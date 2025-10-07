from __future__ import annotations

import json
import logging
import os
import pathlib
import tempfile
import uuid
from collections.abc import Mapping as MappingABC, Sequence as SequenceABC
from datetime import datetime, timezone
from typing import Any, Mapping
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from diaremot.pipeline.orchestrator import run_pipeline as orchestrator_run


logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid integer for %s=%r; using %s", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid float for %s=%r; using %s", name, raw, default)
        return default


DOWNLOAD_CHUNK_SIZE = 1024 * 1024
MAX_DOWNLOAD_MB = _env_int("CONNECTOR_MAX_DOWNLOAD_MB", 128)
MAX_DOWNLOAD_BYTES = 0 if MAX_DOWNLOAD_MB <= 0 else MAX_DOWNLOAD_MB * 1024 * 1024
DOWNLOAD_TIMEOUT = max(1.0, _env_float("CONNECTOR_DOWNLOAD_TIMEOUT", 30.0))

app = FastAPI()
API_KEY = os.getenv("X_API_KEY", "dev-key")
RUNS_DIR = pathlib.Path("runs")
RUNS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(RUNS_DIR)), name="static")


class RunRequest(BaseModel):
    """Payload accepted by the `/run` endpoint."""

    audio_url: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


def _resolve_audio_source(audio_url: str | None, workdir: pathlib.Path) -> pathlib.Path:
    if not audio_url:
        raise HTTPException(status_code=400, detail="audio_url is required")

    parsed = urlparse(audio_url)
    if parsed.scheme in {"http", "https"}:
        suffix = pathlib.Path(parsed.path).suffix or ".wav"
        tmp_handle = tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False, dir=str(workdir)
        )
        tmp_path = pathlib.Path(tmp_handle.name)
        request = Request(audio_url, headers={"User-Agent": "DiaRemotConnector/1.0"})
        try:
            with urlopen(request, timeout=DOWNLOAD_TIMEOUT) as response, tmp_handle:
                if (
                    MAX_DOWNLOAD_BYTES
                    and response.length is not None
                    and response.length > MAX_DOWNLOAD_BYTES
                ):
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "Audio file exceeds connector limit of "
                            f"{MAX_DOWNLOAD_MB} MB"
                        ),
                    )

                total = 0
                while True:
                    chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    total += len(chunk)
                    if MAX_DOWNLOAD_BYTES and total > MAX_DOWNLOAD_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=(
                                "Audio file exceeds connector limit of "
                                f"{MAX_DOWNLOAD_MB} MB"
                            ),
                        )
                    tmp_handle.write(chunk)
                tmp_handle.flush()
        except HTTPException:
            tmp_path.unlink(missing_ok=True)
            raise
        except Exception as exc:  # pragma: no cover - network failure
            tmp_path.unlink(missing_ok=True)
            raise HTTPException(status_code=502, detail=f"Failed to download audio: {exc}")
        return tmp_path

    source_path = pathlib.Path(audio_url).expanduser()
    if not source_path.exists():
        raise HTTPException(status_code=404, detail=f"Audio file not found: {audio_url}")
    return source_path.resolve()


def _prepare_config(overrides: dict[str, Any]) -> dict[str, Any] | None:
    if not overrides:
        return None

    def _normalise_key(key: str) -> str:
        key = key.strip().lower().replace("-", "_")
        if key == "asr_compute_type":
            return "compute_type"
        if key == "asr_cpu_threads":
            return "cpu_threads"
        if key == "chunk_enabled":
            return "auto_chunk_enabled"
        if key == "speech_pad_sec":  # legacy clients
            return "vad_speech_pad_sec"
        return key

    normalised: dict[str, Any] = {}
    for raw_key, value in overrides.items():
        if value is None:
            continue
        normalised[_normalise_key(str(raw_key))] = value

    return normalised or None


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _make_json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat().replace("+00:00", "Z")
    if isinstance(value, MappingABC):
        return {str(k): _make_json_safe(v) for k, v in value.items()}
    if isinstance(value, set):
        return [
            _make_json_safe(v) for v in sorted(value, key=lambda item: repr(item))
        ]
    if isinstance(value, SequenceABC) and not isinstance(value, (str, bytes, bytearray)):
        return [_make_json_safe(v) for v in value]
    return str(value)


def _write_status(
    job_dir: pathlib.Path,
    status: str,
    *,
    manifest: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"status": status, "updated_at": _utcnow()}
    if error:
        payload["error"] = error
    if manifest:
        payload["manifest"] = manifest
        outputs_obj = manifest.get("outputs") or {}
        if not isinstance(outputs_obj, MappingABC):
            outputs_obj = dict(outputs_obj)
        payload["outputs"] = dict(outputs_obj)
        payload["public_urls"] = _build_public_urls(job_dir, payload["outputs"])

    tmp_file = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", delete=False, dir=str(job_dir)
    )
    try:
        json.dump(_make_json_safe(payload), tmp_file, indent=2, sort_keys=True)
        tmp_file.flush()
        os.fsync(tmp_file.fileno())
        tmp_path = pathlib.Path(tmp_file.name)
    finally:
        tmp_file.close()

    tmp_path.replace(job_dir / "status.json")


def _load_status(job_dir: pathlib.Path) -> dict[str, Any] | None:
    status_path = job_dir / "status.json"
    if not status_path.exists():
        return None
    try:
        return json.loads(status_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _build_public_urls(
    job_dir: pathlib.Path, outputs: Mapping[str, Any] | None
) -> dict[str, str]:
    if not outputs:
        return {}

    base_url = f"/static/{job_dir.name}"
    job_root = job_dir.resolve()
    urls: dict[str, str] = {}
    for key, candidate in outputs.items():
        if not isinstance(candidate, str):
            continue
        candidate_path = pathlib.Path(candidate)
        if not candidate_path.exists():
            continue
        try:
            rel_path = candidate_path.resolve().relative_to(job_root)
        except ValueError:
            continue
        urls[key] = f"{base_url}/{rel_path.as_posix()}"
    return urls


def _run_pipeline_job(
    job_id: str,
    audio_url: str | None,
    params: dict[str, Any],
    outdir: pathlib.Path,
) -> None:
    job_dir = outdir
    audio_path: pathlib.Path | None = None
    _write_status(job_dir, "running")

    try:
        audio_path = _resolve_audio_source(audio_url, job_dir)
        config = _prepare_config(params) or {}
        config.setdefault("run_id", job_id)
        config.setdefault("log_dir", str(job_dir / "logs"))
        config.setdefault("cache_root", str(job_dir / "cache"))
        config.setdefault("checkpoint_dir", str(job_dir / "checkpoints"))
        config.setdefault("quiet", True)

        manifest = orchestrator_run(str(audio_path), str(job_dir), config=config)
    except HTTPException as exc:
        logger.warning("Job %s failed: %s", job_id, exc.detail)
        _write_status(job_dir, "failed", error=str(exc.detail))
        return
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Job %s crashed", job_id)
        _write_status(job_dir, "failed", error=str(exc))
        return
    else:
        _write_status(job_dir, "done", manifest=manifest)
    finally:
        if (
            audio_path
            and audio_path.exists()
            and audio_path.is_file()
            and audio_path.parent == job_dir
        ):
            audio_path.unlink(missing_ok=True)


def ensure_auth(x_api_key: str | None):
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.post("/run")
def run_job(
    req: RunRequest,
    bg: BackgroundTasks,
    x_api_key: str | None = Header(None, alias="x-api-key"),
):
    ensure_auth(x_api_key)
    job_id = str(uuid.uuid4())
    outdir = RUNS_DIR / job_id
    outdir.mkdir(parents=True, exist_ok=True)
    _write_status(outdir, "queued")
    bg.add_task(_run_pipeline_job, job_id, req.audio_url, req.params, outdir)
    return {"job_id": job_id, "status": "queued"}


@app.get("/status/{job_id}")
def status(job_id: str, x_api_key: str | None = Header(None, alias="x-api-key")):
    ensure_auth(x_api_key)
    job_dir = RUNS_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    status_payload = _load_status(job_dir) or {"status": "running"}
    response = {
        "job_id": job_id,
        "status": status_payload.get("status", "running"),
        "updated_at": status_payload.get("updated_at"),
    }
    if "error" in status_payload:
        response["error"] = status_payload["error"]
    if status_payload.get("status") == "done":
        response["outputs"] = status_payload.get("outputs", {})
        response["public_urls"] = status_payload.get("public_urls", {})
    return response


@app.get("/results/{job_id}")
def results(job_id: str, x_api_key: str | None = Header(None, alias="x-api-key")):
    ensure_auth(x_api_key)
    job_dir = RUNS_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    status_payload = _load_status(job_dir)
    if not status_payload:
        raise HTTPException(status_code=404, detail="Job status unavailable")
    if status_payload.get("status") != "done":
        raise HTTPException(status_code=409, detail="Job not completed")

    return {
        "job_id": job_id,
        "status": status_payload["status"],
        "updated_at": status_payload.get("updated_at"),
        "outputs": status_payload.get("outputs", {}),
        "public_urls": status_payload.get("public_urls", {}),
        "manifest": status_payload.get("manifest", {}),
    }

