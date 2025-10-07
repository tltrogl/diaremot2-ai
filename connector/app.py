from __future__ import annotations

import os
import pathlib
import tempfile
import uuid
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from diaremot.pipeline.orchestrator import run_pipeline as orchestrator_run


app = FastAPI()
API_KEY = os.getenv("X_API_KEY", "dev-key")


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
        try:
            with urlopen(audio_url) as response, tmp_handle:
                tmp_handle.write(response.read())
                tmp_handle.flush()
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


def run_pipeline(audio_url: str | None, params: dict[str, Any], outdir: str) -> None:
    out_path = pathlib.Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    audio_path = _resolve_audio_source(audio_url, out_path)
    config = _prepare_config(params)

    try:
        orchestrator_run(str(audio_path), str(out_path), config=config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


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
    outdir = pathlib.Path("runs") / job_id
    outdir.mkdir(parents=True, exist_ok=True)
    bg.add_task(run_pipeline, req.audio_url, req.params, str(outdir))
    return {"job_id": job_id, "status": "queued"}


@app.get("/status/{job_id}")
def status(job_id: str, x_api_key: str | None = Header(None, alias="x-api-key")):
    ensure_auth(x_api_key)
    outdir = pathlib.Path("runs") / job_id
    done = (outdir / "summary.html").exists()
    return {"job_id": job_id, "status": "done" if done else "running"}


@app.get("/results/{job_id}")
def results(job_id: str, x_api_key: str | None = Header(None, alias="x-api-key")):
    ensure_auth(x_api_key)
    base = f"/static/{job_id}"
    return {
        "job_id": job_id,
        "files": {
            "csv": f"{base}/diarized_transcript_with_emotion.csv",
            "summary_html": f"{base}/summary.html",
            "speakers": f"{base}/speakers_summary.csv",
        },
    }

