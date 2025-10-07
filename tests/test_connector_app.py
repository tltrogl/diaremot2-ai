from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

MODULE_PATH = Path(__file__).resolve().parent.parent / "connector" / "app.py"
SPEC = importlib.util.spec_from_file_location("connector_app", MODULE_PATH)
assert SPEC and SPEC.loader
connector_app = importlib.util.module_from_spec(SPEC)
sys.modules["connector_app"] = connector_app
SPEC.loader.exec_module(connector_app)

HTTPException = connector_app.HTTPException


def test_prepare_config_normalises_keys():
    overrides = {
        "ASR_compute_type": "int8",
        "asr_cpu_threads": 2,
        "chunk-enabled": True,
        " speech_pad_sec ": 0.5,
        "noop": None,
    }

    result = connector_app._prepare_config(overrides)

    assert result == {
        "compute_type": "int8",
        "cpu_threads": 2,
        "auto_chunk_enabled": True,
        "vad_speech_pad_sec": 0.5,
    }


def test_make_json_safe_coerces_collections(tmp_path):
    base_path = tmp_path / "artifact.txt"
    base_path.write_text("payload", encoding="utf-8")

    payload = {
        "path": base_path,
        "timestamp": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "nested": {"set": {2, 1}, "seq": ("ok", Path("inner"))},
        "other": object(),
    }

    safe = connector_app._make_json_safe(payload)

    assert safe["path"] == str(base_path)
    assert safe["timestamp"] == "2024-01-01T00:00:00Z"
    assert safe["nested"]["set"] == sorted(safe["nested"]["set"])
    assert safe["nested"]["seq"] == ["ok", "inner"]
    assert isinstance(safe["other"], str)
    json.dumps(safe)


def test_build_public_urls_filters_nonlocal_outputs(tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()

    inside = job_dir / "transcript.json"
    inside.write_text("{}", encoding="utf-8")
    outside = tmp_path / "other.json"
    outside.write_text("{}", encoding="utf-8")

    urls = connector_app._build_public_urls(
        job_dir,
        {
            "transcript": str(inside),
            "external": str(outside),
            "missing": str(job_dir / "missing.json"),
            "nonstring": 42,
        },
    )

    assert urls == {
        "transcript": f"/static/{job_dir.name}/transcript.json"
    }


def test_resolve_audio_source_local_path(tmp_path):
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"RIFFdata")

    resolved = connector_app._resolve_audio_source(str(audio_path), tmp_path)

    assert resolved == audio_path.resolve()


def test_resolve_audio_source_streaming_limit(monkeypatch, tmp_path):
    monkeypatch.setattr(connector_app, "MAX_DOWNLOAD_MB", 1)
    monkeypatch.setattr(connector_app, "MAX_DOWNLOAD_BYTES", 3)
    monkeypatch.setattr(connector_app, "DOWNLOAD_CHUNK_SIZE", 2)

    class FakeResponse:
        length = None

        def __init__(self):
            self._chunks = [b"aa", b"bb"]

        def read(self, size):
            return self._chunks.pop(0) if self._chunks else b""

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(request, timeout):
        assert request.full_url == "https://example.com/audio.wav"
        return FakeResponse()

    monkeypatch.setattr(connector_app, "urlopen", fake_urlopen)

    workdir = tmp_path / "job"
    workdir.mkdir()

    with pytest.raises(HTTPException) as excinfo:
        connector_app._resolve_audio_source(
            "https://example.com/audio.wav", workdir
        )

    assert excinfo.value.status_code == 413
    assert not any(workdir.iterdir())


def test_resolve_audio_source_length_guard(monkeypatch, tmp_path):
    monkeypatch.setattr(connector_app, "MAX_DOWNLOAD_MB", 1)
    monkeypatch.setattr(connector_app, "MAX_DOWNLOAD_BYTES", 4)

    class FakeResponse:
        length = 8

        def read(self, size):
            return b""

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(request, timeout):
        return FakeResponse()

    monkeypatch.setattr(connector_app, "urlopen", fake_urlopen)

    workdir = tmp_path / "job"
    workdir.mkdir()

    with pytest.raises(HTTPException) as excinfo:
        connector_app._resolve_audio_source(
            "https://example.com/large.wav", workdir
        )

    assert excinfo.value.status_code == 413
    assert not any(workdir.iterdir())
