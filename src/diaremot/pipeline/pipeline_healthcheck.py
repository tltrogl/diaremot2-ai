#!/usr/bin/env python3
"""DiaRemo pipeline health check utilities.

Lightweight diagnostic script to verify that the DiaRemo pipeline is ready to
run.  The script assumes **faster-whisper** as the default ASR backend and will
fall back to OpenAI's `whisper` package when available.  It performs three
stages:

1. **Dependency check** – import core Python packages and report versions.
2. **Model availability** – ensure critical ONNX models are present.
3. **Smoke test** – optionally run a 1‑second synthetic audio sample through
   the full pipeline.

Usage::

    python pipeline_healthcheck.py [--skip-models] [--skip-smoke] [--local-only]

By default all checks are executed.  Use the flags to skip individual steps or
require models to be available offline.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Dict

# Core packages required for the pipeline
DEPENDENCIES = [
    "torch",
    "librosa",
    "soundfile",
    "numpy",
    "scipy",
    "transformers",
    "faster_whisper",  # preferred ASR backend
    "whisper",  # optional fallback backend
    "onnxruntime",
]

# Minimal set of ONNX models used by the pipeline
MODELS = {
    "speech_emotion": "hf://Dpngtm/wav2vec2-emotion-recognition/model.onnx",
    "vad": "hf://snakers4/silero-vad/silero_vad.onnx",
    "speaker_embedding": "hf://speechbrain/spkrec-ecapa-voxceleb/embedding_model.onnx",
}


def check_dependencies() -> Dict[str, str]:
    """Attempt to import each dependency and return version info."""
    report: Dict[str, str] = {}
    for pkg in DEPENDENCIES:
        try:
            module = importlib.import_module(pkg)
            version = getattr(module, "__version__", "unknown")
            report[pkg] = version
        except Exception as exc:  # pragma: no cover - best effort
            report[pkg] = f"missing ({exc})"
    return report


def check_models(local_only: bool = False) -> Dict[str, str]:
    """Ensure required ONNX models are available."""
    from ..io.onnx_utils import ensure_onnx_model

    report: Dict[str, str] = {}
    for name, src in MODELS.items():
        try:
            path = ensure_onnx_model(src, local_files_only=local_only)
            report[name] = str(path)
        except Exception as exc:  # pragma: no cover - network issues etc.
            report[name] = f"missing ({exc})"
    return report


def run_smoke_test() -> Dict[str, str]:
    """Run a very small end‑to‑end pipeline test."""
    try:
        import numpy as np
        import soundfile as sf
        from .audio_pipeline_core import AudioAnalysisPipelineV2

        sr = 16000
        audio = np.zeros(sr, dtype=np.float32)  # 1‑second silence

        tmp_dir = Path("healthcheck_tmp")
        tmp_dir.mkdir(exist_ok=True)
        wav_path = tmp_dir / "smoke.wav"
        sf.write(wav_path, audio, sr)

        config = {
            "whisper_model": "faster-whisper-tiny.en",
            "noise_reduction": False,
            "beam_size": 1,
            "temperature": 0.0,
            "no_speech_threshold": 0.6,
            "registry_path": str(tmp_dir / "registry.json"),
            "asr_backend": "faster",
        }
        pipeline = AudioAnalysisPipelineV2(config)

        out_dir = tmp_dir / "out"
        out_dir.mkdir(exist_ok=True)
        pipeline.process_audio_file(str(wav_path), str(out_dir))
        return {"success": True, "output": str(out_dir)}
    except Exception as exc:  # pragma: no cover - diagnostics only
        return {"success": False, "error": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(description="DiaRemo pipeline health check")
    parser.add_argument(
        "--skip-models", action="store_true", help="skip model availability checks"
    )
    parser.add_argument(
        "--skip-smoke", action="store_true", help="skip running the pipeline smoke test"
    )
    parser.add_argument(
        "--local-only", action="store_true", help="do not download missing models"
    )
    args = parser.parse_args()

    report = {"dependencies": check_dependencies()}
    if not args.skip_models:
        report["models"] = check_models(local_only=args.local_only)
    if not args.skip_smoke:
        report["smoke_test"] = run_smoke_test()

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
