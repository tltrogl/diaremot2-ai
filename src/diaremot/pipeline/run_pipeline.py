"""Compatibility shims for legacy ``diaremot.pipeline.run_pipeline`` imports."""

from __future__ import annotations

import os
from pathlib import Path

from . import audio_pipeline_core as _core


def _configure_local_cache_env() -> None:
    cache_root = (Path(__file__).resolve().parents[3] / ".cache").resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    targets = {
        "HF_HOME": cache_root / "hf",
        "HUGGINGFACE_HUB_CACHE": cache_root / "hf",
        "TRANSFORMERS_CACHE": cache_root / "transformers",
        "TORCH_HOME": cache_root / "torch",
        "XDG_CACHE_HOME": cache_root,
    }
    for env_name, target in targets.items():
        target_path = target.resolve()
        existing = os.environ.get(env_name)
        if existing:
            try:
                existing_path = Path(existing).resolve()
            except (OSError, RuntimeError, ValueError):
                existing_path = None
            if existing_path is not None:
                if existing_path == target_path:
                    continue
                if existing_path.is_relative_to(cache_root):
                    continue
        target_path.mkdir(parents=True, exist_ok=True)
        os.environ[env_name] = str(target_path)


if "_DIAREMOT_CACHE_ENV_CONFIGURED" not in globals():
    _configure_local_cache_env()
    _DIAREMOT_CACHE_ENV_CONFIGURED = True


try:
    import suppress_warnings as _dia_suppress

    if hasattr(_dia_suppress, "initialize"):
        _dia_suppress.initialize()
except Exception:  # pragma: no cover - legacy guard should never raise
    pass

AudioAnalysisPipelineV2 = _core.AudioAnalysisPipelineV2
DEFAULT_PIPELINE_CONFIG = _core.DEFAULT_PIPELINE_CONFIG


def build_pipeline_config(overrides=None):
    return _core.build_pipeline_config(overrides)


def run_pipeline(input_path, outdir, *, config=None, clear_cache=False):
    return _core.run_pipeline(
        input_path,
        outdir,
        config=config,
        clear_cache=clear_cache,
    )


def resume(checkpoint_path, *, outdir=None, config=None, allow_reprocess=False):
    if allow_reprocess:
        import warnings

        warnings.warn(
            "allow_reprocess flag is ignored by audio_pipeline_core.resume; continuing without reprocessing.",
            RuntimeWarning,
            stacklevel=2,
        )
    return _core.resume(
        checkpoint_path,
        outdir=outdir,
        config=config,
    )


def diagnostics(require_versions=False):
    return _core.diagnostics(require_versions=require_versions)


def verify_dependencies(strict=False):
    return _core.verify_dependencies(strict=strict)


def clear_pipeline_cache(cache_root=None):
    return _core.clear_pipeline_cache(cache_root)


__all__ = [
    "AudioAnalysisPipelineV2",
    "DEFAULT_PIPELINE_CONFIG",
    "build_pipeline_config",
    "run_pipeline",
    "resume",
    "diagnostics",
    "verify_dependencies",
    "clear_pipeline_cache",
]
