"""Configuration defaults and dependency helpers for the DiaRemot pipeline."""

from __future__ import annotations

from importlib import metadata as importlib_metadata
from typing import Any, Iterator

try:  # pragma: no cover - packaging optional during tests
    from packaging.version import Version
except Exception:  # pragma: no cover - defensive fallback
    Version = None  # type: ignore

from .speaker_diarization import DiarizationConfig

DEFAULT_PIPELINE_CONFIG: dict[str, Any] = {
    "registry_path": "speaker_registry.json",
    "ahc_distance_threshold": DiarizationConfig.ahc_distance_threshold,
    "speaker_limit": None,
    "whisper_model": "faster-whisper-tiny.en",
    "asr_backend": "faster",
    "compute_type": "float32",
    "cpu_threads": 1,
    "language": None,
    "language_mode": "auto",
    "ignore_tx_cache": False,
    "quiet": False,
    "disable_affect": False,
    "affect_backend": "onnx",
    "affect_text_model_dir": None,
    "affect_intent_model_dir": None,
    "beam_size": 1,
    "temperature": 0.0,
    "no_speech_threshold": 0.50,
    "noise_reduction": False,
    "enable_sed": True,
    "auto_chunk_enabled": True,
    "chunk_threshold_minutes": 30.0,
    "chunk_size_minutes": 20.0,
    "chunk_overlap_seconds": 30.0,
    "vad_threshold": 0.30,
    "vad_min_speech_sec": 0.8,
    "vad_min_silence_sec": 0.8,
    "vad_speech_pad_sec": 0.2,
    "vad_backend": "auto",
    "disable_energy_vad_fallback": False,
    "energy_gate_db": -33.0,
    "energy_hop_sec": 0.01,
    "max_asr_window_sec": 480,
    "segment_timeout_sec": 300.0,
    "batch_timeout_sec": 1200.0,
    "cpu_diarizer": False,
    "validate_dependencies": False,
    "strict_dependency_versions": False,
    "cache_root": ".cache",
    "cache_roots": [],
    "log_dir": "logs",
    "checkpoint_dir": "checkpoints",
    "target_sr": 16000,
    "loudness_mode": "asr",
}

CORE_DEPENDENCY_REQUIREMENTS: dict[str, str] = {
    "numpy": "1.24",
    "scipy": "1.10",
    "librosa": "0.10",
    "soundfile": "0.12",
    "torch": "2.0",
    "ctranslate2": "3.10",
    "faster_whisper": "1.0",
    "pandas": "2.0",
    "onnxruntime": "1.16",
    "transformers": "4.30",
}

__all__ = [
    "DEFAULT_PIPELINE_CONFIG",
    "CORE_DEPENDENCY_REQUIREMENTS",
    "build_pipeline_config",
    "verify_dependencies",
    "diagnostics",
    "dependency_health_summary",
]


def build_pipeline_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a pipeline configuration merged with overrides."""

    config = dict(DEFAULT_PIPELINE_CONFIG)
    if overrides:
        for key, value in overrides.items():
            if key not in DEFAULT_PIPELINE_CONFIG and value is None:
                # Skip unknown keys explicitly requesting default behaviour
                continue
            if value is not None or key in config:
                config[key] = value
    return config


def _iter_dependency_status() -> Iterator[tuple[str, str, Any, str | None, Exception | None, Exception | None]]:
    for mod, min_ver in CORE_DEPENDENCY_REQUIREMENTS.items():
        import_error: Exception | None = None
        metadata_error: Exception | None = None
        module = None
        try:
            module = __import__(mod.replace("-", "_"))
        except Exception as exc:  # pragma: no cover - defensive import guard
            import_error = exc

        version: str | None = None
        if module is not None:
            try:
                version = importlib_metadata.version(mod)
            except importlib_metadata.PackageNotFoundError:
                version = getattr(module, "__version__", None)
            except Exception as exc:  # pragma: no cover - metadata failure
                metadata_error = exc
        yield mod, min_ver, module, version, import_error, metadata_error


def _verify_core_dependencies(require_versions: bool = False) -> tuple[bool, list[str]]:
    issues: list[str] = []

    for mod, min_ver, module, version, import_error, metadata_error in _iter_dependency_status():
        if import_error is not None or module is None:
            issues.append(f"Missing or failed to import: {mod} ({import_error})")
            continue

        if not require_versions:
            continue

        if version is None:
            reason = metadata_error or "version metadata unavailable"
            issues.append(f"Version unknown for {mod}; require >= {min_ver} ({reason})")
            continue

        if Version is None:
            continue

        try:
            if Version(version) < Version(min_ver):
                issues.append(f"{mod} version {version} < required {min_ver}")
        except Exception as exc:  # pragma: no cover - comparison safety
            issues.append(f"Version check failed for {mod}: {exc}")

    return (len(issues) == 0), issues


def dependency_health_summary() -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}

    for mod, min_ver, module, version, import_error, metadata_error in _iter_dependency_status():
        entry: dict[str, Any] = {"required_min": min_ver}

        if import_error is not None or module is None:
            entry["status"] = "error"
            entry["issue"] = str(import_error)
            summary[mod] = entry
            continue

        entry["status"] = "ok"

        if metadata_error is not None:
            entry["status"] = "warn"
            entry["issue"] = f"version lookup failed: {metadata_error}"

        if version is not None:
            entry["version"] = str(version)
            if Version is not None:
                try:
                    if Version(version) < Version(min_ver):
                        entry["status"] = "warn"
                        entry["issue"] = f"version {version} < required {min_ver}"
                except Exception as exc:  # pragma: no cover - comparison safety
                    entry["status"] = "warn"
                    entry["issue"] = f"version comparison failed: {exc}"
        else:
            entry.setdefault("issue", "version metadata unavailable")

        summary[mod] = entry

    return summary


def verify_dependencies(strict: bool = False) -> tuple[bool, list[str]]:
    """Expose lightweight dependency verification for external callers."""

    return _verify_core_dependencies(require_versions=strict)


def diagnostics(require_versions: bool = False) -> dict[str, Any]:
    """Return diagnostic information about optional runtime dependencies."""

    ok, issues = _verify_core_dependencies(require_versions=require_versions)
    return {
        "ok": ok,
        "issues": issues,
        "summary": dependency_health_summary(),
        "strict_versions": require_versions,
    }
