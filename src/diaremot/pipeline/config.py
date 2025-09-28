"""Configuration defaults and dependency helpers for the DiaRemot pipeline."""

from __future__ import annotations

from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterator

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

try:  # pragma: no cover - packaging optional during tests
    from packaging.version import Version
except Exception:  # pragma: no cover - defensive fallback
    Version = None  # type: ignore

from .speaker_diarization import DiarizationConfig


class PipelineConfig(BaseModel):
    """Validated configuration for the end-to-end pipeline."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    registry_path: Path = Field(default=Path("speaker_registry.json"))
    ahc_distance_threshold: float = Field(default=DiarizationConfig.ahc_distance_threshold, ge=0.0)
    speaker_limit: int | None = Field(default=None, ge=1)
    whisper_model: str = Field(default="faster-whisper-tiny.en")
    asr_backend: str = Field(default="faster")
    compute_type: str = Field(default="float32")
    cpu_threads: int = Field(default=1, ge=1)
    language: str | None = None
    language_mode: str = Field(default="auto")
    ignore_tx_cache: bool = False
    quiet: bool = False
    disable_affect: bool = False
    affect_backend: str = Field(default="onnx")
    affect_text_model_dir: Path | None = None
    affect_intent_model_dir: Path | None = None
    beam_size: int = Field(default=1, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    no_speech_threshold: float = Field(default=0.50, ge=0.0, le=1.0)
    noise_reduction: bool = False
    enable_sed: bool = True
    auto_chunk_enabled: bool = True
    chunk_threshold_minutes: float = Field(default=30.0, gt=0.0)
    chunk_size_minutes: float = Field(default=20.0, gt=0.0)
    chunk_overlap_seconds: float = Field(default=30.0, ge=0.0)
    vad_threshold: float = Field(default=0.30, ge=0.0, le=1.0)
    vad_min_speech_sec: float = Field(default=0.8, ge=0.0)
    vad_min_silence_sec: float = Field(default=0.8, ge=0.0)
    vad_speech_pad_sec: float = Field(default=0.2, ge=0.0)
    vad_backend: str = Field(default="auto")
    disable_energy_vad_fallback: bool = False
    energy_gate_db: float = Field(default=-33.0)
    energy_hop_sec: float = Field(default=0.01, gt=0.0)
    max_asr_window_sec: int = Field(default=480, gt=0)
    segment_timeout_sec: float = Field(default=300.0, gt=0.0)
    batch_timeout_sec: float = Field(default=1200.0, gt=0.0)
    cpu_diarizer: bool = False
    validate_dependencies: bool = False
    strict_dependency_versions: bool = False
    cache_root: Path = Field(default=Path(".cache"))
    cache_roots: list[Path] = Field(default_factory=list)
    log_dir: Path = Field(default=Path("logs"))
    checkpoint_dir: Path = Field(default=Path("checkpoints"))
    target_sr: int = Field(default=16000, gt=0)
    loudness_mode: str = Field(default="asr")
    run_id: str | None = None
    text_emotion_model: str | None = None
    intent_labels: list[str] | None = None

    @field_validator("affect_backend", "asr_backend", "vad_backend", mode="before")
    @classmethod
    def _lower_str(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.lower()
        return value

    @field_validator("affect_backend")
    @classmethod
    def _validate_affect_backend(cls, value: str) -> str:
        allowed = {"auto", "onnx", "torch"}
        if value not in allowed:
            raise ValueError(f"affect_backend must be one of {sorted(allowed)}")
        return value

    @field_validator("vad_backend")
    @classmethod
    def _validate_vad_backend(cls, value: str) -> str:
        allowed = {"auto", "onnx", "torch"}
        if value not in allowed:
            raise ValueError(f"vad_backend must be one of {sorted(allowed)}")
        return value

    @field_validator("loudness_mode")
    @classmethod
    def _validate_loudness_mode(cls, value: str) -> str:
        allowed = {"asr", "broadcast"}
        if value not in allowed:
            raise ValueError(f"loudness_mode must be one of {sorted(allowed)}")
        return value

    @field_validator("cache_roots", mode="before")
    @classmethod
    def _coerce_cache_roots(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, (str, Path)):
            return [value]
        return value

    @model_validator(mode="after")
    def _validate_chunking(self) -> "PipelineConfig":
        if self.chunk_size_minutes * 60.0 <= self.chunk_overlap_seconds:
            raise ValueError("chunk_overlap_seconds must be smaller than chunk_size_minutes * 60")
        if self.chunk_threshold_minutes < self.chunk_size_minutes:
            raise ValueError("chunk_threshold_minutes must be >= chunk_size_minutes")
        return self


DEFAULT_PIPELINE_CONFIG: dict[str, Any] = PipelineConfig().model_dump(mode="python")

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
    "PipelineConfig",
    "build_pipeline_config",
    "verify_dependencies",
    "diagnostics",
    "dependency_health_summary",
]


def build_pipeline_config(overrides: dict[str, Any] | PipelineConfig | None = None) -> dict[str, Any]:
    """Return a validated pipeline configuration merged with overrides."""

    if isinstance(overrides, PipelineConfig):
        return overrides.model_dump(mode="python")

    base = PipelineConfig()
    if not overrides:
        return base.model_dump(mode="python")

    merged: dict[str, Any] = base.model_dump(mode="python")
    for key, value in overrides.items():
        if key not in merged:
            raise ValueError(f"Unknown configuration key: {key}")
        if value is None:
            continue
        merged[key] = value

    try:
        validated = PipelineConfig.model_validate(merged)
    except ValidationError as exc:  # pragma: no cover - surface readable error upstream
        raise ValueError(str(exc)) from exc
    return validated.model_dump(mode="python")


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
