"""Configuration defaults and dependency helpers for the DiaRemot pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

try:  # pragma: no cover - optional during tests
    from packaging.version import Version
except Exception:  # pragma: no cover - defensive fallback
    Version = None  # type: ignore

from .speaker_diarization import DiarizationConfig


def _ensure_path(value: Path | str) -> Path:
    if isinstance(value, Path):
        return value
    return Path(value)


def _ensure_optional_path(value: Path | str | None) -> Path | None:
    if value is None:
        return None
    return _ensure_path(value)


def _ensure_path_list(value: Iterable[Path | str] | Path | str | None) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [_ensure_path(value)]
    return [_ensure_path(item) for item in value]


def _normalise_str(value: Any) -> Any:
    if isinstance(value, str):
        return value.lower()
    return value


def _validate_choice(value: str, allowed: set[str], field_name: str) -> str:
    if value not in allowed:
        raise ValueError(f"{field_name} must be one of {sorted(allowed)}")
    return value


@dataclass(slots=True)
class PipelineConfig:
    """Validated configuration for the end-to-end pipeline."""

    registry_path: Path = field(default_factory=lambda: Path("speaker_registry.json"))
    ahc_distance_threshold: float = DiarizationConfig.ahc_distance_threshold
    speaker_limit: int | None = None
    whisper_model: str = "faster-whisper-tiny.en"
    asr_backend: str = "faster"
    compute_type: str = "float32"
    cpu_threads: int = 1
    language: str | None = None
    language_mode: str = "auto"
    ignore_tx_cache: bool = False
    quiet: bool = False
    disable_affect: bool = False
    affect_backend: str = "onnx"
    affect_text_model_dir: Path | None = None
    affect_intent_model_dir: Path | None = None
    beam_size: int = 1
    temperature: float = 0.0
    no_speech_threshold: float = 0.50
    noise_reduction: bool = False
    enable_sed: bool = True
    auto_chunk_enabled: bool = True
    chunk_threshold_minutes: float = 30.0
    chunk_size_minutes: float = 20.0
    chunk_overlap_seconds: float = 30.0
    vad_threshold: float = 0.30
    vad_min_speech_sec: float = 0.8
    vad_min_silence_sec: float = 0.8
    vad_speech_pad_sec: float = 0.2
    vad_backend: str = "auto"
    disable_energy_vad_fallback: bool = False
    energy_gate_db: float = -33.0
    energy_hop_sec: float = 0.01
    max_asr_window_sec: int = 480
    segment_timeout_sec: float = 300.0
    batch_timeout_sec: float = 1200.0
    cpu_diarizer: bool = False
    validate_dependencies: bool = False
    strict_dependency_versions: bool = False
    cache_root: Path = field(default_factory=lambda: Path(".cache"))
    cache_roots: list[Path] = field(default_factory=list)
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    target_sr: int = 16000
    loudness_mode: str = "asr"
    run_id: str | None = None
    text_emotion_model: str | None = None
    intent_labels: list[str] | None = None

    def __post_init__(self) -> None:
        self.registry_path = _ensure_path(self.registry_path)
        self.affect_text_model_dir = _ensure_optional_path(self.affect_text_model_dir)
        self.affect_intent_model_dir = _ensure_optional_path(self.affect_intent_model_dir)
        self.cache_root = _ensure_path(self.cache_root)
        self.log_dir = _ensure_path(self.log_dir)
        self.checkpoint_dir = _ensure_path(self.checkpoint_dir)
        self.cache_roots = _ensure_path_list(self.cache_roots)

        self.affect_backend = _validate_choice(
            _normalise_str(self.affect_backend), {"auto", "onnx", "torch"}, "affect_backend"
        )
        self.asr_backend = _normalise_str(self.asr_backend)
        self.vad_backend = _validate_choice(
            _normalise_str(self.vad_backend), {"auto", "onnx", "torch"}, "vad_backend"
        )
        self.language_mode = _normalise_str(self.language_mode)
        self.loudness_mode = _validate_choice(
            _normalise_str(self.loudness_mode), {"asr", "broadcast"}, "loudness_mode"
        )

        if self.speaker_limit is not None and self.speaker_limit < 1:
            raise ValueError("speaker_limit must be >= 1 when provided")
        if self.cpu_threads < 1:
            raise ValueError("cpu_threads must be >= 1")
        if self.beam_size < 1:
            raise ValueError("beam_size must be >= 1")
        if not (0.0 <= self.temperature <= 1.0):
            raise ValueError("temperature must be between 0.0 and 1.0")
        if not (0.0 <= self.no_speech_threshold <= 1.0):
            raise ValueError("no_speech_threshold must be between 0.0 and 1.0")
        if not (0.0 <= self.vad_threshold <= 1.0):
            raise ValueError("vad_threshold must be between 0.0 and 1.0")
        if self.vad_min_speech_sec < 0.0:
            raise ValueError("vad_min_speech_sec must be >= 0")
        if self.vad_min_silence_sec < 0.0:
            raise ValueError("vad_min_silence_sec must be >= 0")
        if self.vad_speech_pad_sec < 0.0:
            raise ValueError("vad_speech_pad_sec must be >= 0")
        if self.energy_hop_sec <= 0.0:
            raise ValueError("energy_hop_sec must be > 0")
        if self.max_asr_window_sec <= 0:
            raise ValueError("max_asr_window_sec must be > 0")
        if self.segment_timeout_sec <= 0.0:
            raise ValueError("segment_timeout_sec must be > 0")
        if self.batch_timeout_sec <= 0.0:
            raise ValueError("batch_timeout_sec must be > 0")
        if self.chunk_threshold_minutes <= 0.0:
            raise ValueError("chunk_threshold_minutes must be > 0")
        if self.chunk_size_minutes <= 0.0:
            raise ValueError("chunk_size_minutes must be > 0")
        if self.chunk_overlap_seconds < 0.0:
            raise ValueError("chunk_overlap_seconds must be >= 0")
        if self.chunk_size_minutes * 60.0 <= self.chunk_overlap_seconds:
            raise ValueError("chunk_overlap_seconds must be smaller than chunk_size_minutes * 60")
        if self.chunk_threshold_minutes < self.chunk_size_minutes:
            raise ValueError("chunk_threshold_minutes must be >= chunk_size_minutes")

    def model_dump(self, mode: str = "python") -> dict[str, Any]:  # pragma: no cover - exercised via tests
        return asdict(self)

    @classmethod
    def model_validate(cls, data: Mapping[str, Any] | "PipelineConfig") -> "PipelineConfig":
        if isinstance(data, cls):
            return data
        if not isinstance(data, Mapping):
            raise TypeError("PipelineConfig.model_validate expects a mapping or PipelineConfig instance")
        return cls(**data)


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
    except (TypeError, ValueError) as exc:  # pragma: no cover - surface readable error upstream
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
