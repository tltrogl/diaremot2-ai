"""Core orchestration logic for the DiaRemot audio analysis pipeline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    from importlib import metadata as importlib_metadata
except ImportError:
    import importlib_metadata  # type: ignore

try:
    from packaging.version import Version
except Exception:
    Version = None  # type: ignore

from ..affect.emotion_analyzer import EmotionIntentAnalyzer
from ..affect.intent_defaults import INTENT_LABELS_DEFAULT
from ..affect.sed_panns import PANNSEventTagger, SEDConfig  # type: ignore
from ..summaries.conversation_analysis import (
    ConversationMetrics,
    analyze_conversation_flow,
)
from ..summaries.html_summary_generator import HTMLSummaryGenerator
from ..summaries.pdf_summary_generator import PDFSummaryGenerator
from ..summaries.speakers_summary_builder import build_speakers_summary
from .audio_preprocessing import AudioPreprocessor, PreprocessConfig
from .pipeline_checkpoint_system import PipelineCheckpointManager, ProcessingStage
from .speaker_diarization import DiarizationConfig, SpeakerDiarizer

# Early environment and warning configuration (also done in run_pipeline.py).
# Ensure direct module runs get the same behavior.
try:
    import suppress_warnings as _dia_suppress

    # Initialize environment vars and warning filters (no-op if unavailable)
    if hasattr(_dia_suppress, "initialize"):
        _dia_suppress.initialize()
except Exception:
    # Proceed without suppression if module not available
    pass


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


_configure_local_cache_env()

WINDOWS_MODELS_ROOT = Path("D:/models") if os.name == "nt" else None


def _resolve_default_whisper_model() -> Path:
    env_override = os.environ.get("WHISPER_MODEL_PATH")
    if env_override:
        return Path(env_override)

    candidates = []
    if WINDOWS_MODELS_ROOT:
        candidates.append(WINDOWS_MODELS_ROOT / "faster-whisper-large-v3-turbo-ct2")
    candidates.append(
        Path.home() / "whisper_models" / "faster-whisper-large-v3-turbo-ct2"
    )

    for candidate in candidates:
        if Path(candidate).exists():
            return Path(candidate)

    return Path(candidates[0])


DEFAULT_WHISPER_MODEL = _resolve_default_whisper_model()

try:
    from ..affect import paralinguistics as para
except Exception:
    para = None
CACHE_VERSION = "v3"  # Incremented to handle new checkpoint logic


def _first_existing_path(*candidates: str) -> str | None:
    for cand in candidates:
        if cand and Path(cand).expanduser().resolve().exists():
            return str(Path(cand).expanduser().resolve())
    return None


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
    "AudioAnalysisPipelineV2",
    "build_pipeline_config",
    "run_pipeline",
    "resume",
    "diagnostics",
    "verify_dependencies",
    "clear_pipeline_cache",
]


def build_pipeline_config(
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a pipeline configuration merged with overrides."""

    config = dict(DEFAULT_PIPELINE_CONFIG)
    if overrides:
        for key, value in overrides.items():
            if key not in DEFAULT_PIPELINE_CONFIG and value is None:
                # Skip unknown keys that explicitly request default behaviour
                continue
            if value is not None or key in config:
                config[key] = value
    return config


def clear_pipeline_cache(cache_root: Path | None = None) -> None:
    """Remove cached diarization/transcription artefacts."""

    cache_dir = Path(cache_root) if cache_root else Path(".cache")
    if cache_dir.exists():
        import shutil

        try:
            shutil.rmtree(cache_dir, ignore_errors=True)
        except PermissionError:
            raise RuntimeError(
                "Could not clear cache directory due to insufficient permissions"
            )
    cache_dir.mkdir(parents=True, exist_ok=True)


def verify_dependencies(strict: bool = False) -> tuple[bool, list[str]]:
    """Expose lightweight dependency verification for external callers."""

    return _verify_core_dependencies(require_versions=strict)


def run_pipeline(
    input_path: str,
    outdir: str,
    *,
    config: dict[str, Any] | None = None,
    clear_cache: bool = False,
) -> dict[str, Any]:
    """Execute the pipeline for ``input_path`` writing artefacts to ``outdir``."""

    if clear_cache:
        try:
            clear_pipeline_cache(
                Path(config.get("cache_root", ".cache")) if config else None
            )
        except RuntimeError:
            if config is None:
                config = dict(DEFAULT_PIPELINE_CONFIG)
            config["ignore_tx_cache"] = True

    merged_config = build_pipeline_config(config)
    pipe = AudioAnalysisPipelineV2(merged_config)
    return pipe.process_audio_file(input_path, outdir)


def resume(
    input_path: str,
    outdir: str,
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resume a previous run using available checkpoints/caches."""

    merged_config = build_pipeline_config(config)
    merged_config["ignore_tx_cache"] = False
    pipe = AudioAnalysisPipelineV2(merged_config)
    stage, _data, metadata = pipe.checkpoints.get_resume_point(input_path)
    if metadata is not None:
        pipe.corelog.info(
            "Resuming from %s checkpoint created at %s",
            metadata.stage.value
            if hasattr(metadata.stage, "value")
            else metadata.stage,
            metadata.timestamp,
        )
    return pipe.process_audio_file(input_path, outdir)


def diagnostics(require_versions: bool = False) -> dict[str, Any]:
    """Return diagnostic information about optional runtime dependencies."""

    ok, issues = _verify_core_dependencies(require_versions=require_versions)
    return {
        "ok": ok,
        "issues": issues,
        "summary": _dependency_health_summary(),
        "strict_versions": require_versions,
    }


# Lightweight dependency verification (no heavy imports at module import time)
def _iter_dependency_status():
    for mod, min_ver in CORE_DEPENDENCY_REQUIREMENTS.items():
        import_error: Exception | None = None
        metadata_error: Exception | None = None
        module = None
        try:
            module = __import__(mod.replace("-", "_"))
        except Exception as exc:
            import_error = exc

        version: str | None = None
        if module is not None:
            try:
                version = importlib_metadata.version(mod)
            except importlib_metadata.PackageNotFoundError:
                version = getattr(module, "__version__", None)
            except Exception as exc:
                metadata_error = exc
        yield mod, min_ver, module, version, import_error, metadata_error


def _verify_core_dependencies(require_versions: bool = False):
    """Verify core runtime dependencies are importable (and optionally versioned).

    Returns (ok: bool, issues: List[str]).
    This avoids importing heavy, optional diagnostics modules.
    """

    issues: list[str] = []

    for (
        mod,
        min_ver,
        module,
        version,
        import_error,
        metadata_error,
    ) in _iter_dependency_status():
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
        except Exception as exc:
            issues.append(f"Version check failed for {mod}: {exc}")

    return (len(issues) == 0), issues


# Detailed dependency health summary for logging/reporting
def _dependency_health_summary():
    summary: dict[str, dict[str, Any]] = {}

    for (
        mod,
        min_ver,
        module,
        version,
        import_error,
        metadata_error,
    ) in _iter_dependency_status():
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
                except Exception as exc:
                    entry["status"] = "warn"
                    entry["issue"] = f"version comparison failed: {exc}"
        else:
            entry.setdefault("issue", "version metadata unavailable")

        summary[mod] = entry

    return summary


# Utility functions
def _fmt_hms(seconds: float) -> str:
    seconds = max(0, float(seconds))
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _fmt_hms_ms(ms: float) -> str:
    total_ms = int(round(max(0.0, float(ms))))
    s, ms = divmod(total_ms, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}" if h else f"{m:02d}:{s:02d}.{ms:03d}"


def _compute_audio_sha16(y: np.ndarray) -> str:
    try:
        arr = np.asarray(y, dtype=np.float32)
    except Exception:
        return hashlib.blake2s(b"", digest_size=16).hexdigest()

    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)

    return hashlib.blake2s(arr.tobytes(), digest_size=16).hexdigest()


def _compute_pp_signature(pp_conf: PreprocessConfig) -> dict:
    keys = ["target_sr", "denoise", "loudness_mode"]
    sig = {}
    for k in keys:
        try:
            sig[k] = getattr(pp_conf, k)
        except Exception:
            sig[k] = None
    return sig


def _compute_pipeline_signature(self) -> dict:
    sig = _compute_pp_signature(self.pp_conf)
    sig["transcriber_model"] = getattr(self.tx, "model_size", "unknown")
    sig["registry_hash"] = self._compute_registry_hash()
    return sig


def _atomic_write_json(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _read_json_safe(path: Path):
    path = Path(path)
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


# JSONL Logger
class JSONLWriter:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")

    def emit(self, record: dict[str, Any]) -> None:
        try:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except PermissionError as e:
            print(f"Warning: Could not write to log file {self.path}: {e}")
        except Exception as e:
            print(f"Warning: Error writing to log file {self.path}: {e}")


@dataclass
class RunStats:
    run_id: str
    file_id: str
    schema_version: str = "2.0.0"
    stage_timings_ms: dict[str, float] = field(default_factory=dict)
    stage_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)
    models: dict[str, Any] = field(default_factory=dict)
    config_snapshot: dict[str, Any] = field(default_factory=dict)

    def mark(self, stage: str, elapsed_ms: float, counts: dict[str, int] | None = None):
        self.stage_timings_ms[stage] = self.stage_timings_ms.get(stage, 0.0) + float(
            elapsed_ms
        )
        if counts:
            slot = self.stage_counts.setdefault(stage, {})
            for k, v in counts.items():
                slot[k] = slot.get(k, 0) + int(v)


class CoreLogger:
    def __init__(
        self, run_id: str, jsonl_path: Path, console_level: int = logging.INFO
    ):
        self.run_id = run_id
        self.jsonl = JSONLWriter(jsonl_path)
        self.log = logging.getLogger(f"pipeline.{run_id}")
        self.log.setLevel(console_level)
        if not self.log.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(console_level)
            fmt = logging.Formatter(
                "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M"
            )
            ch.setFormatter(fmt)
            self.log.addHandler(ch)

    def event(self, stage: str, event: str, **fields):
        rec = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "run_id": self.run_id,
            "stage": stage,
            "event": event,
        }
        rec.update(fields)
        self.jsonl.emit(rec)

    def info(self, msg: str):
        self.log.info(msg)

    def warn(self, msg: str):
        self.log.warning(msg)

    def error(self, msg: str):
        self.log.error(msg)


class StageGuard:
    _OPTIONAL_STAGE_EXCEPTION_MAP = {
        "background_sed": (
            ImportError,
            ModuleNotFoundError,
            FileNotFoundError,
            OSError,
        ),
        "registry_update": (
            FileNotFoundError,
            PermissionError,
            OSError,
        ),
        "paralinguistics": (
            ImportError,
            ModuleNotFoundError,
        ),
        "affect_and_assemble": (
            ImportError,
            ModuleNotFoundError,
        ),
        "overlap_interruptions": (
            AttributeError,
            ImportError,
            ModuleNotFoundError,
        ),
        "conversation_analysis": (ValueError,),
        "speaker_rollups": (
            ValueError,
            TypeError,
        ),
    }
    _CRITICAL_STAGES = {"preprocess", "outputs"}
    _TIMEOUT_STAGES = {"diarize", "transcribe"}

    def __init__(self, corelog: CoreLogger, stats: RunStats, stage: str):
        self.corelog = corelog
        self.stats = stats
        self.stage = stage
        self.start = None

    def __enter__(self):
        self.start = time.time()
        self.corelog.event(self.stage, "start")
        self.corelog.info(f"[{self.stage}] start")
        return self

    def done(self, **counts):
        if counts:
            self.stats.mark(self.stage, 0.0, counts)

    def _is_known_nonfatal(self, exc: BaseException) -> bool:
        if isinstance(exc, TimeoutError | subprocess.TimeoutExpired) and (
            self.stage in self._TIMEOUT_STAGES
        ):
            return True
        allowed = self._OPTIONAL_STAGE_EXCEPTION_MAP.get(self.stage, tuple())
        return any(isinstance(exc, exc_cls) for exc_cls in allowed)

    def __exit__(self, exc_type, exc, tb):
        elapsed_ms = (time.time() - self.start) * 1000.0 if self.start else 0.0
        if exc:
            # Always propagate KeyboardInterrupt so Ctrl+C works
            if isinstance(exc, KeyboardInterrupt):
                self.corelog.error("[interrupt] KeyboardInterrupt received; aborting")
                return False
            known_nonfatal = self._is_known_nonfatal(exc)
            trace_hash = hashlib.blake2b(
                f"{self.stage}:{type(exc).__name__}".encode(), digest_size=8
            ).hexdigest()
            self.corelog.event(
                self.stage,
                "error",
                elapsed_ms=elapsed_ms,
                error=f"{type(exc).__name__}: {exc}",
                trace_hash=trace_hash,
                handled=known_nonfatal,
            )
            dur_txt = _fmt_hms_ms(elapsed_ms)
            log_fn = self.corelog.warn if known_nonfatal else self.corelog.error
            log_fn(
                f"[{self.stage}] {'handled ' if known_nonfatal else ''}"
                f"{type(exc).__name__}: {exc} ({dur_txt})"
            )
            self.stats.mark(self.stage, elapsed_ms)
            try:
                msg = f"{self.stage}: {type(exc).__name__}: {exc}"
                self.stats.warnings.append(msg)
                self.stats.errors.append(msg)

                # Record structured failure with fix suggestion
                def _suggest_fix(stage: str, err: BaseException) -> str:
                    txt = str(err).lower()
                    if stage == "preprocess":
                        if "libsndfile" in txt or "soundfile" in txt:
                            return "Install libsndfile: apt-get install libsndfile1 (Linux) or brew install libsndfile (macOS)."
                        if "ffmpeg" in txt or "audioread" in txt:
                            return "Install ffmpeg and ensure it is on PATH."
                        if "file not found" in txt or "no such file" in txt:
                            return "Check input path and permissions."
                        return "Verify audio codec support (try converting to WAV 16kHz mono)."
                    if stage == "transcribe":
                        if isinstance(err, TimeoutError | subprocess.TimeoutExpired):
                            return "Increase --asr-segment-timeout or choose a smaller Whisper model."
                        if "faster_whisper" in txt or "ctranslate2" in txt:
                            return "Install faster-whisper and ctranslate2; confirm CPU wheels are compatible."
                        if "whisper" in txt and "tiny" in txt:
                            return "OpenAI whisper fallback failed; try reinstalling whisper or using a local model."
                        if "model" in txt and ("not found" in txt or "download" in txt):
                            return "Model not found; provide a valid local model path or enable network access."
                        return "Reduce model size, set compute_type=float32, and verify dependencies."
                    if stage == "paralinguistics":
                        return "Install librosa/scipy extras or run with --disable_paralinguistics."
                    if stage == "affect_and_assemble":
                        return "Install emotion/intent model dependencies or run with --disable_affect."
                    if stage == "background_sed":
                        return "Provide SED models locally or disable background SED tagging."
                    if stage == "overlap_interruptions":
                        return "Install paralinguistics extras for overlap metrics or skip this stage."
                    if stage == "conversation_analysis":
                        return "Ensure numpy/pandas are available for analytics or review conversation inputs."
                    if stage == "speaker_rollups":
                        return "Inspect segment data integrity before computing speaker rollups."
                    if stage == "outputs":
                        return "Ensure outdir is writable and disk has space."
                    return "Check logs for details; ensure dependencies and file permissions."

                self.stats.failures.append(
                    {
                        "stage": self.stage,
                        "error": f"{type(exc).__name__}: {exc}",
                        "elapsed_ms": elapsed_ms,
                        "suggestion": _suggest_fix(self.stage, exc),
                    }
                )
                # Stage-specific flags to inform outer control flow
                if self.stage == "preprocess":
                    self.stats.config_snapshot["preprocess_failed"] = True
                if self.stage == "transcribe":
                    self.stats.config_snapshot["transcribe_failed"] = True
            except Exception:
                pass
            swallow = known_nonfatal and self.stage not in self._CRITICAL_STAGES
            return swallow
        else:
            self.corelog.event(self.stage, "stop", elapsed_ms=elapsed_ms)
            dur_txt = _fmt_hms_ms(elapsed_ms)
            self.corelog.info(f"[{self.stage}] ok in {dur_txt}")
            self.stats.mark(self.stage, elapsed_ms)
            return False


# Default data structures
SEGMENT_COLUMNS = [
    "file_id",
    "start",
    "end",
    "speaker_id",
    "speaker_name",
    "text",
    "valence",
    "arousal",
    "dominance",
    "emotion_top",
    "emotion_scores_json",
    "text_emotions_top5_json",
    "text_emotions_full_json",
    "intent_top",
    "intent_top3_json",
    "low_confidence_ser",
    "vad_unstable",
    "affect_hint",
    "asr_logprob_avg",
    "snr_db",
    "wpm",
    "pause_count",
    "pause_time_s",
    "f0_mean_hz",
    "f0_std_hz",
    "loudness_rms",
    "disfluency_count",
    "error_flags",
    # Voice-quality metrics (if available)
    "vq_jitter_pct",
    "vq_shimmer_db",
    "vq_hnr_db",
    "vq_cpps_db",
    "voice_quality_hint",
]


def default_affect() -> dict[str, Any]:
    ser_scores = {"neutral": 1.0}
    text_full = {"neutral": 1.0}
    return {
        "vad": {"valence": 0.0, "arousal": 0.0, "dominance": 0.0},
        "speech_emotion": {
            "top": "neutral",
            "scores_8class": ser_scores,
            "low_confidence_ser": True,
        },
        "text_emotions": {
            "top5": [{"label": "neutral", "score": 1.0}],
            "full_28class": text_full,
        },
        "intent": {
            "top": "status_update",
            "top3": [
                {"label": "status_update", "score": 1.0},
                {"label": "small_talk", "score": 0.0},
                {"label": "opinion", "score": 0.0},
            ],
        },
        "affect_hint": "neutral-status",
    }


def ensure_segment_keys(seg: dict[str, Any]) -> dict[str, Any]:
    for k in SEGMENT_COLUMNS:
        if k not in seg:
            if k in ("error_flags",):
                seg[k] = ""
            elif k in ("low_confidence_ser", "vad_unstable"):
                seg[k] = False
            else:
                seg[k] = None
    return seg


# Main Pipeline Class
class AudioAnalysisPipelineV2:
    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}

        self.run_id = cfg.get("run_id") or (
            time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        )
        self.schema_version = "2.0.0"

        # Paths
        self.log_dir = Path(cfg.get("log_dir", "logs"))
        self.cache_root = Path(cfg.get("cache_root", ".cache"))
        # Support multiple cache roots for reading (first is primary for writes)
        extra_roots = cfg.get("cache_roots", [])
        if isinstance(extra_roots, str | Path):
            extra_roots = [extra_roots]
        self.cache_roots: list[Path] = [self.cache_root] + [
            Path(p) for p in extra_roots
        ]
        self.cache_root.mkdir(parents=True, exist_ok=True)

        # Persist config for later checks
        self.cfg = dict(cfg)

        # Quiet mode env + logging
        self.quiet = bool(cfg.get("quiet", False))
        if self.quiet:
            import os as _os

            _os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            _os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
            _os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            _os.environ.setdefault("CT2_VERBOSE", "0")
            try:
                from transformers.utils.logging import set_verbosity_error as _setv

                _setv()
            except Exception:
                pass

        # Logger & Stats
        self.corelog = CoreLogger(
            self.run_id,
            self.log_dir / "run.jsonl",
            console_level=(logging.WARNING if self.quiet else logging.INFO),
        )
        self.stats = RunStats(
            run_id=self.run_id, file_id="", schema_version=self.schema_version
        )

        # Checkpoint manager
        self.checkpoints = PipelineCheckpointManager(
            cfg.get("checkpoint_dir", "checkpoints")
        )

        # Optional early dependency verification
        if bool(cfg.get("validate_dependencies", False)):
            ok, problems = _verify_core_dependencies(
                require_versions=bool(cfg.get("strict_dependency_versions", False))
            )
            if not ok:
                raise RuntimeError(
                    "Dependency verification failed:\n  - " + "\n  - ".join(problems)
                )

        # Initialize components with error handling
        self._init_components(cfg)

    def _init_components(self, cfg: dict[str, Any]):
        """Initialize pipeline components with graceful error handling"""
        try:
            # Preprocessor
            denoise_mode = (
                "spectral_sub_soft" if cfg.get("noise_reduction", True) else "none"
            )
            self.pp_conf = PreprocessConfig(
                target_sr=cfg.get("target_sr", 16000),
                denoise=denoise_mode,
                loudness_mode=cfg.get("loudness_mode", "asr"),
                auto_chunk_enabled=cfg.get("auto_chunk_enabled", True),
                chunk_threshold_minutes=cfg.get("chunk_threshold_minutes", 30.0),
                chunk_size_minutes=cfg.get("chunk_size_minutes", 20.0),
                chunk_overlap_seconds=cfg.get("chunk_overlap_seconds", 30.0),
            )
            self.pre = AudioPreprocessor(self.pp_conf)

            # Diarizer
            registry_path = cfg.get(
                "registry_path", str(Path("registry") / "speaker_registry.json")
            )
            if not Path(registry_path).is_absolute():
                registry_path = str(Path.cwd() / registry_path)

            ecapa_path = cfg.get("ecapa_model_path")
            search_paths = [
                ecapa_path,
                WINDOWS_MODELS_ROOT / "ecapa_tdnn.onnx"
                if WINDOWS_MODELS_ROOT
                else None,
                Path("models") / "ecapa_tdnn.onnx",
                Path("..") / "models" / "ecapa_tdnn.onnx",
                Path("..") / "diaremot" / "models" / "ecapa_tdnn.onnx",
                Path("..") / ".." / "models" / "ecapa_tdnn.onnx",
            ]
            resolved_path = None
            for candidate in search_paths:
                if not candidate:
                    continue
                candidate_path = Path(candidate).expanduser()
                if not candidate_path.is_absolute():
                    candidate_path = Path.cwd() / candidate_path
                if candidate_path.exists():
                    resolved_path = str(candidate_path.resolve())
                    break
            ecapa_path = resolved_path
            # Create diarization config first
            self.diar_conf = DiarizationConfig(
                target_sr=self.pp_conf.target_sr,
                registry_path=registry_path,
                ahc_distance_threshold=cfg.get("ahc_distance_threshold", 0.02),
                speaker_limit=cfg.get("speaker_limit", None),
                ecapa_model_path=ecapa_path,
                vad_backend=cfg.get("vad_backend", "auto"),
                # Allow CLI to tune VAD
                vad_threshold=cfg.get("vad_threshold", DiarizationConfig.vad_threshold),
                vad_min_speech_sec=cfg.get(
                    "vad_min_speech_sec", DiarizationConfig.vad_min_speech_sec
                ),
                vad_min_silence_sec=cfg.get(
                    "vad_min_silence_sec", DiarizationConfig.vad_min_silence_sec
                ),
                speech_pad_sec=cfg.get(
                    "vad_speech_pad_sec", DiarizationConfig.speech_pad_sec
                ),
                allow_energy_vad_fallback=not bool(
                    cfg.get("disable_energy_vad_fallback", False)
                ),
                energy_gate_db=cfg.get(
                    "energy_gate_db", DiarizationConfig.energy_gate_db
                ),
                energy_hop_sec=cfg.get(
                    "energy_hop_sec", DiarizationConfig.energy_hop_sec
                ),
            )
            # Make Silero VAD less strict to avoid energy-VAD fallback
            try:
                # Only relax defaults if user did not override via CLI
                if "vad_threshold" not in cfg:
                    self.diar_conf.vad_threshold = 0.22
                if "vad_min_speech_sec" not in cfg:
                    self.diar_conf.vad_min_speech_sec = 0.40
                if "vad_min_silence_sec" not in cfg:
                    self.diar_conf.vad_min_silence_sec = 0.40
                if "vad_speech_pad_sec" not in cfg:
                    self.diar_conf.speech_pad_sec = 0.15
            except Exception:
                pass

            # Diarizer: baseline by default; optional CPU-optimized wrapper behind a flag
            self.diar = SpeakerDiarizer(self.diar_conf)
            if bool(cfg.get("cpu_diarizer", False)):
                try:
                    from .cpu_optimized_diarizer import (
                        CPUOptimizationConfig,
                        CPUOptimizedSpeakerDiarizer,
                    )

                    cpu_conf = CPUOptimizationConfig(
                        max_speakers=self.diar_conf.speaker_limit
                    )
                    self.diar = CPUOptimizedSpeakerDiarizer(self.diar, cpu_conf)
                    self.corelog.info("[diarizer] using CPU-optimized wrapper")
                except Exception as _e:
                    self.corelog.warn(
                        f"[diarizer] CPU wrapper unavailable, using baseline: {_e}"
                    )

            # Transcriber - Force CPU-only configuration
            from .transcription_module import AudioTranscriber

            transcriber_config = {
                "model_size": str(cfg.get("whisper_model", DEFAULT_WHISPER_MODEL)),
                # Device selection is handled internally by the transcription backends
                "language": cfg.get("language", None),
                "beam_size": cfg.get("beam_size", 1),
                "temperature": cfg.get("temperature", 0.0),
                "compression_ratio_threshold": cfg.get(
                    "compression_ratio_threshold", 2.5
                ),
                "log_prob_threshold": cfg.get("log_prob_threshold", -1.0),
                "no_speech_threshold": cfg.get("no_speech_threshold", 0.50),
                "condition_on_previous_text": cfg.get(
                    "condition_on_previous_text", False
                ),
                "word_timestamps": cfg.get("word_timestamps", True),
                "max_asr_window_sec": cfg.get("max_asr_window_sec", 480),
                "vad_min_silence_ms": cfg.get("vad_min_silence_ms", 1800),
                "language_mode": cfg.get("language_mode", "auto"),
                # Backend tuning
                "compute_type": cfg.get("compute_type", None),
                "cpu_threads": cfg.get("cpu_threads", None),
                "asr_backend": cfg.get("asr_backend", "auto"),
                # Timeouts
                "segment_timeout_sec": cfg.get("segment_timeout_sec", 300.0),
                "batch_timeout_sec": cfg.get("batch_timeout_sec", 1200.0),
            }

            # Set CPU-only environment variables
            import os

            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            os.environ["TORCH_DEVICE"] = "cpu"

            self.tx = AudioTranscriber(**transcriber_config)

            # Affect analyzer (optional)
            if cfg.get("disable_affect"):
                self.affect = None
            else:
                self.affect = EmotionIntentAnalyzer(
                    text_emotion_model=cfg.get(
                        "text_emotion_model", "SamLowe/roberta-base-go_emotions"
                    ),
                    intent_labels=cfg.get("intent_labels", INTENT_LABELS_DEFAULT),
                )

            # Optional background SED / noise tagger
            self.sed_tagger = None
            try:
                if PANNSEventTagger is not None and bool(cfg.get("enable_sed", True)):
                    self.sed_tagger = PANNSEventTagger(
                        SEDConfig() if SEDConfig else None
                    )
            except Exception:
                self.sed_tagger = None

            # HTML & PDF generators
            self.html = HTMLSummaryGenerator()
            self.pdf = PDFSummaryGenerator()

            # Model tracking
            self.stats.models.update(
                {
                    "preprocessor": getattr(
                        self.pre, "__class__", type(self.pre)
                    ).__name__,
                    "diarizer": getattr(
                        self.diar, "__class__", type(self.diar)
                    ).__name__,
                    "transcriber": getattr(
                        self.tx, "__class__", type(self.tx)
                    ).__name__,
                    "affect": getattr(
                        self.affect, "__class__", type(self.affect)
                    ).__name__,
                }
            )

            # Config snapshot
            self.stats.config_snapshot = {
                "target_sr": self.pp_conf.target_sr,
                "noise_reduction": cfg.get("noise_reduction", True),
                "registry_path": self.diar_conf.registry_path,
                "ahc_distance_threshold": self.diar_conf.ahc_distance_threshold,
                "whisper_model": str(cfg.get("whisper_model", DEFAULT_WHISPER_MODEL)),
                "beam_size": cfg.get("beam_size", 1),
                "temperature": cfg.get("temperature", 0.0),
                "no_speech_threshold": cfg.get("no_speech_threshold", 0.50),
                "intent_labels": cfg.get("intent_labels", INTENT_LABELS_DEFAULT),
            }

        except Exception as e:
            self.corelog.error(f"Component initialization error: {e}")
            # Ensure minimal components exist even on init failure
            try:
                if not hasattr(self, "pre"):
                    self.pre = AudioPreprocessor(PreprocessConfig())
            except Exception:
                self.pre = None
            try:
                if not hasattr(self, "diar"):
                    self.diar = SpeakerDiarizer(DiarizationConfig(target_sr=16000))
            except Exception:
                self.diar = None
            try:
                if not hasattr(self, "tx"):
                    from .transcription_module import AudioTranscriber

                    self.tx = AudioTranscriber()
            except Exception:
                pass
            try:
                if not hasattr(self, "affect"):
                    self.affect = EmotionIntentAnalyzer()
            except Exception:
                pass
            try:
                if not hasattr(self, "pdf"):
                    self.pdf = PDFSummaryGenerator()
            except Exception:
                pass

    def _affect_hint(self, v, a, d, intent):
        try:
            if a is None or v is None:
                return "neutral-status"
            if a > 0.5 and v < 0:
                return "agitated-negative"
            if a < 0.3 and v > 0.2:
                return "calm-positive"
            return f"neutral-{intent}"
        except Exception:
            return "neutral-status"

    def _affect_unified(self, wav: np.ndarray, sr: int, text: str) -> dict[str, Any]:
        try:
            if hasattr(self.affect, "analyze"):
                res = self.affect.analyze(wav=wav, sr=sr, text=text)
                return res or default_affect()

            # Fallback implementation
            return default_affect()

        except Exception as e:
            self.corelog.warn(f"Affect analysis failed: {e}")
            return default_affect()

    def _extract_paraling(
        self, wav: np.ndarray, sr: int, segs: list[dict[str, Any]]
    ) -> dict[int, dict[str, Any]]:
        """Extract paralinguistic features with fallback"""
        results: dict[int, dict[str, Any]] = {}

        try:
            if para and hasattr(para, "extract"):
                out = para.extract(wav, sr, segs) or []
                for i, d in enumerate(out):
                    results[i] = {
                        "wpm": float(d.get("wpm", 0.0) or 0.0),
                        "pause_count": int(d.get("pause_count", 0) or 0),
                        "pause_time_s": float(d.get("pause_time_s", 0.0) or 0.0),
                        "f0_mean_hz": float(d.get("f0_mean_hz", 0.0) or 0.0),
                        "f0_std_hz": float(d.get("f0_std_hz", 0.0) or 0.0),
                        "loudness_rms": float(d.get("loudness_rms", 0.0) or 0.0),
                        "disfluency_count": int(d.get("disfluency_count", 0) or 0),
                        "vq_jitter_pct": float(d.get("vq_jitter_pct", 0.0) or 0.0),
                        "vq_shimmer_db": float(d.get("vq_shimmer_db", 0.0) or 0.0),
                        "vq_hnr_db": float(d.get("vq_hnr_db", 0.0) or 0.0),
                        "vq_cpps_db": float(d.get("vq_cpps_db", 0.0) or 0.0),
                    }
                return results
        except Exception as e:
            self.corelog.warn(f"[paralinguistics] fallback: {e}")

        # Lightweight fallback
        for i, s in enumerate(segs):
            start = float(s.get("start", 0.0) or 0.0)
            end = float(s.get("end", 0.0) or 0.0)
            dur = max(1e-6, end - start)
            txt = s.get("text") or ""
            words = max(0, len(txt.split()))
            wpm = (words / dur) * 60.0 if dur > 0 else 0.0

            i0 = int(start * sr)
            i1 = int(end * sr)
            clip = wav[max(0, i0) : max(0, i1)]
            loud = (
                float(np.sqrt(np.mean(clip.astype(np.float32) ** 2)))
                if clip.size > 0
                else 0.0
            )

            results[i] = {
                "wpm": float(wpm),
                "pause_count": 0,
                "pause_time_s": 0.0,
                "f0_mean_hz": 0.0,
                "f0_std_hz": 0.0,
                "loudness_rms": float(loud),
                "disfluency_count": 0,
            }
        return results

    def _write_outputs(
        self,
        input_audio_path: str,
        outp: Path,
        segments_final: list[dict[str, Any]],
        speakers_summary: list[dict[str, Any]],
        health: Any,
        turns: list[dict[str, Any]],
        overlap_stats: dict[str, Any],
        per_speaker_interrupts: dict[str, Any],
        conv_metrics: ConversationMetrics | None,
        duration_s: float,
    ):
        """Write all output files"""
        # Primary CSV
        self._write_csv(outp / "diarized_transcript_with_emotion.csv", segments_final)

        # JSONL segments
        self._write_jsonl(outp / "segments.jsonl", segments_final)

        # Timeline
        self._write_timeline(outp / "timeline.csv", segments_final)

        # QC report
        self._write_qc(
            outp / "qc_report.json",
            self.stats,
            health,
            n_turns=len(turns),
            n_segments=len(segments_final),
            segments=segments_final,
        )

        # Speakers summary
        if speakers_summary:
            headers = sorted({k for r in speakers_summary for k in r.keys()})
            with (outp / "speakers_summary.csv").open(
                "w", newline="", encoding="utf-8"
            ) as f:
                w = csv.DictWriter(f, fieldnames=headers)
                w.writeheader()
                for r in speakers_summary:
                    w.writerow({k: r.get(k, None) for k in headers})

        # HTML summary
        try:
            html_path = self.html.render_to_html(
                out_dir=str(outp),
                file_id=self.stats.file_id,
                segments=segments_final,
                speakers_summary=speakers_summary,
                overlap_stats=overlap_stats,
            )
        except (RuntimeError, ValueError, OSError, ImportError) as e:
            html_path = None
            self.corelog.warn(
                f"HTML summary skipped: {e}. Verify HTML template assets or install report dependencies."
            )

        # PDF summary
        try:
            pdf_path = self.pdf.render_to_pdf(
                out_dir=str(outp),
                file_id=self.stats.file_id,
                segments=segments_final,
                speakers_summary=speakers_summary,
                overlap_stats=overlap_stats,
            )
        except (RuntimeError, ValueError, OSError, ImportError) as e:
            pdf_path = None
            self.corelog.warn(
                f"PDF summary skipped: {e}. Ensure wkhtmltopdf/LaTeX prerequisites are installed."
            )

        self.checkpoints.create_checkpoint(
            input_audio_path,
            ProcessingStage.SUMMARY_GENERATION,
            {"html": html_path, "pdf": pdf_path},
            progress=90.0,
        )

    def process_audio_file(self, input_audio_path: str, out_dir: str) -> dict[str, Any]:
        """Main processing function with robust checkpoint system"""
        outp = Path(out_dir)
        outp.mkdir(parents=True, exist_ok=True)
        self.stats.file_id = Path(input_audio_path).name

        # Initialize all variables to prevent UnboundLocalError
        y = np.array([])
        sr = 16000
        health = None
        sed_info = None
        tx_out = []
        norm_tx = []
        speakers_summary = []
        turns = []
        segments_final = []
        conv_metrics = None
        duration_s = 0.0
        overlap_stats = {}
        per_speaker_interrupts = {}

        # Checkpoint variables
        audio_sha16 = ""
        pp_sig = {}
        cache_dir = None
        diar_cache = None
        tx_cache = None
        resume_diar = False
        resume_tx = False

        try:
            # ========== 0) Dependency Check (informational) ==========
            with StageGuard(self.corelog, self.stats, "dependency_check"):
                dep_summary = _dependency_health_summary()
                unhealthy = [
                    k for k, v in dep_summary.items() if v.get("status") != "ok"
                ]
                self.corelog.event("dependency_check", "summary", unhealthy=unhealthy)
                if unhealthy:
                    self.corelog.warn(f"Dependency issues detected: {unhealthy}")
                    for k in unhealthy:
                        issue = dep_summary[k].get("issue")
                        if issue:
                            self.stats.warnings.append(f"dep:{k}: {issue}")
                try:
                    self.stats.config_snapshot["dependency_ok"] = len(unhealthy) == 0
                    self.stats.config_snapshot["dependency_summary"] = dep_summary
                except Exception:
                    pass

            # ========== 1) Preprocess ==========
            if not hasattr(self, "pre") or self.pre is None:
                raise RuntimeError(
                    "Preprocessor component unavailable; initialization failed"
                )
            with StageGuard(self.corelog, self.stats, "preprocess"):
                y, sr, health = self.pre.process_file(input_audio_path)
                duration_s = float(len(y) / sr) if sr else 0.0
                self.corelog.info(f"[preprocess] file duration {_fmt_hms(duration_s)}")
                self.corelog.event(
                    "preprocess",
                    "metrics",
                    duration_s=duration_s,
                    snr_db=float(getattr(health, "snr_db", 0.0)) if health else None,
                )
            audio_sha16 = _compute_audio_sha16(y)
            if audio_sha16:
                self.checkpoints.seed_file_hash(input_audio_path, audio_sha16)

            self.checkpoints.create_checkpoint(
                input_audio_path,
                ProcessingStage.AUDIO_PREPROCESSING,
                {"sr": sr},
                progress=5.0,
                file_hash=audio_sha16,
            )
            # ========== 1b) Background SED (optional) ==========
            with StageGuard(self.corelog, self.stats, "background_sed"):
                try:
                    if (
                        getattr(self, "sed_tagger", None) is not None
                        and y.size > 0
                        and sr
                    ):
                        sed_info = self.sed_tagger.tag(y, sr)
                        if sed_info:
                            self.corelog.event(
                                "background_sed",
                                "tags",
                                dominant_label=sed_info.get("dominant_label"),
                                noise_score=sed_info.get("noise_score"),
                            )
                            self.stats.config_snapshot["background_sed"] = sed_info
                except (
                    ImportError,
                    ModuleNotFoundError,
                    RuntimeError,
                    ValueError,
                    OSError,
                ) as e:
                    self.corelog.warn(
                        "[sed] tagging skipped: "
                        f"{e}. Install sed_panns dependencies or disable background SED."
                    )

            # Compute checkpoint signatures
            pp_sig = _compute_pp_signature(self.pp_conf)
            cache_dir = self.cache_root / audio_sha16
            cache_dir.mkdir(parents=True, exist_ok=True)
            diar_path = cache_dir / "diar.json"
            tx_path = cache_dir / "tx.json"

            # Check caches across all configured roots (read), prefer primary root
            diar_cache = _read_json_safe(diar_path)
            tx_cache = _read_json_safe(tx_path)
            diar_cache_src = str(diar_path) if diar_cache else None
            tx_cache_src = str(tx_path) if tx_cache else None
            if not diar_cache or not tx_cache:
                for root in self.cache_roots[1:]:
                    alt_dir = Path(root) / audio_sha16
                    alt_diar = _read_json_safe(alt_dir / "diar.json")
                    alt_tx = _read_json_safe(alt_dir / "tx.json")
                    if not diar_cache and alt_diar:
                        diar_cache = alt_diar
                        diar_cache_src = str(alt_dir / "diar.json")
                    if not tx_cache and alt_tx:
                        tx_cache = alt_tx
                        tx_cache_src = str(alt_dir / "tx.json")
                    if diar_cache and tx_cache:
                        break

            def _cache_matches(obj):
                return (
                    obj
                    and obj.get("version") == CACHE_VERSION
                    and obj.get("audio_sha16") == audio_sha16
                    and obj.get("pp_signature") == pp_sig
                )

            if _cache_matches(tx_cache):
                resume_tx = True
                # Only resume diarization if diar cache also matches
                resume_diar = bool(_cache_matches(diar_cache))
                if resume_diar:
                    self.corelog.info(
                        "[resume] using tx.json+diar.json caches; skipping diarize+ASR"
                    )
                else:
                    self.corelog.info(
                        "[resume] using tx.json cache; skipping ASR and reconstructing turns from tx cache"
                    )
                self.corelog.event(
                    "resume",
                    "tx_cache_hit",
                    audio_sha16=audio_sha16,
                    src=tx_cache_src,
                )
            elif _cache_matches(diar_cache):
                resume_diar = True
                self.corelog.info("[resume] using diar.json cache; skipping diarize")
                self.corelog.event(
                    "resume",
                    "diar_cache_hit",
                    audio_sha16=audio_sha16,
                    src=diar_cache_src,
                )

            # Optionally ignore caches to force diarize/ASR
            if self.cfg.get("ignore_tx_cache"):
                diar_cache = None
                tx_cache = None
                resume_diar = False
                resume_tx = False

            # ========== 2) Diarize ==========
            vad_unstable = False
            with StageGuard(self.corelog, self.stats, "diarize") as g:
                if (
                    resume_tx
                    and (not diar_cache)
                    and tx_cache
                    and tx_cache.get("segments")
                ):
                    # Reconstruct lightweight turns directly from tx cache
                    try:
                        tx_segments = tx_cache.get("segments", []) or []
                        turns = []
                        for d in tx_segments:
                            try:
                                s = float(
                                    d.get("start", d.get("start_time", 0.0)) or 0.0
                                )
                                e = float(d.get("end", d.get("end_time", 0.0)) or 0.0)
                                sid = str(
                                    d.get("speaker_id", d.get("speaker", "Speaker_1"))
                                )
                                sname = d.get("speaker_name") or sid
                                if e < s:
                                    s, e = e, s
                                turns.append(
                                    {
                                        "start": s,
                                        "end": e,
                                        "speaker": sid,
                                        "speaker_name": sname,
                                    }
                                )
                            except (TypeError, ValueError, KeyError, AttributeError):
                                continue
                        # Basic sanity: ensure at least one turn spans audio if nothing valid
                        if not turns:
                            turns = [
                                {
                                    "start": 0.0,
                                    "end": duration_s,
                                    "speaker": "Speaker_1",
                                    "speaker_name": "Speaker_1",
                                }
                            ]
                        g.done(
                            turns=len(turns),
                            speakers_est=len(set([t.get("speaker") for t in turns])),
                        )
                    except (AttributeError, KeyError, TypeError, ValueError) as e:
                        self.corelog.warn(
                            "Failed to reconstruct turns from tx cache: "
                            f"{e}; rerunning diarization to rebuild cache integrity."
                        )
                        try:
                            turns = self.diar.diarize_audio(y, sr) or []
                        except (
                            RuntimeError,
                            ValueError,
                            OSError,
                            subprocess.CalledProcessError,
                        ) as e2:
                            self.corelog.warn(
                                "Diarization failed: "
                                f"{e2}; using single-speaker fallback. Check VAD thresholds or model availability."
                            )
                            turns = [
                                {
                                    "start": 0.0,
                                    "end": duration_s,
                                    "speaker": "Speaker_1",
                                    "speaker_name": "Speaker_1",
                                }
                            ]
                        if not turns:
                            self.corelog.warn(
                                "Diarizer returned 0 turns; using fallback"
                            )
                            turns = [
                                {
                                    "start": 0.0,
                                    "end": duration_s,
                                    "speaker": "Speaker_1",
                                    "speaker_name": "Speaker_1",
                                }
                            ]
                        g.done(
                            turns=len(turns),
                            speakers_est=len(set([t.get("speaker") for t in turns])),
                        )
                elif resume_diar and diar_cache:
                    turns = diar_cache.get("turns", []) or []
                    if not turns:
                        self.corelog.warn(
                            "Cached diar.json has 0 turns; proceeding with fallback"
                        )
                        turns = [
                            {
                                "start": 0.0,
                                "end": duration_s,
                                "speaker": "Speaker_1",
                                "speaker_name": "Speaker_1",
                            }
                        ]

                    try:
                        vad_toggles = sum(1 for t in turns if t.get("is_boundary_flip"))
                    except (TypeError, AttributeError):
                        vad_toggles = 0
                    vad_unstable = (
                        vad_toggles / max(1, int(duration_s / 60) or 1)
                    ) > 60
                    g.done(
                        turns=len(turns),
                        speakers_est=len(set([t.get("speaker") for t in turns])),
                    )
                else:
                    try:
                        turns = self.diar.diarize_audio(y, sr) or []
                    except (
                        RuntimeError,
                        ValueError,
                        OSError,
                        subprocess.CalledProcessError,
                    ) as e:
                        self.corelog.warn(
                            "Diarization failed: "
                            f"{e}; reverting to single-speaker assumption. Verify ECAPA/pyannote assets."
                        )
                        turns = [
                            {
                                "start": 0.0,
                                "end": duration_s,
                                "speaker": "Speaker_1",
                                "speaker_name": "Speaker_1",
                            }
                        ]

                    if not turns:
                        self.corelog.warn("Diarizer returned 0 turns; using fallback")
                        turns = [
                            {
                                "start": 0.0,
                                "end": duration_s,
                                "speaker": "Speaker_1",
                                "speaker_name": "Speaker_1",
                            }
                        ]

                    try:
                        vad_toggles = sum(1 for t in turns if t.get("is_boundary_flip"))
                    except (TypeError, AttributeError):
                        vad_toggles = 0
                    vad_unstable = (
                        vad_toggles / max(1, int(duration_s / 60) or 1)
                    ) > 60
                    g.done(
                        turns=len(turns),
                        speakers_est=len(set([t.get("speaker") for t in turns])),
                    )

                    # Save diarization cache
                    try:
                        # Ensure turns are JSON-serializable (e.g., convert numpy embeddings)
                        def _jsonable_turns(_turns):
                            out = []
                            for _t in _turns:
                                try:
                                    tcopy = dict(_t)
                                except (TypeError, ValueError, AttributeError):
                                    # Fallback for non-dict-like items
                                    continue
                                emb = tcopy.get("embedding")
                                try:
                                    import numpy as _np  # local import to avoid top-level dependency issues

                                    if isinstance(emb, _np.ndarray):
                                        tcopy["embedding"] = emb.tolist()
                                    elif hasattr(emb, "tolist"):
                                        tcopy["embedding"] = emb.tolist()
                                    elif emb is not None and not isinstance(
                                        emb, list | float | int | str | bool
                                    ):
                                        tcopy["embedding"] = None
                                except (
                                    ImportError,
                                    AttributeError,
                                    TypeError,
                                    ValueError,
                                ):
                                    # If numpy not available or any issue, drop embedding
                                    tcopy["embedding"] = None
                                out.append(tcopy)
                            return out

                        _atomic_write_json(
                            diar_path,
                            {
                                "version": CACHE_VERSION,
                                "audio_sha16": audio_sha16,
                                "pp_signature": pp_sig,
                                "turns": _jsonable_turns(turns),
                                "saved_at": time.time(),
                            },
                        )
                    except OSError as e:
                        self.corelog.warn(
                            f"[cache] diar.json write failed: {e}. Ensure cache directory is writable."
                        )

            self.checkpoints.create_checkpoint(
                input_audio_path,
                ProcessingStage.DIARIZATION,
                turns,
                progress=30.0,
            )

            # Registry update (safe)
            with StageGuard(self.corelog, self.stats, "registry_update"):
                try:
                    if (
                        hasattr(self.diar, "registry")
                        and self.diar.registry is not None
                    ):
                        # Update registry centroids if possible
                        pass  # Implementation depends on your registry interface
                except (RuntimeError, ValueError, OSError) as e:
                    self.corelog.warn(
                        f"[registry] update skipped: {e}. Review speaker registry permissions and schema."
                    )

            # ========== 3) Transcribe ==========
            tx_in = []
            speaker_name_map = {}
            for t in turns:
                start = float(t.get("start", t.get("start_time", 0.0)) or 0.0)
                end = float(
                    t.get("end", t.get("end_time", start + 0.5)) or (start + 0.5)
                )
                sid = str(t.get("speaker"))
                sname = t.get("speaker_name") or sid
                speaker_name_map[sid] = sname
                tx_in.append(
                    {
                        "start_time": start,
                        "end_time": end,
                        "speaker_id": sid,
                        "speaker_name": sname,
                    }
                )

            with StageGuard(self.corelog, self.stats, "transcribe") as g:
                if resume_tx and tx_cache:
                    tx_out = []
                    g.done(segments=len(tx_cache.get("segments", []) or []))
                else:
                    try:
                        tx_out = self.tx.transcribe_segments(y, sr, tx_in) or []
                    except (
                        RuntimeError,
                        TimeoutError,
                        subprocess.CalledProcessError,
                    ) as e:
                        self.corelog.warn(
                            "Transcription failed: "
                            f"{e}; generating placeholder segments. Verify faster-whisper setup or tune --asr-segment-timeout."
                        )
                        tx_out = []
                        # Create fallback transcription segments
                        for seg in tx_in:
                            tx_out.append(
                                type(
                                    "TranscriptionSegment",
                                    (),
                                    {
                                        "start_time": seg["start_time"],
                                        "end_time": seg["end_time"],
                                        "text": "[Transcription unavailable]",
                                        "speaker_id": seg["speaker_id"],
                                        "speaker_name": seg["speaker_name"],
                                        "asr_logprob_avg": None,
                                        "snr_db": None,
                                    },
                                )()
                            )

                    g.done(segments=len(tx_out))

            # Normalize transcription output
            if resume_tx and tx_cache and (tx_cache.get("segments") is not None):
                for d in tx_cache.get("segments", []):
                    norm_tx.append(
                        {
                            "start": float(
                                d.get("start", 0.0) or d.get("start_time", 0.0) or 0.0
                            ),
                            "end": float(
                                d.get("end", 0.0) or d.get("end_time", 0.0) or 0.0
                            ),
                            "speaker_id": d.get("speaker_id"),
                            "speaker_name": d.get("speaker_name"),
                            "text": d.get("text", ""),
                            "asr_logprob_avg": d.get("asr_logprob_avg"),
                            "snr_db": d.get("snr_db"),
                            "error_flags": d.get("error_flags", ""),
                        }
                    )
            else:
                for it in tx_out:
                    d = (
                        it.__dict__
                        if hasattr(it, "__dict__")
                        else dict(it)
                        if isinstance(it, dict)
                        else {}
                    )
                    norm_tx.append(
                        {
                            "start": float(
                                d.get("start_time", d.get("start", 0.0)) or 0.0
                            ),
                            "end": float(d.get("end_time", d.get("end", 0.0)) or 0.0),
                            "speaker_id": d.get("speaker_id"),
                            "speaker_name": d.get("speaker_name"),
                            "text": d.get("text", ""),
                            "asr_logprob_avg": d.get("asr_logprob_avg"),
                            "snr_db": d.get("snr_db"),
                            "error_flags": "",
                        }
                    )

            self.checkpoints.create_checkpoint(
                input_audio_path,
                ProcessingStage.TRANSCRIPTION,
                norm_tx,
                progress=60.0,
            )

            # Save transcription cache
            try:
                _atomic_write_json(
                    tx_path,
                    {
                        "version": CACHE_VERSION,
                        "audio_sha16": audio_sha16,
                        "pp_signature": pp_sig,
                        "segments": norm_tx,
                        "saved_at": time.time(),
                    },
                )
            except OSError as e:
                self.corelog.warn(
                    f"[cache] tx.json write failed: {e}. Check disk space and cache directory permissions."
                )

            # ========== 4) Paralinguistics ==========
            para_metrics: dict[int, dict[str, Any]] = {}
            with StageGuard(self.corelog, self.stats, "paralinguistics"):
                if not self.stats.config_snapshot.get("transcribe_failed"):
                    tmp_para_metrics = self._extract_paraling(y, sr, norm_tx)
                    if isinstance(tmp_para_metrics, dict):
                        para_metrics = tmp_para_metrics
                    else:
                        para_metrics = {}

            # Memory cleanup after large operations
            try:
                import gc

                gc.collect()
            except Exception:
                pass

            # ========== 5) Affect Analysis & Assembly ==========
            with StageGuard(self.corelog, self.stats, "affect_and_assemble") as g:
                if self.stats.config_snapshot.get("transcribe_failed"):
                    segments_final = []
                else:
                    for i, seg in enumerate(norm_tx):
                        start = float(seg["start"] or 0.0)
                        end = float(seg["end"] or start)
                        i0 = int(start * sr)
                        i1 = int(end * sr)
                        clip = (
                            y[max(0, i0) : max(0, i1)] if len(y) > 0 else np.array([])
                        )
                        text = seg.get("text") or ""

                        aff = self._affect_unified(clip, sr, text)
                        v = aff["vad"].get("valence", 0.0)
                        a = aff["vad"].get("arousal", 0.0)
                        dmn = aff["vad"].get("dominance", 0.0)
                        ser_top = aff["speech_emotion"].get("top", "neutral")
                        ser_scores = aff["speech_emotion"].get(
                            "scores_8class", {"neutral": 1.0}
                        )
                        low_ser = bool(
                            aff["speech_emotion"].get("low_confidence_ser", False)
                        )
                        tx5 = aff["text_emotions"].get(
                            "top5", [{"label": "neutral", "score": 1.0}]
                        )
                        txfull = aff["text_emotions"].get(
                            "full_28class", {"neutral": 1.0}
                        )
                        intent_top = aff["intent"].get("top", "status_update")
                        intent_top3 = aff["intent"].get("top3", [])
                        hint = aff.get("affect_hint", "neutral-status")

                        pm = para_metrics.get(i, {})
                        row = {
                            "file_id": self.stats.file_id,
                            "start": start,
                            "end": end,
                            "speaker_id": seg.get("speaker_id"),
                            "speaker_name": seg.get("speaker_name"),
                            "text": text,
                            "valence": float(v) if v is not None else None,
                            "arousal": float(a) if a is not None else None,
                            "dominance": float(dmn) if dmn is not None else None,
                            "emotion_top": ser_top,
                            "emotion_scores_json": json.dumps(
                                ser_scores, ensure_ascii=False
                            ),
                            "text_emotions_top5_json": json.dumps(
                                tx5, ensure_ascii=False
                            ),
                            "text_emotions_full_json": json.dumps(
                                txfull, ensure_ascii=False
                            ),
                            "intent_top": intent_top,
                            "intent_top3_json": json.dumps(
                                intent_top3, ensure_ascii=False
                            ),
                            "low_confidence_ser": low_ser,
                            "vad_unstable": bool(vad_unstable),
                            "affect_hint": hint,
                            "asr_logprob_avg": seg.get("asr_logprob_avg"),
                            "snr_db": seg.get("snr_db"),
                            "wpm": pm.get("wpm", 0.0),
                            "pause_count": pm.get("pause_count", 0),
                            "pause_time_s": pm.get("pause_time_s", 0.0),
                            "f0_mean_hz": pm.get("f0_mean_hz", 0.0),
                            "f0_std_hz": pm.get("f0_std_hz", 0.0),
                            "loudness_rms": pm.get("loudness_rms", 0.0),
                            "disfluency_count": pm.get("disfluency_count", 0),
                            "vq_jitter_pct": pm.get("vq_jitter_pct"),
                            "vq_shimmer_db": pm.get("vq_shimmer_db"),
                            "vq_hnr_db": pm.get("vq_hnr_db"),
                            "vq_cpps_db": pm.get("vq_cpps_db"),
                            "voice_quality_hint": pm.get("vq_note"),
                            "error_flags": seg.get("error_flags", ""),
                        }
                        segments_final.append(ensure_segment_keys(row))

                g.done(segments=len(segments_final))

            # ========== 6) Overlap & Interruptions ==========
            with StageGuard(self.corelog, self.stats, "overlap_interruptions"):
                try:
                    if para and hasattr(para, "compute_overlap_and_interruptions"):
                        ov = para.compute_overlap_and_interruptions(turns) or {}
                    else:
                        ov = {}
                    overlap_stats = {
                        "overlap_total_sec": float(ov.get("overlap_total_sec", 0.0)),
                        "overlap_ratio": float(ov.get("overlap_ratio", 0.0)),
                    }
                    per_speaker_interrupts = ov.get("per_speaker", {}) or {}
                except (AttributeError, RuntimeError, ValueError) as e:
                    self.corelog.warn(
                        "[overlap] skipped: "
                        f"{e}. Install paralinguistics extras or validate overlap feature inputs."
                    )
                    overlap_stats = {"overlap_total_sec": 0.0, "overlap_ratio": 0.0}
                    per_speaker_interrupts = {}

            # ========== 7) Conversation Analysis ==========
            with StageGuard(self.corelog, self.stats, "conversation_analysis"):
                try:
                    conv_metrics = analyze_conversation_flow(segments_final, duration_s)
                    self.corelog.event(
                        "conversation_analysis",
                        "metrics",
                        balance=conv_metrics.turn_taking_balance,
                        pace=conv_metrics.conversation_pace_turns_per_min,
                        coherence=conv_metrics.topic_coherence_score,
                    )
                except (RuntimeError, ValueError, ZeroDivisionError) as e:
                    self.corelog.warn(
                        "Conversation analysis failed: "
                        f"{e}. Falling back to neutral conversational metrics."
                    )
                    try:
                        # Construct a minimal metrics object compatible with downstream
                        conv_metrics = ConversationMetrics(
                            turn_taking_balance=0.5,
                            interruption_rate_per_min=0.0,
                            avg_turn_duration_sec=0.0,
                            conversation_pace_turns_per_min=0.0,
                            silence_ratio=0.0,
                            speaker_dominance={},
                            response_latency_stats={},
                            topic_coherence_score=0.0,
                            energy_flow=[],
                        )
                    except (TypeError, ValueError):
                        conv_metrics = None

            # ========== 8) Speaker Rollups ==========
            with StageGuard(self.corelog, self.stats, "speaker_rollups"):
                try:
                    speakers_summary = build_speakers_summary(
                        segments_final, per_speaker_interrupts, overlap_stats
                    )
                    # Ensure it's a list of dicts
                    if isinstance(speakers_summary, dict):
                        speakers_summary = [
                            dict(v, speaker_id=k) for k, v in speakers_summary.items()
                        ]
                    elif not isinstance(speakers_summary, list):
                        speakers_summary = []
                except (RuntimeError, ValueError, TypeError) as e:
                    self.corelog.warn(
                        "Speaker rollups failed: "
                        f"{e}. Inspect segment records or disable speaker summary generation."
                    )
                    speakers_summary = []

            # ========== 9) Write Outputs ==========
            with StageGuard(self.corelog, self.stats, "outputs"):
                self._write_outputs(
                    input_audio_path,
                    outp,
                    segments_final,
                    speakers_summary,
                    health,
                    turns,
                    overlap_stats,
                    per_speaker_interrupts,
                    conv_metrics,
                    duration_s,
                )

                # Mark cache completion
                try:
                    (cache_dir / ".done").write_text("ok", encoding="utf-8")
                except OSError:
                    pass

        except Exception as e:
            self.corelog.error(f"Pipeline failed with unhandled error: {e}")
            # Ensure we have minimal outputs even on failure
            if not segments_final and norm_tx:
                segments_final = [
                    ensure_segment_keys(
                        {
                            "file_id": self.stats.file_id,
                            "start": seg.get("start", 0.0),
                            "end": seg.get("end", 0.0),
                            "speaker_id": seg.get("speaker_id", "Unknown"),
                            "speaker_name": seg.get("speaker_name", "Unknown"),
                            "text": seg.get("text", ""),
                        }
                    )
                    for seg in norm_tx
                ]

            # Try to write what we have
            try:
                self._write_outputs(
                    input_audio_path,
                    outp,
                    segments_final,
                    speakers_summary,
                    health,
                    turns,
                    overlap_stats,
                    per_speaker_interrupts,
                    conv_metrics,
                    duration_s,
                )
            except Exception as write_error:
                self.corelog.error(f"Failed to write outputs: {write_error}")

        # Return manifest
        outputs = {
            "csv": str((outp / "diarized_transcript_with_emotion.csv").resolve()),
            "jsonl": str((outp / "segments.jsonl").resolve()),
            "timeline": str((outp / "timeline.csv").resolve()),
            "summary_html": str((outp / "summary.html").resolve()),
            "summary_pdf": str((outp / "summary.pdf").resolve()),
            "qc_report": str((outp / "qc_report.json").resolve()),
            "speaker_registry": getattr(
                self.diar_conf,
                "registry_path",
                str(Path("registry") / "speaker_registry.json"),
            ),
        }

        spk_path = outp / "speakers_summary.csv"
        if spk_path.exists():
            outputs["speakers_summary"] = str(spk_path.resolve())

        manifest = {
            "run_id": self.run_id,
            "file_id": self.stats.file_id,
            "out_dir": str(outp.resolve()),
            "outputs": outputs,
        }

        # Console one-liner dependency summary
        try:
            dep_ok = bool(self.stats.config_snapshot.get("dependency_ok", True))
            dep_summary = self.stats.config_snapshot.get("dependency_summary", {}) or {}
            unhealthy = [k for k, v in dep_summary.items() if v.get("status") != "ok"]
            if dep_ok and not unhealthy:
                self.corelog.info("[deps] All core dependencies loaded successfully.")
            else:
                self.corelog.warn("[deps] Issues detected: " + ", ".join(unhealthy))
            # Expose in manifest
            manifest["dependency_ok"] = dep_ok and not unhealthy
            manifest["dependency_unhealthy"] = unhealthy
        except Exception:
            pass

        # Transcriber diagnostics (fallback reason, etc.)
        try:
            if hasattr(self, "tx") and hasattr(self.tx, "get_model_info"):
                tx_info = self.tx.get_model_info()
                manifest["transcriber"] = tx_info
                fb = tx_info.get("fallback_triggered")
                if fb:
                    self.corelog.warn(
                        "[tx] Fallback engaged: "
                        + str(tx_info.get("fallback_reason", "unknown"))
                    )
            # Include background SED info if available
            if "background_sed" in getattr(self.stats, "config_snapshot", {}):
                manifest["background_sed"] = self.stats.config_snapshot.get(
                    "background_sed"
                )
        except Exception:
            pass

        self.corelog.event("done", "stop", **manifest)
        # Final stage summary
        try:
            stages = [
                "dependency_check",
                "preprocess",
                "background_sed",
                "diarize",
                "transcribe",
                "paralinguistics",
                "affect_and_assemble",
                "overlap_interruptions",
                "conversation_analysis",
                "speaker_rollups",
                "outputs",
            ]
            failures = {f.get("stage"): f for f in getattr(self.stats, "failures", [])}
            self.corelog.info("[ALERT] Stage summary:")
            for st in stages:
                if st in failures:
                    f = failures[st]
                    ms = float(f.get("elapsed_ms", 0.0))
                    self.corelog.warn(
                        f"  - {st}: FAIL in {_fmt_hms_ms(ms)} — {f.get('error')} | Fix: {f.get('suggestion')}"
                    )
                else:
                    if st in (
                        "paralinguistics",
                        "affect_and_assemble",
                    ) and self.stats.config_snapshot.get("transcribe_failed"):
                        self.corelog.warn(f"  - {st}: SKIPPED (transcribe_failed)")
                    else:
                        ms = float(self.stats.stage_timings_ms.get(st, 0.0))
                        self.corelog.info(f"  - {st}: PASS in {_fmt_hms_ms(ms)}")
        except Exception:
            pass
        self.checkpoints.create_checkpoint(
            input_audio_path, ProcessingStage.COMPLETE, manifest, progress=100.0
        )
        return manifest

    # Helper methods for output generation
    def _write_csv(self, path: Path, rows: list[dict[str, Any]]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=SEGMENT_COLUMNS)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, None) for k in SEGMENT_COLUMNS})

    def _write_jsonl(self, path: Path, segments: list[dict[str, Any]]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for seg in segments:
                f.write(json.dumps(seg, ensure_ascii=False) + "\n")

    def _write_timeline(self, path: Path, segments: list[dict[str, Any]]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["start", "end", "speaker_id"])
            for s in segments:
                w.writerow(
                    [s.get("start", 0.0), s.get("end", 0.0), s.get("speaker_id", "")]
                )

    def _write_qc(
        self,
        path: Path,
        stats: RunStats,
        health: Any,
        n_turns: int,
        n_segments: int,
        segments: list[dict[str, Any]],
    ):
        payload = {
            "run_id": stats.run_id,
            "file_id": stats.file_id,
            "schema_version": stats.schema_version,
            "stage_timings_ms": stats.stage_timings_ms,
            "stage_counts": stats.stage_counts,
            "warnings": stats.warnings,
            "errors": getattr(stats, "errors", []),
            "failures": getattr(stats, "failures", []),
            "models": stats.models,
            "config_snapshot": stats.config_snapshot,
            "audio_health": {
                "snr_db": float(getattr(health, "snr_db", 0.0)) if health else None,
                "silence_ratio": (
                    float(getattr(health, "silence_ratio", 0.0)) if health else None
                ),
                "clipping_detected": (
                    bool(getattr(health, "clipping_detected", False))
                    if health
                    else None
                ),
                "dynamic_range_db": (
                    float(getattr(health, "dynamic_range_db", 0.0)) if health else None
                ),
            },
            "counts": {"turns": int(n_turns), "segments": int(n_segments)},
        }
        # Aggregate voice-quality metrics (if available on segments)
        try:

            def _avg(key: str):
                vals = []
                for s in segments or []:
                    v = s.get(key)
                    if v is None:
                        continue
                    try:
                        vals.append(float(v))
                    except Exception:
                        continue
                return float(sum(vals) / len(vals)) if vals else None

            vq_summary = {
                "vq_jitter_pct_avg": _avg("vq_jitter_pct"),
                "vq_shimmer_db_avg": _avg("vq_shimmer_db"),
                "vq_hnr_db_avg": _avg("vq_hnr_db"),
                "vq_cpps_db_avg": _avg("vq_cpps_db"),
            }
            payload["voice_quality_summary"] = vq_summary
        except Exception:
            pass
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _summarize_speakers(
        self,
        segments: list[dict[str, Any]],
        per_speaker_interrupts: dict[str, dict[str, Any]],
        overlap_stats: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        prof = {}
        for s in segments:
            sid = str(s.get("speaker_id", "Unknown"))
            start = float(s.get("start", 0.0) or 0.0)
            end = float(s.get("end", 0.0) or 0.0)
            dur = max(0.0, end - start)
            p = prof.setdefault(
                sid,
                {
                    "speaker_name": s.get("speaker_name"),
                    "total_duration": 0.0,
                    "word_count": 0,
                    "avg_wpm": 0.0,
                    "avg_valence": 0.0,
                    "avg_arousal": 0.0,
                    "avg_dominance": 0.0,
                    "interruptions_made": 0,
                    "interruptions_received": 0,
                    "overlap_ratio": 0.0,
                },
            )
            p["total_duration"] += dur
            words = len((s.get("text") or "").split())
            p["word_count"] += words

            # Update averages
            for k_src, k_dst in (
                ("valence", "avg_valence"),
                ("arousal", "avg_arousal"),
                ("dominance", "avg_dominance"),
            ):
                val = s.get(k_src, None)
                if val is not None:
                    prev = p[k_dst]
                    cnt = p.get("_n_" + k_dst, 0) + 1
                    p[k_dst] = (prev * (cnt - 1) + float(val)) / float(cnt)
                    p["_n_" + k_dst] = cnt

        # Add interrupt data
        for sid, vals in (per_speaker_interrupts or {}).items():
            p = prof.setdefault(str(sid), {})
            p["interruptions_made"] = int(vals.get("made", 0) or 0)
            p["interruptions_received"] = int(vals.get("received", 0) or 0)

        # Clean up internal counters
        for sid, p in prof.items():
            if not p.get("speaker_name"):
                p["speaker_name"] = sid
            # Remove internal counters
            keys_to_remove = [k for k in p.keys() if k.startswith("_n_")]
            for k in keys_to_remove:
                del p[k]

        return prof

    def _quick_take(
        self, speakers: dict[str, dict[str, Any]], duration_s: float
    ) -> str:
        if not speakers:
            return "No speakers identified."
        most = max(
            speakers.items(), key=lambda kv: float(kv[1].get("total_duration", 0.0))
        )[1]
        tone = "neutral"
        v = float(most.get("avg_valence", 0.0))
        if v > 0.2:
            tone = "positive"
        elif v < -0.2:
            tone = "negative"
        return f"{len(speakers)} speakers over {int(duration_s // 60)} min; most-active tone {tone}."

    def _moments_to_check(self, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not segments:
            return []
        arr = [(i, float(s.get("arousal", 0.0) or 0.0)) for i, s in enumerate(segments)]
        arr.sort(key=lambda kv: kv[1], reverse=True)
        picks = arr[:10]
        out = []
        for i, _ in picks:
            s = segments[i]
            out.append(
                {
                    "timestamp": float(s.get("start", 0.0) or 0.0),
                    "speaker": str(s.get("speaker_id", "Unknown")),
                    "description": (s.get("text") or "")[:180],
                    "type": "peak",
                }
            )
        out.sort(key=lambda m: m["timestamp"])
        return out

    def _action_items(self, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for s in segments:
            text = (s.get("text") or "").lower()
            intent = str(s.get("intent_top") or s.get("intent") or "")
            if (
                intent in {"command", "instruction", "request", "suggestion"}
                or "let's " in text
                or "we will" in text
            ):
                out.append(
                    {
                        "type": "action",
                        "text": s.get("text") or "",
                        "speaker": str(s.get("speaker_id", "Unknown")),
                        "timestamp": float(s.get("start", 0.0) or 0.0),
                        "confidence": 0.8,
                        "intent": intent or "unknown",
                    }
                )
        return out


# CLI entry point


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CPU-first orchestration pipeline for diarization, ASR, and affect scoring."
    )
    parser.add_argument("--input", help="Path to input audio file")
    parser.add_argument("--outdir", help="Directory to write outputs")
    parser.add_argument(
        "--registry_path",
        default=str(Path("registry") / "speaker_registry.json"),
        help="Persistent speaker registry path",
    )
    parser.add_argument(
        "--ahc_distance_threshold",
        type=float,
        default=0.02,
        help="Agglomerative clustering distance threshold",
    )
    parser.add_argument(
        "--speaker_limit",
        type=int,
        default=None,
        help="Optional cap on cluster count",
    )
    parser.add_argument(
        "--whisper_model",
        default=str(DEFAULT_WHISPER_MODEL),
        help="Whisper/Faster-Whisper model size or path",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Override language hint for ASR",
    )
    parser.add_argument(
        "--language_mode",
        default="auto",
        choices=["auto", "manual", "fallback"],
        help="Language detection mode for ASR",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=1,
        help="Beam size for decoding",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for decoding",
    )
    parser.add_argument(
        "--no_speech_threshold",
        type=float,
        default=0.50,
        help="Whisper no-speech probability threshold",
    )
    parser.add_argument(
        "--noise-reduction",
        action="store_true",
        help="Enable preprocessing noise reduction",
    )
    parser.add_argument(
        "--asr_backend",
        choices=["auto", "faster", "openai"],
        default="faster",
        help="ASR backend selector",
    )
    parser.add_argument(
        "--asr-compute-type",
        choices=["float32", "int8", "int8_float16", "float16"],
        default="float32",
        help="CTranslate2 compute_type for faster-whisper",
    )
    parser.add_argument(
        "--asr-cpu-threads",
        type=int,
        default=1,
        help="CPU threads for faster-whisper",
    )
    parser.add_argument(
        "--ignore_tx_cache",
        action="store_true",
        help="Force ASR rerun instead of using cache",
    )
    parser.add_argument(
        "--clear_cache",
        action="store_true",
        help="Clear cached diarization/transcription artefacts before running",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce console verbosity",
    )
    parser.add_argument(
        "--disable_affect",
        action="store_true",
        help="Skip affect analysis",
    )
    parser.add_argument(
        "--affect_backend",
        choices=["auto", "torch", "onnx"],
        default="onnx",
        help="Backend for affect models",
    )
    parser.add_argument(
        "--affect_text_model_dir",
        default=None,
        help="Optional GoEmotions model directory",
    )
    parser.add_argument(
        "--affect_intent_model_dir",
        default=None,
        help="Optional intent model directory",
    )
    parser.add_argument(
        "--cpu_diarizer",
        action="store_true",
        help="Enable CPU-optimised diarizer wrapper",
    )
    parser.add_argument(
        "--asr-window-sec",
        type=int,
        default=480,
        help="Maximum segment length per ASR window",
    )
    parser.add_argument(
        "--asr-segment-timeout",
        type=float,
        default=300.0,
        help="Timeout per ASR segment (seconds)",
    )
    parser.add_argument(
        "--asr-batch-timeout",
        type=float,
        default=1200.0,
        help="Timeout per ASR batch (seconds)",
    )
    parser.add_argument(
        "--disable_sed",
        action="store_true",
        help="Disable sound event detection",
    )
    parser.add_argument(
        "--chunk-enabled",
        dest="chunk_enabled",
        action="store_true",
        help="Enable auto chunking for long audio",
    )
    parser.add_argument(
        "--no-chunk",
        dest="chunk_enabled",
        action="store_false",
        help="Disable auto chunking",
    )
    parser.set_defaults(chunk_enabled=True)
    parser.add_argument(
        "--chunk-threshold-minutes",
        type=float,
        default=30.0,
        help="Chunk only when duration exceeds this threshold",
    )
    parser.add_argument(
        "--chunk-size-minutes",
        type=float,
        default=20.0,
        help="Chunk size in minutes",
    )
    parser.add_argument(
        "--chunk-overlap-seconds",
        type=float,
        default=30.0,
        help="Chunk overlap in seconds",
    )
    parser.add_argument(
        "--verify_deps",
        action="store_true",
        help="Only verify dependency availability",
    )
    parser.add_argument(
        "--strict_dependency_versions",
        action="store_true",
        help="Require minimum dependency versions during verification",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.30,
        help="Silero VAD probability threshold",
    )
    parser.add_argument(
        "--vad-min-speech-sec",
        type=float,
        default=0.8,
        help="Minimum detected speech duration",
    )
    parser.add_argument(
        "--vad-min-silence-sec",
        type=float,
        default=0.8,
        help="Minimum detected silence duration",
    )
    parser.add_argument(
        "--vad-speech-pad-sec",
        type=float,
        default=0.2,
        help="Padding around detected speech",
    )
    parser.add_argument(
        "--no-energy-fallback",
        action="store_true",
        help="Disable energy-based VAD fallback",
    )
    parser.add_argument(
        "--energy-gate-db",
        type=float,
        default=-33.0,
        help="Energy gate for fallback VAD",
    )
    parser.add_argument(
        "--energy-hop-sec",
        type=float,
        default=0.01,
        help="Energy VAD hop length",
    )
    parser.add_argument(
        "--vad-backend",
        choices=["auto", "torch", "onnx"],
        default="auto",
        help="Preferred Silero VAD backend",
    )
    return parser


def _args_to_config(
    args: argparse.Namespace, *, ignore_tx_cache: bool
) -> dict[str, Any]:
    return {
        "registry_path": args.registry_path,
        "ahc_distance_threshold": args.ahc_distance_threshold,
        "speaker_limit": args.speaker_limit,
        "whisper_model": args.whisper_model,
        "asr_backend": args.asr_backend,
        "compute_type": args.asr_compute_type,
        "cpu_threads": int(args.asr_cpu_threads),
        "language": args.language,
        "language_mode": args.language_mode,
        "ignore_tx_cache": ignore_tx_cache,
        "quiet": bool(args.quiet),
        "disable_affect": bool(args.disable_affect),
        "affect_backend": args.affect_backend,
        "affect_text_model_dir": args.affect_text_model_dir,
        "affect_intent_model_dir": args.affect_intent_model_dir,
        "beam_size": args.beam_size,
        "temperature": args.temperature,
        "no_speech_threshold": args.no_speech_threshold,
        "noise_reduction": bool(args.noise_reduction),
        "enable_sed": not bool(args.disable_sed),
        "auto_chunk_enabled": bool(args.chunk_enabled),
        "chunk_threshold_minutes": float(args.chunk_threshold_minutes),
        "chunk_size_minutes": float(args.chunk_size_minutes),
        "chunk_overlap_seconds": float(args.chunk_overlap_seconds),
        "vad_threshold": args.vad_threshold,
        "vad_min_speech_sec": args.vad_min_speech_sec,
        "vad_min_silence_sec": args.vad_min_silence_sec,
        "vad_speech_pad_sec": args.vad_speech_pad_sec,
        "vad_backend": args.vad_backend,
        "disable_energy_vad_fallback": bool(args.no_energy_fallback),
        "energy_gate_db": args.energy_gate_db,
        "energy_hop_sec": args.energy_hop_sec,
        "max_asr_window_sec": int(args.asr_window_sec),
        "segment_timeout_sec": float(args.asr_segment_timeout),
        "batch_timeout_sec": float(args.asr_batch_timeout),
        "cpu_diarizer": bool(args.cpu_diarizer),
    }


def _handle_cache_clear(
    requested: bool, *, cache_root: Path, ignore_tx_cache: bool
) -> bool:
    if not requested:
        return ignore_tx_cache
    import shutil

    try:
        if cache_root.exists():
            shutil.rmtree(cache_root, ignore_errors=True)
        cache_root.mkdir(parents=True, exist_ok=True)
        print("Cache cleared successfully.")
        return ignore_tx_cache
    except PermissionError:
        print(
            "Warning: Could not fully clear cache due to permissions. Ignoring cached results."
        )
        return True
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Warning: Cache clear failed: {exc}. Ignoring cached results.")
        return True


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.verify_deps:
        ok, problems = _verify_core_dependencies(
            require_versions=bool(args.strict_dependency_versions)
        )
        if ok:
            suffix = (
                " with required versions" if args.strict_dependency_versions else ""
            )
            print(f"All core dependencies are importable{suffix}.")
            return 0
        print("Dependency verification failed\n  - " + "\n  - ".join(problems))
        return 1

    if not args.input or not args.outdir:
        parser.error("--input and --outdir are required unless --verify_deps is used")

    ignore_tx_cache = _handle_cache_clear(
        args.clear_cache,
        cache_root=Path(".cache"),
        ignore_tx_cache=bool(args.ignore_tx_cache),
    )

    config = _args_to_config(args, ignore_tx_cache=ignore_tx_cache)
    pipeline = AudioAnalysisPipelineV2(config)
    manifest = pipeline.process_audio_file(args.input, args.outdir)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
