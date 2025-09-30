"""Core orchestration logic for the DiaRemot audio analysis pipeline."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

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
from .config import (
    DEFAULT_PIPELINE_CONFIG,
    build_pipeline_config,
    dependency_health_summary,
    diagnostics as config_diagnostics,
    verify_dependencies as config_verify_dependencies,
)
from .logging_utils import CoreLogger, RunStats, StageGuard, _fmt_hms, _fmt_hms_ms
from .outputs import (
    default_affect,
    ensure_segment_keys,
    write_qc_report,
    write_segments_csv,
    write_segments_jsonl,
    write_timeline_csv,
    write_speakers_summary,
)

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


__all__ = [
    "AudioAnalysisPipelineV2",
    "build_pipeline_config",
    "run_pipeline",
    "resume",
    "diagnostics",
    "verify_dependencies",
    "clear_pipeline_cache",
]


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

    return config_verify_dependencies(strict)


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

    return config_diagnostics(require_versions=require_versions)


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
            ok, problems = config_verify_dependencies(
                strict=bool(cfg.get("strict_dependency_versions", False))
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
        write_segments_csv(
            outp / "diarized_transcript_with_emotion.csv", segments_final
        )

        # JSONL segments
        write_segments_jsonl(outp / "segments.jsonl", segments_final)

        # Timeline
        write_timeline_csv(outp / "timeline.csv", segments_final)

        # QC report
        write_qc_report(
            outp / "qc_report.json",
            self.stats,
            health,
            n_turns=len(turns),
            n_segments=len(segments_final),
            segments=segments_final,
        )

        # Speakers summary
        write_speakers_summary(outp / "speakers_summary.csv", speakers_summary)

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
                dep_summary = dependency_health_summary()
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
