"""Core orchestration logic for the DiaRemot audio analysis pipeline."""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from ..affect.emotion_analyzer import EmotionIntentAnalyzer
from ..affect.intent_defaults import INTENT_LABELS_DEFAULT
from ..affect.sed_panns import PANNSEventTagger, SEDConfig  # type: ignore
from ..summaries.conversation_analysis import ConversationMetrics
from ..summaries.html_summary_generator import HTMLSummaryGenerator
from ..summaries.pdf_summary_generator import PDFSummaryGenerator
from .audio_preprocessing import AudioPreprocessor, PreprocessConfig
from .auto_tuner import AutoTuner
from .config import (
    DEFAULT_PIPELINE_CONFIG,
    build_pipeline_config,
)
from .config import (
    diagnostics as config_diagnostics,
)
from .config import (
    verify_dependencies as config_verify_dependencies,
)
from .logging_utils import CoreLogger, RunStats, StageGuard, _fmt_hms_ms
from .outputs import (
    default_affect,
    ensure_segment_keys,
    write_qc_report,
    write_segments_csv,
    write_segments_jsonl,
    write_speakers_summary,
    write_timeline_csv,
)
from .pipeline_checkpoint_system import PipelineCheckpointManager, ProcessingStage
from .runtime_env import (
    DEFAULT_WHISPER_MODEL,
    WINDOWS_MODELS_ROOT,
    configure_local_cache_env,
)
from .speaker_diarization import DiarizationConfig, SpeakerDiarizer
from .stages import PIPELINE_STAGES, PipelineState

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


configure_local_cache_env()

try:
    from ..affect import paralinguistics as para
except Exception:
    para = None
CACHE_VERSION = "v3"  # Incremented to handle new checkpoint logic


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
        self.cache_version = CACHE_VERSION
        self.paralinguistics_module = para

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

        # Ensure attributes exist even if initialization fails part-way
        self.pre = None
        self.diar = None
        self.tx = None
        self.affect = None
        self.sed_tagger = None
        self.html = None
        self.pdf = None

        affect_kwargs: dict[str, Any] = {
            "text_emotion_model": cfg.get(
                "text_emotion_model", "SamLowe/roberta-base-go_emotions"
            ),
            "intent_labels": cfg.get("intent_labels", INTENT_LABELS_DEFAULT),
            "affect_backend": cfg.get("affect_backend", "auto"),
            "affect_text_model_dir": cfg.get("affect_text_model_dir"),
            "affect_intent_model_dir": cfg.get("affect_intent_model_dir"),
            "analyzer_threads": cfg.get("affect_threads"),
        }

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

            # Background SED / noise tagger (required in the default pipeline)
            try:
                sed_enabled = bool(cfg.get("enable_sed", True))
                if PANNSEventTagger is not None and sed_enabled:
                    self.sed_tagger = PANNSEventTagger(
                        SEDConfig() if SEDConfig else None
                    )
                elif not sed_enabled:
                    self.corelog.warn(
                        "[sed] disabled via configuration; pipeline outputs will lack background tags"
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
                if getattr(self, "pre", None) is None:
                    self.pre = AudioPreprocessor(PreprocessConfig())
            except Exception:
                self.pre = None
            try:
                if getattr(self, "diar", None) is None:
                    self.diar = SpeakerDiarizer(DiarizationConfig(target_sr=16000))
            except Exception:
                self.diar = None
            try:
                if getattr(self, "tx", None) is None:
                    from .transcription_module import AudioTranscriber

                    self.tx = AudioTranscriber()
            except Exception:
                self.tx = None
            try:
                if getattr(self, "affect", None) is None and not cfg.get("disable_affect"):
                    self.affect = EmotionIntentAnalyzer(**affect_kwargs)
            except Exception:
                self.affect = None
            try:
                if getattr(self, "pdf", None) is None:
                    self.pdf = PDFSummaryGenerator()
            except Exception:
                pass
            try:
                if not hasattr(self, "auto_tuner"):
                    self.auto_tuner = AutoTuner()
            except Exception:
                self.auto_tuner = None

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
                        "pause_ratio": float(d.get("pause_ratio", 0.0) or 0.0),
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
                "pause_ratio": 0.0,
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
        """Main processing entry point coordinating modular stages."""

        outp = Path(out_dir)
        outp.mkdir(parents=True, exist_ok=True)
        self.stats.file_id = Path(input_audio_path).name

        state = PipelineState(input_audio_path=input_audio_path, out_dir=outp)

        try:
            for stage in PIPELINE_STAGES:
                with StageGuard(self.corelog, self.stats, stage.name) as guard:
                    stage.runner(self, state, guard)
        except Exception as exc:
            self.corelog.error(f"Pipeline failed with unhandled error: {exc}")
            if not state.segments_final and state.norm_tx:
                state.segments_final = [

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
                    for seg in state.norm_tx
                ]
            try:
                self._write_outputs(
                    input_audio_path,
                    outp,
                    state.segments_final,
                    state.speakers_summary,
                    state.health,
                    state.turns,
                    state.overlap_stats,
                    state.per_speaker_interrupts,
                    state.conv_metrics,
                    state.duration_s,
                )
            except Exception as write_error:
                self.corelog.error(f"Failed to write outputs: {write_error}")

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

        try:
            dep_ok = bool(self.stats.config_snapshot.get("dependency_ok", True))
            dep_summary = self.stats.config_snapshot.get("dependency_summary", {}) or {}
            unhealthy = [k for k, v in dep_summary.items() if v.get("status") != "ok"]
            if dep_ok and not unhealthy:
                self.corelog.info("[deps] All core dependencies loaded successfully.")
            else:
                self.corelog.warn("[deps] Issues detected: " + ", ".join(unhealthy))
            manifest["dependency_ok"] = dep_ok and not unhealthy
            manifest["dependency_unhealthy"] = unhealthy
        except Exception:
            pass

        try:
            if hasattr(self, "tx") and hasattr(self.tx, "get_model_info"):
                tx_info = self.tx.get_model_info()
                manifest["transcriber"] = tx_info
                if tx_info.get("fallback_triggered"):
                    self.corelog.warn(
                        "[tx] Fallback engaged: "
                        + str(tx_info.get("fallback_reason", "unknown"))
                    )
            if "background_sed" in getattr(self.stats, "config_snapshot", {}):
                manifest["background_sed"] = self.stats.config_snapshot.get("background_sed")
        except Exception:
            pass

        self.corelog.event("done", "stop", **manifest)

        try:
            stage_names = [stage.name for stage in PIPELINE_STAGES]
            failures = {f.get("stage"): f for f in getattr(self.stats, "failures", [])}
            self.corelog.info("[ALERT] Stage summary:")
            for st in stage_names:
                if st in failures:
                    failure = failures[st]
                    elapsed_ms = float(failure.get("elapsed_ms", 0.0))
                    self.corelog.warn(
                        f"  - {st}: FAIL in {_fmt_hms_ms(elapsed_ms)} — {failure.get('error')} | Fix: {failure.get('suggestion')}"
                    )
                else:
                    if st in {"paralinguistics", "affect_and_assemble"} and self.stats.config_snapshot.get("transcribe_failed"):
                        self.corelog.warn(f"  - {st}: SKIPPED (transcribe_failed)")
                    else:
                        elapsed_ms = float(self.stats.stage_timings_ms.get(st, 0.0))
                        self.corelog.info(f"  - {st}: PASS in {_fmt_hms_ms(elapsed_ms)}")
        except Exception:
            pass

        self.checkpoints.create_checkpoint(
            input_audio_path,
            ProcessingStage.COMPLETE,
            manifest,
            progress=100.0,
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
