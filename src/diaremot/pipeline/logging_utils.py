"""Shared logging utilities for the audio pipeline."""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def format_duration(seconds: float) -> str:
    seconds = max(0, float(seconds))
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def format_duration_ms(ms: float) -> str:
    total_ms = int(round(max(0.0, float(ms))))
    seconds, ms = divmod(total_ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}" if hours else f"{minutes:02d}:{seconds:02d}.{ms:03d}"


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
        except PermissionError as exc:  # pragma: no cover - best effort logging
            print(f"Warning: Could not write to log file {self.path}: {exc}")
        except Exception as exc:  # pragma: no cover - best effort logging
            print(f"Warning: Error writing to log file {self.path}: {exc}")


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

    def mark(self, stage: str, elapsed_ms: float, counts: dict[str, int] | None = None) -> None:
        self.stage_timings_ms[stage] = self.stage_timings_ms.get(stage, 0.0) + float(elapsed_ms)
        if counts:
            slot = self.stage_counts.setdefault(stage, {})
            for key, value in counts.items():
                slot[key] = slot.get(key, 0) + int(value)


class PipelineLogger:
    """Structured logger that enriches messages with run context."""

    def __init__(self, run_id: str, jsonl_path: Path, console_level: int = logging.INFO):
        self.run_id = run_id
        self.jsonl = JSONLWriter(jsonl_path)
        base_logger = logging.getLogger(f"pipeline.{run_id}")
        base_logger.setLevel(console_level)
        if not base_logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(console_level)
            formatter = logging.Formatter(
                "[%(asctime)s] [%(levelname)s] [run:%(run_id)s] %(message)s", datefmt="%H:%M"
            )
            handler.setFormatter(formatter)
            base_logger.addHandler(handler)
        self._adapter = logging.LoggerAdapter(base_logger, extra={"run_id": run_id})

    def event(self, stage: str, event: str, **fields: Any) -> None:
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "run_id": self.run_id,
            "stage": stage,
            "event": event,
        }
        record.update(fields)
        self.jsonl.emit(record)

    def bind(self, **extra: Any) -> logging.LoggerAdapter:
        context = dict(self._adapter.extra)
        context.update(extra)
        return logging.LoggerAdapter(self._adapter.logger, context)

    def info(self, msg: str, **kwargs: Any) -> None:
        self._adapter.info(msg, **kwargs)

    def warn(self, msg: str, **kwargs: Any) -> None:
        self._adapter.warning(msg, **kwargs)

    def error(self, msg: str, **kwargs: Any) -> None:
        self._adapter.error(msg, **kwargs)


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

    def __init__(self, corelog: PipelineLogger, stats: RunStats, stage: str):
        self.corelog = corelog
        self.stats = stats
        self.stage = stage
        self.start: float | None = None
        context = {"stage": stage}
        file_id = getattr(stats, "file_id", None)
        if file_id:
            context["file_id"] = file_id
        self._logger = corelog.bind(**context)

    def __enter__(self) -> "StageGuard":
        self.start = time.time()
        self.corelog.event(self.stage, "start")
        self._logger.info("[%s] start", self.stage)
        return self

    def done(self, **counts: int) -> None:
        if counts:
            self.stats.mark(self.stage, 0.0, counts)

    def _is_known_nonfatal(self, exc: BaseException) -> bool:
        if isinstance(exc, TimeoutError | subprocess.TimeoutExpired) and (self.stage in self._TIMEOUT_STAGES):
            return True
        allowed = self._OPTIONAL_STAGE_EXCEPTION_MAP.get(self.stage, tuple())
        return any(isinstance(exc, exc_cls) for exc_cls in allowed)

    def __exit__(self, exc_type, exc, tb) -> bool:
        elapsed_ms = (time.time() - self.start) * 1000.0 if self.start else 0.0
        if exc:
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
            duration_text = format_duration_ms(elapsed_ms)
            log_fn = self._logger.warning if known_nonfatal else self._logger.error
            log_fn(
                "[%s] %s%s: %s (%s)",
                self.stage,
                "handled " if known_nonfatal else "",
                type(exc).__name__,
                exc,
                duration_text,
            )
            self.stats.mark(self.stage, elapsed_ms)
            try:
                message = f"{self.stage}: {type(exc).__name__}: {exc}"
                self.stats.warnings.append(message)
                self.stats.errors.append(message)

                def _suggest_fix(stage: str, err: BaseException) -> str:
                    text = str(err).lower()
                    if stage == "preprocess":
                        if "libsndfile" in text or "soundfile" in text:
                            return (
                                "Install libsndfile: apt-get install libsndfile1 (Linux) or brew install libsndfile (macOS)."
                            )
                        if "ffmpeg" in text or "audioread" in text:
                            return "Install ffmpeg and ensure it is on PATH."
                        if "file not found" in text or "no such file" in text:
                            return "Check input path and permissions."
                        return "Verify audio codec support (try converting to WAV 16kHz mono)."
                    if stage == "transcribe":
                        if isinstance(err, TimeoutError | subprocess.TimeoutExpired):
                            return "Increase --asr-segment-timeout or choose a smaller Whisper model."
                        if "faster_whisper" in text or "ctranslate2" in text:
                            return "Install faster-whisper and ctranslate2; confirm CPU wheels are compatible."
                        if "whisper" in text and "tiny" in text:
                            return "OpenAI whisper fallback failed; try reinstalling whisper or using a local model."
                        if "model" in text and ("not found" in text or "download" in text):
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
                if self.stage == "preprocess":
                    self.stats.config_snapshot["preprocess_failed"] = True
                if self.stage == "transcribe":
                    self.stats.config_snapshot["transcribe_failed"] = True
            except Exception:  # pragma: no cover - defensive logging
                pass
            swallow = known_nonfatal and self.stage not in self._CRITICAL_STAGES
            return swallow
        self.corelog.event(self.stage, "stop", elapsed_ms=elapsed_ms)
        duration_text = format_duration_ms(elapsed_ms)
        self._logger.info("[%s] ok in %s", self.stage, duration_text)
        self.stats.mark(self.stage, elapsed_ms)
        return False


__all__ = [
    "JSONLWriter",
    "PipelineLogger",
    "RunStats",
    "StageGuard",
    "format_duration",
    "format_duration_ms",
]

