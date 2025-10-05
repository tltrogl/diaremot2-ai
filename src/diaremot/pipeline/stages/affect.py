"""Affect analysis and assembly stage."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

from ..logging_utils import StageGuard
from ..outputs import ensure_segment_keys
from .base import PipelineState

__all__ = ["run"]


def run(pipeline: AudioAnalysisPipelineV2, state: PipelineState, guard: StageGuard) -> None:
    segments_final: list[dict[str, Any]] = []

    if pipeline.stats.config_snapshot.get("transcribe_failed"):
        state.segments_final = segments_final
        guard.done(segments=0)
        return

    for idx, seg in enumerate(state.norm_tx):
        start = float(seg.get("start") or 0.0)
        end = float(seg.get("end") or start)
        i0 = int(start * state.sr)
        i1 = int(end * state.sr)
        clip = state.y[max(0, i0) : max(0, i1)] if len(state.y) > 0 else np.array([])
        text = seg.get("text") or ""

        aff = pipeline._affect_unified(clip, state.sr, text)
        pm = state.para_metrics.get(idx, {})

        duration_s = _coerce_positive_float(pm.get("duration_s"))
        if duration_s is None:
            duration_s = max(0.0, end - start)

        words = _coerce_int(pm.get("words"))
        if words is None:
            words = len(text.split())

        pause_ratio = _coerce_positive_float(pm.get("pause_ratio"))
        if pause_ratio is None:
            pause_time = _coerce_positive_float(pm.get("pause_time_s")) or 0.0
            pause_ratio = (pause_time / duration_s) if duration_s > 0 else 0.0
        pause_ratio = max(0.0, min(1.0, pause_ratio))

        vad = aff.get("vad", {})
        speech_emotion = aff.get("speech_emotion", {})
        text_emotions = aff.get("text_emotions", {})
        intent = aff.get("intent", {})

        row = {
            "file_id": pipeline.stats.file_id,
            "start": start,
            "end": end,
            "speaker_id": seg.get("speaker_id"),
            "speaker_name": seg.get("speaker_name"),
            "text": text,
            "valence": float(vad.get("valence", 0.0)) if vad.get("valence") is not None else None,
            "arousal": float(vad.get("arousal", 0.0)) if vad.get("arousal") is not None else None,
            "dominance": (
                float(vad.get("dominance", 0.0)) if vad.get("dominance") is not None else None
            ),
            "emotion_top": speech_emotion.get("top", "neutral"),
            "emotion_scores_json": json.dumps(
                speech_emotion.get("scores_8class", {"neutral": 1.0}), ensure_ascii=False
            ),
            "text_emotions_top5_json": json.dumps(
                text_emotions.get("top5", [{"label": "neutral", "score": 1.0}]),
                ensure_ascii=False,
            ),
            "text_emotions_full_json": json.dumps(
                text_emotions.get("full_28class", {"neutral": 1.0}), ensure_ascii=False
            ),
            "intent_top": intent.get("top", "status_update"),
            "intent_top3_json": json.dumps(intent.get("top3", []), ensure_ascii=False),
            "low_confidence_ser": bool(speech_emotion.get("low_confidence_ser", False)),
            "vad_unstable": bool(state.vad_unstable),
            "affect_hint": aff.get("affect_hint", "neutral-status"),
            "asr_logprob_avg": seg.get("asr_logprob_avg"),
            "snr_db": seg.get("snr_db"),
            "wpm": pm.get("wpm", 0.0),
            "duration_s": duration_s,
            "words": words,
            "pause_ratio": pause_ratio,
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

    state.segments_final = segments_final
    guard.done(segments=len(segments_final))


def _coerce_positive_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result) or result < 0:
        return None
    return result


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None
