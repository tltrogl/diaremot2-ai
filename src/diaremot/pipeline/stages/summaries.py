"""Helpers for summary-oriented pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional


@dataclass(slots=True)
class SummariesState:
    """Mutable container for summary stage intermediates."""

    segments: List[Dict[str, Any]] = field(default_factory=list)
    turns: List[Dict[str, Any]] = field(default_factory=list)
    overlap_stats: Dict[str, Any] = field(default_factory=dict)
    per_speaker_interrupts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    speakers_summary: List[Dict[str, Any]] = field(default_factory=list)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def run_overlap(state: SummariesState, paralinguistics_module: Any) -> SummariesState:
    """Populate overlap statistics using the paralinguistics helper."""

    compute: Optional[Callable[[Iterable[Dict[str, Any]]], Dict[str, Any]]] = None
    if paralinguistics_module is not None:
        compute = getattr(paralinguistics_module, "compute_overlap_and_interruptions", None)

    if not callable(compute):
        state.overlap_stats = {"overlap_total_sec": 0.0, "overlap_ratio": 0.0}
        state.per_speaker_interrupts = {}
        return state

    try:
        payload = compute(state.turns or state.segments or []) or {}
    except Exception:
        state.overlap_stats = {"overlap_total_sec": 0.0, "overlap_ratio": 0.0}
        state.per_speaker_interrupts = {}
        return state

    overlap_total = _safe_float(payload.get("overlap_total_sec"))
    overlap_ratio = _safe_float(payload.get("overlap_ratio"))
    state.overlap_stats = {
        "overlap_total_sec": overlap_total,
        "overlap_ratio": overlap_ratio,
    }

    per_speaker_raw = payload.get("by_speaker") or {}
    per_speaker: Dict[str, Dict[str, Any]] = {}
    for speaker_id, stats in per_speaker_raw.items():
        if not isinstance(stats, dict):
            continue
        sid = str(speaker_id)
        per_speaker[sid] = {
            "made": _safe_int(stats.get("interruptions")),
            "received": 0,
            "overlap_sec": _safe_float(stats.get("overlap_sec")),
        }

    for entry in payload.get("interruptions") or []:
        if not isinstance(entry, dict):
            continue
        interrupted = entry.get("interrupted")
        if interrupted is None:
            continue
        sid = str(interrupted)
        if sid not in per_speaker:
            stats = per_speaker_raw.get(sid) if isinstance(per_speaker_raw, dict) else None
            overlap_sec = _safe_float(stats.get("overlap_sec")) if isinstance(stats, dict) else 0.0
            per_speaker[sid] = {"made": 0, "received": 0, "overlap_sec": overlap_sec}
        per_speaker[sid]["received"] = per_speaker[sid].get("received", 0) + 1

    state.per_speaker_interrupts = per_speaker
    return state


__all__ = ["SummariesState", "run_overlap"]
