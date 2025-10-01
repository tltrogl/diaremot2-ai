"""Pipeline summary-related stage helpers for overlap metrics."""

from __future__ import annotations

from typing import Any, Dict


def _ensure_interrupt_bucket(store: Dict[str, Dict[str, Any]], speaker_id: str) -> Dict[str, Any]:
    bucket = store.get(speaker_id)
    if bucket is None:
        bucket = {"made": 0, "received": 0, "overlap_sec": 0.0}
        store[speaker_id] = bucket
    else:
        bucket.setdefault("made", 0)
        bucket.setdefault("received", 0)
        bucket.setdefault("overlap_sec", 0.0)
    return bucket


def run_overlap(
    state: Any,
    paralinguistics_module: Any,
    *,
    min_overlap_sec: float = 0.05,
    interruption_gap_sec: float = 0.15,
) -> None:
    """Compute overlap metrics using the provided paralinguistics module."""

    turns = getattr(state, "turns", None)
    if turns is None:
        turns = getattr(state, "segments", None)
    if turns is None:
        turns = []

    compute = getattr(paralinguistics_module, "compute_overlap_and_interruptions", None)
    if not callable(compute):
        setattr(state, "overlap_stats", {"overlap_total_sec": 0.0, "overlap_ratio": 0.0})
        setattr(state, "per_speaker_interrupts", {})
        return

    result = compute(
        turns,
        min_overlap_sec=min_overlap_sec,
        interruption_gap_sec=interruption_gap_sec,
    ) or {}
    if not isinstance(result, dict):
        result = {}

    overlap_total = float(result.get("overlap_total_sec", 0.0) or 0.0)
    overlap_ratio = float(result.get("overlap_ratio", 0.0) or 0.0)
    setattr(
        state,
        "overlap_stats",
        {"overlap_total_sec": overlap_total, "overlap_ratio": overlap_ratio},
    )

    per_speaker_raw = result.get("by_speaker") or {}
    per_speaker: Dict[str, Dict[str, Any]] = {}

    for speaker, metrics in per_speaker_raw.items():
        sid = str(speaker)
        data = _ensure_interrupt_bucket(per_speaker, sid)
        if isinstance(metrics, dict):
            if "interruptions" in metrics:
                try:
                    data["made"] = int(metrics.get("interruptions", 0) or 0)
                except (TypeError, ValueError):
                    data["made"] = 0
            if "overlap_sec" in metrics:
                try:
                    data["overlap_sec"] = float(metrics.get("overlap_sec", 0.0) or 0.0)
                except (TypeError, ValueError):
                    data["overlap_sec"] = 0.0

    for event in result.get("interruptions") or []:
        if not isinstance(event, dict):
            continue
        interrupter = event.get("interrupter")
        interrupted = event.get("interrupted")
        if interrupter is not None:
            _ensure_interrupt_bucket(per_speaker, str(interrupter))
        if interrupted is not None:
            bucket = _ensure_interrupt_bucket(per_speaker, str(interrupted))
            try:
                bucket["received"] = int(bucket.get("received", 0)) + 1
            except (TypeError, ValueError):
                bucket["received"] = 1

    setattr(state, "per_speaker_interrupts", per_speaker)
