"""Paralinguistics extraction stage."""

from __future__ import annotations

from ..logging_utils import StageGuard
from .base import PipelineState

__all__ = ["run"]


def run(pipeline: "AudioAnalysisPipelineV2", state: PipelineState, guard: StageGuard) -> None:
    metrics: dict[int, dict[str, object]] = {}
    if not pipeline.stats.config_snapshot.get("transcribe_failed"):
        tmp_metrics = pipeline._extract_paraling(state.y, state.sr, state.norm_tx)
        if isinstance(tmp_metrics, dict):
            metrics = tmp_metrics
    state.para_metrics = metrics
    guard.done(count=len(metrics))
