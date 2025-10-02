"""Stage registry for the DiaRemot pipeline."""

from __future__ import annotations

from . import (
    affect,
    asr,
    auto_tune,
    dependency_check,
    diarize,
    paralinguistics,
    preprocess,
    summaries,
)
from .base import PipelineState, StageDefinition

PIPELINE_STAGES: list[StageDefinition] = [
    StageDefinition("dependency_check", dependency_check.run),
    StageDefinition("preprocess", preprocess.run_preprocess),
    StageDefinition("auto_tune", auto_tune.run),
    StageDefinition("background_sed", preprocess.run_background_sed),
    StageDefinition("diarize", diarize.run),
    StageDefinition("transcribe", asr.run),
    StageDefinition("paralinguistics", paralinguistics.run),
    StageDefinition("affect_and_assemble", affect.run),
    StageDefinition("overlap_interruptions", summaries.run_overlap),
    StageDefinition("conversation_analysis", summaries.run_conversation),
    StageDefinition("speaker_rollups", summaries.run_speaker_rollups),
    StageDefinition("outputs", summaries.run_outputs),
]

__all__ = ["PIPELINE_STAGES", "PipelineState", "StageDefinition"]
