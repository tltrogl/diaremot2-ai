
"""Compatibility shim that re-exports the pipeline implementation pieces."""

from __future__ import annotations

from .cli_entry import _args_to_config, _build_arg_parser, main
from .config import (
    CORE_DEPENDENCY_REQUIREMENTS,
    DEFAULT_PIPELINE_CONFIG,
    build_pipeline_config,
    dependency_health_summary,
)
from .logging_utils import CoreLogger, JSONLWriter, RunStats, StageGuard, _fmt_hms, _fmt_hms_ms
from .orchestrator import (
    AudioAnalysisPipelineV2,
    clear_pipeline_cache,
    diagnostics,
    run_pipeline,
    resume,
    verify_dependencies,
)
from .outputs import (
    SEGMENT_COLUMNS,
    default_affect,
    ensure_segment_keys,
    write_qc_report,
    write_segments_csv,
    write_segments_jsonl,
    write_speakers_summary,
    write_timeline_csv,
)

__all__ = [
    "AudioAnalysisPipelineV2",
    "CORE_DEPENDENCY_REQUIREMENTS",
    "DEFAULT_PIPELINE_CONFIG",
    "JSONLWriter",
    "CoreLogger",
    "RunStats",
    "StageGuard",
    "SEGMENT_COLUMNS",
    "_args_to_config",
    "_build_arg_parser",
    "_fmt_hms",
    "_fmt_hms_ms",
    "build_pipeline_config",
    "clear_pipeline_cache",
    "default_affect",
    "dependency_health_summary",
    "diagnostics",
    "ensure_segment_keys",
    "main",
    "resume",
    "run_pipeline",
    "verify_dependencies",
    "write_qc_report",
    "write_segments_csv",
    "write_segments_jsonl",
    "write_speakers_summary",
    "write_timeline_csv",
]
