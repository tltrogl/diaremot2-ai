from __future__ import annotations

import json
import logging

from diaremot.pipeline.logging_utils import (
    CoreLogger,
    JSONLWriter,
    RunStats,
    StageGuard,
)


def test_jsonl_writer_appends_records(tmp_path) -> None:
    target = tmp_path / "events.jsonl"
    writer = JSONLWriter(target)
    writer.emit({"stage": "test", "value": 1})
    writer.emit({"stage": "test", "value": 2})

    lines = target.read_text(encoding="utf-8").strip().splitlines()
    payloads = [json.loads(line) for line in lines]
    assert payloads == [
        {"stage": "test", "value": 1},
        {"stage": "test", "value": 2},
    ]


def test_runstats_mark_aggregates_counts() -> None:
    stats = RunStats(run_id="run", file_id="file.wav")
    stats.mark("stage", 100.0, {"segments": 2})
    stats.mark("stage", 50.0, {"segments": 3})

    assert stats.stage_timings_ms["stage"] == 150.0
    assert stats.stage_counts["stage"]["segments"] == 5


def test_stageguard_swallows_optional_exceptions(tmp_path) -> None:
    logger = CoreLogger("run", tmp_path / "log.jsonl", console_level=logging.CRITICAL)
    stats = RunStats(run_id="run", file_id="file.wav")

    with StageGuard(logger, stats, "background_sed"):
        raise ImportError("sed models missing")

    assert any(f["stage"] == "background_sed" for f in stats.failures)
