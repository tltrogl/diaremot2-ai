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

    with StageGuard(logger, stats, "paralinguistics"):
        raise ImportError("librosa missing")


    assert any(f["stage"] == "paralinguistics" for f in stats.failures)


def test_corelogger_supports_format_args(tmp_path, caplog) -> None:
    logger = CoreLogger("run", tmp_path / "log.jsonl", console_level=logging.DEBUG)

    with caplog.at_level(logging.DEBUG, logger=logger.log.name):
        logger.info("formatted %s %s", "message", 123)
        logger.warn("warn %s", "here")
        logger.error("error %s", "there")

    rendered = [record.getMessage() for record in caplog.records]
    assert "formatted message 123" in rendered
    assert "warn here" in rendered
    assert "error there" in rendered

