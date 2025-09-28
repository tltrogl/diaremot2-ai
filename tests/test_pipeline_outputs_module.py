from __future__ import annotations

import json

from diaremot.pipeline.logging_utils import RunStats
from diaremot.pipeline.outputs import (
    SEGMENT_COLUMNS,
    default_affect,
    ensure_segment_keys,
    write_qc_report,
    write_segments_csv,
    write_segments_jsonl,
    write_timeline_csv,
)


def test_ensure_segment_keys_populates_defaults() -> None:
    seg = {"start": 0.0, "end": 1.0, "speaker_id": "S1"}
    ensure_segment_keys(seg)
    for key in SEGMENT_COLUMNS:
        assert key in seg


def test_writers_produce_files(tmp_path) -> None:
    segments = [
        {
            "file_id": "file.wav",
            "start": 0.0,
            "end": 1.0,
            "speaker_id": "S1",
            "speaker_name": "Speaker 1",
            "text": "hello",
        }
    ]
    ensure_segment_keys(segments[0])

    csv_path = tmp_path / "segments.csv"
    jsonl_path = tmp_path / "segments.jsonl"
    timeline_path = tmp_path / "timeline.csv"

    write_segments_csv(csv_path, segments)
    write_segments_jsonl(jsonl_path, segments)
    write_timeline_csv(timeline_path, segments)

    assert "speaker_id" in csv_path.read_text(encoding="utf-8")
    json_lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert json.loads(json_lines[0])["speaker_id"] == "S1"
    assert "speaker_id" in timeline_path.read_text(encoding="utf-8")


def test_write_qc_report_serialises_payload(tmp_path) -> None:
    stats = RunStats(run_id="run", file_id="file.wav")
    stats.mark("stage", 42.0)

    qc_path = tmp_path / "qc.json"
    write_qc_report(
        qc_path,
        stats,
        health=None,
        n_turns=1,
        n_segments=1,
        segments=[ensure_segment_keys({"file_id": "file.wav"})],
    )

    payload = json.loads(qc_path.read_text(encoding="utf-8"))
    assert payload["run_id"] == "run"
    assert payload["counts"] == {"turns": 1, "segments": 1}
    assert "voice_quality_summary" in payload
    assert default_affect()["speech_emotion"]["top"] == "neutral"
