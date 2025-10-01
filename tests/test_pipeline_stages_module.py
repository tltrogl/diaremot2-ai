"""Tests for the pipeline stages helpers."""

from __future__ import annotations

import types

from diaremot.pipeline.stages.summaries import run_overlap


class _StubParalinguistics:
    def __init__(self, payload):
        self._payload = payload
        self.call_args = None

    def compute_overlap_and_interruptions(self, *args, **kwargs):
        self.call_args = (args, kwargs)
        return self._payload


def test_run_overlap_maps_per_speaker_interruptions():
    payload = {
        "overlap_total_sec": 3.5,
        "overlap_ratio": 0.25,
        "by_speaker": {
            "A": {"overlap_sec": 2.0, "interruptions": 2},
            "B": {"overlap_sec": 1.5, "interruptions": 1},
        },
        "interruptions": [
            {"interrupter": "A", "interrupted": "B", "overlap_sec": 0.8},
            {"interrupter": "A", "interrupted": "B", "overlap_sec": 0.4},
            {"interrupter": "B", "interrupted": "A", "overlap_sec": 0.2},
        ],
    }

    state = types.SimpleNamespace(turns=[{"speaker_id": "A"}])
    stub = _StubParalinguistics(payload)

    run_overlap(state, stub)

    assert state.overlap_stats == {"overlap_total_sec": 3.5, "overlap_ratio": 0.25}
    assert stub.call_args is not None

    per_speaker = state.per_speaker_interrupts
    assert per_speaker["A"] == {"made": 2, "received": 1, "overlap_sec": 2.0}
    assert per_speaker["B"] == {"made": 1, "received": 2, "overlap_sec": 1.5}
