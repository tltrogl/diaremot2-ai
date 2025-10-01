from __future__ import annotations

from types import SimpleNamespace

from diaremot.pipeline.stages import SummariesState, run_overlap


def test_run_overlap_maps_per_speaker_interruptions() -> None:
    state = SummariesState(
        turns=[
            {"start": 0.0, "end": 1.0, "speaker_id": "A"},
            {"start": 0.5, "end": 1.5, "speaker_id": "B"},
        ]
    )

    payload = {
        "overlap_total_sec": 1.25,
        "overlap_ratio": 0.5,
        "by_speaker": {
            "A": {"overlap_sec": 0.75, "interruptions": 2},
            "B": {"overlap_sec": 0.5, "interruptions": 1},
        },
        "interruptions": [
            {"interrupter": "A", "interrupted": "B"},
            {"interrupter": "B", "interrupted": "C"},
        ],
    }

    called = {}

    def fake_compute(turns):
        called["turns"] = turns
        return payload

    module = SimpleNamespace(compute_overlap_and_interruptions=fake_compute)

    run_overlap(state, module)

    assert called["turns"] == state.turns
    assert state.overlap_stats == {
        "overlap_total_sec": 1.25,
        "overlap_ratio": 0.5,
    }
    assert state.per_speaker_interrupts == {
        "A": {"made": 2, "received": 0, "overlap_sec": 0.75},
        "B": {"made": 1, "received": 1, "overlap_sec": 0.5},
        "C": {"made": 0, "received": 1, "overlap_sec": 0.0},
    }
