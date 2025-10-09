from __future__ import annotations

import pytest

from diaremot.sed.sed_panns_onnx import _hysteresis_runs


def test_hysteresis_basic_single_event() -> None:
    probs = [0.1, 0.6, 0.55, 0.2, 0.1]
    runs = _hysteresis_runs(
        probs,
        enter=0.5,
        exit=0.35,
        min_dur=0.2,
        merge_gap=0.05,
        frame_sec=1.0,
        hop_sec=0.5,
    )
    assert runs == [(1, 3, pytest.approx(0.6, rel=1e-6))]


def test_hysteresis_merges_close_runs() -> None:
    probs = [0.55, 0.2, 0.52, 0.25, 0.1]
    runs = _hysteresis_runs(
        probs,
        enter=0.5,
        exit=0.35,
        min_dur=0.2,
        merge_gap=0.3,
        frame_sec=1.0,
        hop_sec=0.5,
    )
    # Events at indices [0] and [2] should merge because the gap is exactly one hop.
    assert runs == [(0, 3, pytest.approx(0.55, rel=1e-6))]


def test_hysteresis_filters_short_events() -> None:
    probs = [0.6, 0.2]
    runs = _hysteresis_runs(
        probs,
        enter=0.5,
        exit=0.35,
        min_dur=1.5,
        merge_gap=0.1,
        frame_sec=1.0,
        hop_sec=0.5,
    )
    assert runs == []
