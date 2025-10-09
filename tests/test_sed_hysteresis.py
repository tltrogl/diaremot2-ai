from diaremot.sed.sed_panns_onnx import _hysteresis_runs


def test_hysteresis_basic_run():
    seq = [0.1, 0.6, 0.7, 0.4, 0.2]
    runs = _hysteresis_runs(seq, enter=0.5, exit=0.35, min_dur=0.3, merge_gap=0.2, hop_sec=0.5)
    assert runs == [(1, 4, 0.7)]


def test_hysteresis_respects_min_duration():
    seq = [0.6, 0.2]
    runs = _hysteresis_runs(seq, enter=0.5, exit=0.35, min_dur=1.1, merge_gap=0.2, hop_sec=0.5)
    assert runs == []


def test_hysteresis_merges_close_runs():
    seq = [0.6, 0.2, 0.6, 0.2]
    runs = _hysteresis_runs(seq, enter=0.5, exit=0.35, min_dur=0.3, merge_gap=0.2, hop_sec=0.5)
    assert runs == [(0, 3, 0.6)]


def test_hysteresis_gap_prevents_merge():
    seq = [0.6, 0.2, 0.2, 0.6, 0.2]
    runs = _hysteresis_runs(seq, enter=0.5, exit=0.35, min_dur=0.3, merge_gap=0.1, hop_sec=0.5)
    assert runs == [(0, 1, 0.6), (3, 4, 0.6)]
