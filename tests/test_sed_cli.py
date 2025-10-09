from __future__ import annotations

import numpy as np
import soundfile as sf

from diaremot.sed import sed_panns_onnx


class _DummyInput:
    def __init__(self) -> None:
        self.name = "input"
        self.shape = [None, 64, None]


class _DummySession:
    def __init__(self, *args, **kwargs) -> None:
        self._input = _DummyInput()

    def get_inputs(self):  # type: ignore[override]
        return [self._input]

    def run(self, _outputs, feed_dict):  # type: ignore[override]
        batch = next(iter(feed_dict.values()))
        batch_size = batch.shape[0]
        return [np.zeros((batch_size, 527), dtype=np.float32)]


def test_cli_generates_csv(monkeypatch, tmp_path) -> None:
    wav_path = tmp_path / "silence.wav"
    sf.write(wav_path, np.zeros(16000, dtype=np.float32), 16000)

    dummy_model = tmp_path / "cnn14.onnx"
    dummy_model.write_text("dummy model")

    monkeypatch.setattr(sed_panns_onnx, "_resolve_model_path", lambda _path: dummy_model)
    monkeypatch.setattr(sed_panns_onnx.ort, "InferenceSession", _DummySession)
    monkeypatch.setattr(
        sed_panns_onnx,
        "_compute_logmel",
        lambda *args, **kwargs: np.zeros((64, 10), dtype=np.float32),
    )

    output_csv = tmp_path / "events.csv"
    assert sed_panns_onnx.main([
        str(wav_path),
        str(output_csv),
        "--median",
        "3",
    ]) == 0

    content = output_csv.read_text().strip().splitlines()
    assert content[0] == "file_id,start,end,label,score"
    assert len(content) == 1
