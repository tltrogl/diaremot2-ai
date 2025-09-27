import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


if "librosa" not in sys.modules:
    librosa_stub = SimpleNamespace(
        util=SimpleNamespace(frame=lambda *args, **kwargs: np.zeros((0,))),
        feature=SimpleNamespace(
            melspectrogram=lambda *args, **kwargs: np.zeros((1, 1))
        ),
        power_to_db=lambda x, ref=1.0: x,
    )
    sys.modules["librosa"] = librosa_stub

if "scipy" not in sys.modules:
    scipy_signal = SimpleNamespace(resample_poly=lambda x, up, down: x)
    sys.modules["scipy"] = SimpleNamespace(signal=scipy_signal)
    sys.modules["scipy.signal"] = scipy_signal

from diaremot.pipeline.speaker_diarization import _SileroWrapper


class _DummyTensor:
    def __init__(self, name: str, shape=None):
        self.name = name
        self.shape = shape or ()


class _DummySession:
    def __init__(self):
        self.calls = []

    def get_inputs(self):
        return [
            _DummyTensor("input", (1, 576)),
            _DummyTensor("state", (2, 1, 128)),
            _DummyTensor("sr", ()),
        ]

    def get_outputs(self):
        return [_DummyTensor("output", (1, 1)), _DummyTensor("state_out", (2, 1, 128))]

    def run(self, output_names, feeds):  # noqa: D401 - mimic onnxruntime signature
        self.calls.append({k: np.array(v, copy=True) for k, v in feeds.items()})
        step = len(self.calls)
        if step <= 2:
            logits = np.array([[0.0, 4.0]], dtype=np.float32)
        else:
            logits = np.array([[4.0, 0.0]], dtype=np.float32)
        state = np.zeros((2, 1, 128), dtype=np.float32)
        return [logits, state]


def test_silero_wrapper_detect_onnx_shapes():
    with mock.patch.object(_SileroWrapper, "_load", lambda self: None):
        wrapper = _SileroWrapper(threshold=0.5, speech_pad_sec=0.0, backend="onnx")

    wrapper.session = _DummySession()
    wrapper._onnx_input_name = "input"
    wrapper._onnx_state_name = "state"
    wrapper._onnx_sr_name = "sr"
    wrapper._onnx_state_output_index = 1
    wrapper._onnx_state_shape = (2, 1, 128)

    wav = np.ones(1600, dtype=np.float32)
    segments = wrapper.detect(wav, 16000, min_speech_sec=0.01, min_silence_sec=0.01)

    assert segments, "Expected ONNX VAD to yield at least one speech segment"
    assert len(wrapper.session.calls) == 4, "Expected 512-sample chunking at 16 kHz"

    first_call = wrapper.session.calls[0]
    assert first_call["input"].shape == (1, 576)
    assert first_call["state"].shape == (2, 1, 128)
    assert first_call["sr"].shape == ()
    assert first_call["sr"].dtype == np.int64

    for start, end in segments:
        assert isinstance(start, float)
        assert isinstance(end, float)
        assert start <= end
