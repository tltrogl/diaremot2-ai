from pathlib import Path

import pytest

try:
    import numpy as np
    from scipy.io import wavfile
except ModuleNotFoundError:  # pragma: no cover - exercised only when deps missing
    pytest.skip("numpy and scipy.io are required for diarization tests", allow_module_level=True)

from diaremot.pipeline import speaker_diarization as sd

if getattr(np, "__stub__", False) or not hasattr(np, "array"):
    pytest.skip("numpy not available", allow_module_level=True)


def test_default_threshold_merges_single_speaker(monkeypatch):
    """Single-speaker audio should yield a single speaker label with defaults."""

    base_embedding = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    base_embedding /= np.linalg.norm(base_embedding)

    class DummyVAD:
        def __init__(self, *_args, **_kwargs):
            pass

        def detect(self, wav, sr, _min_speech, _min_silence):
            duration = len(wav) / sr if sr else 0.0
            return [(0.0, duration)] if duration else []

    class DummyECAPA:
        def __init__(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(sd, "_SileroWrapper", DummyVAD)
    monkeypatch.setattr(sd, "_ECAPAWrapper", DummyECAPA)

    def fake_extract(self, _wav, _sr, speech_regions):
        if not speech_regions:
            return []
        start, end = speech_regions[0]
        mid = (start + end) / 2
        return [
            {
                "start": start,
                "end": mid,
                "embedding": base_embedding.copy(),
                "speaker": None,
                "region_idx": 0,
            },
            {
                "start": mid,
                "end": end,
                "embedding": base_embedding.copy(),
                "speaker": None,
                "region_idx": 1,
            },
        ]

    monkeypatch.setattr(sd.SpeakerDiarizer, "_extract_embedding_windows", fake_extract)

    wav_path = Path("data/sample.wav")
    sr, wav = wavfile.read(wav_path)
    wav = wav.astype(np.float32)
    if np.max(np.abs(wav)) > 0:
        wav /= np.max(np.abs(wav))

    diarizer = sd.SpeakerDiarizer(sd.DiarizationConfig())
    segments = diarizer.diarize_audio(wav, sr)

    assert len(segments) == 1
    assert {seg["speaker"] for seg in segments} == {"Speaker_1"}
