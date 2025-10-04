import math
import types

from diaremot.pipeline.audio_preprocessing import AudioHealth
from diaremot.pipeline.auto_tuner import AutoTuner


def _health(**overrides):
    base = dict(
        snr_db=20.0,
        clipping_detected=False,
        silence_ratio=0.4,
        rms_db=-26.0,
        est_lufs=-24.0,
        dynamic_range_db=18.0,
        floor_clipping_ratio=0.0,
        is_chunked=False,
        chunk_info=None,
    )
    base.update(overrides)
    return AudioHealth(**base)


def test_auto_tuner_low_snr_reduces_vad_and_updates_asr():
    tuner = AutoTuner()
    health = _health(snr_db=5.5)
    audio = [0.005 + 0.03 * math.sin(2 * math.pi * 150 * (i / 16000.0)) for i in range(16000)]
    diar_conf = types.SimpleNamespace(
        vad_threshold=0.25,
        vad_min_speech_sec=0.6,
        vad_min_silence_sec=0.6,
        speech_pad_sec=0.15,
    )
    asr_config = {"beam_size": 1, "no_speech_threshold": 0.5}

    result = tuner.recommend(
        health=health,
        audio=audio,
        sr=16000,
        diar_config=diar_conf,
        asr_config=asr_config,
    )

    assert result.diarization.get("vad_threshold", 1.0) < 0.25
    assert result.diarization.get("vad_min_speech_sec", 1.0) <= 0.45
    assert result.asr.get("beam_size") == 2
    assert result.asr.get("no_speech_threshold", 1.0) < 0.5
    assert "snr<6_lower_vad" in result.notes
    assert result.metrics["peak_dbfs"] <= 0.0


def test_auto_tuner_handles_high_silence_ratio():
    tuner = AutoTuner()
    health = _health(snr_db=24.0, silence_ratio=0.8)
    audio = [0.0] * 16000
    for i in range(4000):
        audio[i] = 0.015 * math.sin(2 * math.pi * 120 * (i / 16000.0))
    diar_conf = types.SimpleNamespace(
        vad_threshold=0.3,
        vad_min_speech_sec=0.5,
        vad_min_silence_sec=0.4,
        speech_pad_sec=0.18,
    )
    asr_config = {"beam_size": 2, "no_speech_threshold": 0.5}

    result = tuner.recommend(
        health=health,
        audio=audio,
        sr=16000,
        diar_config=diar_conf,
        asr_config=asr_config,
    )

    assert result.diarization.get("vad_min_silence_sec", 0.0) >= 0.6
    assert result.asr.get("no_speech_threshold", 0.0) >= 0.55
    assert "high_silence_extend_min_silence" in result.notes
