import json
import types

import math
import numpy as np
import pytest

from diaremot.pipeline.logging_utils import StageGuard
from diaremot.pipeline.orchestrator import AudioAnalysisPipelineV2
from diaremot.pipeline.outputs import default_affect
from diaremot.pipeline.stages import PIPELINE_STAGES, PipelineState, dependency_check
from diaremot.pipeline.stages.affect import run as run_affect_stage
from diaremot.pipeline.stages.summaries import run_overlap


def test_stage_registry_order():
    names = [stage.name for stage in PIPELINE_STAGES]
    assert names == [
        "dependency_check",
        "preprocess",
        "background_sed",
        "diarize",
        "transcribe",
        "paralinguistics",
        "affect_and_assemble",
        "overlap_interruptions",
        "conversation_analysis",
        "speaker_rollups",
        "outputs",
    ]


def test_pipeline_init_recovers_missing_affect(tmp_path, monkeypatch):
    import diaremot.pipeline.orchestrator as orch

    class _StubPreprocessor:
        def __init__(self, *_args, **_kwargs):
            pass

    class _StubDiarizer:
        def __init__(self, *_args, **_kwargs):
            pass

    class _StubTranscriber:
        def __init__(self, *_args, **_kwargs):
            pass

    class _StubHTML:
        def render_to_html(self, *_args, **_kwargs):
            return "out.html"

    class _StubPDF:
        def render_to_pdf(self, *_args, **_kwargs):
            return "out.pdf"

    def _boom(*_args, **_kwargs):
        raise NameError("backend")

    monkeypatch.setattr(orch, "AudioPreprocessor", _StubPreprocessor)
    monkeypatch.setattr(orch, "SpeakerDiarizer", _StubDiarizer)
    monkeypatch.setattr(orch, "PANNSEventTagger", None)
    monkeypatch.setattr(orch, "HTMLSummaryGenerator", _StubHTML)
    monkeypatch.setattr(orch, "PDFSummaryGenerator", _StubPDF)
    monkeypatch.setattr(orch, "EmotionIntentAnalyzer", _boom)
    monkeypatch.setattr(
        "diaremot.pipeline.transcription_module.AudioTranscriber",
        _StubTranscriber,
    )

    pipeline = AudioAnalysisPipelineV2(
        config={
            "log_dir": str(tmp_path / "logs"),
            "checkpoint_dir": str(tmp_path / "chk"),
            "cache_root": tmp_path / "cache",
        }
    )

    # Attributes should exist even when analyzer construction fails
    assert hasattr(pipeline, "affect")
    assert pipeline.affect is None
    # HTML generator may not initialize when affect construction fails, but attribute exists
    assert hasattr(pipeline, "html")
    assert hasattr(pipeline, "pdf") and pipeline.pdf is not None


@pytest.fixture
def stub_pipeline(tmp_path, monkeypatch):
    def _stub_init(self, cfg):
        class _StubPreprocessor:
            def process_file(self, *_args, **_kwargs):
                return (
                    np.zeros(16000, dtype=np.float32),
                    16000,
                    types.SimpleNamespace(snr_db=25.0),
                )

        class _StubDiarizer:
            def diarize_audio(self, *_args, **_kwargs):
                return [
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "speaker": "S1",
                        "speaker_name": "Speaker_1",
                    }
                ]

        class _StubTranscriber:
            def transcribe_segments(self, wav, sr, segments):
                out = []
                for seg in segments:
                    out.append(
                        types.SimpleNamespace(
                            start_time=seg["start_time"],
                            end_time=seg["end_time"],
                            speaker_id=seg["speaker_id"],
                            speaker_name=seg["speaker_name"],
                            text="hello world",
                            asr_logprob_avg=0.0,
                            snr_db=20.0,
                        )
                    )
                return out

            def get_model_info(self):
                return {"fallback_triggered": False}

        class _StubAffect:
            def analyze(self, **_kwargs):
                return default_affect()

        self.pp_conf = types.SimpleNamespace(
            target_sr=16000, denoise="spectral_sub_soft", loudness_mode="asr"
        )
        self.pre = _StubPreprocessor()
        self.diar_conf = types.SimpleNamespace(
            registry_path=str(tmp_path / "registry.json"),
            ahc_distance_threshold=0.12,
            speaker_limit=None,
            vad_threshold=0.22,
            vad_min_speech_sec=0.40,
            vad_min_silence_sec=0.40,
            speech_pad_sec=0.15,
            energy_gate_db=-33.0,
            energy_hop_sec=0.01,
        )
        self.diar = _StubDiarizer()
        self.tx = _StubTranscriber()
        self.affect = _StubAffect()
        self.sed_tagger = None
        self.html = types.SimpleNamespace(render_to_html=lambda *args, **kwargs: None)
        self.pdf = types.SimpleNamespace(render_to_pdf=lambda *args, **kwargs: None)
        from diaremot.pipeline.auto_tuner import AutoTuner

        self.auto_tuner = AutoTuner()
        self.stats.models.update(
            {
                "preprocessor": "StubPreprocessor",
                "diarizer": "StubDiarizer",
                "transcriber": "StubTranscriber",
                "affect": "StubAffect",
            }
        )
        self.stats.config_snapshot = {
            "transcribe_failed": False,
            "dependency_ok": True,
            "dependency_summary": {},
            "auto_tune": {
                "metrics": {},
                "notes": ["pending"],
                "applied": {"diarization": {}, "asr": {}},
            },
        }

    monkeypatch.setattr(AudioAnalysisPipelineV2, "_init_components", _stub_init)

    pipeline = AudioAnalysisPipelineV2(
        config={
            "log_dir": str(tmp_path / "logs"),
            "checkpoint_dir": str(tmp_path / "chk"),
            "cache_root": tmp_path / "cache",
        }
    )
    pipeline.stats.file_id = "sample.wav"
    return pipeline


def test_stage_services_execute_full_cycle(tmp_path, monkeypatch, stub_pipeline):
    pipeline = stub_pipeline

    monkeypatch.setattr(
        dependency_check,
        "dependency_health_summary",
        lambda: {"core": {"status": "ok"}},
    )

    captured = {}

    def _capture_outputs(
        self,
        _input_audio_path,
        _outp,
        segments_final,
        speakers_summary,
        *_rest,
    ):
        captured["segments"] = segments_final
        captured["speakers"] = speakers_summary

    monkeypatch.setattr(AudioAnalysisPipelineV2, "_write_outputs", _capture_outputs)

    out_dir = tmp_path / "out"
    state = PipelineState(input_audio_path="dummy.wav", out_dir=out_dir)

    sed_injection = {
        "top": [
            {"label": "Speech", "score": 0.72},
            {"label": "Music", "score": 0.18},
            {"label": "Typing", "score": 0.10},
        ],
        "dominant_label": "Speech",
        "noise_score": 0.32,
    }

    for stage in PIPELINE_STAGES:
        with StageGuard(pipeline.corelog, pipeline.stats, stage.name) as guard:
            if stage.name == "affect_and_assemble":
                state.sed_info = sed_injection
            stage.runner(pipeline, state, guard)

    assert captured["segments"], "affect stage should produce segments"
    first_segment = captured["segments"][0]
    assert first_segment["text"] == "hello world"
    sed_events = json.loads(first_segment["events_top3_json"])
    assert sed_events and sed_events[0]["label"] == "Speech"
    assert first_segment["noise_tag"] == "Speech"
    assert isinstance(first_segment["snr_db_sed"], float)
    assert state.norm_tx and state.turns
    assert state.overlap_stats is not None
    assert isinstance(state.speakers_summary, list)
    assert pipeline.stats.config_snapshot.get("dependency_ok") is True
    assert "auto_tune" in pipeline.stats.config_snapshot
    assert isinstance(state.tuning_summary, dict)
    assert isinstance(state.tuning_history, list)
odex/update-affect.py-to-handle-sed-fields


def test_affect_stage_uses_paralinguistics_metrics(tmp_path, stub_pipeline):
    pipeline = stub_pipelineC

    state = PipelineState(input_audio_path="dummy.wav", out_dir=tmp_path)
    state.sr = 16000
    state.y = np.zeros(int(3 * state.sr), dtype=np.float32)
    state.norm_tx = [
        {
            "start": 1.0,
            "end": 2.5,
            "speaker_id": "S1",
            "speaker_name": "Speaker_1",
            "text": "alpha beta gamma delta epsilon",
        }
    ]
    state.para_metrics = {
        0: {
            "wpm": 120.0,
            "duration_s": 1.5,
            "words": 5,
            "pause_ratio": 0.25,
            "pause_time_s": 0.375,
        }
    }

    with StageGuard(pipeline.corelog, pipeline.stats, "affect_and_assemble") as guard:
        run_affect_stage(pipeline, state, guard)

    assert state.segments_final, "Affect stage must emit segments"
    seg = state.segments_final[0]
    assert math.isclose(seg["duration_s"], 1.5, rel_tol=1e-6)
    assert seg["words"] == 5
    assert math.isclose(seg["pause_ratio"], 0.25, rel_tol=1e-6)


def test_affect_stage_computes_fallback_metrics(tmp_path, stub_pipeline):
    pipeline = stub_pipeline

    state = PipelineState(input_audio_path="dummy.wav", out_dir=tmp_path)
    state.sr = 16000
    state.y = np.zeros(int(5 * state.sr), dtype=np.float32)
    state.norm_tx = [
        {
            "start": 0.5,
            "end": 3.5,
            "speaker_id": "S1",
            "speaker_name": "Speaker_1",
            "text": "fallback path validation",
        }
    ]
    state.para_metrics = {}

    with StageGuard(pipeline.corelog, pipeline.stats, "affect_and_assemble") as guard:
        run_affect_stage(pipeline, state, guard)

    seg = state.segments_final[0]
    assert math.isclose(seg["duration_s"], 3.0, rel_tol=1e-6)
    assert seg["words"] == 3
    assert seg["pause_ratio"] == 0.0


def test_affect_stage_uses_paralinguistics_metrics(tmp_path, stub_pipeline):
    pipeline = stub_pipeline

    state = PipelineState(input_audio_path="dummy.wav", out_dir=tmp_path)
    state.sr = 16000
    state.y = np.zeros(int(3 * state.sr), dtype=np.float32)
    state.norm_tx = [
        {
            "start": 1.0,
            "end": 2.5,
            "speaker_id": "S1",
            "speaker_name": "Speaker_1",
            "text": "alpha beta gamma delta epsilon",
        }
    ]
    state.para_metrics = {
        0: {
            "wpm": 120.0,
            "duration_s": 1.5,
            "words": 5,
            "pause_ratio": 0.25,
            "pause_time_s": 0.375,
        }
    }

    with StageGuard(pipeline.corelog, pipeline.stats, "affect_and_assemble") as guard:
        run_affect_stage(pipeline, state, guard)

    assert state.segments_final, "Affect stage must emit segments"
    seg = state.segments_final[0]
    assert math.isclose(seg["duration_s"], 1.5, rel_tol=1e-6)
    assert seg["words"] == 5
    assert math.isclose(seg["pause_ratio"], 0.25, rel_tol=1e-6)


def test_affect_stage_computes_fallback_metrics(tmp_path, stub_pipeline):
    pipeline = stub_pipeline

    state = PipelineState(input_audio_path="dummy.wav", out_dir=tmp_path)
    state.sr = 16000
    state.y = np.zeros(int(5 * state.sr), dtype=np.float32)
    state.norm_tx = [
        {
            "start": 0.5,
            "end": 3.5,
            "speaker_id": "S1",
            "speaker_name": "Speaker_1",
            "text": "fallback path validation",
        }
    ]
    state.para_metrics = {}

    with StageGuard(pipeline.corelog, pipeline.stats, "affect_and_assemble") as guard:
        run_affect_stage(pipeline, state, guard)

    seg = state.segments_final[0]
    assert math.isclose(seg["duration_s"], 3.0, rel_tol=1e-6)
    assert seg["words"] == 3
    assert seg["pause_ratio"] == 0.0


def test_run_overlap_maps_interruptions(tmp_path, stub_pipeline):
    pipeline = stub_pipeline

    sample_overlap = {
        "overlap_total_sec": 4.0,
        "overlap_ratio": 0.4,
        "by_speaker": {
            "S1": {"overlap_sec": 2.5, "interruptions": 2},
            "S2": {"overlap_sec": 1.5, "interruptions": 1},
        },
        "interruptions": [
            {"interrupter": "S1", "interrupted": "S2"},
            {"interrupter": "S1", "interrupted": "S2"},
            {"interrupter": "S2", "interrupted": "S1"},
            {"interrupter": "S3", "interrupted": "S1"},
        ],
    }

    class _StubParalinguistics:
        def compute_overlap_and_interruptions(self, turns):
            assert turns == state.turns
            return sample_overlap

    pipeline.paralinguistics_module = _StubParalinguistics()

    out_dir = tmp_path / "out"
    state = PipelineState(input_audio_path="audio.wav", out_dir=out_dir)
    state.turns = [
        {"start": 0.0, "end": 1.0, "speaker_id": "S1"},
        {"start": 1.0, "end": 2.0, "speaker_id": "S2"},
    ]

    with StageGuard(pipeline.corelog, pipeline.stats, "overlap_interruptions") as guard:
        run_overlap(pipeline, state, guard)

    assert state.overlap_stats == {
        "overlap_total_sec": 4.0,
        "overlap_ratio": 0.4,
    }
    assert state.per_speaker_interrupts == {
        "S1": {"made": 2, "received": 2, "overlap_sec": 2.5},
        "S2": {"made": 1, "received": 2, "overlap_sec": 1.5},
        "S3": {"made": 1, "received": 0, "overlap_sec": 0.0},
    }
