from pathlib import Path

import importlib
import logging
import sys
import types
import uuid

import pytest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - exercised only when numpy missing
    pytest.skip("numpy is required for stageguard tests", allow_module_level=True)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _register_stub(name: str, builder):
    if name in sys.modules:
        return sys.modules[name]

    parent = None
    attr = None
    if "." in name:
        parent_name, attr = name.rsplit(".", 1)
        try:
            parent = importlib.import_module(parent_name)
        except Exception:
            parent = types.ModuleType(parent_name)
            parent.__path__ = []  # type: ignore[attr-defined]
            sys.modules[parent_name] = parent

    module = builder()
    sys.modules[name] = module
    if parent is not None and attr is not None:
        setattr(parent, attr, module)
    return module


def _install_audio_pipeline_stubs() -> None:
    def _librosa_builder():
        module = types.ModuleType("librosa")
        module.util = types.SimpleNamespace(frame=lambda *args, **kwargs: [])
        return module

    _register_stub("librosa", _librosa_builder)
    if "librosa" in sys.modules and not hasattr(sys.modules["librosa"], "util"):
        sys.modules["librosa"].util = types.SimpleNamespace(
            frame=lambda *args, **kwargs: []
        )

    def _scipy_builder():
        scipy_stub = types.ModuleType("scipy")
        scipy_stub.signal = types.SimpleNamespace(
            resample_poly=lambda audio, up, down: audio
        )
        return scipy_stub

    _register_stub("scipy", _scipy_builder)
    if "scipy.signal" not in sys.modules:
        sys.modules["scipy.signal"] = types.SimpleNamespace(
            resample_poly=lambda audio, up, down: audio
        )

    def _html_builder():
        module = types.ModuleType("diaremot.summaries.html_summary_generator")

        class _HTMLSummaryGenerator:
            def render_to_html(self, *args, **kwargs):
                return None

        module.HTMLSummaryGenerator = _HTMLSummaryGenerator
        return module

    _register_stub("diaremot.summaries.html_summary_generator", _html_builder)

    def _speakers_builder():
        module = types.ModuleType("diaremot.summaries.speakers_summary_builder")
        module.build_speakers_summary = lambda *args, **kwargs: []
        return module

    _register_stub("diaremot.summaries.speakers_summary_builder", _speakers_builder)

    def _pdf_builder():
        module = types.ModuleType("diaremot.summaries.pdf_summary_generator")

        class _PDFSummaryGenerator:
            def render_to_pdf(self, *args, **kwargs):
                return None

        module.PDFSummaryGenerator = _PDFSummaryGenerator
        return module

    _register_stub("diaremot.summaries.pdf_summary_generator", _pdf_builder)

    def _emotion_builder():
        module = types.ModuleType("diaremot.affect.emotion_analyzer")

        class _EmotionIntentAnalyzer:
            def __init__(self, *args, **kwargs):
                self.init_args = args
                self.init_kwargs = kwargs

        module.EmotionIntentAnalyzer = _EmotionIntentAnalyzer
        return module

    _register_stub("diaremot.affect.emotion_analyzer", _emotion_builder)

    def _conversation_builder():
        module = types.ModuleType("diaremot.summaries.conversation_analysis")

        class _ConversationMetrics:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        def _analyze_conversation_flow(*args, **kwargs):
            return _ConversationMetrics(
                turn_taking_balance=0.5,
                interruption_rate_per_min=0.0,
                avg_turn_duration_sec=0.0,
                conversation_pace_turns_per_min=0.0,
                silence_ratio=0.0,
                speaker_dominance={},
                response_latency_stats={},
                topic_coherence_score=0.0,
                energy_flow=[],
            )

        module.ConversationMetrics = _ConversationMetrics
        module.analyze_conversation_flow = _analyze_conversation_flow
        return module

    _register_stub("diaremot.summaries.conversation_analysis", _conversation_builder)

    def _sed_builder():
        module = types.ModuleType("diaremot.affect.sed_panns")

        class _SEDConfig:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        class _PANNSEventTagger:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

            def tag(self, *args, **kwargs):
                return {}

        module.SEDConfig = _SEDConfig
        module.PANNSEventTagger = _PANNSEventTagger
        return module

    _register_stub("diaremot.affect.sed_panns", _sed_builder)

    def _intent_defaults_builder():
        module = types.ModuleType("diaremot.affect.intent_defaults")
        module.INTENT_LABELS_DEFAULT = [
            "question",
            "request",
            "instruction",
        ]
        return module

    _register_stub("diaremot.affect.intent_defaults", _intent_defaults_builder)

    def _paralinguistics_builder():
        module = types.ModuleType("diaremot.affect.paralinguistics")
        module.PARALINGUISTICS_AVAILABLE = False
        return module

    _register_stub("diaremot.affect.paralinguistics", _paralinguistics_builder)

    def _preprocess_builder():
        module = types.ModuleType("diaremot.pipeline.audio_preprocessing")

        class _PreprocessConfig:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class _AudioPreprocessor:
            def __init__(self, config):
                self.config = config

            def process_file(self, *args, **kwargs):
                return ([], 16000, None)

        module.PreprocessConfig = _PreprocessConfig
        module.AudioPreprocessor = _AudioPreprocessor
        return module

    _register_stub("diaremot.pipeline.audio_preprocessing", _preprocess_builder)

    def _transcribe_builder():
        module = types.ModuleType("diaremot.pipeline.transcription_module")

        class _TranscriptionSegment:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        class _AudioTranscriber:
            def __init__(self, **kwargs):
                self.init_kwargs = kwargs

            def transcribe_segments(self, *args, **kwargs):
                return []

        module.TranscriptionSegment = _TranscriptionSegment
        module.AudioTranscriber = _AudioTranscriber
        return module

    _register_stub("diaremot.pipeline.transcription_module", _transcribe_builder)

    def _checkpoint_builder():
        module = types.ModuleType("diaremot.pipeline.pipeline_checkpoint_system")

        class _ProcessingStage:
            AUDIO_PREPROCESSING = "preprocess"
            DIARIZATION = "diarize"
            TRANSCRIPTION = "transcribe"
            SUMMARY_GENERATION = "summary"
            COMPLETE = "complete"

        class _PipelineCheckpointManager:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

            def create_checkpoint(self, *args, **kwargs):
                return None

        module.ProcessingStage = _ProcessingStage
        module.PipelineCheckpointManager = _PipelineCheckpointManager
        return module

    _register_stub("diaremot.pipeline.pipeline_checkpoint_system", _checkpoint_builder)

    def _diarizer_builder():
        module = types.ModuleType("diaremot.pipeline.speaker_diarization")

        class _DiarizationConfig:
            ahc_distance_threshold = 0.02
            vad_threshold = 0.3
            vad_min_speech_sec = 0.8
            vad_min_silence_sec = 0.8
            speech_pad_sec = 0.2
            energy_gate_db = -33.0
            energy_hop_sec = 0.01

            def __init__(
                self,
                target_sr,
                registry_path,
                ahc_distance_threshold,
                speaker_limit,
                ecapa_model_path,
                vad_backend,
                vad_threshold,
                vad_min_speech_sec,
                vad_min_silence_sec,
                speech_pad_sec,
                allow_energy_vad_fallback,
                energy_gate_db,
                energy_hop_sec,
            ):
                self.target_sr = target_sr
                self.registry_path = registry_path
                self.ahc_distance_threshold = ahc_distance_threshold
                self.speaker_limit = speaker_limit
                self.ecapa_model_path = ecapa_model_path
                self.vad_backend = vad_backend
                self.vad_threshold = vad_threshold
                self.vad_min_speech_sec = vad_min_speech_sec
                self.vad_min_silence_sec = vad_min_silence_sec
                self.speech_pad_sec = speech_pad_sec
                self.allow_energy_vad_fallback = allow_energy_vad_fallback
                self.energy_gate_db = energy_gate_db
                self.energy_hop_sec = energy_hop_sec

        class _SpeakerDiarizer:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        class _SpeakerRegistry:
            def __init__(self, *args, **kwargs):
                pass

        module.DiarizationConfig = _DiarizationConfig
        module.SpeakerDiarizer = _SpeakerDiarizer
        module.SpeakerRegistry = _SpeakerRegistry
        module._agglo = None
        return module

    _register_stub("diaremot.pipeline.speaker_diarization", _diarizer_builder)

    def _cpu_diar_builder():
        module = types.ModuleType("diaremot.pipeline.cpu_optimized_diarizer")

        class _CPUOptimizationConfig:
            def __init__(self, max_speakers=None):
                self.max_speakers = max_speakers

        class _CPUOptimizedSpeakerDiarizer:
            def __init__(self, base_diarizer, config):
                self.base = base_diarizer
                self.config = config

        module.CPUOptimizationConfig = _CPUOptimizationConfig
        module.CPUOptimizedSpeakerDiarizer = _CPUOptimizedSpeakerDiarizer
        return module

    _register_stub("diaremot.pipeline.cpu_optimized_diarizer", _cpu_diar_builder)


_install_audio_pipeline_stubs()

from diaremot.pipeline.logging_utils import CoreLogger, RunStats, StageGuard  # noqa: E402
from diaremot.pipeline.orchestrator import AudioAnalysisPipelineV2  # noqa: E402
from diaremot.pipeline.outputs import default_affect  # noqa: E402

if getattr(np, "__stub__", False) or not hasattr(np, "array"):
    pytest.skip("numpy not available", allow_module_level=True)


def _make_guard(tmp_path, stage: str):
    run_token = uuid.uuid4().hex
    stats = RunStats(run_id=f"test-{run_token}", file_id="file.wav")
    corelog = CoreLogger(
        run_id=f"stageguard-{stage}-{run_token}",
        jsonl_path=tmp_path / f"{stage}.jsonl",
        console_level=logging.CRITICAL,
    )
    return corelog, stats


def test_stageguard_handles_timeout(tmp_path):
    corelog, stats = _make_guard(tmp_path, "transcribe")
    with StageGuard(corelog, stats, "transcribe"):
        raise TimeoutError("segment exceeded deadline")

    assert stats.config_snapshot.get("transcribe_failed") is True
    assert any(f["stage"] == "transcribe" for f in stats.failures)


def test_stageguard_reraises_unexpected(tmp_path):
    corelog, stats = _make_guard(tmp_path, "transcribe")
    with pytest.raises(ValueError):
        with StageGuard(corelog, stats, "transcribe"):
            raise ValueError("unexpected bug")

    assert stats.config_snapshot.get("transcribe_failed") is True
    assert any("unexpected bug" in f["error"] for f in stats.failures)


def test_stageguard_optional_module_missing(tmp_path):
    corelog, stats = _make_guard(tmp_path, "paralinguistics")
    with StageGuard(corelog, stats, "paralinguistics"):
        raise ModuleNotFoundError("librosa not installed")

    assert any(f["stage"] == "paralinguistics" for f in stats.failures)
    assert any("librosa" in f["error"] for f in stats.failures)


def test_stageguard_preprocess_is_critical(tmp_path):
    corelog, stats = _make_guard(tmp_path, "preprocess")
    with pytest.raises(ModuleNotFoundError):
        with StageGuard(corelog, stats, "preprocess"):
            raise ModuleNotFoundError("ffmpeg missing")

    assert stats.config_snapshot.get("preprocess_failed") is True


def test_process_audio_file_handles_missing_paraling(tmp_path, monkeypatch):
    captured = {}

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
            def transcribe_segments(self, wav, sr, tx_in):
                segments = []
                for seg in tx_in:
                    segments.append(
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
                return segments

            def get_model_info(self):
                return {"fallback_triggered": False}

        class _StubAffect:
            def analyze(self, **_kwargs):
                return default_affect()

        self.pp_conf = types.SimpleNamespace(target_sr=16000)
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
        }

    monkeypatch.setattr(AudioAnalysisPipelineV2, "_init_components", _stub_init)

    pipeline = AudioAnalysisPipelineV2(
        config={
            "log_dir": str(tmp_path / "logs"),
            "checkpoint_dir": str(tmp_path / "chk"),
            "cache_root": tmp_path / "cache",
        }
    )

    def _boom(*_args, **_kwargs):
        raise ModuleNotFoundError("paralinguistics extras not installed")

    monkeypatch.setattr(pipeline, "_extract_paraling", _boom)

    def _capture_outputs(
        self,
        _input_audio_path,
        _outp,
        segments_final,
        *_rest,
    ):
        captured["segments"] = segments_final

    monkeypatch.setattr(AudioAnalysisPipelineV2, "_write_outputs", _capture_outputs)

    manifest = pipeline.process_audio_file("dummy.wav", str(tmp_path / "out"))

    assert manifest["run_id"] == pipeline.run_id
    assert captured["segments"]
    segment = captured["segments"][0]
    assert segment["valence"] == 0.0
    assert segment["emotion_top"] == "neutral"
    assert segment["wpm"] == 0.0
    assert segment["pause_count"] == 0


def test_cpu_diarizer_uses_optimized_wrapper(tmp_path, monkeypatch):
    cpu_module = importlib.import_module("diaremot.pipeline.cpu_optimized_diarizer")
    diar_module = importlib.import_module("diaremot.pipeline.speaker_diarization")
    core_module = importlib.import_module("diaremot.pipeline.audio_pipeline_core")

    class _TrackingSpeakerDiarizer:
        def __init__(self, config):
            self.config = config

    monkeypatch.setattr(diar_module, "SpeakerDiarizer", _TrackingSpeakerDiarizer)
    monkeypatch.setattr(core_module, "SpeakerDiarizer", _TrackingSpeakerDiarizer)

    pipeline = AudioAnalysisPipelineV2(
        config={
            "log_dir": str(tmp_path / "logs"),
            "checkpoint_dir": str(tmp_path / "chk"),
            "cache_root": tmp_path / "cache",
            "cpu_diarizer": True,
            "speaker_limit": 3,
        }
    )

    assert isinstance(pipeline.diar, cpu_module.CPUOptimizedSpeakerDiarizer)
    assert isinstance(pipeline.diar.base, _TrackingSpeakerDiarizer)
    assert pipeline.diar.config.max_speakers == 3

