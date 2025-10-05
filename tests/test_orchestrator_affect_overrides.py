from __future__ import annotations

import os
from pathlib import Path

from diaremot.pipeline import orchestrator as orchestrator_mod
from diaremot.pipeline.orchestrator import AudioAnalysisPipelineV2, build_pipeline_config


class _DummyComponent:
    def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - simple stub
        self.args = args
        self.kwargs = kwargs


class _DummyTranscriber:
    def __init__(self, **kwargs) -> None:  # pragma: no cover - simple stub
        self.kwargs = kwargs

    def get_model_info(self) -> dict[str, object]:  # pragma: no cover - simple stub
        return {}


def test_affect_overrides_passed_to_analyzer(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class _RecordingAnalyzer:
        def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - test helper
            captured["args"] = args
            captured["kwargs"] = kwargs

    monkeypatch.setattr(orchestrator_mod, "EmotionIntentAnalyzer", _RecordingAnalyzer)
    monkeypatch.setattr(orchestrator_mod, "AudioPreprocessor", _DummyComponent)
    monkeypatch.setattr(orchestrator_mod, "SpeakerDiarizer", _DummyComponent)
    monkeypatch.setattr(orchestrator_mod, "HTMLSummaryGenerator", lambda: _DummyComponent())
    monkeypatch.setattr(orchestrator_mod, "PDFSummaryGenerator", lambda: _DummyComponent())
    monkeypatch.setattr(orchestrator_mod, "PANNSEventTagger", None)
    monkeypatch.setattr(
        "diaremot.pipeline.transcription_module.AudioTranscriber",
        _DummyTranscriber,
    )

    text_dir = tmp_path / "custom_text"
    intent_dir = tmp_path / "custom_intent"
    text_dir.mkdir()
    intent_dir.mkdir()

    cfg = build_pipeline_config(
        {
            "affect_backend": "torch",
            "affect_text_model_dir": text_dir,
            "affect_intent_model_dir": intent_dir,
            "affect_analyzer_threads": 3,
            "log_dir": tmp_path / "logs",
            "cache_root": tmp_path / "cache",
            "checkpoint_dir": tmp_path / "checkpoints",
        }
    )

    pipeline = AudioAnalysisPipelineV2(cfg)

    assert "kwargs" in captured, "EmotionIntentAnalyzer was not initialised"
    kwargs = captured["kwargs"]
    assert kwargs["affect_backend"] == "torch"
    assert kwargs["affect_text_model_dir"] == os.fspath(text_dir)
    assert kwargs["affect_intent_model_dir"] == os.fspath(intent_dir)
    assert kwargs["analyzer_threads"] == 3

    snapshot = pipeline.stats.config_snapshot
    assert snapshot["affect_backend"] == "torch"
    assert snapshot["affect_text_model_dir"] == os.fspath(text_dir)
    assert snapshot["affect_intent_model_dir"] == os.fspath(intent_dir)
    assert snapshot["affect_analyzer_threads"] == 3
