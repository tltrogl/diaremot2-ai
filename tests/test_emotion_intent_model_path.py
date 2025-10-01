from pathlib import Path

import pytest

from diaremot.affect.emotion_analyzer import EmotionIntentAnalyzer


def test_intent_model_dir_from_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "bart_model"
    target.mkdir()
    monkeypatch.delenv("DIAREMOT_MODEL_DIR", raising=False)
    monkeypatch.setenv("DIAREMOT_INTENT_MODEL_DIR", str(target))
    analyzer = EmotionIntentAnalyzer(affect_intent_model_dir=None)
    assert analyzer.affect_intent_model_dir == str(target)


def test_intent_model_dir_from_model_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("DIAREMOT_INTENT_MODEL_DIR", raising=False)
    model_root = tmp_path / "models"
    bart_dir = model_root / "bart"
    bart_dir.mkdir(parents=True)
    monkeypatch.setenv("DIAREMOT_MODEL_DIR", str(model_root))
    analyzer = EmotionIntentAnalyzer(affect_intent_model_dir=None)
    assert analyzer.affect_intent_model_dir == str(bart_dir)
