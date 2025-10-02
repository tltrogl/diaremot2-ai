from pathlib import Path

import pytest

from diaremot.affect.emotion_analyzer import EmotionIntentAnalyzer


def test_intent_model_dir_from_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "bart_model"
    target.mkdir()
    (target / "config.json").write_text("{}")
    (target / "pytorch_model.bin").write_bytes(b"")
    monkeypatch.delenv("DIAREMOT_MODEL_DIR", raising=False)
    monkeypatch.setenv("DIAREMOT_INTENT_MODEL_DIR", str(target))
    analyzer = EmotionIntentAnalyzer(affect_intent_model_dir=None)
    assert analyzer.affect_intent_model_dir == str(target)


def test_intent_model_dir_from_model_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("DIAREMOT_INTENT_MODEL_DIR", raising=False)
    model_root = tmp_path / "models"
    bart_dir = model_root / "bart"
    nested = bart_dir / "facebook" / "bart-large-mnli"
    nested.mkdir(parents=True)
    (nested / "config.json").write_text("{}")
    (nested / "pytorch_model-00001-of-00002.bin").write_bytes(b"")
    (nested / "pytorch_model.bin.index.json").write_text("{}")
    monkeypatch.setenv("DIAREMOT_MODEL_DIR", str(model_root))
    analyzer = EmotionIntentAnalyzer(affect_intent_model_dir=None)
    assert analyzer.affect_intent_model_dir == str(nested)


def test_intent_model_dir_missing_weights(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    invalid_dir = tmp_path / "bart_model"
    invalid_dir.mkdir()
    monkeypatch.setenv("DIAREMOT_INTENT_MODEL_DIR", str(invalid_dir))
    analyzer = EmotionIntentAnalyzer(affect_intent_model_dir=None)
    assert analyzer.affect_intent_model_dir is None
