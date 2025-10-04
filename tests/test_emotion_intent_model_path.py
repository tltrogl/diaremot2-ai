from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from diaremot.affect.emotion_analyzer import EmotionIntentAnalyzer


def test_intent_model_dir_from_env_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
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


def test_intent_onnx_backend_uses_local_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_dir = tmp_path / "bart_onnx"
    model_dir.mkdir()
    (model_dir / "model.onnx").write_bytes(b"")
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "id2label": {
                    "0": "CONTRADICTION",
                    "1": "NEUTRAL",
                    "2": "ENTAILMENT",
                }
            }
        )
    )

    class _DummyTokenizer:
        def __init__(self) -> None:
            self.last_hypothesis: str | None = None

        def __call__(
            self,
            text: str,
            hypothesis: str,
            *,
            return_tensors: str = "np",
            truncation: bool = True,
        ) -> dict[str, np.ndarray]:
            assert return_tensors == "np"
            assert truncation is True
            self.last_hypothesis = hypothesis
            return {
                "input_ids": np.zeros((1, 4), dtype=np.int64),
                "attention_mask": np.ones((1, 4), dtype=np.int64),
            }

    created_tokenizer: dict[str, _DummyTokenizer] = {}

    class _DummyAutoTokenizer:
        @staticmethod
        def from_pretrained(path: str) -> _DummyTokenizer:
            assert Path(path) == model_dir
            tokenizer = _DummyTokenizer()
            created_tokenizer["instance"] = tokenizer
            return tokenizer

    class _DummyAutoConfig:
        def __init__(self) -> None:
            self.id2label = {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}
            self.hypothesis_template = "This example is {}."

        @classmethod
        def from_pretrained(cls, path: str) -> _DummyAutoConfig:
            assert Path(path) == model_dir
            return cls()

    def _dummy_pipeline(*args, **kwargs):
        raise AssertionError("Torch pipeline should not be used for ONNX backend")

    dummy_module = types.SimpleNamespace(
        AutoTokenizer=_DummyAutoTokenizer,
        AutoConfig=_DummyAutoConfig,
        pipeline=_dummy_pipeline,
    )
    monkeypatch.setitem(sys.modules, "transformers", dummy_module)

    class _DummySession:
        def __init__(self, tokenizer_ref: dict[str, _DummyTokenizer]) -> None:
            self.tokenizer_ref = tokenizer_ref

        def run(self, *_args, **_kwargs):
            tokenizer = self.tokenizer_ref.get("instance")
            hypothesis = tokenizer.last_hypothesis if tokenizer else ""
            if "question" in hypothesis:
                return [np.array([[-1.0, 0.0, 2.0]], dtype=np.float32)]
            return [np.array([[2.0, 0.0, -1.0]], dtype=np.float32)]

    monkeypatch.setattr(
        "diaremot.affect.emotion_analyzer.create_onnx_session",
        lambda _path: _DummySession(created_tokenizer),
    )

    analyzer = EmotionIntentAnalyzer(affect_backend="onnx", affect_intent_model_dir=str(model_dir))

    top, top3 = analyzer._infer_intent("Is this working?")

    assert top == "question"
    assert top3[0]["label"] == "question"
