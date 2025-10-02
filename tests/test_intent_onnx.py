import sys
import types
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class _DummyTokenizer:
    @classmethod
    def from_pretrained(cls, model_dir):
        return cls()

    def __call__(
        self,
        texts,
        hypotheses,
        *,
        return_tensors="np",
        padding=False,
        truncation=False,
    ):
        batch = len(hypotheses)
        seq_len = 4
        input_ids = [[0.0] * seq_len for _ in range(batch)]
        attention_mask = [[1.0] * seq_len for _ in range(batch)]
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _DummyConfig:
    label2id = {"contradiction": 0, "neutral": 1, "entailment": 2}

    @classmethod
    def from_pretrained(cls, model_dir):
        return cls()


def _install_transformers_stub():
    module = types.ModuleType("transformers")
    module.AutoTokenizer = _DummyTokenizer
    module.AutoConfig = _DummyConfig

    def _pipeline(*args, **kwargs):
        class _Pipe:
            def __call__(self, text, candidate_labels, multi_label=False):
                scores = [1.0 / max(1, len(candidate_labels))] * len(candidate_labels)
                return {"labels": list(candidate_labels), "scores": scores}

        return _Pipe()

    module.pipeline = _pipeline
    sys.modules["transformers"] = module


def test_intent_onnx_prefers_local_uint8(tmp_path, monkeypatch):
    _install_transformers_stub()

    from diaremot.affect import emotion_analyzer

    model_dir = tmp_path / "bart"
    model_dir.mkdir()
    preferred = model_dir / "model_uint8.onnx"
    preferred.write_bytes(b"onnx")

    captured = {}

    def _fake_ensure(path_like):
        candidate = Path(path_like)
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        return candidate

    def _fake_session(path, **kwargs):
        captured["path"] = Path(path)

        class _Session:
            def run(self, *_args):
                batch = 3
                logits = np.array(
                    [
                        [6.0, 0.0, 8.0],
                        [0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.5],
                    ],
                    dtype=np.float32,
                )
                return [logits]

        return _Session()

    monkeypatch.setattr(emotion_analyzer, "ensure_onnx_model", _fake_ensure)
    monkeypatch.setattr(emotion_analyzer, "create_onnx_session", _fake_session)

    analyzer = emotion_analyzer.EmotionIntentAnalyzer(
        affect_backend="onnx",
        affect_intent_model_dir=str(model_dir),
        intent_labels=["support", "sales", "other"],
    )

    assert analyzer.intent_labels == ["support", "sales", "other"]

    def _raise_rules(self, text):
        raise RuntimeError("intent rules should not run")

    monkeypatch.setattr(
        emotion_analyzer.EmotionIntentAnalyzer,
        "_intent_rules",
        _raise_rules,
        raising=False,
    )

    top, top3 = analyzer._infer_intent("Need help with installation")

    assert captured["path"] == preferred
    assert top == "support"
    assert top3[0]["label"] == top, top3
    assert {entry["label"] for entry in top3} == {"support", "sales", "other"}


def test_intent_pipeline_fallback_when_no_text(monkeypatch):
    _install_transformers_stub()

    from diaremot.affect import emotion_analyzer

    analyzer = emotion_analyzer.EmotionIntentAnalyzer(affect_backend="onnx")

    monkeypatch.setattr(emotion_analyzer, "create_onnx_session", lambda *args, **kwargs: None)
    monkeypatch.setattr(emotion_analyzer, "ensure_onnx_model", lambda *args, **kwargs: Path("missing"))

    top, top3 = analyzer._infer_intent("")

    assert top in analyzer.intent_labels
    assert len(top3) == min(3, len(analyzer.intent_labels))
