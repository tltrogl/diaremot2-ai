from __future__ import annotations

import os
import sys
import types

import pytest

from diaremot.affect import emotion_analyzer as emo


class DummySession:
    pass


def _stub_session(path: str) -> DummySession:
    return DummySession()


def test_onnx_text_emotion_prefers_local_tokenizer(monkeypatch, tmp_path):
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"onnx")

    monkeypatch.setattr(emo, "_ort_session", _stub_session)

    load_calls: list[tuple[str, dict[str, object]]] = []

    class DummyAutoTokenizer:
        @staticmethod
        def from_pretrained(identifier: str, **kwargs):
            load_calls.append((identifier, kwargs))
            if identifier == os.fspath(tmp_path) and kwargs.get("local_files_only"):
                return "tokenizer-local"
            raise OSError("missing")

    fake_transformers = types.SimpleNamespace(AutoTokenizer=DummyAutoTokenizer)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    model = emo.OnnxTextEmotion(
        os.fspath(model_path),
        tokenizer_source=tmp_path,
        disable_downloads=True,
    )

    assert model.tokenizer == "tokenizer-local"
    assert load_calls[0][0] == os.fspath(tmp_path)
    assert load_calls[0][1].get("local_files_only") is True


def test_onnx_text_emotion_remote_fallback(monkeypatch, tmp_path):
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"onnx")

    monkeypatch.setattr(emo, "_ort_session", _stub_session)

    load_calls: list[str] = []

    class DummyAutoTokenizer:
        @staticmethod
        def from_pretrained(identifier: str, **kwargs):
            load_calls.append(identifier)
            if identifier == os.fspath(tmp_path):
                raise OSError("local missing")
            if identifier == "SamLowe/roberta-base-go_emotions":
                return "tokenizer-remote"
            raise AssertionError(f"Unexpected identifier {identifier}")

    fake_transformers = types.SimpleNamespace(AutoTokenizer=DummyAutoTokenizer)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    model = emo.OnnxTextEmotion(os.fspath(model_path), tokenizer_source=tmp_path)

    assert model.tokenizer == "tokenizer-remote"
    assert load_calls == [os.fspath(tmp_path), os.fspath(tmp_path), "SamLowe/roberta-base-go_emotions"]


def test_onnx_text_emotion_disable_downloads_raises(monkeypatch, tmp_path):
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"onnx")

    monkeypatch.setattr(emo, "_ort_session", _stub_session)

    class DummyAutoTokenizer:
        @staticmethod
        def from_pretrained(identifier: str, **kwargs):
            raise OSError("missing everywhere")

    fake_transformers = types.SimpleNamespace(AutoTokenizer=DummyAutoTokenizer)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    with pytest.raises(RuntimeError) as excinfo:
        emo.OnnxTextEmotion(
            os.fspath(model_path),
            tokenizer_source=tmp_path,
            disable_downloads=True,
        )

    assert "Unable to load text emotion tokenizer" in str(excinfo.value)
