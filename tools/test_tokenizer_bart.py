"""Regression coverage for offline BART tokenizer assets.

The legacy script lived under ``tools/`` and was executed manually to verify
that the Windows production drop of the intent classifier tokeniser could be
loaded.  Pytest attempted to collect it as a test module which broke CI when the
assets or heavy optional dependencies were absent.  This refactor keeps the
coverage while gracefully skipping in environments that do not stage the
transformer models.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Optional

import pytest

try:
    librosa = importlib.import_module("librosa")
except Exception as exc:  # pragma: no cover - import guard
    pytest.skip(f"librosa unavailable: {exc}", allow_module_level=True)

try:
    transformers = importlib.import_module("transformers")
except Exception as exc:  # pragma: no cover - import guard
    pytest.skip(f"transformers unavailable: {exc}", allow_module_level=True)


def _discover_tokenizer_root() -> Optional[Path]:
    """Return the first existing tokenizer directory from known locations."""

    candidates: tuple[Path, ...] = (
        Path(os.environ.get("DIAREMOT_INTENT_MODEL_DIR", "")),
        Path(os.environ.get("DIAREMOT_MODEL_DIR", "")) / "bart",
        Path("models") / "bart",
    )
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    return None


TOKENIZER_ROOT = _discover_tokenizer_root()
if TOKENIZER_ROOT is None:
    pytest.skip("BART tokenizer assets not available", allow_module_level=True)


def test_bart_tokenizer_loads_offline() -> None:
    """Ensure the offline tokenizer bundle can be instantiated."""

    assert librosa.__spec__ is not None  # sanity guard against stubbed modules

    AutoTokenizer = transformers.AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ROOT)
    vocab_size = getattr(tokenizer, "vocab_size", None)
    # The offline bundle can surface ``None`` for sentencepiece tokenizers.
    assert vocab_size is None or vocab_size > 0
