"""
Bootstrap for running this repo reliably without extra flags.

What this does when you run `python -m diaremot.pipeline.cli_entry` from the
repo root:
- Ensures the local `src` package is on sys.path before site-packages, so the
  repo code is used consistently.
- Forces offline Hugging Face behavior (no unintended downloads).
- Forces slow tokenizers (avoids fragile fast `tokenizer.json` parsing) so
  local vocab/merges pairs are used with ONNX.
- Points DiaRemot to your known local model root if present on Windows.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists():
    src_str = str(SRC)
    if src_str not in sys.path:
        # Prepend so local code wins over any installed package
        sys.path.insert(0, src_str)

# Prefer offline/local behaviour by default; can be overridden by user env
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Ensure slow tokenizers so local vocab/merges are used (more robust for ONNX)
os.environ.setdefault("TRANSFORMERS_USE_FAST_TOKENIZERS", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# If a common local model root exists (Windows path from your setup), set it.
try:
    default_models = Path(r"D:\\models")
    if default_models.exists():
        os.environ.setdefault("DIAREMOT_MODEL_DIR", str(default_models))
        os.environ.setdefault("HF_HOME", r"D:\\hf_cache")
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", r"D:\\hf_cache")
        os.environ.setdefault("TRANSFORMERS_CACHE", r"D:\\hf_cache\\transformers")
        os.environ.setdefault("TORCH_HOME", r"D:\\hf_cache\\torch")
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        # Convenience explicit paths if present
        goem = default_models / "text_emotions"
        bart = default_models / "intent"
        ser8 = default_models / "affect" / "ser8"
        vad_dim = default_models / "affect" / "vad_dim"
        if goem.exists():
            os.environ.setdefault("DIAREMOT_TEXT_MODEL_DIR", str(goem))
        if bart.exists():
            os.environ.setdefault("DIAREMOT_INTENT_MODEL_DIR", str(bart))
        if ser8.exists():
            os.environ.setdefault("AFFECT_SER_MODEL_DIR", str(ser8))
        if vad_dim.exists():
            os.environ.setdefault("AFFECT_VAD_DIM_MODEL_DIR", str(vad_dim))
except Exception:
    pass

_numpy_spec = importlib.util.find_spec("numpy")
if _numpy_spec is not None:
    importlib.import_module("numpy")
