#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/workspace}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
. "$VENV_DIR/bin/activate" || { echo "[maint] venv missing"; exit 1; }

python - <<'PY'
from pathlib import Path, sys
need = {
  "silero_vad": Path("/opt/models") / "silero_vad.onnx",
  "ecapa_tdnn": Path("/opt/models") / "ecapa_onnx" / "ecapa_tdnn.onnx",
  "panns_model": Path("/opt/models") / "panns" / "model.onnx",
  "panns_labels": Path("/opt/models") / "panns" / "class_labels_indices.csv",
  "goemotions": Path("/opt/models") / "goemotions-onnx" / "model.onnx",
  "ser8_int8": Path("/opt/models") / "ser8-onnx" / "model.int8.onnx",
  "ser8_fp32": Path("/opt/models") / "ser8-onnx" / "model.onnx",
  "fw_tiny_en": Path("/opt/models") / "faster-whisper-tiny.en" / "model.bin",
  "bart_onnx": Path("/opt/models") / "bart" / "model_uint8.onnx",
  "bart_tok_json": Path("/opt/models") / "bart" / "tokenizer.json",
  "bart_merges": Path("/opt/models") / "bart" / "merges.txt",
  "bart_vocab": Path("/opt/models") / "bart" / "vocab.json",
  "bart_cfg": Path("/opt/models") / "bart" / "config.json",
}
missing = []
for k,p in need.items():
    ok = p.exists()
    if k in ("ser8_int8","ser8_fp32"):
        pass
    else:
        if not ok: missing.append((k,str(p)))
    print(f"{k:12} -> {'OK' if ok else 'MISSING'} :: {p}")

if not (need["ser8_int8"].exists() or need["ser8_fp32"].exists()):
    missing.append(("ser8_model", str(need["ser8_fp32"])))

bart_tok_ok = need["bart_tok_json"].exists() or (need["bart_merges"].exists() and need["bart_vocab"].exists())
if not bart_tok_ok:
    missing.append(("bart_tokenizer", str(need["bart_tok_json"])))

if missing:
    print("\n[maint] Missing required files:")
    for k,p in missing: print(" -", k, ":", p)
    sys.exit(2)

import importlib
for mod in ("diaremot", "diaremot.cli"):
    try:
        importlib.import_module(mod)
        print("[maint] import OK:", mod)
    except Exception as e:
        print("[maint] import FAIL:", mod, "→", e)
        sys.exit(3)
print("[maint] OK")
PY
