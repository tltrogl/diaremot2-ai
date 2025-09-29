#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_ROOT}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[maint] venv missing at $VENV_DIR" >&2
  exit 1
fi

# shellcheck disable=SC1091
. "$VENV_DIR/bin/activate"

python - <<'PY'
import json
import os
from pathlib import Path
import sys

model_root = Path(os.environ.get("DIAREMOT_MODEL_DIR", "/opt/models")).expanduser()
need = {
    "silero_vad": model_root / "silero_vad.onnx",
    "ecapa_tdnn": model_root / "ecapa_onnx" / "ecapa_tdnn.onnx",
    "panns_model": model_root / "panns" / "model.onnx",
    "panns_labels": model_root / "panns" / "class_labels_indices.csv",
    "goemotions": model_root / "goemotions-onnx" / "model.onnx",
    "ser8_int8": model_root / "ser8-onnx" / "model.int8.onnx",
    "ser8_fp32": model_root / "ser8-onnx" / "model.onnx",
    "fw_tiny_en": model_root / "faster-whisper-tiny.en" / "model.bin",
    "bart_onnx": model_root / "bart" / "model_uint8.onnx",
    "bart_tok_json": model_root / "bart" / "tokenizer.json",
    "bart_merges": model_root / "bart" / "merges.txt",
    "bart_vocab": model_root / "bart" / "vocab.json",
    "bart_cfg": model_root / "bart" / "config.json",
}
missing: list[tuple[str, str]] = []

for key, path in need.items():
    exists = path.exists()
    if key in {"ser8_int8", "ser8_fp32"}:
        status = "OK" if exists else "missing"
    else:
        if not exists:
            missing.append((key, str(path)))
        status = "OK" if exists else "MISSING"
    print(f"{key:12} -> {status:8} :: {path}")

if not (need["ser8_int8"].exists() or need["ser8_fp32"].exists()):
    missing.append(("ser8_model", str(need["ser8_fp32"])))

bart_tokenizer_ok = need["bart_tok_json"].exists() or (
    need["bart_merges"].exists() and need["bart_vocab"].exists()
)
if not bart_tokenizer_ok:
    missing.append(("bart_tokenizer", str(need["bart_tok_json"])))

if missing:
    print("\n[maint] Missing required files:")
    for key, path in missing:
        print(f" - {key}: {path}")
    sys.exit(2)

import importlib
for mod in ("diaremot", "diaremot.cli"):
    try:
        importlib.import_module(mod)
        print("[maint] import OK:", mod)
    except Exception as exc:
        print("[maint] import FAIL:", mod, "→", exc)
        sys.exit(3)

print("[maint] Running diagnostics --strict")
from diaremot.cli import core_diagnostics
result = core_diagnostics(require_versions=True)
print(json.dumps(result, indent=2))

if not result.get("ok", True):
    print("[maint] diagnostics FAILED", file=sys.stderr)
    sys.exit(4)

print("[maint] OK")
PY
