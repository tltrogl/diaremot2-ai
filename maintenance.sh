#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$REPO_ROOT}"
export PROJECT_DIR
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[maint] venv missing at $VENV_DIR" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"

python - <<'PY'
import json
import os
from pathlib import Path
from typing import Dict

model_dir = Path(os.environ.get('DIAREMOT_MODEL_DIR', '/opt/models')).expanduser()
if not model_dir.exists():
    fallback = Path(os.environ.get('PROJECT_DIR', '')) / 'models'
    model_dir = fallback.expanduser() if fallback else Path('models').resolve()
paths: Dict[str, Path] = {
    "silero_vad": model_dir / "silero_vad.onnx",
    "ecapa_tdnn": model_dir / "ecapa_onnx" / "ecapa_tdnn.onnx",
    "panns_model": model_dir / "panns" / "model.onnx",
    "panns_labels": model_dir / "panns" / "class_labels_indices.csv",
    "goemotions": model_dir / "goemotions-onnx" / "model.onnx",
    "ser8_int8": model_dir / "ser8-onnx" / "model.int8.onnx",
    "ser8_fp32": model_dir / "ser8-onnx" / "model.onnx",
    "fw_tiny_en": model_dir / "faster-whisper-tiny.en" / "model.bin",
    "bart_onnx": model_dir / "bart" / "model_uint8.onnx",
    "bart_tok_json": model_dir / "bart" / "tokenizer.json",
    "bart_merges": model_dir / "bart" / "merges.txt",
    "bart_vocab": model_dir / "bart" / "vocab.json",
    "bart_cfg": model_dir / "bart" / "config.json",
}
missing = []
for key, path in paths.items():
    exists = path.exists()
    if key in {"ser8_int8", "ser8_fp32"}:
        print(f"{key:12} -> {'OK' if exists else 'MISSING'} :: {path}")
        continue
    if not exists:
        missing.append((key, str(path)))
    print(f"{key:12} -> {'OK' if exists else 'MISSING'} :: {path}")

if not (paths["ser8_int8"].exists() or paths["ser8_fp32"].exists()):
    missing.append(("ser8_model", str(paths["ser8_fp32"])) )

bart_tokeniser_ok = paths["bart_tok_json"].exists() or (
    paths["bart_merges"].exists() and paths["bart_vocab"].exists()
)
if not bart_tokeniser_ok:
    missing.append(("bart_tokenizer", str(paths["bart_tok_json"])) )

if missing:
    print("\n[maint] Missing required files:")
    for key, path in missing:
        print(f" - {key}: {path}")
    raise SystemExit(2)

import importlib
for mod in ("diaremot", "diaremot.cli"):
    try:
        importlib.import_module(mod)
        print(f"[maint] import OK: {mod}")
    except Exception as exc:
        print(f"[maint] import FAIL: {mod} → {exc}")
        raise SystemExit(3)

from diaremot.cli import core_diagnostics

result = core_diagnostics(require_versions=True)
print("[maint] diagnostics:")
print(json.dumps(result, indent=2))
if not result.get("ok", False):
    issues = result.get("issues") or result.get("summary")
    print(f"[maint] diagnostics reported issues: {issues}")
    raise SystemExit(4)

print("[maint] OK")
PY
