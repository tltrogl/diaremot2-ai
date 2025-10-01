#!/usr/bin/env bash
set -Eeuo pipefail
: "${PYTHON:=python}"

if command -v ruff >/dev/null 2>&1; then
  ruff format .
  ruff check --fix .
fi
if command -v pytest >/dev/null 2>&1; then
  pytest -q || true
fi
$PYTHON -m build || true

python - <<'PY'
import json, os, sys, platform, shutil
print(json.dumps({
  "python_version": sys.version.split()[0],
  "platform": platform.platform(),
  "ffmpeg_on_path": bool(shutil.which("ffmpeg")),
  "ruff_on_path": bool(shutil.which("ruff")),
  "pytest_on_path": bool(shutil.which("pytest")),
  "hf_home": os.environ.get("HF_HOME"),
  "transformers_cache": os.environ.get("TRANSFORMERS_CACHE"),
  "torch_home": os.environ.get("TORCH_HOME"),
}, indent=2))
PY