#!/usr/bin/env bash
set -euo pipefail
echo "[info] Environment bootstrap (POSIX)"
command -v python3 >/dev/null 2>&1 || { echo "[ERR] python3 required"; exit 1; }
if ! command -v ffmpeg >/dev/null 2>&1; then echo "[WARN] ffmpeg not found on PATH"; fi

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
pip install -e .

ruff format .
ruff check --fix .
pytest -q || true
echo "[done] Setup complete."
