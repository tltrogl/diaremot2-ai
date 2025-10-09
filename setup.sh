#!/usr/bin/env bash
set -euo pipefail

# Detect package manager and ensure ffmpeg + python build tools
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y ffmpeg build-essential python3-dev
elif command -v apk >/dev/null 2>&1; then
  sudo apk add --no-cache ffmpeg build-base python3-dev
elif command -v yum >/dev/null 2>&1; then
  sudo yum install -y epel-release
  sudo yum install -y ffmpeg ffmpeg-devel gcc gcc-c++ python3-devel
fi

# Prefer uv if available, else pip
if command -v uv >/dev/null 2>&1; then
  PKG="uv pip"
else
  PKG="pip"
fi

python -m venv .ai || true
. ./.ai/bin/activate

# Install Torch CPU FIRST (avoid index flags in requirements)
$PKG install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1+cpu

# Install the rest
$PKG install -r requirements.txt

# Editable install
$PKG install -e .

# Smoke checks
python - << 'PY'
import importlib, sys
mods = ["ctranslate2","onnxruntime","librosa","transformers","sklearn","numpy"]
missing = [m for m in mods if not importlib.util.find_spec(m)]
assert not missing, f"Missing modules: {missing}"
print("Environment OK")
PY
