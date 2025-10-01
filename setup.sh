#!/usr/bin/env bash
set -Eeuo pipefail

log(){ printf '\n==> %s\n' "$*"; }

: "${PYTHON:=python}"

# Caches & CPU threading (no repo mutation)
export HF_HOME="${HF_HOME:-$PWD/.cache/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$PWD/.cache/transformers}"
export TORCH_HOME="${TORCH_HOME:-$PWD/.cache/torch}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Pip UX
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-180}"
export PIP_PROGRESS_BAR=off

pip_run(){
  local attempt=1 delay=2
  while true; do
    if "$PYTHON" -m pip "$@" --root-user-action=ignore; then break; fi
    if [ $attempt -ge 4 ]; then
      echo "pip failed after $attempt attempts: $*" >&2; exit 1
    fi
    sleep "$delay"; attempt=$((attempt+1)); delay=$((delay*2))
  done
}

log "Python: $($PYTHON -V 2>&1 || true)"

log "Upgrade pip/setuptools/wheel"
pip_run install -U pip setuptools wheel

# Respect YOUR pins—do not rewrite requirements.txt
if [ -f requirements.txt ]; then
  # optional: install heavy wheels first to cut risk of timeouts
  grep -E '^(numpy|scipy|llvmlite|numba)=' requirements.txt || true
  pip_run install --only-binary=:all: numpy==2.3.3 scipy==1.16.2 || true
  pip_run install -r requirements.txt
fi

# (Optional) install your package in editable mode if you rely on it
if [ -f pyproject.toml ] || [ -f setup.cfg ] || [ -f setup.py ]; then
  pip_run install -e .
fi

# Provision ffmpeg without apt
if ! command -v ffmpeg >/dev/null 2>&1; then
  log "Installing userland ffmpeg"
  pip_run install imageio-ffmpeg
  FFMPEG_BIN="$("$PYTHON" - <<'PY'
import imageio_ffmpeg, os
p = imageio_ffmpeg.get_ffmpeg_exe()
print(p if os.path.exists(p) else "")
PY
)"
  if [ -n "$FFMPEG_BIN" ] && [ -x "$FFMPEG_BIN" ]; then
    export IMAGEIO_FFMPEG_EXE="$FFMPEG_BIN"
    export PATH="$(dirname "$FFMPEG_BIN"):$PATH"
    log "ffmpeg ready at $FFMPEG_BIN"
  else
    echo "::warning:: imageio-ffmpeg did not expose a binary; some decodes may fail"
  fi
fi

# Final diagnostics (read-only)
"$PYTHON" - <<'PY'
import os, shutil, sys, platform
print("python:", sys.version.split()[0])
print("platform:", platform.platform())
print("ffmpeg_on_path:", bool(shutil.which("ffmpeg")))
print("HF_HOME:", os.environ.get("HF_HOME"))
print("TRANSFORMERS_CACHE:", os.environ.get("TRANSFORMERS_CACHE"))
print("TORCH_HOME:", os.environ.get("TORCH_HOME"))
print("OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"))
PY

log "Setup complete"
