#!/usr/bin/env bash
# DiaRemot — setup for Codex Cloud (internet ON)
set -Eeuo pipefail

log() { printf '\n==> %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT" || die "cannot cd to repo root"

: "${PYTHON:=python}"

# Repo-local caches (good for ephemeral containers)
export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$REPO_ROOT/.cache/transformers}"
export TORCH_HOME="${TORCH_HOME:-$REPO_ROOT/.cache/torch}"

# Conservative threads for CPU-only defaults
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

pip_run() {
  # retries for flaky networks; quiet UX for CI
  $PYTHON -m pip "$@" \
    --disable-pip-version-check \
    --no-input \
    --root-user-action=ignore \
    --progress-bar off
}

log "Python: $($PYTHON -V 2>&1 || echo 'not found')"
$PYTHON - <<'PY' || die "Python >=3.11 required"
import sys; assert sys.version_info[:2] >= (3,11), sys.version
PY

log "Upgrading pip/setuptools/wheel"
for i in 1 2 3; do
  pip_run install -U pip setuptools wheel && break || sleep 2
done

log "Installing runtime dependencies (requirements.txt)"
for i in 1 2 3; do
  pip_run install -r requirements.txt && break || sleep 2
done

log "Editable install"
for i in 1 2 3; do
  pip_run install -e . && break || sleep 2
done

log "Developer tools (ruff/pytest/mypy/build)"
pip_run install -U ruff pytest mypy build || true

# FFmpeg provisioning: prefer system ffmpeg; otherwise, imageio-ffmpeg
if ! command -v ffmpeg >/dev/null 2>&1; then
  log "ffmpeg not found on PATH; provisioning via imageio-ffmpeg"
  pip_run install -U imageio-ffmpeg
  FFMPEG_BIN="$($PYTHON - <<'PY'
import imageio_ffmpeg, sys
print(imageio_ffmpeg.get_ffmpeg_exe())
PY
)"
  if [ -x "$FFMPEG_BIN" ]; then
    export PATH="$(dirname "$FFMPEG_BIN"):$PATH"
    export IMAGEIO_FFMPEG_EXE="$FFMPEG_BIN"
    log "Using bundled FFmpeg at: $FFMPEG_BIN"
  else
    printf '::warning:: Failed to locate imageio-ffmpeg binary; some decodes may fail\n'
  fi
else
  log "ffmpeg found on PATH"
fi

log "Setup complete"

# Minimal diagnostics
python - <<'PY'
import os, shutil, sys, platform
print("== Diagnostics ==")
print("python:", sys.version.split()[0])
print("platform:", platform.platform())
print("ffmpeg_on_path:", bool(shutil.which("ffmpeg")))
print("HF_HOME:", os.environ.get("HF_HOME"))
print("TRANSFORMERS_CACHE:", os.environ.get("TRANSFORMERS_CACHE"))
print("TORCH_HOME:", os.environ.get("TORCH_HOME"))
print("OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"))
PY
