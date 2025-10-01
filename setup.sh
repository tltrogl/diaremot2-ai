#!/usr/bin/env bash
set -Eeuo pipefail
: "${PYTHON:=python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$REPO_ROOT/.cache/transformers}"
export TORCH_HOME="${TORCH_HOME:-$REPO_ROOT/.cache/torch}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

$PYTHON -m pip install -U pip setuptools wheel
$PYTHON -m pip install -r requirements.txt
$PYTHON -m pip install -e .
$PYTHON -m pip install -q ruff pytest || true

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "::warning:: ffmpeg not found on PATH"
fi
echo "Setup complete"