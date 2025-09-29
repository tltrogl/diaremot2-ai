#!/usr/bin/env bash
set -Eeuo pipefail

log() { printf '\n==> %s\n' "$*"; }
err() { printf 'ERROR: %s\n' "$*" >&2; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT" || exit 1
log "Repo root: $REPO_ROOT"

VENV_DIR="${VENV_DIR:-.venv}"
MODEL_DIR="${DIAREMOT_MODEL_DIR:-$REPO_ROOT/models}"

if [[ ! -d "$VENV_DIR" ]]; then
  log "Creating venv at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

if ! command -v ffmpeg >/dev/null 2>&1; then
  printf 'WARN: ffmpeg not on PATH. Install it before running audio processing.\n' >&2
fi

TORCH_IDX="https://download.pytorch.org/whl/cpu"
EXTRA_IDX="https://pypi.org/simple"
python -m pip install --upgrade pip
python -m pip install --index-url "$TORCH_IDX" --extra-index-url "$EXTRA_IDX" -r requirements.txt

export DIAREMOT_MODEL_DIR="$MODEL_DIR"
mkdir -p "$DIAREMOT_MODEL_DIR"
log "Staging models into $DIAREMOT_MODEL_DIR"
python -m diaremot.cli models download-all

python -m diaremot.cli system diagnostics --strict
[[ -f "./data/sample.wav" ]] && python -m diaremot.cli run --audio ./data/sample.wav --tag smoke

log "Setup complete."
