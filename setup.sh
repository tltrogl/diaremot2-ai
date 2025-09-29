#!/usr/bin/env bash
set -Eeuo pipefail

log() { printf '\n==> %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*\n" >&2; exit 1; }

# --- Repo + env ---
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT" || die "cannot cd to repo root"

VENV_DIR="${VENV_DIR:-.venv}"
MODEL_DIR="${DIAREMOT_MODEL_DIR:-$REPO_ROOT/models}"

# --- Versions / indexes ---
TORCH_VERSION="2.4.1"
TORCHAUDIO_VERSION="2.4.1"
TORCH_IDX="https://download.pytorch.org/whl/cpu"
PYPI_IDX="https://pypi.org/simple"

log "Repo root: $REPO_ROOT"

# --- venv ---
if [[ ! -d "$VENV_DIR" ]]; then
  log "Creating venv at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip wheel setuptools

# --- Install exact Torch CPU wheels first (from CPU index only) ---
log "Installing torch==$TORCH_VERSION+cpu and torchaudio==$TORCHAUDIO_VERSION+cpu from CPU index"
python -m pip install \
  --index-url "$TORCH_IDX" \
  --extra-index-url "$PYPI_IDX" \
  "torch==${TORCH_VERSION}+cpu" \
  "torchaudio==${TORCHAUDIO_VERSION}+cpu"

# Constraint file to prevent later steps from altering these pins
TMP_CONSTRAINT="$(mktemp)"
cat > "$TMP_CONSTRAINT" <<EOF
torch==${TORCH_VERSION}+cpu
torchaudio==${TORCHAUDIO_VERSION}+cpu
EOF

# --- Install the rest of dependencies (do NOT override indices globally) ---
if [[ -f "requirements.txt" ]]; then
  log "Installing remaining requirements with constraint on torch/torchaudio"
  python -m pip install \
    --extra-index-url "$PYPI_IDX" \
    --constraint "$TMP_CONSTRAINT" \
    -r requirements.txt
else
  log "requirements.txt not found; skipping general deps"
fi

# --- Verify exact versions ---
python - <<'PY'
import sys
bad = []
try:
    import torch
    assert torch.__version__.startswith("2.4.1"), f"torch version is {torch.__version__}"
    # Ensure CPU build (no CUDA)
    assert torch.cuda.is_available() is False, "CUDA detected; expected CPU-only"
except Exception as e:
    bad.append(f"torch check failed: {e}")

try:
    import torchaudio
    assert torchaudio.__version__.startswith("2.4.1"), f"torchaudio version is {torchaudio.__version__}"
except Exception as e:
    bad.append(f"torchaudio check failed: {e}")

if bad:
    for x in bad: print(x, file=sys.stderr)
    sys.exit(2)
print("Torch pins verified: torch 2.4.1+cpu, torchaudio 2.4.1+cpu")
PY

# --- Models staging (optional) ---
export DIAREMOT_MODEL_DIR="$MODEL_DIR"
mkdir -p "$DIAREMOT_MODEL_DIR"

# If you have a models script, call it here:
# python -m diaremot.cli models download-all || true

# --- Diagnostics / smoke (optional) ---
if command -v ffmpeg >/dev/null 2>&1; then
  log "ffmpeg found"
else
  printf 'WARN: ffmpeg not on PATH. Install it for audio processing.\n' >&2
fi

# python -m diaremot.cli system diagnostics --strict || true
# [[ -f "./data/sample.wav" ]] && python -m diaremot.cli run --audio ./data/sample.wav --tag smoke || true

log "Setup complete."
