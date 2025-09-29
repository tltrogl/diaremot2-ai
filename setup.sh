#!/usr/bin/env bash
# DiaRemot bootstrap: create the virtualenv, install dependencies, stage models, and run diagnostics.
# This script mirrors the manual setup instructions in the README. Re-run it whenever dependency pins
# or model layouts change so documentation, automation, and human workflows stay in sync.

set -Eeuo pipefail

log() { printf '\n==> %s\n' "$*"; }
warn() { printf 'WARN: %s\n' "$*" >&2; }
err() { printf 'ERROR: %s\n' "$*" >&2; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
log "Repo root: $REPO_ROOT"

# ---- sanity checks ----
REQUIRED_FILES=(
  "pyproject.toml"
  "requirements.txt"
  "src/diaremot/__init__.py"
  "src/diaremot/cli.py"
  "src/diaremot/pipeline/orchestrator.py"
  "src/diaremot/pipeline/audio_preprocessing.py"
  "src/diaremot/pipeline/speaker_diarization.py"
  "src/diaremot/pipeline/transcription_module.py"
  "src/diaremot/pipeline/outputs.py"
  "src/diaremot/pipeline/config.py"
  "src/diaremot/affect/emotion_analyzer.py"
  "src/diaremot/affect/paralinguistics.py"
  "src/diaremot/affect/sed_panns.py"
  "src/diaremot/summaries/html_summary_generator.py"
  "src/diaremot/summaries/pdf_summary_generator.py"
  "src/diaremot/summaries/speakers_summary_builder.py"
  "src/diaremot/summaries/conversation_analysis.py"
  "src/diaremot/io/onnx_utils.py"
)
missing=()
for f in "${REQUIRED_FILES[@]}"; do
  [[ -f "$f" ]] || missing+=("$f")
done
if ((${#missing[@]})); then
  err "required files missing:"; printf '  - %s\n' "${missing[@]}" >&2; exit 1
fi

# ---- Python / venv ----
command -v python3 >/dev/null 2>&1 || { err "python3 not found"; exit 1; }
log "Python: $(python3 -V)"

if [[ ! -d .venv ]]; then
  log "Creating virtual environment (.venv)"
  python3 -m venv .venv
else
  log "Using existing virtual environment (.venv)"
fi
# shellcheck disable=SC1091
source .venv/bin/activate

log "Upgrading bootstrap tooling"
python -m pip install --upgrade pip setuptools wheel

log "Installing requirements.txt"
python -m pip install -r requirements.txt

log "Installing diaremot package (editable)"
python -m pip install -e .

# ---- cache normalisation ----
log "Configuring local caches"
CACHE_ROOT="$REPO_ROOT/.cache"
mkdir -p "$CACHE_ROOT/hf" "$CACHE_ROOT/torch" "$CACHE_ROOT/transformers"
export HF_HOME="$CACHE_ROOT/hf"
export HUGGINGFACE_HUB_CACHE="$CACHE_ROOT/hf"
export TRANSFORMERS_CACHE="$CACHE_ROOT/transformers"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT"
export CUDA_VISIBLE_DEVICES=""
export TORCH_DEVICE="cpu"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

# ---- models.zip staging ----
DEFAULT_MODEL_URL="https://github.com/tltrogl/diaremot2-ai/releases/download/v2.AI/models.zip"
DEFAULT_MODEL_SHA="3cc2115f4ef7cd4f9e43cfcec376bf56ea2a8213cb760ab17b27edbc2cac206c"
MODEL_URL="${MODEL_RELEASE_URL:-$DEFAULT_MODEL_URL}"
MODEL_SHA="${MODEL_RELEASE_SHA256:-$DEFAULT_MODEL_SHA}"

MODEL_DIR="${DIAREMOT_MODEL_DIR:-/opt/models}"
mkdir -p "$MODEL_DIR" 2>/dev/null || MODEL_DIR="$REPO_ROOT/models"
mkdir -p "$MODEL_DIR"
REPO_MODELS_ZIP="$REPO_ROOT/models.zip"

log "Staging models into: $MODEL_DIR"
if [[ -f "$REPO_MODELS_ZIP" ]]; then
  log "Using repo copy of models.zip"
else
  log "Downloading models from: $MODEL_URL"
  curl -L --fail --retry 5 -o "$REPO_MODELS_ZIP" "$MODEL_URL"
fi

echo "$MODEL_SHA  $REPO_MODELS_ZIP" | sha256sum -c - || { err "models.zip SHA256 mismatch"; exit 2; }
log "Unpacking models.zip"
unzip -o "$REPO_MODELS_ZIP" -d "$MODEL_DIR" >/dev/null

# Flatten accidental nesting; normalise Silero VAD path
if [[ -d "$MODEL_DIR/models" ]]; then
  cp -a "$MODEL_DIR/models/." "$MODEL_DIR/"
  rm -rf "$MODEL_DIR/models"
fi
if [[ -f "$MODEL_DIR/vad/silero_vad.onnx" && ! -f "$MODEL_DIR/silero_vad.onnx" ]]; then
  mv "$MODEL_DIR/vad/silero_vad.onnx" "$MODEL_DIR/silero_vad.onnx"
fi

log "Models available"
find "$MODEL_DIR" -maxdepth 2 -type f \( -name '*.onnx' -o -name 'model.bin' -o -name '*.json' -o -name '*.csv' \) -print | while IFS= read -r f; do printf '  %s\n' "$f"; done || true

# ---- optional sample asset ----
if command -v ffmpeg >/dev/null 2>&1; then
  if [[ ! -f data/sample.wav ]]; then
    log "Generating 10s 440Hz sample (data/sample.wav)"
    mkdir -p data
    ffmpeg -hide_banner -loglevel error \
      -f lavfi -i "sine=frequency=440:duration=10" \
      -ar 16000 -ac 1 data/sample.wav -y || warn "ffmpeg failed to create sample"
  fi
else
  warn "ffmpeg not found; skipping sample generation"
fi

# ---- import checks ----
log "Verifying core imports"
python - <<'PY'
import importlib
modules = [
    "diaremot.cli",
    "diaremot.pipeline.orchestrator",
    "diaremot.pipeline.audio_preprocessing",
    "diaremot.pipeline.speaker_diarization",
    "diaremot.pipeline.transcription_module",
    "diaremot.pipeline.outputs",
    "diaremot.pipeline.config",
    "diaremot.affect.emotion_analyzer",
    "diaremot.affect.paralinguistics",
    "diaremot.affect.sed_panns",
    "diaremot.summaries.html_summary_generator",
    "diaremot.summaries.pdf_summary_generator",
    "diaremot.summaries.speakers_summary_builder",
    "diaremot.summaries.conversation_analysis",
    "diaremot.io.onnx_utils",
]
failures = []
for module in modules:
    try:
        importlib.import_module(module)
        print(f"OK  import {module}")
    except Exception as exc:  # pragma: no cover - setup guard
        failures.append((module, exc))
if failures:
    print("\nFAILED imports:")
    for module, exc in failures:
        print(f" - {module}: {exc}")
    raise SystemExit(2)
PY

log "Running CLI diagnostics (--strict)"
python -m diaremot.cli system diagnostics --strict || warn "Diagnostics reported issues"

cat <<'MSG'
==> Setup complete.
Activate the environment with:  source .venv/bin/activate
Run the pipeline:              python -m diaremot.cli asr run --input data/sample.wav --outdir outputs/run_$RANDOM
Inspect health:                python -m diaremot.cli system diagnostics --strict
MSG
