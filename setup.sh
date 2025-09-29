#!/usr/bin/env bash
# DiaRemot setup — deterministic CPU bootstrap
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
echo "==> Repo root: $REPO_ROOT"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: required command '$1' not found" >&2
    exit 1
  fi
}

require_cmd python3
require_cmd curl
require_cmd sha256sum

if ! command -v unzip >/dev/null 2>&1; then
  echo "==> 'unzip' not found; Python zipfile fallback will be used"
fi

# ---- sanity: required files ----
REQUIRED_FILES=(
  "pyproject.toml"
  "requirements.txt"
  "src/diaremot/__init__.py"
  "src/diaremot/pipeline/audio_pipeline_core.py"
  "src/diaremot/pipeline/orchestrator.py"
  "src/diaremot/pipeline/transcription_module.py"
  "src/diaremot/pipeline/speaker_diarization.py"
  "src/diaremot/pipeline/audio_preprocessing.py"
  "src/diaremot/affect/emotion_analyzer.py"
  "src/diaremot/affect/paralinguistics.py"
  "src/diaremot/summaries/html_summary_generator.py"
  "src/diaremot/summaries/pdf_summary_generator.py"
  "src/diaremot/summaries/speakers_summary_builder.py"
  "src/diaremot/io/onnx_utils.py"
)
missing=()
for f in "${REQUIRED_FILES[@]}"; do
  [[ -f "$f" ]] || missing+=("$f")
done
if ((${#missing[@]})); then
  echo "ERROR: required files missing:" >&2
  printf '  - %s\n' "${missing[@]}" >&2
  exit 1
fi

# ---- Python / venv ----
echo "==> Python: $(python3 -V)"

if [[ ! -d .venv ]]; then
  echo "==> Creating virtual environment (.venv)"
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

echo "==> Upgrading bootstrap tooling"
python -m pip install --upgrade pip setuptools wheel

echo "==> Installing requirements.txt"
python -m pip install -r requirements.txt

echo "==> Installing DiaRemot (editable)"
python -m pip install -e .

# ---- local caches ----
echo "==> Preparing local caches"
CACHE_ROOT="$REPO_ROOT/.cache"
mkdir -p "$CACHE_ROOT"/hf "$CACHE_ROOT"/torch "$CACHE_ROOT"/transformers
export HF_HOME="$CACHE_ROOT/hf"
export HUGGINGFACE_HUB_CACHE="$CACHE_ROOT/hf"
export TRANSFORMERS_CACHE="$CACHE_ROOT/transformers"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT"
export CUDA_VISIBLE_DEVICES=""
export TORCH_DEVICE="cpu"

# ---- models.zip download/verify ----
MODELS_URL="${MODELS_URL:-https://github.com/tltrogl/diaremot2-ai/releases/download/v2.AI/models.zip}"
MODELS_SHA256="${MODELS_SHA256:-3cc2115f4ef7cd4f9e43cfcec376bf56ea2a8213cb760ab17b27edbc2cac206c}"
REPO_MODELS_ZIP="$REPO_ROOT/models.zip"

if [[ -n "${DIAREMOT_MODEL_DIR:-}" ]]; then
  MODEL_DIR="${DIAREMOT_MODEL_DIR}"
  if ! mkdir -p "$MODEL_DIR"; then
    echo "ERROR: DIAREMOT_MODEL_DIR '$MODEL_DIR' is not writable" >&2
    exit 1
  fi
else
  MODEL_DIR="/opt/models"
  if ! mkdir -p "$MODEL_DIR" 2>/dev/null; then
    MODEL_DIR="$REPO_ROOT/models"
    mkdir -p "$MODEL_DIR"
  fi
  export DIAREMOT_MODEL_DIR="$MODEL_DIR"
fi
export DIAREMOT_MODEL_DIR="${DIAREMOT_MODEL_DIR:-$MODEL_DIR}"

echo "==> Staging models into: $DIAREMOT_MODEL_DIR"
if [[ -f "$REPO_MODELS_ZIP" ]]; then
  echo "   using repo models.zip"
else
  echo "   fetching from release: $MODELS_URL"
  curl -L --fail --retry 5 -o "$REPO_MODELS_ZIP" "$MODELS_URL"
fi

echo "$MODELS_SHA256  $REPO_MODELS_ZIP" | sha256sum -c -

if command -v unzip >/dev/null 2>&1; then
  unzip -o "$REPO_MODELS_ZIP" -d "$DIAREMOT_MODEL_DIR" >/dev/null
else
  python - "$REPO_MODELS_ZIP" "$DIAREMOT_MODEL_DIR" <<'PY'
import sys
from pathlib import Path
import zipfile

zip_path = Path(sys.argv[1])
target = Path(sys.argv[2])
with zipfile.ZipFile(zip_path) as zf:
    zf.extractall(target)
PY
fi

if [[ -d "$DIAREMOT_MODEL_DIR/models" ]]; then
  echo "==> Normalising nested models directory"
  shopt -s dotglob nullglob
  mv "$DIAREMOT_MODEL_DIR"/models/* "$DIAREMOT_MODEL_DIR"/
  shopt -u dotglob nullglob
  rmdir "$DIAREMOT_MODEL_DIR/models"
fi
if [[ -f "$DIAREMOT_MODEL_DIR/vad/silero_vad.onnx" && ! -f "$DIAREMOT_MODEL_DIR/silero_vad.onnx" ]]; then
  mv "$DIAREMOT_MODEL_DIR/vad/silero_vad.onnx" "$DIAREMOT_MODEL_DIR/silero_vad.onnx"
fi

# ---- optional sample audio ----
if command -v ffmpeg >/dev/null 2>&1; then
  if [[ ! -f data/sample.wav ]]; then
    echo "==> Generating 10s 440Hz sample (data/sample.wav)"
    mkdir -p data
    ffmpeg -hide_banner -loglevel error \
      -f lavfi -i "sine=frequency=440:duration=10" \
      -ar 16000 -ac 1 data/sample.wav -y || echo "WARN: ffmpeg failed to create sample"
  fi
else
  echo "WARN: ffmpeg not found; skipping sample generation"
fi

# ---- diagnostics ----
echo "==> Verifying core imports"
python - <<'PY'
import importlib
mods = [
  "diaremot.pipeline.audio_pipeline_core",
  "diaremot.pipeline.orchestrator",
  "diaremot.pipeline.transcription_module",
  "diaremot.pipeline.speaker_diarization",
  "diaremot.pipeline.audio_preprocessing",
  "diaremot.affect.emotion_analyzer",
  "diaremot.affect.paralinguistics",
  "diaremot.summaries.html_summary_generator",
  "diaremot.summaries.pdf_summary_generator",
  "diaremot.summaries.speakers_summary_builder",
  "diaremot.io.onnx_utils",
]
bad = []
for mod in mods:
    try:
        importlib.import_module(mod)
        print("OK  import", mod)
    except Exception as exc:
        bad.append((mod, exc))
if bad:
    print("\nFAILED imports:")
    for mod, exc in bad:
        print(" -", mod, ":", exc)
    raise SystemExit(2)
PY

echo "==> Running diagnostics"
python -m diaremot.cli system diagnostics --strict || true

echo "==> Pipeline dependency probe"
python -m diaremot.pipeline.audio_pipeline_core --verify_deps || true

echo "==> Setup complete."
echo "DIAREMOT_MODEL_DIR=$DIAREMOT_MODEL_DIR"
echo "Run: python -m diaremot.cli asr run --input data/sample.wav --outdir outputs/run_$RANDOM"
