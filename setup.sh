#!/usr/bin/env bash
# DiaRemot Codex setup — zero-touch (uses your v2.AI release)
# - keeps your old flow: venv + reqs + local caches + import checks
# - adds: models.zip download+verify+unpack and path normalization

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
echo "==> Repo root: $REPO_ROOT"

# ---- sanity: required files (same as your prior script) ----
REQUIRED_FILES=(
  "pyproject.toml"
  "requirements.txt"
  "src/diaremot/__init__.py"
  "src/diaremot/pipeline/audio_pipeline_core.py"
  "src/diaremot/pipeline/audio_preprocessing.py"
  "src/diaremot/pipeline/speaker_diarization.py"
  "src/diaremot/pipeline/transcription_module.py"
  "src/diaremot/affect/emotion_analyzer.py"
  "src/diaremot/affect/paralinguistics.py"
  "src/diaremot/summaries/html_summary_generator.py"
  "src/diaremot/summaries/pdf_summary_generator.py"
  "src/diaremot/summaries/speakers_summary_builder.py"
  "src/diaremot/io/onnx_utils.py"
)
missing=(); for f in "${REQUIRED_FILES[@]}"; do [[ -f "$f" ]] || missing+=("$f"); done
if ((${#missing[@]})); then
  echo "ERROR: required files missing:" >&2; printf '  - %s\n' "${missing[@]}" >&2; exit 1
fi

# ---- Python / venv ----
command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 not found"; exit 1; }
echo "==> Python: $(python3 -V)"

if command -v apt-get >/dev/null 2>&1; then
  if [[ "$(id -u)" == "0" ]]; then
    echo "==> Installing system Tk bindings (python3-tk)"
    apt-get update -qq
    apt-get install -y python3-tk >/dev/null
  else
    echo "INFO: install python3-tk via 'sudo apt-get install -y python3-tk' for the GUI"
  fi
fi

echo "==> Creating virtual environment (.venv)"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Upgrading bootstrap tooling"
python -m pip install --upgrade pip setuptools wheel

echo "==> Installing requirements.txt"
python -m pip install -r requirements.txt

# ---- local caches (keep behavior) ----
echo "==> Preparing local caches"
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

# ---- models.zip: download/verify/unpack (zero-touch) ----
MODEL_DIR="/opt/models"
mkdir -p "$MODEL_DIR" 2>/dev/null || MODEL_DIR="$REPO_ROOT/models"  # fallback if /opt not writable
REPO_MODELS_ZIP="$REPO_ROOT/models.zip"

REL_URL="https://github.com/tltrogl/diaremot2-ai/releases/download/v2.AI/models.zip"
REL_SHA="3cc2115f4ef7cd4f9e43cfcec376bf56ea2a8213cb760ab17b27edbc2cac206c"

echo "==> Staging models into: $MODEL_DIR"
if [[ -f "$REPO_MODELS_ZIP" ]]; then
  echo "   using repo models.zip"
else
  echo "   fetching from release: $REL_URL"
  curl -L --fail --retry 5 -o "$REPO_MODELS_ZIP" "$REL_URL"
fi

echo "$REL_SHA  $REPO_MODELS_ZIP" | sha256sum -c - || { echo "ERROR: models.zip SHA256 mismatch"; exit 2; }
echo "==> Unzipping models.zip → $MODEL_DIR"
unzip -o "$REPO_MODELS_ZIP" -d "$MODEL_DIR" >/dev/null

# flatten accidental nesting; normalize VAD path
if [[ -d "$MODEL_DIR/models" ]]; then
  rsync -a "$MODEL_DIR/models/" "$MODEL_DIR/"; rm -rf "$MODEL_DIR/models"
fi
if [[ -f "$MODEL_DIR/vad/silero_vad.onnx" && ! -f "$MODEL_DIR/silero_vad.onnx" ]]; then
  mv "$MODEL_DIR/vad/silero_vad.onnx" "$MODEL_DIR/silero_vad.onnx"
fi

# ---- optional sample (kept from your flow) ----
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

# ---- import checks (kept from your flow) ----
echo "==> Verifying core imports"
python - <<'PY'
import importlib
mods = [
  "diaremot.pipeline.audio_pipeline_core",
  "diaremot.pipeline.audio_preprocessing",
  "diaremot.pipeline.speaker_diarization",
  "diaremot.pipeline.transcription_module",
  "diaremot.affect.emotion_analyzer",
  "diaremot.affect.paralinguistics",
  "diaremot.summaries.html_summary_generator",
  "diaremot.summaries.pdf_summary_generator",
  "diaremot.summaries.speakers_summary_builder",
  "diaremot.io.onnx_utils",
]
bad=[]
for m in mods:
  try: importlib.import_module(m); print("OK  import", m)
  except Exception as e: bad.append((m,e))
if bad:
  print("\nFAILED imports:"); [print(" -",m,":",e) for m,e in bad]; raise SystemExit(2)
PY

echo "==> Pipeline dep check"
python -m diaremot.pipeline.audio_pipeline_core --verify_deps || true

cat <<'MSG'
==> Setup complete (zero-touch).
The agent can now run:
  python -m diaremot.cli run --input data/sample.wav --out outputs/run_$RANDOM
MSG
