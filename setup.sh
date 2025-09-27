#!/usr/bin/env bash
set -euo pipefail

echo "[setup] starting (zero-touch)"

# --- Config ---
PROJECT_DIR="${PROJECT_DIR:-/workspace}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
REQ_TXT="${REQ_TXT:-$PROJECT_DIR/requirements.txt}"

MODEL_DIR="${MODEL_DIR:-/opt/models}"
REPO_MODELS_DIR="$PROJECT_DIR/models"
REPO_MODELS_ZIP="$PROJECT_DIR/models.zip"

# Hard defaults: your public GitHub Release asset
MODEL_RELEASE_URL="https://github.com/tltrogl/diaremot2-ai/releases/download/v2.AI/models.zip"
MODEL_RELEASE_SHA256="3cc2115f4ef7cd4f9e43cfcec376bf56ea2a8213cb760ab17b27edbc2cac206c"

export HF_HOME="${HF_HOME:-/opt/cache/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/opt/cache/pip}"

# --- System deps ---
apt-get update -y
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends   python3-venv python3-dev build-essential   ffmpeg sox libsndfile1   git git-lfs curl ca-certificates unzip rsync   && apt-get clean && rm -rf /var/lib/apt/lists/*

# --- Python / venv ---
python3 -m venv "$VENV_DIR"
. "$VENV_DIR/bin/activate"
python -m pip install -U pip wheel setuptools

# --- Install deps ---
if [ -f "$REQ_TXT" ]; then
  echo "[setup] installing from requirements.txt"
  pip install --no-cache-dir -r "$REQ_TXT"
else
  echo "[setup] editable install from pyproject"
  pip install --no-cache-dir -e "$PROJECT_DIR"
fi

# --- Normalize line endings for scripts from Windows ---
for f in "$PROJECT_DIR"/setup.sh "$PROJECT_DIR"/maintenance.sh; do
  [ -f "$f" ] && sed -i 's/\r$//' "$f" || true
done

# --- Stage models into /opt/models ---
mkdir -p "$MODEL_DIR"

if [ -f "$REPO_MODELS_ZIP" ]; then
  echo "[setup] unpacking repo models.zip → $MODEL_DIR"
  unzip -o "$REPO_MODELS_ZIP" -d "$MODEL_DIR"
elif [ -d "$REPO_MODELS_DIR" ]; then
  echo "[setup] copying repo models/ → $MODEL_DIR"
  rsync -a "$REPO_MODELS_DIR"/ "$MODEL_DIR"/
else
  echo "[setup] fetching models.zip from release"
  echo "[setup] URL: $MODEL_RELEASE_URL"
  curl -L --fail --retry 5 -o "$REPO_MODELS_ZIP" "$MODEL_RELEASE_URL"
  echo "$MODEL_RELEASE_SHA256  $REPO_MODELS_ZIP" | sha256sum -c - || { echo "[setup] Bad models.zip SHA256"; exit 2; }
  unzip -o "$REPO_MODELS_ZIP" -d "$MODEL_DIR"
fi

# Flatten accidental models/models nesting
if [ -d "$MODEL_DIR/models" ]; then
  rsync -a "$MODEL_DIR/models/" "$MODEL_DIR/" && rm -rf "$MODEL_DIR/models"
fi
# Normalize VAD layout
if [ -f "$MODEL_DIR/vad/silero_vad.onnx" ] && [ ! -f "$MODEL_DIR/silero_vad.onnx" ]; then
  mv "$MODEL_DIR/vad/silero_vad.onnx" "$MODEL_DIR/silero_vad.onnx"
fi
# Alias ECAPA alternates if canonical missing
if [ ! -f "$MODEL_DIR/ecapa_onnx/ecapa_tdnn.onnx" ]; then
  for c in voxceleb_ECAPA512.onnx voxceleb_ECAPA512_LM.onnx; do
    if [ -f "$MODEL_DIR/ecapa_onnx/$c" ]; then
      cp "$MODEL_DIR/ecapa_onnx/$c" "$MODEL_DIR/ecapa_onnx/ecapa_tdnn.onnx"
      echo "[setup] aliased $c -> ecapa_tdnn.onnx"
      break
    fi
  done
fi
# Alias PANNs model2 → model.onnx if needed
if [ ! -f "$MODEL_DIR/panns/model.onnx" ] && [ -f "$MODEL_DIR/panns/model2.onnx" ]; then
  cp "$MODEL_DIR/panns/model2.onnx" "$MODEL_DIR/panns/model.onnx"
  echo "[setup] aliased panns/model2.onnx -> model.onnx"
fi

# --- BART tokenizer check (won't run if included in models.zip) ---
if [ ! -f "$MODEL_DIR/bart/tokenizer.json" ] && [ ! -f "$MODEL_DIR/bart/merges.txt" ]; then
  echo "[setup] WARNING: BART tokenizer missing in models.zip; pipeline may download later if internet is allowed."
fi

# --- CPU-only defaults ---
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="false"
echo "export DIAREMOT_MODEL_DIR=\"$MODEL_DIR\"" > /etc/profile.d/diaremot_models.sh
echo "export HF_HOME=\"$HF_HOME\""             > /etc/profile.d/diaremot_hf.sh

echo "[setup] complete"
