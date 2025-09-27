#!/usr/bin/env bash
set -euo pipefail

echo "[setup] starting"

# --- Config ---
PROJECT_DIR="${PROJECT_DIR:-/workspace}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
REQ_TXT="${REQ_TXT:-$PROJECT_DIR/requirements.txt}"

MODEL_DIR="${MODEL_DIR:-/opt/models}"
REPO_MODELS_DIR="$PROJECT_DIR/models"
REPO_MODELS_ZIP="$PROJECT_DIR/models.zip"

# optional: URL + SHA256 to fetch models.zip if not in repo
MODEL_RELEASE_URL="${MODEL_RELEASE_URL:-}"
MODEL_RELEASE_SHA256="${MODEL_RELEASE_SHA256:-}"

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
  # your repo's requirements.txt sets the CPU torch index via --index-url
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
elif [ -n "$MODEL_RELEASE_URL" ]; then
  echo "[setup] fetching models.zip from release URL"
  curl -L --fail --retry 5 -o "$REPO_MODELS_ZIP" "$MODEL_RELEASE_URL"
  if [ -n "$MODEL_RELEASE_SHA256" ]; then
    echo "$MODEL_RELEASE_SHA256  $REPO_MODELS_ZIP" | sha256sum -c - || { echo "Bad models.zip SHA256"; exit 2; }
  fi
  unzip -o "$REPO_MODELS_ZIP" -d "$MODEL_DIR"
else
  echo "[setup] WARNING: no models shipped (models.zip or models/ missing). Set MODEL_RELEASE_URL or commit models."
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

# --- Ensure BART tokenizer assets exist (offline run) ---
# Requires setup internet if not present locally.
if [ ! -f "$MODEL_DIR/bart/tokenizer.json" ] && [ ! -f "$MODEL_DIR/bart/merges.txt" ]; then
  echo "[setup] BART tokenizer missing; fetching to $MODEL_DIR/bart"
  pip install -q transformers huggingface_hub
  python - <<'PY'
import os
from transformers import AutoTokenizer, AutoConfig
dst = os.path.join(os.environ.get("MODEL_DIR","/opt/models"), "bart")
os.makedirs(dst, exist_ok=True)
mid = "facebook/bart-large-mnli"
tok = AutoTokenizer.from_pretrained(mid, use_fast=True)
tok.save_pretrained(dst)                              # tokenizer.json or merges+vocab
AutoConfig.from_pretrained(mid).to_json_file(os.path.join(dst, "config.json"))
print("[setup] wrote BART tokenizer+config to", dst)
PY
fi

# --- CPU-only defaults ---
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="false"
echo "export DIAREMOT_MODEL_DIR=\"$MODEL_DIR\"" > /etc/profile.d/diaremot_models.sh
echo "export HF_HOME=\"$HF_HOME\""             > /etc/profile.d/diaremot_hf.sh

echo "[setup] complete"
