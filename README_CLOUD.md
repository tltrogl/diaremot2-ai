# DiaRemot - Codex Cloud Quickstart (Ubuntu + apt)

IMPORTANT: Use These Pinned Versions
- Always activate the repo venv and `pip install -r requirements.txt`.
- Do NOT upgrade packages; the pipeline is validated on these exact pins:
  - onnxruntime==1.17.1
  - faster-whisper==1.1.0, ctranslate2==4.6.0
  - transformers==4.38.2, tokenizers==0.15.2
  - torch==2.4.1+cpu, torchaudio==2.4.1+cpu, torchvision==0.19.1+cpu
  - librosa==0.10.2.post1, numpy==1.24.4, scipy==1.10.1, numba==0.59.1, llvmlite==0.42.0
  - pandas==2.0.3, scikit-learn==1.3.2
  - praat-parselmouth==0.4.3, panns-inference==0.1.1

Quick check after setup:
```
python - <<'PY'
from importlib.metadata import version
assert version('onnxruntime')=='1.17.1'
assert version('faster-whisper')=='1.1.0'
assert version('ctranslate2')=='4.6.0'
assert version('transformers')=='4.38.2'
assert version('tokenizers')=='0.15.2'
print('Pins OK')
PY
```

This guide is a Codex-Cloud–specific companion to the main README. It assumes an Ubuntu-like runner with `apt` available and focuses on reproducible, CPU-only setup with ONNX-first inference.

Prereqs
- Ubuntu-like environment with `apt`
- Python 3.9–3.11
- Git access to this repo

1) System Utilities (allowed on Codex Cloud)
```
sudo apt-get update -qq
sudo apt-get install -y ffmpeg
```

2) Create and Activate the Virtualenv
```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

3) Install Pinned Python Dependencies
```
python -m pip install -r requirements.txt
```

4) Cache & Environment Defaults
```
export CACHE_ROOT="$(pwd)/.cache"
mkdir -p "$CACHE_ROOT" "$CACHE_ROOT/hf" "$CACHE_ROOT/torch" "$CACHE_ROOT/transformers"
export HF_HOME="$CACHE_ROOT/hf"
export HUGGINGFACE_HUB_CACHE="$CACHE_ROOT/hf"
export TRANSFORMERS_CACHE="$CACHE_ROOT/transformers"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT"
export DIAREMOT_MODEL_DIR="$(pwd)/.cache/models"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export NUMEXPR_MAX_THREADS=${NUMEXPR_MAX_THREADS:-4}
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=""; export TORCH_DEVICE=cpu
```

5) Stage Models (Use Your GitHub Release Asset)
- Recommended: Provide a signed `models.zip` release. Then:
```
export DIAREMOT_AUTO_DOWNLOAD=1
export DIAREMOT_MODELS_URL="https://github.com/OWNER/REPO/releases/download/TAG/models.zip"
export DIAREMOT_MODELS_SHA256="<sha256-hex>"
```
- Or place `models.zip` at repo root, or pre-populate `DIAREMOT_MODEL_DIR` with files.
- Expected files:
  - panns_cnn14.onnx, audioset_labels.csv
  - silero_vad.onnx, ecapa_tdnn.onnx
  - ser_8class.onnx, vad_model.onnx
  - roberta-base-go_emotions.onnx, bart-large-mnli.onnx

6) One-Command Setup
Use the repo script (Codex Cloud–aware). It will install Python deps, ensure env defaults, and fetch/unpack models if configured. It also bootstraps ffmpeg via imageio-ffmpeg when apt is unavailable.
```
chmod +x ./setup.sh
./setup.sh
```

7) Smoke Test
```
python -m diaremot.cli run \
  --input data/sample.wav \
  --outdir ./outputs \
  --asr-compute-type float32
```

Notes
- CPU-only: The pipeline uses `CPUExecutionProvider` for ONNXRuntime and CTranslate2 on CPU.
- If you prefer apt-installed ffmpeg (Codex Cloud): already covered in step 1. The setup script will otherwise provide a portable ffmpeg via `imageio-ffmpeg` in `./.cache/bin`.
- Verify deps quickly:
```
python -m diaremot.pipeline.audio_pipeline_core --verify_deps --strict_dependency_versions
```

Troubleshooting
- Torch `_C` import errors: ensure you used the venv created here; the code lazily imports heavy backends now.
- Librosa lazy_loader error: the code imports `librosa` module and uses `librosa.func` style, which avoids the issue. Ensure you’re on the pinned versions from `requirements.txt`.
