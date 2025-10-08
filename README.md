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

DiaRemot now standardises on canonical cache/model locations. The runtime and setup scripts attempt these paths first, then fall back to repo-local `.cache/` if they are not writable.

| Purpose | Environment variable | Windows default | Linux / GCP default |
|---------|---------------------|-----------------|---------------------|
| Model root | `DIAREMOT_MODEL_DIR` | `D:\models` | `/srv/models` |
| Hugging Face home & hub cache | `HF_HOME`, `HUGGINGFACE_HUB_CACHE` | `D:\hf_cache` | `/srv/.cache/hf` |
| Transformers cache | `TRANSFORMERS_CACHE` | `D:\hf_cache\transformers` | `/srv/.cache/transformers` |
| Torch cache | `TORCH_HOME` | `D:\hf_cache\torch` | `/srv/.cache/torch` |
| Enable HF Transfer | `HF_HUB_ENABLE_HF_TRANSFER` | `1` | `1` |
| Disable tokenizer threads | `TOKENIZERS_PARALLELISM` | `false` | `false` |

**Windows PowerShell**
```powershell
$env:DIAREMOT_MODEL_DIR = "D:\models"
$env:HF_HOME = "D:\hf_cache"
$env:HUGGINGFACE_HUB_CACHE = $env:HF_HOME
$env:TRANSFORMERS_CACHE = "D:\hf_cache\transformers"
$env:TORCH_HOME = "D:\hf_cache\torch"
$env:HF_HUB_ENABLE_HF_TRANSFER = "1"
$env:TOKENIZERS_PARALLELISM = "false"
$env:OMP_NUM_THREADS = ${env:OMP_NUM_THREADS} ?: "4"
$env:MKL_NUM_THREADS = ${env:MKL_NUM_THREADS} ?: "4"
$env:NUMEXPR_MAX_THREADS = ${env:NUMEXPR_MAX_THREADS} ?: "4"
```

**Linux / GCP (bash)**
```bash
export DIAREMOT_MODEL_DIR=${DIAREMOT_MODEL_DIR:-/srv/models}
export HF_HOME=${HF_HOME:-/srv/.cache/hf}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$HF_HOME}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/srv/.cache/transformers}
export TORCH_HOME=${TORCH_HOME:-/srv/.cache/torch}
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export NUMEXPR_MAX_THREADS=${NUMEXPR_MAX_THREADS:-4}
export CUDA_VISIBLE_DEVICES=""; export TORCH_DEVICE=cpu
```

> ℹ️ `/srv` paths are preferred for production VMs. If they are not writable in your shell, DiaRemot automatically reverts to `./.cache/` underneath the repository.

5) Stage Models (Use Your GitHub Release Asset)
- Recommended: Provide a signed `models.zip` release. Then:
```
export DIAREMOT_AUTO_DOWNLOAD=1
export DIAREMOT_MODELS_URL="https://github.com/OWNER/REPO/releases/download/TAG/models.zip"
export DIAREMOT_MODELS_SHA256="<sha256-hex>"
```
- Or place `models.zip` at repo root, or pre-populate `DIAREMOT_MODEL_DIR` with the canonical layout below.
- Expected layout (Windows: `D:\models`, Linux/GCP: `/srv/models`):

```
models/
├── asr_ct2/
│   ├── config.json
│   ├── model.bin
│   ├── tokenizer.json
│   └── vocabulary.json
├── diarization/
│   ├── ecapa_tdnn.onnx
│   └── silero_vad.onnx           # optional; PyTorch Silero fallback auto-loads
├── sed_panns/
│   ├── cnn14.onnx
│   └── labels.csv
├── affect/
│   ├── ser8/
│   │   └── model.onnx
│   └── vad_dim/
│       └── model.onnx
├── intent/
│   ├── model.onnx                # facebook/bart-large-mnli (ONNX)
│   └── tokenizer.json
└── text_emotions/
    ├── model.onnx                # SamLowe/roberta-base-go_emotions (ONNX)
    └── tokenizer.json
```

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

## Adaptive VAD Overrides (Orchestrator vs CLI)

The orchestrator tightens diarization defaults when the CLI does **not** specify overrides. This reduces micro-segments in noisy recordings but can be relaxed via flags:

| Parameter | CLI default | Orchestrator auto-tune | Override flag |
|-----------|-------------|------------------------|---------------|
| `vad_threshold` | 0.30 | **0.35** | `--vad-threshold` |
| `vad_min_speech_sec` | 0.80 s | **0.80 s** (unchanged) | `--vad-min-speech-sec` |
| `vad_min_silence_sec` | 0.80 s | **0.80 s** (unchanged) | `--vad-min-silence-sec` |
| `vad_speech_pad_sec` | 0.20 s | **0.10 s** | `--vad-speech-pad-sec` |

To keep the original CLI defaults, supply all four flags explicitly:

```bash
python -m diaremot.cli run --input data/sample.wav --outdir outputs/ \
  --vad-threshold 0.30 \
  --vad-min-speech-sec 0.80 \
  --vad-min-silence-sec 0.80 \
  --vad-speech-pad-sec 0.20
```

All values verified in `src/diaremot/pipeline/orchestrator.py::_init_components` (strict overrides applied only when a value is absent from the merged config).
Troubleshooting
- Torch `_C` import errors: ensure you used the venv created here; the code lazily imports heavy backends now.
- Librosa lazy_loader error: the code imports `librosa` module and uses `librosa.func` style, which avoids the issue. Ensure you’re on the pinned versions from `requirements.txt`.
