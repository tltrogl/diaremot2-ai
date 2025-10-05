# Windows Model Paths - DiaRemot

**Last updated:** 2025-10-05  
**Verified against:** Actual file system + code search logic

---

## Model Directory Structure

DiaRemot auto-detects models on Windows with NO environment variables required.

### Primary Location: `D:/models/`

```
D:/models/
├── ecapa_onnx/
│   └── ecapa_tdnn.onnx                    # ECAPA speaker embeddings (ONNX)
├── faster-whisper-large-v3-turbo-ct2/     # Whisper ASR (CTranslate2)
├── silero_vad.onnx                        # Silero VAD (ONNX preferred)
└── panns/
    ├── panns_cnn14.onnx                   # PANNs SED (ONNX)
    └── audioset_labels.csv                # 527 AudioSet class labels
```

### Secondary Location: `D:/diaremot/diaremot2-1/models/`

```
D:/diaremot/diaremot2-1/models/
├── ecapa_onnx/
│   └── ecapa_tdnn.onnx                    # ECAPA speaker embeddings (backup)
├── panns/
│   ├── panns_cnn14.onnx                   # PANNs SED (backup)
│   └── audioset_labels.csv
└── bart/
    └── model_int8.onnx                    # BART intent classification (int8 quantized)
```

---

## Model Search Order (Code-Verified)

### 1. ECAPA-TDNN Embeddings
**File:** `speaker_diarization.py::_ECAPAWrapper._load()`

**Search order:**
1. `ECAPA_ONNX_PATH` env var (if set)
2. `D:/models/ecapa_onnx/ecapa_tdnn.onnx`
3. `D:/diaremot/diaremot2-1/models/ecapa_onnx/ecapa_tdnn.onnx`
4. `./models/ecapa_tdnn.onnx` (cwd fallback)

**File size:** ~6.1 MB  
**Backend:** ONNX Runtime

---

### 2. PANNs CNN14 (Sound Event Detection)
**File:** `sed_panns.py::_iter_onnx_candidates()`

**Search order:**
1. `model_dir` config parameter (if set)
2. `D:/diaremot/diaremot2-1/models/panns/`
3. `D:/models/panns/`
4. HuggingFace cache: `.cache/hf/models--qiuqiangkong--panns-tagging-onnx/`

**Files required:**
- `panns_cnn14.onnx` (118 MB)
- `audioset_labels.csv` (527 classes)

**Backend:** ONNX Runtime (preferred), `panns_inference` (PyTorch fallback)

---

### 3. Faster-Whisper ASR
**File:** `runtime_env.py::resolve_default_whisper_model()`

**Search order:**
1. `WHISPER_MODEL_PATH` env var (if set)
2. `D:/models/faster-whisper-large-v3-turbo-ct2/`
3. `~/whisper_models/faster-whisper-large-v3-turbo-ct2/`

**Model:** `tiny.en` (default), `large-v3-turbo` (optional)  
**Backend:** CTranslate2  
**Compute type:** int8 (default)

---

### 4. Silero VAD
**File:** `speaker_diarization.py::_SileroWrapper._load_onnx()`

**Search order:**
1. `SILERO_VAD_ONNX_PATH` env var (if set)
2. `D:/models/silero_vad.onnx`
3. `D:/models/silero/vad.onnx`
4. `./models/silero_vad.onnx` (cwd fallback)

**Fallback:** PyTorch TorchHub (if ONNX unavailable)  
**File size:** ~1.8 MB

---

### 5. BART Intent Classification
**File:** `emotion_analyzer.py::_resolve_intent_model_dir()`

**Search order:**
1. `affect_intent_model_dir` config parameter
2. `D:/diaremot/diaremot2-1/models/bart/model_int8.onnx`
3. HuggingFace model: `facebook/bart-large-mnli`

**Backend:** ONNX (preferred), Transformers (fallback)

---

## Environment Variables (Optional)

**NOT required for model loading** (auto-detected), but can override:

```bash
# Model path overrides (optional)
ECAPA_ONNX_PATH=D:\models\ecapa_onnx\ecapa_tdnn.onnx
WHISPER_MODEL_PATH=D:\models\faster-whisper-large-v3-turbo-ct2
SILERO_VAD_ONNX_PATH=D:\models\silero_vad.onnx

# Performance tuning (recommended)
OMP_NUM_THREADS=4
MKL_NUM_THREADS=4
NUMEXPR_MAX_THREADS=4
TOKENIZERS_PARALLELISM=false
```

**Cache directories** (auto-configured by `runtime_env.py`):
- `HF_HOME` → `.cache/hf`
- `TRANSFORMERS_CACHE` → `.cache/transformers`
- `TORCH_HOME` → `.cache/torch`

---

## Verification Commands

### Check if models exist:

```powershell
# ECAPA
Test-Path "D:\models\ecapa_onnx\ecapa_tdnn.onnx"

# PANNs
Test-Path "D:\models\panns\panns_cnn14.onnx"
Test-Path "D:\models\panns\audioset_labels.csv"

# Whisper
Test-Path "D:\models\faster-whisper-large-v3-turbo-ct2"

# Silero VAD
Test-Path "D:\models\silero_vad.onnx"

# BART (secondary location)
Test-Path "D:\diaremot\diaremot2-1\models\bart\model_int8.onnx"
```

---

**End of Windows Model Paths Reference**
