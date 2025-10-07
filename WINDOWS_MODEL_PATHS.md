# Windows Model Layout & Resolution (DiaRemot)

**Last updated:** 2025-10-07  
**Targets:** Windows 11 Pro / Server 2022, PowerShell 7+

---

## Canonical Directory Tree

DiaRemot now ships with a single canonical model root on Windows: `D:\models`. Place ONNX and CTranslate2 assets inside the subdirectories below. The runtime also scans `%DIAREMOT_MODEL_DIR%` and any CLI-provided overrides before falling back to these defaults.

```
D:\models\
├── asr_ct2\                    # Faster-Whisper tiny-en (CTranslate2 export)
│   ├── config.json
│   ├── model.bin
│   ├── tokenizer.json
│   └── vocabulary.json
├── diarization\
│   ├── ecapa_tdnn.onnx         # Speaker embeddings (ONNX)
│   └── silero_vad.onnx         # Optional; PyTorch Silero fallback auto-loads
├── sed_panns\
│   ├── cnn14.onnx              # PANNs CNN14 event detector
│   └── labels.csv              # AudioSet class labels
├── affect\
│   ├── ser8\
│   │   └── model.onnx          # 8-class speech emotion (ONNX)
│   └── vad_dim\
│       └── model.onnx          # Valence/Arousal/Dominance regression (ONNX)
├── intent\
│   ├── model.onnx              # facebook/bart-large-mnli (quantised/float)
│   └── tokenizer.json
└── text_emotions\
    ├── model.onnx              # SamLowe/roberta-base-go_emotions (ONNX)
    └── tokenizer.json
```

> ℹ️ The setup scripts auto-create this tree and log warnings for any missing files; the pipeline logs neutral outputs plus `manifest["issues"]` entries instead of failing hard.

---

## Model Resolution Order (Code Verified)

All stages honour **CLI flag > environment variable > canonical default** ordering.

### 1. Automatic Speech Recognition (Faster-Whisper CT2)
*File:* `src/diaremot/pipeline/runtime_env.py::resolve_default_whisper_model`

1. `--whisper-model-path` CLI flag → `WHISPER_MODEL_PATH` env override.
2. `%DIAREMOT_MODEL_DIR%\asr_ct2` (canonical) and sibling folders under the same root.
3. `~\whisper_models\tiny.en` (legacy fallback).

**Runtime defaults:** `beam_size=1`, `compute_type=int8`, `vad_filter=True`, `temperature=0.0` on CPU-only hardware.

### 2. Speaker Diarization Assets
*Files:*
- `src/diaremot/pipeline/cpu_optimized_diarizer.py::_ECAPAWrapper`
- `src/diaremot/pipeline/cpu_optimized_diarizer.py::_SileroWrapper`

**Search order:**
1. CLI flags (`--diarization-ecapa-path`, `--silero-vad-onnx-path`).
2. Environment overrides (`ECAPA_ONNX_PATH`, `SILERO_VAD_ONNX_PATH`).
3. Canonical `D:\models\diarization\ecapa_tdnn.onnx` and `D:\models\diarization\silero_vad.onnx`.
4. Repo/local fallbacks under `models/` or `.cache/models/`.

The AHC clustering distance threshold is locked to **0.15** throughout the pipeline.

### 3. Sound Event Detection (PANNs CNN14)
*File:* `src/diaremot/affect/sed_panns.py::PANNSEventTagger`

**Search order:**
1. `--sed-model-dir` CLI override (internal flag from orchestrator config).
2. `SED_MODEL_DIR` / `DIAREMOT_MODEL_DIR` environment overrides.
3. Canonical `D:\models\sed_panns\`.
4. Recursively scan sibling directories for `cnn14.onnx` + `labels.csv`.

Missing assets emit a single warning and yield empty SED artifacts while recording an issue in the manifest.

### 4. Affect: Speech Emotion (SER-8) & V/A/D Regression
*File:* `src/diaremot/affect/emotion_analyzer.py::_resolve_component_dir`

**SER-8 (audio emotion)**
1. `--affect-ser-model-dir` CLI flag.
2. `AFFECT_SER_MODEL_DIR` environment variable.
3. `D:\models\affect\ser8\` (ONNX) or the PyTorch checkpoint via `DIAREMOT_SER_MODEL_DIR`.

**Valence/Arousal/Dominance**
1. `--affect-vad-model-dir` CLI flag.
2. `AFFECT_VAD_DIM_MODEL_DIR` environment variable.
3. `D:\models\affect\vad_dim\`.

Missing models produce neutral outputs plus a manifest issue entry.

### 5. Text Intent Classification (BART-MNLI)
*File:* `src/diaremot/affect/emotion_analyzer.py::_resolve_intent_model_dir`

1. Future `--intent-model-dir` CLI flag.
2. `DIAREMOT_INTENT_MODEL_DIR` environment variable.
3. Canonical `D:\models\intent\` (requires `model.onnx` + `tokenizer.json`).
4. Hugging Face download (`facebook/bart-large-mnli`) if allowed.

### 6. Text Emotions (GoEmotions)
*File:* `src/diaremot/affect/emotion_analyzer.py::_resolve_component_dir`

1. Future `--text-emotions-model-dir` CLI flag.
2. `DIAREMOT_TEXT_EMO_MODEL_DIR` environment variable.
3. Canonical `D:\models\text_emotions\` (ONNX + tokenizer).
4. Hugging Face download (`SamLowe/roberta-base-go_emotions`) if allowed.

---

## Environment Variables & Caches

The Windows bootstrap scripts set the following defaults:

```powershell
$env:DIAREMOT_MODEL_DIR = "D:\models"
$env:HF_HOME = "D:\hf_cache"
$env:HUGGINGFACE_HUB_CACHE = $env:HF_HOME
$env:TRANSFORMERS_CACHE = "D:\hf_cache\transformers"
$env:TORCH_HOME = "D:\hf_cache\torch"
$env:HF_HUB_ENABLE_HF_TRANSFER = "1"
$env:TOKENIZERS_PARALLELISM = "false"
```

If `D:\models` or `D:\hf_cache` cannot be created (e.g., non-admin shells), the setup script falls back to `%REPO%\.cache\`. A warning is printed so you can correct permissions later.

---

## Quick Verification Snippets

```powershell
# Confirm canonical directories
"asr_ct2","diarization","sed_panns","affect","intent","text_emotions" |
  ForEach-Object { Test-Path "D:\models\$_" }

# Check critical files
Test-Path "D:\models\sed_panns\cnn14.onnx"
Test-Path "D:\models\affect\ser8\model.onnx"
Test-Path "D:\models\intent\model.onnx"
Test-Path "D:\models\text_emotions\tokenizer.json"
```

---

**End of reference.**
