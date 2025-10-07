# DiaRemot — CPU-Only Speech Intelligence Pipeline

**Process long-form audio (1-3 hours) on CPU with comprehensive analysis**

---

## Quick Start

```bash
# Install
pip install -r requirements.txt
pip install -e .

# Run
python -m diaremot.cli run --input audio.wav --outdir outputs/

# With options
python -m diaremot.cli run -i audio.wav -o outputs/ \
  --asr-compute-type int8 \
  --vad-threshold 0.35
```

---

## What It Does

**Input:** Long-form audio (conversations, meetings, interviews)  
**Output:** 39-column CSV with per-segment analysis:

- **Diarization** with persistent speaker names
- **Transcription** with ASR confidence
- **Affect** (Valence/Arousal/Dominance + emotions + intent)
- **Voice Quality** (jitter, shimmer, HNR, CPPS)
- **Prosody** (WPM, pauses, pitch, loudness)
- **Sound Events** (music, keyboard, door, TV, etc.)
- **Turn-Taking** (overlaps, interruptions)

Plus: HTML/PDF reports, speaker summaries, timeline, events

---

## Architecture

### ONNX-Preferred with PyTorch Fallback

**Primary:** ONNXRuntime + CTranslate2 (2-5x faster)  
**Fallback:** PyTorch CPU (graceful degradation)

**Why ONNX?** Faster inference, lower memory, smaller deployment  
**Why keep PyTorch?** Works when ONNX unavailable, easier dev/test

---

## Pipeline (11 Stages)

1. **dependency_check** — Validate runtime
2. **preprocess** — Normalize, denoise, auto-chunk
3. **background_sed** — Sound events (PANNs CNN14)
4. **diarize** — Speakers (Silero VAD + ECAPA + AHC)
5. **transcribe** — ASR (faster-whisper tiny.en)
6. **paralinguistics** — Voice quality (Praat) + prosody
7. **affect_and_assemble** — Emotion + intent + assembly
8. **overlap_interruptions** — Turn-taking analysis
9. **conversation_analysis** — Flow metrics
10. **speaker_rollups** — Per-speaker summaries
11. **outputs** — Write CSV/JSON/HTML/PDF

**Note:** No separate auto_tune stage (happens inline in orchestrator)

---

## Model Assets

### Where Models Live

**Auto-discovery search order:**

**Windows:**
1. `DIAREMOT_MODEL_DIR` env var
2. `D:/models/`
3. `D:/diaremot/diaremot2-1/models/`
4. `./models/`
5. `$HOME/models/`

**Unix:**
1. `DIAREMOT_MODEL_DIR`
2. `/models/`
3. `./models/`
4. `$HOME/models/`

### ONNX Models (Preferred)

```
models/
├── panns_cnn14.onnx              # SED (118 MB)
├── audioset_labels.csv            # 527 labels
├── silero_vad.onnx                # VAD (1.8 MB)
├── ecapa_tdnn.onnx                # Embeddings (6.1 MB)
├── ser_8class.onnx                # Audio emotion
├── vad_model.onnx                 # V/A/D
├── roberta-base-go_emotions.onnx  # Text emotion (~500 MB)
└── bart-large-mnli.onnx           # Intent (~1.6 GB)
```

### PyTorch Fallbacks

Auto-download to HF cache when ONNX missing:
- Silero VAD (TorchHub)
- PANNs SED (`panns_inference`)
- Emotion/intent (HuggingFace transformers)

### CTranslate2

- `tiny.en` (39 MB) — default ASR, auto-downloads

---

## CLI Reference

### Basic Usage

```bash
# Simple run
python -m diaremot.cli run -i audio.wav -o outputs/

# Faster ASR (int8)
python -m diaremot.cli run -i audio.wav -o outputs/ --asr-compute-type int8

# Override VAD tuning
python -m diaremot.cli run -i audio.wav -o outputs/ \
  --vad-threshold 0.30 \
  --ahc-distance-threshold 0.12

# Skip optional stages
python -m diaremot.cli run -i audio.wav -o outputs/ \
  --disable-sed \
  --disable-affect

# Resume from checkpoint
python -m diaremot.cli resume -i audio.wav -o outputs/

# Diagnostics
python -m diaremot.cli diagnostics
```

### Key Flags

**ASR:**
- `--asr-compute-type float32|int8|int8_float16` (default: float32)
- `--whisper-model tiny.en|base.en|small.en` (default: tiny.en)

**Diarization:**
- `--vad-threshold` (orchestrator default: 0.35)
- `--ahc-distance-threshold` (orchestrator default: 0.15)
- `--speaker-limit N`

**Pipeline:**
- `--disable-sed` — Skip sound events
- `--disable-affect` — Skip emotion/intent
- `--quiet` — Less logging
- `--clear-cache` — Fresh run

---

## Diarization Tuning

### Orchestrator Overrides

**The orchestrator automatically tunes VAD params** (when user doesn't override):

```python
# CLI defaults (from cli.py)
vad_threshold = 0.30
speech_pad_sec = 0.20
ahc_distance_threshold = 0.12

# Orchestrator overrides (from orchestrator.py::_init_components)
vad_threshold = 0.35  # Stricter (reduce oversegmentation)
speech_pad_sec = 0.10  # Less padding (avoid overlap)
ahc_distance_threshold = 0.15  # Looser (prevent speaker fragmentation)
```

### Override via CLI

```bash
# Use CLI defaults instead of orchestrator tuning
python -m diaremot.cli run -i audio.wav -o outputs/ \
  --vad-threshold 0.30 \
  --vad-speech-pad-sec 0.20 \
  --ahc-distance-threshold 0.12
```

### When to Tune

**Too many micro-segments?** → Use orchestrator defaults (stricter VAD)  
**Missing speech?** → Lower `--vad-threshold 0.25`  
**Too many speakers?** → Raise `--ahc-distance-threshold 0.20`  
**Too few speakers?** → Lower `--ahc-distance-threshold 0.10`

---

## CSV Schema (39 Columns)

```
file_id, start, end, speaker_id, speaker_name, text,
valence, arousal, dominance,
emotion_top, emotion_scores_json,
text_emotions_top5_json, text_emotions_full_json,
intent_top, intent_top3_json,
events_top3_json, noise_tag,
asr_logprob_avg, snr_db, snr_db_sed,
wpm, duration_s, words, pause_ratio,
low_confidence_ser, vad_unstable, affect_hint,
pause_count, pause_time_s,
f0_mean_hz, f0_std_hz,
loudness_rms, disfluency_count,
error_flags,
vq_jitter_pct, vq_shimmer_db, vq_hnr_db, vq_cpps_db, voice_quality_hint
```

**Key columns:**
- `affect_hint`: "calm-positive", "agitated-negative", etc.
- `vq_*`: Voice quality from Praat
- `events_top3_json`: Top sound events
- `noise_tag`: Dominant background

---

## ONNX Conversion

### HuggingFace → ONNX

**Text classification:**
```bash
optimum-cli export onnx \
  --model SamLowe/roberta-base-go_emotions \
  --task text-classification \
  --opset 14 \
  ./models/roberta-go_emotions-onnx/
```

**Zero-shot:**
```bash
optimum-cli export onnx \
  --model facebook/bart-large-mnli \
  --task zero-shot-classification \
  --opset 14 \
  ./models/bart-mnli-onnx/
```

### PyTorch Audio → ONNX

```python
import torch

model.eval()
dummy_input = torch.randn(1, 1, 16000)

torch.onnx.export(
    model, dummy_input, "model.onnx",
    opset_version=14,
    input_names=['audio'],
    output_names=['logits'],
    dynamic_axes={'audio': {0: 'batch', 2: 'time'}}
)
```

---

## Troubleshooting

### VAD Issues

**Problem:** Fragmented segments  
**Fix:** Use orchestrator defaults (stricter threshold)
```bash
# Default behavior uses vad_threshold=0.35
python -m diaremot.cli run -i audio.wav -o outputs/
```

**Problem:** Missing speech  
**Fix:** Lower threshold
```bash
python -m diaremot.cli run -i audio.wav -o outputs/ --vad-threshold 0.25
```

### Speaker Issues

**Problem:** Too many speakers  
**Fix:** Loosen clustering or set limit
```bash
python -m diaremot.cli run -i audio.wav -o outputs/ \
  --ahc-distance-threshold 0.20 \
  --speaker-limit 5
```

### Performance Issues

**Problem:** ASR too slow  
**Fix:** Use int8 quantization
```bash
python -m diaremot.cli run -i audio.wav -o outputs/ --asr-compute-type int8
```

**Problem:** OOM on long files  
**Fix:** Auto-chunking enabled by default (>30 min files)

### Model Issues

**Problem:** ONNX model not found  
**Fix:** Set model dir
```bash
export DIAREMOT_MODEL_DIR=/path/to/models
python -m diaremot.cli diagnostics
```

**Problem:** Paralinguistics failed  
**Fix:** Install Praat
```bash
pip install praat-parselmouth
```

---

## Environment

**Required:**
```bash
export DIAREMOT_MODEL_DIR=/path/to/models
export HF_HOME=./.cache/
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
```

**Optional (quiet mode):**
```bash
export CT2_VERBOSE=0
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TRANSFORMERS_VERBOSITY=error
```

---

## Critical Notes

1. **11 stages total** (no separate auto_tune)
2. **Orchestrator tunes VAD inline** (vad_threshold=0.35, ahc_distance=0.15)
3. **Main CLI uses float32** (not int8) for ASR
4. **Schema is 39 columns** — don't modify without migration
5. **Paralinguistics required** — cannot skip
6. **SED enabled by default** — use `--disable-sed` to skip
7. **ONNX preferred** — PyTorch is fallback

---

## License

See LICENSE file.
