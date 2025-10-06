
**Intent Classification:**
- `intent_top` — Top intent label
- `intent_top3_json` — Top-3 intents with confidence (JSON)

**Sound Events:**
- `events_top3_json` — Top-3 background sounds detected (JSON)
- `noise_tag` — Dominant background class (music, keyboard, silence, etc.)

**Quality Metrics:**
- `asr_logprob_avg` — ASR confidence (average log probability)
- `snr_db` — Signal-to-noise ratio estimate (dB)
- `snr_db_sed` — SNR from SED noise score
- `low_confidence_ser` — Boolean flag for low SER confidence
- `vad_unstable` — Boolean flag for VAD instability
- `error_flags` — Comma-separated processing errors

**Prosody & Paralinguistics:**
- `wpm` — Words per minute
- `duration_s` — Segment duration
- `words` — Word count
- `pause_count` — Number of pauses
- `pause_time_s` — Total pause duration
- `pause_ratio` — Pause time / duration
- `f0_mean_hz` — Mean pitch
- `f0_std_hz` — Pitch variability
- `loudness_rms` — RMS loudness
- `disfluency_count` — Filler word count

**Voice Quality (Praat):**
- `vq_jitter_pct` — Jitter (%)
- `vq_shimmer_db` — Shimmer (dB)
- `vq_hnr_db` — Harmonics-to-Noise Ratio (dB)
- `vq_cpps_db` — Cepstral Peak Prominence (dB)

**Interpretations:**
- `affect_hint` — Human-readable affect state (e.g., "calm-positive")
- `voice_quality_hint` — Voice quality interpretation (e.g., "clear-modal")

**CRITICAL:** Do NOT modify this schema without a migration plan. Breaking changes require version bump and backward compatibility layer.

---

## Installation

### Prerequisites

**Required:**
- Python 3.11
- FFmpeg (on PATH) — `ffmpeg -version` must work
- 4+ GB RAM
- 4+ CPU cores recommended

**Optional:**
- wkhtmltopdf — For PDF summary generation

### Local Install (Windows PowerShell)

```powershell
# 1. Create virtual environment
py -3.11 -m venv .venv
. .\.venv\Scripts\Activate.ps1

# 2. Upgrade pip
python -m pip install -U pip wheel setuptools

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install package in editable mode
pip install -e .

# 5. Install dev tools (optional)
pip install ruff pytest mypy
```

### Linux/macOS

```bash
# 1. Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 2. Upgrade pip
python -m pip install -U pip wheel setuptools

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install package
pip install -e .

# 5. Dev tools
pip install ruff pytest mypy
```

### Codex Cloud

```bash
./setup.sh          # Install dependencies
./maint-codex.sh    # Maintenance tasks
python -m diaremot.cli run --input data/sample.wav --outdir outputs/
```

---

## Environment Variables

### Required

```bash
export DIAREMOT_MODEL_DIR=/path/to/models  # Model root directory
export HF_HOME=./.cache                     # HuggingFace cache
export HUGGINGFACE_HUB_CACHE=./.cache/hub   # HF Hub cache
export TRANSFORMERS_CACHE=./.cache/transformers
export TORCH_HOME=./.cache/torch

# Threading control (CPU optimization)
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_MAX_THREADS=4

# Disable tokenizer parallelism warnings
export TOKENIZERS_PARALLELISM=false
```

### Model Search Paths

The runtime automatically searches for models in this order:

1. `$DIAREMOT_MODEL_DIR` (if set)
2. `D:/models` (Windows only)
3. `/models` (Linux/macOS)
4. `./models` (project-level directory)
5. `$HOME/models`

Place ONNX and CTranslate2 assets in any of these locations.

---

## Model Assets

### ONNX Models (Preferred Runtime)

**Expected under `$DIAREMOT_MODEL_DIR`:**

```
$DIAREMOT_MODEL_DIR/
├── panns_cnn14.onnx              # SED (118 MB)
├── audioset_labels.csv            # SED labels (527 classes)
├── silero_vad.onnx                # VAD (1.8 MB)
├── ecapa_tdnn.onnx                # Speaker embeddings (6.1 MB)
├── ser_8class.onnx                # Audio emotion
├── vad_model.onnx                 # V/A/D emotion
├── roberta-base-go_emotions.onnx  # Text emotion (~500 MB)
└── bart-large-mnli.onnx           # Intent (~1.6 GB)
```

### CTranslate2 Models

Faster-Whisper auto-downloads to HuggingFace cache:

```
$HF_HOME/hub/models--guillaumekln--faster-whisper-tiny.en/
```

**Model:** `tiny.en` (39 MB)  
**Compute types:** float32 (default), int8, int8_float16

### PyTorch Fallback Models

When ONNX models unavailable, auto-download from:

- **Silero VAD:** TorchHub (`snakers4/silero-vad`)
- **PANNs SED:** `panns_inference` library → `~/panns_data/`
- **Emotion/Intent:** HuggingFace Hub (transformers library)

### Converting Models to ONNX

**Text models (HuggingFace):**
```bash
# GoEmotions (text emotion)
optimum-cli export onnx \
  --model SamLowe/roberta-base-go_emotions \
  --task text-classification \
  --opset 14 \
  ./models/roberta-base-go_emotions-onnx/

# BART-MNLI (intent)
optimum-cli export onnx \
  --model facebook/bart-large-mnli \
  --task zero-shot-classification \
  --opset 14 \
  ./models/bart-large-mnli-onnx/
```

**Audio models (PyTorch):**
```python
import torch
import onnx

# Example: SER model
dummy_input = torch.randn(1, 1, 16000)
torch.onnx.export(
    model,
    dummy_input,
    "ser_8class.onnx",
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=['audio'],
    output_names=['logits'],
    dynamic_axes={'audio': {0: 'batch_size', 2: 'time'}}
)
```

---

## CLI Usage

### Basic Commands

```bash
# Basic run (default: float32, all stages enabled)
python -m diaremot.cli run --input audio.wav --outdir outputs/

# Fast mode (int8 quantization)
python -m diaremot.cli run --input audio.wav --outdir outputs/ --asr-compute-type int8

# Override orchestrator VAD tuning
python -m diaremot.cli run --input audio.wav --outdir outputs/ \
  --vad-threshold 0.30 \
  --vad-min-speech-sec 0.80 \
  --ahc-distance-threshold 0.12

# Use preset profile
python -m diaremot.cli run --input audio.wav --outdir outputs/ --profile fast

# Disable optional stages
python -m diaremot.cli run --input audio.wav --outdir outputs/ \
  --disable-sed \
  --disable-affect

# Resume from checkpoint
python -m diaremot.cli resume --input audio.wav --outdir outputs/

# Check dependencies
python -m diaremot.cli diagnostics
python -m diaremot.cli diagnostics --strict
```

### Key Flags

**Input/Output:**
- `--input / -i` — Audio file path (WAV, MP3, M4A, FLAC, etc.)
- `--outdir / -o` — Output directory (created if doesn't exist)

**ASR Configuration:**
- `--asr-compute-type` — `float32` (default) | `int8` | `int8_float16`
- `--asr-model` — faster-whisper model name (default: `tiny.en`)

**Diarization (overrides orchestrator defaults):**
- `--vad-threshold` — Override 0.35 default (lower = more sensitive)
- `--vad-min-speech-sec` — Override 0.80s default
- `--vad-min-silence-sec` — Override 0.80s default
- `--speech-pad-sec` — Override 0.10s default
- `--ahc-distance-threshold` — Override 0.15 default (lower = more speakers)
- `--speaker-limit` — Maximum speakers to detect

**Stage Control:**
- `--disable-sed` — Skip sound event detection
- `--disable-affect` — Skip emotion/intent analysis
- `--profile` — Preset config (default|fast|accurate|offline|path/to/json)

**System:**
- `--quiet` — Reduce console verbosity
- `--clear-cache` — Clear cache before running
- `--resume` — Resume from checkpoint (auto-detect checkpoint dir)

### Profiles

**Built-in profiles:**

**default:**
```json
{
  "asr_compute_type": "float32",
  "vad_threshold": null,  // Use orchestrator default (0.35)
  "enable_sed": true,
  "enable_affect": true
}
```

**fast:**
```json
{
  "asr_compute_type": "int8",
  "vad_threshold": 0.40,  // Even stricter
  "ahc_distance_threshold": 0.20,  // Fewer speakers
  "enable_sed": false,
  "enable_affect": false
}
```

**accurate:**
```json
{
  "asr_compute_type": "float32",
  "vad_threshold": 0.25,  // More sensitive
  "ahc_distance_threshold": 0.10,  // More speakers
  "enable_sed": true,
  "enable_affect": true
}
```

**Custom profile:**
```bash
python -m diaremot.cli run -i audio.wav -o outputs/ --profile /path/to/custom.json
```

---

## Troubleshooting

### Common Issues

**Issue:** "VAD threshold too low, getting fragmented segments"

**Symptoms:** Many short segments (<2s), same speaker split into multiple IDs

**Fix:** Use stricter VAD threshold or orchestrator defaults:
```bash
# Option 1: Let orchestrator use 0.35 default (don't set --vad-threshold)
python -m diaremot.cli run -i audio.wav -o outputs/

# Option 2: Set even stricter
python -m diaremot.cli run -i audio.wav -o outputs/ --vad-threshold 0.40
```

---

**Issue:** "Too many speakers detected"

**Symptoms:** 10+ speakers in a 2-person conversation

**Fix:** Loosen AHC distance threshold or set speaker limit:
```bash
# Looser clustering (fewer speakers)
python -m diaremot.cli run -i audio.wav -o outputs/ --ahc-distance-threshold 0.20

# Hard speaker limit
python -m diaremot.cli run -i audio.wav -o outputs/ --speaker-limit 5
```

---

**Issue:** "ASR too slow"

**Symptoms:** Transcription taking >2x audio length

**Fix:** Use int8 quantization:
```bash
python -m diaremot.cli run -i audio.wav -o outputs/ --asr-compute-type int8
```

**Performance comparison (1 hour audio, 4-core CPU):**
- float32: ~10 minutes
- int8: ~5 minutes
- Quality difference: <2% WER increase

---

**Issue:** "Paralinguistics stage failed / Praat errors"

**Symptoms:** Voice quality metrics all 0.0

**Fix:** Ensure Praat-Parselmouth installed:
```bash
pip install praat-parselmouth
```

**Common Praat failures:**
- Very short segments (<0.5s) → Insufficient pitch periods
- Very noisy audio → HNR calculation fails
- Whispered speech → F0 detection fails

**Fallback behavior:** WPM computed from ASR text, voice quality set to 0.0

---

**Issue:** "ONNX model not found, using PyTorch fallback"

**Symptoms:** Warnings about missing ONNX files, slower inference

**Fix:** Verify model paths:
```bash
# Check model directory
ls -lh $DIAREMOT_MODEL_DIR/

# Expected files
panns_cnn14.onnx              # 118 MB
audioset_labels.csv           # ~100 KB
silero_vad.onnx              # 1.8 MB
ecapa_tdnn.onnx              # 6.1 MB
```

**Convert models if missing** (see "Converting Models to ONNX" section)

---

**Issue:** "Speaker names not persistent across files"

**Symptoms:** Same speaker gets different names in different audio files

**Fix:** Use shared speaker registry:
```bash
# Process all files with same registry
python -m diaremot.cli run -i audio1.wav -o outputs/ --speaker-registry speakers.json
python -m diaremot.cli run -i audio2.wav -o outputs/ --speaker-registry speakers.json
```

Registry file is auto-created and updated. Speaker centroids persist across runs.

---

**Issue:** "Out of memory on long audio files"

**Symptoms:** Process killed during transcription or affect analysis

**Fix:** Auto-chunking should handle this, but if still failing:
```bash
# Reduce affect window size (edit pipeline config)
# Or process in smaller segments manually

# Split audio first
ffmpeg -i long_audio.wav -f segment -segment_time 1800 -c copy part_%03d.wav

# Process each part
for part in part_*.wav; do
    python -m diaremot.cli run -i "$part" -o "outputs/${part%.wav}/"
done
```

---

### Diagnostic Commands

```bash
# Check all dependencies (non-strict)
python -m diaremot.cli diagnostics

# Strict version checking
python -m diaremot.cli diagnostics --strict

# Verify model files exist
ls -lh $DIAREMOT_MODEL_DIR/

# Test ONNX runtime
python -c "import onnxruntime; print(onnxruntime.__version__)"

# Test faster-whisper
python -c "from faster_whisper import WhisperModel; print('OK')"

# Test Praat
python -c "import parselmouth; print(parselmouth.__version__)"

# Clear cache and retry
python -m diaremot.cli run -i audio.wav -o outputs/ --clear-cache

# Check logs
tail -f outputs/processing.log
```

---

## Performance Tuning

### CPU Optimization

**Thread control:**
```bash
export OMP_NUM_THREADS=4        # OpenMP threads
export MKL_NUM_THREADS=4        # Intel MKL threads
export NUMEXPR_MAX_THREADS=4    # NumPy/NumExpr
```

**Recommendations:**
- 4-core CPU: Set to 4
- 8-core CPU: Set to 4-6 (avoid hyperthreading overhead)
- 16+ core CPU: Set to 6-8

**ASR threading:**
- faster-whisper: Always 1 thread (CTranslate2 limitation)
- Don't set higher; no performance gain

### Memory Optimization

**Auto-chunking thresholds:**
- Preprocess: 30 minutes (auto-chunks longer files)
- ASR: 8 minutes per window (max_asr_window_sec)
- Affect: 30 seconds per window

**Reduce memory if needed:**
```python
# Edit pipeline/config.py
class AudioConfig:
    chunk_threshold_minutes = 20.0  # Chunk earlier
    
class ASRConfig:
    max_asr_window_sec = 300  # 5 minutes instead of 8
```

### Inference Speed

**Fastest configuration:**
```bash
python -m diaremot.cli run -i audio.wav -o outputs/ \
  --profile fast \
  --asr-compute-type int8
```

**Expected throughput (4-core CPU, 1-hour audio):**
- float32 + all stages: ~12 minutes
- int8 + no SED/affect: ~4 minutes

**Bottlenecks:**
1. Diarization (30% of time) — ECAPA embedding extraction
2. Transcription (25% of time) — CTranslate2 inference
3. Affect (20% of time) — ONNX emotion models
4. Paralinguistics (15% of time) — Praat analysis
5. SED (10% of time) — PANNs inference

---

## Output Files Reference

### diarized_transcript_with_emotion.csv

**Primary output** — 39-column CSV with per-segment data

**Use cases:**
- Import into annotation tools (ELAN, Praat, Audacity)
- Data analysis with pandas/R
- Manual review and correction
- Training data for downstream models

**Scrub-friendly:** Load in spreadsheet software, sort by timestamp, filter by speaker/emotion

---

### segments.jsonl

**JSON Lines** format — one JSON object per segment

**Advantages over CSV:**
- Nested data structures (emotion scores, events)
- No escaping issues
- Stream-friendly (process line-by-line)

**Use cases:**
- Programmatic access (Python, JavaScript, etc.)
- Database import (MongoDB, Elasticsearch)
- Streaming processing pipelines

---

### speakers_summary.csv

**Per-speaker rollup** statistics

**Columns:**
- speaker_id, speaker_name
- total_duration_s, turn_count
- avg_valence, avg_arousal, avg_dominance
- emotion_distribution (JSON)
- intent_distribution (JSON)
- avg_wpm, avg_f0_hz, avg_loudness_rms
- avg_vq_jitter, avg_vq_shimmer, avg_vq_hnr, avg_vq_cpps
- interruptions_made, interruptions_received
- background_event_exposure (JSON)

**Use cases:**
- Speaker comparison
- Conversation balance analysis
- Voice quality assessment
- Emotional profile per speaker

---

### summary.html

**Interactive HTML report**

**Sections:**
1. **Quick Take** — Key metrics (duration, speaker count, dominant emotions, etc.)
2. **Speaker Snapshots** — Per-speaker cards with stats
3. **Moments to Check** — Flagged segments (high arousal, overlaps, background events)
4. **Timeline** — Visual timeline with color-coded emotions
5. **Event Log** — Sound event timeline

**Features:**
- Click timestamps to jump to moments
- Filter by speaker/emotion
- Export-friendly (self-contained HTML)

---

### summary.pdf

**PDF version** of HTML report

**Requires:** wkhtmltopdf on PATH

**Generation:**
```bash
# Install wkhtmltopdf first
# Windows: choco install wkhtmltopdf
# macOS: brew install wkhtmltopdf
# Linux: apt-get install wkhtmltopdf

# Then run pipeline normally
python -m diaremot.cli run -i audio.wav -o outputs/
```

PDF auto-generated if wkhtmltopdf detected.

---

### speaker_registry.json

**Persistent speaker identities** across files

**Structure:**
```json
{
  "Speaker_A": {
    "centroid": [0.1, 0.2, ..., 0.192],  // 192-dim ECAPA vector
    "count": 15,  // Times seen
    "last_updated": "2025-10-05T12:00:00Z"
  },
  "Speaker_B": { ... }
}
```

**Use cases:**
- Multi-file speaker tracking
- Speaker verification
- Cross-conversation analysis

**Update:** Auto-updates on each run if `--speaker-registry` specified

---

### events_timeline.csv & events.jsonl

**Sound event timeline** from SED

**Columns (CSV):**
- start, end, duration_s
- event_label (collapsed class)
- confidence
- original_label (AudioSet class)

**Use cases:**
- Context reconstruction
- Meeting quality analysis (keyboard noise, phone rings)
- Ambient sound tracking

---

### timeline.csv

**Fast-scrub timeline** — simplified view

**Columns:**
- timestamp, speaker_name, text
- emotion_summary, voice_quality_summary
- background_event

**Use cases:**
- Quick review
- Annotation tool import
- Client-facing summaries

---

### qc_report.json

**Quality control metrics**

**Includes:**
- Pipeline stage success/failure
- Segment count, average duration
- VAD statistics (speech ratio, silence ratio)
- ASR confidence distribution
- Error flags summary
- Processing time per stage

**Use cases:**
- Automated QA
- Performance monitoring
- Debugging failed runs

---

## Architecture Notes

### ONNX-Preferred Inference

**Why ONNX?**
- 2-5x faster CPU inference vs PyTorch
- Lower memory footprint
- No torch.jit complexity
- Smaller deployment size

**Fallback Strategy:**
1. Try ONNX model (ONNXRuntime)
2. If unavailable, use PyTorch (transformers, TorchHub)
3. If both fail, disable stage gracefully (with warning)

**No GPU Support:**
- DiaRemot is CPU-only by design
- Target deployment: servers, edge devices, laptops
- No CUDA/ROCm dependencies

### VAD Parameter Philosophy

**Orchestrator overrides exist for good reasons:**

**Stricter VAD (0.35 vs 0.30):**
- Reduces false positives in noisy environments
- Prevents fragment over-segmentation
- Better for podcast/interview scenarios

**Less padding (0.10 vs 0.20):**
- Avoids segment overlap
- Cleaner turn boundaries
- Better for high-interruption conversations

**Looser AHC (0.15 vs 0.12):**
- Prevents single speaker fragmentation
- More robust to voice variation (phone vs in-person)
- Better generalization across audio quality

**When to override:**
- Clean studio recordings → Use CLI defaults (0.30, 0.12)
- Noisy real-world audio → Use orchestrator defaults (0.35, 0.15)
- Custom scenarios → Experiment and tune

### Auto-Chunking Strategy

**Why auto-chunk?**
- Long files (>30 min) cause memory pressure
- ASR performance degrades on very long sequences
- Affect models have attention limits

**Chunking points:**
- Preprocess: 30 minutes
- ASR: 8 minutes (max_asr_window_sec)
- Affect: 30 seconds

**Seamless stitching:**
- Chunks processed independently
- Results merged with overlap handling
- Timestamps adjusted to global timeline

---

## Development

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test
pytest tests/test_pipeline.py -v

# With coverage
pytest tests/ --cov=src/diaremot --cov-report=html
```

### Linting

```bash
# Check code
ruff check src/ tests/

# Auto-fix
ruff check src/ tests/ --fix

# Format
ruff format src/ tests/
```

### Type Checking

```bash
mypy src/diaremot/
```

---

## License

See LICENSE file.

---

## Support

**Issues:** https://github.com/yourusername/diaremot/issues  
**Docs:** See `AGENTS.md` for agent/AI instructions  
**Changelog:** See `CHANGELOG.md` for version history
> Codex Cloud Note
> If you are running on Codex Cloud, use the Codex‑specific guide: README_CLOUD.md. It contains the exact pinned package versions and the allowed apt instructions for ffmpeg. This README targets general/local setups.
