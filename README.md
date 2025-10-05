# DiaRemot — CPU‑Only Speech Intelligence Pipeline

Process long, real‑world audio **on CPU** and produce a diarized transcript with per‑segment:
- **Tone (Valence/Arousal/Dominance)**
- **Speech emotion (8‑class)**
- **Text emotions (GoEmotions, 28)**
- **Intent** (zero‑shot over fixed labels)
- **Sound‑event context (SED: music, keyboard, door, TV, etc.)**
- **Paralinguistics (REQUIRED)**: speech rate (WPM), pauses, and voice‑quality via **Praat‑Parselmouth**: **jitter**, **shimmer**, **HNR**, **CPPS**
- **Persistent speaker names across files**

## What You Get

The pipeline produces comprehensive outputs for each audio file:

### Primary Outputs
- `diarized_transcript_with_emotion.csv` — The main 39-column file with all segment-level features. This is your scrub-friendly data source.
- `segments.jsonl` — Per-segment JSON records with audio clips, text, and all feature overlaps
- `speakers_summary.csv` — Rolled-up statistics per speaker: average V/A/D, emotion mix, intent distribution, WPM, SNR, voice quality metrics
- `summary.html` — Interactive HTML report with Quick Take, Speaker Snapshots, Moments to Check (high arousal), and Action Items
- `speaker_registry.json` — Persistent speaker identity via ECAPA embeddings, enabling cross-file speaker tracking
- `events_timeline.csv` + `events.jsonl` — Timeline of sound events detected by PANNs (music starts, keyboard typing, doors, phone rings, etc.)
- `timeline.csv` — Fast-scrubbing index of start/end times and speaker IDs
- `qc_report.json` — Pipeline health metrics, stage timings, warnings, model versions

### Optional Outputs
- `summary.pdf` — PDF version of the HTML summary (requires `wkhtmltopdf` installed)

## Model Set (CPU‑Optimized)

All models run on CPU. The pipeline intelligently selects ONNX models when available (2-5x faster) and falls back to PyTorch when needed.

### Diarization Stack
**Components:** Silero VAD + ECAPA-TDNN embeddings + Agglomerative Hierarchical Clustering (AHC)

**VAD (Voice Activity Detection):**
- Prefers ONNX Silero models when available
- Falls back to TorchHub Silero VAD if ONNX unavailable
- Ultimate fallback to energy-based VAD heuristic if models missing

**Parameter Tuning:**
The orchestrator applies intelligent defaults when you don't override via CLI:

CLI defaults:
```
vad_threshold: 0.30
vad_min_speech_sec: 0.80
vad_min_silence_sec: 0.80
speech_pad_sec: 0.20
ahc_distance_threshold: 0.12
```

Orchestrator overrides (for better performance):
```
vad_threshold: 0.35          # Stricter to reduce micro-segmentation
vad_min_speech_sec: 0.80     # Same as CLI
vad_min_silence_sec: 0.80    # Same as CLI
speech_pad_sec: 0.10         # Less padding to avoid overlaps
ahc_distance_threshold: 0.15 # Looser clustering to prevent speaker fragmentation
```

**Why the overrides?** Testing on real-world audio showed that the orchestrator's stricter VAD threshold reduces over-segmentation (splitting single utterances into fragments), while the looser AHC threshold prevents false speaker splits (treating one person as multiple speakers).

**How to override:** Use CLI flags to restore original defaults:
```bash
python -m diaremot.cli run -i audio.wav -o outputs/ \
  --vad-threshold 0.30 \
  --vad-min-speech-sec 0.80 \
  --ahc-distance-threshold 0.12
```

### ASR (Automatic Speech Recognition)
**Model:** Faster‑Whisper `tiny.en` via CTranslate2  
**Default quantization:** `float32` (main CLI entry point)  
**Optional:** Use `--asr-compute-type int8` for 2-3x speedup with minimal quality loss

**Why float32 default?** Better transcription accuracy on challenging audio (accents, background noise, soft speech). The speed difference is acceptable for most use cases.

**When to use int8:** Large batch processing where speed matters more than perfect accuracy, or when processing clear audio with minimal noise.

### Affect Analysis Models

**Tone (V/A/D — Valence/Arousal/Dominance):**
- Model: `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`
- Backend: HuggingFace transformers (CPU)
- Output: Three continuous dimensions from -1 to 1

**Speech Emotion (8-class):**
- Model: `Dpngtm/wav2vec2-emotion-recognition`
- Backend: HuggingFace transformers (CPU)
- Classes: angry, happy, sad, neutral, fearful, disgusted, surprised, calm

**Text Emotions (28-class):**
- Model: `SamLowe/roberta-base-go_emotions`
- Backend: ONNX (preferred) or HuggingFace transformers (fallback)
- Output: Full 28-class distribution; top 5 kept in CSV

**Intent (Zero-Shot Classification):**
- Model: `facebook/bart-large-mnli`
- Backend: ONNX (preferred via local exports) or HuggingFace (fallback)
- Fallback: Rule-based heuristics if models unavailable
- Labels: question, request, instruction, command, complaint, apology, opinion, agreement, disagreement, suggestion, status_update, small_talk, gratitude, greeting, farewell

### Sound Event Detection (SED)

**Model:** PANNs CNN14 (AudioSet-pretrained)  
**Backend:** ONNX via onnxruntime (preferred), PyTorch via `panns_inference` (fallback)  
**Enabled by default:** Yes (disable with `--disable-sed`)

**Processing parameters:**
- Frame duration: 1.0 second
- Hop length: 0.5 seconds (50% overlap)
- Median filtering: 3-5 frames to smooth detections
- Hysteresis thresholds: enter at 0.50, exit at 0.35
- Minimum event duration: 0.30 seconds
- Merge gap: Events separated by ≤0.20s are merged
- Label collapse: AudioSet's 527 classes → ~20 semantic groups

**Semantic groups include:**
- Speech and vocalizations (speech, laughter, crying, coughing)
- Music and instruments
- Domestic sounds (door, keyboard, phone, alarm, appliance)
- Ambient noise (traffic, rain, wind)
- Animals
- Alerts and notifications

**Why enabled by default?** Sound event context is critical for understanding conversation dynamics. A phone ringing, keyboard typing, or door opening often correlates with topic shifts, interruptions, or meeting transitions.

### Paralinguistics (REQUIRED)

**Engine:** Praat-Parselmouth (Python wrapper around Praat acoustic analysis toolkit)  
**Status:** Required stage — cannot be skipped

**Voice Quality Metrics (Praat-based):**
- `vq_jitter_pct`: Period-to-period pitch variation (voice stability)
- `vq_shimmer_db`: Amplitude variation (voice roughness)
- `vq_hnr_db`: Harmonics-to-Noise Ratio (voice clarity)
- `vq_cpps_db`: Cepstral Peak Prominence Smoothed (voice quality indicator)

**Prosodic Features:**
- `wpm`: Words per minute (speaking rate)
- `pause_count`: Number of pauses detected
- `pause_time_s`: Total pause duration in seconds
- `pause_ratio`: Proportion of segment duration that is pauses
- `f0_mean_hz`: Mean fundamental frequency (pitch)
- `f0_std_hz`: Pitch variability
- `loudness_rms`: RMS loudness
- `disfluency_count`: Filler words and hesitations (um, uh, like, you know)

**Fallback behavior:** If Praat analysis fails (corrupted audio segment, extreme noise), the pipeline computes WPM from ASR text and sets voice quality metrics to 0.0 to prevent pipeline failure.

## Installation

### Prerequisites
1. **Python 3.11** (tested and recommended)
2. **FFmpeg** on system PATH — verify with `ffmpeg -version`
3. Virtual environment (strongly recommended)

### Installation Steps (Windows PowerShell)

```powershell
# Create virtual environment
py -3.11 -m venv .venv

# Activate environment
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install build tools
python -m pip install -U pip wheel setuptools

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

### Installation Steps (Linux/macOS)

```bash
# Create virtual environment
python3.11 -m venv .venv

# Activate environment
source .venv/bin/activate

# Upgrade pip and install build tools
python -m pip install -U pip wheel setuptools

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

### Development Tools (Optional)

```bash
pip install ruff pytest mypy
```

## Environment Setup

### Required Environment Variables

The pipeline expects these environment variables to be set. Most are auto-configured, but you can override for custom setups:

```bash
# Model directory (primary location for ONNX models)
DIAREMOT_MODEL_DIR=/workspace/models

# HuggingFace cache locations
HF_HOME=.cache/hf
HUGGINGFACE_HUB_CACHE=.cache/hf
TRANSFORMERS_CACHE=.cache/transformers

# PyTorch cache
TORCH_HOME=.cache/torch

# Thread control for CPU parallelism
OMP_NUM_THREADS=4
MKL_NUM_THREADS=4
NUMEXPR_MAX_THREADS=4

# Prevent tokenizer parallelism warnings
TOKENIZERS_PARALLELISM=false
```

### Model Search Paths

The runtime automatically searches for models in this order:
1. `$DIAREMOT_MODEL_DIR` (if set)
2. `D:/models` (Windows) or `/models` (Unix-like systems)
3. Project-level `models/` directory
4. `$HOME/models`

Place your ONNX and CTranslate2 assets (e.g., `faster-whisper/tiny.en`, `bart/model_uint8.onnx`, `panns/model.onnx`) in any of these locations.

## Usage Examples

### Basic Usage

```bash
# Process audio file with default settings
python -m diaremot.cli run --input audio.wav --outdir outputs/

# Short form
python -m diaremot.cli run -i audio.wav -o outputs/
```

### Performance Tuning

```bash
# Use int8 quantization for 2-3x faster ASR
python -m diaremot.cli run -i audio.wav -o outputs/ --asr-compute-type int8

# Disable sound event detection (faster, less context)
python -m diaremot.cli run -i audio.wav -o outputs/ --disable-sed

# Disable affect analysis (faster, no emotions/intent)
python -m diaremot.cli run -i audio.wav -o outputs/ --disable-affect

# Use fast profile (combines optimizations)
python -m diaremot.cli run -i audio.wav -o outputs/ --profile fast
```

### VAD and Clustering Tuning

```bash
# Override orchestrator's VAD threshold (use CLI default 0.30)
python -m diaremot.cli run -i audio.wav -o outputs/ --vad-threshold 0.30

# Adjust speaker clustering sensitivity
python -m diaremot.cli run -i audio.wav -o outputs/ --ahc-distance-threshold 0.12

# Increase minimum speech duration (reduce fragmention)
python -m diaremot.cli run -i audio.wav -o outputs/ --vad-min-speech-sec 1.0

# Combine multiple VAD adjustments
python -m diaremot.cli run -i audio.wav -o outputs/ \
  --vad-threshold 0.25 \
  --vad-min-speech-sec 1.0 \
  --vad-min-silence-sec 0.5 \
  --vad-speech-pad-sec 0.15
```

### Resume and Cache Management

```bash
# Resume from checkpoint after interruption
python -m diaremot.cli resume --input audio.wav --outdir outputs/

# Clear cache and reprocess everything
python -m diaremot.cli run -i audio.wav -o outputs/ --clear-cache

# Ignore cached results (force reprocess)
python -m diaremot.cli run -i audio.wav -o outputs/ --ignore-tx-cache
```

### Diagnostics

```bash
# Check dependencies (loose version check)
python -m diaremot.cli diagnostics

# Strict version requirements check
python -m diaremot.cli diagnostics --strict
```

### Configuration Profiles

```bash
# Use built-in fast profile
python -m diaremot.cli run -i audio.wav -o outputs/ --profile fast

# Use built-in accurate profile
python -m diaremot.cli run -i audio.wav -o outputs/ --profile accurate

# Use built-in offline profile (prefer ONNX, no downloads)
python -m diaremot.cli run -i audio.wav -o outputs/ --profile offline

# Use custom JSON profile
python -m diaremot.cli run -i audio.wav -o outputs/ --profile /path/to/config.json
```

## CSV Schema Reference

The primary output is `diarized_transcript_with_emotion.csv` with exactly **39 columns**.

**Schema source:** `src/diaremot/pipeline/outputs.py::SEGMENT_COLUMNS`

### Complete Column List

```
file_id                     — Input filename
start                       — Segment start time (seconds)
end                         — Segment end time (seconds)
speaker_id                  — Speaker cluster ID (e.g., SPEAKER_00)
speaker_name                — Persistent speaker name from registry
text                        — Transcribed text

valence                     — Emotional valence (-1 negative to +1 positive)
arousal                     — Emotional arousal (-1 calm to +1 excited)
dominance                   — Emotional dominance (-1 submissive to +1 dominant)

emotion_top                 — Top speech emotion (angry, happy, sad, etc.)
emotion_scores_json         — Full 8-class emotion distribution (JSON)

text_emotions_top5_json     — Top 5 text emotions from GoEmotions (JSON)
text_emotions_full_json     — Full 28-class text emotion distribution (JSON)

intent_top                  — Top intent (question, request, complaint, etc.)
intent_top3_json            — Top 3 intents with scores (JSON)

events_top3_json            — Top 3 sound events detected (JSON)
noise_tag                   — Dominant background sound class

asr_logprob_avg             — ASR confidence (average log probability)
snr_db                      — Signal-to-noise ratio estimate (dB)
snr_db_sed                  — SNR from SED noise score

wpm                         — Words per minute (speaking rate)
duration_s                  — Segment duration (seconds)
words                       — Word count
pause_ratio                 — Proportion of time spent in pauses

low_confidence_ser          — Speech emotion recognition confidence flag
vad_unstable                — VAD instability indicator
affect_hint                 — Human-readable affect summary

pause_count                 — Number of pauses detected
pause_time_s                — Total pause duration (seconds)

f0_mean_hz                  — Mean pitch (Hz)
f0_std_hz                   — Pitch variability (Hz)

loudness_rms                — RMS loudness
disfluency_count            — Count of filler words/hesitations

error_flags                 — Processing error indicators

vq_jitter_pct               — Voice jitter (period variation %)
vq_shimmer_db               — Voice shimmer (amplitude variation dB)
vq_hnr_db                   — Harmonics-to-noise ratio (dB)
vq_cpps_db                  — Cepstral peak prominence smoothed (dB)
voice_quality_hint          — Human-readable voice quality interpretation
```

### Key Column Details

**Sound Events (`events_top3_json`):**
- JSON array of top-k AudioSet clusters detected in this segment
- Example: `[{"label": "Music", "score": 0.85}, {"label": "Speech", "score": 0.72}]`

**Noise Tag (`noise_tag`):**
- Dominant background class from SED analysis
- Values: Speech, Music, Silence, Keyboard, Phone, Door, etc.

**SNR Estimates:**
- `snr_db`: Traditional SNR estimate
- `snr_db_sed`: Approximate SNR derived from SED noise classification scores

**Quality Flags:**
- `low_confidence_ser`: True if speech emotion recognition confidence below threshold
- `vad_unstable`: True if VAD detected unstable speech regions
- `error_flags`: Comma-separated error codes if processing issues occurred

**Affect Hint (`affect_hint`):**
- Human-readable affect state: "calm-positive", "agitated-negative", "neutral-status", etc.
- Derived from V/A/D values and intent

**Voice Quality Interpretation (`voice_quality_hint`):**
- Human-readable summary of voice quality metrics
- Example: "clear-stable", "rough-strained", "breathy-unstable"

## Pipeline Stages (Detailed)

The pipeline executes exactly **11 stages** in order. Each stage is modular and can be inspected/debugged independently.

**Source:** `src/diaremot/pipeline/stages/__init__.py::PIPELINE_STAGES`

### Stage 1: dependency_check

**Module:** `stages/dependency_check.py`

Validates runtime dependencies before processing begins:
- `onnxruntime >= 1.16.0`
- `faster-whisper >= 1.0.0` (includes CTranslate2)
- `transformers` (tokenizers only)
- Praat-Parselmouth

If strict mode enabled (`--diagnostics --strict`), enforces minimum versions. Otherwise logs warnings for version mismatches.

### Stage 2: preprocess

**Module:** `stages/preprocess.py::run_preprocess`

Audio normalization and preparation:
1. Load audio file (supports all FFmpeg formats)
2. Resample to 16 kHz mono
3. High-pass filter (80-120 Hz) to remove DC offset and rumble
4. Optional gentle noise reduction (spectral subtraction)
5. Gated gain for low-RMS speech segments
6. Gentle compression to even out dynamics
7. Loudness normalization to -20 LUFS (optimized for ASR)
8. Auto-chunking for files >30 minutes (default threshold)

**Outputs:**
- `state.y`: Normalized audio array
- `state.sr`: Sample rate (16000)
- `state.health`: Audio health metrics (SNR, clipping, dynamic range)
- `state.duration_s`: File duration in seconds

**Cache management:** Computes audio hash for cache lookups. Checks for cached diarization and transcription results.

### Stage 3: background_sed

**Module:** `stages/preprocess.py::run_background_sed`

Sound event detection using PANNs CNN14:
1. Extract 1-second mel-spectrogram frames (0.5s hop)
2. Run ONNX inference (or PyTorch fallback)
3. Apply median filtering (3-5 frames) to smooth detections
4. Hysteresis thresholding (0.50 enter, 0.35 exit)
5. Merge events separated by <0.20s
6. Filter events shorter than 0.30s
7. Collapse AudioSet 527 classes → ~20 semantic groups

**Outputs:**
- `state.sed_info`: Detected events with timestamps
- Dominant background label and noise score stored in `pipeline.stats.config_snapshot`

**Failure mode:** If models missing or inference fails, logs warning and continues. SED data remains empty.

### Stage 4: diarize

**Module:** `stages/diarize.py`

Speaker segmentation using Silero VAD + ECAPA-TDNN + AHC:
1. **VAD:** Detect speech regions (ONNX → Torch → Energy fallback)
2. **Embedding extraction:** Extract ECAPA-TDNN speaker embeddings for each speech region
3. **Clustering:** Agglomerative hierarchical clustering to group embeddings by speaker
4. **Registry matching:** Compare cluster centroids to speaker registry for persistent naming
5. **Post-processing:** Merge short turns, apply collar, enforce minimum turn duration

**Outputs:**
- `state.turns`: List of diarized turns with speaker IDs and timestamps
- Updated speaker registry with new/matched speakers

**Cache:** If cached diarization available, skips this stage entirely.

### Stage 5: transcribe

**Module:** `stages/asr.py`

ASR using faster-whisper (CTranslate2):
1. Iterate over diarized turns (not full audio)
2. For each turn, run faster-whisper with VAD filtering
3. Extract word-level timestamps
4. Compute ASR confidence (log probabilities)
5. Detect language (if auto mode)

**Outputs:**
- `state.norm_tx`: Transcribed turns with text, timestamps, speaker IDs

**Cache:** If cached transcription available, reconstructs turns from cache and skips ASR.

**Chunking:** Long audio segments (>480s default) are processed in windows with overlap to prevent timeout.

### Stage 6: paralinguistics

**Module:** `stages/paralinguistics.py`

Voice quality and prosody extraction using Praat-Parselmouth:
1. For each transcribed segment, extract audio clip
2. Run Praat acoustic analysis:
   - Jitter (pitch period variation)
   - Shimmer (amplitude variation)
   - HNR (harmonics-to-noise ratio)
   - CPPS (cepstral peak prominence)
3. Compute prosodic features:
   - WPM from word count and duration
   - Pause detection and quantification
   - F0 (pitch) statistics
   - RMS loudness
   - Disfluency count (filler words)

**Outputs:**
- `state.para_metrics`: Dictionary mapping segment index → paralinguistic features

**Fallback:** If Praat fails, computes WPM from text and sets voice quality metrics to 0.0.

**Critical:** This stage is required and cannot be skipped. Voice quality metrics are part of the core 39-column schema.

### Stage 7: affect_and_assemble

**Module:** `stages/affect.py`

Affect analysis and segment assembly:
1. For each transcribed segment:
   - Extract audio clip
   - Run audio emotion models (V/A/D, 8-class SER)
   - Run text emotion model (GoEmotions 28-class)
   - Run intent classification (BART zero-shot)
   - Compute affect hint from V/A/D + intent
2. Merge with paralinguistic metrics
3. Merge with SED events
4. Build final 39-column segment record

**Outputs:**
- `state.segments_final`: Complete segment list with all 39 columns populated

**Failure mode:** If affect models unavailable, populates neutral defaults and sets confidence flags.

### Stage 8: overlap_interruptions

**Module:** `stages/summaries.py::run_overlap`

Turn-taking analysis:
1. Detect overlapping speech regions
2. Classify interruptions vs. cooperative overlaps
3. Compute per-speaker interruption statistics

**Outputs:**
- `state.overlap_stats`: Overall overlap metrics
- `state.per_speaker_interrupts`: Per-speaker interruption counts

### Stage 9: conversation_analysis

**Module:** `stages/summaries.py::run_conversation`

Conversation flow metrics:
1. Compute turn-taking balance (dominance ratio)
2. Measure response latencies
3. Detect topic shifts (via intent/emotion changes)
4. Calculate conversation health scores

**Outputs:**
- `state.conv_metrics`: ConversationMetrics object with flow statistics

### Stage 10: speaker_rollups

**Module:** `stages/summaries.py::run_speaker_rollups`

Per-speaker aggregation:
1. Group segments by speaker
2. Compute averages: V/A/D, WPM, SNR, voice quality
3. Build emotion distribution histograms
4. Aggregate intent distributions
5. Calculate speaking time and word counts

**Outputs:**
- `state.speakers_summary`: Per-speaker summary records

### Stage 11: outputs

**Module:** `stages/summaries.py::run_outputs`

File writing:
1. `diarized_transcript_with_emotion.csv` — 39-column CSV
2. `segments.jsonl` — JSONL with full segment data
3. `speakers_summary.csv` — Per-speaker aggregates
4. `timeline.csv` — Fast-scrubbing index
5. `qc_report.json` — Pipeline health report
6. `summary.html` — Interactive HTML report
7. `summary.pdf` — PDF report (optional, requires wkhtmltopdf)
8. `events_timeline.csv`, `events.jsonl` — SED event timelines

All outputs written to `--outdir` directory.

## Configuration Tuning

### When to Tune VAD

**Symptoms of over-segmentation:**
- Segments are very short (<2 seconds)
- Single utterances split across multiple rows
- High segment count relative to audio duration

**Solution:** Increase `vad-min-speech-sec` and/or decrease `vad-threshold`:
```bash
python -m diaremot.cli run -i audio.wav -o outputs/ \
  --vad-threshold 0.25 \
  --vad-min-speech-sec 1.0
```

**Symptoms of under-segmentation:**
- Long segments with multiple speakers
- Missed turn boundaries
- Low segment count

**Solution:** Decrease `vad-min-speech-sec` and/or increase `vad-threshold`:
```bash
python -m diaremot.cli run -i audio.wav -o outputs/ \
  --vad-threshold 0.40 \
  --vad-min-speech-sec 0.5
```

### When to Tune AHC Clustering

**Symptoms of over-clustering (too many speakers):**
- Same person assigned multiple speaker IDs
- Speaker switches mid-utterance
- More speakers than actually present

**Solution:** Increase `ahc-distance-threshold`:
```bash
python -m diaremot.cli run -i audio.wav -o outputs/ \
  --ahc-distance-threshold 0.20
```

**Symptoms of under-clustering (too few speakers):**
- Different people grouped as same speaker
- Speaker mixing in transcript
- Fewer speakers than actually present

**Solution:** Decrease `ahc-distance-threshold`:
```bash
python -m diaremot.cli run -i audio.wav -o outputs/ \
  --ahc-distance-threshold 0.10
```

## Troubleshooting

### Pipeline Fails at dependency_check

**Problem:** Missing or incompatible dependencies

**Solution:**
```bash
# Check what's missing
python -m diaremot.cli diagnostics --strict

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Transcription Quality Poor

**Problem:** Incorrect language, low confidence scores

**Solutions:**
- Specify language explicitly: `--language en`
- Try different compute types: `--asr-compute-type float32`
- Check audio quality in preprocessing health metrics
- Increase beam size for better accuracy: `--beam-size 4`

### Too Many/Few Speakers Detected

**See:** "When to Tune AHC Clustering" section above

### Processing Very Slow

**Solutions:**
1. Use int8 quantization: `--asr-compute-type int8`
2. Disable SED: `--disable-sed`
3. Disable affect: `--disable-affect`
4. Use fast profile: `--profile fast`
5. Reduce beam size: `--beam-size 1` (already default)

### Models Not Found

**Problem:** "Model not found" or "Failed to load model" errors

**Solution:**
```bash
# Set model directory
export DIAREMOT_MODEL_DIR=/path/to/models

# Verify models exist
ls $DIAREMOT_MODEL_DIR

# Or place models in default locations:
# - D:/models (Windows)
# - /models (Unix)
# - ./models (project directory)
```

### Cache Issues

**Problem:** Stale cached results, inconsistent outputs

**Solution:**
```bash
# Clear all caches
python -m diaremot.cli run -i audio.wav -o outputs/ --clear-cache

# Or manually delete cache:
rm -rf .cache/
```

## Performance Benchmarks

Approximate processing times on modern CPU (Intel i7-10700, 8 cores):

| Audio Duration | Config | Processing Time | Real-Time Factor |
|----------------|--------|-----------------|------------------|
| 10 minutes | float32, all stages | ~8 minutes | 0.8x |
| 10 minutes | int8, all stages | ~5 minutes | 0.5x |
| 10 minutes | int8, no SED/affect | ~3 minutes | 0.3x |
| 60 minutes | float32, all stages | ~50 minutes | 0.83x |
| 60 minutes | int8, all stages | ~30 minutes | 0.5x |
| 180 minutes | float32, all stages | ~160 minutes | 0.89x |
| 180 minutes | int8, all stages | ~95 minutes | 0.53x |

**Notes:**
- Real-time factor <1.0 means faster than real-time
- Performance varies with audio complexity, number of speakers, speech density
- Auto-chunking engages for files >30 minutes, improving parallelism
- ONNX models (when available) provide 2-5x speedup over PyTorch fallbacks

## Advanced Usage

### Custom Configuration Profiles

Create a JSON file with overrides:

```json
{
  "whisper_model": "faster-whisper-tiny.en",
  "beam_size": 4,
  "temperature": 0.0,
  "compute_type": "int8",
  "vad_threshold": 0.25,
  "ahc_distance_threshold": 0.18,
  "affect_backend": "onnx",
  "enable_sed": true,
  "noise_reduction": true,
  "auto_chunk_enabled": true,
  "chunk_threshold_minutes": 20.0
}
```

Use with:
```bash
python -m diaremot.cli run -i audio.wav -o outputs/ --profile /path/to/custom.json
```

### Batch Processing

```bash
# Process multiple files
for file in data/*.wav; do
  outdir="outputs/$(basename "$file" .wav)"
  python -m diaremot.cli run -i "$file" -o "$outdir"
done
```

### Speaker Registry Management

The speaker registry enables persistent speaker identification across files:

```bash
# Use custom registry location
python -m diaremot.cli run -i audio.wav -o outputs/ \
  --registry-path /path/to/speaker_registry.json

# Registry format (JSON):
{
  "speakers": {
    "SPEAKER_00": {
      "centroid": [0.12, -0.45, ...],  # ECAPA embedding
      "count": 15,                      # Number of occurrences
      "name": "John Doe",              # Optional custom name
      "last_seen": "2024-10-05T10:30:00Z"
    }
  }
}
```

### Integration with Other Tools

Export to JSON for downstream processing:
```bash
# segments.jsonl already in JSONL format
jq -s '.' outputs/segments.jsonl > outputs/segments.json

# Convert CSV to JSON
python -c "
import csv, json
with open('outputs/diarized_transcript_with_emotion.csv') as f:
    print(json.dumps(list(csv.DictReader(f)), indent=2))
" > outputs/transcript.json
```

## Frequently Asked Questions

**Q: Can I use GPU?**
A: No. DiaRemot is CPU-only by design for maximum compatibility and deployment flexibility.

**Q: Why is my speaker registry growing indefinitely?**
A: Set `--speaker-limit N` to cap the number of tracked speakers. Old/rare speakers will be pruned.

**Q: Can I skip paralinguistics to speed up processing?**
A: No. Paralinguistics is a required stage as voice quality metrics are part of the core 39-column schema.

**Q: How do I change the ASR model?**
A: Use `--whisper-model faster-whisper-base.en` or any faster-whisper compatible model ID.

**Q: What's the difference between `run` and `resume`?**
A: `run` starts fresh (optionally using cache). `resume` specifically continues from the last checkpoint after interruption.

**Q: Can I use this on non-English audio?**
A: Partially. ASR supports multiple languages (set `--language`), but emotion/intent models are English-trained.

**Q: Why are there three SNR estimates?**
A: `snr_db` is traditional SNR, `snr_db_sed` is derived from SED noise classification. Use whichever correlates better with perceived quality in your domain.

## License

[Include your license information here]

## Citation

If you use DiaRemot in research, please cite:

```bibtex
[Include citation information if applicable]
```

## Support

For issues, feature requests, or questions:
- GitHub Issues: [repository URL]
- Documentation: [docs URL]
- Email: [support email]
