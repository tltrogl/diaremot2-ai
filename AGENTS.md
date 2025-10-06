# AGENTS.md - DiaRemot Agent Instructions (Codex / AI Agents)

IMPORTANT for Codex Cloud runs
- Use AGENTS_CLOUD.md instead of this file when executing on Codex Cloud. It includes Codex‑specific allowances (apt for ffmpeg) and the exact pinned dependency versions to install from requirements.txt. This AGENTS.md is the general policy.


_Last updated: 2025-10-05_

**Role:** System Architect / Maintainer for DiaRemot  
As the agent, you must **plan → implement → verify → report** in each change cycle. You are building *real code*, not mocks.  

---

## Truth & Integrity (non-negotiable)
- Only produce **correct, factual, non-fabricated** outputs.
- Do **not simulate** logs or results; only what you actually ran.
- If uncertain, state it and propose concrete diagnostic tests.
- Internet is **ON**: you may research; cite sources or include "source of truth" notes.
- Do not leak any secrets, credentials, or private links.

---

## Environment & Shell
- Execution environment: **Codex Cloud (ephemeral)** — filesystem resets; cache only under `./.cache/`.
- Primary shell: **bash**. (You may generate Windows/PowerShell variants when needed.)
- Install dependencies via `pip install -r requirements.txt`. Do not rely on `apt` or system packages.

### Required Environment Variables
These must be defined (or defaulted) before executing:
```
DIAREMOT_MODEL_DIR
HF_HOME
HUGGINGFACE_HUB_CACHE
TRANSFORMERS_CACHE
TORCH_HOME
OMP_NUM_THREADS
MKL_NUM_THREADS
NUMEXPR_MAX_THREADS
TOKENIZERS_PARALLELISM = false
```

---

## Pipeline Architecture: ONNX-Preferred with PyTorch Fallback

**DiaRemot is CPU-only** with the following inference strategy:

### Primary Stack (Preferred)
- **ONNXRuntime** for all audio/text models (VAD, embeddings, emotion, intent)
- **CTranslate2** for ASR (faster-whisper tiny.en)
- **librosa/scipy/numpy** for audio preprocessing/feature extraction
- **Praat-Parselmouth** for voice quality analysis

### Fallback Stack (When ONNX Unavailable)
- **PyTorch CPU** for model inference if ONNX models missing:
  - Silero VAD (TorchHub)
  - PANNs SED (`panns_inference` library)
  - Emotion models (HuggingFace transformers)
- **HuggingFace `pipeline()`** for text models if ONNX unavailable

### Preprocessing (No PyTorch)
All audio preprocessing uses:
- **librosa** - mel-spectrograms, MFCC, resampling
- **scipy** - filtering, signal processing
- **numpy** - array operations, normalization
- **soundfile** - audio I/O

**PyTorch is NOT used for preprocessing.** If you see torch imports in preprocessing code, that's a bug.

### Why ONNX-Preferred?
- 2-5x faster CPU inference vs PyTorch
- Lower memory footprint for 1-3 hour audio files
- No torch.jit complexity
- Smaller deployment size when ONNX-only

### Why Keep PyTorch Fallback?
- Graceful degradation when ONNX models unavailable
- Development/testing without ONNX conversion
- Backward compatibility with existing pipelines

### Agent Guidelines
**When adding new models:**
1. **Primary path:** Implement ONNX inference first
2. **Fallback path:** Add PyTorch option for robustness
3. **Prefer ONNX** but don't break pipeline if unavailable
4. **Never use PyTorch for preprocessing** - use librosa/scipy

---

## Pipeline Contract (must remain true)

**Every run should include all 11 stages by default**, unless explicitly overridden.

The canonical stage list is defined in `src/diaremot/pipeline/stages/__init__.py::PIPELINE_STAGES`:

### 1. **dependency_check**
Validate runtime dependencies:
- `onnxruntime >= 1.16.0`
- `faster-whisper >= 1.0.0` (includes CTranslate2)
- `transformers` (tokenizers only, no inference)
- Praat-Parselmouth

### 2. **preprocess**
Audio normalization, denoising, auto-chunking (for files >30 min)

**Config:** `PreprocessConfig` in `audio_preprocessing.py`
- `target_sr`: 16000
- `denoise`: "spectral_sub_soft" | "none"
- `loudness_mode`: "asr" (normalize to -20 LUFS)
- `auto_chunk_enabled`: true
- `chunk_threshold_minutes`: 30.0

### 3. **background_sed** (Sound Event Detection)
**Model:** PANNs CNN14  
**Runtime:** ONNXRuntime (preferred), PyTorch fallback (`panns_inference` library)  
**Parameters:**
- Frame: 1.0 s
- Hop: 0.5 s
- Thresholds: enter 0.50, exit 0.35
- Min duration: 0.30 s
- Merge gap: 0.20 s
- Label collapse: AudioSet 527 → ~20 semantic groups (speech, music, laughter, crying, door, phone, etc.)

**Enabled by default.** User must explicitly `--disable-sed` to skip.

**Assets required:**
- ONNX: `panns_cnn14.onnx` (118 MB) + `audioset_labels.csv`
- PyTorch fallback: `Cnn14_mAP=0.431.pth`

### 4. **diarize** (Speaker Segmentation)
**VAD:** Silero VAD  
**Embeddings:** ECAPA-TDNN  
**Clustering:** Agglomerative Hierarchical Clustering (AHC)  
**Runtime:** ONNXRuntime (preferred), PyTorch TorchHub (fallback)

**CLI Default parameters** (`src/diaremot/cli.py`):
- `vad_threshold = 0.30`
- `vad_min_speech_sec = 0.80`
- `vad_min_silence_sec = 0.80`
- `speech_pad_sec = 0.20`
- `ahc_distance_threshold = 0.12`

**Orchestrator overrides** (`src/diaremot/pipeline/orchestrator.py::_init_components()`, lines 234-244):
Applied when user doesn't set CLI flags:
- `vad_threshold = 0.35` (stricter to reduce oversegmentation)
- `vad_min_speech_sec = 0.80` (same)
- `vad_min_silence_sec = 0.80` (same)
- `speech_pad_sec = 0.10` (less padding to avoid overlap)
- `ahc_distance_threshold = 0.15` (looser to prevent speaker fragmentation)
- `collar_sec = 0.25`
- `min_turn_sec = 1.50`

**User can override orchestrator tuning:**
```bash
python -m diaremot.cli run -i audio.wav -o outputs/ \
  --vad-threshold 0.30 \
  --ahc-distance-threshold 0.12
```

**Assets required:**
- ONNX: `silero_vad.onnx` (1.8 MB), `ecapa_tdnn.onnx` (6.1 MB)
- PyTorch fallback: TorchHub downloads

### 5. **transcribe** (ASR)
**Model:** faster-whisper `tiny.en` (39 MB)  
**Runtime:** CTranslate2 (float32 default for main CLI, int8 optional)  
**Parameters:**
- `beam_size = 1` (greedy decoding)
- `temperature = 0.0` (deterministic)
- `no_speech_threshold = 0.50`
- `compute_type = float32` (default for main CLI; override with `--asr-compute-type int8`)
- `vad_filter = True` (uses built-in Silero VAD)
- `max_asr_window_sec = 480` (8 minutes)

**Runs on:** Diarized speech turns only (not full audio)

### 6. **paralinguistics** (Voice Quality + Prosody)
**Runtime:** Praat-Parselmouth (native C++ library)  
**Metrics extracted:**
- Voice quality: jitter (%), shimmer (dB), HNR (dB), CPPS (dB)
- Prosody: WPM, duration_s, words, pause_count, pause_time_s, pause_ratio
- Pitch: f0_mean_hz, f0_std_hz
- Loudness: loudness_rms
- Disfluencies: disfluency_count

**Fallback:** If Praat fails, compute WPM from ASR text and set voice quality metrics to 0.0

**REQUIRED STAGE:** Cannot be skipped. Must populate all 14 paralinguistic fields in CSV.

### 7. **affect_and_assemble** (Emotion + Intent)
**Audio emotion:** 8-class Speech Emotion Recognition  
**VAD emotion:** Valence/Arousal/Dominance  
**Text emotion:** GoEmotions 28-class  
**Intent:** Zero-shot classification (BART-MNLI)  
**Runtime:** ONNXRuntime (preferred), HuggingFace transformers (fallback)

**Assets required (ONNX):**
- `ser_8class.onnx` (audio emotion)
- `vad_model.onnx` (valence/arousal/dominance)
- `roberta-base-go_emotions.onnx` (text emotion)
- `bart-large-mnli.onnx` (intent classification)

**Fallback:** HuggingFace transformers `pipeline()` if ONNX unavailable

**Assembles:** Final segment dicts with all 39 CSV columns

### 8. **overlap_interruptions**
Turn-taking analysis, interruption detection, overlap statistics

**Outputs:**
- `overlap_stats`: Dict with overlap_count, total_overlap_duration_s
- `per_speaker_interrupts`: Dict mapping speaker_id → {made, received}

### 9. **conversation_analysis**
Flow metrics (turn-taking balance, response latencies, dominance)

**Outputs:**
- `conv_metrics`: ConversationMetrics object

### 10. **speaker_rollups**
Per-speaker summaries (total duration, V/A/D averages, emotion mix, WPM, voice quality)

**Outputs:**
- `speakers_summary`: List of dicts with speaker-level aggregates

### 11. **outputs**
Write final files:
- `diarized_transcript_with_emotion.csv` (39 columns)
- `segments.jsonl`
- `speakers_summary.csv`
- `summary.html`
- `summary.pdf` (optional, requires wkhtmltopdf)
- `speaker_registry.json`
- `events_timeline.csv` & `events.jsonl`
- `timeline.csv`, `qc_report.json`

---

## Schema Contract

**Canonical segment schema:** `src/diaremot/pipeline/outputs.py::SEGMENT_COLUMNS` (39 columns)

```python
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

**You must conform exactly to these column names unless extending forward-compatibly.**

---

## Model Assets / File Paths

All ONNX models expected under `DIAREMOT_MODEL_DIR`:

```
$DIAREMOT_MODEL_DIR/
├── panns_cnn14.onnx              # SED (118 MB)
├── audioset_labels.csv            # SED labels (527 classes)
├── silero_vad.onnx                # VAD (1.8 MB)
├── ecapa_tdnn.onnx                # Speaker embeddings (6.1 MB)
├── ser_8class.onnx                # Audio emotion (size varies)
├── vad_model.onnx                 # V/A/D emotion (size varies)
├── roberta-base-go_emotions.onnx  # Text emotion (~500 MB)
└── bart-large-mnli.onnx           # Intent classification (~1.6 GB)
```

**CTranslate2 models** (faster-whisper downloads automatically to HF cache):
- `tiny.en` (39 MB) — default ASR model

**PyTorch fallback models** (auto-downloaded when ONNX unavailable):
- TorchHub: Silero VAD
- `panns_inference`: PANNs CNN14
- HuggingFace: emotion/intent models

**Missing assets should:**
1. Log warning (not silent failure)
2. Fallback to PyTorch if available
3. If PyTorch also unavailable, disable that stage gracefully
4. Never crash the pipeline

---

## ONNX Runtime Configuration

**CPU optimization flags:**
```python
import onnxruntime as ort

session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session_options.intra_op_num_threads = min(4, os.cpu_count())
session_options.inter_op_num_threads = 1
session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

providers = ['CPUExecutionProvider']
session = ort.InferenceSession(model_path, sess_options=session_options, providers=providers)
```

**CRITICAL:** Do NOT use GPUExecutionProvider. DiaRemot is CPU-only.

---

## CLI / Entry Point Contract

**Do not break or rename:**
- `python -m diaremot.pipeline.run_pipeline`
- `python -m diaremot.pipeline.cli_entry`
- `python -m diaremot.cli` (Typer app)

**Example CLI command:**
```bash
python -m diaremot.cli run \
  --input audio.wav \
  --outdir ./results \
  --vad-threshold 0.35 \
  --asr-compute-type float32
```

---

## Operating Procedure (Plan→Implement→Verify→Report)

### 1. Plan (5-10 bullets)
- Files touched
- Signatures changed
- Data shapes / schemas affected
- Test plan
- ONNX model conversion steps (if adding new models)
- VAD parameter impact (if modifying diarization)

### 2. Implement
- Minimal diff
- Keep module boundaries
- Consistent style (ruff-compliant)
- **Prefer ONNX, include PyTorch fallback**
- **Respect orchestrator overrides** — don't accidentally remove them

### 3. Verify
```bash
# Lint
ruff check src/ tests/

# Type check (if using mypy)
mypy src/diaremot/

# Unit tests
pytest tests/ -v

# Integration test (smoke run)
python -m diaremot.cli run --input data/sample.wav --outdir /tmp/test

# Verify CSV schema
python -c "from diaremot.pipeline.outputs import SEGMENT_COLUMNS; print(len(SEGMENT_COLUMNS))"  # Should be 39
```

### 4. Report (single response)
- Short summary (1-2 paragraphs)
- Diffs/patch list
- Commands run + exit codes
- Key logs (tail ~200 lines)
- Generated artifact paths (CSV, HTML, JSON)
- Risks, assumptions, follow-up notes

**If any stage fails, fix before reporting.** Do not produce incomplete code or half-baked logs.

---

## Research / Dependency Guidelines

- Use **official docs / primary sources** (ONNX Model Zoo, HuggingFace, CTranslate2 docs)
- Summarize findings succinctly with "Source of truth" notes
- Do not quietly download multi-GB assets; require explicit caching
- When pinning or updating dependencies, show exact `pip` resolution, version before/after, and justify trade-offs

**Example: Adding a new ONNX model**
1. Document source (HuggingFace repo, ONNX Model Zoo URL)
2. Show conversion command (e.g., `optimum-cli export onnx ...`)
3. Verify model works with ONNXRuntime
4. Add PyTorch fallback for robustness
5. Benchmark inference time (CPU-only)
6. Add to `DIAREMOT_MODEL_DIR` manifest

---

## Hard Constraints & Safety

- **No GPU usage** — CPU-only execution
- **No system-level installs** (apt, brew)
- **No secrets/keys** in code or logs
- **Must preserve behavior** of existing functions and modules (extend, not remove)
- **ASR default:** `compute_type = float32` for main CLI (not int8)
- **11 stages exactly** — do not add/remove stages without updating PIPELINE_STAGES
- **Do not rename or break output schemas** or filenames without coordinated migration steps
- **All agent actions must be auditably reproducible** — logs and commands must align with outputs
- **ONNX-preferred, PyTorch fallback** for all models except ASR (CTranslate2) and paralinguistics (Praat)
- **Orchestrator VAD overrides** — must remain when user doesn't set CLI flags

---

## Reporting Checklist (always include)

- ✅ Only factual, reproducible changes; no hallucinated logs
- ✅ Ruff / lint / tests passed (report summary)
- ✅ Full pipeline run (all 11 stages) completed
- ✅ No broken stage; no regression introduced
- ✅ Schema maintained or extended forward-compatibly (39 columns)
- ✅ SED label collapse preserved
- ✅ ONNX models validated with ONNXRuntime
- ✅ PyTorch fallback tested when ONNX unavailable
- ✅ Orchestrator VAD overrides preserved (if modifying diarization)
- ✅ All assumptions, risks, version bumps, file paths documented
- ✅ No private credentials or secrets in artifacts or logs

---

## Example Directive & Expected Planning Style

> **Directive:** "Add zero-shot emotion classification using ONNX model with HuggingFace fallback."

Your plan response might look like:

```
Plan:
- Convert HuggingFace `roberta-base-go_emotions` to ONNX using optimum-cli
- Modify `affect/text_analyzer.py` to:
  - Primary: Load ONNX model via ONNXRuntime
  - Fallback: Use HuggingFace transformers pipeline() if ONNX missing
- Tokenize text using HuggingFace tokenizer (fast, no inference)
- Update `pipeline/config.py` to reference ONNX model path
- Benchmark inference time: HF pipeline vs ONNX (expect 2-3x speedup)
- Write unit tests:
  - ONNX inference matches HF pipeline output
  - Fallback triggers correctly when ONNX missing
- Integration test: run full pipeline on sample audio, verify CSV schema unchanged (39 columns)
- Document ONNX conversion steps in README
- Lint / typecheck / build

Then implement, verify, report with logs/patches.
```

---

## Common Pitfalls to Avoid

1. **Using HuggingFace `pipeline()` as primary path** → Use ONNX first, pipeline() as fallback
2. **Loading PyTorch models without ONNX alternative** → Always provide both paths
3. **Not validating ONNX model outputs** → Always test against reference implementation
4. **Silent failures when both ONNX and PyTorch missing** → Log warning and disable stage gracefully
5. **Hardcoding model paths** → Use `DIAREMOT_MODEL_DIR` environment variable
6. **Not benchmarking ONNX conversion** → Always compare inference time vs PyTorch
7. **Breaking schema when adding ONNX models** → Maintain 39-column CSV contract
8. **Using PyTorch for preprocessing** → Use librosa/scipy/numpy instead
9. **Claiming auto_tune is a stage** → It's NOT in PIPELINE_STAGES; orchestrator applies tuning inline
10. **Forgetting orchestrator overrides VAD params** → Check `orchestrator.py::_init_components()` for actual values
11. **Removing orchestrator VAD overrides** → They exist for good reason (reduce oversegmentation)
12. **Wrong compute_type default** → Main CLI uses float32, not int8

---

## Quick Reference: ONNX Conversion Commands

### HuggingFace to ONNX (using optimum-cli)
```bash
# Text classification (e.g., GoEmotions)
optimum-cli export onnx \
  --model SamLowe/roberta-base-go_emotions \
  --task text-classification \
  --opset 14 \
  ./models/roberta-base-go_emotions-onnx/

# Zero-shot classification (e.g., BART-MNLI)
optimum-cli export onnx \
  --model facebook/bart-large-mnli \
  --task zero-shot-classification \
  --opset 14 \
  ./models/bart-large-mnli-onnx/
```

### PyTorch Audio Models to ONNX
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

## When to Escalate

If you encounter:
- ONNX conversion failures (opset compatibility, dynamic shapes)
- ONNXRuntime errors (shape mismatches, unsupported ops)
- Significant accuracy degradation vs PyTorch (>5% difference)
- Performance regression (ONNX slower than PyTorch)
- Orchestrator VAD overrides causing issues on specific audio types

**Then:**
1. Document the exact error + reproducible test case
2. Check ONNX/optimum GitHub issues for known bugs
3. Try different opset versions (11, 12, 14, 16)
4. Consider quantization-aware training if int8 fails
5. For VAD issues, document audio characteristics (SNR, speech style, etc.)
6. Flag for human review if unresolvable

---

**Remember:** This is production code serving real users. Precision matters. ONNX-preferred with PyTorch fallback is the architecture. Orchestrator VAD overrides exist to reduce oversegmentation. When in doubt, verify before claiming.
