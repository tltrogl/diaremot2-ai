# AGENTS.md — DiaRemot Agent Instructions (Codex / AI Agents)

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

The canonical stage list is defined in `src/diaremot/pipeline/stages/__init__.py::PIPELINE_STAGES` (lines 13-24):

```python
PIPELINE_STAGES: list[StageDefinition] = [
    StageDefinition("dependency_check", dependency_check.run),
    StageDefinition("preprocess", preprocess.run_preprocess),
    StageDefinition("background_sed", preprocess.run_background_sed),
    StageDefinition("diarize", diarize.run),
    StageDefinition("transcribe", asr.run),
    StageDefinition("paralinguistics", paralinguistics.run),
    StageDefinition("affect_and_assemble", affect.run),
    StageDefinition("overlap_interruptions", summaries.run_overlap),
    StageDefinition("conversation_analysis", summaries.run_conversation),
    StageDefinition("speaker_rollups", summaries.run_speaker_rollups),
    StageDefinition("outputs", summaries.run_outputs),
]
```

**CRITICAL:** This is the authoritative stage list. Count: exactly **11 stages**. There is NO `auto_tune` stage.

### Stage Details

### 1. **dependency_check**
**Module:** `stages/dependency_check.py`  
Validate runtime dependencies:
- `onnxruntime >= 1.16.0`
- `faster-whisper >= 1.0.0` (includes CTranslate2)
- `transformers` (tokenizers only, no inference)
- Praat-Parselmouth

If strict mode enabled, enforces minimum versions. Otherwise logs warnings.

### 2. **preprocess**
**Module:** `stages/preprocess.py::run_preprocess`  
Audio normalization, denoising, auto-chunking (for files >30 min)

**Processing steps:**
1. Load audio (all FFmpeg formats)
2. Resample to 16 kHz mono
3. High-pass filter (80-120 Hz)
4. Optional noise reduction (spectral subtraction)
5. Gated gain for low-RMS speech
6. Gentle compression
7. Loudness normalization (-20 LUFS for ASR)
8. Auto-chunk if duration > 30 minutes

**Outputs:**
- `state.y`: Normalized audio array
- `state.sr`: Sample rate (16000)
- `state.health`: Audio health metrics
- `state.duration_s`: File duration

**Cache management:** Computes audio SHA-16 hash, checks for cached diarization/transcription

### 3. **background_sed** (Sound Event Detection)
**Module:** `stages/preprocess.py::run_background_sed`  
**Model:** PANNs CNN14  
**Runtime:** ONNXRuntime (preferred), PyTorch fallback (`panns_inference` library)  

**Parameters:**
- Frame: 1.0 s
- Hop: 0.5 s
- Thresholds: enter 0.50, exit 0.35
- Min duration: 0.30 s
- Merge gap: 0.20 s
- Label collapse: AudioSet 527 → ~20 semantic groups (speech, music, laughter, crying, door, phone, keyboard, etc.)

**Assets required:**
- ONNX: `panns_cnn14.onnx` (118 MB) + `audioset_labels.csv`
- PyTorch fallback: `Cnn14_mAP=0.431.pth`

**Enabled by default:** Yes. Disable with `--disable-sed`.

**Failure mode:** If models missing or inference fails, logs warning and continues. SED data remains empty but pipeline doesn't crash.

### 4. **diarize** (Speaker Segmentation)
**Module:** `stages/diarize.py`  
**VAD:** Silero VAD  
**Embeddings:** ECAPA-TDNN  
**Clustering:** Agglomerative Hierarchical Clustering (AHC)  
**Runtime:** ONNXRuntime (preferred), PyTorch TorchHub (fallback)

**IMPORTANT: Parameter Configuration Reality**

**CLI Default parameters** (from `src/diaremot/cli.py`):
```python
vad_threshold: 0.30
vad_min_speech_sec: 0.80
vad_min_silence_sec: 0.80
speech_pad_sec: 0.20
ahc_distance_threshold: 0.12
```

**Orchestrator overrides** (from `src/diaremot/pipeline/orchestrator.py::_init_components()` lines 227-242):
Applied when user doesn't override via CLI:
```python
vad_threshold: 0.35          # Stricter to reduce micro-segmentation
vad_min_speech_sec: 0.80     # Same as CLI
vad_min_silence_sec: 0.80    # Same as CLI
speech_pad_sec: 0.10         # Less padding to avoid overlaps
ahc_distance_threshold: 0.15 # Looser to prevent speaker fragmentation
collar_sec: 0.25
min_turn_sec: 1.50
```

**Why the overrides?**
Testing on real-world audio showed:
- Higher VAD threshold (0.35) reduces over-segmentation
- Lower padding (0.10s) prevents turn overlaps
- Higher AHC distance (0.15) prevents false speaker splits

**How users override:** CLI flags take precedence:
```bash
python -m diaremot.cli run -i audio.wav -o outputs/ \
  --vad-threshold 0.30 \
  --ahc-distance-threshold 0.12
```

**Assets required:**
- ONNX: `silero_vad.onnx` (1.8 MB), `ecapa_tdnn.onnx` (6.1 MB)
- PyTorch fallback: TorchHub downloads

**Cache:** If cached diarization available, skips this stage entirely.

### 5. **transcribe** (ASR)
**Module:** `stages/asr.py`  
**Model:** faster-whisper `tiny.en` (39 MB)  
**Runtime:** CTranslate2  

**Default quantization:** **float32** for main CLI  
**Source:** `src/diaremot/cli.py` line 174 shows:
```python
asr_compute_type: str = typer.Option("float32", help="CT2 compute type for faster-whisper.")
```

**NOT int8 as some old docs claimed.** User can override with `--asr-compute-type int8`.

**Parameters:**
- `beam_size = 1` (greedy decoding)
- `temperature = 0.0` (deterministic)
- `no_speech_threshold = 0.50`
- `vad_filter = True` (uses built-in Silero VAD)

**Runs on:** Diarized speech turns only (not full audio)

**Cache:** If cached transcription available, reconstructs turns from cache and skips ASR.

**Chunking:** Long segments (>480s default) processed in windows with overlap to prevent timeout.

### 6. **paralinguistics** (Voice Quality + Prosody)
**Module:** `stages/paralinguistics.py`  
**Runtime:** Praat-Parselmouth (native C++ library)  
**Status:** **Required stage** — cannot be skipped

**Metrics extracted:**

Voice quality (Praat-based):
- `vq_jitter_pct`: Period-to-period pitch variation (voice stability)
- `vq_shimmer_db`: Amplitude variation (voice roughness)
- `vq_hnr_db`: Harmonics-to-Noise Ratio (voice clarity)
- `vq_cpps_db`: Cepstral Peak Prominence Smoothed (voice quality)

Prosodic features:
- `wpm`: Words per minute (speaking rate)
- `pause_count`: Number of pauses detected
- `pause_time_s`: Total pause duration
- `pause_ratio`: Proportion of segment that is pauses
- `f0_mean_hz`: Mean fundamental frequency (pitch)
- `f0_std_hz`: Pitch variability
- `loudness_rms`: RMS loudness
- `disfluency_count`: Filler words (um, uh, like, you know)

**Fallback:** If Praat analysis fails (corrupted segment, extreme noise), computes WPM from ASR text and sets voice quality metrics to 0.0 to prevent pipeline failure.

### 7. **affect_and_assemble** (Emotion + Intent)
**Module:** `stages/affect.py`  

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

**Segment assembly:** Merges audio affect, text affect, paralinguistics, and SED events into final 39-column segment records.

### 8. **overlap_interruptions**
**Module:** `stages/summaries.py::run_overlap`  
Turn-taking analysis, interruption detection, overlap statistics

### 9. **conversation_analysis**
**Module:** `stages/summaries.py::run_conversation`  
Flow metrics (turn-taking balance, response latencies, dominance)

### 10. **speaker_rollups**
**Module:** `stages/summaries.py::run_speaker_rollups`  
Per-speaker summaries (total duration, V/A/D averages, emotion mix, WPM, voice quality)

### 11. **outputs**
**Module:** `stages/summaries.py::run_outputs`  
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

**Migration rule:** Always append new columns to end. Never insert in middle. Provide default values for new columns.

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
Before writing any code:
- **Files touched** (with line numbers where changes will occur)
- **Signatures changed** (function/method signatures)
- **Data shapes / schemas affected** (especially CSV schema)
- **Test plan** (unit tests, integration tests, smoke tests)
- **ONNX model conversion steps** (if adding new models)
- **Breaking changes** (and migration strategy)

Example planning response:
```markdown
Plan:
- Modify `src/diaremot/pipeline/outputs.py` lines 48-49 to add new column
- Update `src/diaremot/pipeline/stages/affect.py` lines 70-75 to populate new field
- Add unit test in `tests/test_outputs.py` lines 150-165
- Add integration test: full pipeline run, verify new column exists
- Breaking change: CSV schema 39 → 40 columns (migration: append to end)
```

### 2. Implement
- **Minimal diff** — touch only necessary code
- **Keep module boundaries** — don't merge unrelated logic
- **Consistent style** — ruff-compliant
- **Prefer ONNX, include PyTorch fallback** for new models
- **No TODO placeholders** — complete all code paths
- **Verify against source** — check actual code, not old docs

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

# Verify outputs
ls /tmp/test/diarized_transcript_with_emotion.csv
head -1 /tmp/test/diarized_transcript_with_emotion.csv | tr ',' '\n' | wc -l
```

### 4. Report (single response)
Include all of these sections:

**Summary:**
- 1-2 paragraph overview of what changed and why

**Source Code References:**
- File paths with line numbers
- Code snippets showing changes
- Before/after comparisons

**Commands Run:**
```bash
$ command1
output1

$ command2  # exit code: 0
output2
```

**Generated Artifacts:**
- List all output files with paths
- Key metrics or interesting findings

**Verification:**
- How you confirmed it works
- Test results
- Edge cases checked

**Risks & Assumptions:**
- What could break
- What's untested
- Dependencies on other changes

**Follow-up:**
- Recommended next steps
- Additional testing needed
- Documentation updates required

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
7. Document in AI_INDEX.yaml

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

---

## Reporting Checklist (always include)

- ✅ Only factual, reproducible changes; no hallucinated logs
- ✅ Ruff / lint / tests passed (report summary)
- ✅ Full pipeline run (all 11 stages) completed
- ✅ No broken stage; no regression introduced
- ✅ Schema maintained or extended forward-compatibly
- ✅ SED label collapse preserved
- ✅ ONNX models validated with ONNXRuntime
- ✅ PyTorch fallback tested when ONNX unavailable
- ✅ All assumptions, risks, version bumps, file paths documented
- ✅ No private credentials or secrets in artifacts or logs
- ✅ Cited source code with file paths and line numbers

---

## Example Directive & Expected Planning Style

> **Directive:** "Add zero-shot emotion classification using ONNX model with HuggingFace fallback."

Your plan response should look like:

```markdown
Plan:
1. **Files to modify:**
   - `src/diaremot/affect/text_analyzer.py` (lines 45-80): Add ONNX inference path
   - `src/diaremot/affect/text_analyzer.py` (lines 110-130): Add HF fallback
   - `src/diaremot/pipeline/config.py` (lines 25-30): Add ONNX model path config
   - `tests/test_text_analyzer.py` (lines 60-95): Add ONNX/HF fallback tests

2. **ONNX conversion steps:**
   - Convert `roberta-base-go_emotions` to ONNX using optimum-cli
   - Command: `optimum-cli export onnx --model SamLowe/roberta-base-go_emotions --task text-classification --opset 14 ./models/roberta-onnx/`
   - Expected output: model.onnx (~500 MB)

3. **Implementation approach:**
   - Primary: Load ONNX model via ONNXRuntime
   - Tokenize text using HuggingFace tokenizer (fast, no inference)
   - Run ONNX inference on token IDs
   - Fallback: If ONNX load fails, use HuggingFace transformers pipeline()
   - Benchmark: Expect 2-3x speedup vs HF pipeline

4. **Testing plan:**
   - Unit test: ONNX inference matches HF pipeline output (tolerance: 0.01)
   - Unit test: Fallback triggers correctly when ONNX missing
   - Integration test: Full pipeline run on sample audio, verify CSV schema unchanged
   - Performance test: Benchmark ONNX vs HF on 100 segments

5. **Breaking changes:**
   - None (backward compatible)
   - CSV schema unchanged (39 columns)
   - If ONNX unavailable, falls back to existing HF path

6. **Documentation updates:**
   - `AI_INDEX.yaml`: Add ONNX model entry
   - `README.md`: Document ONNX model path requirement
   - `CLAUDE.md`: Update model list with ONNX/HF backends
```

Then implement, verify, and report with actual logs and results.

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
11. **Wrong default compute_type** → Main CLI uses float32, not int8
12. **Wrong stage count** → 11 stages, not 12
13. **Fabricating logs** → Only report what actually executed
14. **Not citing source code** → Always provide file paths and line numbers

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

**Then:**
1. Document the exact error + reproducible test case
2. Check ONNX/optimum GitHub issues for known bugs
3. Try different opset versions (11, 12, 14, 16)
4. Consider quantization-aware training if int8 fails
5. Flag for human review if unresolvable

---

**Remember:** This is production code serving real users. Precision matters. ONNX-preferred with PyTorch fallback is the architecture. Always verify source code before claiming. Cite file paths and line numbers. When in doubt, check the actual implementation in `src/`.
