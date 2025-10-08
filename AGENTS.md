# AGENTS.md — DiaRemot AI Assistant Instructions

**Last Updated:** 2025-10-08  
**Verified Against:** Live codebase at D:\diaremot\diaremot2-ai  
**Role:** You execute user instructions with precision. No autonomy, no assumptions.

---

## Rule Zero: Listen to the User

Before doing **anything**:

1. Read the complete user instruction
2. Ask clarifying questions if ambiguous (don't guess)
3. Write a 5-10 bullet plan showing your approach
4. Execute with surgical precision (minimal diffs)
5. Verify everything (lint, test, smoke run)
6. Report factually (files changed, commands run, real outputs only)

**You are the assistant. The user is the architect.** Execute their vision.

---

## What DiaRemot Actually Is

**Purpose:** CPU-only speech intelligence pipeline that processes 1-3 hour audio files into diarized transcripts with comprehensive acoustic and linguistic analysis.

**Core Technology Stack:**
- Python 3.11
- **ONNXRuntime 1.17.1** (primary inference engine)
- **CTranslate2 4.6.0 + faster-whisper 1.1.0** (ASR)
- PyTorch 2.4.1+cpu (minimal use, fallback only)
- **Praat-Parselmouth 0.4.3** (voice quality analysis)
- librosa 0.10.2.post1, scipy, numpy (audio preprocessing)

**Primary Output:** `diarized_transcript_with_emotion.csv` (39 columns, never modify schema without migration)

---

## Pipeline Architecture: Exactly 11 Stages

**Source:** `src/diaremot/pipeline/stages/__init__.py::PIPELINE_STAGES`

1. **dependency_check** — Verify runtime dependencies exist and meet version requirements
2. **preprocess** — Audio normalization (16kHz, -20 LUFS), optional denoising, auto-chunk if >30min
3. **background_sed** — Sound event detection (PANNs CNN14 ONNX, 527 AudioSet classes → ~20 groups)
4. **diarize** — Speaker segmentation (Silero VAD → ECAPA-TDNN embeddings → AHC clustering)
5. **transcribe** — Speech-to-text (faster-whisper tiny.en via CTranslate2)
6. **paralinguistics** — Voice quality (Praat: jitter/shimmer/HNR/CPPS) + prosody (WPM, F0, pauses)
7. **affect_and_assemble** — Audio emotion (V/A/D + 8-class SER) + text emotion (GoEmotions 28) + intent
8. **overlap_interruptions** — Turn-taking analysis, interruption detection
9. **conversation_analysis** — Flow metrics, dominance patterns, engagement scores
10. **speaker_rollups** — Per-speaker aggregations across all segments
11. **outputs** — Generate CSV, JSONL, HTML, PDF, QC reports

**CRITICAL FACTS:**
- `auto_tune.py` exists but is **NOT a pipeline stage** (VAD tuning happens inline in orchestrator)
- Paralinguistics stage is **required** and must never fail silently
- SED is **enabled by default** (disable with `--disable-sed`)
- Schema has **exactly 39 columns** (breaking changes require migration plan)

---

## The Diarization Two-Stage Strategy

**Design Goal:** Detect quiet speech without creating over-segmentation.

This is solved through a **two-stage detect-then-consolidate approach**, not a single parameter.

### Stage 1: Detection (VAD Threshold)

**Lower threshold = More sensitive = Catches more speech**

The VAD threshold controls what the system accepts as "probably speech." When you set a lower threshold like 0.22-0.25, you're casting a wide net that catches quiet utterances, hesitations, soft-spoken sections, and trailing words. This inevitably creates many small fragments because the VAD fires on every little phrase bounded by pauses.

**Trade-off:** Lower thresholds catch quiet speech but create more fragments. Higher thresholds (0.35+) miss quiet speech entirely but create fewer fragments.

**Orchestrator default:** `vad_threshold = 0.22` (generous detection to catch quiet speech, including soft-spoken sections)

### Stage 2: Consolidation (AHC Clustering)

**Higher threshold = More forgiving = Merges more fragments**

After detection creates fragments, the AHC clustering decides which fragments belong to the same speaker. A higher distance threshold (0.15-0.20) means the system is more willing to say "these fragments sound similar enough to be the same person" and merge them together.

**Trade-off:** Higher AHC thresholds merge fragments from the same speaker (fixing over-segmentation) but risk merging different speakers. Lower AHC thresholds (0.02-0.05) keep fragments separate, creating massive over-segmentation even for a single speaker.

**Orchestrator default:** `ahc_distance_threshold = 0.18` (forgiving clustering to merge fragments into coherent turns)

### Why This Combination Works

**Loose detection (0.22) + Loose clustering (0.18) = Catch quiet speech, then merge fragments intelligently**

The VAD detects even quiet utterances as separate fragments. The clustering then recognizes these fragments came from the same speaker and merges them back into continuous turns. This solves the core problem: you get the quiet speech without the fragmentation.

**Alternative combinations and why they fail:**
- **Strict detection (0.35) + Loose clustering (0.18):** Misses quiet speech entirely, so clustering has nothing to merge
- **Loose detection (0.22) + Strict clustering (0.02):** Catches quiet speech but refuses to merge fragments, massive over-segmentation
- **Strict detection (0.35) + Strict clustering (0.02):** Worst of both worlds, misses speech and fragments what it finds

### Additional Safeguards

**Minimum durations filter out noise blips:**
- `vad_min_speech_sec = 0.40` — Ignore very short bursts (breathing, clicks)
- `vad_min_silence_sec = 0.40` — Don't split on tiny pauses within a thought

**Modest padding avoids artificial overlaps:**
- `speech_pad_sec = 0.15` — Small buffer around speech without creating segment collisions

**Post-processing cleans up the timeline:**
- `collar_sec = 0.25` — Trim segment edges to avoid micro-overlaps
- `min_turn_sec = 1.50` — Merge very short turns from same speaker
- `max_gap_to_merge_sec = 1.00` — Bridge small silences between same-speaker turns

---

## Entry Points and CLI Behavior

**Three ways to run the pipeline:**
1. `python -m diaremot.cli run` (main Typer CLI app, **defaults to float32**)
2. `python -m diaremot.pipeline.run_pipeline` (direct orchestrator call)
3. `python -m diaremot.pipeline.cli_entry` (legacy argparse CLI)

### Critical CLI Defaults

**ASR compute type:**
- **Main CLI default:** `float32` (higher accuracy, slower)
- Override with: `--asr-compute-type int8` (2x faster, <2% WER penalty)
- **NEVER claim default is int8** — that's factually wrong

**Diarization parameters** (CLI defaults in `src/diaremot/cli.py`):
```python
vad_threshold: 0.30          # Base CLI default
vad_min_speech_sec: 0.80     # Base CLI default
vad_min_silence_sec: 0.80    # Base CLI default  
speech_pad_sec: 0.20         # Base CLI default
ahc_distance_threshold: 0.12 # Base CLI default
```

**Orchestrator overrides** (in `orchestrator.py::_init_components`, applied when user doesn't set CLI flags):
```python
vad_threshold: 0.22           # Looser (catch quiet speech)
vad_min_speech_sec: 0.40      # Shorter (accept brief utterances)
vad_min_silence_sec: 0.40     # Shorter (don't split on tiny pauses)
speech_pad_sec: 0.15          # Modest (avoid overlap)
ahc_distance_threshold: 0.18  # Looser (merge fragments from same speaker)
```

**To override orchestrator tuning:**
```bash
python -m diaremot.cli run -i audio.wav -o outputs/ \
  --vad-threshold 0.30 \
  --ahc-distance-threshold 0.12
```

---

## CSV Schema: 39 Columns (Sacred)

**Source:** `src/diaremot/pipeline/outputs.py::SEGMENT_COLUMNS`

**Primary file:** `diarized_transcript_with_emotion.csv`

**Column groups:**
- **Temporal:** file_id, start, end, duration_s
- **Speaker:** speaker_id, speaker_name
- **Content:** text, words, language
- **ASR metrics:** asr_logprob_avg (confidence), low_confidence_ser flag
- **Audio emotion:** valence, arousal, dominance, emotion_top, emotion_scores_json
- **Text emotion:** text_emotions_top5_json, text_emotions_full_json
- **Intent:** intent_top, intent_top3_json
- **Sound events:** events_top3_json, noise_tag
- **Voice quality (Praat):** vq_jitter_pct, vq_shimmer_db, vq_hnr_db, vq_cpps_db, voice_quality_hint
- **Prosody:** wpm, pause_count, pause_time_s, pause_ratio, f0_mean_hz, f0_std_hz, loudness_rms, disfluency_count
- **Signal quality:** snr_db, snr_db_sed, vad_unstable flag
- **Hints:** affect_hint, error_flags

**NEVER modify this schema without:**
1. Migration plan for existing CSV files
2. Appending new columns to the end only (never insert/reorder)
3. Updating `ensure_segment_keys()` default values
4. Writing unit test verifying column count
5. Documenting in README and AGENTS.md

---

## Key Dependencies (Exact Versions Required)

**From `requirements.txt`:**
```
onnxruntime==1.17.1          # Primary inference engine
faster-whisper==1.1.0        # ASR
ctranslate2==4.6.0           # ASR backend
torch==2.4.1+cpu             # Fallback only
transformers==4.38.2         # Text models
praat-parselmouth==0.4.3     # Voice quality
librosa==0.10.2.post1        # Audio I/O
scipy==1.10.1                # Signal processing
numpy==1.24.4                # Arrays
```

**Do NOT upgrade versions without explicit user approval and testing.**

---

## Environment Variables (Required)

```bash
DIAREMOT_MODEL_DIR           # Model directory (ONNX files go here)
HF_HOME                      # HuggingFace cache
TRANSFORMERS_CACHE           # Transformers cache  
TORCH_HOME                   # PyTorch cache
OMP_NUM_THREADS=4            # OpenMP parallelism
MKL_NUM_THREADS=4            # Intel MKL threads
NUMEXPR_MAX_THREADS=4        # NumPy threads
TOKENIZERS_PARALLELISM=false # Disable tokenizer warnings
```

---

## Models and Inference Backends

### ONNX-First Strategy

DiaRemot prefers ONNX models (via ONNXRuntime) for 2-5x faster CPU inference vs PyTorch. Fallback to PyTorch/HuggingFace only when ONNX unavailable.

**ONNX models expected in `$DIAREMOT_MODEL_DIR`:**
- `panns_cnn14.onnx` (118 MB) — SED
- `audioset_labels.csv` — SED label mapping (527 classes)
- `silero_vad.onnx` (1.8 MB) — VAD
- `ecapa_tdnn.onnx` (6.1 MB) — Speaker embeddings
- `ser_8class.onnx` — Audio emotion (8-class SER)
- `vad_model.onnx` — V/A/D emotion
- `roberta-base-go_emotions.onnx` (~500 MB) — Text emotion (28 classes)
- `bart-large-mnli.onnx` (~1.6 GB) — Intent classification

**Auto-download models (HuggingFace cache):**
- faster-whisper tiny.en (39 MB, CTranslate2 format)

**PyTorch fallbacks (TorchHub / transformers):**
- Silero VAD: `snakers4/silero-vad`
- PANNs: `panns_inference` library
- Emotion/intent: HuggingFace models when ONNX missing

---

## Hard Constraints (Never Break These)

1. **Schema stability:** 39 columns is sacred, extend only with migration
2. **Entry points:** Never rename `cli.py::app`, `run_pipeline`, `cli_entry`
3. **CPU-only:** No GPU code, no CUDA dependencies
4. **11 stages:** Exactly this count in PIPELINE_STAGES
5. **Main CLI ASR default:** `compute_type=float32` (NOT int8)
6. **Paralinguistics required:** Stage must never be skipped silently
7. **SED default-enabled:** `enable_sed=True` unless user explicitly disables
8. **No PyTorch in preprocessing:** Use librosa/scipy/numpy only
9. **Diarization two-stage strategy:** Maintain loose detection + loose clustering for quiet speech detection

---

## Development Workflow (Required Process)

### 1. Planning Phase
Before writing any code:
- List files you'll modify with line numbers if possible
- Describe data structure changes
- Outline test strategy
- Identify breaking changes
- Document VAD/clustering parameter effects if modifying diarization

### 2. Implementation Phase
- **Minimal diffs** — change only what's necessary
- **Preserve module boundaries** — don't merge unrelated logic
- **Match existing style** — follow ruff conventions
- **No placeholder TODOs** — complete all code paths
- **Respect orchestrator overrides** — understand when they apply vs CLI defaults

### 3. Verification Phase (Mandatory)

```bash
# Lint code
ruff check src/ tests/

# Type check (if mypy configured)
mypy src/diaremot/

# Unit tests
pytest tests/ -v

# Smoke test (if sample audio available)
python -m diaremot.cli run --input data/sample.wav --outdir /tmp/smoke_test/

# Verify schema unchanged
python -c "from diaremot.pipeline.outputs import SEGMENT_COLUMNS; print(len(SEGMENT_COLUMNS))"  # Must be 39
```

### 4. Reporting Phase

**Include in every response:**
- **Summary:** 1-2 paragraphs of what changed and why
- **Files modified:** List with line counts (`+15 lines`, etc.)
- **Commands executed:** With actual exit codes
- **Key logs:** Real output from commands (tail ~50 lines relevant parts)
- **Artifact paths:** Files generated with actual paths
- **Risks/assumptions:** What could break, what's untested, edge cases
- **Follow-up:** Recommended next steps or additional testing needed

**Example:**
```
## Summary
Fixed orchestrator AHC clustering threshold from 0.02 to 0.18 to prevent over-segmentation
while maintaining ability to detect quiet speech with existing loose VAD threshold (0.22).

## Files Modified
- src/diaremot/pipeline/orchestrator.py (+1 line) — Changed ahc_distance_threshold default from 0.02 to 0.18

## Commands Executed
ruff check src/ tests/  # exit 0
pytest tests/test_diarization.py -v  # 12 passed

## Risks
- May slightly increase risk of merging different speakers with very similar voices
- Should test on multi-speaker recordings to verify behavior
- No regression in existing test cases

## Follow-up
- Test on real-world multi-speaker audio
- Monitor for false speaker merges in production
```

---

## Common Pitfalls to Avoid

1. **Claiming auto_tune is a stage** — It's NOT in PIPELINE_STAGES (VAD tuning is inline)
2. **Wrong VAD defaults** — Orchestrator uses 0.22 (loose), not CLI's 0.30 or mythical 0.35
3. **Wrong compute_type** — Main CLI defaults to float32, NOT int8
4. **Wrong stage count** — 11 stages, not 12
5. **Modifying schema carelessly** — Always append columns, never insert/reorder
6. **Ignoring orchestrator overrides** — Check `_init_components()` for actual runtime values
7. **Breaking entry points** — `cli.py::app` must remain as Typer app instance
8. **Skipping paralinguistics** — Stage is required, must never fail silently
9. **Using PyTorch for preprocessing** — Use librosa/scipy/numpy exclusively
10. **Forgetting SED is default-enabled** — Must be explicitly disabled with `--disable-sed`
11. **Misunderstanding VAD threshold direction** — Lower = more sensitive, higher = stricter
12. **Misunderstanding AHC threshold direction** — Higher = merge more, lower = fragment more

---

## SED Parameters (Background Sound Event Detection)

**Model:** PANNs CNN14 (ONNX preferred, PyTorch fallback via `panns_inference`)

**Frame analysis:**
- Frame size: 1.0 seconds
- Hop size: 0.5 seconds (50% overlap)
- Enter threshold: 0.50 (confidence to start detecting an event)
- Exit threshold: 0.35 (confidence to stop detecting an event)
- Minimum duration: 0.30 seconds (filter out brief noises)
- Merge gap: 0.20 seconds (bridge short silence within same event)

**Label collapse:** AudioSet 527 classes → ~20 semantic groups (speech, music, laughter, keyboard, door, phone, TV, appliances, etc.)

**Enable/disable:**
- Default: **Enabled**
- Disable: `--disable-sed`

---

## When to Search for More Info

Before making claims about:
- Specific parameter values in code
- Line numbers in source files
- Function signatures and return types
- Model architectures or sizes
- Dependency version numbers
- Default configuration values

**Always verify against actual source code.** Use project knowledge search or ask the user for confirmation.

---

## Reporting Checklist (Every Response Must Include)

- ✅ Only factual, reproducible changes (no hallucinated logs)
- ✅ Ruff/lint passed (or specific errors listed)
- ✅ Tests passed (or specific failures explained)
- ✅ Full pipeline run completed if relevant (or errors documented)
- ✅ No schema regressions (39 columns maintained)
- ✅ All assumptions, risks, version bumps documented
- ✅ No secrets/credentials in artifacts or logs
- ✅ Orchestrator override behavior preserved (if modifying diarization)
- ✅ Stage count remains 11 (if modifying pipeline)

---

## Key Files Reference

- **Pipeline orchestration:** `src/diaremot/pipeline/orchestrator.py`
- **Stage registry:** `src/diaremot/pipeline/stages/__init__.py`
- **CSV schema:** `src/diaremot/pipeline/outputs.py` (SEGMENT_COLUMNS)
- **Main CLI:** `src/diaremot/cli.py` (Typer app)
- **Config dataclasses:** `src/diaremot/pipeline/config.py`
- **Diarization:** `src/diaremot/pipeline/speaker_diarization.py` (DiarizationConfig)
- **Emotion analysis:** `src/diaremot/affect/emotion_analyzer.py`
- **Paralinguistics:** `src/diaremot/affect/paralinguistics.py`
- **SED:** `src/diaremot/affect/sed_panns.py`

---

## Remember

You are the assistant. The user is the architect. Execute their vision with precision and honesty. Never fabricate logs, test results, or capabilities. When uncertain, verify. When wrong, admit it. When successful, report factually.

**This is production code serving real users. Precision matters. Quality matters. Truth matters.**
