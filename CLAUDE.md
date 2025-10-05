# CLAUDE.md — AI Assistant Instructions for DiaRemot

**Last updated:** 2025-10-05  
**For:** Claude, AI coding assistants, and automated agents

---

## Project Overview

**DiaRemot** is a CPU-only speech intelligence pipeline processing long-form audio (1–3 hours) into diarized transcripts with comprehensive affect, paralinguistic, and sound event analysis.

**Core stack:** Python 3.11 | PyTorch CPU | ONNX | CTranslate2 | Praat-Parselmouth

**Key outputs:** 
- 39-column CSV with per-segment emotions, intent, voice quality, SED events
- HTML/PDF summaries with speaker analytics
- Persistent speaker registry across files

---

## Truth & Correctness Requirements

### Critical Rules
1. **Never fabricate logs, outputs, or test results** — only report what actually executed
2. **Never simulate code execution** — run actual commands and report real exit codes
3. **Always verify against source code** before claiming behavior
4. **Cite specific files/line numbers** when referencing implementation details
5. **Flag assumptions explicitly** — distinguish fact from speculation

### When Uncertain
- State "I need to verify this by checking [file]"
- Propose diagnostic tests to confirm behavior
- Never guess at API signatures or parameters

---

## Architecture Reference

### 11-Stage Pipeline (canonical order)
**Source:** `src/diaremot/pipeline/stages/__init__.py::PIPELINE_STAGES`

1. **dependency_check** — Validate runtime dependencies
2. **preprocess** — Audio normalization, denoising, auto-chunking
3. **background_sed** — PANNs CNN14 sound event detection
4. **diarize** — Silero VAD + ECAPA-TDNN + AHC clustering
5. **transcribe** — Faster-Whisper tiny-en (CTranslate2)
6. **paralinguistics** — Praat-Parselmouth voice quality + prosody
7. **affect_and_assemble** — Audio/text affect + segment assembly
8. **overlap_interruptions** — Turn-taking analysis
9. **conversation_analysis** — Flow metrics, dominance
10. **speaker_rollups** — Per-speaker summaries
11. **outputs** — Write CSV, JSON, HTML, PDF files

**CRITICAL:** There is NO `auto_tune` stage. The `auto_tune.py` module exists but is NOT in PIPELINE_STAGES. VAD tuning happens inline in orchestrator's `__init__` method.

### Stage Details

#### 1. dependency_check
**Module:** `src/diaremot/pipeline/stages/dependency_check.py`
**Purpose:** Validate core dependencies are importable and optionally check versions
**Critical dependencies:**
- `onnxruntime >= 1.16.0`
- `faster-whisper >= 1.0.0` (includes CTranslate2)
- `transformers` (tokenizers only)
- `praat-parselmouth`
- `librosa`, `scipy`, `numpy`, `soundfile`

#### 2. preprocess
**Module:** `src/diaremot/pipeline/stages/preprocess.py::run_preprocess`
**Purpose:** Audio normalization, denoising, health assessment, auto-chunking
**Config:** `PreprocessConfig` in `audio_preprocessing.py`
**Parameters:**
- `target_sr`: 16000 Hz
- `denoise`: "spectral_sub_soft" | "none"
- `loudness_mode`: "asr" (normalize to -20 LUFS)
- `auto_chunk_enabled`: true
- `chunk_threshold_minutes`: 30.0
- `chunk_size_minutes`: 20.0
- `chunk_overlap_seconds`: 30.0

**Outputs to state:**
- `y`: Preprocessed audio array
- `sr`: Sample rate (16000)
- `health`: AudioHealth object (SNR, clipping, dynamic range)
- `duration_s`: Total duration
- `audio_sha16`: 16-char hash for caching
- `cache_dir`: Cache directory for this file

#### 3. background_sed
**Module:** `src/diaremot/pipeline/stages/preprocess.py::run_background_sed`
**Purpose:** Sound event detection for ambient context
**Model:** PANNs CNN14
**Runtime:** ONNXRuntime (preferred), PyTorch (`panns_inference` fallback)
**Parameters:**
- Frame: 1.0s
- Hop: 0.5s
- Enter threshold: 0.50
- Exit threshold: 0.35
- Min duration: 0.30s
- Merge gap: 0.20s
- Label collapse: AudioSet 527 → ~20 semantic groups

**Enabled by default.** Disable with `--disable-sed`.

**Outputs to state:**
- `sed_info`: Dict with dominant_label, noise_score, event timeline

#### 4. diarize
**Module:** `src/diaremot/pipeline/stages/diarize.py`
**Purpose:** Speaker segmentation via VAD + embedding + clustering
**Components:**
- **VAD:** Silero (ONNX → Torch → Energy fallback)
- **Embeddings:** ECAPA-TDNN (ONNX)
- **Clustering:** Agglomerative Hierarchical Clustering

**CLI Defaults** (`src/diaremot/cli.py`):
```python
vad_threshold: 0.30
vad_min_speech_sec: 0.80
vad_min_silence_sec: 0.80
speech_pad_sec: 0.20
ahc_distance_threshold: 0.12
```

**Orchestrator Overrides** (`src/diaremot/pipeline/orchestrator.py::_init_components`, lines 234-244):
Applied when user doesn't set CLI flags:
```python
vad_threshold: 0.35  # Stricter to reduce oversegmentation
vad_min_speech_sec: 0.80  # Same
vad_min_silence_sec: 0.80  # Same
speech_pad_sec: 0.10  # Less padding to avoid overlap
ahc_distance_threshold: 0.15  # Looser to prevent speaker fragmentation
```

**Override orchestrator tuning:**
```bash
python -m diaremot.cli run -i audio.wav -o outputs/ \
  --vad-threshold 0.30 \
  --ahc-distance-threshold 0.12
```

**Outputs to state:**
- `turns`: List of DiarizedTurn objects
- `resume_diar`: Boolean (true if using cached diarization)

#### 5. transcribe
**Module:** `src/diaremot/pipeline/stages/asr.py`
**Purpose:** Speech-to-text via Faster-Whisper
**Model:** `tiny.en` (39 MB)
**Backend:** CTranslate2
**Default compute_type:** `float32` (main CLI), `int8` (optional override)

**Parameters:**
- `beam_size`: 1 (greedy decoding)
- `temperature`: 0.0 (deterministic)
- `no_speech_threshold`: 0.50
- `vad_filter`: true (built-in Silero VAD)
- `max_asr_window_sec`: 480 (8 minutes)
- `segment_timeout_sec`: 300.0
- `batch_timeout_sec`: 1200.0

**Runs on:** Diarized speech turns only (not full audio)

**Outputs to state:**
- `norm_tx`: List of transcript segments with text, timestamps, speaker_id, speaker_name
- `resume_tx`: Boolean (true if using cached transcription)

#### 6. paralinguistics
**Module:** `src/diaremot/pipeline/stages/paralinguistics.py`
**Purpose:** Voice quality and prosody analysis
**Runtime:** Praat-Parselmouth (required)
**Metrics extracted:**
- **Voice quality:** jitter (%), shimmer (dB), HNR (dB), CPPS (dB)
- **Prosody:** WPM, duration_s, words, pause_count, pause_time_s, pause_ratio
- **Pitch:** f0_mean_hz, f0_std_hz
- **Loudness:** loudness_rms
- **Disfluencies:** disfluency_count

**Fallback:** If Praat fails, compute WPM from text and set voice quality metrics to 0.0

**Outputs to state:**
- `para_metrics`: Dict mapping segment index → metrics dict

#### 7. affect_and_assemble
**Module:** `src/diaremot/pipeline/stages/affect.py`
**Purpose:** Emotion and intent analysis, final segment assembly
**Models:**
- **Audio VAD:** `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` (valence/arousal/dominance)
- **Speech emotion:** `Dpngtm/wav2vec2-emotion-recognition` (8-class)
- **Text emotion:** `SamLowe/roberta-base-go_emotions` (28-class)
- **Intent:** `facebook/bart-large-mnli` (zero-shot)

**Runtime:** ONNXRuntime (preferred), HuggingFace transformers (fallback)

**Outputs to state:**
- `segments_final`: List of complete segment dicts with all 39 CSV columns

#### 8. overlap_interruptions
**Module:** `src/diaremot/pipeline/stages/summaries.py::run_overlap`
**Purpose:** Detect overlaps, interruptions, turn-taking patterns

**Outputs to state:**
- `overlap_stats`: Dict with overlap_count, total_overlap_duration_s
- `per_speaker_interrupts`: Dict mapping speaker_id → {made, received}

#### 9. conversation_analysis
**Module:** `src/diaremot/pipeline/stages/summaries.py::run_conversation`
**Purpose:** Flow metrics, dominance, engagement

**Outputs to state:**
- `conv_metrics`: ConversationMetrics object

#### 10. speaker_rollups
**Module:** `src/diaremot/pipeline/stages/summaries.py::run_speaker_rollups`
**Purpose:** Per-speaker summary statistics

**Outputs to state:**
- `speakers_summary`: List of dicts with speaker-level aggregates

#### 11. outputs
**Module:** `src/diaremot/pipeline/stages/summaries.py::run_outputs`
**Purpose:** Write all output files

**Files written:**
- `diarized_transcript_with_emotion.csv` (39 columns)
- `segments.jsonl`
- `speakers_summary.csv`
- `timeline.csv`
- `qc_report.json`
- `summary.html` (optional, requires templates)
- `summary.pdf` (optional, requires wkhtmltopdf)
- `speaker_registry.json`
- `events_timeline.csv`, `events.jsonl` (if SED enabled)

### CSV Schema (39 columns)
**Source:** `src/diaremot/pipeline/outputs.py::SEGMENT_COLUMNS`

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

**Do NOT modify schema without migration plan.**

### Key Models
- **Diarization:** Silero VAD (Torch/ONNX) + ECAPA-TDNN (ONNX)
- **ASR:** `faster-whisper-tiny.en` (CTranslate2, float32 default)
- **Tone (V/A/D):** `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`
- **Speech emotion:** `Dpngtm/wav2vec2-emotion-recognition`
- **Text emotions:** `SamLowe/roberta-base-go_emotions`
- **Intent:** `facebook/bart-large-mnli` (prefers ONNX, fallback to HF)
- **SED:** PANNs CNN14 (ONNX)
- **Paralinguistics:** Praat-Parselmouth (jitter/shimmer/HNR/CPPS)

---

## CLI Interface

### Standard Arguments
**Source:** `src/diaremot/cli.py`

```bash
# Main app (default compute_type=float32)
python -m diaremot.cli run --input <file> --outdir <dir>

# Use int8 for faster ASR
python -m diaremot.cli run --input <file> --outdir <dir> --asr-compute-type int8

# Override orchestrator's VAD tuning
python -m diaremot.cli run --input <file> --outdir <dir> \
  --vad-threshold 0.30 \
  --ahc-distance-threshold 0.12
```

**Critical flags:**
- `--input` / `-i` — Audio file path
- `--outdir` / `-o` — Output directory
- `--asr-compute-type` — `float32` (default) | `int8` | `int8_float16`
- `--vad-threshold` — Override orchestrator default (0.35)
- `--vad-min-speech-sec` — Override default (0.80)
- `--speech-pad-sec` — Override orchestrator default (0.10)
- `--ahc-distance-threshold` — Override orchestrator default (0.15)
- `--profile` — `default` | `fast` | `accurate` | `offline` | path to JSON
- `--disable-sed` — Skip sound event detection
- `--disable-affect` — Skip emotion/intent analysis
- `--quiet` — Reduce console verbosity
- `--clear-cache` — Clear cache before running

### Environment Variables (required)
```bash
DIAREMOT_MODEL_DIR=/workspace/models
HF_HOME=.cache/hf
HUGGINGFACE_HUB_CACHE=.cache/hf
TRANSFORMERS_CACHE=.cache/transformers
TORCH_HOME=.cache/torch
OMP_NUM_THREADS=4
MKL_NUM_THREADS=4
NUMEXPR_MAX_THREADS=4
TOKENIZERS_PARALLELISM=false
```

---

## Development Workflow

### 1. Planning Phase
Before any implementation:
- List affected files with line numbers
- Describe data structure changes
- Outline test strategy
- Identify breaking changes
- Document VAD parameter effects if modifying diarization

### 2. Implementation Phase
- **Minimal diffs** — touch only necessary code
- **Preserve module boundaries** — don't merge unrelated logic
- **Match existing style** — follow `ruff` conventions
- **No placeholder TODOs** — complete all code paths
- **Respect orchestrator overrides** — understand when they apply

### 3. Verification Phase (mandatory)
```bash
# Lint
ruff check src/ tests/

# Type check (if mypy configured)
mypy src/

# Unit tests
pytest tests/ -v

# Smoke test (if sample audio available)
python -m diaremot.cli run --input data/sample.wav --outdir outputs/

# Verify CSV schema unchanged
python -c "from diaremot.pipeline.outputs import SEGMENT_COLUMNS; print(len(SEGMENT_COLUMNS))"  # Should be 39
```

### 4. Reporting Phase
Include in every response:
- **Summary:** 1–2 paragraphs of what changed
- **Files modified:** List with brief descriptions
- **Commands executed:** With exit codes
- **Key logs:** Tail ~200 lines of relevant output
- **Artifact paths:** CSV/JSON/HTML files generated
- **Risks/assumptions:** What could break, what's untested
- **Follow-up:** Recommended next steps

---

## Hard Constraints

### Never Break These
1. **Schema stability:** `SEGMENT_COLUMNS` is sacred — extend only with migration
2. **Entry points:** Don't rename `cli.py::app`, `run_pipeline`, `cli_entry`
3. **CPU-only:** No GPU code, no CUDA dependencies
4. **11 stages:** Exactly this count in PIPELINE_STAGES
5. **Main CLI default:** `compute_type=float32` (not int8)
6. **Paralinguistics required:** Stage must never be skipped silently
7. **SED default-enabled:** `enable_sed=True` unless user explicitly disables

### File Paths (respect these)
- **Models:** `${DIAREMOT_MODEL_DIR}/*.onnx`
- **Speaker registry:** Default `speaker_registry.json` (configurable)
- **Cache:** `.cache/hf/`, `.cache/transformers/`, `.cache/torch/`
- **Outputs:** `<outdir>/diarized_transcript_with_emotion.csv`, etc.

### Performance Guardrails
- **OMP/MKL threads:** 4 max
- **ASR threads:** 1
- **ASR chunking:** 10 min windows (480s)
- **Affect chunking:** 30 sec windows
- **Auto-chunk threshold:** 30 min files

---

## Common Modifications

### Adding a CSV Column
1. Update `outputs.py::SEGMENT_COLUMNS` — append to end
2. Update `affect.py::run()` — populate field in segment dict
3. Update `outputs.py::ensure_segment_keys()` — add default value
4. Add unit test verifying column exists
5. Update README CSV schema section
6. Document migration path for existing CSVs

### Adding a Pipeline Stage
1. Create `stages/<n>.py` with `run(pipeline, state, guard)` signature
2. Import in `stages/__init__.py`
3. Insert `StageDefinition("name", module.run)` at correct position in `PIPELINE_STAGES`
4. Update `AI_INDEX.yaml` pipeline_spec
5. Add integration test
6. Update README stage count

### Adding a Model
1. Add to `affect/emotion_analyzer.py` or create new analyzer module
2. Update `models:` section in `AI_INDEX.yaml`
3. Add ONNX export/loading logic if applicable
4. Document HF fallback behavior
5. Add to `requirements.txt` if new dependency
6. Test both ONNX and fallback paths

### Modifying VAD Parameters
1. Identify whether change should be:
   - CLI default (in `cli.py`)
   - Orchestrator override (in `orchestrator.py::_init_components`)
   - Both
2. Document rationale (e.g., "reduces oversegmentation")
3. Test with known problematic audio
4. Update README and CLAUDE.md
5. Consider adding override flag if not already exists

---

## Testing Strategy

### Unit Tests
```python
# Test paralinguistics metrics
def test_paralinguistics_silent_audio():
    audio = np.zeros(16000)  # 1 sec silence
    result = extract_voice_quality(audio, sr=16000)
    assert result["vq_jitter_pct"] is not None
    assert result["vq_hnr_db"] is not None

# Test CSV schema integrity
def test_segment_columns_count():
    from diaremot.pipeline.outputs import SEGMENT_COLUMNS
    assert len(SEGMENT_COLUMNS) == 39

# Test orchestrator overrides apply
def test_orchestrator_vad_override():
    from diaremot.pipeline.orchestrator import AudioAnalysisPipelineV2
    pipe = AudioAnalysisPipelineV2({})
    # When user doesn't set vad_threshold via config
    assert pipe.diar_conf.vad_threshold == 0.35
```

### Integration Tests
```bash
# Full pipeline smoke test
python -m diaremot.cli run \
  --input tests/fixtures/sample_30sec.wav \
  --outdir /tmp/smoke_test

# Verify outputs exist
ls /tmp/smoke_test/diarized_transcript_with_emotion.csv
ls /tmp/smoke_test/qc_report.json

# Verify CSV has 39 columns
head -1 /tmp/smoke_test/diarized_transcript_with_emotion.csv | tr ',' '\n' | wc -l  # Should be 39
```

### Regression Tests
- Keep `tests/fixtures/sample_30sec.wav` as canonical test audio
- Maintain CSV golden file for schema validation
- Check `qc_report.json` for stage failures
- Validate orchestrator overrides still apply correctly

---

## Diagnostic Commands

### Check Dependencies
```bash
python -m diaremot.cli diagnostics --strict
```

### Verify Models Downloaded
```bash
ls -lh $DIAREMOT_MODEL_DIR/
ls -lh .cache/hf/hub/models--*
```

### Clear Cache
```bash
rm -rf .cache/
python -m diaremot.cli run --input <file> --outdir <dir> --clear-cache
```

### Resume Failed Run
```bash
python -m diaremot.cli resume --input <file> --outdir <dir>
```

### Debug VAD Issues
```bash
# Use CLI defaults instead of orchestrator overrides
python -m diaremot.cli run --input <file> --outdir <dir> \
  --vad-threshold 0.30 \
  --speech-pad-sec 0.20 \
  --ahc-distance-threshold 0.12
```

---

## Reporting Checklist

Before submitting any response with code changes:

- [ ] Only factual, reproducible outputs (no simulated logs)
- [ ] `ruff check` passed (or errors listed)
- [ ] Tests passed (or failures explained)
- [ ] Full pipeline run completed (or errors documented)
- [ ] No schema regressions (39 columns maintained)
- [ ] All assumptions/risks documented
- [ ] CLI examples tested
- [ ] No secrets/credentials in artifacts
- [ ] Orchestrator override behavior preserved (if modifying diarization)
- [ ] Stage count remains 11 (if modifying pipeline)

---

## Example Response Template

```markdown
## Summary
Modified VAD threshold calculation in orchestrator to use median audio energy instead of fixed 0.35 value for adaptive tuning.

## Files Modified
- `src/diaremot/pipeline/orchestrator.py` (+15 lines) — Added `_compute_adaptive_vad_threshold()` method
- `tests/test_orchestrator.py` (+23 lines) — Unit tests for adaptive threshold

## Commands Executed
```bash
ruff check src/ tests/  # exit 0
pytest tests/test_orchestrator.py -v  # 5 passed
python -m diaremot.cli run --input data/sample.wav --outdir /tmp/test  # exit 0
```

## Outputs Generated
- `/tmp/test/diarized_transcript_with_emotion.csv` — 39 columns verified
- `/tmp/test/qc_report.json` — Shows adaptive_vad_threshold: 0.32 in config_snapshot

## Key Logs
```
[orchestrator] _compute_adaptive_vad_threshold: median_energy_db=-28.5, threshold=0.32
[diarize] vad_threshold=0.32 (adaptive), vad_min_speech_sec=0.80
[diarize] detected 127 speech regions
```

## Risks
- Adaptive threshold may be too low for very noisy audio (consider min threshold of 0.25)
- User-specified `--vad-threshold` still overrides adaptive value (expected behavior)
- Not tested on >2hr audio files

## Follow-up
- Add CLI flag `--disable-adaptive-vad` for users who want fixed thresholds
- Test on corpus with wide SNR range (-10 to +30 dB)
- Consider capping adaptive threshold at 0.45 for ultra-quiet audio
```

---

## Quick Reference

### Key Files
- **Pipeline orchestration:** `src/diaremot/pipeline/orchestrator.py`
- **Stage registry:** `src/diaremot/pipeline/stages/__init__.py`
- **CSV schema:** `src/diaremot/pipeline/outputs.py`
- **CLI:** `src/diaremot/cli.py`
- **Config:** `src/diaremot/pipeline/config.py`
- **Models:** `src/diaremot/affect/emotion_analyzer.py`
- **Diarization:** `src/diaremot/pipeline/speaker_diarization.py`
- **Paralinguistics:** `src/diaremot/affect/paralinguistics.py`

### Documentation
- **User guide:** `README.md`
- **Agent instructions:** `AGENTS.md`
- **Architecture index:** `AI_INDEX.yaml`
- **This file:** `CLAUDE.md`

### Critical Line Numbers (for reference)
- **PIPELINE_STAGES list:** `src/diaremot/pipeline/stages/__init__.py` lines 12-24
- **SEGMENT_COLUMNS list:** `src/diaremot/pipeline/outputs.py` lines 11-49
- **Orchestrator VAD overrides:** `src/diaremot/pipeline/orchestrator.py` lines 234-244
- **CLI compute_type default:** `src/diaremot/cli.py` line 217

---

## Common Pitfalls to Avoid

1. **Claiming auto_tune is a stage** — It's NOT in PIPELINE_STAGES
2. **Wrong VAD defaults** — Orchestrator overrides to 0.35, not CLI's 0.30
3. **Wrong compute_type** — Main CLI defaults to float32, not int8
4. **Wrong stage count** — 11 stages, not 12
5. **Modifying schema carelessly** — Always append, never insert/reorder
6. **Ignoring orchestrator overrides** — Check `_init_components()` for actual values
7. **Breaking entry points** — `cli.py::app` must remain as Typer app
8. **Skipping paralinguistics** — Stage is required, must not fail silently
9. **Using PyTorch for preprocessing** — Use librosa/scipy/numpy
10. **Forgetting SED is default-enabled** — Must be explicitly disabled

---

**Remember:** This is production code serving real users. Precision matters. When in doubt, verify before claiming.
