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

**CRITICAL:** There are exactly **11 stages**. There is NO `auto_tune` stage in this list.

### The auto_tune Confusion

There EXISTS a module `src/diaremot/pipeline/stages/auto_tune.py` with a `run()` function. However:

1. **It is NOT imported** in `stages/__init__.py`
2. **It is NOT in PIPELINE_STAGES** list
3. **It is NOT called** during normal pipeline execution
4. **The orchestrator applies VAD tuning** directly in `_init_components()` method

**Conclusion:** The `auto_tune` module is **dead code** or **unused legacy**. Do not claim it runs as a stage.

### VAD Parameter Reality

**What the docs previously claimed (WRONG):**
- "Adaptive VAD tuning stage applies thresholds 0.22/0.40/0.40/0.15"
- "auto_tune is stage 3"

**What actually happens:**

**CLI Defaults** (`src/diaremot/cli.py` lines 188-191):
```python
vad_threshold: float = typer.Option(0.30, help="Silero VAD probability threshold.")
vad_min_speech_sec: float = typer.Option(0.8, help="Minimum detected speech duration.")
vad_min_silence_sec: float = typer.Option(0.8, help="Minimum detected silence duration.")
vad_speech_pad_sec: float = typer.Option(0.2, help="Padding added around VAD speech regions.")
ahc_distance_threshold: float = typer.Option(0.12, help="Agglomerative clustering distance threshold.")
```

**Orchestrator Overrides** (`src/diaremot/pipeline/orchestrator.py::AudioAnalysisPipelineV2._init_components()` lines 227-242):
```python
# Fix VAD oversegmentation: stricter thresholds, longer minimums, less padding
try:
    # Only apply if user did not override via CLI
    if "vad_threshold" not in cfg:
        self.diar_conf.vad_threshold = 0.35  # Much stricter to avoid micro-snippets
    if "vad_min_speech_sec" not in cfg:
        self.diar_conf.vad_min_speech_sec = 0.8  # Longer minimum to merge breaths
    if "vad_min_silence_sec" not in cfg:
        self.diar_conf.vad_min_silence_sec = 0.8  # Longer gaps required
    if "vad_speech_pad_sec" not in cfg:
        self.diar_conf.speech_pad_sec = 0.1  # Less padding to avoid overlap
except Exception:
    pass
```

**AHC Distance Override** (`src/diaremot/pipeline/orchestrator.py` line 218):
```python
self.diar_conf = DiarizationConfig(
    # ...
    ahc_distance_threshold=cfg.get(
        "ahc_distance_threshold", 0.15  # Much looser clustering to prevent speaker fragmentation
    ),
    # ...
)
```

**Ground truth:**
- Orchestrator applies `0.35/0.8/0.8/0.1` for VAD when user doesn't override
- Orchestrator applies `0.15` for AHC distance (not 0.12 from CLI default)
- This happens in `__init__` method, NOT in a separate stage
- User can override with CLI flags: `--vad-threshold 0.30 --ahc-distance-threshold 0.12`

### CSV Schema (39 columns)
**Source:** `src/diaremot/pipeline/outputs.py::SEGMENT_COLUMNS` (lines 10-49)

```python
SEGMENT_COLUMNS = [
    "file_id",
    "start",
    "end",
    "speaker_id",
    "speaker_name",
    "text",
    "valence",
    "arousal",
    "dominance",
    "emotion_top",
    "emotion_scores_json",
    "text_emotions_top5_json",
    "text_emotions_full_json",
    "intent_top",
    "intent_top3_json",
    "events_top3_json",
    "noise_tag",
    "asr_logprob_avg",
    "snr_db",
    "snr_db_sed",
    "wpm",
    "duration_s",
    "words",
    "pause_ratio",
    "low_confidence_ser",
    "vad_unstable",
    "affect_hint",
    "pause_count",
    "pause_time_s",
    "f0_mean_hz",
    "f0_std_hz",
    "loudness_rms",
    "disfluency_count",
    "error_flags",
    "vq_jitter_pct",
    "vq_shimmer_db",
    "vq_hnr_db",
    "vq_cpps_db",
    "voice_quality_hint",
]
```

**Do NOT modify schema without migration plan.**

### Key Models

**Diarization:**
- Silero VAD (ONNX preferred, TorchHub fallback, energy heuristic ultimate fallback)
- ECAPA-TDNN (ONNX)
- Source: `src/diaremot/pipeline/speaker_diarization.py`

**ASR:**
- faster-whisper tiny.en (CTranslate2)
- Default compute_type: **float32** for main CLI (NOT int8 as some old docs claimed)
- Source: `src/diaremot/pipeline/transcription_module.py`, `src/diaremot/cli.py` line 174

**Tone (V/A/D):**
- `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`
- Source: `src/diaremot/affect/emotion_analyzer.py`

**Speech emotion (8-class):**
- `Dpngtm/wav2vec2-emotion-recognition`
- Source: `src/diaremot/affect/emotion_analyzer.py`

**Text emotions (28-class):**
- `SamLowe/roberta-base-go_emotions`
- Backends: ONNX (preferred) → HuggingFace transformers (fallback)
- Source: `src/diaremot/affect/text_analyzer.py`

**Intent:**
- `facebook/bart-large-mnli` (zero-shot)
- Backends: ONNX (local exports preferred) → HuggingFace (fallback) → rule-based heuristics
- Source: `src/diaremot/affect/intent_analyzer.py`

**SED:**
- PANNs CNN14 (AudioSet 527 classes)
- Backends: ONNX → PyTorch `panns_inference`
- Label collapse: 527 → ~20 semantic groups
- Source: `src/diaremot/affect/sed_panns.py`

**Paralinguistics:**
- Praat-Parselmouth (native C++ library)
- Metrics: jitter, shimmer, HNR, CPPS, WPM, pauses, pitch, loudness
- Source: `src/diaremot/affect/paralinguistics.py`

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

# Resume from checkpoint
python -m diaremot.cli resume --input <file> --outdir <dir>
```

**Critical flags:**
- `--input` / `-i` — Audio file path
- `--outdir` / `-o` — Output directory
- `--asr-compute-type` — `float32` (default) | `int8` | `int8_float16`
- `--vad-threshold` — Override orchestrator default (0.35)
- `--vad-min-speech-sec` — Override orchestrator default (0.8)
- `--vad-speech-pad-sec` — Override orchestrator default (0.1)
- `--ahc-distance-threshold` — Override orchestrator default (0.15)
- `--profile` — `default` | `fast` | `accurate` | `offline` | path to JSON
- `--disable-sed` — Skip sound event detection
- `--disable-affect` — Skip emotion/intent analysis

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
- **List affected files** with line numbers
- **Describe data structure changes** (especially schema)
- **Outline test strategy** (unit, integration, smoke)
- **Identify breaking changes** and migration path

### 2. Implementation Phase
- **Minimal diffs** — touch only necessary code
- **Preserve module boundaries** — don't merge unrelated logic
- **Match existing style** — follow `ruff` conventions
- **No placeholder TODOs** — complete all code paths
- **Verify source truth** — check actual code, not docs

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
```

### 4. Reporting Phase
Include in every response:
- **Summary:** 1–2 paragraphs of what changed
- **Files modified:** List with brief descriptions and line ranges
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
4. **11 stages exactly:** Do NOT claim 12 stages or auto_tune as a stage
5. **Main CLI default:** `compute_type=float32` (not int8)
6. **Paralinguistics required:** Stage must never be skipped silently
7. **SED default-enabled:** `enable_sed=True` unless user explicitly disables

### File Paths (respect these)

**Models:** `${DIAREMOT_MODEL_DIR}/*.onnx`  
Search order:
1. `$DIAREMOT_MODEL_DIR`
2. `D:/models` (Windows) or `/models` (Unix)
3. Project `models/` directory
4. `$HOME/models`

**Speaker registry:** Default `speaker_registry.json` (configurable via `--registry-path`)

**Cache:** `.cache/hf/`, `.cache/transformers/`, `.cache/torch/`

**Outputs:** `<outdir>/diarized_transcript_with_emotion.csv`, etc.

### Performance Guardrails
- **OMP/MKL threads:** 4 max
- **ASR threads:** 1
- **ASR chunking:** 10 min windows (480 seconds default)
- **Affect chunking:** 30 sec windows
- **Auto-chunk threshold:** 30 min files

---

## Common Modifications

### Adding a CSV Column

**Steps:**
1. Update `outputs.py::SEGMENT_COLUMNS` — **append to end** (never insert in middle)
2. Update `affect.py::run()` — populate field in segment dict
3. Update `outputs.py::ensure_segment_keys()` — add default value
4. Add unit test verifying column exists and populated
5. Update README CSV schema section

**Example:**
```python
# outputs.py
SEGMENT_COLUMNS = [
    # ... existing 39 columns ...
    "new_column_name",  # Column 40
]

# affect.py::run()
row = {
    # ... existing fields ...
    "new_column_name": computed_value,
}

# outputs.py::ensure_segment_keys()
defaults: dict[str, Any] = {
    # ... existing defaults ...
    "new_column_name": 0.0,  # or None, or ""
}
```

### Adding a Pipeline Stage

**Steps:**
1. Create `stages/<stage_name>.py` with `run(pipeline, state, guard)` signature
2. Import in `stages/__init__.py`
3. Insert `StageDefinition("stage_name", module.run)` at correct position in `PIPELINE_STAGES`
4. Update `AI_INDEX.yaml` pipeline_spec
5. Add integration test
6. Update CLAUDE.md and AGENTS.md stage counts

**Example:**
```python
# stages/new_stage.py
def run(pipeline: AudioAnalysisPipelineV2, state: PipelineState, guard: StageGuard) -> None:
    # Stage logic here
    guard.done()

# stages/__init__.py
from . import new_stage

PIPELINE_STAGES: list[StageDefinition] = [
    # ... existing stages ...
    StageDefinition("new_stage", new_stage.run),
    # ... remaining stages ...
]
```

### Adding a Model

**Steps:**
1. Add to `affect/emotion_analyzer.py` or create new analyzer module
2. Implement ONNX loading (preferred) + PyTorch fallback
3. Update `models:` section in `AI_INDEX.yaml`
4. Document HF fallback behavior
5. Add to `requirements.txt` if new dependency
6. Add unit tests for both ONNX and PyTorch paths

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
    
# Test schema conformance
def test_csv_schema_columns():
    from diaremot.pipeline.outputs import SEGMENT_COLUMNS
    assert len(SEGMENT_COLUMNS) == 39
    assert "vq_jitter_pct" in SEGMENT_COLUMNS
```

### Integration Tests
```bash
# Full pipeline smoke test
python -m diaremot.cli run \
  --input tests/fixtures/sample_30sec.wav \
  --outdir /tmp/smoke_test

# Verify outputs exist
ls /tmp/smoke_test/diarized_transcript_with_emotion.csv
ls /tmp/smoke_test/segments.jsonl
ls /tmp/smoke_test/speakers_summary.csv

# Verify CSV has 39 columns
head -1 /tmp/smoke_test/diarized_transcript_with_emotion.csv | tr ',' '\n' | wc -l
```

### Regression Tests
- Keep `tests/fixtures/sample_30sec.wav` as canonical test audio
- Maintain CSV golden file for schema validation
- Check `qc_report.json` for stage failures
- Verify all 11 stages completed

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

### Inspect Pipeline State
```bash
# Check which stages completed
cat outputs/qc_report.json | jq '.stage_timings_ms'

# Check for warnings/errors
cat outputs/qc_report.json | jq '.warnings, .errors, .failures'
```

---

## Research Guidelines

### When to Search Documentation
- Model API changes (check HuggingFace model cards)
- ONNX export procedures (check official ONNX docs)
- Praat-Parselmouth APIs (check library docs)
- CTranslate2 quantization (check CT2 docs)

### Cite Sources
```markdown
Per HuggingFace model card for `SamLowe/roberta-base-go_emotions`:
- Returns 28-class emotion distribution
- Use `return_all_scores=True` to get full dist

Source: https://huggingface.co/SamLowe/roberta-base-go_emotions
```

### Dependency Updates
Before updating `requirements.txt`:
1. Check breaking changes in changelog
2. Test locally with new version
3. Document version bump rationale
4. Update `AI_INDEX.yaml` if model loading changes

---

## Reporting Checklist

Before submitting any response with code changes:

- [ ] Only factual, reproducible outputs (no simulated logs)
- [ ] `ruff check` passed (or errors listed)
- [ ] Tests passed (or failures explained)
- [ ] Full pipeline run completed (or errors documented)
- [ ] No schema regressions
- [ ] All assumptions/risks documented
- [ ] CLI examples tested
- [ ] No secrets/credentials in artifacts
- [ ] Verified source code before claiming behavior
- [ ] Cited file paths and line numbers

---

## Example Response Template

```markdown
## Summary
Added `speaking_rate_category` field to CSV schema to classify WPM into slow/normal/fast bins.

## Files Modified
- `src/diaremot/pipeline/outputs.py` (+2 lines, lines 48-49) — Added column to SEGMENT_COLUMNS
- `src/diaremot/pipeline/stages/affect.py` (+8 lines, lines 67-74) — Compute category from WPM
- `tests/test_paralinguistics.py` (+12 lines, lines 89-100) — Unit test for categorization

## Source Code References
From `src/diaremot/pipeline/outputs.py` line 49:
```python
SEGMENT_COLUMNS = [
    # ... existing 39 columns ...
    "speaking_rate_category",  # NEW: Column 40
]
```

From `src/diaremot/pipeline/stages/affect.py` lines 67-74:
```python
# Compute speaking rate category
wpm = pm.get("wpm", 0.0)
if wpm < 100:
    rate_category = "slow"
elif wpm > 160:
    rate_category = "fast"
else:
    rate_category = "normal"
row["speaking_rate_category"] = rate_category
```

## Commands Executed
```bash
$ ruff check src/ tests/
All checks passed!

$ pytest tests/test_paralinguistics.py -v
test_speaking_rate_slow PASSED
test_speaking_rate_normal PASSED
test_speaking_rate_fast PASSED
3 passed in 0.15s

$ python -m diaremot.cli run --input data/sample.wav --outdir /tmp/test
[Pipeline completed successfully]
```

## Outputs Generated
- `/tmp/test/diarized_transcript_with_emotion.csv` — New column verified present (40 columns total)
- Column values: "slow" (WPM <100), "normal" (100–160), "fast" (>160)

## Verification
```bash
$ head -1 /tmp/test/diarized_transcript_with_emotion.csv | tr ',' '\n' | wc -l
40

$ head -2 /tmp/test/diarized_transcript_with_emotion.csv | tail -1 | cut -d',' -f40
normal
```

## Risks
- **Backward compatibility:** Old CSVs won't have this column (safe - column appended to end)
- **Thresholds:** (100/160 WPM) are subjective — may need tuning for different domains
- **Schema migration:** Consumers expecting 39 columns need update

## Follow-up
- Consider making thresholds configurable via CLI flag (e.g., `--wpm-slow-threshold`, `--wpm-fast-threshold`)
- Add visualization in HTML summary showing distribution of rate categories
- Update documentation (README, CLAUDE.md, AGENTS.md) to reflect 40-column schema
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

### Documentation
- **User guide:** `README.md`
- **Agent instructions:** `AGENTS.md`
- **Architecture index:** `AI_INDEX.yaml`
- **This file:** `CLAUDE.md`

### Support
For questions not covered here, check:
1. `AI_INDEX.yaml` for architecture mapping
2. `AGENTS.md` for detailed pipeline contract
3. Source code docstrings
4. Project knowledge base

---

## Common Mistakes to Avoid

1. **Claiming auto_tune is a stage** → It's NOT in PIPELINE_STAGES
2. **Wrong stage count** → 11 stages, not 12
3. **Wrong VAD defaults** → Orchestrator uses 0.35, not 0.22 or 0.30
4. **Wrong AHC default** → Orchestrator uses 0.15, not 0.12
5. **Wrong compute_type** → Main CLI defaults to float32, not int8
6. **Ignoring orchestrator overrides** → Check `_init_components()` for actual values
7. **Not verifying source code** → Always check actual implementation
8. **Fabricating logs** → Only report what actually ran
9. **Breaking schema** → Always append columns, never insert
10. **Claiming features without evidence** → Cite file paths and line numbers

---

**Remember:** This is production code serving real users. Precision matters. When in doubt, verify source code before claiming behavior. Always cite file paths and line numbers.
