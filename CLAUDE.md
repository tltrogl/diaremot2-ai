# CLAUDE.md — AI Assistant Instructions for DiaRemot

**Last updated:** 2025-10-04  
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

### 12-Stage Pipeline (canonical order)
**Source:** `src/diaremot/pipeline/stages/__init__.py::PIPELINE_STAGES`

1. **dependency_check** — Validate runtime dependencies
2. **preprocess** — Audio normalization, denoising, auto-chunking
3. **auto_tune** — Adaptive VAD parameter tuning
4. **background_sed** — PANNs CNN14 sound event detection
5. **diarize** — Silero VAD + ECAPA-TDNN + AHC clustering
6. **transcribe** — Faster-Whisper tiny-en (CTranslate2)
7. **paralinguistics** — Praat-Parselmouth voice quality + prosody
8. **affect_and_assemble** — Audio/text affect + segment assembly
9. **overlap_interruptions** — Turn-taking analysis
10. **conversation_analysis** — Flow metrics, dominance
11. **speaker_rollups** — Per-speaker summaries
12. **outputs** — Write CSV, JSON, HTML, PDF files

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
- **ASR:** `faster-whisper-tiny.en` (CTranslate2, int8 quantization)
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
python -m diaremot.cli run --input <file> --outdir <dir> --asr-compute-type int8

# ASR app (default compute_type=int8)  
python -m diaremot.cli asr run --input <file> --outdir <dir>

# Legacy entry point
python -m diaremot.pipeline.cli_entry --input <file> --outdir <dir> --asr-compute-type int8
```

**Critical flags:**
- `--input` / `-i` — Audio file path
- `--outdir` / `-o` — Output directory
- `--asr-compute-type` — `float32` | `int8` | `int8_float16`
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
- List affected files with line numbers
- Describe data structure changes
- Outline test strategy
- Identify breaking changes

### 2. Implementation Phase
- **Minimal diffs** — touch only necessary code
- **Preserve module boundaries** — don't merge unrelated logic
- **Match existing style** — follow `ruff` conventions
- **No placeholder TODOs** — complete all code paths

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
4. **int8 ASR default:** For asr_app and cli_entry (main app uses float32)
5. **Paralinguistics required:** Stage must never be skipped silently
6. **SED default-enabled:** `enable_sed=True` unless user explicitly disables

### File Paths (respect these)
- **Models:** `${DIAREMOT_MODEL_DIR}/*.onnx`
- **Speaker registry:** Default `speaker_registry.json` (configurable)
- **Cache:** `.cache/hf/`, `.cache/transformers/`, `.cache/torch/`
- **Outputs:** `<outdir>/diarized_transcript_with_emotion.csv`, etc.

### Performance Guardrails
- **OMP/MKL threads:** 4 max
- **ASR threads:** 1
- **ASR chunking:** 10 min windows
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

### Adding a Pipeline Stage
1. Create `stages/<name>.py` with `run(pipeline, state, guard)` signature
2. Import in `stages/__init__.py`
3. Insert `StageDefinition("name", module.run)` at correct position in `PIPELINE_STAGES`
4. Update `AI_INDEX.yaml` pipeline_spec
5. Add integration test

### Adding a Model
1. Add to `affect/emotion_analyzer.py` or create new analyzer module
2. Update `models:` section in `AI_INDEX.yaml`
3. Add ONNX export/loading logic if applicable
4. Document HF fallback behavior
5. Add to `requirements.txt` if new dependency

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
```

### Integration Tests
```bash
# Full pipeline smoke test
python -m diaremot.cli run \
  --input tests/fixtures/sample_30sec.wav \
  --outdir /tmp/smoke_test \
  --asr-compute-type int8

# Verify outputs exist
ls /tmp/smoke_test/diarized_transcript_with_emotion.csv
```

### Regression Tests
- Keep `tests/fixtures/sample_30sec.wav` as canonical test audio
- Maintain CSV golden file for schema validation
- Check `qc_report.json` for stage failures

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

---

## Example Response Template

```markdown
## Summary
Added `speaking_rate_category` field to CSV schema to classify WPM into slow/normal/fast bins.

## Files Modified
- `src/diaremot/pipeline/outputs.py` (+2 lines) — Added column to SEGMENT_COLUMNS
- `src/diaremot/pipeline/stages/affect.py` (+8 lines) — Compute category from WPM
- `tests/test_paralinguistics.py` (+12 lines) — Unit test for categorization

## Commands Executed
```bash
ruff check src/ tests/  # exit 0
pytest tests/test_paralinguistics.py -v  # 3 passed
python -m diaremot.cli run --input data/sample.wav --outdir /tmp/test  # exit 0
```

## Outputs Generated
- `/tmp/test/diarized_transcript_with_emotion.csv` — New column verified present
- Column values: "slow" (WPM <100), "normal" (100–160), "fast" (>160)

## Risks
- Backward compatibility: Old CSVs won't have this column (safe - column appended to end)
- Thresholds (100/160 WPM) are subjective — may need tuning

## Follow-up
- Consider making thresholds configurable via CLI flag
- Add visualization in HTML summary
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

**Remember:** This is production code serving real users. Precision matters. When in doubt, verify before claiming.
