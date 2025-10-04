# AI_INDEX.yaml Complete Corrections
**Generated:** 2025-10-04  
**Source verification:** cli.py, stages/__init__.py, outputs.py, AGENTS.md

---

## 1. PIPELINE STAGES - Add missing `auto_tune` stage

**Current AI_INDEX.yaml lists:**
```yaml
pipeline_spec:
  pre_vad:
  background_sed:
  diarize:
  asr:
  audio_affect:
  text_analysis:
  paralinguistics:
  speaker_registry:
```

**Actual implementation (stages/__init__.py):**
```python
PIPELINE_STAGES = [
    StageDefinition("dependency_check", dependency_check.run),
    StageDefinition("preprocess", preprocess.run_preprocess),
    StageDefinition("auto_tune", auto_tune.run),              # ← MISSING IN AI_INDEX
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

**Fix:** Add `auto_tune` stage after `preprocess` (from AGENTS.md):
```yaml
  auto_tune:
    name: Adaptive VAD Tuning
    description: Adaptive VAD parameter tuning based on audio characteristics
    runs_after: preprocess
    runs_before: background_sed
```

---

## 2. PARALINGUISTICS METRICS - Add missing fields

**Current:**
```yaml
paralinguistics:
  metrics:
  - WPM
  - pause_time_s
  - f0_mean_hz
  - f0_std_hz
  - loudness_rms
  - disfluency_count
  - jitter%
  - shimmer dB
  - HNR dB
  - CPPS dB
```

**Actual (outputs.py SEGMENT_COLUMNS):**
```python
"wpm",
"duration_s",           # ← MISSING
"words",                # ← MISSING
"pause_ratio",          # ← MISSING
"pause_count",          # ← MISSING (separate from pause_time_s)
"pause_time_s",
"f0_mean_hz",
"f0_std_hz",
"loudness_rms",
"disfluency_count",
"vq_jitter_pct",
"vq_shimmer_db",
"vq_hnr_db",
"vq_cpps_db",
```

**Fix:**
```yaml
paralinguistics:
  required: true
  tooling: Praat‑Parselmouth
  metrics:
  - WPM
  - duration_s
  - words
  - pause_count
  - pause_time_s
  - pause_ratio
  - f0_mean_hz
  - f0_std_hz
  - loudness_rms
  - disfluency_count
  - vq_jitter_pct (jitter%)
  - vq_shimmer_db (shimmer dB)
  - vq_hnr_db (HNR dB)
  - vq_cpps_db (CPPS dB)
  notes: Mandatory stage. Failures are surfaced; nulls only if a hard error is explicitly handled.
```

---

## 3. CLI ARGUMENTS - Standardize documentation

**Current inconsistencies in README/docs:**
- `--input` vs `--audio`
- `--outdir` vs `--out`
- `--asr-compute-type` vs `--compute-type`

**Actual (cli.py):**
```python
@app.command()
def run(
    input: Path = typer.Option(..., "--input", "-i", help="Path to input audio file."),
    outdir: Path = typer.Option(..., "--outdir", "-o", help="Directory to write outputs."),
    asr_compute_type: str = typer.Option("float32", help="CT2 compute type for faster-whisper."),
```

**HOWEVER, there's also an asr_app subcommand:**
```python
@asr_app.command("run")
def asr_run(
    input: Path = typer.Option(..., "--input", "-i", ...),
    outdir: Path = typer.Option(..., "--outdir", "-o", ...),
    asr_compute_type: Literal["float32", "int8", "int8_float16"] = typer.Option("int8", ...),
```

**Correct CLI examples:**
```bash
# Main app (asr_compute_type defaults to float32)
python -m diaremot.cli run --input data/sample.wav --outdir outputs/ --asr-compute-type int8

# ASR subcommand (asr_compute_type defaults to int8)
python -m diaremot.cli asr run --input data/sample.wav --outdir outputs/
```

---

## 4. ASR COMPUTE_TYPE DEFAULT - Clarify dual defaults

**AI_INDEX states:**
```yaml
asr:
  compute_type: int8
```

**Reality (cli.py):**
- `cli.py::run()` default: **`float32`**
- `cli.py::asr_run()` default: **`int8`**
- `cli_entry.py::_build_arg_parser()` default: **`int8`**

**AGENTS.md states:**
> default `compute_type = int8` (config default; transcriber fallback is `float32`)

**Fix:** Update AI_INDEX to clarify:
```yaml
asr:
  backend: faster‑whisper (CTranslate2)
  model: tiny‑en
  compute_type: int8  # Default for asr_app.run and cli_entry.py; main cli.run defaults to float32
  compute_type_note: "CLI default varies: asr_app uses int8, main app uses float32. Override with --asr-compute-type"
```

---

## 5. DIARIZATION PARAMETERS - Add orchestrator overrides

**AI_INDEX current:**
```yaml
diarize:
  params:
    vad_threshold: 0.3
    vad_min_speech_sec: 0.8
    vad_min_silence_sec: 0.8
    speech_pad_sec: 0.2
    ahc_distance_threshold: 0.12
    collar_sec: 0.25
    min_turn_sec: 1.5
```

**AGENTS.md documents orchestrator overrides:**
```
vad_threshold = 0.22 (relaxed from 0.30)
vad_min_speech_sec = 0.40 (relaxed from 0.80)
vad_min_silence_sec = 0.40 (relaxed from 0.80)
speech_pad_sec = 0.15 (relaxed from 0.20)
ahc_distance_threshold = 0.02 (orchestrator override; DiarizationConfig default is 0.12)
```

**Fix:** Add note about orchestrator overrides:
```yaml
diarize:
  stack: Silero VAD + ECAPA‑TDNN (embeddings) + Agglomerative clustering (AHC)
  params:
    vad_threshold: 0.3  # CLI default; orchestrator may override to 0.22
    vad_min_speech_sec: 0.8  # CLI default; orchestrator may override to 0.40
    vad_min_silence_sec: 0.8  # CLI default; orchestrator may override to 0.40
    speech_pad_sec: 0.2  # CLI default; orchestrator may override to 0.15
    ahc_distance_threshold: 0.12  # DiarizationConfig default; orchestrator may override to 0.02
    collar_sec: 0.25
    min_turn_sec: 1.5
  notes: Orchestrator may apply adaptive overrides based on audio characteristics
```

---

## 6. STAGE COUNT - Correct from 12 to actual count

**AGENTS.md claims:**
> Every run should include all 12 stages by default

**Actual (stages/__init__.py):**
12 stages total (correct):
1. dependency_check
2. preprocess
3. auto_tune
4. background_sed
5. diarize
6. transcribe
7. paralinguistics
8. affect_and_assemble
9. overlap_interruptions
10. conversation_analysis
11. speaker_rollups
12. outputs

**Fix:** AGENTS.md is CORRECT. AI_INDEX.yaml is MISSING `dependency_check` and `auto_tune` stages.

---

## 7. REPO LAYOUT - Update entry points

**Current:**
```yaml
entry_points:
  typer_app: diaremot.cli:app
  python_module: python -m diaremot.pipeline.run_pipeline
  cli_entry: python -m diaremot.pipeline.cli_entry
```

**Verified (all exist):**
- ✅ `diaremot.cli:app`
- ✅ `diaremot.pipeline.run_pipeline`
- ✅ `diaremot.pipeline.cli_entry`

**No changes needed.**

---

## SUMMARY OF ALL REQUIRED CHANGES

1. **Add `dependency_check` stage** (stage 1)
2. **Add `auto_tune` stage** (stage 3, after preprocess)
3. **Add missing paralinguistics metrics:** `duration_s`, `words`, `pause_count`, `pause_ratio`
4. **Standardize CLI args:** Use `--input`, `--outdir`, `--asr-compute-type`
5. **Clarify compute_type defaults:** float32 (main app) vs int8 (asr_app)
6. **Add diarization orchestrator override notes**
7. **Confirm stage count = 12** (AI_INDEX currently only documents 8)

---

## VERIFICATION CHECKLIST

- ✅ Actual stage count: 12 (AGENTS.md correct)
- ✅ CLI args verified: `--input`, `--outdir`, `--asr-compute-type`
- ✅ Paralinguistics columns verified against SEGMENT_COLUMNS
- ✅ Default compute_type: varies by entry point
- ✅ All file paths exist
- ✅ Model names match implementation
- ✅ Output files match
