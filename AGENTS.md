# AGENTS.md — DiaRemot Agent Instructions (Codex / AI Agents)

_Last updated: 2025-10-04_

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

## Pipeline Architecture: ONNX-Only Inference

**CRITICAL:** DiaRemot is a **CPU-only, ONNX-first** pipeline. All inference must use:
- **ONNXRuntime** for audio/text models (VAD, embeddings, emotion, intent)
- **CTranslate2** for ASR (faster-whisper)
- **NO PyTorch inference** (PyTorch used only for preprocessing/feature extraction)
- **NO HuggingFace pipeline()** calls for inference

### Rationale
- ONNX: 2-5x faster CPU inference vs PyTorch
- CTranslate2: Optimized Transformer inference (int8 quantization)
- Reduced memory footprint for long-form audio (1-3 hours)
- Deployment-ready: no torch.jit, no model.eval() calls

---

## Pipeline Contract (must remain true)
**Every run should include all 12 stages by default**, unless explicitly overridden.

The canonical stage list is defined in `src/diaremot/pipeline/stages/__init__.py::PIPELINE_STAGES`:

### 1. **dependency_check**
Validate runtime dependencies:
- `onnxruntime >= 1.16.0`
- `faster-whisper >= 1.0.0` (includes CTranslate2)
- `transformers` (tokenizers only, no inference)
- Praat-Parselmouth

### 2. **preprocess**
Audio normalization, denoising, auto-chunking (for files >30 min)

### 3. **auto_tune**
Adaptive VAD parameter tuning based on audio characteristics

### 4. **background_sed** (Sound Event Detection)
**Model:** PANNs CNN14 ONNX (`panns_cnn14.onnx`)  
**Runtime:** ONNXRuntime  
**Parameters:**
- Frame: 1.0 s
- Hop: 0.5 s
- Thresholds: enter 0.50, exit 0.35
- Min duration: 0.30 s
- Merge gap: 0.20 s
- Label collapse: AudioSet 527 → ~20 semantic groups (speech, music, laughter, crying, door, phone, etc.)

**Assets required:**
- `panns_cnn14.onnx` (118 MB)
- `audioset_labels.csv` (527 class labels)

### 5. **diarize** (Speaker Segmentation)
**VAD:** Silero VAD ONNX (`silero_vad.onnx`)  
**Embeddings:** ECAPA-TDNN ONNX (`ecapa_tdnn.onnx`)  
**Clustering:** Agglomerative Hierarchical Clustering (AHC)  
**Runtime:** ONNXRuntime for both VAD and embeddings

**Default parameters** (orchestrator overrides):
- `vad_threshold = 0.22` (relaxed from CLI default 0.30)
- `vad_min_speech_sec = 0.40` (relaxed from 0.80)
- `vad_min_silence_sec = 0.40` (relaxed from 0.80)
- `speech_pad_sec = 0.15` (relaxed from 0.20)
- `ahc_distance_threshold = 0.02` (orchestrator override; DiarizationConfig default is 0.12)
- `collar_sec = 0.25`
- `min_turn_sec = 1.50`

**Assets required:**
- `silero_vad.onnx` (1.8 MB)
- `ecapa_tdnn.onnx` (6.1 MB)

### 6. **transcribe** (ASR)
**Model:** faster-whisper `tiny.en` (39 MB)  
**Runtime:** CTranslate2 (int8 quantization by default)  
**Parameters:**
- `beam_size = 1` (greedy decoding)
- `temperature = 0.0` (deterministic)
- `no_speech_threshold = 0.50`
- `compute_type = int8` (default; fallback to float32 if int8 unavailable)
- `vad_filter = True` (uses built-in Silero VAD)

**Runs on:** Diarized speech turns only (not full audio)

**CRITICAL:** ASR uses CTranslate2, NOT ONNX. This is the only stage exempt from ONNX requirement.

### 7. **paralinguistics** (Voice Quality + Prosody)
**Runtime:** Praat-Parselmouth (native C++ library, not ONNX)  
**Metrics extracted:**
- Voice quality: jitter (%), shimmer (dB), HNR (dB), CPPS (dB)
- Prosody: WPM, duration_s, words, pause_count, pause_time_s, pause_ratio
- Pitch: f0_mean_hz, f0_std_hz
- Loudness: loudness_rms
- Disfluencies: disfluency_count

**Fallback:** If Praat fails, compute WPM from ASR text and set voice quality metrics to 0.0

### 8. **affect_and_assemble** (Emotion + Intent)
**Audio emotion:** 8-class Speech Emotion Recognition ONNX  
**VAD emotion:** Valence/Arousal/Dominance ONNX  
**Text emotion:** GoEmotions 28-class ONNX (`roberta-base-go_emotions.onnx`)  
**Intent:** Zero-shot classification ONNX (`bart-large-mnli.onnx`)  
**Runtime:** ONNXRuntime for all models

**CRITICAL:** Do NOT use HuggingFace `pipeline()` or PyTorch for inference. All text models must be ONNX.

**Assets required:**
- `ser_8class.onnx` (audio emotion)
- `vad_model.onnx` (valence/arousal/dominance)
- `roberta-base-go_emotions.onnx` (text emotion)
- `bart-large-mnli.onnx` (intent classification)

### 9. **overlap_interruptions**
Turn-taking analysis, interruption detection, overlap statistics

### 10. **conversation_analysis**
Flow metrics (turn-taking balance, response latencies, dominance)

### 11. **speaker_rollups**
Per-speaker summaries (total duration, V/A/D averages, emotion mix, WPM, voice quality)

### 12. **outputs**
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

**Missing assets should:**
1. Log warning (not silent failure)
2. Fallback to disabling that stage (e.g., skip SED if `panns_cnn14.onnx` missing)
3. Never crash the pipeline

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
  --vad-threshold 0.22 \
  --compute-type int8
```

---

## Operating Procedure (Plan→Implement→Verify→Report)

### 1. Plan (5-10 bullets)
- Files touched
- Signatures changed
- Data shapes / schemas affected
- Test plan
- ONNX model conversion steps (if adding new models)

### 2. Implement
- Minimal diff
- Keep module boundaries
- Consistent style (ruff-compliant)
- **No PyTorch inference** (only ONNX/CTranslate2)

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
4. Benchmark inference time (CPU-only)
5. Add to `DIAREMOT_MODEL_DIR` manifest

---

## Hard Constraints & Safety

- **No GPU usage** — CPU-only execution
- **No system-level installs** (apt, brew)
- **No secrets/keys** in code or logs
- **Must preserve behavior** of existing functions and modules (extend, not remove)
- **ASR must default to `compute_type = int8`** unless benchmarked improvement documented
- **Do not rename or break output schemas** or filenames without coordinated migration steps
- **All agent actions must be auditably reproducible** — logs and commands must align with outputs
- **ONNX-first for all inference** except ASR (CTranslate2) and paralinguistics (Praat)

---

## Reporting Checklist (always include)

- ✅ Only factual, reproducible changes; no hallucinated logs
- ✅ Ruff / lint / tests passed (report summary)
- ✅ Full pipeline run (all 12 stages) completed
- ✅ No broken stage; no regression introduced
- ✅ Schema maintained or extended forward-compatibly
- ✅ SED label collapse preserved
- ✅ All ONNX models validated with ONNXRuntime
- ✅ No PyTorch inference introduced
- ✅ All assumptions, risks, version bumps, file paths documented
- ✅ No private credentials or secrets in artifacts or logs

---

## Example Directive & Expected Planning Style

> **Directive:** "Add zero-shot emotion classification using ONNX model instead of HuggingFace pipeline."

Your plan response might look like:

```
Plan:
- Convert HuggingFace `roberta-base-go_emotions` to ONNX using optimum-cli
- Modify `affect/text_analyzer.py` to load ONNX model via ONNXRuntime
- Replace `pipeline("text-classification")` calls with ONNX inference
- Tokenize text using HuggingFace tokenizer (fast, no inference)
- Update `pipeline/config.py` to reference ONNX model path
- Benchmark inference time: HF pipeline vs ONNX (expect 2-3x speedup)
- Write unit test: verify output schema matches old HF pipeline
- Integration test: run full pipeline on sample audio, verify CSV schema unchanged
- Document ONNX conversion steps in README
- Lint / typecheck / build

Then implement, verify, report with logs/patches.
```

---

## Common Pitfalls to Avoid

1. **Using HuggingFace `pipeline()` for inference** → Use ONNX instead
2. **Loading PyTorch models with `model.eval()`** → Convert to ONNX
3. **Not validating ONNX model outputs** → Always test against reference implementation
4. **Silent failures when ONNX model missing** → Log warning and disable stage gracefully
5. **Hardcoding model paths** → Use `DIAREMOT_MODEL_DIR` environment variable
6. **Not benchmarking ONNX conversion** → Always compare inference time vs PyTorch
7. **Breaking schema when adding ONNX models** → Maintain 39-column CSV contract

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

**Remember:** This is production code serving real users. Precision matters. ONNX-first is non-negotiable. When in doubt, verify before claiming.
