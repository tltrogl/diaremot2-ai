# AGENTS.md — DiaRemot Agent Instructions (Codex / AI Agents)

_Last updated: 2025‑10‑04_

**Role:** System Architect / Maintainer for DiaRemot
As the agent, you must **plan → implement → verify → report** in each change cycle. You are building *real code*, not mocks.

---

## Truth & Integrity (non‑negotiable)
- Only produce **correct, factual, non‑fabricated** outputs.
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

## Pipeline Contract (must remain true)
**Every run should include all 12 stages by default**, unless explicitly overridden.

The canonical stage list is defined in `src/diaremot/pipeline/stages/__init__.py::PIPELINE_STAGES`:

1. **dependency_check** — Validate runtime dependencies (transformers, onnxruntime, etc.)
2. **preprocess** — Audio normalization, denoising, auto-chunking (for files >30 min)
3. **auto_tune** — Adaptive VAD parameter tuning based on audio characteristics
4. **background_sed** — PANNs CNN14 ONNX, frame = 1.0 s, hop = 0.5 s; thresholds: enter 0.50, exit 0.35; min_dur 0.30 s; merge_gap 0.20 s. Collapse AudioSet → ~20 groups.
5. **diarize** — Silero VAD → ECAPA‑TDNN embeddings → AHC clustering. **Critical:** Default parameters in orchestrator:
   - `vad_threshold = 0.22` (relaxed from 0.30)
   - `vad_min_speech_sec = 0.40` (relaxed from 0.80)
   - `vad_min_silence_sec = 0.40` (relaxed from 0.80)
   - `speech_pad_sec = 0.15` (relaxed from 0.20)
   - `ahc_distance_threshold = 0.02` (orchestrator override; DiarizationConfig default is 0.12)
   - `collar_sec = 0.25`
   - `min_turn_sec = 1.50`
6. **transcribe** — `tiny.en` via faster-whisper in CTranslate2, default `compute_type = int8` (config default; transcriber fallback is `float32`). Parameters: `beam_size` = 1, `temperature = 0.0`, `no_speech_threshold = 0.50`. Runs on diarized turns only.
7. **paralinguistics** — Via Praat‑Parselmouth: jitter, shimmer, HNR, CPPS; plus prosody (WPM, duration_s, words, pause_count, pause_time_s, pause_ratio, f0_mean_hz, f0_std_hz, loudness_rms, disfluency_count).
8. **affect_and_assemble** — Audio affect (V/A/D, 8-class SER), text analysis (GoEmotions 28 via `roberta-base-go_emotions`, Intent zero-shot via `bart-large-mnli` or ONNX). Assembles final segment records.
9. **overlap_interruptions** — Turn-taking analysis, interruption detection, overlap statistics.
10. **conversation_analysis** — Flow metrics (turn-taking balance, response latencies, dominance).
11. **speaker_rollups** — Per-speaker summaries (total duration, V/A/D averages, emotion mix, WPM, voice quality).
12. **outputs** — Write final files:
    - `diarized_transcript_with_emotion.csv` (39 columns)
    - `segments.jsonl`
    - `speakers_summary.csv`
    - `summary.html`
    - `summary.pdf` (optional, requires wkhtmltopdf)
    - `speaker_registry.json`
    - `events_timeline.csv` & `events.jsonl`
    - `timeline.csv`, `qc_report.json`

**Schema guidance:** The canonical segment schema is defined in `src/diaremot/pipeline/outputs.py::SEGMENT_COLUMNS` (39 columns). You must conform exactly to those column names unless extending forward-compatibly.

**Model assets / file paths:**
Default ONNX / label files expected under `DIAREMOT_MODEL_DIR`, e.g. `ecapa_tdnn.onnx`, `panns_cnn14.onnx`, `audioset_labels.csv`. Missing assets should lead to warning logs or fallback to *disabling SED*, *not* silent failure.

**CLI / entry point contract:**
Do not break or rename:
- `python -m diaremot.pipeline.run_pipeline`
- `python -m diaremot.pipeline.cli_entry`
- `python -m diaremot.cli` (Typer app)

---

## Operating Procedure (Plan→Implement→Verify→Report)
1. **Plan**: 5–10 bullets including files touched, signatures, data shapes, test plan.
2. **Implement**: minimal diff; keep module boundaries; consistent style.
3. **Verify**: run tests, lint, build, and smoke-run sample audio.
4. **Report** (single response):
   - Short summary (1–2 paragraphs)
   - Diffs/patch list
   - Commands run + exit codes
   - Key logs (tail ~200 lines)
   - Generated artifact paths (CSV, HTML, JSON)
   - Risks, assumptions, follow-up notes

If any stage fails, fix before reporting. Do not produce incomplete code or half-baked logs.

---

## Research / Dependency Guidelines
- Use **official docs / primary sources** (model readmes, HF, ONNX).
- Summarize findings succinctly with "Source of truth" notes.
- Do not quietly download multi-GB assets; require explicit caching.
- When pinning or updating dependencies, show exact `pip` resolution, version before/after, and justify trade-offs.

---

## Hard Constraints & Safety
- No GPU usage; no system‑level installs (apt, brew).
- No secrets/keys in code or logs.
- Must preserve behavior of existing functions and modules (extend, not remove).
- ASR must default to `compute_type = int8`, unless a compelling, benchmarked improvement is documented.
- Do not rename or break output schemas or filenames without coordinated migration steps.
- All agent actions must be auditably reproducible — logs and commands must align with outputs.

---

## Reporting Checklist (always include)
- ✅ Only factual, reproducible changes; no hallucinated logs
- ✅ Ruff / lint / tests passed (report summary)
- ✅ Full pipeline run (all 12 stages) completed
- ✅ No broken stage; no regression introduced
- ✅ Schema maintained or extended forward‑compatibly
- ✅ SED label collapse preserved
- ✅ All assumptions, risks, version bumps, file paths documented
- ✅ No private credentials or secrets in artifacts or logs

---

## Example directive & expected planning style

> **Directive:** "Add jitter/shimmer/HNR/CPPS extraction to paralinguistics, integrate into segment schema, write basic tests for silent audio."

Your plan response might look like:

```
Plan:
- Modify `affect/paralinguistics.py` to call Parselmouth APIs and compute jitter/shimmer/HNR/CPPS.
- Update `pipeline/outputs.py` → append new columns to `SEGMENT_COLUMNS`.
- In `pipeline/cli_entry` / orchestrator, ensure paralinguistics stage is called.
- Write unit tests: e.g. a synthetic 1-second tone and check zero jitter/shimmer outputs.
- Add CSV parsing integration test: run sample audio pipeline and verify new columns exist.
- Lint / typecheck / build.

Then implement, verify, report with logs/patches.
```
