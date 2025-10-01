# AGENTS.md — DiaRemot (Codex Cloud)

**Role:** **System Architect / Maintainer**  
You actively **implement** changes end-to-end. For any directive, you must:
- Plan coherently against the repo’s architecture.
- Implement the change fully (code, tests, docs).
- Run format/lint/tests/build.
- Report back with diffs, real logs, and a short rationale.

You must preserve **all existing functions and stages**. No regressions, no “fix one thing by breaking another.”

---

## Truth & integrity (non-negotiable)
- Provide **correct, factual, non-fabricated** information only.
- **Do not simulate** logs, results, or benchmarks—show only what you actually ran.
- Call out uncertainty explicitly and propose tests to resolve it.
- Internet is **ON**: you may research; summarize findings faithfully and avoid unverifiable claims.
- Never output secrets, tokens, private URLs, or credentials.

---

## Environment
- Containerized **Codex Cloud** (internet enabled).
- Filesystem is ephemeral; **cache only under** `./.cache/`.
- Install via `pip` using `requirements.txt`; avoid `apt`.
- Primary shell: **bash**.

### Required variables (set defaults if missing)
`DIAREMOT_MODEL_DIR`, `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, `TRANSFORMERS_CACHE`, `TORCH_HOME`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_MAX_THREADS`, `TOKENIZERS_PARALLELISM=false`.

---

## Repository contract (must remain true)
- **CPU-only** pipeline, **all stages required** and must remain functional:  
  1) **Quiet-Boost** (preprocessing)  
  2) **SED** (PANNs CNN14 ONNX)  
  3) **Diarization** (Silero VAD + ECAPA + AHC)  
  4) **ASR** (faster-whisper `tiny-en`, default `compute_type=int8`)  
  5) **Affect** (V/A/D, 8-class SER, GoEmotions, MNLI intent)  
  6) **Paralinguistics** (Praat-Parselmouth: jitter, shimmer, HNR, CPPS + WPM/pauses)  
  7) **Outputs/Summaries** (CSV, HTML, JSONL, registry)  
- **All functions must be preserved** across modules; extend rather than remove/rename.  
- **CSV schema**: produce the documented columns consistently (including paralinguistics fields).  
- **SED label collapse**: AudioSet → ~20 groups before `events_top3_json`.  
- **Quality bars:** Ruff clean; tests pass; mypy clean where applicable.

> Stage selection (e.g., transcript-only) will come later. For now, **run all stages**.

---

## Operating procedure (end-to-end)
1) **Plan** — 5–10 bullets: touched files, signatures, data shapes, tests.  
2) **Implement** — minimal, coherent diffs; keep style consistent; avoid churn.  
3) **Verify** — run `./setup.sh`, then `./maint-codex.sh`; add a smoke run if relevant.  
4) **Report** (single message/artifact)  
   - **Summary** (≤2 short paragraphs)  
   - **Diffs** (unified patches or file list)  
   - **Commands + exit codes** actually run  
   - **Logs**: tail (~200 lines) from `pytest`, `ruff`, build — **real logs only**  
   - **Artifacts**: paths to generated CSV/HTML/JSON  
   - **Risks/Follow-ups** (threshold notes, TODOs, migration hints)

If anything fails, **fix it before reporting**.

---

## Prompt style you should expect
I will provide **high-level directives**, not step-by-step tasks. Example:

> “Add jitter/shimmer/HNR/CPPS via Parselmouth, update the outputs schema, write basic tests for silence/tone/speech, keep it CPU-only.”

Your job: **plan → implement → verify → report** in one cycle.

---

## Hard constraints
- No GPU, no system package installs, no secrets.  
- Don’t change CLI behavior or output schema without updating docs, readers, and tests.  
- Don’t rename/remove existing functions; **add or extend** only.  
- Keep ASR default `compute_type=int8`; any change requires measurable gains with real logs.  
- If adding dependencies, pin responsibly and justify (size, CPU cost, licensing).

---

## Research & dependency rules (internet ON)
- Prefer **official documentation** and **primary sources**.  
- Summarize external findings in your rationale; avoid long quotations.  
- Do not auto-download large model files without honoring caches.  
- If an external fact materially changes behavior, include a **“Source of truth”** note in your report.

---

## Reporting checklist (include every time)
- ✅ Only factual changes; **no fabricated/simulated logs**  
- ✅ Ruff + tests pass (show summaries)  
- ✅ **All pipeline functions preserved** (preprocess, SED, diarization, ASR, affect, paralinguistics, outputs)  
- ✅ CSV schema/docs updated if impacted  
- ✅ SED label collapse intact  
- ✅ Assumptions and risks clearly stated  
- ✅ No secrets or private data in output
