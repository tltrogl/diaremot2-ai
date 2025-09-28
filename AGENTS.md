# AGENTS.md — DiaRemot repository guidelines

## Overview
- DiaRemot is a CPU-only speech intelligence pipeline that performs ASR, diarization, affect analysis, sound event detection, and summary generation.
- Primary tooling lives under `src/diaremot/` with Typer-powered CLIs exposed via `python -m diaremot.cli` or the `diaremot` console script.
- The agent runtime **has internet access**; use it responsibly for documentation lookups or artifact downloads while keeping runs deterministic.

## Environment
- Python 3.10–3.11 supported; dependency pins are validated with CPython 3.11 and that version is preferred for development and CI.
- Execution assumes x86_64 CPUs with AVX2 so the PyTorch CPU wheels function correctly.
- Ensure `ffmpeg` is on `PATH` so audio decoding, resampling, and sample generation work as expected.
- Models should reside under `$DIAREMOT_MODEL_DIR` (defaults to `/opt/models`). See the README for the exact directory layout.

## Setup
- `./setup.sh` — full bootstrap: creates `.venv`, installs `requirements.txt`, stages `models.zip`, normalises caches, and validates imports. Invoke with `bash ./setup.sh` when running from PowerShell.
- `./maintenance.sh` — lightweight re-validation for warm containers (checks models + imports without reinstalling dependencies).
- Both scripts mirror the documented manual steps; keep them aligned with the README whenever workflow changes.

## Running the pipeline
```bash
# Preferred Typer CLI usage (works via module or console script)
python -m diaremot.cli asr run --input "data/sample.wav" --outdir "outputs/run1"
# After editable/install step
# diaremot asr run --input "data/sample.wav" --outdir "outputs/run1"

# Resume cached checkpoints
python -m diaremot.cli asr resume --input "data/sample.wav" --outdir "outputs/run1"

# Regenerate reports without re-running inference
python -m diaremot.cli report gen --manifest "outputs/run1/manifest.json" --format pdf --format html
```
Diagnostics:
```bash
python -m diaremot.cli system diagnostics --strict
# Console script aliases also exist: diaremot system diagnostics --strict / diaremot-diagnostics --strict
```

## Testing & QA
- Run `pytest -q` for unit coverage once dependencies are installed.
- Execute `python -m diaremot.cli system diagnostics --strict` before shipping changes to confirm dependency and model health.
- Keep documentation (README, AGENTS.md, requirements, pyproject) synchronised with functional updates.

## Expectations
- Stage or download model assets during setup; runtime code should not attempt to fetch models dynamically.
- Fail fast: surface non-zero exit codes on errors and ensure logging clearly communicates missing prerequisites.
- Prefer Typer CLI pathways over bespoke scripts so tooling remains consistent across local and automated environments.
