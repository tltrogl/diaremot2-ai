# DiaRemot repository guidelines

## Overview
- DiaRemot is a CPU-only speech intelligence pipeline that delivers ASR, diarisation, affect scoring, background event tagging, and summary generation.
- The primary entry point is the Typer CLI exposed via `python -m diaremot.cli` or the `diaremot` console script once the package is installed.
- The agent runtime **has internet access**. Use it for documentation lookups or fetching pinned artefacts while keeping the workflow deterministic.

## Environment
- Supported interpreters: CPython 3.10–3.11 (pins validated on 3.11).
- Target x86_64 hosts with AVX2; PyTorch CPU wheels are pulled from `--index-url https://download.pytorch.org/whl/cpu`.
- Ensure `ffmpeg` is present on `PATH` for decoding and sample generation.
- Models live under `${DIAREMOT_MODEL_DIR:-/opt/models}`. Keep this path configurable and mirror any structural changes in the README and setup scripts.

## Setup & maintenance scripts
- `./setup.sh`
  - Creates `.venv`, upgrades `pip/setuptools/wheel`, installs `requirements.txt`, and performs an editable install of the package.
  - Downloads and verifies `models.zip`, unpacking into `$DIAREMOT_MODEL_DIR` (defaults to `/opt/models` with a repository fallback when unwritable).
  - Runs import and diagnostics smoke tests. Invoke via `bash ./setup.sh`, including from PowerShell sessions.
- `./maintenance.sh` / `./maint-codex.sh`
  - Activate `.venv`, validate staged models, and confirm CLI imports. Keep both scripts aligned with the README and diagnostics guidance.
  - `maint-codex.sh` is the lightweight check executed in Codex/agent containers; prefer idempotent checks and informative failure messages.

## Running the pipeline
```bash
python -m diaremot.cli asr run --input data/sample.wav --outdir outputs/run1
# or, after installation:
diaremot asr run --input data/sample.wav --outdir outputs/run1

diaremot asr resume --input data/sample.wav --outdir outputs/run1
python -m diaremot.cli report gen --manifest outputs/run1/manifest.json --format pdf --format html
python -m diaremot.cli system diagnostics --strict
```

## Testing & QA
- Execute `pytest -q` once dependencies are installed.
- Run `python -m diaremot.cli system diagnostics --strict` (or `diaremot-diagnostics --strict`) before shipping changes to confirm dependency versions, ffmpeg presence, and model coverage.
- Keep documentation (`README.md`, scoped `AGENTS.md` files) and dependency manifests (`requirements.txt`, `pyproject.toml`) synchronized with behaviour and automation changes.

## Expectations
- Stage or download model assets during setup; runtime code must not fetch weights implicitly.
- Propagate configuration via `PipelineConfig` instead of duplicating literals across modules.
- Prefer the shared CLI for orchestration over bespoke scripts to preserve consistent UX across environments.
