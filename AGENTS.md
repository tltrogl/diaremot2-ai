# DiaRemot repository guidelines

## Mission profile
DiaRemot delivers a CPU-only speech intelligence pipeline that ingests long-form audio and emits
diarised transcripts, affect overlays, sound-event tags, conversation metrics, and HTML/PDF
briefings. All code and docs in this repository should assume **no GPU availability** and a
containerised Linux host with internet access for documentation lookups and deterministic artifact
downloads.

## Tooling & environment
- Target CPython 3.11 (3.10 remains supported). Keep `requirements.txt` and `pyproject.toml`
in lock-step with the modules imported under `src/diaremot`.
- The pinned wheels target x86_64 CPUs with AVX2; do not introduce GPU-only dependencies.
- Ensure `ffmpeg` is available on `PATH`. Optional helpers such as PyAV improve duration probing but
the pipeline must continue working without them.
- Models live under `$DIAREMOT_MODEL_DIR` (defaults to `/opt/models`). The layout is documented in
the README and mirrored in `setup.sh`/`maintenance.sh`.
- Local caches default to `./.cache` to keep Hugging Face, Torch, and tokenizer data inside the
repository. Respect this when adding new modules.

## Workflows
- `./setup.sh` performs the canonical bootstrap: venv creation, dependency install, editable
package installation, model staging, cache normalisation, and CLI diagnostics. Keep the script and
the README perfectly aligned.
- `./maintenance.sh` and `./maint-codex.sh` are the lightweight health checks for warm containers;
they validate model assets and run the Typer system diagnostics in strict mode.
- Preferred entry points are Typer commands exposed through `python -m diaremot.cli` or the
installed `diaremot` console script. Document new behaviour via these interfaces instead of ad-hoc
scripts.

## Testing & QA
- Run `pytest -q` before shipping changes. Add focused tests under `tests/` for new behaviour.
- Execute `python -m diaremot.cli system diagnostics --strict` to validate dependency health and
model presence.
- Keep documentation (README, AGENTS, setup/maintenance scripts, requirements, pyproject) in sync
with functional changes. Documentation drift is considered a regression.
