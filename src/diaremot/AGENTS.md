# AGENTS.md — DiaRemot package guidance

## Scope
These instructions cover code under `src/diaremot/` including the Typer CLI, pipeline modules, affect models, and report generators.

## Setup & environment
- Target Python 3.10–3.11 (validated on 3.11). Keep dependency changes mirrored in `pyproject.toml` and `requirements.txt`.
- Ensure CPU execution paths stay deterministic and compatible with the pinned CPU PyTorch wheels.
- Expect models under `$DIAREMOT_MODEL_DIR` (defaults to `/opt/models`). Do not hard-code other paths.

## CLI conventions
- Prefer Typer entry points defined in `diaremot.cli`:
  - `python -m diaremot.cli asr run --input <file> --outdir <dir>` for the end-to-end pipeline.
  - `python -m diaremot.cli asr resume --input <file> --outdir <dir>` to continue cached work.
  - `python -m diaremot.cli report gen --manifest <manifest> --format pdf --format html` for summary regeneration.
- Maintain backward-compatible aliases only when necessary; deprecate unused flags with clear warnings before removal.

## Testing expectations
- Keep `pytest -q` passing once dependencies are installed; write tests under `tests/` alongside fixtures.
- Provide meaningful logging and exceptions so diagnostics surfaces actionable guidance.

## Documentation discipline
- Update module docstrings, README sections, and CLIs whenever behaviour changes.
- Reference shared constants/configuration to avoid drift across subsystems (e.g., reuse `PipelineConfig`).
