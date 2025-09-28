# AGENTS.md — Build / Run / Test (Codex)

## Setup
- **Setup script**: `./setup.sh` (creates venv, installs requirements, stages models)
- **Maintenance**: `./maintenance.sh` (warm container health + model check)
- **Python**: 3.11 recommended (repo pins support 3.9–3.11)
- Ensure `ffmpeg` is available on PATH for audio decoding and chunking.

## Run
```bash
# Preferred Typer CLI group (ASR + diarization + affect)
python -m diaremot.cli asr run --input "data/sample.wav" --outdir "outputs/run1"

# Console-script entrypoint (after editable install)
diaremot asr run --input "data/sample.wav" --outdir "outputs/run1"

# Resume using cached checkpoints
diaremot asr resume --input "data/sample.wav" --outdir "outputs/run1"

# Regenerate summaries without inference
diaremot report gen --manifest "outputs/run1/manifest.json" --format pdf --format html
```
Diagnostics:
```bash
python -m diaremot.cli system diagnostics --strict
# or: diaremot system diagnostics --strict / diaremot-diagnostics --strict
```

## Expectations
- Models live under `$DIAREMOT_MODEL_DIR` (default `/opt/models`) as described in `README.md`.
- Agent runtime has **no internet**; all model fetching must happen in `setup.sh`.
- Exit with non-zero status on failure.
