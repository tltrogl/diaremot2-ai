# AGENTS.md — Build / Run / Test (Codex)

## Setup
- **Setup script**: `./setup.sh`  (cold container)
- **Maintenance**: `./maint-codex.sh` (warm container check)
- **Python**: 3.11 (repo pins support 3.9–3.11)

## Run — choose ONE
```bash
# Preferred: CLI (single file)
python -m diaremot.cli run --input "data/sample.wav" --outdir "outputs/run1"

# Or: directory of wavs
# python -m diaremot.cli run --input "samples/" --outdir "outputs/run1"
```

Diagnostics:
```bash
python -m diaremot.cli diagnostics
# or: diaremot-diagnostics
```

## Expectations
- Models present under `$DIAREMOT_MODEL_DIR` (default `/opt/models`) per README.
- Agent has **no internet**; all fetching is done in `setup.sh`.
- On failure, exit non-zero.
