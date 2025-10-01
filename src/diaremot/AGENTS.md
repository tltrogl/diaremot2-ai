# AGENTS.md — Build / Run / Test (Codex)

## Setup
- **Setup script**: `./setup.sh`  (cold container)
- **Maintenance**: `./maintenance.sh` (warm cache)
- **Python**: 3.11 (repo supports 3.9–3.11 per pins)

## Run — choose ONE and delete the rest
```bash
# Preferred: CLI
python -m diaremot.cli run --input "samples/" --out "outputs/run1"

# If console scripts are installed:
# diaremot run --input "samples/" --out "outputs/run1"

# Alternate (legacy module entry if you call it directly):
# python -m diaremot.pipeline.run_pipeline --input "samples/" --out "outputs/run1"
```
Diagnostics:
```bash
python -m diaremot.cli diagnostics
# or: diaremot-diagnostics
```

## Expectations
- Models present under `$DIAREMOT_MODEL_DIR` (default `/opt/models`) per README.
- Agent has **no internet**; all fetching is done in `setup.sh`.
- On failure, exit non‑zero.






