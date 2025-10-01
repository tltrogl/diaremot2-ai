# AGENTS.md — DiaRemot (CPU‑only Speech ML)

**Paralinguistics is REQUIRED.** Use Praat‑Parselmouth for jitter/shimmer/HNR/CPPS + prosody (WPM/pauses).

## Setup & maintenance
```bash
./setup.sh
./maint-codex.sh
```

## Run
```bash
python -m diaremot.cli diagnostics --strict
python -m diaremot.cli run --audio data/sample.wav --tag smoke --compute-type int8
```

## Style & checks
- Ruff: `ruff format .` and `ruff check --fix .`
- mypy: `mypy src tests`
- pytest: `pytest -q`

## REQUIRED tasks
- Paralinguistics: `affect/paralinguistics.py` must compute WPM/pauses and jitter/shimmer/HNR/CPPS; emit
  `vq_jitter_pct,vq_shimmer_db,vq_hnr_db,vq_cpps_db,voice_quality_hint`.
- SED collapse: AudioSet→~20 groups in `sed_panns.py` before `events_top3_json`.
- ASR default: `compute_type=int8` on CPU.