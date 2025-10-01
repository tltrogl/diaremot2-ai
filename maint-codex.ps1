#!/usr/bin/env bash
# DiaRemot — maintenance for Codex Cloud (internet ON)
# format → lint → test → build → diagnostics → CSV schema check
set -Eeuo pipefail

log() { printf '\n==> %s\n' "$*"; }
json() { python - <<'PY'
import json, os, sys, platform, shutil
print(json.dumps({
  "python_version": sys.version.split()[0],
  "platform": platform.platform(),
  "ffmpeg_on_path": bool(shutil.which("ffmpeg")),
  "ruff_on_path": bool(shutil.which("ruff")),
  "pytest_on_path": bool(shutil.which("pytest")),
  "mypy_on_path": bool(shutil.which("mypy")),
  "hf_home": os.environ.get("HF_HOME"),
  "transformers_cache": os.environ.get("TRANSFORMERS_CACHE"),
  "torch_home": os.environ.get("TORCH_HOME"),
  "omp_threads": os.environ.get("OMP_NUM_THREADS"),
}, indent=2))
PY
}

: "${PYTHON:=python}"

# ---------- Formatting & Lint ----------
log "Format (ruff)"
if command -v ruff >/dev/null 2>&1; then
  ruff format .
  ruff check --fix .
else
  echo "::notice:: ruff not installed; skipping format/lint"
fi

log "Type check (mypy)"
if command -v mypy >/dev/null 2>&1; then
  mypy src || true
else
  echo "::notice:: mypy not installed; skipping type check"
fi

# ---------- Tests ----------
log "Tests (pytest)"
if command -v pytest >/dev/null 2>&1; then
  # -q keeps logs compact; remove -q if you need verbose
  pytest -q || (echo "::error:: pytest failed" && exit 1)
else
  echo "::notice:: pytest not installed; skipping tests"
fi

# ---------- Build ----------
log "Build"
$PYTHON -m build || true

# ---------- Diagnostics ----------
log "Diagnostics"
json

# ---------- CSV Schema Guard ----------
# This guard ensures the primary CSV contains all required columns.
# You can specify the CSV explicitly via OUTPUT_CSV, or we’ll try to find the common filename.
#
# Env knobs:
#   OUTPUT_CSV=/path/to/diarized_transcript_with_emotion.csv
#   ALLOW_MISSING_CSV=true   # if true, skip check when no CSV is found
#   EXTRA_REQUIRED_COLS="colA,colB"  # comma-separated additional required columns
#
log "CSV schema check"
: "${ALLOW_MISSING_CSV:=false}"
: "${OUTPUT_CSV:=}"

# Find a plausible output CSV if not provided
if [[ -z "${OUTPUT_CSV}" ]]; then
  # Search by typical name anywhere under repo, newest first
  mapfile -t CANDIDATES < <(ls -1t **/diarized_transcript_with_emotion.csv 2>/dev/null || true)
  if [[ ${#CANDIDATES[@]} -gt 0 ]]; then
    OUTPUT_CSV="${CANDIDATES[0]}"
    echo "::notice:: OUTPUT_CSV not set; using ${OUTPUT_CSV}"
  fi
fi

REQUIRED_BASE="file_id,start,end,speaker_id,speaker_name,text,valence,arousal,dominance,emotion_top,emotion_scores_json,text_emotions_top5_json,text_emotions_full_json,intent_top,intent_top3_json,events_top3_json,noise_tag,asr_logprob_avg,snr_db,snr_db_sed,wpm,duration_s,words,pause_ratio,vq_jitter_pct,vq_shimmer_db,vq_hnr_db,vq_cpps_db,voice_quality_hint"
# Append optional extras if requested
if [[ -n "${EXTRA_REQUIRED_COLS:-}" ]]; then
  REQUIRED_BASE="${REQUIRED_BASE},${EXTRA_REQUIRED_COLS}"
fi

if [[ -z "${OUTPUT_CSV}" ]]; then
  if [[ "${ALLOW_MISSING_CSV}" == "true" ]]; then
    echo "::notice:: No output CSV found; skipping schema check"
    exit 0
  else
    echo "::error:: No output CSV found and ALLOW_MISSING_CSV=false"
    exit 2
  fi
fi

# Inline Python to validate header
$PYTHON - <<PY
import csv, sys, os
csv_path = os.environ.get("OUTPUT_CSV")
required = [c.strip() for c in os.environ["REQUIRED_BASE"].split(",") if c.strip()]
if not os.path.isfile(csv_path):
    print(f"::error:: CSV not found: {csv_path}")
    sys.exit(3)
with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    try:
        header = next(reader)
    except StopIteration:
        print(f"::error:: CSV is empty: {csv_path}")
        sys.exit(3)

missing = [c for c in required if c not in header]
if missing:
    print("::error:: Missing required columns:", ", ".join(missing))
    print("CSV:", csv_path)
    print("Header:", ",".join(header))
    sys.exit(4)

# Extra: sanity on ordering (warn only)
order_warnings = []
for col in ("file_id","start","end","speaker_id","speaker_name","text"):
    if col in header and header.index(col) > 10:
        order_warnings.append(col)
if order_warnings:
    print(f"::notice:: Unusual column ordering for: {', '.join(order_warnings)}")

print(f"::notice:: CSV schema OK: {csv_path}")
PY
