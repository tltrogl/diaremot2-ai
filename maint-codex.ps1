#!/usr/bin/env bash
# DiaRemot — maintenance for Codex Cloud (internet ON)
# format → lint → typecheck → test → build → diagnostics → smoke-run → CSV schema check
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

# ---------- Type check ----------
log "Type check (mypy)"
if command -v mypy >/dev/null 2>&1; then
  mypy src || true
else
  echo "::notice:: mypy not installed; skipping type check"
fi

# ---------- Tests ----------
log "Tests (pytest)"
if command -v pytest >/dev/null 2>&1; then
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

# ---------- Smoke run (synthesize audio → full pipeline) ----------
SMOKE_DIR="${SMOKE_DIR:-/tmp/diaremot_smoke}"
SMOKE_WAV="$SMOKE_DIR/audio.wav"
mkdir -p "$SMOKE_DIR"

log "Synthesize 16 kHz mono WAV (silence→tone→noise→silence) at: $SMOKE_WAV"
$PYTHON - <<'PY'
import numpy as np, soundfile as sf, os
sr = 16000
def seg_silence(d): return np.zeros(int(d*sr), dtype=np.float32)
def seg_tone(d, f=220.0, a=0.15):
    t = np.arange(int(d*sr))/sr
    return (a*np.sin(2*np.pi*f*t)).astype(np.float32)
def seg_noise(d, a=0.02):
    return (a*np.random.randn(int(d*sr))).astype(np.float32)

x = np.concatenate([
    seg_silence(1.0),
    seg_tone(3.0, 220.0, 0.15),
    seg_noise(1.0, 0.02),
    seg_silence(1.0),
])
os.makedirs(os.environ["SMOKE_DIR"], exist_ok=True)
sf.write(os.environ["SMOKE_DIR"] + "/audio.wav", x, sr, subtype="PCM_16")
print("wrote:", os.environ["SMOKE_DIR"] + "/audio.wav")
PY

log "Run pipeline (CPU-only, int8 ASR) on smoke WAV"
set +e
$PYTHON -m diaremot.cli run \
  --audio "$SMOKE_WAV" \
  --tag smoke \
  --compute-type int8
PIPE_STATUS=$?
set -e
if [[ $PIPE_STATUS -ne 0 ]]; then
  echo "::error:: pipeline run failed on smoke sample (exit $PIPE_STATUS)"
  exit 1
fi

# Try to locate the newest primary CSV
log "Locate output CSV"
: "${OUTPUT_CSV:=}"
if [[ -z "${OUTPUT_CSV}" ]]; then
  # newest diarized_transcript_with_emotion.csv anywhere under repo
  mapfile -t CANDIDATES < <(ls -1t **/diarized_transcript_with_emotion.csv 2>/dev/null || true)
  if [[ ${#CANDIDATES[@]} -gt 0 ]]; then
    OUTPUT_CSV="${CANDIDATES[0]}"
    echo "::notice:: using OUTPUT_CSV=${OUTPUT_CSV}"
  else
    echo "::error:: No diarized_transcript_with_emotion.csv found after smoke run"
    exit 2
  fi
fi
export OUTPUT_CSV

# ---------- CSV Schema Guard ----------
log "CSV schema check"
REQUIRED_BASE="file_id,start,end,speaker_id,speaker_name,text,valence,arousal,dominance,emotion_top,emotion_scores_json,text_emotions_top5_json,text_emotions_full_json,intent_top,intent_top3_json,events_top3_json,noise_tag,asr_logprob_avg,snr_db,snr_db_sed,wpm,duration_s,words,pause_ratio,vq_jitter_pct,vq_shimmer_db,vq_hnr_db,vq_cpps_db,voice_quality_hint"
# Allow override or additions
: "${EXTRA_REQUIRED_COLS:=}"
if [[ -n "$EXTRA_REQUIRED_COLS" ]]; then
  REQUIRED_BASE="${REQUIRED_BASE},${EXTRA_REQUIRED_COLS}"
fi
export REQUIRED_BASE

$PYTHON - <<'PY'
import csv, sys, os
csv_path = os.environ["OUTPUT_CSV"]
required = [c.strip() for c in os.environ["REQUIRED_BASE"].split(",") if c.strip()]
try:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
except FileNotFoundError:
    print(f"::error:: CSV not found: {csv_path}")
    sys.exit(3)

if not header:
    print(f"::error:: CSV is empty: {csv_path}")
    sys.exit(3)

missing = [c for c in required if c not in header]
if missing:
    print("::error:: Missing required columns:", ", ".join(missing))
    print("CSV:", csv_path)
    print("Header:", ",".join(header))
    sys.exit(4)

print(f"::notice:: CSV schema OK: {csv_path}")
PY

log "maint-codex.sh finished successfully."
