#!/usr/bin/env bash
# DiaRemot — setup for Codex Cloud (internet ON, pinned deps, robust installs)
set -Eeuo pipefail

log() { printf '\n==> %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT" || die "cannot cd to repo root"
: "${PYTHON:=python}"

# ---------- Write pinned requirements ----------
# Core pins chosen for CPU wheels & interop:
# - numba 0.62.1 <-> llvmlite 0.45.0
# - transformers 4.56.2 <-> tokenizers 0.22.1
# - onnxruntime 1.23.0 (manylinux_2_27+)
# - ctranslate2 4.6.0 + faster-whisper 1.2.0
# - numpy 2.3.3, scipy 1.16.2 (ABI compatible wheels)
cat > requirements.txt <<'REQ'
ctranslate2==4.6.0
faster-whisper==1.2.0
onnxruntime==1.23.0
numpy==2.3.3
scipy==1.16.2
librosa==0.11.0
soundfile==0.13.1
transformers==4.56.2
tokenizers==0.22.1
huggingface_hub==0.35.3
scikit-learn==1.7.2
reportlab==4.4.4
ffmpeg-python==0.2.0
typer==0.19.2
praat-parselmouth==0.4.6
av==15.1.0
soxr==1.0.0
tqdm==4.67.1
coloredlogs==15.0.1
flatbuffers==25.9.23
protobuf==6.32.1
sympy==1.14.0
filelock==3.19.1
requests==2.32.5
audioread==3.0.1
numba==0.62.1
llvmlite==0.45.0
joblib==1.5.2
decorator==5.2.1
pooch==1.8.2
lazy_loader==0.4
msgpack==1.1.1
cffi==2.0.0
regex==2025.9.18
safetensors==0.6.2
threadpoolctl==3.6.0
pillow==11.3.0
charset-normalizer==3.4.3
future==1.0.0
markdown-it-py==4.0.0
mdurl==0.1.2
humanfriendly==10.0
mpmath==1.3.0
REQ

# Optional constraints for extra determinism of transitives pulled by build/editable
cat > constraints.txt <<'CON'
# pin extra transitives if pip tries newer minors; extend as needed
fsspec==2025.9.0
hf-xet==1.1.10
typing-extensions==4.15.0
packaging==25.0
click==8.2.1
rich==14.1.0
shellingham==1.5.4
pooch==1.8.2
CON

# ---------- Caches & CPU threading ----------
mkdir -p "$REPO_ROOT/.cache"/{hf,transformers,torch} || true
export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$REPO_ROOT/.cache/transformers}"
export TORCH_HOME="${TORCH_HOME:-$REPO_ROOT/.cache/torch}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# ---------- pip UX for CI + retries ----------
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-180}"
export PIP_PROGRESS_BAR=off

pip_run() {
  # Usage: pip_run install [-c constraints.txt] <pkgs...>
  local attempt=1 delay=2
  while true; do
    if "$PYTHON" -m pip "$@" --root-user-action=ignore; then
      break
    fi
    if [ $attempt -ge 4 ]; then
      die "pip failed after ${attempt} attempts: $*"
    fi
    log "pip failed (attempt ${attempt}); retrying in ${delay}s..."
    sleep "$delay"
    attempt=$((attempt+1))
    delay=$((delay*2))
  done
}

# ---------- Sanity ----------
log "Python: $($PYTHON -V 2>&1 || echo 'not found')"
"$PYTHON" - <<'PY' || die "Python >=3.11 required"
import sys; assert sys.version_info[:2] >= (3,11), sys.version
PY

log "Upgrading pip/setuptools/wheel"
pip_run install -U pip setuptools wheel

# ---------- Chunked installs (with constraints) ----------
log "Install heavy numeric/runtime wheels"
pip_run install -c constraints.txt \
  "numpy==2.3.3" "scipy==1.16.2" "llvmlite==0.45.0" "numba==0.62.1"

log "Install ONNX/ASR core"
pip_run install -c constraints.txt \
  "onnxruntime==1.23.0" "ctranslate2==4.6.0" "faster-whisper==1.2.0"

log "Install media libs"
pip_run install -c constraints.txt \
  "av==15.1.0" "soundfile==0.13.1" "ffmpeg-python==0.2.0" "soxr==1.0.0"

log "Install NLP stack"
pip_run install -c constraints.txt \
  "tokenizers==0.22.1" "transformers==4.56.2" "huggingface_hub==0.35.3" \
  "typer==0.19.2" "scikit-learn==1.7.2"

log "Install audio analysis & reporting"
pip_run install -c constraints.txt \
  "librosa==0.11.0" "reportlab==4.4.4" "praat-parselmouth==0.4.6"

# Final sync to ensure remaining pins land (idempotent)
log "Sync residual requirements.txt"
pip_run install -c constraints.txt -r requirements.txt

log "Editable install"
pip_run install -c constraints.txt -e .

log "Developer tools (ruff/pytest/mypy/build)"
pip_run install -U ruff pytest mypy build || true

# ---------- FFmpeg provisioning ----------
if ! command -v ffmpeg >/dev/null 2>&1; then
  log "ffmpeg not found; provisioning via imageio-ffmpeg"
  pip_run install -U imageio-ffmpeg
  FFMPEG_BIN="$("$PYTHON" - <<'PY'
import imageio_ffmpeg
print(imageio_ffmpeg.get_ffmpeg_exe())
PY
)"
  if [ -x "$FFMPEG_BIN" ]; then
    export PATH="$(dirname "$FFMPEG_BIN"):$PATH"
    export IMAGEIO_FFMPEG_EXE="$FFMPEG_BIN"
    log "Using bundled FFmpeg at: $FFMPEG_BIN"
  else
    printf '::warning:: imageio-ffmpeg did not expose a binary; some decodes may fail\n'
  fi
else
  log "ffmpeg found on PATH"
fi

# ---------- AGENTS.md ----------
log "Writing AGENTS.md"
cat > "$REPO_ROOT/AGENTS.md" <<'EOF'
# AGENTS.md — DiaRemot (Codex Cloud)

**Role:** **System Architect / Maintainer**  
You actively **implement** changes end-to-end. For any directive, you must:
- Plan coherently against the repo’s architecture.
- Implement the change fully (code, tests, docs).
- Run format/lint/tests/build.
- Report back with diffs, real logs, and a short rationale.

You must preserve **all existing functions and stages**. No regressions, no “fix one thing by breaking another.”

---

## Truth & integrity (non-negotiable)
- Provide **correct, factual, non-fabricated** information only.
- **Do not simulate** logs, results, or benchmarks—show only what you actually ran.
- Call out uncertainty explicitly and propose tests to resolve it.
- Internet is **ON**: you may research; summarize findings faithfully and avoid unverifiable claims.
- Never output secrets, tokens, private URLs, or credentials.

---

## Environment
- Containerized **Codex Cloud** (internet enabled).
- Filesystem is ephemeral; **cache only under** `./.cache/`.
- Install via `pip` using `requirements.txt`; avoid `apt`.
- Primary shell: **bash**.

### Required variables (set defaults if missing)
`DIAREMOT_MODEL_DIR`, `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, `TRANSFORMERS_CACHE`, `TORCH_HOME`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_MAX_THREADS`, `TOKENIZERS_PARALLELISM=false`.

---

## Repository contract (must remain true)
- **CPU-only** pipeline, **all stages required** and must remain functional:
  1) **Quiet-Boost** (preprocessing)
  2) **SED** (PANNs CNN14 ONNX)
  3) **Diarization** (Silero VAD + ECAPA + AHC)
  4) **ASR** (faster-whisper `tiny-en`, default `compute_type=int8`)
  5) **Affect** (V/A/D, 8-class SER, GoEmotions, MNLI intent)
  6) **Paralinguistics** (Praat-Parselmouth: jitter, shimmer, HNR, CPPS + WPM/pauses)
  7) **Outputs/Summaries** (CSV, HTML, JSONL, registry)
- **All functions must be preserved** across modules; extend rather than remove/rename.
- **CSV schema**: produce the documented columns consistently (including paralinguistics fields).
- **SED label collapse**: AudioSet → ~20 groups before `events_top3_json`.
- **Quality bars:** Ruff clean; tests pass; mypy clean where applicable.

> Stage selection (e.g., transcript-only) will come later. For now, **run all stages**.

---

## Operating procedure (end-to-end)
1) **Plan** — 5–10 bullets: touched files, signatures, data shapes, tests.
2) **Implement** — minimal, coherent diffs; keep style consistent; avoid churn.
3) **Verify** — run `./setup.sh`, then `./maint-codex.sh`; add a smoke run if relevant.
4) **Report** (single message/artifact)
   - **Summary** (≤2 short paragraphs)
   - **Diffs** (unified patches or file list)
   - **Commands + exit codes** actually run
   - **Logs**: tail (~200 lines) from `pytest`, `ruff`, build — **real logs only**
   - **Artifacts**: paths to generated CSV/HTML/JSON
   - **Risks/Follow-ups** (threshold notes, TODOs, migration hints)

If anything fails, **fix it before reporting**.

---

## Prompt style you should expect
I will provide **high-level directives**, not step-by-step tasks. Example:

> “Add jitter/shimmer/HNR/CPPS via Parselmouth, update the outputs schema, write basic tests for silence/tone/speech, keep it CPU-only.”

Your job: **plan → implement → verify → report** in one cycle.

---

## Hard constraints
- No GPU, no system package installs, no secrets.
- Don’t change CLI behavior or output schema without updating docs, readers, and tests.
- Don’t rename/remove existing functions; **add or extend** only.
- Keep ASR default `compute_type=int8`; any change requires measurable gains with real logs.
- If adding dependencies, pin responsibly and justify (size, CPU cost, licensing).

---

## Research & dependency rules (internet ON)
- Prefer **official documentation** and **primary sources**.
- Summarize external findings in your rationale; avoid long quotations.
- Do not auto-download large model files without honoring caches.
- If an external fact materially changes behavior, include a **“Source of truth”** note in your report.

---

## Reporting checklist (include every time)
- ✅ Only factual changes; **no fabricated/simulated logs**
- ✅ Ruff + tests pass (show summaries)
- ✅ **All pipeline functions preserved** (preprocess, SED, diarization, ASR, affect, paralinguistics, outputs)
- ✅ CSV schema/docs updated if impacted
- ✅ SED label collapse intact
- ✅ Assumptions and risks clearly stated
- ✅ No secrets or private data in output
EOF

# ---------- maint-codex.sh ----------
log "Writing maint-codex.sh"
cat > "$REPO_ROOT/maint-codex.sh" <<'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail
log() { printf '\n==> %s\n' "$*"; }
: "${PYTHON:=python}"

log "Format & Lint (ruff)"
if command -v ruff >/dev/null 2>&1; then
  ruff format .
  ruff check --fix .
else
  echo "::notice:: ruff not installed; skipping"
fi

log "Type check (mypy)"
if command -v mypy >/dev/null 2>&1; then
  mypy src || true
else
  echo "::notice:: mypy not installed; skipping"
fi

log "Tests (pytest)"
if command -v pytest >/dev/null 2>&1; then
  pytest -q || true
else
  echo "::notice:: pytest not installed; skipping"
fi

log "Build (PEP 517)"
$PYTHON -m build || true

# Optional CSV header check (if any CSV exists)
REQ_COLS="vq_jitter_pct,vq_shimmer_db,vq_hnr_db,vq_cpps_db,voice_quality_hint"
CSV_CANDIDATE="$(ls -1 *.csv 2>/dev/null | head -n1 || true)"
if [ -n "${CSV_CANDIDATE}" ]; then
  log "CSV header check on ${CSV_CANDIDATE}"
  IFS=, read -r -a HEAD <<< "$(head -n1 "${CSV_CANDIDATE}")"
  for col in ${REQ_COLS//,/ }; do
    if ! printf '%s\0' "${HEAD[@]}" | grep -Fzxq "$col"; then
      echo "::warning:: Missing CSV column: $col in ${CSV_CANDIDATE}"
    fi
  done
else
  echo "::notice:: No CSV present yet; header check skipped"
fi
EOF
chmod +x "$REPO_ROOT/maint-codex.sh"

# ---------- Install with pins & constraints ----------
log "Installing pinned requirements (chunked)"
# heavy numeric/runtime
"$PYTHON" -m pip install --root-user-action=ignore -c constraints.txt \
  numpy==2.3.3 scipy==1.16.2 llvmlite==0.45.0 numba==0.62.1

# rest via file to produce clear pip logs and reuse wheels
"$PYTHON" -m pip install --root-user-action=ignore -c constraints.txt -r requirements.txt

# ---------- Editable install & tools ----------
log "Editable install (again to bind to resolved env)"
"$PYTHON" -m pip install --root-user-action=ignore -c constraints.txt -e .
"$PYTHON" -m pip install --root-user-action=ignore -U ruff pytest mypy build || true

# ---------- FFmpeg provisioning ----------
if ! command -v ffmpeg >/dev/null 2>&1; then
  log "ffmpeg not found; provisioning via imageio-ffmpeg"
  "$PYTHON" -m pip install --root-user-action=ignore -U imageio-ffmpeg
  FFMPEG_BIN="$("$PYTHON" - <<'PY'
import imageio_ffmpeg
print(imageio_ffmpeg.get_ffmpeg_exe())
PY
)"
  if [ -x "$FFMPEG_BIN" ]; then
    export PATH="$(dirname "$FFMPEG_BIN"):$PATH"
    export IMAGEIO_FFMPEG_EXE="$FFMPEG_BIN"
    log "Using bundled FFmpeg at: $FFMPEG_BIN"
  else
    printf '::warning:: imageio-ffmpeg did not expose a binary; some decodes may fail\n'
  fi
else
  log "ffmpeg found on PATH"
fi

log "Setup complete"

# ---------- Diagnostics ----------
"$PYTHON" - <<'PY'
import os, shutil, sys, platform
print("== Diagnostics ==")
print("python:", sys.version.split()[0])
print("platform:", platform.platform())
print("ffmpeg_on_path:", bool(shutil.which("ffmpeg")))
print("HF_HOME:", os.environ.get("HF_HOME"))
print("TRANSFORMERS_CACHE:", os.environ.get("TRANSFORMERS_CACHE"))
print("TORCH_HOME:", os.environ.get("TORCH_HOME"))
print("OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"))
PY
