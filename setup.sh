#!/usr/bin/env bash
# DiaRemot setup (AGENTS.md‑compliant, CPU‑only, ONNX‑preferred)
# - No apt/brew usage; pip only
# - Cross‑platform venv activation (posix/Windows Git Bash)
# - Defines required env vars (with safe defaults)
# - Optional models.zip staging, gated by env flags
# - Optional FFmpeg bootstrap via imageio-ffmpeg into ./.cache/bin (Codex Cloud)

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
echo "==> Repo root: $REPO_ROOT"

# ---- Sanity: required files present ----
REQUIRED_FILES=(
  "pyproject.toml"
  "requirements.txt"
  "src/diaremot/cli.py"
  "src/diaremot/pipeline/stages/__init__.py"
)
missing=(); for f in "${REQUIRED_FILES[@]}"; do [[ -f "$f" ]] || missing+=("$f"); done
if ((${#missing[@]})); then
  echo "ERROR: required files missing:" >&2; printf '  - %s\n' "${missing[@]}" >&2; exit 1
fi

# ---- Python / venv (no system package managers) ----
PYTHON_BIN="${PYTHON_BIN:-python3}"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || PYTHON_BIN=python
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "ERROR: Python not found in PATH" >&2; exit 1; }
echo "==> Using Python: $($PYTHON_BIN -V)"

if [[ ! -d .venv ]]; then
  echo "==> Creating virtual environment (.venv)"
  "$PYTHON_BIN" -m venv .venv
fi

# shellcheck disable=SC1091
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
elif [[ -f .venv/Scripts/activate ]]; then
  source .venv/Scripts/activate
else
  echo "ERROR: venv activation script not found" >&2; exit 1
fi

echo "==> Upgrading pip/setuptools/wheel"
python -m pip install --upgrade pip setuptools wheel

echo "==> Installing requirements.txt"
python -m pip install -r requirements.txt

if [[ -f pyproject.toml ]]; then
  echo "==> Installing diaremot package in editable mode"
  python -m pip install -e .
fi

# ---- Local caches + required env vars (defaults) ----
echo "==> Preparing local caches and env vars"
CACHE_ROOT="${CACHE_ROOT:-$REPO_ROOT/.cache}"
mkdir -p "$CACHE_ROOT" "$CACHE_ROOT/hf" "$CACHE_ROOT/torch" "$CACHE_ROOT/transformers"

export HF_HOME="${HF_HOME:-$CACHE_ROOT/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$CACHE_ROOT/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$CACHE_ROOT/transformers}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_ROOT/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT}"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=""   # CPU‑only
export TORCH_DEVICE="cpu"

# Threads defaults per AGENTS.md (cap to 4)
_cpu_n="$( (command -v nproc >/dev/null && nproc) || (getconf _NPROCESSORS_ONLN 2>/dev/null) || echo 4 )"
_cpu_n=$(( _cpu_n > 4 ? 4 : _cpu_n ))
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${_cpu_n}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${_cpu_n}}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-${_cpu_n}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# ---- Models: stage into $DIAREMOT_MODEL_DIR (no silent downloads) ----
export DIAREMOT_MODEL_DIR="${DIAREMOT_MODEL_DIR:-$REPO_ROOT/models}"
mkdir -p "$DIAREMOT_MODEL_DIR"

need_model_copy=1
for must in \
  panns_cnn14.onnx \
  audioset_labels.csv \
  silero_vad.onnx \
  ecapa_tdnn.onnx \
  ser_8class.onnx \
  vad_model.onnx \
  roberta-base-go_emotions.onnx \
  bart-large-mnli.onnx; do
  if [[ -f "$DIAREMOT_MODEL_DIR/$must" ]]; then continue; else need_model_copy=0; break; fi
done

# Portable sha256 helper
sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}';
  elif command -v shasum   >/dev/null 2>&1; then shasum -a 256 "$1" | awk '{print $1}';
  else python - "$1" <<'PY'
import hashlib, sys
h=hashlib.sha256()
with open(sys.argv[1],'rb') as f:
    for chunk in iter(lambda: f.read(8192), b''):
        h.update(chunk)
print(h.hexdigest())
PY
  fi
}

unzip_portable() {
  local zip="$1"; local dest="$2"
  if command -v unzip >/dev/null 2>&1; then unzip -o "$zip" -d "$dest" >/dev/null
  else python -m zipfile -e "$zip" "$dest"
  fi
}

if (( need_model_copy == 0 )); then
  echo "==> Models not fully present under $DIAREMOT_MODEL_DIR"
  MODELS_ZIP_PATH="${DIAREMOT_MODELS_ZIP:-$REPO_ROOT/models.zip}"
  MODELS_URL="${DIAREMOT_MODELS_URL:-}"
  MODELS_SHA="${DIAREMOT_MODELS_SHA256:-}"
  AUTO_DL="${DIAREMOT_AUTO_DOWNLOAD:-0}"

  if [[ -f "$MODELS_ZIP_PATH" ]]; then
    echo "==> Unpacking models.zip from repo into $DIAREMOT_MODEL_DIR"
    [[ -n "$MODELS_SHA" ]] && {
      got=$(sha256_file "$MODELS_ZIP_PATH"); [[ "$got" == "$MODELS_SHA" ]] || { echo "ERROR: SHA256 mismatch for models.zip" >&2; exit 2; };
    }
    unzip_portable "$MODELS_ZIP_PATH" "$DIAREMOT_MODEL_DIR"
  elif [[ "$AUTO_DL" == "1" && -n "$MODELS_URL" ]]; then
    echo "==> Downloading models.zip (per DIAREMOT_AUTO_DOWNLOAD=1)"
    mkdir -p "$CACHE_ROOT"
    dest="$CACHE_ROOT/models.zip"
    if command -v curl >/dev/null 2>&1; then curl -L --fail --retry 5 -o "$dest" "$MODELS_URL"; 
    elif command -v wget >/dev/null 2>&1; then wget -O "$dest" "$MODELS_URL"; 
    else python - "$MODELS_URL" "$dest" <<'PY'
import sys, urllib.request
urllib.request.urlretrieve(sys.argv[1], sys.argv[2])
PY
    fi
    [[ -n "$MODELS_SHA" ]] && { got=$(sha256_file "$dest"); [[ "$got" == "$MODELS_SHA" ]] || { echo "ERROR: SHA256 mismatch for models.zip" >&2; exit 2; }; }
    unzip_portable "$dest" "$DIAREMOT_MODEL_DIR"
  else
    echo "WARN: Models missing and no models.zip provided."
    echo "      Provide models at $DIAREMOT_MODEL_DIR or set DIAREMOT_MODELS_ZIP=/path/models.zip"
    echo "      (Optional) set DIAREMOT_AUTO_DOWNLOAD=1 and DIAREMOT_MODELS_URL to enable download."
  fi
fi

# Normalize layout if a nested 'models/' directory was in the zip
if [[ -d "$DIAREMOT_MODEL_DIR/models" ]]; then
  echo "==> Normalizing staged models layout"
  cp -R "$DIAREMOT_MODEL_DIR/models/." "$DIAREMOT_MODEL_DIR/" || true
  rm -rf "$DIAREMOT_MODEL_DIR/models"
fi

# ---- Optional: generate tiny sample (if ffmpeg present) ----
# In Codex Cloud (no apt), bootstrap ffmpeg locally if missing and allowed.
if ! command -v ffmpeg >/dev/null 2>&1; then
  if [[ "${DIAREMOT_BOOTSTRAP_FFMPEG:-1}" == "1" ]]; then
    echo "==> Bootstrapping FFmpeg via imageio-ffmpeg into $CACHE_ROOT/bin"
    python - <<'PY'
import os, sys, subprocess
try:
    import imageio_ffmpeg as iif
except Exception:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', 'imageio-ffmpeg==0.4.9'])
    import imageio_ffmpeg as iif
exe = iif.get_ffmpeg_exe()
print(exe)
PY
    FFMPEG_EXE_PATH=$(python - <<'PY'
import imageio_ffmpeg as iif
print(iif.get_ffmpeg_exe())
PY
)
    mkdir -p "$CACHE_ROOT/bin"
    ln -sf "$FFMPEG_EXE_PATH" "$CACHE_ROOT/bin/ffmpeg"
    export PATH="$CACHE_ROOT/bin:${PATH}"
    echo "   -> ffmpeg is now at $CACHE_ROOT/bin/ffmpeg"
  else
    echo "WARN: ffmpeg not found and DIAREMOT_BOOTSTRAP_FFMPEG=0; skipping ffmpeg install"
  fi
fi

if command -v ffmpeg >/dev/null 2>&1; then
  if [[ ! -f data/sample.wav ]]; then
    echo "==> Generating 10s 440Hz sample (data/sample.wav)"
    mkdir -p data
    ffmpeg -hide_banner -loglevel error \
      -f lavfi -i "sine=frequency=440:duration=10" \
      -ar 16000 -ac 1 data/sample.wav -y || echo "WARN: ffmpeg failed to create sample"
  fi
fi

# ---- Import and dependency checks (non‑fatal summary) ----
echo "==> Verifying core imports"
python - <<'PY' || true
import importlib
mods = [
  "diaremot.pipeline.audio_pipeline_core",
  "diaremot.pipeline.audio_preprocessing",
  "diaremot.pipeline.speaker_diarization",
  "diaremot.pipeline.transcription_module",
  "diaremot.affect.emotion_analyzer",
  "diaremot.affect.paralinguistics",
  "diaremot.io.onnx_utils",
]
bad=[]
for m in mods:
  try:
    importlib.import_module(m)
    print("OK  import", m)
  except Exception as e:
    bad.append((m, e))
if bad:
  print("\nFAILED imports:")
  for m,e in bad:
    print(" -", m, ":", e)
PY

echo "==> Checking pinned dependency versions"
python - <<'PY'
from importlib import metadata
from packaging.version import Version
from diaremot.pipeline.config import CORE_DEPENDENCY_REQUIREMENTS

issues: list[str] = []
for name, minimum in CORE_DEPENDENCY_REQUIREMENTS.items():
    try:
        version = metadata.version(name)
    except metadata.PackageNotFoundError:
        issues.append(f"{name} not installed (required >= {minimum})")
        continue
    print(f"   - {name} {version} (required >= {minimum})")
    try:
        if Version(version) < Version(minimum):
            issues.append(f"{name} {version} < required {minimum}")
    except Exception as exc:  # pragma: no cover - defensive guard
        issues.append(f"{name} version comparison failed: {exc}")

if issues:
    print("Dependency version issues detected:")
    for item in issues:
        print(f" * {item}")
    raise SystemExit(2)
PY

echo "==> Pipeline dependency check (--verify_deps)"
python -m diaremot.pipeline.audio_pipeline_core --verify_deps --strict_dependency_versions || true

cat <<'MSG'
==> Setup complete.
Environment variables set (cached under ./.cache). Models staged in $DIAREMOT_MODEL_DIR.
Run a quick smoke test:
  python -m diaremot.cli run --input data/sample.wav --outdir ./outputs --asr-compute-type float32
MSG
