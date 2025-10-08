#!/usr/bin/env bash
# Codex bootstrap for DiaRemot (Linux, CPU-only)
# - sets up a virtualenv using requirements.txt
# - pins Hugging Face/Torch caches inside the repo
# - optionally generates a 10s sine-wave sample under data/
# - runs pipeline dependency verification to ensure torch._C loads cleanly

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
echo "==> Repo root: $REPO_ROOT"

REQUIRED_FILES=(
  "pyproject.toml"
  "requirements.txt"
  "src/diaremot/__init__.py"
  "src/diaremot/pipeline/audio_pipeline_core.py"
  "src/diaremot/pipeline/audio_preprocessing.py"
  "src/diaremot/pipeline/speaker_diarization.py"
  "src/diaremot/pipeline/transcription_module.py"
  "src/diaremot/affect/emotion_analyzer.py"
  "src/diaremot/affect/paralinguistics.py"
  "src/diaremot/summaries/html_summary_generator.py"
  "src/diaremot/summaries/pdf_summary_generator.py"
  "src/diaremot/summaries/speakers_summary_builder.py"
  "src/diaremot/io/onnx_utils.py"
)
missing=()
for f in "${REQUIRED_FILES[@]}"; do
  [[ -f "$f" ]] || missing+=("$f")
done
if ((${#missing[@]})); then
  echo "ERROR: required project files missing:" >&2
  printf '  - %s\n' "${missing[@]}" >&2
  exit 1
fi

eq_exit() {
  echo "ERROR: $1" >&2
  exit 1
}

command -v python3 >/dev/null 2>&1 || eq_exit "python3 not found in PATH"
echo "==> Python: $(python3 -V)"

echo "==> Creating virtual environment (.venv)"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Upgrading bootstrap tooling"
python -m pip install --upgrade pip setuptools wheel

echo "==> Installing requirements.txt"
python -m pip install -r requirements.txt

echo "==> Preparing local caches"
CACHE_ROOT="$REPO_ROOT/.cache"
mkdir -p "$CACHE_ROOT"

prefer_dir() {
  local primary="$1" fallback="$2" label="$3"
  if mkdir -p "$primary" 2>/dev/null; then
    printf '%s\n' "$primary"
    return 0
  fi
  echo "WARN: unable to initialise $label at $primary; using $fallback" >&2
  mkdir -p "$fallback"
  printf '%s\n' "$fallback"
}

HF_HOME="${HF_HOME:-$(prefer_dir /srv/.cache/hf "$CACHE_ROOT/hf" 'HF cache')}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$(prefer_dir /srv/.cache/transformers "$CACHE_ROOT/transformers" 'Transformers cache')}"
TORCH_HOME="${TORCH_HOME:-$(prefer_dir /srv/.cache/torch "$CACHE_ROOT/torch" 'Torch cache')}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT}"
HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
DIAREMOT_MODEL_DIR="${DIAREMOT_MODEL_DIR:-$(prefer_dir /srv/models "$REPO_ROOT/models" 'model root')}"

mkdir -p "$DIAREMOT_MODEL_DIR" \
         "$DIAREMOT_MODEL_DIR/asr_ct2" \
         "$DIAREMOT_MODEL_DIR/diarization" \
         "$DIAREMOT_MODEL_DIR/sed_panns" \
         "$DIAREMOT_MODEL_DIR/affect/ser8" \
         "$DIAREMOT_MODEL_DIR/affect/vad_dim" \
         "$DIAREMOT_MODEL_DIR/intent" \
         "$DIAREMOT_MODEL_DIR/text_emotions"

export HF_HOME HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE TORCH_HOME XDG_CACHE_HOME
export HF_HUB_ENABLE_HF_TRANSFER TOKENIZERS_PARALLELISM DIAREMOT_MODEL_DIR
export CUDA_VISIBLE_DEVICES=""
export TORCH_DEVICE="cpu"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

if command -v ffmpeg >/dev/null 2>&1; then
  if [[ ! -f data/sample.wav ]]; then
    echo "==> Generating 10s 440Hz sine sample (data/sample.wav)"
    mkdir -p data
    ffmpeg -hide_banner -loglevel error \
      -f lavfi -i "sine=frequency=440:duration=10" \
      -ar 16000 -ac 1 data/sample.wav -y || echo "WARN: ffmpeg failed to create sample audio"
  fi
else
  echo "WARN: ffmpeg not found; skipping sample audio generation"
fi

echo "==> Verifying core imports"
python - <<'PY'
import importlib
modules = [
    "diaremot.pipeline.audio_pipeline_core",
    "diaremot.pipeline.audio_preprocessing",
    "diaremot.pipeline.speaker_diarization",
    "diaremot.pipeline.transcription_module",
    "diaremot.affect.emotion_analyzer",
    "diaremot.affect.paralinguistics",
    "diaremot.summaries.html_summary_generator",
    "diaremot.summaries.pdf_summary_generator",
    "diaremot.summaries.speakers_summary_builder",
    "diaremot.io.onnx_utils",
]
failed = []
for name in modules:
    try:
        importlib.import_module(name)
        print("OK  import", name)
    except Exception as exc:
        failed.append((name, exc))
if failed:
    print("\nFAILED imports:")
    for name, exc in failed:
        print(f" - {name}: {exc}")
    raise SystemExit(2)
PY

echo "==> Running pipeline dependency check"
python -m diaremot.pipeline.audio_pipeline_core --verify_deps

cat <<'MSG'
==> Setup complete.
Activate the venv with "source .venv/bin/activate" (prompt should show (.venv)).
To smoke-test the pipeline: python -m diaremot.cli run --input data/sample.wav --outdir outputs/run_$RANDOM
MSG