#!/usr/bin/env bash
# DiaRemot maintenance utilities (Linux, CPU-only)

set -Eeuo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

VENV="$REPO_ROOT/.venv"
if [[ -f "$VENV/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
fi

CACHE_ROOT="$REPO_ROOT/.cache"
mkdir -p "$CACHE_ROOT/hf" "$CACHE_ROOT/torch" "$CACHE_ROOT/transformers"
export HF_HOME="$CACHE_ROOT/hf"
export HUGGINGFACE_HUB_CACHE="$CACHE_ROOT/hf"
export TRANSFORMERS_CACHE="$CACHE_ROOT/transformers"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT"
export CUDA_VISIBLE_DEVICES=""
export TORCH_DEVICE="cpu"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

REQUIRED_FILES=(
  "pyproject.toml"
  "src/diaremot/pipeline/audio_pipeline_core.py"
  "src/diaremot/pipeline/speaker_diarization.py"
  "src/diaremot/pipeline/transcription_module.py"
  "src/diaremot/affect/emotion_analyzer.py"
)
need_files() {
  local missing=()
  for f in "${REQUIRED_FILES[@]}"; do [[ -f "$f" ]] || missing+=("$f"); done
  if ((${#missing[@]})); then
    echo "Missing files:"; printf '  - %s\n' "${missing[@]}"; return 2
  fi
}

cmd="${1:-help}"; arg="${2:-}"

case "$cmd" in
check)
  need_files || exit $?
  echo "Python: $(python -V)"
  echo "Pip:    $(python -m pip -V)"
  python - <<'PY'
import importlib
mods = [
    "diaremot.pipeline.audio_pipeline_core",
    "diaremot.pipeline.audio_preprocessing",
    "diaremot.pipeline.speaker_diarization",
    "diaremot.pipeline.transcription_module",
    "diaremot.affect.emotion_analyzer",
    "diaremot.affect.paralinguistics",
    "diaremot.summaries.html_summary_generator",
    "diaremot.summaries.pdf_summary_generator",
    "diaremot.summaries.speakers_summary_builder",
]
for name in mods:
    try:
        importlib.import_module(name)
        print("OK  import", name)
    except Exception as exc:
        print("WARN import", name, "->", exc)
PY
  python -m diaremot.pipeline.audio_pipeline_core --verify_deps
  ;;
prewarm)
  mode="${arg:-light}"; mode=${mode,,}
  python - <<PY
import os
mode = "${mode}"
print(f"Prewarm mode: {mode}")

def log(status, name):
    print(f"{status:<4} {name}", flush=True)

def trycall(fn, name):
    try:
        fn(); log("OK", name)
    except Exception as exc:
        log("WARN", f"{name}: {exc}")

def warm_fw():
    from faster_whisper import WhisperModel
    WhisperModel(os.getenv("DIAREMOT_WHISPER_MODEL","tiny.en"), device="cpu", compute_type=os.getenv("DIAREMOT_COMPUTE_TYPE","int8"))

def warm_ctranslate():
    import ctranslate2; ctranslate2.contains_model

def warm_ecapa():
    from diaremot.pipeline.speaker_diarization import _ECAPAWrapper
    _ECAPAWrapper()

def warm_silero():
    import torch
    torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True)

for fn,name in [(warm_fw,"faster-whisper"),(warm_ctranslate,"ctranslate2"),(warm_ecapa,"ecapa_tdnn"),(warm_silero,"silero_vad")]:
    trycall(fn,name)

if mode == "full":
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoFeatureExtractor, AutoModelForAudioClassification
    def preload(model_id):
        AutoTokenizer.from_pretrained(model_id)
        try:
            AutoModelForSequenceClassification.from_pretrained(model_id)
        except Exception:
            AutoFeatureExtractor.from_pretrained(model_id)
            AutoModelForAudioClassification.from_pretrained(model_id)
    for mid in [
        "SamLowe/roberta-base-go_emotions",
        "facebook/bart-large-mnli",
        "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
        "Dpngtm/wav2vec2-emotion-recognition",
    ]:
        trycall(lambda m=mid: preload(m), mid)
PY
  ;;
update)
  need_files || exit $?
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install -r requirements.txt
  ;;
freeze)
  python -m pip freeze --all | sed '/^pkg-resources==/d' > requirements.lock.txt
  echo "Wrote requirements.lock.txt"
  ;;
clean)
  target="${arg:-caches}"
  case "$target" in
    caches) rm -rf "$CACHE_ROOT"; mkdir -p "$CACHE_ROOT" ;;
    venv)   rm -rf "$VENV" ;;
    all)    rm -rf "$CACHE_ROOT" "$VENV"; mkdir -p "$CACHE_ROOT" ;;
    *) echo "Unknown clean target: $target"; exit 2 ;;
  esac
  ;;
doctor)
  "$0" check
  python - <<'PY'
from diaremot.pipeline.audio_pipeline_core import AudioAnalysisPipelineV2
pipe = AudioAnalysisPipelineV2()
print("OK  pipeline instantiated (default config)")
PY
  ;;
smoke)
  need_files || exit $?
  if [[ ! -f data/sample.wav ]]; then
    echo "ERROR: data/sample.wav missing; run setup-codex.sh first" >&2
    exit 2
  fi
  run_dir="outputs/smoke_$(date +%s)"
  python -m diaremot.cli run --input data/sample.wav --outdir "$run_dir"
  echo "Artifacts written to $run_dir"
  ;;
*)
  echo "Usage: $0 {check|prewarm [light|full]|update|freeze|clean [caches|venv|all]|doctor|smoke}" >&2
  exit 2
  ;;
esac

echo "Done."