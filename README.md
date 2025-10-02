# DiaRemot — CPU‑Only Speech Intelligence Pipeline

Process long, real‑world audio **on CPU** and produce a diarized transcript with per‑segment:
- **Tone (Valence/Arousal/Dominance)**
- **Speech emotion (8‑class)**
- **Text emotions (GoEmotions, 28)**
- **Intent** (zero‑shot over fixed labels)
- **Sound‑event context (SED: music, keyboard, door, TV, etc.)**
- **Paralinguistics (REQUIRED)**: speech rate (WPM), pauses, and voice‑quality via **Praat‑Parselmouth**: **jitter**, **shimmer**, **HNR**, **CPPS**
- **Persistent speaker names across files**

Outputs:
- `diarized_transcript_with_emotion.csv` — primary, scrub‑friendly
- `segments.jsonl` — per‑segment payload (audio + text + SED overlaps)
- `speakers_summary.csv` — per‑speaker rollups (V/A/D, emotion mix, intents, WPM, SNR, voice‑quality)
- `summary.html` — Quick Take, Speaker Snapshots, Moments to Check (SED), Action Items
- `speaker_registry.json` — persistent names via centroids
- `events_timeline.csv` + `events.jsonl` — SED events
- `timeline.csv`, `qc_report.json` — fast scrubbing + health checks

## Model set (CPU‑friendly)

- **Diarization**: Diart (Silero VAD + ECAPA‑TDNN embeddings + AHC). Prefers ONNX, Torch fallback for Silero VAD.
- **ASR**: Faster‑Whisper `tiny‑en` via CTranslate2 (`compute_type=int8`).
- **Tone (V/A/D)**: `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`.
- **Speech emotion (8‑class)**: `Dpngtm/wav2vec2-emotion-recognition`.
- **Text emotions (28)**: `SamLowe/roberta-base-go_emotions` (full distribution; keep top‑5).
- **Intent**: Prefers local ONNX exports (e.g., `model_uint8.onnx` under `affect_intent_model_dir` such as `D:\\diaremot\\diaremot2-1\\models\\bart\\`) and falls back to the `facebook/bart-large-mnli` Hugging Face pipeline when no ONNX asset is available.
- **SED**: PANNs CNN14 (ONNX) on onnxruntime; 1.0s frames, 0.5s hop; median filter 3–5; hysteresis 0.50/0.35; `min_dur=0.30s`; `merge_gap≤0.20s`; collapse AudioSet→~20 labels.
- **Paralinguistics (REQUIRED)**: **Praat‑Parselmouth** for jitter/shimmer/HNR/CPPS + prosody (WPM/pauses).

## Install (local; Windows PowerShell shown)

1) **Python 3.11**; **FFmpeg on PATH** (`ffmpeg -version`).
2) Create venv and install:
```powershell
py -3.11 -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install -U pip wheel setuptools
python -m pip install -r requirements.txt
python -m pip install -e .
```

Tkinter bindings are required for the desktop GUI. Install the platform-specific
system package before launching the app:

```bash
# Debian/Ubuntu: sudo apt-get update && sudo apt-get install -y python3-tk
# macOS (Homebrew): brew install python-tk@3.11  # adjust to your Python version
# Windows: ensure the "tcl/tk" Optional Feature is selected in the Python installer
```

### 3) Stage the pretrained models

Set `DIAREMOT_MODEL_DIR` to the folder that contains the ONNX/CT2 assets listed
below. The CLI and GUI both look in this path when loading models.

```bash
export DIAREMOT_MODEL_DIR=/opt/models         # Linux/macOS
# Windows PowerShell
#   $env:DIAREMOT_MODEL_DIR = "D:\models"
```

### 4) Run the pipeline

```bash
# via CLI (preferred)
python -m diaremot.cli run --input data/sample.wav --outdir outputs/run1

# If console scripts are installed:
# diaremot run --input data/sample.wav --outdir outputs/run1

# Desktop GUI launcher
# python -m diaremot.gui.app
# or, if installed as a script:
# diaremot-gui
```

For a Start Menu–friendly executable on Windows, see the packaging workflow below.

---

## Build a Windows desktop executable

Use PyInstaller to bundle the Tkinter GUI into a single-folder distribution.
Run from an activated virtual environment on Windows so collected DLLs match
the target platform.

```powershell
python -m pip install --upgrade pip
python -m pip install .[desktop]
pyinstaller packaging/diaremot_gui.spec --noconfirm
```

Artifacts are written to `dist/DiaRemotDesktop/`. Ship that folder (or a zip)
alongside your models directory and a shortcut that runs `DiaRemotDesktop.exe`.
Before first launch, ensure `DIAREMOT_MODEL_DIR` points to the directory that
contains the ONNX and CT2 assets.

Entrypoints are provided by `src/diaremot/cli.py` and `[project.scripts]` in `pyproject.toml`.
`run_pipeline` and `resume` are also proxied through the CLI.

---

## Models (single source of truth)

All model files live under **`$DIAREMOT_MODEL_DIR`** (default `/opt/models`). Layout must be:

```
{MODEL_DIR}/silero_vad.onnx
{MODEL_DIR}/ecapa_onnx/ecapa_tdnn.onnx
{MODEL_DIR}/panns/model.onnx
{MODEL_DIR}/panns/class_labels_indices.csv
{MODEL_DIR}/goemotions-onnx/model.onnx
{MODEL_DIR}/ser8-onnx/model.onnx        # or model.int8.onnx
{MODEL_DIR}/faster-whisper-tiny.en/model.bin
{MODEL_DIR}/bart/model_uint8.onnx       # or model.onnx
# BART tokenizer assets (required offline)
{MODEL_DIR}/bart/tokenizer.json         # or merges.txt + vocab.json
{MODEL_DIR}/bart/tokenizer_config.json
{MODEL_DIR}/bart/special_tokens_map.json
{MODEL_DIR}/bart/config.json
```

Ship models via one of:
1. **Repo**: include `models/` or `models.zip` in repo root.
2. **Release**: upload `models.zip` as a GitHub Release asset; `setup.sh` downloads it.
3. **Custom host**: download during `setup.sh` with checksum verification.

**Codex note:** models persist in the warm container cache (~12h).

---

## Environment variables

- `DIAREMOT_MODEL_DIR` (default `/opt/models`) — base directory above
- `OMP_NUM_THREADS=1` — avoid CPU oversubscription
- `TOKENIZERS_PARALLELISM=false` — silence HF tokenizer parallelism
- `HF_HOME=/opt/cache/hf` — Hugging Face cache path (optional)

Optional (for release download in setup):
- `MODEL_RELEASE_URL` — direct URL to `models.zip`
- `MODEL_RELEASE_SHA256` — SHA256 for integrity check

---

## Codex Cloud settings

- **Base image**: `universal` (Python 3.11)
- **Setup script**: `./setup.sh` — installs deps, stages `models/` into `/opt/models`
- **Maintenance**: `./maint-codex.sh` — quick health/import check
- **Internet**: Agent **OFF** (deterministic). `setup.sh` performs any required fetches.
- **AGENTS.md** contains concrete run/diagnostic commands.
