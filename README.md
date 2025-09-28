# DiaRemot — CPU‑Only Speech + Affect Pipeline (Codex‑Ready)

This project runs **on CPU** and produces a diarized transcript with per‑segment affect:
- **ASR**: Faster‑Whisper tiny.en (CT2)
- **Diarization**: Silero VAD (ONNX) + ECAPA‑TDNN embeddings (ONNX)
- **SED**: PANNs (ONNX) + labels CSV
- **SER**: 8‑class speech emotion (INT8/FP32 ONNX)
- **Text emotion**: GoEmotions (ONNX)
- **Intent/zero‑shot**: BART (ONNX classifier)

Everything is wired for **OpenAI Codex Cloud**: reproducible container, cached models, and no Windows‑only paths.

---

## Installation & Quickstart

### 1. Create a Python environment

```bash
# Python 3.11 recommended (3.9–3.11 supported by your repo pins)
python -V

# venv
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\\Scripts\\activate
```

### 2. Install DiaRemot and dependencies

```bash
# install (uses your repo's requirements.txt with CPU torch index)
pip install -U pip wheel setuptools
pip install -r requirements.txt
pip install -e .
```

Tkinter bindings are required for the desktop GUI. Install the platform-specific
system package before launching the app:

```bash
# Debian/Ubuntu: sudo apt-get update && sudo apt-get install -y python3-tk
# macOS (Homebrew): brew install python-tk@3.11  # adjust version to your Python
# Windows: ensure "tcl/tk" feature is selected in the Python installer
```

### 3. Stage the pretrained models

Set `DIAREMOT_MODEL_DIR` to the folder that contains the ONNX/CT2 assets listed
below. The CLI and GUI both look in this path when loading models.

```bash
export DIAREMOT_MODEL_DIR=/opt/models   # or ./models locally
```

### 4. Run the pipeline

```bash
# run via CLI (preferred)
python -m diaremot.cli run --input samples/ --out outputs/run1
# or, if scripts are installed:
# diaremot run --input samples/ --out outputs/run1

# desktop GUI launcher
python -m diaremot.gui.app
# or, with scripts installed:
# diaremot-gui

# diagnostics
python -m diaremot.cli diagnostics
# or: diaremot-diagnostics
```

### 5. Launch the desktop GUI

```bash
python -m diaremot.gui.app
# or, with scripts installed:
# diaremot-gui
```

For a Start Menu friendly executable on Windows, see the packaging workflow
below.

---

## Build a Windows desktop executable

Use PyInstaller to bundle the Tkinter GUI into a single-folder distribution.
The process should be run from an activated virtual environment on Windows so
the collected DLLs match the target platform.

```powershell
python -m pip install --upgrade pip
python -m pip install .[desktop]
pyinstaller packaging/diaremot_gui.spec --noconfirm
```

Artifacts are written to `dist/DiaRemotDesktop/`. Ship that folder (or create a
zip) alongside your models directory and a shortcut that runs `DiaRemotDesktop.exe`.
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
# BART tokenizer assets (required for offline)
{MODEL_DIR}/bart/tokenizer.json         # or merges.txt + vocab.json
{MODEL_DIR}/bart/tokenizer_config.json
{MODEL_DIR}/bart/special_tokens_map.json
{MODEL_DIR}/bart/config.json
```

Ship models via one of:
1. **Repo**: include `models/` or `models.zip` in repo root.
2. **Release**: upload `models.zip` as a GitHub Release asset; `setup.sh` can download it.
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
- **Maintenance**: `./maintenance.sh` — quick health/import check
- **Internet**: Agent **OFF** (deterministic). Setup can fetch if needed.
- **AGENTS.md** shows concrete run commands via the CLI.
