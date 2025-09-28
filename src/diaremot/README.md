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

## Quickstart (local or Codex shell)

```bash
# Python 3.11 recommended (3.9–3.11 supported by your repo pins)
python -V

# venv
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\Scripts\activate

# install (uses your repo's requirements.txt with CPU torch index)
pip install -U pip wheel setuptools
pip install -r requirements.txt
pip install -e .

# models base dir
export DIAREMOT_MODEL_DIR=/opt/models   # or ./models locally

# run via CLI (preferred)
python -m diaremot.cli asr run --input samples/ --out outputs/run1
# or, if scripts are installed:
# diaremot asr run --input samples/ --out outputs/run1
# Backward-compatible aliases for ``diaremot run``/``resume`` remain available.

# diagnostics
python -m diaremot.cli system diagnostics
# or: diaremot system diagnostics / diaremot-diagnostics
```

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
