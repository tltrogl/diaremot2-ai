# DiaRemot — CPU-Only Speech Intelligence Pipeline

DiaRemot is a production-oriented **speech intelligence stack that runs fully on CPU**. It ingests long-form recordings and produces:

- **Diarized transcripts** powered by Faster-Whisper with Silero VAD + ECAPA-TDNN diarization.
- **Affect enrichment**: acoustic emotion (SER), text emotion (GoEmotions), voice-quality metrics, and custom intent classification.
- **Sound event detection** using PANNs for background context.
- **Actionable summaries**: HTML/PDF briefings, CSV rollups, JSONL transcripts, QC health diagnostics, and an extensible manifest describing every artefact.

The repository is tuned for deterministic execution inside the OpenAI Codex environment (no GPU, no internet at runtime) and mirrors the local developer workflow.

---

## 1. Requirements at a Glance

| Requirement | Notes |
| --- | --- |
| Python | 3.9–3.11 supported (3.11 recommended). |
| CPU | x86_64 with AVX2 for Torch/CT2 wheels from the PyTorch CPU index. |
| System tools | `ffmpeg` for decoding compressed audio and chunk extraction. |
| Models | Pre-packaged ONNX/CT2 assets staged under `$DIAREMOT_MODEL_DIR` (defaults to `/opt/models`). See [Model Layout](#3-model-layout).
| Disk | ~5 GB for models + temporary cache (`.cache/`). |

Use `setup.sh` to bootstrap a Codex-compatible environment end-to-end (venv, requirements, model download, import checks). Run `maintenance.sh` in warm containers to re-validate model presence and imports.

---

## 2. Quickstart

```bash
# Python environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .

# Model staging (defaults to /opt/models)
export DIAREMOT_MODEL_DIR=/opt/models  # or ./models if you prefer a local path
# Populate the directory using setup.sh, a release models.zip, or manual sync.

# Run a job (ASR + diarization + affect)
diaremot asr run --input data/sample.wav --outdir outputs/sample_run
# Equivalent module form if console scripts are unavailable:
python -m diaremot.cli asr run --input data/sample.wav --outdir outputs/sample_run

# Resume a partially completed run
diaremot asr resume --input data/sample.wav --outdir outputs/sample_run

# Diagnostics and dependency health (includes ffmpeg + model checks)
diaremot system diagnostics --strict

# Regenerate summaries without re-running inference
diaremot report gen --manifest outputs/sample_run/manifest.json --format pdf --format html
```

Codex agents should call `python -m diaremot.cli asr run ...` after executing `setup.sh`. Warm-container maintenance uses `./maintenance.sh`.

---

## 3. Model Layout

The pipeline expects the following structure under `$DIAREMOT_MODEL_DIR` (default `/opt/models`):

```
silero_vad.onnx
ecapa_onnx/ecapa_tdnn.onnx
panns/model.onnx
panns/class_labels_indices.csv
goemotions-onnx/model.onnx
ser8-onnx/model.int8.onnx  # FP32 fallback: ser8-onnx/model.onnx
faster-whisper-tiny.en/model.bin
bart/model_uint8.onnx      # FP32 fallback: bart/model.onnx
bart/tokenizer.json        # or merges.txt + vocab.json
bart/tokenizer_config.json
bart/special_tokens_map.json
bart/config.json
```

Ship models via one of:

1. **Bundled assets** – commit `models/` or `models.zip` to the repo.
2. **Release download** – host `models.zip` and configure `setup.sh` to fetch + checksum verify.
3. **Custom staging** – populate the directory prior to running the CLI.

`maintenance.sh` validates that either the INT8 or FP32 SER/BART weights are present alongside tokenizer assets.

---

## 4. CLI Surface

The Typer-based CLI exposes domain-specific groups:

### `diaremot asr`
- `run` – complete ASR + diarization + affect pipeline. Supports profiles (`--profile fast|accurate|offline`), Faster-Whisper backend selection, Silero VAD tuning, energy-VAD fallback control, automatic chunking for long files, noise reduction, and affect backend overrides.
- `resume` – pick up from checkpoints in `--outdir` (retains cached diarization/transcription artefacts).

Key options map directly onto the validated [`PipelineConfig`](src/diaremot/pipeline/config.py): chunk sizing, CPU threading, cache roots, ASR timeouts, SED enablement, and affect model directories.

### `diaremot vad`
- `debug` – lightweight Silero VAD inspection with JSON output for troubleshooting thresholds.

### `diaremot report`
- `gen` – rebuild HTML/PDF summaries from a manifest, optionally synthesizing speaker summaries if CSV rollups are absent.

### `diaremot system`
- `diagnostics` – dependency/model health checks (`--strict` enforces version minimums) and ffmpeg availability reporting.

The root command preserves `diaremot run ...` for backwards compatibility, delegating to `asr run`.

---

## 5. Outputs

Each run writes a manifest (`manifest.json`) with the canonical list of artefacts:

- `diarized_transcript_with_emotion.csv` and `segments.jsonl` for per-segment metadata (speaker, timestamps, text, SER/GoEmotions scores, voice-quality metrics, intent tags, VAD confidence, etc.).
- `timeline.csv` for diarization-only consumption.
- `summary.html` / `summary.pdf` built from the HTML/PDF generators.
- `speakers_summary.csv` aggregating per-speaker statistics when available.
- `qc_report.json` containing stage timings, dependency health, audio quality metrics, and summary voice-quality statistics.
- `speaker_registry.json` (path surfaced in the manifest) for persistent speaker embeddings.

The pipeline also persists intermediate checkpoints under `checkpoints/` and logs in `logs/` to support resume and audit flows.

---

## 6. Configuration & Environment

Key environment variables:

- `DIAREMOT_MODEL_DIR` – override model root (default `/opt/models`).
- `OMP_NUM_THREADS=1` – recommended to prevent CPU oversubscription.
- `TOKENIZERS_PARALLELISM=false` – silences Hugging Face tokenizer warnings.
- `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, `TRANSFORMERS_CACHE`, `TORCH_HOME`, `XDG_CACHE_HOME` – automatically pointed at `.cache/` for offline determinism.

`PipelineConfig` supports additional tuning such as `auto_chunk_enabled`, `chunk_threshold_minutes`, `cache_roots`, `noise_reduction`, and `cpu_diarizer`. See [`src/diaremot/pipeline/config.py`](src/diaremot/pipeline/config.py) for exhaustive fields and validation rules.

---

## 7. Development Workflow

- Run unit tests with `pytest -q`.
- Use `python -m diaremot.cli system diagnostics --strict` to confirm dependency health before shipping new builds.
- The codebase lives under `src/diaremot/` (Typer CLI, pipeline orchestrator, affect modules, summaries). Tests reside in `tests/` and stub third-party components (e.g., ReportLab) for offline CI.

When contributing, update this README, `requirements.txt`, and `pyproject.toml` alongside functional changes so Codex instructions stay authoritative.

---

## 8. Support & Licensing

The package is published as `diaremot` with version `2.1.0` (see `pyproject.toml`). Licensing is proprietary; ensure distribution complies with your organisation's policies.
