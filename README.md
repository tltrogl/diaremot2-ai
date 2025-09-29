# DiaRemot — CPU-Only Speech Intelligence Pipeline

DiaRemot is a production-grade **speech intelligence stack that runs entirely on CPUs**. It ingests long-form recordings and produces diarised transcripts enriched with acoustic emotion, text intent, and background sound event context. The orchestration is delivered through a Typer-based CLI so the same workflow runs identically in local shells, CI, and containerised agent environments.

---

## 1. Capabilities at a glance

| Capability | Implementation highlights |
| --- | --- |
| Transcription & diarisation | Faster-Whisper CT2 models with Silero VAD, ECAPA-TDNN embeddings, and optional CPU diariser fallbacks. |
| Affect enrichment | Acoustic SER (INT8 + FP32 fallbacks), text emotion via GoEmotions, optional intent classification, and per-segment voice-quality metrics. |
| Sound event tagging | Optional PANNs ONNX tagging for ambient context with automatic cache hygiene. |
| Summaries & reports | HTML/PDF briefings, diarisation CSV/JSONL exports, QC diagnostics, and speaker registry updates. |
| Resume & caching | Checkpoint-aware pipeline with resumable stages, deterministic cache layout, and manifest-driven regeneration. |

The runtime is deterministic, GPU-free, and expects models to be pre-staged. Internet access is available for documentation lookups or retrieving pinned artefacts when required.

---

## 2. System requirements

| Requirement | Notes |
| --- | --- |
| Python | CPython 3.10–3.11 (pins validated on 3.11). |
| CPU | x86_64 with AVX2 to satisfy the PyTorch CPU wheels sourced from the PyTorch CPU index. |
| System tools | `ffmpeg` for decoding/compression handling. `curl` and `unzip` (or Python’s `zipfile`) are used by `setup.sh`. |
| Models | Stage weights under `$DIAREMOT_MODEL_DIR` (defaults to `/opt/models`). See [§4 Model assets](#4-model-assets). |
| Disk | ~5 GB for models plus `.cache/` for HF/Torch artefacts. |

---

## 3. Environment preparation

`setup.sh` mirrors the manual steps below. From PowerShell sessions, invoke it with `bash ./setup.sh`.

### PowerShell

```powershell
# Interpreter (3.11 recommended)
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip setuptools wheel
# requirements.txt carries the PyTorch CPU index pins
python -m pip install -r requirements.txt
python -m pip install -e .

# Model staging (defaults to /opt/models)
$env:DIAREMOT_MODEL_DIR = "C:/diaremot/models"  # choose a writable path
bash ./setup.sh  # optional: downloads models.zip + runs diagnostics

# Run a pipeline job
python -m diaremot.cli asr run --input data/sample.wav --outdir outputs/sample_run
# Console script after editable install
diaremot asr run --input data/sample.wav --outdir outputs/sample_run
```

### POSIX shell

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .

export DIAREMOT_MODEL_DIR=${DIAREMOT_MODEL_DIR:-/opt/models}
bash ./setup.sh  # optional end-to-end bootstrap

python -m diaremot.cli asr run --input data/sample.wav --outdir outputs/sample_run
```

`setup.sh` is idempotent: it validates required files, installs dependencies, stages models, exports cache variables, and finishes with import/diagnostics checks.

---

## 4. Model assets

The pipeline expects the following structure beneath `$DIAREMOT_MODEL_DIR` (default `/opt/models`). Either unpack `models.zip` from the release artefact or stage weights manually.

```
silero_vad.onnx
ecapa_onnx/ecapa_tdnn.onnx
panns/model.onnx
panns/class_labels_indices.csv
goemotions-onnx/model.onnx
ser8-onnx/model.int8.onnx    # FP32 fallback: ser8-onnx/model.onnx
faster-whisper-tiny.en/model.bin
bart/model_uint8.onnx        # FP32 fallback: bart/model.onnx
bart/tokenizer.json          # or merges.txt + vocab.json
bart/tokenizer_config.json
bart/config.json
```

`maintenance.sh`/`maint-codex.sh` confirm that either the INT8 or FP32 SER/BART weights are present along with tokenizer assets. Adjust the scripts and README together if the layout changes.

---

## 5. CLI quick reference

The Typer CLI groups commands by domain and exposes the same operations through module execution or the installed console script.

```bash
# End-to-end pipeline
python -m diaremot.cli asr run --input input.wav --outdir outputs/run1
# Resume from checkpoints
python -m diaremot.cli asr resume --input input.wav --outdir outputs/run1
# Regenerate summaries without inference
python -m diaremot.cli report gen --manifest outputs/run1/manifest.json --format pdf --format html
# Diagnostics
python -m diaremot.cli system diagnostics --strict
```

Useful flags:
- `--profile fast|accurate|offline` toggles built-in overrides (beam size, affect backend, CT2 compute types).
- `--chunk-enabled/--chunk-threshold-minutes/--chunk-size-minutes` control long-form chunking.
- `--affect-backend onnx|torch|auto` switches SER implementation.
- `--enable-sed/--disable-affect/--noise-reduction` toggle optional subsystems.

The legacy `diaremot run ...` alias remains available for backwards compatibility and forwards to `asr run`.

---

## 6. Outputs

Each successful run writes a manifest (`manifest.json`) describing artefacts and metadata:

- `segments.jsonl` and `diarized_transcript_with_emotion.csv` – diarised transcript with affect & intent scores.
- `timeline.csv` – diarisation timeline for analytics tooling.
- `summary.html` / `summary.pdf` – HTML/PDF executive reports built by the summary generators.
- `speakers_summary.csv` – aggregated per-speaker statistics (auto-built during reporting if absent).
- `qc_report.json` – dependency checks, audio quality metrics, and stage timings.
- `speaker_registry.json` – persistent ECAPA embeddings for future runs.

Intermediate checkpoints live under `checkpoints/` and logs in `logs/` to support resumable workflows.

---

## 7. Diagnostics & maintenance

- `bash ./setup.sh` – full bootstrap: installs dependencies, stages models, and runs smoke checks.
- `bash ./maintenance.sh` – warm-container validation (models + imports) using the default model root.
- `bash ./maint-codex.sh` – Codex/agent-friendly variant that honours `$DIAREMOT_MODEL_DIR` and surfaces actionable errors.
- `python -m diaremot.cli system diagnostics --strict` – JSON diagnostics covering dependency versions, ffmpeg availability, and model presence.

For stricter validation you can also invoke `python -m diaremot.pipeline.validate_system_complete`, which exercises the same dependency verifiers used by the orchestrator.

---

## 8. Development workflow

1. Keep `requirements.txt` and `pyproject.toml` in sync whenever dependencies change.
2. Run `pytest -q` locally once dependencies are installed.
3. Execute `python -m diaremot.cli system diagnostics --strict` before shipping updates.
4. Update documentation (`README.md`, AGENTS guidance, package docs) alongside behavioural changes so automation remains trustworthy.

---

## 9. Support & licensing

The package is published as `diaremot` (version `2.1.0`; see `pyproject.toml`). Licensing is proprietary—coordinate distribution with your organisation’s policies.
