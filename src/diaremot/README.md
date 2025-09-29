# DiaRemot — CPU-Only Speech Intelligence Pipeline

DiaRemot ingests long-form audio on a CPU-only host and produces a full intelligence package:

- **Diarised transcription** using Faster-Whisper (CT2) with Silero VAD and ECAPA-TDNN embeddings.
- **Affect enrichment** combining acoustic SER, text emotion (GoEmotions), intent classification,
  paralinguistic measurements, and voice-quality metrics.
- **Sound event detection** via PANNs to provide background context.
- **Conversation analytics** including interruptions, dominance, turn-taking balance, and energy flow.
- **Rich artefacts**: CSV/JSONL transcripts, speaker rollups, QC diagnostics, and HTML/PDF summaries.

The repository targets reproducible execution in containers or bare-metal Linux hosts without GPUs.
Pinned wheels come from the PyTorch CPU index and all tooling is validated on CPython 3.11 (3.10
remains supported).

---

## 1. Requirements

| Component | Details |
| --- | --- |
| Python | 3.11 preferred (3.10 supported). |
| CPU | x86_64 with AVX2 so the PyTorch CPU wheels and CT2 builds work. |
| System tools | `ffmpeg` (decode/transcode) + `unzip`, `curl`, `rsync`. Optional: PyAV improves metadata probing. |
| Disk | ~5 GB for models plus `.cache/` (Hugging Face/Torch/tokenizers). |
| Models | Pre-packaged ONNX/CT2 assets staged under `$DIAREMOT_MODEL_DIR` (defaults to `/opt/models`). |

Environment variables respected by the tooling:

- `DIAREMOT_MODEL_DIR` — override the model root (defaults to `/opt/models`, `setup.sh` falls back to
  `<repo>/models` when `/opt` is not writable).
- `MODEL_RELEASE_URL` / `MODEL_RELEASE_SHA256` — override the release asset that `setup.sh` pulls.
- `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, `TRANSFORMERS_CACHE`, `TORCH_HOME`, `XDG_CACHE_HOME` — all
  default to `./.cache` during setup to keep downloads inside the repo.

---

## 2. Model layout

`setup.sh` (and the runtime) expect the following structure. The SHA-verified `models.zip` published
with the repository already matches this layout.

```
${DIAREMOT_MODEL_DIR}/
├── silero_vad.onnx
├── ecapa_onnx/
│   └── ecapa_tdnn.onnx
├── panns/
│   ├── model.onnx
│   └── class_labels_indices.csv
├── goemotions-onnx/
│   └── model.onnx
├── ser8-onnx/
│   ├── model.onnx          # FP32 fallback
│   └── model.int8.onnx      # INT8 preferred
├── faster-whisper-tiny.en/
│   └── model.bin
└── bart/
    ├── model_uint8.onnx
    ├── config.json
    ├── tokenizer.json       # or merges.txt + vocab.json
    ├── merges.txt
    └── vocab.json
```

> Tip: `maint-codex.sh` and `maintenance.sh` both validate this inventory and fail fast if a required
asset is missing.

---

## 3. Bootstrap

### One-command setup

```bash
bash ./setup.sh
```

The script performs:

1. Repository sanity checks (core modules must exist).
2. Virtualenv creation (re-used if already present) and dependency installation from `requirements.txt`.
3. Editable install of `diaremot` so console scripts (`diaremot`, `diaremot-diagnostics`) are
   available.
4. Cache normalisation under `./.cache` for Hugging Face, Torch, and tokenizers.
5. Model staging: downloads or reuses `models.zip`, verifies the SHA256, unpacks into
   `$DIAREMOT_MODEL_DIR`, and normalises the directory structure.
6. Optional 10 s synthetic sample via `ffmpeg` (`data/sample.wav`).
7. Import validation for the major pipeline modules.
8. `python -m diaremot.cli system diagnostics --strict` to confirm dependency health.

Adjust the model download by exporting `MODEL_RELEASE_URL` / `MODEL_RELEASE_SHA256` before running the
script. All steps mirror the manual instructions below—keep the README and script in sync.

### Manual steps (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .
$env:DIAREMOT_MODEL_DIR = "C:/diaremot/models"  # or another writable path
Expand-Archive models.zip -DestinationPath $env:DIAREMOT_MODEL_DIR
python -m diaremot.cli system diagnostics --strict
```

### Manual steps (POSIX shell)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .
export DIAREMOT_MODEL_DIR=${DIAREMOT_MODEL_DIR:-/opt/models}
unzip -o models.zip -d "$DIAREMOT_MODEL_DIR"
python -m diaremot.cli system diagnostics --strict
```

---

## 4. Running the pipeline

Preferred entry points are provided by `src/diaremot/cli.py` (Typer). Module execution works even
without installation because `setup.sh` extends `PYTHONPATH`, but installing the package exposes the
`diaremot` console script as well.

```bash
# End-to-end run (creates outputs under outputs/run1)
python -m diaremot.cli asr run --input data/sample.wav --outdir outputs/run1

# Resume a previous run (uses cached checkpoints, ignores --clear-cache)
python -m diaremot.cli asr resume --input data/sample.wav --outdir outputs/run1

# Regenerate summaries from a manifest (no re-processing)
python -m diaremot.cli report gen --manifest outputs/run1/manifest.json --format pdf --format html

# Inspect VAD behaviour with Silero
python -m diaremot.cli vad debug --input data/sample.wav --json

# Dependency and model diagnostics
python -m diaremot.cli system diagnostics --strict
```

Useful switches for `asr run`:

- `--profile fast|accurate|offline` — load bundled configuration overrides.
- `--whisper-model faster-whisper-medium.en` — use another CT2 bundle staged under
  `$DIAREMOT_MODEL_DIR`.
- `--chunk-enabled false` — disable automatic chunking for short files.
- `--affect-backend torch` — prefer Torch affect models over ONNX.
- `--disable-sed` or `--disable-affect` — trim the processing stack when debugging.

All CLI options are validated by `PipelineConfig`; invalid values produce descriptive errors before
processing begins.

---

## 5. Output artefacts

Each run writes a manifest JSON and a suite of artefacts in the chosen output directory:

| File | Description |
| --- | --- |
| `manifest.json` | High-level summary (paths, run id, dependency health). |
| `diarized_transcript_with_emotion.csv` | Main per-segment table (timestamps, transcript, affect, SER, VQ metrics). |
| `segments.jsonl` | JSONL mirror of the transcript for downstream automation. |
| `speakers_summary.csv` | Aggregated speaker rollups (dominance, words, emotions, interruptions). |
| `timeline.csv` | Turn-by-turn diarisation timeline. |
| `qc_report.json` | Diagnostics (overlap, cache health, dependency issues). |
| `summary.html` / `summary.pdf` | Executive summaries with key moments, charts, and voice metrics. |
| `speaker_registry.json` | Shared identity registry (path configurable via `--registry-path`). |

Conversation analytics (turn-taking balance, interruption rates, response latency, topic coherence)
and overlap statistics feed both the QC report and the HTML/PDF summaries. If a stage fails, the
pipeline still tries to emit partial outputs so downstream tooling can surface actionable errors.

---

## 6. Maintenance & diagnostics

- `maintenance.sh` / `maint-codex.sh` — lightweight health checks for already-provisioned
  environments. They validate the model inventory, import the package, and run the Typer diagnostics
  with `--strict`. Exit codes are non-zero if models or dependencies are missing.
- `python -m diaremot.pipeline.pipeline_diagnostic` — deeper introspection tool used internally by CI
  (checks `ffmpeg`, optional Python modules, and prints remediation steps).
- `pytest -q` — unit tests. Run inside the activated virtualenv once dependencies are installed.

---

## 7. Development notes

- Source lives under `src/diaremot`. Key modules include:
  - `pipeline/orchestrator.py` — core execution engine with caching, checkpoints, and manifest
    assembly.
  - `affect/emotion_analyzer.py` — wraps text affect, SER, SED, and voice-quality extraction.
  - `summaries/` — HTML/PDF generation, speaker rollups, and conversation flow analysis.
- Configuration is validated by `PipelineConfig` (`pipeline/config.py`). Stick to the provided keys
  when introducing new CLI options or defaults.
- The repository keeps caches local (`.cache/`) to avoid polluting global environments. Respect the
  same pattern for new downloads.
- When adding dependencies, update both `requirements.txt` and `pyproject.toml`, then adjust
  `setup.sh`, documentation, and diagnostics accordingly.

---

## 8. Troubleshooting

| Symptom | Likely cause / fix |
| --- | --- |
| `ffmpeg` errors or missing sample file | Install `ffmpeg` and ensure it is on `PATH`. `setup.sh` warns but continues. |
| Diagnostics report missing modules | Re-run `./setup.sh` or `python -m pip install -e .` inside the venv. |
| `models.zip` SHA mismatch | Update `MODEL_RELEASE_URL`/`MODEL_RELEASE_SHA256`, or refresh the asset in the GitHub release. |
| Transcription fallback warnings | Check the manifest `dependency_unhealthy` list and the logs under `outputs/<run>/logs`. |
| Conversation analysis skipped | Requires valid transcripts; inspect QC report for earlier stage failures. |

Optional helpers:

- **PyAV (`av`)** — improves duration probing for exotic containers. Install via `python -m pip install av`.
- **panns_inference** — optional PyTorch-based SED fallback. The default ONNX runtime does not need it.

For additional context see `src/diaremot/README.md`, which mirrors the developer-focused details used
by the Typer CLI.
