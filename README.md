# DiaRemot — CPU-first speech intelligence pipeline

DiaRemot ingests longform speech recordings on CPU-only hosts and produces:

- **Diarized, emotion-tagged transcripts** ready for analysis.
- **Speaker-level rollups** plus overlap/interrupt metrics.
- **Timeline CSV, HTML & PDF summaries, JSONL segments, and QC reports.**

The system prioritises resiliency on constrained machines: all heavy components run on CPU, stages checkpoint to disk, and cached diarization/transcription results can be reused across reruns.

---

## Pipeline architecture

| Stage | Module / Class | Description | Primary outputs |
| ----- | --------------- | ----------- | --------------- |
| 0. Dependency probe | `audio_pipeline_core._dependency_health_summary` | Reports missing/unsupported libraries before work begins. | Issues logged & persisted in QC report.
| 1. Preprocess | `AudioPreprocessor.process_file` | Decodes audio (ffmpeg fallback), denoises, normalises loudness, auto-chunks long files. | 16 kHz mono signal, `AudioHealth` metrics, optional temp chunks.
| 1b. Background SED | `PANNSEventTagger` (optional) | Tags non-speech events for context. | Noise / tag metadata appended to manifest.
| 2. Diarization | `SpeakerDiarizer` (`cpu_optimized_diarizer` wrapper optional) | Generates time-aligned speaker turns, manages speaker registry. | Turn list cached to `.cache/<sha>/diar.json`, registry updates.
| 3. Transcription | `AudioTranscriber.transcribe_segments` | Runs Faster-Whisper (default `faster-whisper-tiny.en`) with batching, using diarization to guide segments. | Normalised ASR segments cached to `.cache/<sha>/tx.json`.
| 4. Paralinguistics | `paralinguistics.extract` (best effort) | Computes speech-rate, loudness, disfluency, and voice-quality hints per segment. | Supplemental metrics keyed by segment index.
| 5. Affect assembly | `EmotionIntentAnalyzer.analyze` | Combines audio & text affect models (ONNX/Torch selectable) into unified affect payload per segment. | Affect-rich segment rows (valence/arousal/emotion/intent).
| 6. Overlap metrics | `paralinguistics.compute_overlap_and_interruptions` | Measures speaker overlap and interruptions. | Global overlap stats and per-speaker interruption counts.
| 7. Conversation metrics | `analyze_conversation_flow` | Calculates turn-taking balance, pace, topic coherence. | `ConversationMetrics` used in summaries.
| 8. Speaker rollups | `build_speakers_summary` | Aggregates speaker durations, pace, affect hints, overlaps. | `speakers_summary.csv` rows.
| 9. Reporting | `HTMLSummaryGenerator` / `PDFSummaryGenerator` & `_write_outputs` | Writes canonical outputs & QC artefacts. | CSV/JSONL timeline, summaries, HTML+PDF, QC JSON, logs, checkpoints.

Cache-aware resume logic lives in `PipelineCheckpointManager` and automatically reuses diarization/ASR caches when the preprocessed audio fingerprint (`audio_sha16`) and preprocessing signature match.

---

## Entry points

- **CLI (primary)** — `python -m diaremot.cli run --input INPUT.wav --outdir outputs/run_YYYYMMDD_HHMMSS`
  - `run`, `resume`, and `diagnostics` commands map directly to `audio_pipeline_core.run_pipeline`, `resume`, and `diagnostics` respectively.
  - Built-in profiles: `default`, `fast`, `accurate`, `offline` (all now default to `faster-whisper-tiny.en`).
- **Python API** — `from diaremot.pipeline.audio_pipeline_core import run_pipeline, AudioAnalysisPipelineV2`
  - Call `run_pipeline(path, outdir, config=overrides)` for a one-shot run.
  - Instantiate `AudioAnalysisPipelineV2(config)` and call `process_audio_file` for finer-grained control or integration tests.

`diaremot.pipeline.run_pipeline` remains as a thin compatibility shim that forwards to `audio_pipeline_core` while ensuring cache directories are initialised.

---

## Requirements

- **OS:** Windows 10/11 or Linux (x86_64 tested).
- **Python:** 3.11.
- **CPU:** AVX2-capable CPUs recommended; GPU is not required.
- **Disk:** ≥6 GB free for models, caches, and artefacts.
- **FFmpeg:** required for non-PCM inputs and chunk extraction.

### Optional caches

Set these env vars to persist model downloads locally:

```powershell
$env:HF_HOME = "$PWD/.cache/hf"
$env:TRANSFORMERS_CACHE = "$PWD/.cache/transformers"
```

The pipeline also honours `WHISPER_MODEL_PATH` (override Faster-Whisper path) and will default to repository-local `.cache/` directories if unset.

---

## Installation

> Packaging metadata is still evolving. If editable installs fail, fall back to `pip install -r requirements.txt`.

### Windows (PowerShell)
```powershell
python --version              # expect 3.11.x
python -m venv .venv
. .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
# Optional editable install (if pyproject/setup is published):
# pip install -e .
```

### Linux/macOS (bash)
```bash
python3 --version             # expect 3.11.x
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
# Optional editable install (if packaging is present):
# pip install -e .
```

If PyTorch CPU wheels are missing:
```bash
pip install --index-url https://download.pytorch.org/whl/cpu \
  torch==2.4.1+cpu torchaudio==2.4.1+cpu torchvision==0.19.1+cpu
```

---

## Running the pipeline

### 1. Diagnostics
```powershell
diaremot-diagnostics --strict
# or
python -m diaremot.cli diagnostics --strict
```

### 2. Full run
```powershell
$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
python -m diaremot.cli run --input data\sample.wav --outdir "outputs\run_$ts"
```

### 3. Resume a prior run
```powershell
python -m diaremot.cli resume --input data\sample.wav --outdir "outputs\run_$ts"
```

Outputs created under `--outdir`:

- `diarized_transcript_with_emotion.csv`
- `segments.jsonl`
- `timeline.csv`
- `speakers_summary.csv`
- `summary.html`
- `summary.pdf`
- `qc_report.json`
- `logs/` (stage logs & diagnostics)
- `artifacts/` (figures, audio snippets)
- `checkpoints/` (stage checkpoints for resume)

The JSON manifest printed to stdout includes the run ID, resolved output paths, dependency health summary, and transcriber metadata.

---

## Configuration & profiles

Use `--profile` (built-in or path to a JSON overlay) or individual CLI flags to tune behaviour. A JSON profile only overrides the keys you specify; everything else inherits from the defaults returned by `build_pipeline_config`.

Common knobs:

- **ASR:** `--asr-backend`, `--asr-model`, `--asr-compute-type`, `--asr-cpu-threads`, `--language`, `--language-mode`, `--beam-size`, `--temperature`, `--no-speech-threshold`.
- **Diarization:** `--ahc-distance-threshold`, `--speaker-limit`, `--vad-threshold`, `--vad-backend`, `--disable-energy-fallback`, `--cpu-diarizer`.
- **Preprocessing:** `--noise-reduction`, `--chunk-enabled/--no-chunk`, chunk durations & overlaps.
- **Affect stack:** `--disable-affect`, `--affect-backend auto|onnx|torch`, `--affect-text-model-dir`, `--affect-intent-model-dir`.
- **Caching & resiliency:** `--ignore-tx-cache`, `--clear-cache`, `--threads`, `--quiet`.

Programmatic runs can pass the same keys via the `config` dict. See `DEFAULT_PIPELINE_CONFIG` in `audio_pipeline_core.py` for the complete list.

---

## Caching, checkpoints & resume

- Caches live under `.cache/<audio_sha16>/` (primary root configurable via `cache_root`, additional read-only roots via `cache_roots`).
  - `diar.json` stores diarization turns; `tx.json` stores normalised ASR segments; `.done` marks a fully successful run.
- `checkpoints/` records per-stage payloads through `PipelineCheckpointManager` and powers the `resume` command.
- Use `--ignore-tx-cache` to force re-diarization/transcription even if caches match, or `--clear-cache` to purge caches before running.

---

## Troubleshooting quick hits

- Missing wheels? Install PyTorch CPU builds from the CPU index (see above).
- FFmpeg decode errors? Ensure `ffmpeg`/`ffprobe` are on PATH.
- VAD too chatty/strict? Adjust `--vad-threshold` or toggle energy fallback.
- Affect models failing on ONNX? Point `--affect-text-model-dir` / `--affect-intent-model-dir` to exported weights or switch `--affect-backend torch`.
- Crash mid-run? Re-run with `resume` and inspect `logs/last_stage.txt` and `qc_report.json`.

---

## License & support

The project is currently **proprietary** (see `pyproject.toml`). When reporting issues, include the command line, `diaremot-diagnostics --strict` output, and the tail of `logs/` for the failing run.
