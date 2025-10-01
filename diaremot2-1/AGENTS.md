# AGENTS.md

Working notes for DiaRemot (CPU-first diarization + ASR + affect pipeline).

## Entry points & architecture
- **Primary CLI:** `python -m diaremot.cli`. Commands `run`, `resume`, `diagnostics` proxy to `diaremot.pipeline.audio_pipeline_core`.
- **Programmatic:** import `AudioAnalysisPipelineV2` or `run_pipeline` from `diaremot.pipeline.audio_pipeline_core`.
- Pipeline stages (see `audio_pipeline_core.AudioAnalysisPipelineV2`):
  1. Preprocess (`AudioPreprocessor`) → caches & health metrics.
  2. Diarization (`SpeakerDiarizer` or CPU wrapper) → `turns` cache.
  3. Transcription (`AudioTranscriber`, default `faster-whisper-tiny.en`) → `segments` cache.
  4. Paralinguistics/Affect (`paralinguistics`, `EmotionIntentAnalyzer`) → affect-rich rows.
  5. Reporting (`HTMLSummaryGenerator`, `PDFSummaryGenerator`) → CSV/JSONL/HTML/PDF/QC artefacts.
- Cache+checkpoint system lives under `.cache/<audio_sha16>/` and `checkpoints/` and powers `resume`.

## Setup
1. `python -m venv .venv`
2. Activate: Windows `.\.venv\Scripts\Activate.ps1`; POSIX `source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. Optional: set `HF_HOME`, `TRANSFORMERS_CACHE`, `WHISPER_MODEL_PATH` for local model storage.
5. Sanity check: `python -m diaremot.cli diagnostics --strict`

> Missing Torch wheels? Install from the CPU index: `pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1+cpu torchaudio==2.4.1+cpu torchvision==0.19.1+cpu`.

## Running
```powershell
$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
python -m diaremot.cli run --input data\sample.wav --outdir "outputs\run_$ts"
```
Resume with the same arguments via `python -m diaremot.cli resume ...`.

Expected contents in `--outdir`:
- `diarized_transcript_with_emotion.csv`
- `segments.jsonl`
- `timeline.csv`
- `speakers_summary.csv`
- `summary.html`
- `summary.pdf`
- `qc_report.json`
- `logs/`, `artifacts/`, `checkpoints/`

## Smoke / verification
- CLI banners: `python -m diaremot.cli --help`, `python -m diaremot.cli run --help`
- Quick WAV (≤30 s): `python -m diaremot.cli run --input tests\data\short.wav --outdir outputs\ci_test`
  - Expect CSV with ≥1 data row, HTML/PDF summaries, and QC JSON.

## Conventions
- Python 3.11, prefer type hints.
- Profiles (`--profile`) are JSON overlays that merge with `DEFAULT_PIPELINE_CONFIG`.
- `EmotionIntentAnalyzer` honours `affect_backend`, `affect_text_model_dir`, `affect_intent_model_dir` from the config.
- Logs land in `logs/run.jsonl`; aggregate QC in `qc_report.json`.

## Troubleshooting
- FFmpeg missing? Install `ffmpeg` / add to PATH.
- Affect backends: switch via `--affect-backend auto|onnx|torch`.
- Force fresh ASR/diarization: `--ignore-tx-cache` or `--clear-cache`.
- Crash late? `resume` then inspect `logs/last_stage.txt` and `qc_report.json`.

## License
Proprietary (match `pyproject.toml`).

