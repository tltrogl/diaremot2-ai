# AGENTS_CLOUD.md — DiaRemot in Codex Cloud

This file teaches Codex to act as your build‑and‑run assistant for the DiaRemot repo. Codex should **plan → verify → apply → test** in small loops, prefer **PowerShell for Windows snippets** but run **bash** inside Codex Cloud.

## Mission
- Stand up a CPU‑only environment.
- Run full pipeline on provided audio.
- Edit code on request; send diffs back.
- Keep dependencies minimal and deterministic.

## Operating Rules
1. **Plan then do**: summarize intended changes, then apply them.
2. **Atomic edits**: when modifying files, produce a unified diff.
3. **Validate**: run smoke tests (`--help`, import checks, 10–20s audio slice).
4. **Explain failures**: include the exact stderr and your fix.
5. **Don’t install what we don’t use**: no `torchvision`/`torchaudio` unless imports demand it.

## Environment Setup
In Codex Cloud (Ubuntu‑like), run:
```bash
bash setup.sh
```
What it ensures:
- Python 3.11+ with `uv` or `pip`, ffmpeg present
- CPU Torch installed explicitly (no GPU wheels)
- Repo installed in editable mode

## Run Maps
### 1) End‑to‑end run (CLI)
```bash
python -m diaremot.cli run --input data/sample.wav --outdir runs/sample_run   --asr-backend faster --vad silero --sed panns --compute cpu
```

### 2) Stage-by-stage (for debugging)
```bash
python -m diaremot.pipeline.audio_preprocessing input.wav runs/boosted.wav --target-sr 16000
python -m diaremot.sed.sed_panns_onnx runs/boosted.wav runs/events_timeline.csv
python -m diaremot.pipeline.speaker_diarization runs/boosted.wav runs/diarization.json
python -m diaremot.pipeline.transcription runs/boosted.wav runs/asr.json --ct2 --tiny-en
python -m diaremot.affect.affect_audio runs/boosted.wav runs/affect_audio.json
python -m diaremot.affect.affect_text runs/asr.json runs/text_emotions.json
python -m diaremot.affect.intent_zero_shot runs/asr.json runs/intent.json
```

## Model Assets (CPU)
- **ASR**: Faster‑Whisper tiny‑en (CTranslate2 INT8)
- **Diarization**: Silero VAD (ONNX) + ECAPA‑TDNN (ONNX)
- **SED**: PANNs CNN14 (ONNX), YAMNet fallback
- **SER**: 8‑class (Torch‑based)
- **Text**: GoEmotions (Transformers), BART MNLI for intent

Codex: cache models once; prefer local paths (`D:/models` on Windows or `./models` in Codex Cloud).

## Verification Checklist
- `python -c "import diaremot; print(diaremot.__version__)"`
- `python -m diaremot.cli --help`
- Short run (`<=20s` wav) completes and writes: CSV, JSONL, HTML, registry
- No imports of `torchvision`/`torchaudio`

## Common Pitfalls
- `librosa.core.resample` error → update to `librosa.resample` (0.10+).
- `uv pip` + `--index-url` in requirements → install Torch **separately** first (see `setup.sh`).
