Codex: you are my repo assistant. Use plan → verify → apply loops, produce unified diffs, and run smoke tests after each change. Prefer bash in Codex Cloud; PowerShell snippets for Windows users.

Responsibilities

Stand up CPU-only env (setup.sh).

Run pipeline E2E and per-stage.

Edit code on request and post diffs.

Diagnose failures with exact stderr and next-step fixes.

Setup
Always show details
bash setup.sh
python -m diaremot.cli --help

Run Maps
End-to-End
Always show details
python -m diaremot.cli run --input data/sample.wav --outdir runs/sample_run \
  --asr-backend faster --vad silero --sed panns --compute cpu

Per-Stage
Always show details
python -m diaremot.pipeline.audio_preprocessing input.wav runs/boosted.wav --target-sr 16000
python -m diaremot.sed.sed_panns_onnx runs/boosted.wav runs/events_timeline.csv --enter 0.5 --exit 0.35
python -m diaremot.pipeline.speaker_diarization runs/boosted.wav runs/diarization.json
python -m diaremot.pipeline.transcription runs/boosted.wav runs/asr.json --ct2 --tiny-en
python -m diaremot.affect.affect_audio runs/boosted.wav runs/affect_audio.json
python -m diaremot.affect.affect_text runs/asr.json runs/text_emotions.json
python -m diaremot.affect.intent_zero_shot runs/asr.json runs/intent.json

Models & Caching

ASR: Faster-Whisper tiny-en (CTranslate2 INT8)

Diarization: Silero VAD (ONNX) + ECAPA-TDNN (ONNX)

SED: PANNs CNN14 (ONNX), YAMNet fallback

SER: Wav2Vec2-based (Torch)

Text: GoEmotions + BART-MNLI

Verification

python -c "import diaremot, onnxruntime, ctranslate2"

Process 10–20s WAV; emit CSV/JSONL/HTML; speaker_registry.json updated

No imports of torchvision/torchaudio

Failure Handling

Paste stderr, show offending lines, propose single-change patch, rerun smoke, repeat.
"""
