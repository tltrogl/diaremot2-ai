# DiaRemot — CPU‑Only Speech Intelligence Pipeline

Diarization + transcription + affect (valence/arousal/dominance, speech emotion, text emotions, intent) + sound‑event context (SED) — all **CPU‑only**, built for 1–3 hour noisy recordings.


```mermaid
flowchart LR
    A[Quiet-Boost\nPre-VAD] --> B[SED (PANNs CNN14 ONNX)\n1s/0.5s hop + hysteresis]
    A --> C[Silero VAD (ONNX)]
    C --> D[Diart + ECAPA-TDNN (ONNX)\nAHC clustering]
    D --> E[Transcription (Faster-Whisper tiny-en, CT2 INT8)]
    D --> F[Affect (audio): V/A/D + 8-class SER]
    E --> G[Affect (text): GoEmotions 28 + Intent ZS]
    B -. overlaps .-> E
    B -. overlaps .-> F
    E --> H[Outputs: CSV/JSONL/HTML, Speaker Registry]
```


## Key Features
- **Quiet‑Boost** preprocessing: HPF 80–120 Hz, light denoise, gated gain for soft voices, gentle compression, LUFS normalize, resample → **16 kHz mono**.
- **SED first** (PANNs CNN14 ONNX; YAMNet fallback). Attach event overlaps to segments.
- **Diarization** (Diart on CPU): Silero VAD → ECAPA‑TDNN embeddings → Agglomerative clustering.
- **ASR**: Faster‑Whisper `tiny-en` (CTranslate2 INT8). Run **only** on diarized speech.
- **Affect (audio)**: V/A/D via `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`; 8‑class SER via `Dpngtm/wav2vec2-emotion-recognition`.
- **Affect (text)**: GoEmotions (28) via `SamLowe/roberta-base-go_emotions`, plus **intent** via `facebook/bart-large-mnli` zero‑shot.
- **Persistent speaker registry**: ECAPA centroids and cosine matching across runs.
- **Human‑friendly reporting**: summary.html (Quick Take, Speaker Snapshots, Moments to Check), CSVs for scrubbing, JSONL for programmatic use.

## Repository Map
```
src/diaremot/
  pipeline/
    audio_preprocessing.py   # Quiet-Boost
    speaker_diarization.py   # Silero VAD + ECAPA + AHC
    transcription.py         # Faster-Whisper (CT2)
  affect/
    affect_audio.py          # V/A/D, aggregation windows
    ser_dpngtm.py            # 8-class speech emotion (Torch)
    affect_text.py           # GoEmotions
    intent_zero_shot.py      # BART MNLI zero-shot
  sed/
    sed_panns_onnx.py        # PANNs CNN14 via onnxruntime
    sed_yamnet_tf.py         # Fallback
  io/download_utils.py       # Helper for model assets
  cli.py                     # CLI entry
```

## Outputs
- `diarized_transcript_with_emotion.csv` — Primary scrub-friendly table.
- `segments.jsonl` — Per-segment payload.
- `speakers_summary.csv` — Per-speaker rollups.
- `summary.html` — Narrative + moments to check.
- `speaker_registry.json` — Persistent centroids and metadata.
- `events_timeline.csv` + `events.jsonl` — SED details.
- `timeline.csv`, `qc_report.json` — health checks.

## Quickstart (Codex Cloud)
Use the Codex‑ready setup script to get a clean CPU environment with ffmpeg and pinned wheels:
```bash
bash setup.sh
# or: ./setup.sh
```

Process an example file:
```bash
python -m diaremot.pipeline.audio_preprocessing data/sample.wav runs/sample_16k.wav --target-sr 16000
python -m diaremot.cli run --input data/sample.wav --outdir runs/sample_run
```

## Known Issues & Fixes
- **librosa resample API** changed in 0.10+. If you see `librosa.core.resample missing`, update code to `librosa.resample` or pin librosa < 0.10. Docs here assume code is **0.10+ ready**.
- **Torch extras**: `torchvision`/`torchaudio` are **not used** by this repo and are removed from requirements for faster, smaller CPU installs.

## License
Proprietary — see project headers.
