# DiaRemot — CPU‑Only Speech Intelligence Pipeline

Process long, real‑world audio **on CPU** and produce a diarized transcript with per‑segment:
- **Tone (Valence/Arousal/Dominance)**
- **Speech emotion (8‑class)**
- **Text emotions (GoEmotions, 28)**
- **Intent** (zero‑shot over fixed labels)
- **Sound‑event context (SED: music, keyboard, door, TV, etc.)**
- **Paralinguistics (REQUIRED)**: speech rate (WPM), pauses, and voice‑quality via **Praat‑Parselmouth**: **jitter**, **shimmer**, **HNR**, **CPPS**
- **Persistent speaker names across files**

Outputs:
- `diarized_transcript_with_emotion.csv` — primary, scrub‑friendly
- `segments.jsonl` — per‑segment payload (audio + text + SED overlaps)
- `speakers_summary.csv` — per‑speaker rollups (V/A/D, emotion mix, intents, WPM, SNR, voice‑quality)
- `summary.html` — Quick Take, Speaker Snapshots, Moments to Check (SED), Action Items
- `speaker_registry.json` — persistent names via centroids
- `events_timeline.csv` + `events.jsonl` — SED events
- `timeline.csv`, `qc_report.json` — fast scrubbing + health checks

## Model set (CPU‑friendly)

- **Diarization**: Diart (Silero VAD + ECAPA‑TDNN embeddings + AHC). Prefers ONNX, Torch fallback for Silero VAD.
  - **Adaptive VAD tuning**: Pipeline automatically relaxes VAD thresholds for soft-speech scenarios:
    - `vad_threshold`: 0.22 (relaxed from CLI default 0.30)
    - `vad_min_speech_sec`: 0.40s (relaxed from 0.80s)
    - `vad_min_silence_sec`: 0.40s (relaxed from 0.80s)
    - `speech_pad_sec`: 0.15s (relaxed from 0.20s)
  - Override adaptive tuning via CLI: `--vad-threshold 0.3 --vad-min-speech-sec 0.8`
- **ASR**: Faster‑Whisper `tiny.en` via CTranslate2 (default `compute_type=int8`).
- **Tone (V/A/D)**: `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`.
- **Speech emotion (8‑class)**: `Dpngtm/wav2vec2-emotion-recognition`.
- **Text emotions (28)**: `SamLowe/roberta-base-go_emotions` (full distribution; keep top‑5).
- **Intent**: Prefers local ONNX exports (e.g., `model_uint8.onnx` under `affect_intent_model_dir` such as `D:\\diaremot\\diaremot2-1\\models\\bart\\`) and falls back to the `facebook/bart-large-mnli` Hugging Face pipeline when no ONNX asset is available.
- **SED**: PANNs CNN14 (ONNX) on onnxruntime; 1.0s frames, 0.5s hop; median filter 3–5; hysteresis 0.50/0.35; `min_dur=0.30s`; `merge_gap≤0.20s`; collapse AudioSet→~20 labels.
- **Paralinguistics (REQUIRED)**: **Praat‑Parselmouth** for jitter/shimmer/HNR/CPPS + prosody (WPM/pauses).

## Install (local; Windows PowerShell shown)

1) **Python 3.11**; **FFmpeg on PATH** (`ffmpeg -version`).
2) Create venv and install:
```powershell
py -3.11 -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install -U pip wheel setuptools
pip install -r requirements.txt
pip install -e .
```
3) Dev tools:
```powershell
pip install ruff pytest mypy
```

## Codex Cloud usage

```bash
./setup.sh
./maint-codex.sh
python -m diaremot.cli run --input data/sample.wav --outdir outputs/ --asr-compute-type int8
```

## Environment variables

- `DIAREMOT_MODEL_DIR` — models root (e.g., `/workspace/models`).
- `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, `TRANSFORMERS_CACHE`, `TORCH_HOME` — `./.cache/`.
- Threads: `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_MAX_THREADS`.
- `TOKENIZERS_PARALLELISM=false`.

## CSV schema (primary)

The canonical schema is defined in `src/diaremot/pipeline/outputs.py::SEGMENT_COLUMNS` (39 columns):

```
file_id,start,end,speaker_id,speaker_name,text,
valence,arousal,dominance,
emotion_top,emotion_scores_json,
text_emotions_top5_json,text_emotions_full_json,
intent_top,intent_top3_json,
events_top3_json,noise_tag,
asr_logprob_avg,snr_db,snr_db_sed,
wpm,duration_s,words,pause_ratio,
low_confidence_ser,vad_unstable,affect_hint,
pause_count,pause_time_s,
f0_mean_hz,f0_std_hz,
loudness_rms,disfluency_count,
error_flags,
vq_jitter_pct,vq_shimmer_db,vq_hnr_db,vq_cpps_db,voice_quality_hint
```

**Key columns:**
- `events_top3_json` — Top-k AudioSet clusters detected per segment
- `noise_tag` — Dominant background class
- `snr_db_sed` — Approximate SNR from SED noise score
- `low_confidence_ser` — Speech emotion recognition confidence flag
- `vad_unstable` — VAD instability indicator
- `affect_hint` — Affect state summary (e.g., "calm-positive", "agitated-negative")
- `pause_count`, `pause_time_s` — Pause metrics from paralinguistics
- `f0_mean_hz`, `f0_std_hz` — Pitch statistics
- `loudness_rms` — RMS loudness
- `disfluency_count` — Filler words and hesitations
- `error_flags` — Processing error indicators
- `vq_jitter_pct`, `vq_shimmer_db`, `vq_hnr_db`, `vq_cpps_db` — Voice quality (Praat-Parselmouth)
- `voice_quality_hint` — Voice quality interpretation