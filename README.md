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
- **ASR**: Faster‑Whisper `tiny‑en` via CTranslate2 (`compute_type=int8`).
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
python -m diaremot.cli run --audio data/sample.wav --tag smoke --compute-type int8
```

## Environment variables

- `DIAREMOT_MODEL_DIR` — models root (e.g., `/workspace/models`).
- `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, `TRANSFORMERS_CACHE`, `TORCH_HOME` — `./.cache/`.
- Threads: `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_MAX_THREADS`.
- `TOKENIZERS_PARALLELISM=false`.

## CSV schema (primary)

```
file_id,start,end,speaker_id,speaker_name,text,
valence,arousal,dominance,
emotion_top,emotion_scores_json,
text_emotions_top5_json,text_emotions_full_json,
intent_top,intent_top3_json,
events_top3_json,noise_tag,
asr_logprob_avg,snr_db,snr_db_sed,
wpm,duration_s,words,pause_ratio,
vq_jitter_pct,vq_shimmer_db,vq_hnr_db,vq_cpps_db,voice_quality_hint
```

`events_top3_json` carries the top-k AudioSet clusters detected globally or per
segment (when available), `noise_tag` surfaces the dominant background class,
and `snr_db_sed` converts the SED noise score into an approximate SNR value for
triage. The CSV also exposes normalized segment duration, token counts, and the
pause ratio derived from paralinguistics, keeping downstream consumers aligned
with the contract.