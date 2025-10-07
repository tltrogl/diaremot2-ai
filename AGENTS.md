# AGENTS_CLOUD.md — DiaRemot Agent Instructions (Codex Cloud)

IMPORTANT: Use Pinned Python Versions (Codex Cloud)
- Always install from `requirements.txt` inside the repo venv. Do not upgrade.
- Key pins that MUST be present in Codex Cloud runs:
  - onnxruntime==1.17.1
  - faster-whisper==1.1.0, ctranslate2==4.6.0
  - transformers==4.38.2, tokenizers==0.15.2
  - torch==2.4.1+cpu, torchaudio==2.4.1+cpu, torchvision==0.19.1+cpu
  - librosa==0.10.2.post1, numpy==1.24.4, scipy==1.10.1, numba==0.59.1, llvmlite==0.42.0
  - pandas==2.0.3, scikit-learn==1.3.2
  - praat-parselmouth==0.4.3, panns-inference==0.1.1

Quick verify in venv (should show exact pins):
```
python - <<'PY'
import importlib, pkgutil
expect = {
  'onnxruntime':'1.17.1', 'faster-whisper':'1.1.0', 'ctranslate2':'4.6.0',
  'transformers':'4.38.2', 'tokenizers':'0.15.2',
  'torch':'2.4.1+cpu', 'torchaudio':'2.4.1+cpu', 'torchvision':'0.19.1+cpu',
  'librosa':'0.10.2.post1','numpy':'1.24.4','scipy':'1.10.1','numba':'0.59.1','llvmlite':'0.42.0',
  'pandas':'2.0.3','scikit-learn':'1.3.2','praat-parselmouth':'0.4.3','panns-inference':'0.1.1'
}
from importlib.metadata import version
ok=True
for k,v in expect.items():
    try:
        got=version(k)
        if got!=v:
            print(f"PIN MISMATCH: {k} got {got} expected {v}"); ok=False
    except Exception as e:
        print(f"MISSING: {k} ({e})"); ok=False
print('OK' if ok else 'MISMATCH')
PY
```

Last updated: 2025-10-06

Role: System Architect / Maintainer for DiaRemot on Codex Cloud
Plan → Implement → Verify → Report for every change. Build real, working code.

Important Differences vs generic AGENTS.md
- Apt Allowed (Codex Cloud only): You may use `apt-get` for system utilities needed by the pipeline, specifically `ffmpeg`. Prefer pip wheels for Python packages. Do not rely on apt on Windows/macOS.
- CPU Only: Continue enforcing CPU-only execution. Do not enable GPU providers.
- Ephemeral FS: Cache only under `./.cache/`. Models and tool binaries may be staged there.

Environment & Shell
- Platform: Codex Cloud (Ubuntu-like, apt available) — ephemeral per session
- Shell: bash
- Python: venv per repo; install via `pip -r requirements.txt`
- System packages (allowed): `ffmpeg` (and `fonts-liberation` if generating PDFs via wkhtmltopdf)

Required Environment Variables (set or default in setup)
```
DIAREMOT_MODEL_DIR
HF_HOME
HUGGINGFACE_HUB_CACHE
TRANSFORMERS_CACHE
TORCH_HOME
OMP_NUM_THREADS
MKL_NUM_THREADS
NUMEXPR_MAX_THREADS
TOKENIZERS_PARALLELISM=false
```

Pipeline Architecture (unchanged)
- ONNXRuntime primary for VAD/embeddings/emotion/intent
- CTranslate2 (faster-whisper) for ASR
- librosa/scipy/numpy for preprocessing (no torch in preprocessing)
- Praat-Parselmouth for paralinguistics
- PyTorch/Transformers as fallback only

Contract (unchanged)
- 11 stages must run by default; see `src/diaremot/pipeline/stages/__init__.py::PIPELINE_STAGES`
- Segment CSV schema is 39 columns as defined in `src/diaremot/pipeline/outputs.py::SEGMENT_COLUMNS`
- Paralinguistics stage must emit all 14 metrics: `wpm`, `duration_s`, `words`, `pause_count`, `pause_time_s`, `pause_ratio`, `f0_mean_hz`, `f0_std_hz`, `loudness_rms`, `disfluency_count`, `vq_jitter_pct`, `vq_shimmer_db`, `vq_hnr_db`, `vq_cpps_db`
- ASR default compute type for main CLI is `float32`

Models & Assets
- Default model directory: `./.cache/models` (via `DIAREMOT_MODEL_DIR`)
- Use your GitHub Release asset `models.zip` (recommended). Provide its URL and SHA256.
  - Env: `DIAREMOT_AUTO_DOWNLOAD=1`, `DIAREMOT_MODELS_URL=https://github.com/OWNER/REPO/releases/download/TAG/models.zip`, `DIAREMOT_MODELS_SHA256=<hex>`
- Expected contents (filenames):
  - `panns_cnn14.onnx`, `audioset_labels.csv`
  - `silero_vad.onnx`, `ecapa_tdnn.onnx`
  - `ser_8class.onnx`, `vad_model.onnx`
  - `roberta-base-go_emotions.onnx`, `bart-large-mnli.onnx`

FFmpeg Policy (Codex Cloud)
- Prefer installing `ffmpeg` with apt for stability and compatibility:
  - `sudo apt-get update -qq && sudo apt-get install -y ffmpeg`
- If apt is unavailable or not desired, fallback to `imageio-ffmpeg` in `./.cache/bin` (setup.sh supports this).

Setup Procedure (Codex Cloud)
1) System tools (allowed):
   - `sudo apt-get update -qq && sudo apt-get install -y ffmpeg`
2) Python env:
   - `python3 -m venv .venv && source .venv/bin/activate`
   - `python -m pip install --upgrade pip setuptools wheel`
   - `python -m pip install -r requirements.txt`
3) Caches/env defaults:
   - Set `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, `TRANSFORMERS_CACHE`, `TORCH_HOME` to `./.cache/*`
   - Threads: cap `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_MAX_THREADS` to ≤4
   - `TOKENIZERS_PARALLELISM=false`, `CUDA_VISIBLE_DEVICES=""`, `TORCH_DEVICE=cpu`
4) Models:
   - Provide `models.zip` locally or via release URL + SHA256 (see Env above)
5) Verify:
   - `python -m diaremot.pipeline.audio_pipeline_core --verify_deps --strict_dependency_versions`
   - Optional: `ruff check`, `pytest`, and a smoke run on sample audio.

Operating Procedure (Codex Cloud)
Plan (5–10 bullets) → Minimal diffs → Verify (lint/tests/smoke) → Report (diffs, cmds, logs, artifacts).

Safety & Constraints (still apply)
- CPU-only; ONNX primary; PyTorch fallback only
- Do not break CLI entry points (`python -m diaremot.cli`, etc.)
- Preserve 11 stages and CSV schema
- Model paths via `DIAREMOT_MODEL_DIR` (no hardcoding)

Troubleshooting Notes (Codex Cloud)
- If `torch._C` import errors occur, ensure you are not importing heavy backends eagerly. The codebase now lazy-loads transcription backends.
- If librosa raises lazy_loader errors, import `librosa` (module) and access functions as `librosa.func` (already implemented).
