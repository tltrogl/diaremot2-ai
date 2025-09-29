# DiaRemot — CPU-Only Speech Intelligence Pipeline

Process 1–3 hr, noisy + soft-voice recordings **on CPU** → diarized transcript with per-segment tone (V/A/D),
speech emotion (8-class), text emotions (GoEmotions, 28), intent (14-label), sound events (SED), and persistent
speaker names. Outputs include CSV/JSONL/HTML with QC signals.

## Installation (Windows • PowerShell first)
1) **Python 3.11–3.12** (recommended)
2) **FFmpeg on PATH** (required). Verify:
```powershell
ffmpeg -version
```
3) **Virtual environment + install** (derived from real imports)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

### Linux/macOS
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
pip install -e .
```

## Quick run
```powershell
python -m diaremot --input "D:\audio\call.wav" --out "outputs\run_$(Get-Date -Format yyyyMMdd_HHmmss)"
```

## Development
- **Format/Lint**: Ruff
```powershell
ruff format .
ruff check --fix .
```
- **Tests**:
```powershell
pytest -q
```
- **Build**:
```powershell
python -m build
```

## Dependency Policy
- `requirements.txt` is generated from the *actual* imports in `src/` (no trust in pyproject pins).
- If a runtime error calls out a missing module, add it here and re-install.
- Torch/Torchaudio/Torchvision are CPU-mode by default; CUDA is not required.

## Notes
- FFmpeg must be on PATH for decode/encode and any `ffmpeg-python` bindings.
- If duplicate pipeline launchers exist (e.g., root `audio_pipeline_core.py` vs. `src/diaremot/pipeline/...`), prefer the package entry point.
