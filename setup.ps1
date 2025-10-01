Param(
  [string]$Py="py"
)
Write-Host "[info] Environment bootstrap (Windows/PowerShell)"
try { & $Py -3.11 -c "import sys; print(sys.version)" | Out-Null } catch { Write-Error "Python 3.11 required"; exit 1 }

$ff = Get-Command ffmpeg -ErrorAction SilentlyContinue
if (-not $ff) { Write-Warning "FFmpeg not found on PATH. Install it and re-run." }

& $Py -3.11 -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
pip install -e .

ruff format .
ruff check --fix .
pytest -q
Write-Host "[done] Setup complete."
