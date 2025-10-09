param([string]$PythonExe = "")

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
function Have($name) { Get-Command $name -ErrorAction SilentlyContinue | Out-Null }

# 0) Pick python.exe — param > py -0p > common locations
if (-not $PythonExe) {
  $candidates = @()
  if (Have py) {
    try {
      $list = & py -0p 2>$null
      if ($list) { $candidates += ($list -split "`r?`n") }
    } catch {}
  }
  $candidates += @(
    "D:\Program Files\Python311\python.exe",
    "C:\Program Files\Python311\python.exe",
    "$env:LocalAppData\Programs\Python\Python311\python.exe"
  )
  $PythonExe = ($candidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1)
}
if (-not $PythonExe -or -not (Test-Path $PythonExe)) {
  throw "No usable python.exe found. Pass -PythonExe or install: winget install -e --id Python.Python.3.11"
}
"`nUsing interpreter: $PythonExe"
& $PythonExe -V

# 1) Create venv if missing
if (-not (Test-Path .\.ai)) { & $PythonExe -m venv .ai }

# 2) Use venv tools explicitly (no PATH reliance)
$VenvPy  = (Resolve-Path .\.ai\Scripts\python.exe).Path
$VenvPip = (Resolve-Path .\.ai\Scripts\pip.exe).Path
# Ensure uv present and prefer it over pip
$VenvDir = Split-Path -Parent $VenvPy
$UvExe   = Join-Path $VenvDir "uv.exe"
# Install/upgrade uv quietly (no-op if already current)
& $PKG install -U uv
if (Test-Path $UvExe) { $PKG = "$UvExe pip" } else { $PKG = $VenvPip }
"Venv python: $VenvPy"
"Venv pip:    $VenvPip"

# 3) Torch CPU first, then the rest, then editable package
& $PKG install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1+cpu
& $PKG install -r .\requirements.txt
& $PKG install -e .

# 4) Smoke check
$code = @"
import importlib.util
mods = ["torch","ctranslate2","onnxruntime","librosa","transformers","sklearn","numpy","pandas"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
assert not missing, f"Missing modules: {missing}"
import torch
print("Torch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
print("Env OK")
"@
& $VenvPy -c $code


# --- Auto-activate venv for this session ---
try {
   = Join-Path (Split-Path -Parent D:\diaremot\diaremot2-ai\.ai\Scripts\python.exe) 'Activate.ps1'
  if (Test-Path ) {
    . 
    'Venv activated for this session.' | Write-Host
  } else {
    'Activate script not found at ' +  | Write-Warning
  }
} catch {
  'Auto-activation failed: ' +  | Write-Warning
}"Setup complete."
