param(
  [switch]$PersistEnv = $false,
  [switch]$AutoDownloadModels = $false,
  [string]$ModelsUrl,
  [string]$ModelsZip,
  [string]$ModelsSha256,
  [string]$Python
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent $PSCommandPath
Push-Location $RepoRoot
Write-Host "==> Repo root: $RepoRoot"

function Set-Env([string]$Name,[string]$Value) {
  $env:$Name = $Value
  if ($PersistEnv) { [Environment]::SetEnvironmentVariable($Name, $Value, 'User') }
}

function Get-SHA256([string]$Path) {
  try { (Get-FileHash -Algorithm SHA256 -Path $Path).Hash.ToLower() } catch { '' }
}

function Expand-Zip([string]$Zip,[string]$Dest) {
  if (Get-Command Expand-Archive -ErrorAction SilentlyContinue) {
    Expand-Archive -Force -Path $Zip -DestinationPath $Dest
  } else {
    Add-Type -AssemblyName System.IO.Compression.FileSystem | Out-Null
    [System.IO.Compression.ZipFile]::ExtractToDirectory($Zip, $Dest, $true)
  }
}

# ---- Python / venv ----
function Resolve-Python() {
  if ($Python) { return $Python }
  if (Get-Command py -ErrorAction SilentlyContinue) { return 'py' }
  if (Get-Command python3 -ErrorAction SilentlyContinue) { return 'python3' }
  return 'python'
}

$py = Resolve-Python
Write-Host "==> Using Python launcher: $py"

if (-not (Test-Path .venv)) {
  Write-Host "==> Creating virtual environment (.venv)"
  if ($py -eq 'py') { & $py -3 -m venv .venv } else { & $py -m venv .venv }
}

. .\.venv\Scripts\Activate.ps1
Write-Host "==> Python: $(python -V)"

Write-Host "==> Upgrading pip/setuptools/wheel"
python -m pip install --upgrade pip setuptools wheel

Write-Host "==> Installing requirements.txt"
python -m pip install -r requirements.txt

# ---- Local caches + required env vars ----
Write-Host "==> Preparing local caches and env vars"
$cacheRoot = if ($env:CACHE_ROOT) { $env:CACHE_ROOT } else { Join-Path $RepoRoot '.cache' }
New-Item -Force -ItemType Directory -Path $cacheRoot,$(Join-Path $cacheRoot 'hf'),$(Join-Path $cacheRoot 'torch'),$(Join-Path $cacheRoot 'transformers') | Out-Null

Set-Env HF_HOME                 ($env:HF_HOME                 ? $env:HF_HOME                 : (Join-Path $cacheRoot 'hf'))
Set-Env HUGGINGFACE_HUB_CACHE   ($env:HUGGINGFACE_HUB_CACHE   ? $env:HUGGINGFACE_HUB_CACHE   : (Join-Path $cacheRoot 'hf'))
Set-Env TRANSFORMERS_CACHE      ($env:TRANSFORMERS_CACHE      ? $env:TRANSFORMERS_CACHE      : (Join-Path $cacheRoot 'transformers'))
Set-Env TORCH_HOME              ($env:TORCH_HOME              ? $env:TORCH_HOME              : (Join-Path $cacheRoot 'torch'))
Set-Env XDG_CACHE_HOME          ($env:XDG_CACHE_HOME          ? $env:XDG_CACHE_HOME          : $cacheRoot)
Set-Env PYTHONPATH              ("$RepoRoot/src" + ($(if ($env:PYTHONPATH) { ';' + $env:PYTHONPATH } else { '' })))
Set-Env CUDA_VISIBLE_DEVICES    ''
Set-Env TORCH_DEVICE            'cpu'

$cpu = [Environment]::ProcessorCount
$threads = [Math]::Min(4, [Math]::Max(1, $cpu))
Set-Env OMP_NUM_THREADS         ($env:OMP_NUM_THREADS         ? $env:OMP_NUM_THREADS         : "$threads")
Set-Env MKL_NUM_THREADS         ($env:MKL_NUM_THREADS         ? $env:MKL_NUM_THREADS         : "$threads")
Set-Env NUMEXPR_MAX_THREADS     ($env:NUMEXPR_MAX_THREADS     ? $env:NUMEXPR_MAX_THREADS     : "$threads")
Set-Env TOKENIZERS_PARALLELISM  ($env:TOKENIZERS_PARALLELISM  ? $env:TOKENIZERS_PARALLELISM  : 'false')

Set-Env DIAREMOT_MODEL_DIR      ($env:DIAREMOT_MODEL_DIR      ? $env:DIAREMOT_MODEL_DIR      : (Join-Path $RepoRoot 'models'))
New-Item -Force -ItemType Directory -Path $env:DIAREMOT_MODEL_DIR | Out-Null

# ---- Models staging (gated) ----
$required = @(
  'panns_cnn14.onnx', 'audioset_labels.csv', 'silero_vad.onnx', 'ecapa_tdnn.onnx',
  'ser_8class.onnx', 'vad_model.onnx', 'roberta-base-go_emotions.onnx', 'bart-large-mnli.onnx'
)
$needModels = $false
foreach ($f in $required) { if (-not (Test-Path (Join-Path $env:DIAREMOT_MODEL_DIR $f))) { $needModels = $true; break } }

if ($needModels) {
  Write-Host "==> Models not fully present under $env:DIAREMOT_MODEL_DIR"
  $zipPath = if ($ModelsZip) { $ModelsZip } elseif ($env:DIAREMOT_MODELS_ZIP) { $env:DIAREMOT_MODELS_ZIP } else { Join-Path $RepoRoot 'models.zip' }
  $dl = ($AutoDownloadModels -or ($env:DIAREMOT_AUTO_DOWNLOAD -eq '1'))
  $url = if ($ModelsUrl) { $ModelsUrl } else { $env:DIAREMOT_MODELS_URL }
  $sha = if ($ModelsSha256) { $ModelsSha256 } else { $env:DIAREMOT_MODELS_SHA256 }

  if (Test-Path $zipPath) {
    if ($sha) {
      $got = Get-SHA256 $zipPath
      if ($got -ne $sha) { throw "models.zip SHA256 mismatch: got $got expected $sha" }
    }
    Write-Host "==> Unpacking models.zip into $env:DIAREMOT_MODEL_DIR"
    Expand-Zip -Zip $zipPath -Dest $env:DIAREMOT_MODEL_DIR
  } elseif ($dl -and $url) {
    $dest = Join-Path $cacheRoot 'models.zip'
    Write-Host "==> Downloading models.zip to $dest"
    Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $dest
    if ($sha) {
      $got = Get-SHA256 $dest; if ($got -ne $sha) { throw "models.zip SHA256 mismatch: got $got expected $sha" }
    }
    Expand-Zip -Zip $dest -Dest $env:DIAREMOT_MODEL_DIR
  } else {
    Write-Warning "Models missing and no models.zip provided. Set DIAREMOT_MODELS_ZIP or pass -ModelsZip."
    Write-Warning "Optionally, pass -AutoDownloadModels -ModelsUrl <URL> [-ModelsSha256 <sha>] to enable download."
  }

  $nested = Join-Path $env:DIAREMOT_MODEL_DIR 'models'
  if (Test-Path $nested) {
    Write-Host "==> Normalizing staged models layout"
    Copy-Item -Recurse -Force (Join-Path $nested '*') $env:DIAREMOT_MODEL_DIR
    Remove-Item -Recurse -Force $nested
  }
}

# ---- Optional sample (if ffmpeg available) ----
if (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
  if (-not (Test-Path 'data/sample.wav')) {
    Write-Host "==> Generating 10s 440Hz sample (data/sample.wav)"
    New-Item -Force -ItemType Directory -Path 'data' | Out-Null
    ffmpeg -hide_banner -loglevel error -f lavfi -i "sine=frequency=440:duration=10" -ar 16000 -ac 1 'data/sample.wav' -y | Out-Null
  }
}

# ---- Import and dependency checks ----
Write-Host "==> Verifying core imports"
$code = @'
import importlib, sys
mods = [
  "diaremot.pipeline.audio_pipeline_core",
  "diaremot.pipeline.audio_preprocessing",
  "diaremot.pipeline.speaker_diarization",
  "diaremot.pipeline.transcription_module",
  "diaremot.affect.emotion_analyzer",
  "diaremot.affect.paralinguistics",
  "diaremot.io.onnx_utils",
]
bad=[]
for m in mods:
    try:
        importlib.import_module(m)
        print("OK  import", m)
    except Exception as e:
        bad.append((m, e))
if bad:
    print("\nFAILED imports:")
    for m,e in bad:
        print(" -", m, ":", e)
    sys.exit(2)
'@
$tmp = New-TemporaryFile
Set-Content -Path $tmp -Value $code -Encoding UTF8
try { python $tmp } catch { Write-Warning $_ }
Remove-Item $tmp -Force

Write-Host "==> Pipeline dependency check (--verify_deps)"
try { python -m diaremot.pipeline.audio_pipeline_core --verify_deps --strict_dependency_versions } catch { Write-Warning $_ }

Write-Host @"
==> Setup complete.
Models (if provided) staged in $env:DIAREMOT_MODEL_DIR.
Run a smoke test in this shell:
  python -m diaremot.cli run --input data/sample.wav --outdir .\outputs --asr-compute-type float32
"@

Pop-Location
