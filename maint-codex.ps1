Write-Host "[maint] format + lint"
ruff format .
ruff check --fix .

Write-Host "[maint] tests"
pytest -q

Write-Host "[maint] build"
python -m build

Write-Host "[maint] diagnostics"
$diag = [ordered]@{
  python = (python -c "import sys; print(sys.version.split()[0])")
  ffmpeg = [bool](Get-Command ffmpeg -ErrorAction SilentlyContinue)
  ruff   = [bool](Get-Command ruff -ErrorAction SilentlyContinue)
  pytest = [bool](Get-Command pytest -ErrorAction SilentlyContinue)
} | ConvertTo-Json -Depth 3
$diag
