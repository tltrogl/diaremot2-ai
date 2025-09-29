#!/usr/bin/env bash
set -euo pipefail
echo "[maint] format + lint"
ruff format .
ruff check --fix .

echo "[maint] tests"
pytest -q

echo "[maint] build"
python -m build

echo "[maint] diagnostics"
python - <<'PY'
import json, shutil, sys
print(json.dumps({
  "python": sys.version.split()[0],
  "ffmpeg": bool(shutil.which("ffmpeg")),
  "ruff": bool(shutil.which("ruff")),
  "pytest": bool(shutil.which("pytest")),
}, indent=2))
PY
