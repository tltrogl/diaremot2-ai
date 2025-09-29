# AGENTS.md — Quality, Automation & CI

## Goals
- Keep code formatted and linted
- Prevent doc drift and broken installs
- One-command maintenance flows (Windows/PS and POSIX)

## Tools
- **Ruff** for lint + format
- **PyTest** for tests
- **FFmpeg** presence required

## Pre-commit (recommended)
```bash
pip install pre-commit
pre-commit install
```
Hooks:
- ruff (lint + format)
- end-of-file-fixer, trailing-whitespace

## CI (suggested)
- OS: ubuntu-latest, windows-latest
- Python: 3.11
- Steps: setup-python, cache pip, `pip install -r requirements.txt -e .`,
  then `ruff format --check . && ruff check . && pytest -q`

## Release
- Bump version when ready in pyproject
- `python -m build`, then publish
- Re-run import scan to update `requirements.txt` if modules changed
