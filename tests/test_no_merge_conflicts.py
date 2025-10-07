"""Regression guard to catch unresolved merge markers in source files."""

from __future__ import annotations

from pathlib import Path

import pytest


CONFLICT_MARKERS = ("<<<<<<<", "=======", ">>>>>>>")


def _iter_python_sources() -> list[Path]:
    roots = (Path("src"), Path("tests"))
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        files.extend(sorted(root.rglob("*.py")))
    return files


@pytest.mark.parametrize("path", _iter_python_sources(), ids=lambda p: str(p))
def test_python_files_do_not_contain_merge_conflict_markers(path: Path) -> None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    for raw_line in lines:
        stripped = raw_line.lstrip()
        if stripped.startswith(CONFLICT_MARKERS[0]):
            pytest.fail(f"Found merge conflict marker '<<<<<<<' in {path}")
        if stripped.startswith(CONFLICT_MARKERS[1]) and stripped.strip() == CONFLICT_MARKERS[1]:
            pytest.fail(f"Found merge conflict marker '=======' in {path}")
        if stripped.startswith(CONFLICT_MARKERS[2]):
            pytest.fail(f"Found merge conflict marker '>>>>>>>' in {path}")
