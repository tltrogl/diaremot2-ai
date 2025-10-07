"""Shared utilities for DiaRemot BART maintenance scripts."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

_ENV_KEYS = (
    "BART_MODEL_DIR",
    "DIAREMOT_BART_MODEL_DIR",
    "DIAREMOT_MODEL_DIR",
)


def _candidate_dirs(explicit: Path | None) -> Iterable[Path]:
    if explicit:
        yield explicit

    for key in _ENV_KEYS:
        value = os.getenv(key)
        if not value:
            continue
        path = Path(value).expanduser()
        if key == "DIAREMOT_MODEL_DIR":
            path = path / "bart"
        yield path

    repo_root = Path(__file__).resolve().parents[1]
    yield repo_root / "models" / "bart"
    yield repo_root / "runs" / "models" / "bart"


def resolve_bart_dir(path: Path | None, *, must_exist: bool = False) -> Path:
    """Resolve the directory containing the BART checkpoint.

    Parameters
    ----------
    path:
        Optional explicit directory supplied by the caller.
    must_exist:
        If ``True`` the returned path has to exist, otherwise a
        ``FileNotFoundError`` is raised with the candidate list embedded.
    """

    explicit = path.expanduser() if path else None
    for candidate in _candidate_dirs(explicit):
        if candidate.exists():
            return candidate

    # Fall back to the first candidate even if it does not exist.
    fallback = next(iter(_candidate_dirs(explicit)))
    if must_exist:
        raise FileNotFoundError(f"Could not find a BART model directory near {fallback}")
    return fallback


def describe_bart_candidates(path: Path | None) -> str:
    """Return a human readable list of the directories we considered."""
    explicit = path.expanduser() if path else None
    parts = [f" - {candidate}" for candidate in _candidate_dirs(explicit)]
    return "\n".join(parts)
