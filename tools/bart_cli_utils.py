"""Shared utilities for DiaRemot BART maintenance scripts."""
<<<<<<< HEAD

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
=======
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional
>>>>>>> 7b611bc33ae14a4cd702cb5f9355008663373325

_ENV_KEYS = (
    "BART_MODEL_DIR",
    "DIAREMOT_BART_MODEL_DIR",
    "DIAREMOT_MODEL_DIR",
)


<<<<<<< HEAD
def _candidate_dirs(explicit: Path | None) -> Iterable[Path]:
=======
def _candidate_dirs(explicit: Optional[Path]) -> Iterable[Path]:
>>>>>>> 7b611bc33ae14a4cd702cb5f9355008663373325
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


<<<<<<< HEAD
def resolve_bart_dir(path: Path | None, *, must_exist: bool = False) -> Path:
=======
def resolve_bart_dir(path: Optional[Path], *, must_exist: bool = False) -> Path:
>>>>>>> 7b611bc33ae14a4cd702cb5f9355008663373325
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


<<<<<<< HEAD
def describe_bart_candidates(path: Path | None) -> str:
=======
def describe_bart_candidates(path: Optional[Path]) -> str:
>>>>>>> 7b611bc33ae14a4cd702cb5f9355008663373325
    """Return a human readable list of the directories we considered."""
    explicit = path.expanduser() if path else None
    parts = [f" - {candidate}" for candidate in _candidate_dirs(explicit)]
    return "\n".join(parts)
