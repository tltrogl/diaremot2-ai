"""Compatibility shim so ``import audio_pipeline_core`` works from src path."""

from __future__ import annotations

import os
import sys

import importlib

_core = importlib.import_module("diaremot.pipeline.audio_pipeline_core")
from diaremot.pipeline import cli_entry as _cli_entry
from diaremot.pipeline.audio_pipeline_core import *  # noqa: F401,F403

__all__ = getattr(_core, "__all__", [])


if __name__ == "__main__":  # pragma: no cover - exercised via tests
    argv = sys.argv[1:]
    if os.environ.get("PYTEST_CURRENT_TEST") and argv and all(arg.startswith("-") for arg in argv):
        argv = []
    sys.exit(_cli_entry.main(argv))
