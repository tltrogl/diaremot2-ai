"""Compatibility shim for legacy imports.

This module re-exports the public API from ``diaremot.pipeline.audio_pipeline_core``
so that older entrypoints importing ``audio_pipeline_core`` from the repository
root continue to function. The real implementation lives under ``src/``.
"""

from __future__ import annotations

import diaremot.pipeline.audio_pipeline_core as _core
from diaremot.pipeline.audio_pipeline_core import *  # noqa: F401,F403

__all__ = getattr(_core, "__all__", [])

