"""Shared helpers for configuring cache-related environment variables."""

from __future__ import annotations

from pathlib import Path

from . import runtime_env

__all__ = ["configure_local_cache_env"]


_CACHE_ROOT: Path | None = None


def configure_local_cache_env() -> Path:
    """Delegate to :mod:`runtime_env`'s robust cache configuration helper."""

    global _CACHE_ROOT
    cache_root = runtime_env.configure_local_cache_env()
    _CACHE_ROOT = cache_root
    return cache_root


_CACHE_ROOT = configure_local_cache_env()
