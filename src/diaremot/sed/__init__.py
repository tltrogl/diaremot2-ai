"""Sound event detection helpers (PANNs CNN14 + fallbacks)."""

from .sed_panns_onnx import run_sed  # re-export for convenience

__all__ = ["run_sed"]
