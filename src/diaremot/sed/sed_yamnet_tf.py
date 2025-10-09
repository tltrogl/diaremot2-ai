"""TensorFlow YAMNet fallback for environments where CNN14 is unavailable."""

from __future__ import annotations


def run_sed(
    wav_path: str,
    *,
    sr_target: int = 16000,
    frame_sec: float = 1.0,
    hop_sec: float = 0.5,
    enter_thresh: float = 0.50,
    exit_thresh: float = 0.35,
    min_dur: float = 0.30,
    merge_gap: float = 0.20,
    topk: int = 3,
) -> list[dict[str, float]]:
    """Placeholder fallback for the TensorFlow YAMNet backend.

    This repository ships without TensorFlow by default to keep the CPU footprint
    light. The CLI exposes ``--backend yamnet`` for parity with legacy builds, but
    using it without installing TensorFlow/YAMNet will raise this error.
    """

    raise RuntimeError(
        "YAMNet backend is not installed. Install TensorFlow + YAMNet or use the "
        "default PANNs backend."
    )
