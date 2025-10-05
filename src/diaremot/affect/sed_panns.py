"""Optional background sound tagging via PANNs models."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Literal, Optional

import numpy as np

from ..io.onnx_utils import create_onnx_session
from ..pipeline.runtime_env import DEFAULT_MODELS_ROOT, iter_model_roots

DEFAULT_PANNS_MODEL_DIR = DEFAULT_MODELS_ROOT / "panns"

logger = logging.getLogger(__name__)

try:  # pragma: no cover - auxiliary dependency
    import librosa  # type: ignore[import-not-found]

    _HAVE_LIBROSA = True
except Exception:  # pragma: no cover - env dependent
    _HAVE_LIBROSA = False

try:  # pragma: no cover - dependency detection
    import importlib.util as _ort_util  # type: ignore[import-not-found]

    _HAVE_ORT = _ort_util.find_spec("onnxruntime") is not None
except Exception:  # pragma: no cover - env dependent
    _HAVE_ORT = False

if TYPE_CHECKING:  # pragma: no cover - typing only
    from onnxruntime import InferenceSession as _OrtSession
else:  # pragma: no cover - runtime fallback
    _OrtSession = Any

labels: list[str] = []


_NOISE_KEYWORDS = (
    "music",
    "noise",
    "typing",
    "keyboard",
    "applause",
    "traffic",
    "engine",
    "vehicle",
    "crowd",
    "siren",
    "wind",
    "rain",
    "tv",
)


EvalStrategy = Literal["head", "uniform"]


@dataclass
class SEDConfig:
    """Configuration for the sound event detection stage."""

    top_k: int = 3
    run_on_suspect_only: bool = True
    min_duration_sec: float = 0.25
    # Prefer a local model directory by default for offline reliability
    model_dir: Optional[Path] = DEFAULT_PANNS_MODEL_DIR
    # By default, analyze the full audio instead of truncating to a head slice.
    max_eval_sec: Optional[float] = None
    eval_strategy: EvalStrategy = "head"

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if self.min_duration_sec < 0:
            raise ValueError("min_duration_sec must be >= 0")
        if self.max_eval_sec is not None and self.max_eval_sec <= 0:
            raise ValueError("max_eval_sec must be > 0 when provided")
        if self.eval_strategy not in {"head", "uniform"}:
            raise ValueError("eval_strategy must be 'head' or 'uniform'")
        if self.model_dir is not None:
            self.model_dir = Path(self.model_dir)


def _load_label_file(label_path: Path) -> list[str]:
    """Extract label names from a PANNs CSV manifest."""

    with label_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        labels_from_csv = []
        for row in reader:
            value = row.get("display_name") or row.get("name") or row.get("mid") or ""
            labels_from_csv.append(str(value))
    return labels_from_csv


def _iter_env_roots(env_vars: Iterable[str]) -> Iterator[Path]:
    for env_var in env_vars:
        value = os.getenv(env_var)
        if not value:
            continue
        root = Path(value).expanduser()
        if root.exists():
            yield root


class PANNSEventTagger:
    """Lightweight wrapper for PANNs AudioSet tagging on CPU using ONNX."""

    def __init__(self, cfg: Optional[SEDConfig] = None, backend: str = "auto"):
        self.cfg = cfg or SEDConfig()

        backend = (backend or "auto").strip().lower()
        allowed_backends = {"auto", "onnx", "none"}
        if backend not in allowed_backends:
            raise ValueError(
                f"backend must be one of {sorted(allowed_backends)}; received '{backend}'"
            )
        if backend == "auto":
            # Prefer ONNX Runtime when both backends are available
            backend = "onnx" if _HAVE_ORT else "none"
        self.backend = backend
        self._session: Optional[_OrtSession] = None
        self._labels: Optional[list[str]] = None
        self.available = backend != "none"
        self._ensure_model()
        if not self.available:
            logger.warning(
                "PANNs event tagging unavailable: ONNX Runtime models missing"
            )

    def _ensure_model(self) -> None:
        if not self.available:
            return

        if not self.available:
            return

        if self.backend != "onnx":
            self.available = False
            self.backend = "none"
            return

        if self._init_onnx_backend():
            self.available = True
            return

        self.available = False
        self.backend = "none"

    def _iter_onnx_candidates(self) -> Iterator[tuple[Path, Path]]:
        seen: set[Path] = set()
        model_dir = self.cfg.model_dir
        if model_dir:
            candidate = model_dir / "model.onnx"
            labels_path = model_dir / "class_labels_indices.csv"
            if candidate.exists() and labels_path.exists():
                seen.add(candidate)
                yield candidate, labels_path

        for root in iter_model_roots():
            base = root / "panns"
            candidate = base / "model.onnx"
            labels_path = base / "class_labels_indices.csv"
            if candidate.exists() and labels_path.exists() and candidate not in seen:
                seen.add(candidate)
                yield candidate, labels_path

        env_roots = _iter_env_roots(
            (
                "DIAREMOT_PANNS_DIR",
                "DIAREMOT_HF_HUB_DIR",
                "HF_MODELS_HUB",
                "HF_HOME",
                "HUGGINGFACE_HUB_CACHE",
            )
        )
        for root in env_roots:
            repo_dir = root / "models--qiuqiangkong--panns-tagging-onnx"
            search_roots = [repo_dir] if repo_dir.exists() else [root]
            for search_root in search_roots:
                try:
                    for model_path in search_root.rglob("model.onnx"):
                        if model_path in seen:
                            continue
                        labels_path = model_path.parent / "class_labels_indices.csv"
                        if labels_path.exists():
                            seen.add(model_path)
                            yield model_path, labels_path
                except Exception as exc:
                    logger.info("Failed loading ONNX from env root %s: %s", root, exc)

    def _init_onnx_backend(self) -> bool:
        if not _HAVE_ORT:
            return False
        if self._session is not None and self._labels:
            return True

        for model_path, label_path in self._iter_onnx_candidates():
            try:
                session = create_onnx_session(model_path)
                labels_loaded = _load_label_file(label_path)
            except Exception as exc:
                logger.info("Failed loading ONNX model from %s: %s", model_path, exc)
                continue

            if not labels_loaded:
                logger.info("Skipping ONNX candidate without labels: %s", model_path)
                continue

            self._session = session
            self._labels = labels_loaded
            return True

        logger.info(
            "PANNs ONNX assets not found locally; skipping remote download fallback"
        )
        return False

    def _resample_to_32k(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        if sr == 32000:
            return audio.astype(np.float32, copy=False), sr
        if _HAVE_LIBROSA:
            try:
                y = librosa.resample(
                    audio.astype(np.float32), orig_sr=sr, target_sr=32000
                )
                return y.astype(np.float32), 32000
            except Exception:
                pass
        if sr == 16000:
            return np.repeat(audio.astype(np.float32), 2), 32000
        ratio = 32000 / float(sr)
        new_len = int(round(len(audio) * ratio))
        if new_len <= 1:
            return audio.astype(np.float32), sr
        x_old = np.linspace(0, len(audio) - 1, len(audio), dtype=np.float32)
        x_new = np.linspace(0, len(audio) - 1, new_len, dtype=np.float32)
        y = np.interp(x_new, x_old, audio).astype(np.float32)
        return y, 32000

    def tag(self, audio_16k_mono: np.ndarray, sr: int) -> Optional[dict[str, Any]]:
        if not self.available:
            return None
        if audio_16k_mono is None or audio_16k_mono.size == 0:
            return None
        if (len(audio_16k_mono) / max(1, sr)) < self.cfg.min_duration_sec:
            return None
        self._ensure_model()
        if not self.available:
            return None

        # Subsample/limit audio duration to keep SED fast on long files
        y_in = audio_16k_mono
        total_sec = len(y_in) / float(sr)
        if self.cfg.max_eval_sec and total_sec > self.cfg.max_eval_sec:
            max_samples = int(self.cfg.max_eval_sec * sr)
            if self.cfg.eval_strategy == "uniform" and (self.cfg.max_eval_sec or 0) > 0:
                # Take N uniform slices totaling max_eval_sec
                slices = 5
                seg_len = max_samples // slices
                step = len(y_in) // slices
                parts = []
                for i in range(slices):
                    start = i * step
                    end = min(start + seg_len, len(y_in))
                    parts.append(y_in[start:end])
                y_in = np.concatenate(parts) if parts else y_in[:max_samples]
            else:
                # Default: head slice
                y_in = y_in[:max_samples]

        y, sr32 = self._resample_to_32k(y_in, sr)

        if self.backend != "onnx" or self._session is None or not self._labels:
            return None
        try:
            inp = self._session.get_inputs()[0].name
            clip = self._session.run(None, {inp: y[np.newaxis, :]})[0][0]
        except Exception:
            return None
        map_labels = self._labels

        if clip.size == 0:
            return None
        # If labels are missing or mismatched, synthesize generic labels
        if not map_labels or len(map_labels) != clip.size:
            try:
                n = int(clip.size)
            except Exception:
                n = 0
            map_labels = [f"class_{i}" for i in range(n)]
        top_idx = clip.argsort()[-self.cfg.top_k :][::-1]
        top = [{"label": str(map_labels[i]), "score": float(clip[i])} for i in top_idx]

        noise_score = 0.0
        for i, s in enumerate(clip):
            lab = str(map_labels[i]).lower()
            if any(k in lab for k in _NOISE_KEYWORDS):
                noise_score += float(s)

        return {
            "top": top,
            "dominant_label": top[0]["label"] if top else None,
            "noise_score": float(noise_score),
        }
