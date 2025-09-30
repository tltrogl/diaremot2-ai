"""Optional background sound tagging via PANNs models."""

from __future__ import annotations

import csv
from contextlib import contextmanager
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np

from ..io.onnx_utils import create_onnx_session

if os.name == "nt":
    _WINDOWS_PANNS_DIRS = [
        Path("D:/diaremot/diaremot2-1/models/panns"),
        Path("D:/models/panns"),
    ]
    for _cand in _WINDOWS_PANNS_DIRS:
        if _cand.exists():
            DEFAULT_PANNS_MODEL_DIR = _cand
            break
    else:
        DEFAULT_PANNS_MODEL_DIR = _WINDOWS_PANNS_DIRS[0]
else:
    DEFAULT_PANNS_MODEL_DIR = Path("models/panns")

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import librosa  # type: ignore[import-not-found]

    _HAVE_LIBROSA = True
except Exception:  # pragma: no cover - env dependent
    _HAVE_LIBROSA = False

try:  # pragma: no cover - optional dependency
    import onnxruntime as ort  # type: ignore[import-not-found]

    _HAVE_ORT = True
except Exception:  # pragma: no cover - env dependent
    _HAVE_ORT = False
    ort = None  # type: ignore[assignment]

try:
    # Cheap check without importing (avoids triggering wget in panns_inference)
    import importlib.util as _ilu  # type: ignore[import-not-found]

    _HAVE_PANNS = _ilu.find_spec("panns_inference") is not None
except Exception:
    _HAVE_PANNS = False

if TYPE_CHECKING:  # pragma: no cover - typing only
    from panns_inference import AudioTagging as _PannsAudioTagging
else:  # pragma: no cover - runtime fallback
    _PannsAudioTagging = Any

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
    """Configuration for optional sound event detection."""

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


class PANNSEventTagger:
    """Lightweight wrapper for PANNs AudioSet tagging on CPU.

    Prefers an ONNX Runtime model when available, falling back to the
    original `panns_inference` PyTorch implementation. The ``backend``
    parameter accepts ``auto``, ``onnx``, ``pytorch``, or ``none``.
    - Accepts 16 kHz mono audio; resamples to 32 kHz if librosa available or
      uses simple upsampling fallbacks.
    - Returns top-K labels with scores and a coarse noise score.
    """

    def __init__(self, cfg: Optional[SEDConfig] = None, backend: str = "auto"):
        self.cfg = cfg or SEDConfig()

        backend = (backend or "auto").strip().lower()
        allowed_backends = {"auto", "onnx", "pytorch", "none"}
        if backend not in allowed_backends:
            raise ValueError(
                f"backend must be one of {sorted(allowed_backends)}; received '{backend}'"
            )
        if backend == "auto":
            # Prefer ONNX Runtime when both backends are available
            if _HAVE_ORT:
                backend = "onnx"
            elif _HAVE_PANNS:
                backend = "pytorch"
            else:
                backend = "none"
        self.backend = backend
        self._tagger: Optional[_PannsAudioTagging] = None
        self._session: Optional["ort.InferenceSession"] = None
        self._labels: Optional[list[str]] = None
        self.available = backend != "none"
        self._ensure_model()
        if not self.available:
            logger.warning(
                "PANNs event tagging unavailable: neither ONNX nor PyTorch backend could be initialized"
            )

    def _ensure_model(self):
        if not self.available:
            return
        if self.backend == "onnx":
            if self._session is not None and self._labels is not None:
                return
            if not _HAVE_ORT:
                self.available = False
            else:
                # 1) Prefer explicit local directory if provided
                if self.cfg.model_dir:
                    mp = self.cfg.model_dir / "model.onnx"
                    lp = self.cfg.model_dir / "class_labels_indices.csv"
                    if mp.exists() and lp.exists():
                        try:
                            self._session = create_onnx_session(mp)
                            with lp.open() as f:
                                reader = csv.DictReader(f)
                                self._labels = [
                                    str(row.get("display_name", "")) for row in reader
                                ]
                            return
                        except Exception as exc:
                            logger.info("Failed loading local ONNX model: %s", exc)
                            self._session = None
                            self._labels = None
                # 2) Search common env roots for HF cache-based files
                if self._session is None or not self._labels:
                    env_roots = [
                        os.getenv("DIAREMOT_PANNS_DIR"),
                        os.getenv("DIAREMOT_HF_HUB_DIR"),
                        os.getenv("HF_MODELS_HUB"),  # user-provided custom hub path
                        os.getenv("HF_HOME"),
                        os.getenv("HUGGINGFACE_HUB_CACHE"),
                    ]
                    for root in (Path(p) for p in env_roots if p):
                        try:
                            if not root.exists():
                                continue
                            # Prefer the specific repo path if present
                            repo_dir = root / "models--qiuqiangkong--panns-tagging-onnx"
                            candidates = []
                            if repo_dir.exists():
                                for mp in repo_dir.rglob("model.onnx"):
                                    lp = mp.parent / "class_labels_indices.csv"
                                    if lp.exists():
                                        candidates.append((mp, lp))
                            else:
                                # Generic fallback: search under root
                                for mp in root.rglob("model.onnx"):
                                    lp = mp.parent / "class_labels_indices.csv"
                                    if lp.exists():
                                        candidates.append((mp, lp))
                            if candidates:
                                # Pick the first valid pair
                                mp, lp = candidates[0]
                                self._session = create_onnx_session(mp)
                                with lp.open() as f:
                                    reader = csv.DictReader(f)
                                    self._labels = [
                                        row.get("display_name", "") for row in reader
                                    ]
                                return
                        except Exception as exc:
                            logger.info(
                                "Failed loading ONNX from env root %s: %s", root, exc
                            )
                if self._session is None or not self._labels:
                    logger.info(
                        "PANNs ONNX assets not found locally; skipping remote download fallback"
                    )
                    self._session = None
                    self._labels = None
            if self._session is None or not self._labels:
                if _HAVE_PANNS:
                    self.backend = "pytorch"
                    self.available = True
                    self._ensure_model()
                else:
                    self.available = False
        elif self.backend == "pytorch":
            if self._tagger is not None:
                return
            if not _HAVE_PANNS:
                if _HAVE_ORT:
                    self.backend = "onnx"
                    self.available = True
                    self._ensure_model()
                else:
                    self.available = False
                return
            # Ensure labels exist in HOME/panns_data to avoid wget in panns_inference
            try:
                home_panns = Path.home() / "panns_data"
                home_panns.mkdir(parents=True, exist_ok=True)
                src_labels = None
                if self.cfg.model_dir:
                    cand = self.cfg.model_dir / "class_labels_indices.csv"
                    if cand.exists():
                        src_labels = cand
                if (
                    src_labels
                    and not (home_panns / "class_labels_indices.csv").exists()
                ):
                    (home_panns / "class_labels_indices.csv").write_bytes(
                        src_labels.read_bytes()
                    )
            except Exception:
                pass

            # Resolve checkpoint file path if available
            ckpt_path: Optional[Path] = None
            if self.cfg.model_dir and Path(self.cfg.model_dir).exists():
                for name in [
                    "Cnn14_mAP=0.431.pth",
                    "Cnn14_mAP%3D0.431.pth",
                    "Cnn14_DecisionLevelMax.pth",
                ]:
                    cand = Path(self.cfg.model_dir) / name
                    if cand.exists():
                        ckpt_path = cand
                        break
                if ckpt_path is None:
                    # Any .pth in the directory
                    for cand in Path(self.cfg.model_dir).glob("*.pth"):
                        ckpt_path = cand
                        break

            try:
                # Lazy import after labels are in place to avoid wget
                from panns_inference import AudioTagging as _AT  # type: ignore

                # Suppress verbose prints from panns_inference (e.g., checkpoint path)
                @contextmanager
                def _suppress_stdout_stderr():
                    _old_out, _old_err = sys.stdout, sys.stderr
                    try:
                        with open(os.devnull, "w") as devnull:
                            sys.stdout = devnull
                            sys.stderr = devnull
                            yield
                    finally:
                        sys.stdout = _old_out
                        sys.stderr = _old_err

                with _suppress_stdout_stderr():
                    self._tagger = _AT(
                        checkpoint_path=str(ckpt_path) if ckpt_path else None,
                        device="cpu",
                    )
            except Exception as exc:
                logger.info("Failed initializing PyTorch backend: %s", exc)
                self._tagger = None
                if _HAVE_ORT:
                    self.backend = "onnx"
                    self.available = True
                    self._ensure_model()
                else:
                    self.available = False
        else:
            self.available = False

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

        if self.backend == "onnx":
            if self._session is None or not self._labels:
                return None
            try:
                inp = self._session.get_inputs()[0].name
                clip = self._session.run(None, {inp: y[np.newaxis, :]})[0][0]
            except Exception:
                return None
            map_labels = self._labels
        else:
            if self._tagger is None:
                return None
            try:
                # panns_inference expects shape [B, T] at 32 kHz and returns
                # (clipwise_output[B,527], embedding[B,2048]) as numpy arrays
                cw, _emb = self._tagger.inference(y[np.newaxis, :])  # type: ignore
                clip = np.asarray(cw[0], dtype=np.float32)
            except Exception:
                return None
            # Prefer labels from the tagger; fallback to imported default
            map_labels = list(getattr(self._tagger, "labels", labels)) or []  # type: ignore[arg-type]

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
