# emotion_analyzer.py - CPU-only, production-hardened
# Fixes & upgrades:
#   * Dynamic VAD index mapping (read id2label/label2id; no brittle ordering)
#   * Parallel inference for SER/Text/Intent (ThreadPoolExecutor) on CPU
#   * Unified .analyze(wav, sr, text) -> schema-stable dict (matches core expectations)
#   * SER confidence calibration via entropy + top-margin heuristics
#   * Input validation, robust fallbacks, consistent 8-class + 28-class outputs
#   * Cross-modal "affect_hint" uses valence/arousal proxies (soft, not brittle)

# Models (CPU):
#   - VAD (dimensional): audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim -> Valence/Arousal/Dominance
#   - SER (8-class audio): Dpngtm/wav2vec2-emotion-recognition
#   - Text emotions (28): SamLowe/roberta-base-go_emotions (multi-label)
#   - Intent (14+1): facebook/bart-large-mnli (zero-shot) over standardized labels

# Notes:
#   - All models are lazily loaded on first use to reduce memory footprint.
#   - If any model import/inference fails, we fall back to light, CPU-cheap heuristics.
#   - Output schema follows the pipeline spec used by audio_pipeline_core.AudioAnalysisPipelineV2

from __future__ import annotations

<<<<<<< HEAD
=======
from __future__ import annotations

import json
import json
>>>>>>> 7b611bc33ae14a4cd702cb5f9355008663373325
import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
<<<<<<< HEAD
=======
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
>>>>>>> 7b611bc33ae14a4cd702cb5f9355008663373325

# Optional heavy deps are imported lazily inside methods.
# librosa is cheap enough; used for fallbacks and basic features.
import librosa
import numpy as np

<<<<<<< HEAD
from ..io.onnx_utils import create_onnx_session, ensure_onnx_model
from ..utils.model_paths import iter_model_roots, iter_model_subpaths
from .intent_defaults import INTENT_LABELS_DEFAULT
=======
from .ser_dpngtm import SERDpngtm
from .intent_defaults import INTENT_LABELS_DEFAULT
from ..io.onnx_utils import create_onnx_session

# Preprocessing: strictly librosa/scipy/numpy
try:
    import librosa  # type: ignore
except ImportError:  # pragma: no cover - handled gracefully at runtime
    librosa = None  # type: ignore
>>>>>>> 7b611bc33ae14a4cd702cb5f9355008663373325

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setLevel(logging.INFO)
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

# If a local model directory exists (DIAREMOT_MODEL_DIR or the known default),
# prefer offline mode to ensure we load local ONNX exports instead of trying
# to fetch from Hugging Face hub. This prevents unintended network lookups and
# guarantees model resolution uses the local model root.
try:
    _model_root = os.environ.get("DIAREMOT_MODEL_DIR")
    if not _model_root:
        for candidate in iter_model_roots():
            if candidate.exists():
                _model_root = str(candidate)
                break
    if _model_root and Path(_model_root).expanduser().exists():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        logger.info("HF hub forced offline; using local model root: %s", _model_root)
except Exception:
    # Never raise during import; best-effort only
    pass

INTENT_HYPOTHESIS_TEMPLATE = "This example is about {}."


# --------------------------
# Label spaces / constants
# --------------------------

SER_LABELS_8 = [
    "angry",
    "happy",
    "sad",
    "neutral",
    "fearful",
    "surprised",
    "disgusted",
    "calm",
]

GOEMOTIONS_LABELS = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]

GOEMOTIONS_POSITIVE = {
    "admiration",
    "amusement",
    "approval",
    "caring",
    "desire",
    "excitement",
    "gratitude",
    "joy",
    "love",
    "optimism",
    "pride",
    "relief",
}
GOEMOTIONS_NEGATIVE = {
    "anger",
    "annoyance",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "fear",
    "grief",
    "nervousness",
    "remorse",
    "sadness",
}
GOEMOTIONS_AMBIGUOUS = {"confusion", "curiosity", "realization", "surprise"}

# --------------------------
# Helpers
# --------------------------

DEFAULT_INTENT_MODEL = "facebook/bart-large-mnli"


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    s = e.sum() or 1.0
    return (e / s).astype(np.float64)


<<<<<<< HEAD
def _entropy(probs: dict[str, float]) -> float:
    p = np.clip(np.array(list(probs.values()), dtype=np.float64), 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))
=======
def _normalize_backend(value: Optional[str]) -> str:
    if not value:
        return "auto"
    normalized = value.lower()
    if normalized not in {"auto", "onnx", "torch"}:
        return "auto"
    return normalized


def _intent_dir_has_assets(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    config = path / "config.json"
    if not config.exists():
        return False
    if (path / "model.onnx").exists() or (path / "model_uint8.onnx").exists():
        return True
    if (path / "pytorch_model.bin").exists():
        return True
    shard = list(path.glob("pytorch_model-*.bin"))
    index = path / "pytorch_model.bin.index.json"
    return bool(shard and index.exists())


def _intent_candidate_dirs(explicit: Optional[str]) -> Iterable[Path]:
    candidates: list[Path] = []

    def _add(candidate: Optional[str | Path]) -> None:
        if not candidate:
            return
        path = Path(candidate).expanduser()
        candidates.append(path)

    _add(explicit)
    _add(os.getenv("DIAREMOT_INTENT_MODEL_DIR"))

    model_root = os.getenv("DIAREMOT_MODEL_DIR")
    if model_root:
        root = Path(model_root).expanduser()
        _add(root)
        _add(root / "bart")
        _add(root / "bart-large-mnli")
        _add(root / "facebook" / "bart-large-mnli")
        _add(root / "bart" / "facebook" / "bart-large-mnli")

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        yield candidate


def _resolve_intent_model_dir(explicit: Optional[str]) -> Optional[str]:
    if explicit:
        path = Path(explicit).expanduser()
        if path.exists():
            return str(path)
    for candidate in _intent_candidate_dirs(explicit):
        if _intent_dir_has_assets(candidate):
            return str(candidate)
    return None


def _find_label_index(id2label: Dict[int, str], target: str) -> Optional[int]:
    target_lower = target.lower()
    for idx, label in id2label.items():
        if str(label).lower() == target_lower:
            return int(idx)
    return None


def _ort_session(path: str):
    try:
        import onnxruntime as ort  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(f"onnxruntime not available: {exc}") from exc

    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = min(4, os.cpu_count() or 1)
    sess_options.inter_op_num_threads = 1
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    # DiaRemot is CPU-only per AGENTS.md. Do not use GPU providers.
    providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(path, sess_options=sess_options, providers=providers)
>>>>>>> 7b611bc33ae14a4cd702cb5f9355008663373325


def _norm_entropy(probs: dict[str, float]) -> float:
    H = _entropy(probs)
    Hmax = math.log(len(probs)) if probs else 1.0
    return float(H / max(1e-9, Hmax))


def _top_margin(probs: dict[str, float]) -> float:
    vals = sorted(probs.values(), reverse=True)
    if len(vals) < 2:
        return 1.0
    return float(vals[0] - vals[1])


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    """Normalize a mapping of scores, guarding against bad inputs."""

    if not scores:
        return {}

    clean: dict[str, float] = {}
    for label, value in scores.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = 0.0
        if not math.isfinite(numeric) or numeric < 0.0:
            numeric = 0.0
        clean[label] = numeric

    total = sum(clean.values())
    if total <= 0.0:
        uniform = 1.0 / len(clean)
        return {label: uniform for label in clean}

    inv_total = 1.0 / total
    return {label: value * inv_total for label, value in clean.items()}


def _topk_distribution(dist: dict[str, float], k: int) -> list[dict[str, float]]:
    if k <= 0 or not dist:
        return []
    ordered = sorted(dist.items(), key=lambda kv: (-kv[1], kv[0]))
    return [{"label": label, "score": float(score)} for label, score in ordered[:k]]


def _ensure_16k_mono(audio: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    if audio is None or not isinstance(audio, np.ndarray) or audio.size == 0:
        return np.zeros(0, dtype=np.float32), 16000
    x = audio.astype(np.float32, copy=False)
    if sr != 16000:
        try:
            x = librosa.resample(x, orig_sr=sr, target_sr=16000)
            sr = 16000
        except Exception:
            pass
    if x.ndim > 1:
        x = np.mean(x, axis=0).astype(np.float32)
    return x, sr


def _trim_max_len(audio: np.ndarray, sr: int, max_sec: float = 30.0) -> np.ndarray:
    nmax = int(max_sec * sr)
    return audio[:nmax] if audio.size > nmax else audio


def _text_tokens(text: str) -> list[str]:
    if not text:
        return []
    out, buf = [], []
    for ch in text:
        if ch.isalnum() or ch in ["'", "-"]:
            buf.append(ch.lower())
        else:
            if buf:
                out.append("".join(buf))
                buf = []
    if buf:
        out.append("".join(buf))
    return out


# --------------------------
# Dataclass (internal use)
# --------------------------


@dataclass
class EmotionAnalysisResult:
    valence: float
    arousal: float
    dominance: float
    ser_top: str
    ser_probs: dict[str, float]
    text_full: dict[str, float]
    text_top5: list[dict[str, float]]
    intent_top: str
    intent_top3: list[dict[str, float]]
    flags: dict[str, bool]
    affect_hint: str


# --------------------------
# Analyzer
# --------------------------


_HF_MODEL_WEIGHT_PATTERNS = (
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
    "pytorch_model-*.bin",
    "model.safetensors",
    "model-*.safetensors",
    "tf_model.h5",
    "model.ckpt.index",
    "flax_model.msgpack",
)


# Back-compat alias expected by orchestrator with extended intent controls
class EmotionIntentAnalyzer(EmotionAnalyzer):
    def __init__(
        self,
        *,
        text_emotion_model: str = "SamLowe/roberta-base-go_emotions",
        intent_labels: Sequence[str] | None = None,
        affect_backend: Optional[str] = None,
        affect_text_model_dir: Optional[str] = None,
        affect_intent_model_dir: Optional[str] = None,
        analyzer_threads: Optional[int] = None,
        disable_downloads: Optional[bool] = None,
        model_dir: Optional[str] = None,
    ) -> None:
        super().__init__(model_dir=model_dir, disable_downloads=disable_downloads)

        self.text_emotion_model = text_emotion_model
        labels = intent_labels or INTENT_LABELS_DEFAULT
        self.intent_labels: List[str] = [str(label) for label in labels]
        self.affect_backend = _normalize_backend(affect_backend)
        self.analyzer_threads = analyzer_threads

        self.affect_text_model_dir = affect_text_model_dir
        if affect_text_model_dir:
            text_dir = os.fspath(affect_text_model_dir)
            self.path_text_onnx = os.path.join(text_dir, "roberta-base-go_emotions.onnx")

        self.affect_intent_model_dir = _resolve_intent_model_dir(affect_intent_model_dir)

        self._intent_session: Optional[object] = None
        self._intent_tokenizer: Optional[Callable[..., Dict[str, np.ndarray]]] = None
        self._intent_config: Optional[object] = None
        self._intent_pipeline: Optional[Callable[..., object]] = None
        self._intent_entail_idx: Optional[int] = None
        self._intent_contra_idx: Optional[int] = None
        self._intent_hypothesis_template: str = "This example is {}."

    # ---- Intent helpers ----
    def _lazy_intent(self) -> None:
        backend = self.affect_backend
        if backend == "onnx":
            self._ensure_intent_onnx(strict=False)
        elif backend == "torch":
            self._ensure_intent_pipeline()
        else:
            if not self._ensure_intent_onnx(strict=False):
                self._ensure_intent_pipeline()

    def _select_onnx_model(self, model_dir: Path) -> Optional[Path]:
        for name in ("model_uint8.onnx", "model.onnx"):
            candidate = model_dir / name
            if candidate.exists():
                return candidate
        remaining = list(model_dir.glob("*.onnx"))
        return remaining[0] if remaining else None

    def _ensure_intent_onnx(self, *, strict: bool) -> bool:
        if self._intent_session is not None and self._intent_tokenizer is not None:
            return True

        model_dir_str = self.affect_intent_model_dir
        if not model_dir_str:
            if strict:
                logger.warning("Intent ONNX backend requested but no model directory is configured")
            return False

        model_dir = Path(model_dir_str)
        model_path = self._select_onnx_model(model_dir)
        if model_path is None:
            if strict:
                logger.warning("Intent ONNX backend missing model.onnx in %s", model_dir)
            return False

        threads = self.analyzer_threads or 1
        try:
            self._intent_session = create_onnx_session(model_path, threads=threads)
        except Exception as exc:  # pragma: no cover - runtime dependent
            logger.warning("Intent ONNX session unavailable: %s", exc)
            self._intent_session = None
            return False

        try:
            from transformers import AutoConfig, AutoTokenizer  # type: ignore
        except ModuleNotFoundError as exc:
            logger.warning("transformers unavailable for intent tokenizer: %s", exc)
            self._intent_session = None
            return False

        try:
            self._intent_tokenizer = AutoTokenizer.from_pretrained(model_dir_str)
            self._intent_config = AutoConfig.from_pretrained(model_dir_str)
        except Exception as exc:  # noqa: BLE001 - dependent on HF cache state
            logger.warning("Failed to load intent tokenizer/config: %s", exc)
            self._intent_session = None
            self._intent_tokenizer = None
            self._intent_config = None
            return False

        id2label_raw = getattr(self._intent_config, "id2label", {})
        if isinstance(id2label_raw, dict):
            id2label = {int(k): str(v) for k, v in id2label_raw.items()}
        else:
            id2label = {int(idx): str(label) for idx, label in enumerate(id2label_raw)}
        self._intent_entail_idx = _find_label_index(id2label, "entailment")
        self._intent_contra_idx = _find_label_index(id2label, "contradiction")

        template = getattr(self._intent_config, "hypothesis_template", None)
        if isinstance(template, str) and "{}" in template:
            self._intent_hypothesis_template = template
        else:
            self._intent_hypothesis_template = "This example is {}."

        return True

    def _ensure_intent_pipeline(self) -> bool:
        if self._intent_pipeline is not None:
            return True
        pipeline = _maybe_import_transformers_pipeline()
        if pipeline is None:
            return False
        try:
            self._intent_pipeline = pipeline(
                task="zero-shot-classification",
                model=DEFAULT_INTENT_MODEL,
                multi_label=True,
            )
        except Exception as exc:  # noqa: BLE001 - HF backend specific
            logger.warning("Intent pipeline unavailable: %s", exc)
            self._intent_pipeline = None
            return False
        return True

    def _intent_default_prediction(self) -> Tuple[str, List[Dict[str, float]]]:
        if not self.intent_labels:
            return "", []
        topn = min(3, len(self.intent_labels))
        default_labels = self.intent_labels[:topn]
        score = 1.0 / topn if topn else 0.0
        entries = [{"label": label, "score": score} for label in default_labels]
        return default_labels[0], entries

    def _infer_intent_with_onnx(self, text: str) -> Tuple[str, List[Dict[str, float]]]:
        if self._intent_session is None or self._intent_tokenizer is None:
            return self._intent_default_prediction()

        entail_idx = self._intent_entail_idx
        contra_idx = self._intent_contra_idx
        results: List[Dict[str, float]] = []

        for label in self.intent_labels:
            hypothesis = self._intent_hypothesis_template.format(label)
            encoded = self._intent_tokenizer(
                text,
                hypothesis,
                return_tensors="np",
                truncation=True,
            )
            inputs = {name: np.asarray(value) for name, value in encoded.items()}
            logits = self._intent_session.run(None, inputs)[0]
            arr = np.array(logits, dtype=np.float32).ravel()

            if (
                entail_idx is not None
                and contra_idx is not None
                and 0 <= entail_idx < arr.size
                and 0 <= contra_idx < arr.size
            ):
                pair = np.array([arr[contra_idx], arr[entail_idx]], dtype=np.float32)
                score = float(_softmax(pair)[-1])
            elif entail_idx is not None and 0 <= entail_idx < arr.size:
                probs = _softmax(arr)
                score = float(probs[entail_idx])
            else:
                probs = _softmax(arr)
                score = float(np.max(probs))

            results.append({"label": label, "score": score})

        results.sort(key=lambda item: item["score"], reverse=True)
        top_label = results[0]["label"] if results else ""
        return top_label, results[: min(3, len(results))]

    def _infer_intent_with_pipeline(self, text: str) -> Tuple[str, List[Dict[str, float]]]:
        if self._intent_pipeline is None:
            return self._intent_default_prediction()

        candidates = self.intent_labels or ["other"]
        raw = self._intent_pipeline(text, candidate_labels=candidates, multi_label=True)
        entries: List[Dict[str, float]]
        if isinstance(raw, dict) and "labels" in raw and "scores" in raw:
            entries = [
                {"label": str(label), "score": float(score)}
                for label, score in zip(raw["labels"], raw["scores"])
            ]
        else:
            entries = []
            for item in raw:
                if isinstance(item, dict):
                    label = str(item.get("label", ""))
                    score = float(item.get("score", 0.0))
                else:
                    label = str(getattr(item, "label", ""))
                    score = float(getattr(item, "score", 0.0))
                entries.append({"label": label, "score": score})

        entries.sort(key=lambda item: item["score"], reverse=True)
        top_label = entries[0]["label"] if entries else ""
        return top_label, entries[: min(3, len(entries))]

    # ---- Public API extension ----
    def _infer_intent(self, text: str) -> Tuple[str, List[Dict[str, float]]]:
        clean_text = (text or "").strip()
        if not clean_text:
            return self._intent_default_prediction()

        self._lazy_intent()

        if self._intent_session is not None and self._intent_tokenizer is not None:
            try:
                return self._infer_intent_with_onnx(clean_text)
            except Exception as exc:  # noqa: BLE001 - runtime dependent
                logger.warning("Intent ONNX inference failed: %s", exc)

        if self._intent_pipeline is not None:
            try:
                return self._infer_intent_with_pipeline(clean_text)
            except Exception as exc:  # noqa: BLE001 - runtime dependent
                logger.warning("Intent pipeline inference failed: %s", exc)

        return self._intent_default_prediction()


__all__ = __all__ + ("EmotionIntentAnalyzer", "create_onnx_session")
