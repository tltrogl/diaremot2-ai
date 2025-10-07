"""Emotion analysis utilities (ONNX-first, HF fallback).

This module adheres to DiaRemot's ONNX-preferred architecture and CPU-only
constraint. It provides text emotion (GoEmotions 28), audio SER (8-class), and
V/A/D estimates, returning fields consumed by Stage 7.
"""

from __future__ import annotations

import json
import logging
import os
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .intent_defaults import INTENT_LABELS_DEFAULT
from .ser_dpngtm import SERDpngtm

# Preprocessing: strictly librosa/scipy/numpy
try:
    import librosa  # type: ignore
except ImportError:  # pragma: no cover - handled gracefully at runtime
    librosa = None  # type: ignore

logger = logging.getLogger(__name__)


DEFAULT_TEXT_EMOTION_MODEL = "SamLowe/roberta-base-go_emotions"

if not hasattr(np, "int64"):
    np.int64 = int  # type: ignore[attr-defined]
if not hasattr(np, "ones"):
    np.ones = lambda shape, dtype=None: [  # type: ignore[attr-defined]
        1.0
        for _ in range(
            shape if isinstance(shape, int) else int(math.prod(shape))
        )
    ]


def _softmax(x: Sequence[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32) if hasattr(np, "asarray") else list(x)
    arr_list = list(arr)
    if not arr_list:
        return np.asarray([], dtype=np.float32) if hasattr(np, "asarray") else []
    max_val = max(arr_list)
    shifted = [value - max_val for value in arr_list]
    exps = [math.exp(value) for value in shifted]
    denom = sum(exps) or 1.0
    probs = [value / denom for value in exps]
    return np.asarray(probs, dtype=np.float32) if hasattr(np, "asarray") else probs


def _topk(labels: List[str], probs: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
    k = min(k, len(labels))
    idx = np.argpartition(-probs, k - 1)[:k]
    idx = idx[np.argsort(-probs[idx])]
    return [(labels[i], float(probs[i])) for i in idx]


def _json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _sr_target() -> int:
    # Keep consistent with PreprocessConfig.target_sr in AGENTS.md
    return 16000


# GoEmotions 28 labels (SamLowe/roberta-base-go_emotions)
GOEMOTIONS_LABELS: List[str] = [
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


# Default 8-class SER labels (common mapping; can be overridden by model-specific labels)
SER8_LABELS: List[str] = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
]


def _resolve_model_dir() -> str:
    d = os.environ.get("DIAREMOT_MODEL_DIR", "")
    if d:
        return d
    # Windows-friendly default under repo-local cache
    local = os.path.join(".cache", "models")
    os.makedirs(local, exist_ok=True)
    return local


def _ort_session(path: str, *, intra_op_threads: Optional[int] = None):
    try:
        import onnxruntime as ort  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(f"onnxruntime not available: {exc}") from exc

    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if intra_op_threads is None:
        sess_options.intra_op_num_threads = min(4, os.cpu_count() or 1)
    else:
        sess_options.intra_op_num_threads = max(1, int(intra_op_threads))
    sess_options.inter_op_num_threads = 1
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    # DiaRemot is CPU-only per AGENTS.md. Do not use GPU providers.
    providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(path, sess_options=sess_options, providers=providers)


def create_onnx_session(path: str, *, intra_op_threads: Optional[int] = None):
    """Factory exposed for unit tests and dependency injection."""

    return _ort_session(path, intra_op_threads=intra_op_threads)


def _maybe_import_transformers_pipeline():
    try:
        from transformers import pipeline  # type: ignore

        return pipeline
    except Exception:
        return None


@dataclass
class EmotionOutputs:
    # Numeric affect (if available)
    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0

    # Audio SER
    emotion_top: str = "neutral"
    emotion_scores_json: str = _json({})

    # Text emotions (GoEmotions)
    text_emotions_top5_json: str = _json([])
    text_emotions_full_json: str = _json({})


class OnnxTextEmotion:
    def __init__(
        self,
        model_path: str,
        labels: Sequence[str] = GOEMOTIONS_LABELS,
        tokenizer_name: str = DEFAULT_TEXT_EMOTION_MODEL,
    ):
        self.labels = list(labels)
        self.sess = _ort_session(model_path)

        from transformers import AutoTokenizer  # type: ignore

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __call__(self, text: str) -> Tuple[List[Tuple[str, float]], Dict[str, float]]:
        enc = self.tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            padding="max_length",
            max_length=128,
        )
        inputs = {self.sess.get_inputs()[0].name: enc["input_ids"].astype(np.int64)}
        # Handle attention_mask if present
        if len(self.sess.get_inputs()) > 1 and "attention_mask" in enc:
            inputs[self.sess.get_inputs()[1].name] = enc["attention_mask"].astype(np.int64)
        # Optional token_type_ids
        if len(self.sess.get_inputs()) > 2 and "token_type_ids" in enc:
            inputs[self.sess.get_inputs()[2].name] = enc["token_type_ids"].astype(np.int64)

        out = self.sess.run(None, inputs)
        logits = out[0]
        if logits.ndim == 2:
            logits = logits[0]
        probs = _softmax(logits.astype(np.float32))
        full = {self.labels[i]: float(probs[i]) for i in range(len(self.labels))}
        top5 = _topk(self.labels, probs, k=5)
        return top5, full


class HfTextEmotionFallback:
    def __init__(self, model_name: str = DEFAULT_TEXT_EMOTION_MODEL):
        pipeline = _maybe_import_transformers_pipeline()
        if pipeline is None:
            raise RuntimeError("transformers pipeline() unavailable for fallback")
        self.pipe = pipeline(
            task="text-classification",
            model=model_name,
            top_k=None,
            truncation=True,
        )

    def __call__(self, text: str) -> Tuple[List[Tuple[str, float]], Dict[str, float]]:
        out = self.pipe(text)[0]
        # HF returns list of dicts with 'label' and 'score'
        full = {d["label"].lower(): float(d["score"]) for d in out}
        # Ensure all 28 labels exist
        for lab in GOEMOTIONS_LABELS:
            full.setdefault(lab, 0.0)
        arr = np.array([full[lab] for lab in GOEMOTIONS_LABELS], dtype=np.float32)
        arr = arr / (arr.sum() + 1e-8)
        top5 = _topk(GOEMOTIONS_LABELS, arr, 5)
        return top5, {lab: float(arr[i]) for i, lab in enumerate(GOEMOTIONS_LABELS)}


class TorchAudioEmotion:
    """Torch/HF-backed SER implementation (primary backend)."""

    def __init__(
        self,
        labels: List[str] = SER8_LABELS,
        *,
        model_dir: Optional[str] = None,
        disable_downloads: bool = False,
    ) -> None:
        self.labels = labels
        allow_downloads = not disable_downloads
        self._backend = SERDpngtm(
            model_dir=model_dir,
            allow_downloads=allow_downloads,
        )

    @staticmethod
    def _ensure_mono_16k(y: np.ndarray, sr: int) -> np.ndarray:
        if y.ndim > 1:
            y = np.mean(y, axis=-1)
        target_sr = _sr_target()
        if librosa is not None and sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        return y.astype(np.float32)

    def _normalise_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        base: Dict[str, float] = {lab: 0.0 for lab in self.labels}
        for key, value in scores.items():
            label = key.lower()
            if label in base:
                base[label] = float(value)
            else:
                base[key] = float(value)
        total = float(sum(base.values()))
        if total > 0.0:
            for lab in list(base):
                base[lab] = base[lab] / total
        return base

    def __call__(self, y: np.ndarray, sr: int) -> Tuple[str, Dict[str, float]]:
        y16k = self._ensure_mono_16k(y, sr)
        _top, raw_scores = self._backend.predict_16k_f32(y16k)
        scores = self._normalise_scores(raw_scores)
        top_label = max(scores.items(), key=lambda item: item[1])[0]
        return top_label, scores


class OnnxAudioEmotion:
    def __init__(self, model_path: str, labels: List[str] = SER8_LABELS):
        self.labels = labels
        self.sess = _ort_session(model_path)

    @staticmethod
    def _ensure_mono_16k(y: np.ndarray, sr: int) -> np.ndarray:
        if y.ndim > 1:
            y = np.mean(y, axis=-1)
        target_sr = _sr_target()
        if librosa is not None and sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        return y.astype(np.float32)

    def _as_waveform_input(self, y: np.ndarray) -> Dict[str, np.ndarray]:
        # Try to feed as [1, 1, T]
        inp_name = self.sess.get_inputs()[0].name
        return {inp_name: y[None, None, :]}

    def _as_mel_input(self, y: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        if librosa is None:
            return None
        sr = _sr_target()
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=64)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-6)
        x = mel_db.astype(np.float32)[None, None, :, :]  # [1,1,64,frames]
        inp_name = self.sess.get_inputs()[0].name
        return {inp_name: x}

    def __call__(self, y: np.ndarray, sr: int) -> Tuple[str, Dict[str, float]]:
        y = self._ensure_mono_16k(y, sr)
        # Try raw waveform first
        inputs = self._as_waveform_input(y)
        try:
            out = self.sess.run(None, inputs)
        except Exception:  # noqa: BLE001 - onnxruntime errors vary by build
            # Try mel fallback shape
            mel_inputs = self._as_mel_input(y)
            if mel_inputs is None:
                raise
            out = self.sess.run(None, mel_inputs)

        logits = out[0]
        if logits.ndim == 2:
            logits = logits[0]
        probs = _softmax(logits.astype(np.float32))
        full = {self.labels[i]: float(probs[i]) for i in range(len(self.labels))}
        top_label = max(full.items(), key=lambda kv: kv[1])[0]
        return top_label, full


class OnnxVADEmotion:
    def __init__(self, model_path: str):
        self.sess = _ort_session(model_path)

    def __call__(self, y: np.ndarray, sr: int) -> Tuple[float, float, float]:
        # Keep simple; feed pooled features or raw waveform
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        target_sr = _sr_target()
        if librosa is not None and sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        x = y.astype(np.float32)[None, None, :]
        inp_name = self.sess.get_inputs()[0].name
        out = self.sess.run(None, {inp_name: x})
        arr = np.array(out[0]).astype(np.float32).ravel()
        # Expect [3] -> V, A, D; clip to [-1,1]
        if arr.size >= 3:
            v, a, d = arr[:3]
        else:
            v = a = d = 0.0
        v = float(np.clip(v, -1.0, 1.0))
        a = float(np.clip(a, -1.0, 1.0))
        d = float(np.clip(d, -1.0, 1.0))
        return v, a, d


class EmotionAnalyzer:
    """
    Emotion analyzer favouring ONNX for text/VAD with Torch SER primary fallback.

    Produces fields required by Stage 7 (affect_and_assemble):
    - valence, arousal, dominance
    - emotion_top, emotion_scores_json
    - text_emotions_top5_json, text_emotions_full_json
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        disable_downloads: Optional[bool] = None,
        *,
        text_emotion_model: str = DEFAULT_TEXT_EMOTION_MODEL,
        affect_text_model_dir: Optional[str] = None,
    ):
        self.model_dir = model_dir or _resolve_model_dir()
        self.disable_downloads = bool(disable_downloads or False)
        self.text_emotion_model_name = text_emotion_model
        self.affect_text_model_dir = affect_text_model_dir

        # Try ONNX/Torch for each component (lazily initialised)
        self._text_model: Optional[OnnxTextEmotion] = None
        self._text_fallback: Optional[HfTextEmotionFallback] = None
        self._audio_model: Optional[Callable[[np.ndarray, int], Tuple[str, Dict[str, float]]]] = (
            None
        )
        self._vad_model: Optional[OnnxVADEmotion] = None

        # Paths
        text_root = Path(affect_text_model_dir or self.model_dir)
        self.path_text_onnx = str(text_root / "roberta-base-go_emotions.onnx")
        self.path_ser8_onnx = os.path.join(self.model_dir, "ser_8class.onnx")
        self.path_vad_onnx = os.path.join(self.model_dir, "vad_model.onnx")

        # Torch SER location (optional local snapshot)
        self.path_ser_torch: Optional[str] = os.getenv("DIAREMOT_SER_MODEL_DIR")
        if not self.path_ser_torch:
            candidate = os.path.join(self.model_dir, "dpngtm_ser")
            if os.path.isdir(candidate):
                self.path_ser_torch = candidate

        # Allow explicit override from env (exported ONNX path)
        env_ser = os.getenv("DIAREMOT_SER_ONNX")
        if env_ser:
            self.path_ser8_onnx = env_ser

    # Initialize lazily upon first use to avoid import overhead when unused

    # ---- Lazy initializers ----
    def _ensure_text_model(self):
        if self._text_model is not None or self._text_fallback is not None:
            return
        try:
            self._text_model = OnnxTextEmotion(
                self.path_text_onnx,
                labels=GOEMOTIONS_LABELS,
                tokenizer_name=self.text_emotion_model_name,
            )
        except (FileNotFoundError, RuntimeError) as exc:
            logger.warning("Text emotion ONNX unavailable: %s", exc)
            if self.disable_downloads:
                self._text_fallback = None
            else:
                try:
                    self._text_fallback = HfTextEmotionFallback(
                        model_name=self.text_emotion_model_name
                    )
                    logger.warning("Using HuggingFace fallback for text emotion.")
                except Exception as fb_exc:  # noqa: BLE001
                    logger.warning("HF fallback unavailable: %s", fb_exc)
                    self._text_fallback = None

    def _ensure_audio_model(self):
        if self._audio_model is not None:
            return
        # Primary: Torch/HF backend
        try:
            self._audio_model = TorchAudioEmotion(
                labels=SER8_LABELS,
                model_dir=self.path_ser_torch,
                disable_downloads=self.disable_downloads,
            )
            return
        except RuntimeError as exc:
            logger.warning("Audio SER torch backend unavailable: %s", exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Audio SER torch backend failed to initialise: %s", exc)

        # Fallback: ONNXRuntime
        try:
            self._audio_model = OnnxAudioEmotion(self.path_ser8_onnx)
        except (FileNotFoundError, RuntimeError) as exc:
            logger.warning("Audio SER ONNX unavailable: %s", exc)
            self._audio_model = None

    def _ensure_vad_model(self):
        if self._vad_model is not None:
            return
        try:
            self._vad_model = OnnxVADEmotion(self.path_vad_onnx)
        except (FileNotFoundError, RuntimeError) as exc:
            logger.warning("V/A/D ONNX unavailable: %s", exc)
            self._vad_model = None

    # ---- Public API ----
    def analyze_text(self, text: str) -> Tuple[List[Tuple[str, float]], Dict[str, float]]:
        self._ensure_text_model()
        if self._text_model is not None:
            return self._text_model(text)
        if self._text_fallback is not None:
            return self._text_fallback(text)
        # No model available
        zeros = {lab: 0.0 for lab in GOEMOTIONS_LABELS}
        return [], zeros

    def analyze_audio(
        self, y: Optional[np.ndarray], sr: Optional[int]
    ) -> Tuple[str, Dict[str, float]]:
        if y is None or sr is None:
            return "neutral", {lab: 0.0 for lab in SER8_LABELS}
        self._ensure_audio_model()
        if self._audio_model is None:
            return "neutral", {lab: 0.0 for lab in SER8_LABELS}
        try:
            return self._audio_model(y, sr)
        except Exception:  # noqa: BLE001 - backend-specific runtime errors vary
            return "neutral", {lab: 0.0 for lab in SER8_LABELS}

    def analyze_vad_emotion(
        self, y: Optional[np.ndarray], sr: Optional[int]
    ) -> Tuple[float, float, float]:
        if y is None or sr is None:
            return 0.0, 0.0, 0.0
        self._ensure_vad_model()
        if self._vad_model is None:
            return 0.0, 0.0, 0.0
        try:
            return self._vad_model(y, sr)
        except Exception:  # noqa: BLE001 - onnxruntime errors vary by build
            return 0.0, 0.0, 0.0

    def analyze_segment(
        self, text: str, audio_wave: Optional[np.ndarray], sr: Optional[int]
    ) -> EmotionOutputs:
        """Analyze a single segment (text + audio)."""
        top5, full_text = self.analyze_text(text or "")
        emo_top, emo_full = self.analyze_audio(audio_wave, sr)
        v, a, d = self.analyze_vad_emotion(audio_wave, sr)

        return EmotionOutputs(
            valence=v,
            arousal=a,
            dominance=d,
            emotion_top=emo_top,
            emotion_scores_json=_json(emo_full),
            text_emotions_top5_json=_json(top5),
            text_emotions_full_json=_json(full_text),
        )


__all__ = (
    "EmotionAnalyzer",
    "EmotionOutputs",
    "GOEMOTIONS_LABELS",
    "SER8_LABELS",
)


class EmotionIntentAnalyzer(EmotionAnalyzer):
    """Extends :class:`EmotionAnalyzer` with text intent classification support."""

    def __init__(
        self,
        *,
        affect_backend: Optional[str] = None,
        affect_text_model_dir: Optional[str] = None,
        affect_intent_model_dir: Optional[str] = None,
        text_emotion_model: str = DEFAULT_TEXT_EMOTION_MODEL,
        intent_labels: Optional[Sequence[str]] = None,
        analyzer_threads: Optional[int] = None,
        disable_downloads: Optional[bool] = None,
        model_dir: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            model_dir=model_dir,
            disable_downloads=disable_downloads,
            text_emotion_model=text_emotion_model,
            affect_text_model_dir=affect_text_model_dir,
        )

        # Preserve legacy kwargs silently for backwards compatibility
        if kwargs:
            logger.debug("Unused EmotionIntentAnalyzer kwargs ignored: %s", sorted(kwargs))

        self.affect_backend = (affect_backend or "auto").lower()
        self.analyzer_threads = analyzer_threads
        self.intent_labels = self._normalise_intent_labels(intent_labels)
        self.intent_hypothesis_template = "This example is {}."
        self.affect_intent_model_dir = self._discover_intent_model_dir(
            affect_intent_model_dir
        )

        self._intent_session = None
        self._intent_tokenizer = None
        self._intent_config = None
        self._intent_pipeline = None
        self._intent_backend: Optional[str] = None

    # ------------------------------------------------------------------
    # Intent asset discovery helpers
    # ------------------------------------------------------------------
    def _normalise_intent_labels(
        self, labels: Optional[Sequence[str]]
    ) -> List[str]:
        if labels is None:
            source = INTENT_LABELS_DEFAULT
        else:
            source = labels
        normalised = [str(label).strip() for label in source if str(label).strip()]
        return normalised or list(INTENT_LABELS_DEFAULT)

    def _discover_intent_model_dir(self, override: Optional[str]) -> Optional[str]:
        candidates: Iterable[Optional[str]] = (
            override,
            os.getenv("DIAREMOT_INTENT_MODEL_DIR"),
        )
        model_root_candidates: List[Optional[str]] = [os.getenv("DIAREMOT_MODEL_DIR")]
        if self.model_dir:
            model_root_candidates.append(self.model_dir)
        for root in model_root_candidates:
            if root:
                candidates = (*candidates, os.path.join(root, "bart"))

        seen: set[str] = set()
        for candidate in candidates:
            if not candidate:
                continue
            path = str(Path(candidate).expanduser())
            if path in seen:
                continue
            seen.add(path)
            resolved = self._resolve_intent_dir(Path(path))
            if resolved is not None:
                return str(resolved)
        return None

    def _resolve_intent_dir(self, base: Path, max_depth: int = 3) -> Optional[Path]:
        if not base.exists():
            return None
        queue: List[Tuple[Path, int]] = [(base, 0)]
        visited: set[Path] = set()
        while queue:
            current, depth = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            if self._intent_assets_present(current):
                return current
            if depth >= max_depth:
                continue
            try:
                for child in current.iterdir():
                    if child.is_dir():
                        queue.append((child, depth + 1))
            except PermissionError:
                continue
        return None

    def _intent_assets_present(self, directory: Path) -> bool:
        if not directory.is_dir():
            return False
        onnx_names = ["model_uint8.onnx", "model.onnx"]
        has_explicit_onnx = any((directory / name).is_file() for name in onnx_names)
        has_generic_onnx = bool(list(directory.glob("*.onnx")))
        if has_explicit_onnx or has_generic_onnx:
            return True
        config_path = directory / "config.json"
        has_config = config_path.is_file()
        has_weights = bool(list(directory.glob("pytorch_model*.bin"))) or bool(
            list(directory.glob("*.safetensors"))
        )
        return has_config and has_weights

    # ------------------------------------------------------------------
    # Backend initialisation
    # ------------------------------------------------------------------
    def _ensure_intent_backend(self) -> Optional[str]:
        if self._intent_backend is not None:
            return self._intent_backend

        backend = self.affect_backend
        if backend in {"auto", "onnx"}:
            if self._init_intent_onnx():
                self._intent_backend = "onnx"
                return self._intent_backend
            if backend == "onnx":
                self._intent_backend = None
                return None

        if backend in {"auto", "hf", "pipeline", "torch"}:
            if self._init_intent_pipeline():
                self._intent_backend = "hf"
                return self._intent_backend

        self._intent_backend = None
        return None

    def _init_intent_onnx(self) -> bool:
        if not self.affect_intent_model_dir:
            return False

        model_dir = Path(self.affect_intent_model_dir)
        config_data: Dict[str, object] = {}
        config_path = model_dir / "config.json"
        if config_path.is_file():
            try:
                config_data = json.loads(config_path.read_text())
            except Exception as exc:  # noqa: BLE001
                logger.warning("Intent config unreadable at %s: %s", config_path, exc)
                config_data = {}

        template = config_data.get("hypothesis_template")
        if isinstance(template, str) and "{}" in template:
            self.intent_hypothesis_template = template

        try:
            from transformers import AutoConfig, AutoTokenizer  # type: ignore
        except Exception as exc:  # noqa: BLE001
            logger.warning("transformers unavailable for intent tokenizer: %s", exc)
            return False

        try:
            self._intent_tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self._intent_config = AutoConfig.from_pretrained(model_dir)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load intent tokenizer/config from %s: %s", model_dir, exc)
            self._intent_tokenizer = None
            self._intent_config = None
            return False

        onnx_path = self._select_intent_onnx_path(model_dir)
        if onnx_path is None:
            logger.warning("Intent ONNX weights missing under %s", model_dir)
            return False

        session_kwargs: Dict[str, object] = {}
        if self.analyzer_threads is not None:
            session_kwargs["intra_op_threads"] = self.analyzer_threads
        try:
            self._intent_session = create_onnx_session(str(onnx_path), **session_kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to initialise intent ONNX session: %s", exc)
            self._intent_session = None
            return False

        return True

    def _select_intent_onnx_path(self, model_dir: Path) -> Optional[Path]:
        preferred = ["model_uint8.onnx", "model_quantized.onnx", "model.onnx"]
        for name in preferred:
            candidate = model_dir / name
            if candidate.is_file():
                return candidate
        for candidate in model_dir.glob("*.onnx"):
            if candidate.is_file():
                return candidate
        return None

    def _init_intent_pipeline(self) -> bool:
        pipeline_fn = _maybe_import_transformers_pipeline()
        if pipeline_fn is None:
            return False

        model_ref = self.affect_intent_model_dir or "facebook/bart-large-mnli"
        try:
            self._intent_pipeline = pipeline_fn(
                task="zero-shot-classification",
                model=model_ref,
                device=-1,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to initialise transformers intent pipeline: %s", exc)
            self._intent_pipeline = None
            return False

        return True

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _intent_default_response(self) -> Tuple[str, List[Dict[str, float]]]:
        labels = [label for label in self.intent_labels if label]
        if not labels:
            return "", []
        top_candidates = labels[:3]
        weight = 1.0 / len(top_candidates)
        return labels[0], [
            {"label": label, "score": float(weight)} for label in top_candidates
        ]

    def _infer_intent(self, text: str) -> Tuple[str, List[Dict[str, float]]]:
        text = text.strip()
        if not text:
            return self._intent_default_response()

        backend = self._ensure_intent_backend()

        if backend == "onnx" and self._intent_session and self._intent_tokenizer:
            try:
                return self._infer_intent_onnx(text)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Intent ONNX inference failed: %s", exc)

        if backend == "hf" and self._intent_pipeline is not None:
            try:
                return self._infer_intent_pipeline(text)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Transformers intent pipeline failed: %s", exc)

        return self._intent_default_response()

    def _infer_intent_onnx(self, text: str) -> Tuple[str, List[Dict[str, float]]]:
        tokenizer = self._intent_tokenizer
        session = self._intent_session
        config = self._intent_config
        assert tokenizer is not None and session is not None

        entail_idx = None
        if config is not None:
            label2id = getattr(config, "label2id", None) or {}
            if isinstance(label2id, dict):
                for label, idx in label2id.items():
                    if str(label).lower() == "entailment":
                        entail_idx = int(idx)
                        break
            if entail_idx is None:
                id2label = getattr(config, "id2label", None) or {}
                if isinstance(id2label, dict):
                    for idx, label in id2label.items():
                        if str(label).lower() == "entailment":
                            entail_idx = int(idx)
                            break
        if entail_idx is None:
            entail_idx = -1

        scores: List[float] = []
        for label in self.intent_labels:
            hypothesis = self.intent_hypothesis_template.format(label)
            encoded = tokenizer(
                text,
                hypothesis,
                return_tensors="np",
                truncation=True,
            )

            feeds: Dict[str, np.ndarray] = {}
            input_names: Iterable[str]
            if hasattr(session, "get_inputs"):
                input_names = [meta.name for meta in session.get_inputs()]
            else:
                input_names = encoded.keys()
            for name in input_names:
                if name not in encoded:
                    continue
                value = encoded[name]
                try:
                    dtype = getattr(np, "int64", int)
                    feeds[name] = np.asarray(value, dtype=dtype)
                except Exception:  # noqa: BLE001
                    feeds[name] = value

            raw_logits = session.run(None, feeds)[0]
            logits_array = (
                np.asarray(raw_logits, dtype=np.float32)
                if hasattr(np, "asarray")
                else raw_logits
            )
            if hasattr(logits_array, "ndim") and getattr(logits_array, "ndim") > 1:
                logits_array = logits_array[0]
            elif not hasattr(logits_array, "ndim"):
                if logits_array and isinstance(logits_array[0], (list, tuple)):
                    logits_array = logits_array[0]
            logits_vector = (
                np.asarray(logits_array, dtype=np.float32)
                if hasattr(np, "asarray")
                else list(logits_array)
            )
            probs = _softmax(logits_vector)
            probs_list = list(probs.tolist()) if hasattr(probs, "tolist") else list(probs)
            if entail_idx >= 0 and entail_idx < len(probs_list):
                score = float(probs_list[entail_idx])
            else:
                score = float(probs_list[-1])
            scores.append(score)

        return self._rank_intent_scores(scores)

    def _infer_intent_pipeline(self, text: str) -> Tuple[str, List[Dict[str, float]]]:
        pipeline = self._intent_pipeline
        assert pipeline is not None

        raw = pipeline(text, self.intent_labels, multi_label=True)
        if isinstance(raw, list):
            payload = raw[0] if raw else {}
        else:
            payload = raw
        labels = payload.get("labels", []) if isinstance(payload, dict) else []
        scores = payload.get("scores", []) if isinstance(payload, dict) else []

        score_map = {str(label): float(score) for label, score in zip(labels, scores)}
        ordered_scores = [score_map.get(label, 0.0) for label in self.intent_labels]
        return self._rank_intent_scores(ordered_scores)

    def _rank_intent_scores(self, scores: Sequence[float]) -> Tuple[str, List[Dict[str, float]]]:
        if not scores:
            return self._intent_default_response()
        arr = np.asarray(scores, dtype=np.float32) if hasattr(np, "asarray") else list(scores)
        arr_list = list(arr)
        if hasattr(np, "argsort"):
            indices = list(np.argsort(arr)[::-1])
        else:
            indices = [idx for idx, _ in sorted(enumerate(arr_list), key=lambda item: item[1], reverse=True)]
        top_label = self.intent_labels[indices[0]] if indices else self.intent_labels[0]
        top3 = []
        for idx in indices[:3]:
            top3.append({"label": self.intent_labels[idx], "score": float(arr_list[idx])})
        return top_label, top3


__all__ = __all__ + ("EmotionIntentAnalyzer",)
