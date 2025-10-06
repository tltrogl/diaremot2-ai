import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from .ser_onnx import SEROnnx
import numpy as np

# Preprocessing: strictly librosa/scipy/numpy
try:
    import librosa  # type: ignore
except ImportError:  # pragma: no cover - handled gracefully at runtime
    librosa = None  # type: ignore

logger = logging.getLogger(__name__)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


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


"""Emotion analysis utilities (ONNX-first, HF fallback).

This module adheres to DiaRemot's ONNX-preferred architecture and CPU-only
constraint. It provides text emotion (GoEmotions 28), audio SER (8-class), and
V/A/D estimates, returning fields consumed by Stage 7.
"""

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
    def __init__(self, model_path: str, labels: List[str] = GOEMOTIONS_LABELS):
        self.labels = labels
        self.sess = _ort_session(model_path)
        self.ser_onnx = SEROnnx()  # uses DIAREMOT_SER_ONNX and auto-picks DML

        # Tokenizer only (no HF inference)
        from transformers import AutoTokenizer  # type: ignore

        self.tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")

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
    def __init__(self):
        pipeline = _maybe_import_transformers_pipeline()
        if pipeline is None:
            raise RuntimeError("transformers pipeline() unavailable for fallback")
        self.pipe = pipeline(
            task="text-classification",
            model="SamLowe/roberta-base-go_emotions",
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


class OnnxAudioEmotion:
    def __init__(self, model_path: str, labels: List[str] = SER8_LABELS):
        self.labels = labels
        self.sess = _ort_session(model_path)

    @staticmethod
    def _ensure_mono_16k(y: np.ndarray, sr: int) -> np.ndarray:
        if y.ndim > 1:
            y = np.mean(y, axis=1)
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
    ONNX-first emotion analyzer with graceful fallbacks.

    Produces fields required by Stage 7 (affect_and_assemble):
    - valence, arousal, dominance
    - emotion_top, emotion_scores_json
    - text_emotions_top5_json, text_emotions_full_json
    """

    def __init__(self, model_dir: Optional[str] = None, disable_downloads: Optional[bool] = None):
        self.model_dir = model_dir or _resolve_model_dir()
        self.disable_downloads = bool(disable_downloads or False)

        # Try ONNX for each component
        self._text_model: Optional[OnnxTextEmotion] = None
        self._text_fallback: Optional[HfTextEmotionFallback] = None
        self._audio_model: Optional[OnnxAudioEmotion] = None
        self._vad_model: Optional[OnnxVADEmotion] = None

        # Paths
        self.path_text_onnx = os.path.join(self.model_dir, "roberta-base-go_emotions.onnx")
        self.path_ser8_onnx = os.path.join(self.model_dir, "ser_8class.onnx")
        self.path_vad_onnx = os.path.join(self.model_dir, "vad_model.onnx")

        # Allow explicit override from env (your exported ONNX path)
        _env_ser = os.getenv("DIAREMOT_SER_ONNX")
        if _env_ser:
            self.path_ser8_onnx = _env_ser

    # Initialize lazily upon first use to avoid import overhead when unused

    # ---- Lazy initializers ----
    def _ensure_text_model(self):
        if self._text_model is not None or self._text_fallback is not None:
            return
        try:
            self._text_model = OnnxTextEmotion(self.path_text_onnx)
        except (FileNotFoundError, RuntimeError) as exc:
            logger.warning("Text emotion ONNX unavailable: %s", exc)
            if self.disable_downloads:
                self._text_fallback = None
            else:
                try:
                    self._text_fallback = HfTextEmotionFallback()
                    logger.warning("Using HuggingFace fallback for text emotion.")
                except Exception as fb_exc:  # noqa: BLE001
                    logger.warning("HF fallback unavailable: %s", fb_exc)
                    self._text_fallback = None

    def _ensure_audio_model(self):
        if self._audio_model is not None:
            return
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
        except Exception:  # noqa: BLE001 - onnxruntime errors vary by build
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

# Back-compat alias expected by orchestrator
class EmotionIntentAnalyzer(EmotionAnalyzer):
    pass

__all__ = __all__ + ("EmotionIntentAnalyzer",)
