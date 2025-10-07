"""Emotion analysis utilities (ONNX-first, HF fallback).

This module adheres to DiaRemot's ONNX-preferred architecture and CPU-only
constraint. It provides text emotion (GoEmotions 28), audio SER (8-class), and
V/A/D estimates, returning fields consumed by Stage 7.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .ser_dpngtm import SERDpngtm
from .intent_defaults import INTENT_LABELS_DEFAULT
from ..io.onnx_utils import create_onnx_session
from ..pipeline.runtime_env import DEFAULT_MODELS_ROOT

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

DEFAULT_INTENT_MODEL = "facebook/bart-large-mnli"


def _resolve_model_dir() -> Path:
    d = os.environ.get("DIAREMOT_MODEL_DIR")
    if d:
        return Path(d).expanduser()
    return Path(DEFAULT_MODELS_ROOT)


def _resolve_component_dir(
    cli_value: Optional[str], env_key: str, *default_subpath: str
) -> Path:
    candidates: list[Path] = []
    if cli_value:
        candidates.append(Path(cli_value).expanduser())
    env_value = os.getenv(env_key)
    if env_value:
        candidates.append(Path(env_value).expanduser())
    model_root = _resolve_model_dir()
    if default_subpath:
        candidates.append(model_root.joinpath(*default_subpath))
    else:
        candidates.append(model_root)

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    return candidates[0]


def _select_first_existing(directory: Path, names: Sequence[str]) -> Path:
    for name in names:
        candidate = directory / name
        if candidate.exists():
            return candidate
    return directory / names[0]


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
        _add(root / "intent")
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
        labels: List[str] = GOEMOTIONS_LABELS,
        *,
        tokenizer_source: Optional[str | os.PathLike[str]] = None,
        disable_downloads: bool = False,
    ):
        self.labels = labels
        self.sess = _ort_session(model_path)
        self.tokenizer = self._load_tokenizer(
            model_path,
            tokenizer_source=tokenizer_source,
            disable_downloads=disable_downloads,
        )

    def _load_tokenizer(
        self,
        model_path: str,
        *,
        tokenizer_source: Optional[str | os.PathLike[str]],
        disable_downloads: bool,
    ):
        from transformers import AutoTokenizer  # type: ignore

        candidates: list[tuple[str, dict[str, object]]] = []
        errors: list[str] = []

        if tokenizer_source:
            local_dir = Path(tokenizer_source).expanduser()
        else:
            local_dir = Path(model_path).expanduser().parent

        local_dir_str = os.fspath(local_dir)
        candidates.append((local_dir_str, {"local_files_only": True}))
        if not disable_downloads:
            candidates.append((local_dir_str, {"local_files_only": False}))
            candidates.append(("SamLowe/roberta-base-go_emotions", {}))

        for identifier, kwargs in candidates:
            try:
                return AutoTokenizer.from_pretrained(identifier, **kwargs)
            except Exception as exc:  # noqa: BLE001 - HF backend specific
                errors.append(f"{identifier}: {exc}")

        details = "; ".join(errors)
        raise RuntimeError(
            "Unable to load text emotion tokenizer; attempted candidates: " + details
        )

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
        text_model_dir: Optional[str] = None,
        ser_model_dir: Optional[str] = None,
        vad_model_dir: Optional[str] = None,
    ):
        base_dir = Path(model_dir).expanduser() if model_dir else _resolve_model_dir()
        self.model_dir = str(base_dir)
        self.disable_downloads = bool(disable_downloads or False)
        self.issues: list[str] = []

        self.text_model_dir = _resolve_component_dir(
            text_model_dir, "DIAREMOT_TEXT_EMO_MODEL_DIR", "text_emotions"
        )
        self.ser_model_dir = _resolve_component_dir(
            ser_model_dir, "AFFECT_SER_MODEL_DIR", "affect", "ser8"
        )
        self.vad_model_dir = _resolve_component_dir(
            vad_model_dir, "AFFECT_VAD_DIM_MODEL_DIR", "affect", "vad_dim"
        )

        # Paths
        self.path_text_onnx = str(
            _select_first_existing(
                self.text_model_dir,
                ("model.onnx", "roberta-base-go_emotions.onnx"),
            )
        )
        self.path_ser8_onnx = str(
            _select_first_existing(
                self.ser_model_dir,
                ("model.onnx", "ser_8class.onnx"),
            )
        )
        self.path_vad_onnx = str(
            _select_first_existing(
                self.vad_model_dir,
                ("model.onnx", "vad_model.onnx"),
            )
        )

        # Try ONNX/Torch for each component (lazily initialised)
        self._text_model: Optional[OnnxTextEmotion] = None
        self._text_fallback: Optional[HfTextEmotionFallback] = None
        self._audio_model: Optional[Callable[[np.ndarray, int], Tuple[str, Dict[str, float]]]] = (
            None
        )
        self._vad_model: Optional[OnnxVADEmotion] = None

        # Torch SER location (optional local snapshot)
        self.path_ser_torch: Optional[str] = os.getenv("DIAREMOT_SER_MODEL_DIR")
        if not self.path_ser_torch:
            candidate = base_dir / "dpngtm_ser"
            if os.path.isdir(candidate):
                self.path_ser_torch = os.fspath(candidate)

        # Allow explicit override from env (exported ONNX path)
        env_ser = os.getenv("DIAREMOT_SER_ONNX")
        if env_ser:
            self.path_ser8_onnx = env_ser

    # Initialize lazily upon first use to avoid import overhead when unused

    # ---- Lazy initializers ----
    def _record_issue(self, message: str) -> None:
        if message not in self.issues:
            self.issues.append(message)

    def _ensure_text_model(self):
        if self._text_model is not None or self._text_fallback is not None:
            return
        try:
            self._text_model = OnnxTextEmotion(
                self.path_text_onnx,
                tokenizer_source=self.text_model_dir,
                disable_downloads=self.disable_downloads,
            )
        except (FileNotFoundError, RuntimeError) as exc:
            logger.warning("Text emotion ONNX unavailable: %s", exc)
            self._record_issue(
                f"Text emotion ONNX missing under {self.text_model_dir}"
            )
            if self.disable_downloads:
                self._text_fallback = None
            else:
                try:
                    self._text_fallback = HfTextEmotionFallback()
                    logger.warning("Using HuggingFace fallback for text emotion.")
                except Exception as fb_exc:  # noqa: BLE001
                    logger.warning("HF fallback unavailable: %s", fb_exc)
                    self._text_fallback = None
                    self._record_issue("Text emotion fallback unavailable; outputs neutral")

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
        if self._audio_model is None:
            self._record_issue(
                f"Speech emotion model unavailable under {self.ser_model_dir}"
            )

    def _ensure_vad_model(self):
        if self._vad_model is not None:
            return
        try:
            self._vad_model = OnnxVADEmotion(self.path_vad_onnx)
        except (FileNotFoundError, RuntimeError) as exc:
            logger.warning("V/A/D ONNX unavailable: %s", exc)
            self._vad_model = None
            self._record_issue(
                f"Valence/arousal/dominance model unavailable under {self.vad_model_dir}"
            )

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


# Back-compat alias expected by orchestrator with extended intent controls
class EmotionIntentAnalyzer(EmotionAnalyzer):
    def __init__(
        self,
        *,
        text_emotion_model: str = "SamLowe/roberta-base-go_emotions",
        intent_labels: Sequence[str] | None = None,
        affect_backend: Optional[str] = None,
        affect_text_model_dir: Optional[str] = None,
        affect_ser_model_dir: Optional[str] = None,
        affect_vad_model_dir: Optional[str] = None,
        affect_intent_model_dir: Optional[str] = None,
        analyzer_threads: Optional[int] = None,
        disable_downloads: Optional[bool] = None,
        model_dir: Optional[str] = None,
    ) -> None:
        super().__init__(
            model_dir=model_dir,
            disable_downloads=disable_downloads,
            text_model_dir=affect_text_model_dir,
            ser_model_dir=affect_ser_model_dir,
            vad_model_dir=affect_vad_model_dir,
        )

        self.text_emotion_model = text_emotion_model
        labels = intent_labels or INTENT_LABELS_DEFAULT
        self.intent_labels: List[str] = [str(label) for label in labels]
        self.affect_backend = _normalize_backend(affect_backend)
        self.analyzer_threads = analyzer_threads

        self.affect_text_model_dir = os.fspath(self.text_model_dir)
        self.affect_ser_model_dir = os.fspath(self.ser_model_dir)
        self.affect_vad_model_dir = os.fspath(self.vad_model_dir)

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
            self._record_issue("Intent model directory not configured")
            return False

        model_dir = Path(model_dir_str)
        model_path = self._select_onnx_model(model_dir)
        if model_path is None:
            if strict:
                logger.warning("Intent ONNX backend missing model.onnx in %s", model_dir)
            self._record_issue(f"Intent ONNX model missing under {model_dir}")
            return False

        threads = self.analyzer_threads or 1
        try:
            self._intent_session = create_onnx_session(model_path, threads=threads)
        except Exception as exc:  # pragma: no cover - runtime dependent
            logger.warning("Intent ONNX session unavailable: %s", exc)
            self._intent_session = None
            self._record_issue(f"Intent ONNX session unavailable: {exc}")
            return False

        try:
            from transformers import AutoConfig, AutoTokenizer  # type: ignore
        except ModuleNotFoundError as exc:
            logger.warning("transformers unavailable for intent tokenizer: %s", exc)
            self._intent_session = None
            self._record_issue("Transformers package missing for intent tokenizer")
            return False

        try:
            self._intent_tokenizer = AutoTokenizer.from_pretrained(model_dir_str)
            self._intent_config = AutoConfig.from_pretrained(model_dir_str)
        except Exception as exc:  # noqa: BLE001 - dependent on HF cache state
            logger.warning("Failed to load intent tokenizer/config: %s", exc)
            self._intent_session = None
            self._intent_tokenizer = None
            self._intent_config = None
            self._record_issue(
                f"Intent tokenizer/config unavailable under {model_dir_str}: {exc}"
            )
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
            self._record_issue("Transformers pipeline unavailable for intent analysis")
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
            self._record_issue(f"Intent transformers pipeline unavailable: {exc}")
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
