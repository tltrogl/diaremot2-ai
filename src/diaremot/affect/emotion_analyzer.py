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

import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

# Optional heavy deps are imported lazily inside methods.
# librosa is cheap enough; used for fallbacks and basic features.
import librosa
import numpy as np

from ..io.onnx_utils import create_onnx_session, ensure_onnx_model
from ..utils.model_paths import iter_model_roots, iter_model_subpaths
from .intent_defaults import INTENT_LABELS_DEFAULT

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


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    s = e.sum() or 1.0
    return (e / s).astype(np.float64)


def _entropy(probs: dict[str, float]) -> float:
    p = np.clip(np.array(list(probs.values()), dtype=np.float64), 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


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


def _is_hf_model_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not (path / "config.json").exists():
        return False
    # Accept standard HF weight files OR ONNX exports in the same folder
    for pattern in _HF_MODEL_WEIGHT_PATTERNS:
        if any(path.glob(pattern)):
            return True
    for onnx_name in ("model_uint8.onnx", "model_int8.onnx", "model.onnx"):
        if (path / onnx_name).exists():
            return True
    return False


def _locate_hf_model_dir(base: Path, max_depth: int = 2) -> Path | None:
    """Search breadth-first for a Hugging Face model directory containing weights."""

    try:
        base = base.expanduser().resolve()
    except Exception:
        base = base.expanduser()

    if not base.exists():
        return None

    queue: list[tuple[Path, int]] = [(base, 0)]
    seen: set[Path] = set()

    while queue:
        current, depth = queue.pop(0)
        if current in seen:
            continue
        seen.add(current)

        if _is_hf_model_dir(current):
            return current

        if depth >= max_depth:
            continue

        try:
            for child in current.iterdir():
                if child.is_dir():
                    queue.append((child, depth + 1))
        except Exception:
            continue

    return None


def _resolve_intent_model_dir(
    explicit_dir: str | None,
) -> str | None:
    """Resolve the intent model directory with environment overrides."""

    if explicit_dir:
        explicit_path = Path(explicit_dir).expanduser()
        located = _locate_hf_model_dir(explicit_path)
        if located:
            return str(located)
        if not explicit_path.exists():
            # Caller provided a model identifier instead of a local directory.
            return explicit_dir

    env_override = os.environ.get("DIAREMOT_INTENT_MODEL_DIR")
    if env_override:
        env_path = Path(env_override).expanduser()
        located = _locate_hf_model_dir(env_path)
        if located:
            return str(located)
        # Respect explicit override even if incomplete
        return None

    for root in iter_model_roots():
        candidate = _locate_hf_model_dir(root / "bart")
        if candidate:
            return str(candidate)

    return None


def _resolve_vad_model_dir(explicit_dir: str | None) -> str | None:
    if explicit_dir:
        explicit_path = Path(explicit_dir).expanduser()
        if explicit_path.exists():
            return str(explicit_path)

    env_override = os.environ.get("DIAREMOT_VAD_MODEL_DIR")
    if env_override:
        env_path = Path(env_override).expanduser()
        if env_path.exists():
            return str(env_path)

    for candidate in iter_model_subpaths("vad-onnx"):
        if candidate.exists():
            return str(candidate)
    return None


def _resolve_text_model_dir(explicit_dir: str | None) -> str | None:
    """Resolve the text emotion model directory or HF id.

    Preference order:
      1) Explicit directory if it exists
      2) DIAREMOT_TEXT_MODEL_DIR env var (if directory exists)
      3) Known model roots (e.g., DIAREMOT_MODEL_DIR, ./models, OS defaults)
      4) Return the original explicit value (could be an HF repo id)
    """

    if explicit_dir:
        p = Path(explicit_dir).expanduser()
        if p.exists():
            return str(p)
        # Could be an HF id; return as-is
        return explicit_dir

    env_override = os.environ.get("DIAREMOT_TEXT_MODEL_DIR")
    if env_override:
        env_path = Path(env_override).expanduser()
        if env_path.exists():
            return str(env_path)

    for candidate in iter_model_subpaths("goemotions-onnx"):
        if candidate.exists():
            return str(candidate)

    return None


class EmotionIntentAnalyzer:
    def __init__(
        self,
        device: str = "cpu",
        text_emotion_model: str = "SamLowe/roberta-base-go_emotions",
        intent_labels: list[str] | None = None,
        affect_backend: str = "auto",
        affect_text_model_dir: str | None = None,
        affect_intent_model_dir: str | None = None,
        affect_ser_model_dir: str | None = None,
        affect_vad_model_dir: str | None = None,
        analyzer_threads: int | None = None,
    ):
        # CPU-only; we still honor thread settings
        try:
            import torch

            torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
        except Exception:
            pass

        self.device = "cpu"
        # Be robust to a None/empty override coming from config
        self.text_model_name = text_emotion_model or "SamLowe/roberta-base-go_emotions"
        labels = list(intent_labels) if intent_labels else list(INTENT_LABELS_DEFAULT)
        self.intent_labels = list(dict.fromkeys(labels))
        self.analyzer_threads = None
        if analyzer_threads is not None:
            try:
                override = int(analyzer_threads)
            except (TypeError, ValueError):
                override = None
            else:
                if override > 0:
                    self.analyzer_threads = override

        # Model names (can be swapped if you want lighter variants)
        self.vad_model_name = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
        self.ser_model_name = "Dpngtm/wav2vec2-emotion-recognition"
        self.intent_model_name = "facebook/bart-large-mnli"

        # Lazy handles
        self._vad_processor = None
        self._vad_model = None
        self._vad_session = None  # ONNX runtime session
        a = None  # noqa
        self._vad_idx = None  # {'valence': i, 'arousal': j, 'dominance': k}

        self._ser_processor = None
        self._ser_model = None
        self._ser_session = None  # ONNX runtime session

        self._text_pipeline = None  # transformers/optimum pipeline
        self._text_session = None  # ONNX runtime session
        self._text_tokenizer = None
        self._intent_pipeline = None
        self._intent_session = None
        self._intent_tokenizer = None
        self._intent_hypothesis_template = "This example is {}."
        self._intent_entail_idx: int | None = None
        self._intent_contra_idx: int | None = None
        self.affect_backend = str(affect_backend).lower()
        self.affect_text_model_dir = _resolve_text_model_dir(affect_text_model_dir)
        self.affect_intent_model_dir = _resolve_intent_model_dir(affect_intent_model_dir)
        self.affect_ser_model_dir = affect_ser_model_dir
        self.affect_vad_model_dir = _resolve_vad_model_dir(affect_vad_model_dir)
        # Optional explicit local SER model directory
        try:
            if affect_ser_model_dir and Path(affect_ser_model_dir).exists():
                self.ser_model_name = str(Path(affect_ser_model_dir))
        except Exception:
            pass

    # -------- public API: unified --------
    def analyze(self, wav: np.ndarray, sr: int, text: str) -> dict[str, object]:
        """
        Unified interface expected by the core.
        Returns:
          {
            "vad": {"valence": float, "arousal": float, "dominance": float},
            "speech_emotion": {"top": str, "scores_8class": {label: prob}, "low_confidence_ser": bool},
            "text_emotions": {"top5": [{"label": str, "score": float}, ...], "full_28class": {label: prob}},
            "intent": {"top": str, "top3": [{"label": str, "score": float}, ...]},
            "affect_hint": str
          }
        """
        res = self._analyze_all(wav, sr, text)
        return {
            "vad": {
                "valence": res.valence,
                "arousal": res.arousal,
                "dominance": res.dominance,
            },
            "speech_emotion": {
                "top": res.ser_top,
                "scores_8class": res.ser_probs,
                "low_confidence_ser": self._ser_low_confidence(res.ser_probs),
            },
            "text_emotions": {"top5": res.text_top5, "full_28class": res.text_full},
            "intent": {"top": res.intent_top, "top3": res.intent_top3},
            "affect_hint": res.affect_hint,
        }

    # -------- internal orchestrator with parallelism --------
    def _analyze_all(self, wav: np.ndarray, sr: int, text: str) -> EmotionAnalysisResult:
        x, sr = _ensure_16k_mono(wav, sr)
        x = _trim_max_len(x, sr, 30.0)
        if self.analyzer_threads is not None:
            workers = self.analyzer_threads
        else:
            workers = min(4, os.cpu_count() or 1)
        workers = max(1, workers)
        normalized_text = text or ""

        if workers == 1:
            valence, arousal, dominance = self._infer_vad(x, sr)
            ser_top, ser_probs = self._infer_ser(x, sr)
            text_full, text_top5 = self._infer_text(normalized_text)
            intent_top, intent_top3 = self._infer_intent(normalized_text)
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                fut_vad = ex.submit(self._infer_vad, x, sr)
                fut_ser = ex.submit(self._infer_ser, x, sr)
                fut_txt = ex.submit(self._infer_text, normalized_text)
                fut_int = ex.submit(self._infer_intent, normalized_text)

                valence, arousal, dominance = fut_vad.result()
                ser_top, ser_probs = fut_ser.result()
                text_full, text_top5 = fut_txt.result()
                intent_top, intent_top3 = fut_int.result()

        # Cross-modal hint using polarity sums + SER polarity buckets
        pol = self._polarity_sums(text_full)
        hint = self._affect_hint(valence, arousal, ser_probs, pol)

        # Optional voice-quality hint via paralinguistics
        vq_hint = None
        try:
            try:
                from . import paralinguistics as para  # type: ignore
            except Exception:
                para = None  # type: ignore
            if para is not None and hasattr(para, "compute_segment_features_v2"):
                feats = para.compute_segment_features_v2(
                    x, sr, 0.0, float(len(x) / sr), text, para.ParalinguisticsConfig()
                )
                jitter = float(feats.get("vq_jitter_pct", 0.0) or 0.0)
                shimmer = float(feats.get("vq_shimmer_db", 0.0) or 0.0)
                hnr = float(feats.get("vq_hnr_db", 0.0) or 0.0)
                cpps = float(feats.get("vq_cpps_db", 0.0) or 0.0)
                if (jitter >= 1.2 or shimmer >= 0.5) and hnr <= 12.0:
                    vq_hint = "voice: tense/rough"
                elif cpps <= 10.0 or hnr <= 10.0:
                    vq_hint = "voice: breathy/hoarse"
                else:
                    vq_hint = "voice: steady"
        except Exception:
            vq_hint = None

        flags = {}
        if vq_hint:
            flags["voice_quality_hint"] = vq_hint

        return EmotionAnalysisResult(
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            ser_top=ser_top,
            ser_probs=ser_probs,
            text_full=text_full,
            text_top5=text_top5,
            intent_top=intent_top,
            intent_top3=intent_top3,
            flags=flags,
            affect_hint=hint if not vq_hint else f"{hint} | {vq_hint}",
        )

    # -------- VAD (dimensional, dynamic mapping) --------
    def _load_vad_onnx(self) -> bool:
        if (
            self._vad_session is not None
            and self._vad_processor is not None
            and self._vad_idx is not None
        ):
            return True

        model_dir = self.affect_vad_model_dir
        candidates: list[str | Path] = []
        names = ("model_uint8.onnx", "model_int8.onnx", "model.onnx")
        if model_dir:
            base = Path(model_dir)
            if base.exists():
                for name in names:
                    cand = base / name
                    if cand.exists():
                        candidates.append(cand)
                if not candidates:
                    candidates.append(base / names[-1])
        if not candidates:
            candidates = [f"hf://{self.vad_model_name}/{name}" for name in names]

        import os as _os

        local_only = str(_os.getenv("HF_HUB_OFFLINE", "")).lower() not in ("", "0", "false") or str(
            _os.getenv("TRANSFORMERS_OFFLINE", "")
        ).lower() not in ("", "0", "false")

        try:
            from transformers import AutoConfig, AutoFeatureExtractor

            extractor_source = None
            if model_dir and Path(model_dir).exists():
                extractor_source = str(Path(model_dir))
            cfg_source = extractor_source or self.vad_model_name

            self._vad_processor = AutoFeatureExtractor.from_pretrained(
                extractor_source or self.vad_model_name
            )
            cfg = AutoConfig.from_pretrained(cfg_source)

            from ..io.onnx_utils import create_onnx_session, ensure_onnx_model

            session = None
            last_exc: Exception | None = None
            for cand in candidates:
                try:
                    model_path = ensure_onnx_model(cand, local_files_only=local_only)
                    session = create_onnx_session(model_path, threads=self.analyzer_threads or 1)
                    last_exc = None
                    break
                except Exception as exc:  # pragma: no cover - runtime dependency issues
                    last_exc = exc
                    continue
            if session is None:
                if last_exc is not None:
                    logger.warning("VAD ONNX model unavailable (%s); will use fallback", last_exc)
                self._vad_processor = None
                return False

            self._vad_session = session
            self._vad_model = None
            self._vad_idx = self._build_vad_index_map(cfg)
            logger.info("VAD ONNX model loaded")
            return True
        except Exception as exc:  # pragma: no cover - runtime dependency issues
            logger.warning("VAD ONNX model unavailable (%s); will use fallback", exc)
            self._vad_session = None
            self._vad_processor = None
            return False

    # -------- VAD (dimensional, dynamic mapping) --------
    def _lazy_vad(self):
        backend = (self.affect_backend or "auto").lower()
        if backend in {"onnx", "auto"}:
            if self._load_vad_onnx():
                return
            if backend == "onnx":
                logger.warning("VAD ONNX model unavailable; falling back to transformers backend")
        if self._vad_model is not None and self._vad_processor is not None:
            return
        try:
            from transformers import (
                AutoConfig,
                AutoModelForAudioClassification,
                AutoProcessor,
            )

            cfg_source = self.vad_model_name
            if self.affect_vad_model_dir and Path(self.affect_vad_model_dir).exists():
                cfg_source = str(Path(self.affect_vad_model_dir))

            cfg = AutoConfig.from_pretrained(cfg_source)
            self._vad_processor = AutoProcessor.from_pretrained(cfg_source)
            self._vad_session = None
            self._vad_model = AutoModelForAudioClassification.from_pretrained(self.vad_model_name)
            # Build index map dynamically
            self._vad_idx = self._build_vad_index_map(cfg)
            logger.info(f"VAD label order mapping: {self._vad_idx}")
        except Exception as e:
            logger.warning(f"VAD model unavailable ({e}); will use fallback")
            self._vad_model = None
            self._vad_processor = None
            self._vad_idx = None

    @staticmethod
    def _build_vad_index_map(cfg) -> dict[str, int]:
        def _norm(s: str) -> str:
            return str(s).lower().strip()

        id2label = getattr(cfg, "id2label", None) or {}
        if isinstance(id2label, dict) and id2label:
            # keys may be int or str→int
            mapping = {}
            for k, v in id2label.items():
                name = _norm(v)
                try:
                    idx = int(k)
                except Exception:
                    try:
                        idx = int(str(k))
                    except Exception:
                        continue
                if "arous" in name:
                    mapping["arousal"] = idx
                elif "domin" in name:
                    mapping["dominance"] = idx
                elif "valen" in name:
                    mapping["valence"] = idx
            if set(mapping.keys()) == {"valence", "arousal", "dominance"}:
                return mapping
        # Fallback known ordering used by this HF head: [arousal, dominance, valence]
        return {"arousal": 0, "dominance": 1, "valence": 2}

    def _infer_vad(self, audio: np.ndarray, sr: int) -> tuple[float, float, float]:
        if audio is None or audio.size == 0 or sr <= 0:
            return 0.0, 0.0, 0.0
        # Try model; fallback to acoustic proxy
        self._lazy_vad()
        try:
            scores = None
            if (
                self._vad_session is not None
                and self._vad_processor is not None
                and self._vad_idx is not None
            ):
                inputs = self._vad_processor(
                    audio, sampling_rate=sr, return_tensors="np", padding=True
                )
                ort_inputs = {k: v for k, v in inputs.items()}
                logits = self._vad_session.run(None, ort_inputs)[0]
                if logits.ndim == 2:
                    logits = logits[0]
                scores = 1.0 / (1.0 + np.exp(-logits))
            elif self._vad_model is not None and self._vad_processor is not None:
                import torch

                inputs = self._vad_processor(
                    audio, sampling_rate=sr, return_tensors="pt", padding=True
                )
                with torch.inference_mode():
                    logits = self._vad_model(**inputs).logits
                    scores = torch.sigmoid(logits).squeeze().cpu().numpy().astype(np.float64)
            else:
                raise RuntimeError("vad model missing")
            # Map to V/A/D in [-1,1]
            idx = self._vad_idx or {"arousal": 0, "dominance": 1, "valence": 2}

            def _unit(z):
                return float(np.clip(2.0 * float(z) - 1.0, -1.0, 1.0))

            arousal = _unit(scores[idx["arousal"]]) if len(scores) > idx["arousal"] else 0.0
            dominance = _unit(scores[idx["dominance"]]) if len(scores) > idx["dominance"] else 0.0
            valence = _unit(scores[idx["valence"]]) if len(scores) > idx["valence"] else 0.0
            return valence, arousal, dominance
        except Exception:
            return self._vad_proxy(audio, sr)

    @staticmethod
    def _vad_proxy(audio: np.ndarray, sr: int) -> tuple[float, float, float]:
        # Cheap acoustic proxies (bounded, robust)
        try:
            sc = (
                float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
                if audio.size
                else 0.0
            )
            valence = float(np.clip((sc - 1000.0) / 2000.0, -1.0, 1.0))

            rms = float(np.mean(librosa.feature.rms(y=audio))) if audio.size else 0.0
            tempo = float(librosa.beat.tempo(y=audio, sr=sr)[0]) if audio.size > sr else 120.0
            energy = float(np.clip(rms * 12.0, 0.0, 1.0))
            arousal = float(
                np.clip(
                    (0.6 * energy + 0.4 * np.clip((tempo - 60) / 120, 0, 1)) * 2 - 1,
                    -1,
                    1,
                )
            )

            f0 = librosa.yin(audio, fmin=70, fmax=300) if audio.size else np.array([np.nan])
            f0m = float(np.nanmean(f0)) if np.isfinite(f0).any() else float("nan")
            dominance = float(np.clip((0 if math.isnan(f0m) else (f0m - 150) / 100), -1, 1))
            return valence, arousal, dominance
        except Exception:
            return 0.0, 0.0, 0.0

    # -------- SER (8-class) --------
    def _lazy_ser(self):
        """Load SER (audio classification) with robust processor fallback.

        Some Transformers versions incorrectly try to load a tokenizer for
        wav2vec2 classification checkpoints. Prefer AutoProcessor, but fall back
        to AutoFeatureExtractor when needed (Torch and ONNX paths).
        """
        if self.affect_backend == "onnx":
            if self._ser_session is not None and self._ser_processor is not None:
                return
            try:
                # Prefer AutoProcessor, fall back to AutoFeatureExtractor
                from transformers import AutoFeatureExtractor, AutoProcessor

                try:
                    proc = AutoProcessor.from_pretrained(self.ser_model_name)
                except Exception:
                    proc = AutoFeatureExtractor.from_pretrained(self.ser_model_name)
                self._ser_processor = proc

                if Path(self.ser_model_name).exists():
                    ident = Path(self.ser_model_name) / "model.onnx"
                else:
                    ident = f"hf://{self.ser_model_name}/model.onnx"
                import os as _os

                _off = str(_os.getenv("HF_HUB_OFFLINE", "")).lower() not in (
                    "",
                    "0",
                    "false",
                ) or str(_os.getenv("TRANSFORMERS_OFFLINE", "")).lower() not in ("", "0", "false")
                model_path = ensure_onnx_model(ident, local_files_only=_off)
                self._ser_session = create_onnx_session(model_path)
            except Exception as e:
                logger.warning(f"SER ONNX model unavailable ({e}); will use fallback")
                self._ser_processor = None
                self._ser_session = None
        else:
            if self._ser_model is not None and self._ser_processor is not None:
                return
            try:
                from transformers import (
                    AutoFeatureExtractor,
                    AutoModelForAudioClassification,
                    AutoProcessor,
                )

                try:
                    proc = AutoProcessor.from_pretrained(self.ser_model_name)
                except Exception:
                    proc = AutoFeatureExtractor.from_pretrained(self.ser_model_name)
                self._ser_processor = proc
                self._ser_model = AutoModelForAudioClassification.from_pretrained(
                    self.ser_model_name
                )
            except Exception as e:
                logger.warning(f"SER model unavailable ({e}); will use fallback")
                self._ser_processor = None
                self._ser_model = None

    def _infer_ser(self, audio: np.ndarray, sr: int) -> tuple[str, dict[str, float]]:
        if audio is None or audio.size == 0 or sr <= 0:
            return "neutral", {lbl: (1.0 if lbl == "neutral" else 0.0) for lbl in SER_LABELS_8}
        self._lazy_ser()
        try:
            if self.affect_backend == "onnx":
                if self._ser_session is None or self._ser_processor is None:
                    raise RuntimeError("ser model missing")
                inputs = self._ser_processor(
                    audio, sampling_rate=sr, return_tensors="np", padding=True
                )
                ort_inputs = {k: v for k, v in inputs.items()}
                probs = self._ser_session.run(None, ort_inputs)[0].squeeze().astype(np.float64)
                # ensure probabilities
                ex = np.exp(probs - np.max(probs))
                probs = ex / (ex.sum() or 1.0)
            else:
                if self._ser_model is None or self._ser_processor is None:
                    raise RuntimeError("ser model missing")
                import torch
                import torch.nn.functional as F

                inputs = self._ser_processor(
                    audio, sampling_rate=sr, return_tensors="pt", padding=True
                )
                with torch.inference_mode():
                    probs = (
                        F.softmax(self._ser_model(**inputs).logits, dim=-1)
                        .squeeze()
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                    )
            # Map to our label order (best effort)
            prob_dict = {}
            # Try config-based id2label if available
            id2label = getattr(getattr(self._ser_model, "config", None), "id2label", None) or {}
            if isinstance(id2label, dict) and id2label:
                for i, lbl in enumerate(SER_LABELS_8):
                    # find closest by name
                    match = None
                    for k, v in id2label.items():
                        if str(v).lower().startswith(lbl[:4]):  # loose match: 'angr'→'angry'
                            try:
                                idx = int(k)
                                match = idx
                                break
                            except Exception:
                                pass
                    if match is not None and match < len(probs):
                        prob_dict[lbl] = float(probs[match])
                # fill any missing using sequential as fallback
            if len(prob_dict) != len(SER_LABELS_8):
                for i, lbl in enumerate(SER_LABELS_8):
                    if lbl not in prob_dict:
                        prob_dict[lbl] = float(probs[i]) if i < len(probs) else 0.0
            prob_dict = _normalize_scores(prob_dict)
            top = max(prob_dict, key=prob_dict.get) if prob_dict else "neutral"
            return top, prob_dict
        except Exception:
            return self._ser_proxy(audio, sr)

    @staticmethod
    def _ser_proxy(audio: np.ndarray, sr: int) -> tuple[str, dict[str, float]]:
        # cheap acoustic color proxy
        try:
            sc = (
                float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
                if audio.size
                else 0.0
            )
            rms = float(np.mean(librosa.feature.rms(y=audio))) if audio.size else 0.0
            hi = float(np.clip((sc - 1500) / 1500, 0, 1))
            en = float(np.clip(rms * 15, 0, 1))
            emo = (
                "happy"
                if hi > 0.5 and en > 0.3
                else ("angry" if en > 0.6 else ("sad" if hi < 0.3 else "neutral"))
            )
            probs = {lbl: 0.01 for lbl in SER_LABELS_8}
            probs[emo] = 0.7
            probs["neutral"] = max(probs["neutral"], 0.2)
            return emo, _normalize_scores(probs)
        except Exception:
            return "neutral", {lbl: (1.0 if lbl == "neutral" else 0.0) for lbl in SER_LABELS_8}

    @staticmethod
    def _ser_low_confidence(probs: dict[str, float]) -> bool:
        if not probs:
            return True
        top = max(probs.values())
        margin = _top_margin(probs)
        Hn = _norm_entropy(probs)
        return bool(top < 0.50 or margin < 0.15 or Hn > 0.85)

    # -------- Text Emotions (28, multi-label) --------
    def _lazy_text(self):
        # Make text loading robust: prefer a local ONNX export first (if present)
        # to avoid failing early when tokenizer.json is malformed. If ONNX session
        # creation succeeds we keep it and attempt to load tokenizer/config but
        # treat tokenizer failures as non-fatal (we can still use ONNX session
        # for inference when tokenizer is available later).
        model_dir = self.affect_text_model_dir or self.text_model_name
        names = ("model_uint8.onnx", "model_int8.onnx", "model.onnx")

        # Fast-path: if a local directory with ONNX exports exists, try those first.
        try:
            if Path(model_dir).exists():
                candidates = []
                base = Path(model_dir)
                for n in names:
                    p = base / n
                    if p.exists():
                        candidates.append(p)
                if not candidates:
                    candidates.append(base / names[-1])

                import os as _os

                _off = str(_os.getenv("HF_HUB_OFFLINE", "")).lower() not in (
                    "",
                    "0",
                    "false",
                ) or str(_os.getenv("TRANSFORMERS_OFFLINE", "")).lower() not in ("", "0", "false")

                last_exc: Exception | None = None
                for cand in candidates:
                    try:
                        mp = ensure_onnx_model(cand, local_files_only=_off)
                        self._text_session = create_onnx_session(mp)
                        last_exc = None
                        break
                    except Exception as e:
                        last_exc = e
                        continue

                # If we created a local ONNX session, attempt to load tokenizer/config
                # but tolerate tokenizer parsing errors (log and keep session).
                if self._text_session is not None:
                    try:
                        from transformers import AutoTokenizer

                        try:
                            self._text_tokenizer = AutoTokenizer.from_pretrained(model_dir)
                        except Exception as tok_exc:
                            logger.warning(
                                "Text tokenizer failed to load (%s); ONNX session available and will be used when tokenizer becomes available",
                                tok_exc,
                            )
                            self._text_tokenizer = None
                    except Exception:
                        # No transformers available; keep session and proceed
                        self._text_tokenizer = None
                    return
                # If creation failed for all local candidates, fall through to network/transformers path
        except Exception as e:
            logger.debug("local ONNX text candidate scan failed: %s", e)

        # Original flow: try to use ONNX (possibly via remote hf://) or transformers pipeline
        if self.affect_backend == "onnx":
            if self._text_session is not None and self._text_tokenizer is not None:
                return
            try:
                from transformers import AutoTokenizer

                model_dir_ref = model_dir
                # Prefer loading tokenizer (remote or local). If tokenizer fails, ensure_onnx_model may still provide a session
                try:
                    self._text_tokenizer = AutoTokenizer.from_pretrained(model_dir_ref)
                except Exception as tok_exc:
                    logger.warning(
                        "Text tokenizer load failed (%s); will still try ONNX model retrieval",
                        tok_exc,
                    )
                    self._text_tokenizer = None

                if Path(model_dir_ref).exists():
                    ident = Path(model_dir_ref) / "model.onnx"
                else:
                    ident = f"hf://{model_dir_ref}/model.onnx"
                import os as _os

                _off = str(_os.getenv("HF_HUB_OFFLINE", "")).lower() not in (
                    "",
                    "0",
                    "false",
                ) or str(_os.getenv("TRANSFORMERS_OFFLINE", "")).lower() not in ("", "0", "false")
                model_path = ensure_onnx_model(ident, local_files_only=_off)
                self._text_session = create_onnx_session(model_path)
            except Exception as e:
                logger.warning(f"Text ONNX model unavailable ({e}); will use fallback")
                self._text_session = None
                # keep tokenizer as-is (may be None)
        else:
            if self._text_pipeline is not None:
                return
            try:
                from transformers import AutoTokenizer

                model_dir_ref = model_dir
                try:
                    self._text_tokenizer = AutoTokenizer.from_pretrained(model_dir_ref)
                except Exception as tok_exc:
                    logger.warning(
                        "Text tokenizer load failed (%s); will try pipeline/ONNX fallback", tok_exc
                    )
                    self._text_tokenizer = None

                # Try to create ONNX session (remote or local); keep pipeline None to prefer lightweight use
                if Path(model_dir_ref).exists():
                    ident = Path(model_dir_ref) / "model.onnx"
                else:
                    ident = f"hf://{model_dir_ref}/model.onnx"
                import os as _os

                _off = str(_os.getenv("HF_HUB_OFFLINE", "")).lower() not in (
                    "",
                    "0",
                    "false",
                ) or str(_os.getenv("TRANSFORMERS_OFFLINE", "")).lower() not in ("", "0", "false")
                model_path = ensure_onnx_model(ident, local_files_only=_off)
                self._text_session = create_onnx_session(model_path)
                self._text_pipeline = None
            except Exception as e:
                logger.warning(f"Text emotion model unavailable ({e}); will use fallback")
                self._text_session = None
                self._text_tokenizer = None
                self._text_pipeline = None

    def _infer_text(self, text: str) -> tuple[dict[str, float], list[dict[str, float]]]:
        if text is None:
            text = ""
        self._lazy_text()
        try:
            if self.affect_backend == "onnx":
                if self._text_session is None or self._text_tokenizer is None or not text.strip():
                    raise RuntimeError("text model missing or text empty")
                enc = self._text_tokenizer(text, return_tensors="np", truncation=True)
                ort_inputs = {k: v for k, v in enc.items()}
                logits = self._text_session.run(None, ort_inputs)[0]
                if logits.ndim == 2:
                    logits = logits[0]
                probs = 1.0 / (1.0 + np.exp(-logits))
                dist = {
                    lbl: float(probs[i])
                    for i, lbl in enumerate(GOEMOTIONS_LABELS)
                    if i < len(probs)
                }
                dist = _normalize_scores(dist)
                top5 = _topk_distribution(dist, 5)
                return dist, top5
            else:
                if self._text_pipeline is None or not text.strip():
                    raise RuntimeError("text pipeline missing or text empty")
                # pipeline with top_k=None yields list of dicts with 'label','score' for all classes
                out = self._text_pipeline(text, truncation=True)[0]
                dist = {lbl: 0.0 for lbl in GOEMOTIONS_LABELS}
                for item in out:
                    lab = str(item["label"]).lower().strip()
                    if lab in dist:
                        dist[lab] = float(item["score"])
                # normalize to 1.0 for stability (multi-label but we want comparable numbers)
                dist = _normalize_scores(dist)
                top5 = _topk_distribution(dist, 5)
                return dist, top5
        except Exception:
            # keyword fallback
            dist = self._text_keyword_fallback(text)
            dist = _normalize_scores(dist)
            top5 = _topk_distribution(dist, 5)
            return dist, top5

    @staticmethod
    def _text_keyword_fallback(text: str) -> dict[str, float]:
        t = f" {text.lower()} "
        buckets = {
            "joy": [" happy ", " excited ", " wonderful ", " amazing ", " love "],
            "sadness": [" sad ", " down ", " depressed ", " upset "],
            "anger": [" angry ", " mad ", " furious ", " annoyed ", " hate "],
            "fear": [" scared ", " afraid ", " worried ", " nervous ", " anxious "],
            "surprise": [" wow ", " unexpected ", " shocked ", " surprised "],
            "disgust": [
                " disgusting ",
                " awful ",
                " terrible ",
                " horrible ",
                " gross ",
            ],
            "neutral": [" okay ", " fine ", " alright ", " normal "],
        }
        scores = {k: 0.0 for k in buckets}
        for k, kws in buckets.items():
            for w in kws:
                if w in t:
                    scores[k] += 1.0
        # map to 28-class roughly
        dist = {lbl: 0.0 for lbl in GOEMOTIONS_LABELS}
        if scores["joy"]:
            for lbl in [
                "joy",
                "admiration",
                "amusement",
                "approval",
                "optimism",
                "gratitude",
                "excitement",
                "love",
                "pride",
                "relief",
                "caring",
            ]:
                if lbl in dist:
                    dist[lbl] += scores["joy"]
        if scores["sadness"]:
            for lbl in ["sadness", "grief", "remorse", "disappointment"]:
                dist[lbl] += scores["sadness"]
        if scores["anger"]:
            for lbl in ["anger", "annoyance", "disapproval", "disgust"]:
                dist[lbl] += scores["anger"]
        if scores["fear"]:
            for lbl in ["fear", "nervousness"]:
                dist[lbl] += scores["fear"]
        if scores["surprise"]:
            for lbl in ["surprise", "realization"]:
                dist[lbl] += scores["surprise"]
        if scores["disgust"]:
            dist["disgust"] += scores["disgust"]
        if scores["neutral"] or sum(scores.values()) == 0:
            dist["neutral"] += max(1.0, scores["neutral"])
        return dist

    # -------- Intent (zero-shot or rule-based) --------
    @staticmethod
    def _intent_onnx_candidates(model_dir: str | Path) -> list[str | Path]:
        base = Path(model_dir)
        names = ("model_uint8.onnx", "model_int8.onnx", "model.onnx")
        candidates: list[str | Path] = []
        if base.exists():
            for name in names:
                candidate = base / name
                if candidate.exists():
                    candidates.append(candidate)
            if not candidates:
                candidates.append(base / names[0])
            return candidates

        base_str = str(model_dir).rstrip("/\\")
        if not base_str:
            return []
        if base_str.startswith("hf://"):
            prefix = base_str
        else:
            prefix = f"hf://{base_str}"
        for name in names:
            candidates.append(f"{prefix}/{name}")
        return candidates

    def _lazy_intent(self):
        model_name = self.affect_intent_model_dir or self.intent_model_name
        if self.affect_backend == "onnx":
            # Fast path: if the configured model_name points at a local directory
            # containing ONNX exports, try ONNX candidates first and avoid
            # attempting to instantiate the (potentially broken) tokenizer
            # which can raise parsing errors for malformed tokenizer.json files.
            if Path(model_name).exists():
                try:
                    _off = str(__import__("os").getenv("HF_HUB_OFFLINE", "")).lower() not in (
                        "",
                        "0",
                        "false",
                    ) or str(__import__("os").getenv("TRANSFORMERS_OFFLINE", "")).lower() not in (
                        "",
                        "0",
                        "false",
                    )
                    candidates = self._intent_onnx_candidates(model_name)
                    last_exc = None
                    for ident in candidates:
                        try:
                            mp = ensure_onnx_model(ident, local_files_only=_off)
                            self._intent_session = create_onnx_session(mp)
                            last_exc = None
                            break
                        except Exception as e:
                            last_exc = e
                            continue
                    if self._intent_session is not None:
                        # Successfully created a local ONNX session; load config and tokenizer
                        # so ONNX inference has everything it needs.
                        try:
                            from transformers import AutoConfig, AutoTokenizer

                            # Tokenizer is required for premise/hypothesis encoding.
                            # Some exported folders only contain a tokenizer.json that older
                            # `tokenizers` builds cannot parse. Try fast first, then slow.
                            try:
                                self._intent_tokenizer = AutoTokenizer.from_pretrained(model_name)
                            except Exception:
                                # Attempt slow tokenizer that uses vocab.json/merges.txt
                                self._intent_tokenizer = AutoTokenizer.from_pretrained(
                                    model_name, use_fast=False
                                )

                            cfg = AutoConfig.from_pretrained(model_name)
                            template = getattr(cfg, "hypothesis_template", None)
                            if template:
                                self._intent_hypothesis_template = str(template)
                            label_map = {}
                            for key, value in getattr(cfg, "id2label", {}).items():
                                try:
                                    idx = int(key)
                                except Exception:
                                    continue
                                label_map[idx] = str(value).lower()
                            if not label_map:
                                label_map = {
                                    idx: label.lower()
                                    for idx, label in enumerate(
                                        ["contradiction", "neutral", "entailment"]
                                    )
                                }
                            entail_idx = next(
                                (i for i, lab in label_map.items() if "entail" in lab), None
                            )
                            contra_idx = next(
                                (i for i, lab in label_map.items() if "contradict" in lab), None
                            )
                            if entail_idx is None or contra_idx is None:
                                raise RuntimeError(
                                    "intent ONNX labels missing entailment/contradiction"
                                )
                            self._intent_entail_idx = entail_idx
                            self._intent_contra_idx = contra_idx
                            self._intent_pipeline = None
                            return
                        except Exception:
                            # If tokenizer/config read fails, we still keep the ONNX session
                            # but mark ONNX intent as unavailable to trigger safe fallback later.
                            self._intent_entail_idx = None
                            self._intent_contra_idx = None
                            self._intent_tokenizer = None
                            self._intent_pipeline = None
                            return
                    if self._intent_session is None and last_exc is not None:
                        # Fall through to the original ONNX/transformers flow and log
                        logger.warning(
                            "Intent ONNX model unavailable (%s); falling back to transformers for intent only",
                            last_exc,
                        )
                        self._intent_session = None
                        self._intent_tokenizer = None
                        self._intent_entail_idx = None
                        self._intent_contra_idx = None
                except Exception as e:
                    logger.warning(
                        "Intent ONNX candidate scan failed (%s); falling back to transformers",
                        e,
                    )
                    self._intent_session = None
                    self._intent_tokenizer = None
                    self._intent_entail_idx = None
                    self._intent_contra_idx = None
                    # continue to transformers fallback below
            # Original behaviour: attempt ONNX candidate resolution (remote) then transformers
            if (
                self._intent_session is not None
                and self._intent_tokenizer is not None
                and self._intent_entail_idx is not None
                and self._intent_contra_idx is not None
            ):
                return
            try:
                from transformers import AutoConfig, AutoTokenizer

                # Prefer fast; fall back to slow if tokenizer.json is incompatible
                try:
                    self._intent_tokenizer = AutoTokenizer.from_pretrained(model_name)
                except Exception:
                    self._intent_tokenizer = AutoTokenizer.from_pretrained(
                        model_name, use_fast=False
                    )
                cfg = AutoConfig.from_pretrained(model_name)
                import os as _os

                _off = str(_os.getenv("HF_HUB_OFFLINE", "")).lower() not in (
                    "",
                    "0",
                    "false",
                ) or str(_os.getenv("TRANSFORMERS_OFFLINE", "")).lower() not in ("", "0", "false")
                # Try local/remote candidates (uint8/int8/default)
                if Path(model_name).exists():
                    candidates = self._intent_onnx_candidates(model_name)
                else:
                    candidates = self._intent_onnx_candidates(f"hf://{model_name}")
                last_exc = None
                for ident in candidates:
                    try:
                        mp = ensure_onnx_model(ident, local_files_only=_off)
                        self._intent_session = create_onnx_session(mp)
                        last_exc = None
                        break
                    except Exception as e:
                        last_exc = e
                        continue
                if self._intent_session is None and last_exc is not None:
                    raise last_exc
                template = getattr(cfg, "hypothesis_template", None)
                if template:
                    self._intent_hypothesis_template = str(template)
                label_map = {}
                for key, value in getattr(cfg, "id2label", {}).items():
                    try:
                        idx = int(key)
                    except Exception:
                        continue
                    label_map[idx] = str(value).lower()
                if not label_map:
                    label_map = {
                        idx: label.lower()
                        for idx, label in enumerate(["contradiction", "neutral", "entailment"])
                    }
                entail_idx = next(
                    (i for i, lab in label_map.items() if "entail" in lab),
                    None,
                )
                contra_idx = next(
                    (i for i, lab in label_map.items() if "contradict" in lab),
                    None,
                )
                if entail_idx is None or contra_idx is None:
                    raise RuntimeError("intent ONNX labels missing entailment/contradiction")
                self._intent_entail_idx = entail_idx
                self._intent_contra_idx = contra_idx
                self._intent_pipeline = None
                return
            except Exception as e:
                logger.warning(
                    "Intent ONNX model unavailable (%s); falling back to transformers for intent only",
                    e,
                )
                self._intent_session = None
                self._intent_tokenizer = None
                self._intent_entail_idx = None
                self._intent_contra_idx = None

        if self._intent_pipeline is not None:
            return
        try:
            from transformers import pipeline

            self._intent_pipeline = pipeline(
                "zero-shot-classification",
                model=model_name,
                tokenizer=model_name,
            )
        except Exception as e:
            logger.warning(f"Intent model unavailable ({e}); will use fallback")
            self._intent_pipeline = None

    def _infer_intent(self, text: str) -> tuple[str, list[dict[str, float]]]:
        t = (text or "").strip()
        self._lazy_intent()
        try:
            if self.affect_backend == "onnx":
                if (
                    self._intent_session is None
                    or self._intent_tokenizer is None
                    or self._intent_entail_idx is None
                    or self._intent_contra_idx is None
                    or not t
                ):
                    raise RuntimeError("intent ONNX backend unavailable or text empty")
                scores: dict[str, float] = {}
                for label in self.intent_labels:
                    hypothesis = self._intent_hypothesis_template.format(label.replace("_", " "))
                    enc = self._intent_tokenizer(
                        t,
                        hypothesis,
                        return_tensors="np",
                        truncation=True,
                    )
                    ort_inputs = {k: v for k, v in enc.items()}
                    logits = self._intent_session.run(None, ort_inputs)[0]
                    if logits.ndim == 2:
                        logits = logits[0]
                    logits = logits.astype(np.float64)
                    ex = np.exp(logits - np.max(logits))
                    probs_local = ex / (ex.sum() or 1.0)
                    entail_score = float(probs_local[self._intent_entail_idx])
                    scores[label] = entail_score
                probs = _normalize_scores(scores)
                top = max(probs, key=probs.get)
                top3 = _topk_distribution(probs, 3)
                return top, top3

            if self._intent_pipeline is None or not t:
                raise RuntimeError("intent pipeline missing or text empty")
            res = self._intent_pipeline(t, candidate_labels=self.intent_labels, multi_label=False)
            seq_labels = res["labels"]
            seq_scores = res["scores"]
            probs = {
                str(lbl): float(score) for lbl, score in zip(seq_labels, seq_scores, strict=False)
            }
            # reorder to our label list for stability & normalize
            probs = {lbl: probs.get(lbl, 0.0) for lbl in self.intent_labels}
            probs = _normalize_scores(probs)
            top = max(probs, key=probs.get)
            top3 = _topk_distribution(probs, 3)
            return top, top3
        except Exception:
            # rule fallback
            probs = self._intent_rules(t)
            probs = _normalize_scores(probs)
            top = max(probs, key=probs.get)
            top3 = _topk_distribution(probs, 3)
            return top, top3

    def _intent_rules(self, t: str) -> dict[str, float]:
        t = (t or "").lower()
        probs = {lbl: 0.0 for lbl in self.intent_labels}
        if any(q in t for q in ["?", " what ", " how ", " when ", " where ", " why ", " who "]):
            probs["question"] = 1.0
        if any(p in t for p in [" please ", " can you ", " could you ", " would you "]):
            probs["request"] = max(probs["request"], 1.0)
        if any(g in t for g in [" hi ", " hello ", " hey ", " good morning ", " good afternoon "]):
            probs["greeting"] = 1.0
        if any(b in t for b in [" bye ", " goodbye ", " see you ", " talk later "]):
            probs["farewell"] = 1.0
        if any(a in t for a in [" sorry ", " apologize ", " my bad "]):
            probs["apology"] = 1.0
        if any(c in t for c in ["thanks", " thank you ", " appreciate "]):
            probs["gratitude"] = 1.0
        if any(k in t for k in [" i think ", " i feel ", " in my opinion ", " imo "]):
            probs["opinion"] = 1.0
        if any(k in t for k in [" yes ", " yeah ", " correct ", " agreed ", " agree "]):
            probs["agreement"] = 1.0
        if any(k in t for k in [" no ", " not really ", " disagree ", " incorrect "]):
            probs["disagreement"] = 1.0
        if any(k in t for k in [" you should ", " we should ", " i suggest ", " recommendation "]):
            probs["suggestion"] = 1.0
        if any(k in t for k in [" do this ", " step ", " first ", " then "]) and "?" not in t:
            probs["instruction"] = 0.7
        if any(k in t for k in [" now ", " immediately ", " must ", " need to "]):
            probs["command"] = max(probs["command"], 0.5)
        # fallbacks
        if max(probs.values() or [0.0]) == 0.0:
            probs["status_update"] = 1.0
        return probs

    # -------- Cross-modal hint --------
    def _polarity_sums(self, dist: dict[str, float]) -> dict[str, float]:
        pos = sum(dist.get(label, 0.0) for label in GOEMOTIONS_POSITIVE)
        neg = sum(dist.get(label, 0.0) for label in GOEMOTIONS_NEGATIVE)
        amb = sum(dist.get(label, 0.0) for label in GOEMOTIONS_AMBIGUOUS)
        neu = dist.get("neutral", 0.0)
        total = pos + neg + amb + neu or 1.0
        return {
            "positive": float(pos / total),
            "negative": float(neg / total),
            "ambiguous": float(amb / total),
            "neutral": float(neu / total),
        }

    def _affect_hint(
        self, v: float, a: float, ser_probs: dict[str, float], pol: dict[str, float]
    ) -> str:
        baseline = "neutral-status"

        # SER polarity buckets
        ser_pos = ser_probs.get("happy", 0.0) + ser_probs.get("surprised", 0.0)
        ser_neg = (
            ser_probs.get("angry", 0.0)
            + ser_probs.get("sad", 0.0)
            + ser_probs.get("fearful", 0.0)
            + ser_probs.get("disgusted", 0.0)
        )
        ser_neu = ser_probs.get("neutral", 0.0) + ser_probs.get("calm", 0.0)
        ser_label = max(
            {"positive": ser_pos, "negative": ser_neg, "neutral": ser_neu},
            key=lambda k: {
                "positive": ser_pos,
                "negative": ser_neg,
                "neutral": ser_neu,
            }[k],
        )

        # Text polarity
        text_label = max(pol, key=pol.get) if pol else "neutral"
        text_conf = pol.get(text_label, 0.0)

        if text_conf < 0.45:
            return baseline
        if ser_label == "neutral" and text_label == "neutral":
            return baseline
        if ser_label == "neutral" and text_label != "neutral":
            return f"text-dominant-{text_label}"

        v_sign_text = 1 if text_label == "positive" else (-1 if text_label == "negative" else 0)
        v_sign_audio = 1 if v > 0.15 else (-1 if v < -0.15 else 0)
        aligned = (v_sign_text == v_sign_audio) or (v_sign_text == 0 and abs(v) < 0.15)
        polarity_tag = text_label if text_label in ("positive", "negative") else ser_label
        return (
            f"affect-convergent-{polarity_tag}" if aligned else f"affect-divergent-{polarity_tag}"
        )


# End class


# Convenience: batch adapter (kept for compatibility)
def analyze_segment_batch(analyzer: EmotionIntentAnalyzer, segments: list[dict]) -> list[dict]:
    out = []
    for seg in segments:
        try:
            res = analyzer.analyze(
                seg.get("audio", np.zeros(0, dtype=np.float32)),
                int(seg.get("sr", 16000)),
                seg.get("text", ""),
            )
            out.append(res)
        except Exception as e:
            logger.error(f"segment failed: {e}")
            out.append(
                {
                    "vad": {"valence": 0.0, "arousal": 0.0, "dominance": 0.0},
                    "speech_emotion": {
                        "top": "neutral",
                        "scores_8class": _normalize_scores(
                            {lbl: (1.0 if lbl == "neutral" else 0.0) for lbl in SER_LABELS_8}
                        ),
                        "low_confidence_ser": True,
                    },
                    "text_emotions": {
                        "top5": [{"label": "neutral", "score": 1.0}],
                        "full_28class": {"neutral": 1.0},
                    },
                    "intent": {
                        "top": "status_update",
                        "top3": [{"label": "status_update", "score": 1.0}],
                    },
                    "affect_hint": "none",
                }
            )
    return out


if __name__ == "__main__":
    # Smoke test (no model downloads here)
    print("updated_emotion_analyzer.py ready (CPU-only).")
