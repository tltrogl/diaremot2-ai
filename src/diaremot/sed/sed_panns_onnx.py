"""Sound event detection using PANNs CNN14 exported to ONNX.

The implementation mirrors the production pipeline defaults:

* Audio is expected at 16 kHz mono; other sample rates are resampled via ``librosa.resample``.
* Frames are carved with a 1.0 s window and 0.5 s hop (configurable).
* Each frame is transformed into a 64-bin log-mel spectrogram. We normalise per-frame
  by subtracting the mean and dividing by the standard deviation computed along the
  time axis for each mel bin (i.e., statistics per frequency band).
* Posterior smoothing uses a small median filter (default width = 5 frames).
* Hysteresis thresholding converts per-class posteriors into events which are then
  merged if separated by short gaps and pruned if shorter than ``min_dur``.

A lightweight CLI is provided: ``python -m diaremot.sed.sed_panns_onnx``. It emits
``events_timeline.csv`` compatible with the pipeline expectations and optionally a
frame-level JSONL for debugging.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy.signal import medfilt

try:  # Optional runtime helper; absent during unit tests.
    from ..pipeline.runtime_env import iter_model_roots
except Exception:  # pragma: no cover - fallback for minimal installs
    def iter_model_roots() -> tuple[Path, ...]:  # type: ignore
        return (Path.cwd(),)


logger = logging.getLogger(__name__)

_MODEL_FILENAMES = (
    "Cnn14_16k.onnx",
    "cnn14_16k.onnx",
    "cnn14.onnx",
    "Cnn14.onnx",
)
_MODEL_SUBDIRS = (
    "sed_panns",
    "panns",
    "panns_cnn14",
    "cnn14",
    "sed",
    "models",
    "",
)

# fmt: off
COARSE_INDEX_TO_LABEL: dict[int, str] = {
    7: "keyboard",
    16: "laughter",
    17: "laughter",
    18: "laughter",
    19: "laughter",
    21: "laughter",
    27: "music",
    28: "music",
    30: "crowd",
    32: "music",
    33: "music",
    34: "music",
    35: "music",
    53: "footsteps",
    63: "applause",
    66: "crowd",
    67: "applause",
    69: "crowd",
    74: "barking",
    75: "barking",
    77: "barking",
    80: "barking",
    137: "music",
    140: "music",
    141: "music",
    142: "music",
    143: "music",
    144: "music",
    145: "music",
    147: "music",
    148: "music",
    149: "music",
    151: "music",
    152: "keyboard",
    153: "keyboard",
    154: "keyboard",
    155: "keyboard",
    156: "keyboard",
    157: "keyboard",
    158: "keyboard",
    160: "keyboard",
    161: "music",
    162: "music",
    163: "music",
    164: "music",
    165: "music",
    167: "music",
    168: "music",
    169: "music",
    170: "music",
    171: "music",
    179: "music",
    180: "music",
    184: "music",
    186: "music",
    187: "music",
    188: "music",
    191: "music",
    193: "music",
    194: "music",
    196: "music",
    197: "music",
    198: "music",
    199: "music",
    203: "vehicle",
    206: "wind",
    208: "music",
    209: "music",
    214: "music",
    216: "music",
    217: "music",
    218: "music",
    219: "music",
    220: "music",
    226: "music",
    227: "music",
    228: "music",
    230: "music",
    231: "music",
    233: "music",
    234: "music",
    235: "music",
    236: "music",
    237: "music",
    238: "music",
    239: "music",
    240: "music",
    241: "music",
    242: "music",
    243: "music",
    245: "music",
    246: "music",
    247: "music",
    248: "music",
    249: "music",
    250: "music",
    251: "music",
    252: "music",
    253: "music",
    254: "music",
    256: "music",
    258: "music",
    259: "music",
    260: "music",
    261: "music",
    262: "music",
    263: "music",
    264: "music",
    265: "music",
    267: "music",
    268: "music",
    269: "music",
    270: "music",
    272: "music",
    273: "music",
    274: "music",
    275: "music",
    276: "music",
    277: "music",
    278: "music",
    279: "music",
    280: "music",
    281: "music",
    282: "music",
    283: "wind",
    285: "wind",
    288: "water",
    289: "water",
    291: "water",
    292: "water",
    293: "water",
    294: "water",
    300: "vehicle",
    301: "vehicle",
    302: "vehicle",
    305: "vehicle",
    306: "vehicle",
    307: "vehicle",
    308: "vehicle",
    310: "vehicle",
    311: "wind",
    313: "vehicle",
    314: "vehicle",
    315: "vehicle",
    316: "vehicle",
    318: "vehicle",
    320: "vehicle",
    321: "vehicle",
    322: "vehicle",
    323: "vehicle",
    324: "siren_alarm",
    325: "vehicle",
    326: "vehicle",
    327: "vehicle",
    329: "vehicle",
    330: "vehicle",
    331: "vehicle",
    332: "vehicle",
    333: "vehicle",
    334: "vehicle",
    336: "engine",
    337: "vehicle",
    338: "engine",
    339: "vehicle",
    340: "vehicle",
    341: "vehicle",
    342: "vehicle",
    343: "engine",
    344: "engine",
    348: "engine",
    349: "engine",
    350: "engine",
    351: "engine",
    352: "engine",
    353: "engine",
    354: "door",
    355: "door",
    357: "door",
    358: "door",
    359: "door",
    360: "water",
    364: "cooking",
    365: "cooking",
    366: "cooking",
    367: "cooking",
    368: "cooking",
    369: "cooking",
    370: "water",
    371: "water",
    384: "typing",
    385: "typing",
    386: "typing",
    388: "siren_alarm",
    389: "phone",
    390: "phone",
    391: "phone",
    392: "phone",
    393: "phone",
    395: "siren_alarm",
    396: "siren_alarm",
    397: "siren_alarm",
    398: "siren_alarm",
    399: "siren_alarm",
    400: "siren_alarm",
    426: "impact",
    445: "water",
    448: "water",
    456: "water",
    460: "impact",
    466: "impact",
    467: "impact",
    469: "impact",
    470: "impact",
    486: "door",
    490: "cooking",
    491: "mouse_click",
    492: "mouse_click",
    501: "water",
    524: "tv",
}
# fmt: on

COARSE_LABELS: tuple[str, ...] = (
    "music",
    "keyboard",
    "door",
    "tv",
    "phone",
    "vehicle",
    "siren_alarm",
    "laughter",
    "footsteps",
    "impact",
    "barking",
    "wind",
    "water",
    "crowd",
    "engine",
    "cooking",
    "typing",
    "mouse_click",
    "applause",
    "other_env",
)


@dataclass(slots=True)
class _FrameResult:
    start: float
    end: float
    topk: list[tuple[str, float]]


def _resolve_model_path(model_path: str | Path | None) -> Path:
    if model_path is not None:
        path = Path(model_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"PANNs ONNX model not found at {path}. Set --model-path or place cnn14.onnx "
                "under the models directory."
            )
        return path

    candidates: list[Path] = []
    module_root = Path(__file__).resolve().parent
    for name in _MODEL_FILENAMES:
        candidates.append(module_root / name)
    for root in iter_model_roots():
        for subdir in _MODEL_SUBDIRS:
            for name in _MODEL_FILENAMES:
                candidates.append(Path(root) / subdir / name)

    for cand in candidates:
        if cand.exists():
            return cand.resolve()

    raise FileNotFoundError(
        "Unable to locate the CNN14 ONNX model. Download the DiaRemot model bundle "
        "(models.zip) or pass --model-path explicitly."
    )


def _load_audio(path: Path, sr_target: int) -> tuple[np.ndarray, int]:
    y, sr = sf.read(path, always_2d=False)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    y = y.astype(np.float32, copy=False)
    if sr != sr_target:
        y = librosa.resample(y, orig_sr=sr, target_sr=sr_target)
        sr = sr_target
    return y, sr


def _frame_audio(y: np.ndarray, frame_len: int, hop_len: int) -> list[np.ndarray]:
    if y.size == 0:
        return []
    frames: list[np.ndarray] = []
    total = len(y)
    for start in range(0, max(total - frame_len + 1, 1), hop_len):
        end = start + frame_len
        if end > total:
            segment = np.zeros(frame_len, dtype=np.float32)
            segment[: total - start] = y[start:]
        else:
            segment = y[start:end]
        frames.append(segment.astype(np.float32, copy=False))
        if end >= total:
            break
    return frames


def _compute_logmel(
    frame: np.ndarray,
    sr: int,
    mel_bins: int,
    n_fft: int = 1024,
    hop_length: int | None = None,
    fmin: float = 50.0,
    fmax: float | None = None,
) -> np.ndarray:
    hop = hop_length or 320
    mel = librosa.feature.melspectrogram(
        y=frame,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window="hann",
        center=False,
        power=2.0,
        n_mels=mel_bins,
        fmin=fmin,
        fmax=fmax or sr / 2.0,
        htk=False,
    )
    log_mel = np.log(mel + 1.0e-6)
    mean = log_mel.mean(axis=1, keepdims=True)
    std = log_mel.std(axis=1, keepdims=True)
    log_mel = (log_mel - mean) / (std + 1.0e-6)
    return log_mel.astype(np.float32)


def _prepare_input(batch: np.ndarray, input_shape: Sequence[int | None]) -> np.ndarray:
    rank = len(input_shape)
    arr = batch.astype(np.float32, copy=False)
    if rank == 4:
        c_axis = input_shape[1]
        h_axis = input_shape[2]
        w_axis = input_shape[3]
        if c_axis in (1, None):
            if h_axis == arr.shape[1] or h_axis is None:
                arr = arr[:, np.newaxis, :, :]
            elif w_axis == arr.shape[1] or w_axis is None:
                arr = arr[:, np.newaxis, :, :].transpose(0, 1, 3, 2)
            else:
                raise ValueError(f"Unexpected ONNX input shape {input_shape} for log-mel tensor")
        elif c_axis == arr.shape[1]:
            pass
        else:
            raise ValueError(f"Unsupported channel dimension in ONNX input: {input_shape}")
    elif rank == 3:
        h_axis = input_shape[1]
        w_axis = input_shape[2]
        if h_axis == arr.shape[1] or h_axis is None:
            pass
        elif w_axis == arr.shape[1] or w_axis is None:
            arr = arr.transpose(0, 2, 1)
        else:
            raise ValueError(f"Unexpected ONNX input shape {input_shape}")
    else:
        raise ValueError(f"Unsupported ONNX input rank {rank}")
    return arr


def _median_filter(probabilities: np.ndarray, kernel: int) -> np.ndarray:
    if kernel <= 1:
        return probabilities
    if kernel % 2 == 0:
        kernel += 1
    filtered = np.empty_like(probabilities)
    for class_idx in range(probabilities.shape[1]):
        filtered[:, class_idx] = medfilt(probabilities[:, class_idx], kernel_size=kernel)
    return filtered


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _collapse_labels(
    frame_posteriors: np.ndarray,
    mapping: dict[int, str],
    topk: int,
) -> tuple[list[list[tuple[str, float]]], list[dict[str, float]]]:
    num_classes = frame_posteriors.shape[1]
    label_map = {idx: mapping.get(idx, "other_env") for idx in range(num_classes)}
    frame_topk: list[list[tuple[str, float]]] = []
    frame_scores: list[dict[str, float]] = []
    for frame in frame_posteriors:
        scores: dict[str, float] = {label: 0.0 for label in COARSE_LABELS}
        scores.setdefault("applause", 0.0)  # ensure explicit label present
        scores.setdefault("other_env", 0.0)
        for idx, value in enumerate(frame):
            label = label_map[idx]
            prev = scores.get(label, 0.0)
            if value > prev:
                scores[label] = float(value)
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        frame_topk.append(ordered[:topk])
        frame_scores.append(scores)
    return frame_topk, frame_scores


def _hysteresis_runs(
    probs: Sequence[float],
    enter: float,
    exit: float,
    min_dur: float,
    merge_gap: float,
    frame_sec: float,
    hop_sec: float,
) -> list[tuple[int, int, float]]:
    active = False
    start_idx = 0
    max_score = 0.0
    raw_runs: list[tuple[int, int, float]] = []
    for idx, value in enumerate(probs):
        if not active and value >= enter:
            active = True
            start_idx = idx
            max_score = float(value)
            continue
        if active:
            max_score = max(max_score, float(value))
            if value <= exit:
                active = False
                raw_runs.append((start_idx, idx, max_score))
    if active:
        raw_runs.append((start_idx, len(probs), max_score))

    if not raw_runs:
        return []

    merged: list[tuple[int, int, float]] = []
    for start, end, score in raw_runs:
        if not merged:
            merged.append((start, end, score))
            continue
        prev_start, prev_end, prev_score = merged[-1]
        prev_end_time = (prev_end - 1) * hop_sec + frame_sec if prev_end > prev_start else prev_start * hop_sec
        start_time = start * hop_sec
        if start_time - prev_end_time <= merge_gap:
            merged[-1] = (prev_start, end, max(prev_score, score))
        else:
            merged.append((start, end, score))

    results: list[tuple[int, int, float]] = []
    for start, end, score in merged:
        if end <= start:
            continue
        start_time = start * hop_sec
        end_time = (end - 1) * hop_sec + frame_sec
        if end_time - start_time >= min_dur:
            results.append((start, end, score))
    return results


def _run_inference(
    session: ort.InferenceSession,
    features: np.ndarray,
    batch_size: int = 16,
) -> np.ndarray:
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    input_shape = tuple(input_meta.shape)
    outputs: list[np.ndarray] = []
    for start in range(0, len(features), batch_size):
        batch = features[start : start + batch_size]
        tensor = _prepare_input(batch, input_shape)
        logits = session.run(None, {input_name: tensor})[0]
        outputs.append(np.asarray(logits, dtype=np.float32))
    return np.concatenate(outputs, axis=0)


def _build_frame_results(
    topk: list[list[tuple[str, float]]],
    frame_sec: float,
    hop_sec: float,
    total_duration: float,
) -> list[_FrameResult]:
    frames: list[_FrameResult] = []
    for idx, choices in enumerate(topk):
        start = idx * hop_sec
        end = min(start + frame_sec, total_duration)
        frames.append(_FrameResult(start=start, end=end, topk=choices))
    return frames


def _run_sed_internal(
    wav_path: Path,
    *,
    sr_target: int,
    frame_sec: float,
    hop_sec: float,
    mel_bins: int,
    median_size: int,
    enter_thresh: float,
    exit_thresh: float,
    min_dur: float,
    merge_gap: float,
    topk: int,
    threads: int | None,
    model_path: str | Path | None,
    return_frames: bool,
) -> tuple[list[dict[str, float]], list[_FrameResult]]:
    start_time = time.perf_counter()
    model = _resolve_model_path(model_path)
    sess_options = ort.SessionOptions()
    if threads:
        sess_options.intra_op_num_threads = int(threads)
    session = ort.InferenceSession(str(model), sess_options=sess_options, providers=["CPUExecutionProvider"])

    audio, sr = _load_audio(wav_path, sr_target)
    total_duration = len(audio) / float(sr) if sr else 0.0
    if len(audio) < int(frame_sec * sr):
        logger.warning("Audio shorter than one frame; no events emitted.")
        return [], []

    frame_len = int(round(frame_sec * sr))
    hop_len = max(1, int(round(hop_sec * sr)))
    frames = _frame_audio(audio, frame_len, hop_len)
    if not frames:
        logger.warning("No frames extracted; no events emitted.")
        return [], []

    features = np.stack([
        _compute_logmel(frame, sr=sr, mel_bins=mel_bins)
        for frame in frames
    ])

    logits = _run_inference(session, features)
    probs = _sigmoid(logits)
    probs = _median_filter(probs, median_size)

    frame_topk, frame_scores = _collapse_labels(probs, COARSE_INDEX_TO_LABEL, topk)
    label_series: dict[str, np.ndarray] = {
        label: np.array([scores.get(label, 0.0) for scores in frame_scores], dtype=np.float32)
        for label in COARSE_LABELS
    }

    events: list[dict[str, float]] = []
    for label, series in label_series.items():
        if label == "other_env":
            continue
        runs = _hysteresis_runs(
            series,
            enter=enter_thresh,
            exit=exit_thresh,
            min_dur=min_dur,
            merge_gap=merge_gap,
            frame_sec=frame_sec,
            hop_sec=hop_sec,
        )
        for start_idx, end_idx, score in runs:
            start = start_idx * hop_sec
            end = min((end_idx - 1) * hop_sec + frame_sec, total_duration)
            events.append({
                "start": round(start, 6),
                "end": round(end, 6),
                "label": label,
                "score": float(score),
            })

    events.sort(key=lambda item: (item["start"], item["label"]))
    elapsed = time.perf_counter() - start_time
    logger.info(
        "Processed %s frames in %.2fs (audio %.2fs)",
        len(frames),
        elapsed,
        total_duration,
    )

    frame_results = _build_frame_results(frame_topk, frame_sec, hop_sec, total_duration) if return_frames else []
    return events, frame_results


def run_sed(
    wav_path: str,
    *,
    sr_target: int = 16000,
    frame_sec: float = 1.0,
    hop_sec: float = 0.5,
    mel_bins: int = 64,
    median_size: int = 5,
    enter_thresh: float = 0.50,
    exit_thresh: float = 0.35,
    min_dur: float = 0.30,
    merge_gap: float = 0.20,
    topk: int = 3,
    threads: int | None = None,
    model_path: str | None = None,
) -> list[dict[str, float]]:
    """Return merged sound events for ``wav_path``.

    Parameters
    ----------
    wav_path:
        Input audio file (mono recommended). The file is resampled to ``sr_target`` if needed.
    sr_target:
        Target sampling rate for feature extraction (default ``16 kHz``).
    frame_sec / hop_sec:
        Sliding window parameters in seconds.
    mel_bins:
        Number of mel filter banks.
    median_size:
        Width of the temporal median filter applied to per-class posteriors.
    enter_thresh / exit_thresh:
        Hysteresis thresholds (probability space) for entering/exiting an event.
    min_dur:
        Minimum event duration in seconds after merging small gaps.
    merge_gap:
        Maximum silence allowed between two runs for them to be merged.
    topk:
        Number of coarse labels to keep per frame for diagnostics.
    threads:
        Optional intra-op thread count for ONNX Runtime.
    model_path:
        Optional explicit path to the CNN14 ONNX model.
    """

    events, _ = _run_sed_internal(
        Path(wav_path),
        sr_target=sr_target,
        frame_sec=frame_sec,
        hop_sec=hop_sec,
        mel_bins=mel_bins,
        median_size=median_size,
        enter_thresh=enter_thresh,
        exit_thresh=exit_thresh,
        min_dur=min_dur,
        merge_gap=merge_gap,
        topk=topk,
        threads=threads,
        model_path=model_path,
        return_frames=False,
    )
    return events


def _write_csv(events: Iterable[dict[str, float]], csv_path: Path, file_id: str) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file_id", "start", "end", "label", "score"])
        for event in events:
            writer.writerow(
                [
                    file_id,
                    f"{event['start']:.3f}",
                    f"{event['end']:.3f}",
                    event["label"],
                    f"{event['score']:.6f}",
                ]
            )


def _write_debug_jsonl(frames: Sequence[_FrameResult], jsonl_path: Path, file_id: str) -> None:
    if not frames:
        return
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for idx, frame in enumerate(frames):
            payload = {
                "file_id": file_id,
                "frame_index": idx,
                "start": round(frame.start, 6),
                "end": round(frame.end, 6),
                "topk": [{"label": label, "score": float(score)} for label, score in frame.topk],
            }
            handle.write(json.dumps(payload) + "\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sound event detection via PANNs CNN14 (ONNX).")
    parser.add_argument("input", type=Path, help="Input WAV/FLAC path.")
    parser.add_argument("output", type=Path, help="Output CSV path (events_timeline.csv).")
    parser.add_argument("--frame", dest="frame_sec", type=float, default=1.0, help="Frame length in seconds (default: 1.0).")
    parser.add_argument("--hop", dest="hop_sec", type=float, default=0.5, help="Hop length in seconds (default: 0.5).")
    parser.add_argument("--mel", dest="mel_bins", type=int, default=64, help="Number of mel bins (default: 64).")
    parser.add_argument("--median", dest="median_size", type=int, default=5, help="Median filter width in frames (odd, default: 5).")
    parser.add_argument("--enter", dest="enter_thresh", type=float, default=0.50, help="Enter threshold (default: 0.50).")
    parser.add_argument("--exit", dest="exit_thresh", type=float, default=0.35, help="Exit threshold (default: 0.35).")
    parser.add_argument("--min-dur", dest="min_dur", type=float, default=0.30, help="Minimum event duration in seconds (default: 0.30).")
    parser.add_argument("--merge-gap", dest="merge_gap", type=float, default=0.20, help="Maximum merge gap in seconds (default: 0.20).")
    parser.add_argument("--topk", dest="topk", type=int, default=3, help="Top-K labels to retain per frame (default: 3).")
    parser.add_argument("--threads", dest="threads", type=int, default=None, help="Optional intra-op thread count for onnxruntime.")
    parser.add_argument("--model-path", dest="model_path", type=Path, default=None, help="Explicit path to cnn14.onnx.")
    parser.add_argument("--backend", choices=["panns", "yamnet"], default="panns", help="Backend to use (default: panns).")
    parser.add_argument("--debug-jsonl", dest="debug_jsonl", type=Path, default=None, help="Optional frame-level JSONL dump.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    if args.backend == "yamnet":
        from . import sed_yamnet_tf  # Lazy import to avoid TensorFlow costs

        events = sed_yamnet_tf.run_sed(
            str(args.input),
            sr_target=16000,
            frame_sec=args.frame_sec,
            hop_sec=args.hop_sec,
            enter_thresh=args.enter_thresh,
            exit_thresh=args.exit_thresh,
            min_dur=args.min_dur,
            merge_gap=args.merge_gap,
            topk=args.topk,
        )
        frame_results: list[_FrameResult] = []
    else:
        events, frames = _run_sed_internal(
            args.input,
            sr_target=16000,
            frame_sec=args.frame_sec,
            hop_sec=args.hop_sec,
            mel_bins=args.mel_bins,
            median_size=args.median_size,
            enter_thresh=args.enter_thresh,
            exit_thresh=args.exit_thresh,
            min_dur=args.min_dur,
            merge_gap=args.merge_gap,
            topk=args.topk,
            threads=args.threads,
            model_path=args.model_path,
            return_frames=args.debug_jsonl is not None,
        )
        frame_results = frames

    file_id = args.input.stem
    _write_csv(events, args.output, file_id=file_id)
    if args.debug_jsonl:
        _write_debug_jsonl(frame_results, args.debug_jsonl, file_id=file_id)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
