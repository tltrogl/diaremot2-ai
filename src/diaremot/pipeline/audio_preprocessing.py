# audio_preprocessing.py with integrated auto-chunking for long files
# Extended version that automatically splits very long audio files
# and processes them in manageable chunks to prevent memory issues

from __future__ import annotations
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, filtfilt
from scipy.ndimage import median_filter
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
from pathlib import Path
import warnings
import logging
import tempfile
import time
import os
import subprocess

logger = logging.getLogger(__name__)

# Quieter logs from third-parties
warnings.filterwarnings("ignore", category=UserWarning, module="librosa.core.audio")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa.core.audio")

# ---------------------------
# Dataclasses / Config
# ---------------------------


@dataclass
class PreprocessConfig:
    target_sr: int = 16000
    mono: bool = True

    # Auto-chunking for long audio files
    auto_chunk_enabled: bool = True
    chunk_threshold_minutes: float = 30.0  # Split audio longer than this
    chunk_size_minutes: float = 20.0  # Each chunk duration
    chunk_overlap_seconds: float = 30.0  # Overlap between chunks
    chunk_temp_dir: Optional[str] = None  # Use system temp if None

    # High-pass
    hpf_hz: float = 80.0
    hpf_order: int = 2

    # Denoise (soft spectral subtraction with temporal smoothing + backoff)
    denoise: str = "spectral_sub_soft"  # "spectral_sub_soft" | "none"
    denoise_alpha_db: float = 3.0  # over-subtraction in dB
    denoise_beta: float = 0.06  # spectral floor as fraction of noise (0..1)
    mask_exponent: float = 1.0  # 1 ~ Wiener-ish
    smooth_t: int = 3  # median smoothing width (frames) for mask
    high_clip_backoff: float = 0.12  # backoff if floor_clipping_ratio exceeds this

    # Noise tracking
    noise_update_alpha: float = 0.10  # EMA for noise profile updates (lower = smoother)
    min_noise_frames: int = 30  # min non-speech frames to trust VAD noise estimate

    # VAD (RMS-gated; CPU-friendly)
    use_vad: bool = True
    frame_ms: int = 20
    hop_ms: int = 10
    vad_rel_db: float = 12.0  # speech if rms_db > noise_floor_db + vad_rel_db
    vad_floor_percentile: float = 20.0

    # Gated upward gain
    gate_db: float = -45.0  # below this, do not boost
    target_db: float = -23.0  # aim per-frame towards this
    max_boost_db: float = 18.0  # cap upward gain
    gain_smooth_ms: int = 250
    gain_smooth_method: str = "hann"  # "hann" | "exp"
    exp_smooth_alpha: float = 0.15

    # Compression (transparent)
    comp_ratio: float = 2.0
    comp_thresh_db: float = -26.0
    comp_knee_db: float = 6.0

    # Loudness norm (approximate)
    loudness_mode: str = "asr"  # "asr" -> hotter (-20 LUFS equiv), "broadcast" -> -23
    lufs_target_asr: float = -20.0
    lufs_target_broadcast: float = -23.0

    # QC / metrics
    oversample_factor: int = 4  # for intersample peak check
    silence_db: float = -60.0  # below counts as silence


@dataclass
class AudioHealth:
    snr_db: float
    clipping_detected: bool
    silence_ratio: float
    rms_db: float
    est_lufs: float
    dynamic_range_db: float
    floor_clipping_ratio: float
    is_chunked: bool = False
    chunk_info: Optional[Dict] = None


@dataclass
class ChunkInfo:
    chunk_id: int
    start_time: float
    end_time: float
    duration: float
    overlap_start: float
    overlap_end: float
    temp_path: str


# ---------------------------
# Utility helpers
# ---------------------------


PCM_FORMATS = {"WAV", "WAVEX", "AIFF", "AIFFC"}
PCM_FALLBACK_SUBTYPES = {"PCM", "FLOAT", "DOUBLE"}


def _is_uncompressed_pcm(info: sf.Info) -> bool:
    """Return True when the file is a WAV/AIFF PCM variant we can read directly."""

    try:
        fmt = (info.format or "").upper()
        subtype = (info.subtype or "").upper()
    except AttributeError:
        return False

    if fmt not in PCM_FORMATS:
        return False

    return any(token in subtype for token in PCM_FALLBACK_SUBTYPES)


def _load_uncompressed_with_soundfile(
    source: Path, target_sr: int, mono: bool
) -> Tuple[np.ndarray, int]:
    """Read PCM WAV/AIFF directly via libsndfile."""

    y, sr = sf.read(source, always_2d=False, dtype="float32")
    if mono and y.ndim > 1:
        y = np.mean(y, axis=1)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr, res_type="kaiser_best")
        sr = target_sr
    return y.astype(np.float32), sr


def _decode_with_ffmpeg(
    source: Path, target_sr: int, mono: bool
) -> Tuple[np.ndarray, int]:
    """Decode arbitrary containers via ffmpeg → temp wav → soundfile."""

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = tmp.name

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-ac",
        "1" if mono else "2",
        "-ar",
        str(target_sr),
        "-f",
        "wav",
        "-loglevel",
        "quiet",  # Suppress ffmpeg output
        tmp_wav,
    ]

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=300,  # 5 minute timeout
        )
        y, sr = sf.read(tmp_wav, always_2d=False, dtype="float32")
        if mono and y.ndim > 1:
            y = np.mean(y, axis=1)
        logger.debug(f"Decoded {source} via primary ffmpeg path")
        return y.astype(np.float32), sr
    except subprocess.TimeoutExpired as exc:
        logger.error(f"FFmpeg timeout for {source}")
        raise RuntimeError(f"Audio decoding timeout for {source}") from exc
    except subprocess.CalledProcessError as exc:
        stderr_output = exc.stderr.decode() if exc.stderr else "No error details"
        logger.debug(f"FFmpeg returned non-zero exit for {source}: {stderr_output}")
        raise RuntimeError(
            f"Cannot decode audio file {source}. FFmpeg error: {stderr_output}"
        ) from exc
    finally:
        try:
            os.remove(tmp_wav)
        except Exception:
            pass


def _safe_load_audio(
    path: str, target_sr: int, mono: bool = True
) -> Tuple[np.ndarray, int]:
    p = Path(path)

    info: Optional[sf.Info] = None
    try:
        info = sf.info(p)
    except Exception as exc_info:
        logger.debug(f"soundfile could not inspect {p}: {exc_info}")

    if info and _is_uncompressed_pcm(info):
        logger.debug(f"Loading PCM audio directly via soundfile: {p}")
        return _load_uncompressed_with_soundfile(p, target_sr=target_sr, mono=mono)

    try:
        logger.debug(f"Decoding {p} via ffmpeg → WAV")
        return _decode_with_ffmpeg(p, target_sr=target_sr, mono=mono)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg is required to decode compressed audio containers; install ffmpeg"
        ) from exc
    except RuntimeError as exc:
        raise RuntimeError(f"Cannot decode audio file {p} via ffmpeg: {exc}") from exc


def _get_audio_duration(path: str) -> float:
    """Get audio duration with robust fallback chain - FIXED VERSION"""
    try:
        info = sf.info(path)
        if info and info.duration and info.duration > 0:
            logger.debug(f"Got duration via soundfile metadata: {info.duration}s")
            return float(info.duration)
    except Exception as exc:
        logger.debug(f"soundfile failed for duration: {exc}")

    try:
        import av  # type: ignore

        with av.open(path) as container:
            dur = None
            for s in container.streams:
                if s.type == "audio" and s.duration and s.time_base:
                    dur = float(s.duration * s.time_base)
                    break
            if dur is None and container.duration is not None:
                dur = float(container.duration) / 1e6
            if dur is not None and dur > 0:
                logger.debug(f"Got duration via PyAV: {dur}s")
                return dur
    except Exception as exc_av:
        logger.debug(f"PyAV failed for duration: {exc_av}")

    try:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            duration = float(result.stdout.strip())
            logger.debug(f"Got duration via ffprobe: {duration}s")
            return duration
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        FileNotFoundError,
        ValueError,
    ) as exc_probe:
        logger.debug(f"ffprobe failed for duration: {exc_probe}")

    logger.warning(f"Could not determine duration for {path}, using 0.0")
    return 0.0


def _create_audio_chunks(audio_path: str, config: PreprocessConfig) -> List[ChunkInfo]:
    logger.info(f"[chunks] Creating audio chunks for long file: {audio_path}")

    # Load audio file info (robust to compressed containers)
    info: Optional[sf.Info] = None
    duration = _get_audio_duration(audio_path)

    try:
        info = sf.info(audio_path)
    except Exception as exc_info:
        logger.debug(f"soundfile could not inspect {audio_path}: {exc_info}")

    if info and info.samplerate:
        sr = int(info.samplerate)
    else:
        sr = int(config.target_sr)

    logger.info(
        f"[chunks] Audio duration: {duration / 60:.1f} minutes; threshold={config.chunk_threshold_minutes} min; size={config.chunk_size_minutes} min; overlap={config.chunk_overlap_seconds}s"
    )

    chunk_duration = config.chunk_size_minutes * 60.0
    overlap_duration = config.chunk_overlap_seconds

    # Create temp directory for chunks
    if config.chunk_temp_dir:
        temp_dir = Path(config.chunk_temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = Path(tempfile.mkdtemp(prefix="audio_chunks_"))

    chunks = []
    chunk_id = 0
    start_time = 0.0

    while start_time < duration:
        end_time = min(start_time + chunk_duration, duration)

        # Calculate actual start/end with overlap
        if chunk_id > 0:
            actual_start = max(0.0, start_time - overlap_duration)
        else:
            actual_start = start_time

        if end_time < duration:
            actual_end = min(duration, end_time + overlap_duration)
        else:
            actual_end = end_time

        # Extract chunk - FIXED VERSION
        try:
            t0 = time.time()
            # Try direct chunk extraction with ffmpeg (faster for M4A)
            temp_chunk_raw = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_chunk_raw.close()

            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                audio_path,
                "-ss",
                str(actual_start),
                "-t",
                str(actual_end - actual_start),
                "-acodec",
                "pcm_s16le",
                "-ar",
                str(sr),
                "-ac",
                "1",
                "-loglevel",
                "quiet",
                temp_chunk_raw.name,
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=120)
            if result.returncode == 0:
                chunk_audio, _ = sf.read(temp_chunk_raw.name, dtype="float32")
                logger.info(
                    f"[chunks] Extracted chunk {chunk_id} via ffmpeg in {time.time()-t0:.2f}s ({actual_start:.1f}s→{actual_end:.1f}s)"
                )
            else:
                raise subprocess.CalledProcessError(result.returncode, cmd)

        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            FileNotFoundError,
        ):
            if info and _is_uncompressed_pcm(info):
                logger.debug(
                    f"ffmpeg chunk extraction failed; reading PCM chunk via soundfile for chunk {chunk_id}"
                )
                frames_start = int(round(actual_start * sr))
                frames_end = int(round(actual_end * sr))
                frames = max(frames_end - frames_start, 0)
                with sf.SoundFile(audio_path) as snd:
                    snd.seek(frames_start)
                    chunk_audio = snd.read(frames, dtype="float32", always_2d=False)
                if chunk_audio.ndim > 1:
                    chunk_audio = np.mean(chunk_audio, axis=1)
                chunk_audio = chunk_audio.astype(np.float32, copy=False)
                logger.info(
                    f"[chunks] Extracted chunk {chunk_id} via soundfile in {time.time()-t0:.2f}s ({actual_start:.1f}s→{actual_end:.1f}s)"
                )
            else:
                raise RuntimeError(
                    "ffmpeg chunk extraction failed and only PCM WAV/AIFF fallback is supported"
                )
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_chunk_raw.name)
            except Exception:
                pass

        # Save chunk to temp file
        chunk_filename = (
            f"chunk_{chunk_id:03d}_{int(start_time):04d}s-{int(end_time):04d}s.wav"
        )
        chunk_path = temp_dir / chunk_filename
        sf.write(chunk_path, chunk_audio, sr)

        chunk_info = ChunkInfo(
            chunk_id=chunk_id,
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            overlap_start=start_time - actual_start,
            overlap_end=actual_end - end_time,
            temp_path=str(chunk_path),
        )
        chunks.append(chunk_info)

        logger.info(
            f"[chunks] Saved chunk {chunk_id}: {chunk_filename} ({chunk_info.duration:.1f}s)"
        )

        # Move to next chunk
        start_time = end_time
        chunk_id += 1

    logger.info(f"[chunks] Created {len(chunks)} chunks in {temp_dir}")
    return chunks


def _merge_chunked_audio(
    chunks: List[Tuple[np.ndarray, ChunkInfo]], target_sr: int
) -> np.ndarray:
    logger.info(f"Merging {len(chunks)} processed chunks")

    if not chunks:
        return np.array([], dtype=np.float32)

    if len(chunks) == 1:
        return chunks[0][0]

    # Sort chunks by start time
    chunks.sort(key=lambda x: x[1].start_time)

    merged_parts = []

    for i, (chunk_audio, chunk_info) in enumerate(chunks):
        if i == 0:
            # First chunk: use everything
            merged_parts.append(chunk_audio)
        else:
            # Subsequent chunks: skip overlap from beginning
            overlap_samples = int(chunk_info.overlap_start * target_sr)
            if overlap_samples < len(chunk_audio):
                chunk_audio = chunk_audio[overlap_samples:]
            merged_parts.append(chunk_audio)

    # Concatenate all parts
    merged = np.concatenate(merged_parts, axis=0)
    logger.info(f"Merged audio: {len(merged) / target_sr:.1f}s total")

    return merged.astype(np.float32)


def _cleanup_chunks(chunks: List[ChunkInfo]) -> None:
    """Robust cleanup of temporary chunk files with retry logic."""
    logger.info(f"Cleaning up {len(chunks)} temporary chunk files")
    temp_dirs = set()
    failed_cleanups = []

    for chunk in chunks:
        chunk_path = Path(chunk.temp_path)
        if chunk_path.exists():
            temp_dirs.add(chunk_path.parent)
            try:
                chunk_path.unlink()
            except (OSError, PermissionError) as e:
                logger.warning(f"Failed to remove {chunk_path}: {e}")
                failed_cleanups.append(chunk_path)

    # Retry failed cleanups once after brief delay
    if failed_cleanups:
        import time

        time.sleep(0.1)
        for chunk_path in failed_cleanups:
            try:
                if chunk_path.exists():
                    chunk_path.unlink()
            except Exception:
                logger.warning(f"Permanent cleanup failure: {chunk_path}")

    # Remove empty temp directories
    for temp_dir in temp_dirs:
        try:
            if temp_dir.exists() and not any(temp_dir.iterdir()):
                temp_dir.rmdir()
                logger.debug(f"Removed temp directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Could not remove temp directory {temp_dir}: {e}")


def _butter_highpass(y: np.ndarray, sr: int, freq: float, order: int = 2) -> np.ndarray:
    if freq <= 0:
        return y
    nyq = 0.5 * sr
    Wn = min(0.999, max(1e-6, freq / nyq))
    b, a = butter(order, Wn, btype="high", analog=False)
    return filtfilt(b, a, y)


def _db(x: float) -> float:
    return float(20.0 * np.log10(max(1e-12, x)))


def _rms_db(y: np.ndarray) -> float:
    return _db(float(np.sqrt(np.mean(np.square(y)) + 1e-12)))


def _frame_params(sr: int, frame_ms: int, hop_ms: int) -> Tuple[int, int]:
    n_fft = int(round(frame_ms * 0.001 * sr))
    n_fft = max(256, 1 << int(np.ceil(np.log2(max(8, n_fft)))))
    hop = int(round(hop_ms * 0.001 * sr))
    hop = max(1, min(hop, n_fft // 2))
    return n_fft, hop


def _hann_smooth(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    w = np.hanning(win)
    w = w / (w.sum() + 1e-12)
    pad = win // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xpad, w, mode="valid")


def _exp_smooth(x: np.ndarray, alpha: float) -> np.ndarray:
    y = np.empty_like(x)
    acc = x[0]
    for i, v in enumerate(x):
        acc = alpha * v + (1 - alpha) * acc
        y[i] = acc
    return y


def _interp_per_sample(env: np.ndarray, hop: int, length: int) -> np.ndarray:
    # Map per-frame env → per-sample linearly; env length equals num frames
    t_env = np.arange(len(env)) * hop
    t = np.arange(length)
    return np.interp(t, t_env, env, left=env[0], right=env[-1])


def _oversampled_clip_detect(
    y: np.ndarray, factor: int = 4, thresh: float = 0.999
) -> bool:
    if factor <= 1:
        return bool(np.any(np.abs(y) >= thresh))
    # Linear oversample (cheap; good enough for QC)
    idx = np.arange(len(y), dtype=np.float64)
    fine = np.linspace(0, len(y) - 1, num=(len(y) - 1) * factor + 1)
    y2 = np.interp(fine, idx, y.astype(np.float64))
    return bool(np.any(np.abs(y2) >= thresh))


def _percentile_db(y_abs: np.ndarray, p: float) -> float:
    return _db(float(np.percentile(y_abs, p)))


def _dynamic_range_db(y: np.ndarray) -> float:
    y_abs = np.abs(y) + 1e-12
    hi = _percentile_db(y_abs, 95.0)
    lo = _percentile_db(y_abs, 5.0)
    return max(0.0, hi - lo)


def _estimate_loudness_lufs_approx(y: np.ndarray, sr: int) -> float:
    # Approx integrated loudness: 400ms RMS with -10 dB relative gate, no K-weight
    win = int(0.400 * sr)
    hop = int(0.100 * sr)
    if win <= 0 or len(y) < win:
        return _rms_db(y)
    frames = librosa.util.frame(y, frame_length=win, hop_length=hop).T
    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    loud = 20 * np.log10(rms + 1e-12)
    ungated_mean = np.mean(loud)
    gated = loud[loud > (ungated_mean - 10.0)]
    lufs = float(np.mean(gated) if len(gated) else ungated_mean)
    return lufs


def _simple_vad(
    y: np.ndarray, sr: int, frame_ms: int, hop_ms: int, floor_pct: float, rel_db: float
) -> np.ndarray:
    # Framewise RMS; speech if above noise floor + margin
    n_fft, hop = _frame_params(sr, frame_ms, hop_ms)
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop, window="hann")
    mag = np.abs(S)
    rms = np.sqrt(np.mean(mag**2, axis=0) + 1e-12)
    rms_db = 20 * np.log10(rms + 1e-12)
    floor = np.percentile(rms_db, floor_pct)
    speech = rms_db > (floor + rel_db)
    return speech.astype(np.bool_)


# ---------------------------
# Denoise: soft spectral subtraction (VAD-aware)
# ---------------------------


def _spectral_subtract_soft_vad(
    y: np.ndarray,
    sr: int,
    speech_mask: Optional[np.ndarray],
    alpha_db: float,
    beta: float,
    p: float,
    smooth_t: int,
    noise_ema_alpha: float,
    min_noise_frames: int,
    frame_ms: int,
    hop_ms: int,
    backoff_thresh: float,
) -> Tuple[np.ndarray, float]:
    n_fft, hop = _frame_params(sr, frame_ms, hop_ms)
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop, window="hann")
    mag, phase = np.abs(S), np.angle(S)

    # Noise estimate using VAD if available, otherwise percentile
    if speech_mask is not None and np.sum(~speech_mask) >= max(min_noise_frames, 5):
        noise_mag = np.median(mag[:, ~speech_mask], axis=1, keepdims=True)
    else:
        noise_mag = np.percentile(mag, 10, axis=1, keepdims=True)

    # Over-subtraction factor (linear)
    alpha = 10.0 ** (alpha_db / 20.0)

    residual = mag - alpha * noise_mag
    floor = beta * noise_mag
    clean_mag = np.maximum(residual, floor)

    # Soft gain mask 0..1
    M = (clean_mag / (clean_mag + alpha * noise_mag + 1e-12)) ** p

    # Temporal median smoothing to reduce musical noise
    if smooth_t > 1:
        M = median_filter(M, size=(1, smooth_t))

    # Compute floor-clipping ratio (how many bins at floor)
    floor_hits = (residual <= floor).sum()
    total_bins = residual.size
    floor_ratio = float(floor_hits) / float(total_bins + 1e-12)

    # Backoff if too high
    if floor_ratio > backoff_thresh:
        alpha *= 0.75
        beta2 = min(0.08, beta * 1.25)
        residual2 = mag - alpha * noise_mag
        floor2 = beta2 * noise_mag
        clean_mag2 = np.maximum(residual2, floor2)
        M = (clean_mag2 / (clean_mag2 + alpha * noise_mag + 1e-12)) ** p
        if smooth_t > 1:
            M = median_filter(M, size=(1, smooth_t))
        floor_hits = (residual2 <= floor2).sum()
        total_bins = residual2.size
        floor_ratio = float(floor_hits) / float(total_bins + 1e-12)
        logger.warning(
            "[denoise] High floor clipping; applied backoff (ratio=%.3f)", floor_ratio
        )

    S_hat = M * mag * np.exp(1j * phase)
    y_hat = librosa.istft(S_hat, hop_length=hop, window="hann", length=len(y))
    return y_hat.astype(np.float32), float(floor_ratio)


# ---------------------------
# Core processor with auto-chunking
# ---------------------------


class AudioPreprocessor:
    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or PreprocessConfig()

    def process_file(self, path: str) -> Tuple[np.ndarray, int, Optional[AudioHealth]]:
        # Check if file needs chunking
        duration = _get_audio_duration(path)
        threshold_seconds = self.config.chunk_threshold_minutes * 60.0

        if self.config.auto_chunk_enabled and duration > threshold_seconds:
            logger.info(
                f"Long audio detected ({duration / 60:.1f}min), using auto-chunking"
            )
            return self._process_file_chunked(path)
        else:
            logger.info(f"Processing audio normally ({duration / 60:.1f}min)")
            y, sr = _safe_load_audio(
                path, target_sr=self.config.target_sr, mono=self.config.mono
            )
            return self.process_array(y, sr)

    def _process_file_chunked(
        self, path: str
    ) -> Tuple[np.ndarray, int, Optional[AudioHealth]]:
        # Create chunks
        chunks_info = _create_audio_chunks(path, self.config)

        if not chunks_info:
            logger.warning("No chunks created, falling back to normal processing")
            y, sr = _safe_load_audio(
                path, target_sr=self.config.target_sr, mono=self.config.mono
            )
            return self.process_array(y, sr)

        processed_chunks = []
        chunk_healths = []

        try:
            # Process each chunk
            for chunk_info in chunks_info:
                logger.info(
                    f"Processing chunk {chunk_info.chunk_id}/{len(chunks_info) - 1}"
                )

                # Load and process chunk
                y_chunk, sr = _safe_load_audio(
                    chunk_info.temp_path,
                    target_sr=self.config.target_sr,
                    mono=self.config.mono,
                )

                y_processed, sr_processed, health = self.process_array(y_chunk, sr)

                processed_chunks.append((y_processed, chunk_info))
                if health:
                    chunk_healths.append(health)

            # Merge processed chunks
            merged_audio = _merge_chunked_audio(processed_chunks, self.config.target_sr)

            # Calculate combined health metrics
            combined_health = self._combine_chunk_health(
                chunk_healths, len(chunks_info)
            )
            combined_health.is_chunked = True
            combined_health.chunk_info = {
                "num_chunks": len(chunks_info),
                "chunk_duration_minutes": self.config.chunk_size_minutes,
                "total_duration_minutes": len(merged_audio)
                / self.config.target_sr
                / 60.0,
                "overlap_seconds": self.config.chunk_overlap_seconds,
            }

            logger.info(
                f"Chunked processing complete: {len(merged_audio) / self.config.target_sr:.1f}s total"
            )

            return merged_audio, self.config.target_sr, combined_health

        finally:
            # Clean up temporary files
            _cleanup_chunks(chunks_info)

    def _combine_chunk_health(
        self, chunk_healths: List[AudioHealth], num_chunks: int
    ) -> AudioHealth:
        if not chunk_healths:
            return AudioHealth(
                snr_db=0.0,
                clipping_detected=False,
                silence_ratio=1.0,
                rms_db=-60.0,
                est_lufs=-60.0,
                dynamic_range_db=0.0,
                floor_clipping_ratio=0.0,
                is_chunked=True,
            )

        # Average most metrics, take worst-case for others
        avg_snr = float(np.mean([h.snr_db for h in chunk_healths]))
        any_clipping = any(h.clipping_detected for h in chunk_healths)
        avg_silence = float(np.mean([h.silence_ratio for h in chunk_healths]))
        avg_rms = float(np.mean([h.rms_db for h in chunk_healths]))
        avg_lufs = float(np.mean([h.est_lufs for h in chunk_healths]))
        avg_dynamic_range = float(np.mean([h.dynamic_range_db for h in chunk_healths]))
        max_floor_clipping = float(
            np.max([h.floor_clipping_ratio for h in chunk_healths])
        )

        return AudioHealth(
            snr_db=avg_snr,
            clipping_detected=any_clipping,
            silence_ratio=avg_silence,
            rms_db=avg_rms,
            est_lufs=avg_lufs,
            dynamic_range_db=avg_dynamic_range,
            floor_clipping_ratio=max_floor_clipping,
            is_chunked=True,
        )

    def process_array(
        self, y: np.ndarray, sr: int
    ) -> Tuple[np.ndarray, int, Optional[AudioHealth]]:
        if y is None or len(y) == 0:
            return np.zeros(1, dtype=np.float32), sr, None
        y = y.astype(np.float32, copy=False)

        # 1) High-pass
        y = _butter_highpass(y, sr, self.config.hpf_hz, self.config.hpf_order)

        # 2) VAD
        speech_mask = (
            _simple_vad(
                y,
                sr,
                self.config.frame_ms,
                self.config.hop_ms,
                floor_pct=self.config.vad_floor_percentile,
                rel_db=self.config.vad_rel_db,
            )
            if self.config.use_vad
            else None
        )

        # 3) Denoise
        if self.config.denoise == "spectral_sub_soft":
            y_d, floor_ratio = _spectral_subtract_soft_vad(
                y,
                sr,
                speech_mask,
                alpha_db=self.config.denoise_alpha_db,
                beta=self.config.denoise_beta,
                p=self.config.mask_exponent,
                smooth_t=self.config.smooth_t,
                noise_ema_alpha=self.config.noise_update_alpha,
                min_noise_frames=self.config.min_noise_frames,
                frame_ms=self.config.frame_ms,
                hop_ms=self.config.hop_ms,
                backoff_thresh=self.config.high_clip_backoff,
            )
        else:
            y_d, floor_ratio = y, 0.0

        # 4) Gated upward gain (per-frame), smoothed (hann/exp), mapped to samples
        n_fft, hop = _frame_params(sr, self.config.frame_ms, self.config.hop_ms)
        S2 = librosa.stft(y_d, n_fft=n_fft, hop_length=hop, window="hann")
        mag2 = np.abs(S2)
        frame_rms = np.sqrt(np.mean(mag2**2, axis=0) + 1e-12)
        frame_db = 20 * np.log10(frame_rms + 1e-12)

        gate_db = float(self.config.gate_db)
        target_db = float(self.config.target_db)
        max_boost = float(self.config.max_boost_db)

        gain_db = np.zeros_like(frame_db)
        needs_boost = (frame_db > gate_db) & (frame_db < target_db)
        gain_db[needs_boost] = np.minimum(target_db - frame_db[needs_boost], max_boost)

        smooth_len = max(1, int(round(self.config.gain_smooth_ms / self.config.hop_ms)))
        if self.config.gain_smooth_method == "hann":
            gain_db_sm = _hann_smooth(gain_db, smooth_len)
        else:
            gain_db_sm = _exp_smooth(gain_db, alpha=float(self.config.exp_smooth_alpha))

        gain_lin = np.power(10.0, gain_db_sm / 20.0)
        env = _interp_per_sample(gain_lin, hop, len(y_d))
        y_boost = y_d * env.astype(np.float32)

        # 5) Compression (transparent)
        thr = float(self.config.comp_thresh_db)
        ratio = float(self.config.comp_ratio)
        knee = float(self.config.comp_knee_db)

        S3 = librosa.stft(y_boost, n_fft=n_fft, hop_length=hop, window="hann")
        mag3 = np.abs(S3)
        lvl_db = 20 * np.log10(np.sqrt(np.mean(mag3**2, axis=0)) + 1e-12)

        over = lvl_db - thr
        comp_gain_db = np.zeros_like(over)
        lower = -knee / 2.0
        upper = knee / 2.0
        for i, o in enumerate(over):
            if o <= lower:
                comp_gain_db[i] = 0.0
            elif o < upper:
                t = (o - lower) / (knee + 1e-12)
                desired = thr + o / ratio
                comp_gain_db[i] = desired - (thr + o)
                comp_gain_db[i] *= t  # knee smoothing
            else:
                comp_gain_db[i] = (thr + o / ratio) - (thr + o)

        comp_gain_lin = np.power(10.0, comp_gain_db / 20.0)
        comp_env = _interp_per_sample(comp_gain_lin, hop, len(y_boost))
        y_comp = y_boost * comp_env.astype(np.float32)

        # 6) Loudness normalization
        current_lufs = _estimate_loudness_lufs_approx(y_comp, sr)
        if self.config.loudness_mode == "asr":
            target_lufs = self.config.lufs_target_asr
        else:
            target_lufs = self.config.lufs_target_broadcast

        loudness_gain_db = target_lufs - current_lufs
        loudness_gain_db = np.clip(loudness_gain_db, -12.0, 12.0)  # Safety limits
        loudness_gain_lin = 10.0 ** (loudness_gain_db / 20.0)
        y_norm = y_comp * float(loudness_gain_lin)

        # 7) Final safety limiting
        peak = np.max(np.abs(y_norm))
        if peak > 0.95:
            safety_gain = 0.95 / peak
            y_norm = y_norm * safety_gain
            logger.warning(
                f"Applied safety limiting: {20 * np.log10(safety_gain):.1f} dB"
            )

        # 8) Quality metrics
        y_final = y_norm.astype(np.float32)

        # SNR estimation
        signal_power = np.mean(y_final**2)
        noise_estimate = np.percentile(y_final**2, 10)  # Bottom 10% as noise estimate
        snr_db = (
            10 * np.log10((signal_power / max(noise_estimate, 1e-12)))
            if signal_power > 0
            else 0.0
        )

        # Silence detection
        silence_thresh = 10.0 ** (self.config.silence_db / 20.0)
        silence_frames = np.sum(np.abs(y_final) < silence_thresh)
        silence_ratio = silence_frames / len(y_final) if len(y_final) > 0 else 1.0

        # Clipping detection with oversampling
        clipping_detected = _oversampled_clip_detect(
            y_final, self.config.oversample_factor
        )

        # Dynamic range
        dynamic_range_db = _dynamic_range_db(y_final)

        # RMS and estimated LUFS
        rms_db = _rms_db(y_final)
        est_lufs = _estimate_loudness_lufs_approx(y_final, sr)

        health = AudioHealth(
            snr_db=float(snr_db),
            clipping_detected=bool(clipping_detected),
            silence_ratio=float(silence_ratio),
            rms_db=float(rms_db),
            est_lufs=float(est_lufs),
            dynamic_range_db=float(dynamic_range_db),
            floor_clipping_ratio=float(floor_ratio),
        )

        return y_final, sr, health


# Example usage and testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Audio preprocessing with auto-chunking"
    )
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("output", help="Output audio file")
    parser.add_argument(
        "--target-sr", type=int, default=16000, help="Target sample rate"
    )
    parser.add_argument(
        "--denoise", choices=["none", "spectral_sub_soft"], default="spectral_sub_soft"
    )
    parser.add_argument("--loudness-mode", choices=["asr", "broadcast"], default="asr")
    parser.add_argument(
        "--chunk-threshold",
        type=float,
        default=30.0,
        help="Auto-chunk threshold (minutes)",
    )
    parser.add_argument(
        "--chunk-size", type=float, default=20.0, help="Chunk size (minutes)"
    )
    parser.add_argument(
        "--no-chunking", action="store_true", help="Disable auto-chunking"
    )

    args = parser.parse_args()

    # Create config
    config = PreprocessConfig(
        target_sr=args.target_sr,
        denoise=args.denoise,
        loudness_mode=args.loudness_mode,
        auto_chunk_enabled=not args.no_chunking,
        chunk_threshold_minutes=args.chunk_threshold,
        chunk_size_minutes=args.chunk_size,
    )

    # Create preprocessor
    preprocessor = AudioPreprocessor(config)

    print(f"Processing {args.input}...")
    start_time = time.time()

    try:
        y_processed, sr_processed, health = preprocessor.process_file(args.input)

        # Save output
        sf.write(args.output, y_processed, sr_processed)

        elapsed = time.time() - start_time
        duration = len(y_processed) / sr_processed

        print(f"✓ Processing complete in {elapsed:.1f}s")
        print(f"  Output: {args.output}")
        print(f"  Duration: {duration:.1f}s")
        print(f"  Sample rate: {sr_processed} Hz")

        if health:
            print("  Audio Health:")
            print(f"    SNR: {health.snr_db:.1f} dB")
            print(f"    RMS: {health.rms_db:.1f} dB")
            print(f"    Est. LUFS: {health.est_lufs:.1f}")
            print(f"    Dynamic range: {health.dynamic_range_db:.1f} dB")
            print(f"    Silence ratio: {health.silence_ratio:.1%}")
            print(f"    Clipping detected: {health.clipping_detected}")
            if health.is_chunked:
                print(f"    Processed in chunks: {health.chunk_info['num_chunks']}")

    except Exception as e:
        print(f"✗ Processing failed: {e}")
        import traceback

        traceback.print_exc()
