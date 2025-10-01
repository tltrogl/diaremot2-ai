"""Preprocessing stages (audio + background SED tagging)."""

from __future__ import annotations

from pathlib import Path

from ..logging_utils import StageGuard, _fmt_hms
from ..pipeline_checkpoint_system import ProcessingStage
from .base import PipelineState
from .utils import compute_audio_sha16, compute_pp_signature, read_json_safe

__all__ = ["run_preprocess", "run_background_sed"]


def run_preprocess(
    pipeline: "AudioAnalysisPipelineV2", state: PipelineState, guard: StageGuard
) -> None:
    if not hasattr(pipeline, "pre") or pipeline.pre is None:
        raise RuntimeError("Preprocessor component unavailable; initialization failed")

    y, sr, health = pipeline.pre.process_file(state.input_audio_path)
    state.y = y
    state.sr = sr
    state.health = health
    state.duration_s = float(len(y) / sr) if sr else 0.0
    pipeline.corelog.info(f"[preprocess] file duration {_fmt_hms(state.duration_s)}")
    pipeline.corelog.event(
        "preprocess",
        "metrics",
        duration_s=state.duration_s,
        snr_db=float(getattr(health, "snr_db", 0.0)) if health else None,
    )
    guard.done(duration_s=state.duration_s)

    state.audio_sha16 = compute_audio_sha16(state.y)
    if state.audio_sha16:
        pipeline.checkpoints.seed_file_hash(state.input_audio_path, state.audio_sha16)

    pipeline.checkpoints.create_checkpoint(
        state.input_audio_path,
        ProcessingStage.AUDIO_PREPROCESSING,
        {"sr": state.sr},
        progress=5.0,
        file_hash=state.audio_sha16,
    )

    state.pp_sig = compute_pp_signature(pipeline.pp_conf)
    cache_root = pipeline.cache_root
    if not state.audio_sha16:
        cache_dir = cache_root / "nohash"
    else:
        cache_dir = cache_root / state.audio_sha16
    cache_dir.mkdir(parents=True, exist_ok=True)
    state.cache_dir = cache_dir
    diar_path = cache_dir / "diar.json"
    tx_path = cache_dir / "tx.json"

    state.diar_cache = read_json_safe(diar_path)
    state.tx_cache = read_json_safe(tx_path)
    diar_cache_src = str(diar_path) if state.diar_cache else None
    tx_cache_src = str(tx_path) if state.tx_cache else None

    if not state.diar_cache or not state.tx_cache:
        for root in pipeline.cache_roots[1:]:
            alt_dir = Path(root) / state.audio_sha16
            alt_diar = read_json_safe(alt_dir / "diar.json")
            alt_tx = read_json_safe(alt_dir / "tx.json")
            if not state.diar_cache and alt_diar:
                state.diar_cache = alt_diar
                diar_cache_src = str(alt_dir / "diar.json")
            if not state.tx_cache and alt_tx:
                state.tx_cache = alt_tx
                tx_cache_src = str(alt_dir / "tx.json")
            if state.diar_cache and state.tx_cache:
                break

    def _cache_matches(obj: dict[str, object] | None) -> bool:
        return (
            bool(obj)
            and obj.get("version") == pipeline.cache_version
            and obj.get("audio_sha16") == state.audio_sha16
            and obj.get("pp_signature") == state.pp_sig
        )

    if _cache_matches(state.tx_cache):
        state.resume_tx = True
        state.resume_diar = bool(_cache_matches(state.diar_cache))
        if state.resume_diar:
            pipeline.corelog.info(
                "[resume] using tx.json+diar.json caches; skipping diarize+ASR"
            )
        else:
            pipeline.corelog.info(
                "[resume] using tx.json cache; skipping ASR and reconstructing turns from tx cache"
            )
        pipeline.corelog.event(
            "resume",
            "tx_cache_hit",
            audio_sha16=state.audio_sha16,
            src=tx_cache_src,
        )
    elif _cache_matches(state.diar_cache):
        state.resume_diar = True
        pipeline.corelog.info("[resume] using diar.json cache; skipping diarize")
        pipeline.corelog.event(
            "resume",
            "diar_cache_hit",
            audio_sha16=state.audio_sha16,
            src=diar_cache_src,
        )

    if pipeline.cfg.get("ignore_tx_cache"):
        state.diar_cache = None
        state.tx_cache = None
        state.resume_diar = False
        state.resume_tx = False


def run_background_sed(
    pipeline: "AudioAnalysisPipelineV2", state: PipelineState, guard: StageGuard
) -> None:
    try:
        if (
            getattr(pipeline, "sed_tagger", None) is not None
            and state.y.size > 0
            and state.sr
        ):
            sed_info = pipeline.sed_tagger.tag(state.y, state.sr)
            if sed_info:
                pipeline.corelog.event(
                    "background_sed",
                    "tags",
                    dominant_label=sed_info.get("dominant_label"),
                    noise_score=sed_info.get("noise_score"),
                )
                pipeline.stats.config_snapshot["background_sed"] = sed_info
                state.sed_info = sed_info
    except (
        ImportError,
        ModuleNotFoundError,
        RuntimeError,
        ValueError,
        OSError,
    ) as exc:
        pipeline.corelog.warn(
            "[sed] tagging skipped: "
            f"{exc}. Install sed_panns dependencies; background SED is required."
        )
    finally:
        guard.done()
