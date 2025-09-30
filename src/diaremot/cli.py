"""Command line interface for the DiaRemot audio intelligence pipeline."""

from __future__ import annotations

import csv
import json
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import typer

from .pipeline.config import PipelineConfig

app = typer.Typer(
    help="""High level CLI wrapper for the DiaRemot audio pipeline.\n\nThe CLI is now organised into domain-centric groups. For example:\n\n  • diaremot asr run --input sample.wav --outdir outputs/run1\n  • diaremot vad debug --input sample.wav\n  • diaremot report gen --manifest outputs/run1/manifest.json""",
    rich_markup_mode="markdown",
)
asr_app = typer.Typer(
    help="Automatic speech recognition, diarization and affect pipeline commands."
)
vad_app = typer.Typer(help="Voice activity detection tooling.")
report_app = typer.Typer(help="Reporting and summary generation utilities.")
system_app = typer.Typer(help="System level diagnostics and maintenance commands.")

app.add_typer(asr_app, name="asr")
app.add_typer(vad_app, name="vad")
app.add_typer(report_app, name="report")
app.add_typer(system_app, name="system")

BUILTIN_PROFILES: Dict[str, Dict[str, Any]] = {
    "default": {},
    "fast": {
        "whisper_model": "faster-whisper-tiny.en",
        "beam_size": 1,
        "temperature": 0.0,
        "affect_backend": "torch",
        "enable_sed": False,
    },
    "accurate": {
        "whisper_model": "faster-whisper-tiny.en",
        "beam_size": 4,
        "temperature": 0.0,
        "no_speech_threshold": 0.2,
    },
    "offline": {
        "affect_backend": "onnx",
        "disable_affect": False,
        "ignore_tx_cache": False,
    },
}

_REPORT_FORMATS = {"pdf", "html"}


@lru_cache()
def _core():
    try:
        from .pipeline import orchestrator as _orch
        from .pipeline import config as _config
    except ModuleNotFoundError:
        try:
            return import_module("diaremot.pipeline.audio_pipeline_core")
        except ModuleNotFoundError:
            return import_module("audio_pipeline_core")
    else:
        return SimpleNamespace(
            build_pipeline_config=_config.build_pipeline_config,
            diagnostics=_orch.diagnostics,
            resume=_orch.resume,
            run_pipeline=_orch.run_pipeline,
        )


def core_build_config(overrides: Optional[Dict[str, Any]] = None) -> PipelineConfig:
    data = _core().build_pipeline_config(overrides or {})
    return PipelineConfig.model_validate(data)


def core_diagnostics(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return _core().diagnostics(*args, **kwargs)


def core_resume(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return _core().resume(*args, **kwargs)


def core_run_pipeline(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return _core().run_pipeline(*args, **kwargs)


def _load_profile(profile: Optional[str]) -> Dict[str, Any]:
    if not profile:
        return {}

    if profile in BUILTIN_PROFILES:
        return dict(BUILTIN_PROFILES[profile])

    profile_path = Path(profile)
    if not profile_path.exists():
        raise typer.BadParameter(f"Profile '{profile}' not found.")

    try:
        data = json.loads(profile_path.read_text())
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise typer.BadParameter(
            f"Profile file '{profile}' is not valid JSON: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise typer.BadParameter(
            f"Profile file '{profile}' must contain a JSON object of overrides."
        )
    return data


def _normalise_path(value: Optional[Path]) -> Optional[str]:
    if value is None:
        return None
    return str(value.expanduser().resolve())


def _merge_configs(
    profile_overrides: Dict[str, Any], cli_overrides: Dict[str, Any]
) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(profile_overrides)
    defaults = _default_config().model_dump(mode="python")
    for key, value in cli_overrides.items():
        if value is None:
            continue
        if key in profile_overrides and defaults.get(key) == value:
            continue
        merged[key] = value
    return merged


def _manifest_output_root(manifest_path: Path, manifest_data: Dict[str, Any]) -> Path:
    root_value = manifest_data.get("out_dir")
    if root_value:
        root_path = Path(root_value).expanduser()
        if not root_path.is_absolute():
            root_path = (manifest_path.parent / root_path).resolve()
        return root_path
    return manifest_path.parent.resolve()


def _resolve_manifest_path(root: Path, value: Any) -> Optional[Path]:
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (root / path).resolve()
    return path


def _validate_assets(
    input_path: Path, output_dir: Path, config: PipelineConfig
) -> None:
    errors: List[str] = []

    if not input_path.exists():
        errors.append(f"Input file '{input_path}' does not exist.")

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # pragma: no cover - filesystem failure
        errors.append(f"Unable to create output directory '{output_dir}': {exc}")

    registry_path = config.registry_path.expanduser()
    try:
        registry_path.resolve().parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # pragma: no cover - filesystem failure
        errors.append(f"Unable to prepare speaker registry directory: {exc}")

    if config.affect_backend == "onnx":
        for key, path_value in {
            "affect_text_model_dir": config.affect_text_model_dir,
            "affect_intent_model_dir": config.affect_intent_model_dir,
        }.items():
            if path_value and not path_value.expanduser().exists():
                errors.append(
                    f"Configured {key}='{path_value}' but the path is missing."
                )

    cache_root = config.cache_root.expanduser()
    try:
        cache_root.resolve().mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # pragma: no cover - filesystem failure
        errors.append(f"Unable to prepare cache directory '{cache_root}': {exc}")

    if errors:
        raise typer.BadParameter("\n".join(errors))


@lru_cache()
def _default_config() -> PipelineConfig:
    return core_build_config({})


def _common_options(**kwargs: Any) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {
        "registry_path": kwargs.get("registry_path"),
        "ahc_distance_threshold": kwargs.get("ahc_distance_threshold"),
        "speaker_limit": kwargs.get("speaker_limit"),
        "whisper_model": kwargs.get("whisper_model"),
        "compute_type": kwargs.get("asr_compute_type"),
        "cpu_threads": kwargs.get("asr_cpu_threads"),
        "language": kwargs.get("language"),
        "language_mode": kwargs.get("language_mode"),
        "ignore_tx_cache": kwargs.get("ignore_tx_cache"),
        "quiet": kwargs.get("quiet"),
        "disable_affect": kwargs.get("disable_affect"),
        "affect_backend": kwargs.get("affect_backend"),
        "affect_text_model_dir": kwargs.get("affect_text_model_dir"),
        "affect_intent_model_dir": kwargs.get("affect_intent_model_dir"),
        "beam_size": kwargs.get("beam_size"),
        "temperature": kwargs.get("temperature"),
        "no_speech_threshold": kwargs.get("no_speech_threshold"),
        "noise_reduction": kwargs.get("noise_reduction"),
        "enable_sed": kwargs.get("enable_sed"),
        "auto_chunk_enabled": kwargs.get("chunk_enabled"),
        "chunk_threshold_minutes": kwargs.get("chunk_threshold_minutes"),
        "chunk_size_minutes": kwargs.get("chunk_size_minutes"),
        "chunk_overlap_seconds": kwargs.get("chunk_overlap_seconds"),
        "vad_threshold": kwargs.get("vad_threshold"),
        "vad_min_speech_sec": kwargs.get("vad_min_speech_sec"),
        "vad_min_silence_sec": kwargs.get("vad_min_silence_sec"),
        "vad_speech_pad_sec": kwargs.get("vad_speech_pad_sec"),
        "vad_backend": kwargs.get("vad_backend"),
        "disable_energy_vad_fallback": kwargs.get("disable_energy_vad_fallback"),
        "energy_gate_db": kwargs.get("energy_gate_db"),
        "energy_hop_sec": kwargs.get("energy_hop_sec"),
        "max_asr_window_sec": kwargs.get("asr_window_sec"),
        "segment_timeout_sec": kwargs.get("asr_segment_timeout"),
        "batch_timeout_sec": kwargs.get("asr_batch_timeout"),
        "cpu_diarizer": kwargs.get("cpu_diarizer"),
    }

    backend = kwargs.get("affect_backend")
    if backend is not None:
        overrides["affect_backend"] = str(backend).lower()

    asr_backend = kwargs.get("asr_backend")
    if asr_backend is not None:
        overrides["asr_backend"] = str(asr_backend).lower()

    vad_backend = kwargs.get("vad_backend")
    if vad_backend is not None:
        overrides["vad_backend"] = str(vad_backend).lower()

    return overrides


@asr_app.command("run")
def asr_run(
    input: Path = typer.Option(..., "--input", "-i", help="Path to input audio file."),
    outdir: Path = typer.Option(
        ..., "--outdir", "-o", help="Directory to write outputs."
    ),
    profile: Optional[str] = typer.Option(
        None,
        "--profile",
        help=f"Configuration profile to load ({', '.join(BUILTIN_PROFILES)} or path to JSON).",
    ),
    registry_path: Path = typer.Option(
        Path("speaker_registry.json"), help="Speaker registry path."
    ),
    ahc_distance_threshold: float = typer.Option(
        0.12, help="Agglomerative clustering distance threshold."
    ),
    speaker_limit: Optional[int] = typer.Option(
        None, help="Maximum number of speakers to keep."
    ),
    whisper_model: str = typer.Option(
        "faster-whisper-tiny.en", help="Whisper/Faster-Whisper model identifier."
    ),
    asr_backend: str = typer.Option("faster", help="ASR backend", show_default=True),
    asr_compute_type: str = typer.Option(
        "float32", help="CT2 compute type for faster-whisper."
    ),
    asr_cpu_threads: int = typer.Option(1, help="CPU threads for ASR backend."),
    language: Optional[str] = typer.Option(None, help="Override ASR language"),
    language_mode: str = typer.Option("auto", help="Language detection mode"),
    ignore_tx_cache: bool = typer.Option(
        False,
        "--ignore-tx-cache",
        help="Ignore cached diarization/transcription results.",
        is_flag=True,
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        help="Reduce console verbosity.",
        is_flag=True,
    ),
    disable_affect: bool = typer.Option(
        False,
        "--disable-affect",
        help="Skip affect analysis.",
        is_flag=True,
    ),
    affect_backend: str = typer.Option(
        "onnx", help="Affect backend (auto/torch/onnx)."
    ),
    affect_text_model_dir: Optional[Path] = typer.Option(
        None, help="Path to GoEmotions model directory."
    ),
    affect_intent_model_dir: Optional[Path] = typer.Option(
        None, help="Path to intent model directory."
    ),
    beam_size: int = typer.Option(1, help="Beam size for ASR decoding."),
    temperature: float = typer.Option(0.0, help="Sampling temperature for ASR."),
    no_speech_threshold: float = typer.Option(
        0.50, help="No-speech threshold for Whisper."
    ),
    noise_reduction: bool = typer.Option(
        False,
        "--noise-reduction",
        help="Enable gentle noise reduction.",
        is_flag=True,
    ),
    disable_sed: bool = typer.Option(
        False,
        "--disable-sed",
        help="Disable background sound event tagging.",
        is_flag=True,
    ),
    chunk_enabled: Optional[bool] = typer.Option(
        None,
        "--chunk-enabled",
        help="Set automatic chunking of long files (true/false).",
    ),
    chunk_threshold_minutes: float = typer.Option(
        30.0, help="Chunking activation threshold."
    ),
    chunk_size_minutes: float = typer.Option(20.0, help="Chunk size in minutes."),
    chunk_overlap_seconds: float = typer.Option(
        30.0, help="Overlap between chunks in seconds."
    ),
    vad_threshold: float = typer.Option(0.30, help="Silero VAD probability threshold."),
    vad_min_speech_sec: float = typer.Option(
        0.8, help="Minimum detected speech duration."
    ),
    vad_min_silence_sec: float = typer.Option(
        0.8, help="Minimum detected silence duration."
    ),
    vad_speech_pad_sec: float = typer.Option(
        0.2, help="Padding added around VAD speech regions."
    ),
    vad_backend: str = typer.Option(
        "auto", help="Silero VAD backend (auto/torch/onnx)."
    ),
    disable_energy_vad_fallback: bool = typer.Option(
        False,
        "--disable-energy-fallback",
        help="Disable energy VAD fallback when Silero VAD fails.",
        is_flag=True,
    ),
    energy_gate_db: float = typer.Option(-33.0, help="Energy VAD gating threshold."),
    energy_hop_sec: float = typer.Option(0.01, help="Energy VAD hop length."),
    asr_window_sec: int = typer.Option(
        480, help="Maximum audio length per ASR window."
    ),
    asr_segment_timeout: float = typer.Option(300.0, help="Timeout per ASR segment."),
    asr_batch_timeout: float = typer.Option(
        1200.0, help="Timeout for a batch of ASR segments."
    ),
    cpu_diarizer: bool = typer.Option(
        False,
        "--cpu-diarizer",
        help="Enable CPU optimised diarizer wrapper.",
        is_flag=True,
    ),
    clear_cache: bool = typer.Option(
        False,
        "--clear-cache",
        help="Clear cached diarization/transcription data before running.",
        is_flag=True,
    ),
):
    cli_overrides = _common_options(
        registry_path=_normalise_path(registry_path),
        ahc_distance_threshold=ahc_distance_threshold,
        speaker_limit=speaker_limit,
        whisper_model=whisper_model,
        asr_backend=asr_backend,
        asr_compute_type=asr_compute_type,
        asr_cpu_threads=asr_cpu_threads,
        language=language,
        language_mode=language_mode,
        ignore_tx_cache=ignore_tx_cache,
        quiet=quiet,
        disable_affect=disable_affect,
        affect_backend=affect_backend,
        affect_text_model_dir=_normalise_path(affect_text_model_dir),
        affect_intent_model_dir=_normalise_path(affect_intent_model_dir),
        beam_size=beam_size,
        temperature=temperature,
        no_speech_threshold=no_speech_threshold,
        noise_reduction=noise_reduction,
        enable_sed=not disable_sed,
        chunk_enabled=chunk_enabled,
        chunk_threshold_minutes=chunk_threshold_minutes,
        chunk_size_minutes=chunk_size_minutes,
        chunk_overlap_seconds=chunk_overlap_seconds,
        vad_threshold=vad_threshold,
        vad_min_speech_sec=vad_min_speech_sec,
        vad_min_silence_sec=vad_min_silence_sec,
        vad_speech_pad_sec=vad_speech_pad_sec,
        vad_backend=vad_backend,
        disable_energy_vad_fallback=disable_energy_vad_fallback,
        energy_gate_db=energy_gate_db,
        energy_hop_sec=energy_hop_sec,
        asr_window_sec=asr_window_sec,
        asr_segment_timeout=asr_segment_timeout,
        asr_batch_timeout=asr_batch_timeout,
        cpu_diarizer=cpu_diarizer,
    )

    profile_overrides = _load_profile(profile)
    merged = _merge_configs(profile_overrides, cli_overrides)

    try:
        config = core_build_config(merged)
    except ValueError as exc:
        raise typer.BadParameter(f"Configuration error: {exc}") from exc

    _validate_assets(input, outdir, config)

    try:
        manifest = core_run_pipeline(
            str(input),
            str(outdir),
            config=config.model_dump(mode="python"),
            clear_cache=clear_cache,
        )
    except Exception as exc:  # pragma: no cover - runtime failure
        typer.secho(f"Pipeline execution failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    typer.echo(json.dumps(manifest, indent=2))


app.command("run")(asr_run)


@asr_app.command("resume")
def asr_resume(
    input: Path = typer.Option(..., "--input", "-i", help="Original input audio file."),
    outdir: Path = typer.Option(
        ..., "--outdir", "-o", help="Output directory used in the previous run."
    ),
    profile: Optional[str] = typer.Option(
        None,
        "--profile",
        help=f"Configuration profile to load ({', '.join(BUILTIN_PROFILES)} or path to JSON).",
    ),
    registry_path: Path = typer.Option(
        Path("speaker_registry.json"), help="Speaker registry path."
    ),
    ahc_distance_threshold: float = typer.Option(
        0.12, help="Agglomerative clustering distance threshold."
    ),
    speaker_limit: Optional[int] = typer.Option(
        None, help="Maximum number of speakers to keep."
    ),
    whisper_model: str = typer.Option(
        "faster-whisper-tiny.en", help="Whisper/Faster-Whisper model identifier."
    ),
    asr_backend: str = typer.Option("faster", help="ASR backend", show_default=True),
    asr_compute_type: str = typer.Option(
        "float32", help="CT2 compute type for faster-whisper."
    ),
    asr_cpu_threads: int = typer.Option(1, help="CPU threads for ASR backend."),
    language: Optional[str] = typer.Option(None, help="Override ASR language"),
    language_mode: str = typer.Option("auto", help="Language detection mode"),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        help="Reduce console verbosity.",
        is_flag=True,
    ),
    disable_affect: bool = typer.Option(
        False,
        "--disable-affect",
        help="Skip affect analysis.",
        is_flag=True,
    ),
    affect_backend: str = typer.Option(
        "onnx", help="Affect backend (auto/torch/onnx)."
    ),
    affect_text_model_dir: Optional[Path] = typer.Option(
        None, help="Path to GoEmotions model directory."
    ),
    affect_intent_model_dir: Optional[Path] = typer.Option(
        None, help="Path to intent model directory."
    ),
    beam_size: int = typer.Option(1, help="Beam size for ASR decoding."),
    temperature: float = typer.Option(0.0, help="Sampling temperature for ASR."),
    no_speech_threshold: float = typer.Option(
        0.50, help="No-speech threshold for Whisper."
    ),
    noise_reduction: bool = typer.Option(
        False,
        "--noise-reduction",
        help="Enable gentle noise reduction.",
        is_flag=True,
    ),
    disable_sed: bool = typer.Option(
        False,
        "--disable-sed",
        help="Disable background sound event tagging.",
        is_flag=True,
    ),
    chunk_enabled: Optional[bool] = typer.Option(
        None,
        "--chunk-enabled",
        help="Set automatic chunking of long files (true/false).",
    ),
    chunk_threshold_minutes: float = typer.Option(
        30.0, help="Chunking activation threshold."
    ),
    chunk_size_minutes: float = typer.Option(20.0, help="Chunk size in minutes."),
    chunk_overlap_seconds: float = typer.Option(
        30.0, help="Overlap between chunks in seconds."
    ),
    vad_threshold: float = typer.Option(0.30, help="Silero VAD probability threshold."),
    vad_min_speech_sec: float = typer.Option(
        0.8, help="Minimum detected speech duration."
    ),
    vad_min_silence_sec: float = typer.Option(
        0.8, help="Minimum detected silence duration."
    ),
    vad_speech_pad_sec: float = typer.Option(
        0.2, help="Padding added around VAD speech regions."
    ),
    vad_backend: str = typer.Option(
        "auto", help="Silero VAD backend (auto/torch/onnx)."
    ),
    disable_energy_vad_fallback: bool = typer.Option(
        False,
        "--disable-energy-fallback",
        help="Disable energy VAD fallback when Silero VAD fails.",
        is_flag=True,
    ),
    energy_gate_db: float = typer.Option(-33.0, help="Energy VAD gating threshold."),
    energy_hop_sec: float = typer.Option(0.01, help="Energy VAD hop length."),
    asr_window_sec: int = typer.Option(
        480, help="Maximum audio length per ASR window."
    ),
    asr_segment_timeout: float = typer.Option(300.0, help="Timeout per ASR segment."),
    asr_batch_timeout: float = typer.Option(
        1200.0, help="Timeout for a batch of ASR segments."
    ),
    cpu_diarizer: bool = typer.Option(
        False,
        "--cpu-diarizer",
        help="Enable CPU optimised diarizer wrapper.",
        is_flag=True,
    ),
):
    cli_overrides = _common_options(
        registry_path=_normalise_path(registry_path),
        ahc_distance_threshold=ahc_distance_threshold,
        speaker_limit=speaker_limit,
        whisper_model=whisper_model,
        asr_backend=asr_backend,
        asr_compute_type=asr_compute_type,
        asr_cpu_threads=asr_cpu_threads,
        language=language,
        language_mode=language_mode,
        ignore_tx_cache=False,
        quiet=quiet,
        disable_affect=disable_affect,
        affect_backend=affect_backend,
        affect_text_model_dir=_normalise_path(affect_text_model_dir),
        affect_intent_model_dir=_normalise_path(affect_intent_model_dir),
        beam_size=beam_size,
        temperature=temperature,
        no_speech_threshold=no_speech_threshold,
        noise_reduction=noise_reduction,
        enable_sed=not disable_sed,
        chunk_enabled=chunk_enabled,
        chunk_threshold_minutes=chunk_threshold_minutes,
        chunk_size_minutes=chunk_size_minutes,
        chunk_overlap_seconds=chunk_overlap_seconds,
        vad_threshold=vad_threshold,
        vad_min_speech_sec=vad_min_speech_sec,
        vad_min_silence_sec=vad_min_silence_sec,
        vad_speech_pad_sec=vad_speech_pad_sec,
        vad_backend=vad_backend,
        disable_energy_vad_fallback=disable_energy_vad_fallback,
        energy_gate_db=energy_gate_db,
        energy_hop_sec=energy_hop_sec,
        asr_window_sec=asr_window_sec,
        asr_segment_timeout=asr_segment_timeout,
        asr_batch_timeout=asr_batch_timeout,
        cpu_diarizer=cpu_diarizer,
    )

    profile_overrides = _load_profile(profile)
    merged = _merge_configs(profile_overrides, cli_overrides)

    try:
        config = core_build_config(merged)
    except ValueError as exc:
        raise typer.BadParameter(f"Configuration error: {exc}") from exc

    _validate_assets(input, outdir, config)

    try:
        manifest = core_resume(
            str(input),
            str(outdir),
            config=config.model_dump(mode="python"),
        )
    except Exception as exc:  # pragma: no cover - runtime failure
        typer.secho(f"Pipeline resume failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    typer.echo(json.dumps(manifest, indent=2))


app.command("resume")(asr_resume)


@system_app.command("diagnostics")
def diagnostics(
    strict: bool = typer.Option(False, help="Require minimum dependency versions."),
) -> None:
    """Run dependency diagnostics and emit a JSON summary."""

    result = core_diagnostics(require_versions=strict)
    typer.echo(json.dumps(result, indent=2))


app.command("diagnostics")(diagnostics)


def main_diagnostics() -> None:
    """Console script entry point for :func:`diagnostics`."""

    typer.run(diagnostics)


def _load_segments_jsonl(path: Path) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    if not path.exists():
        return segments
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                segments.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return segments


def _load_segments_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _load_speakers_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


@report_app.command("gen")
def report_gen(
    manifest: Path = typer.Option(
        ..., "--manifest", "-m", help="Manifest JSON from a pipeline run."
    ),
    outdir: Optional[Path] = typer.Option(
        None, "--outdir", help="Override output directory."
    ),
    format: List[str] = typer.Option(
        ["pdf"], "--format", "-f", help="Formats to (re)generate: pdf or html."
    ),
    force: bool = typer.Option(
        False, "--force", help="Regenerate even if the target file already exists."
    ),
) -> None:
    try:
        manifest_data = json.loads(manifest.read_text())
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(
            f"Manifest '{manifest}' is not valid JSON: {exc}"
        ) from exc

    outputs = manifest_data.get("outputs", {}) or {}
    output_root = _manifest_output_root(manifest, manifest_data)
    base_outdir = Path(outdir).expanduser().resolve() if outdir else output_root
    base_outdir.mkdir(parents=True, exist_ok=True)

    segments: List[Dict[str, Any]] = []
    segments_path = _resolve_manifest_path(output_root, outputs.get("jsonl"))
    if segments_path:
        segments = _load_segments_jsonl(segments_path)
    if not segments:
        csv_path = _resolve_manifest_path(output_root, outputs.get("csv"))
        if csv_path:
            segments = _load_segments_csv(csv_path)

    if not segments:
        raise typer.BadParameter("Manifest does not reference any segment outputs.")

    speakers_summary: List[Dict[str, Any]] = []
    speakers_path = _resolve_manifest_path(output_root, outputs.get("speakers_summary"))
    if speakers_path:
        speakers_summary = _load_speakers_csv(speakers_path)

    if not speakers_summary:
        from .summaries.speakers_summary_builder import build_speakers_summary

        speakers_summary = build_speakers_summary(segments, {}, {})

    overlap_stats: Dict[str, Any] = {}
    qc_path = _resolve_manifest_path(output_root, outputs.get("qc_report"))
    if qc_path and qc_path.exists():
        try:
            qc_data = json.loads(qc_path.read_text())
            overlap_stats = qc_data.get("overlap_stats", {}) or {}
        except Exception:
            overlap_stats = {}

    formats = {fmt.lower() for fmt in format}
    unknown_formats = formats - _REPORT_FORMATS
    if unknown_formats:
        raise typer.BadParameter(
            f"Unsupported report format(s): {', '.join(sorted(unknown_formats))}"
        )

    results: Dict[str, str] = {}
    file_id = manifest_data.get("file_id") or base_outdir.name

    if "pdf" in formats:
        from .summaries.pdf_summary_generator import PDFSummaryGenerator

        target = base_outdir / "summary.pdf"
        if force or not target.exists():
            pdf_path = PDFSummaryGenerator().render_to_pdf(
                out_dir=str(base_outdir),
                file_id=file_id,
                segments=segments,
                speakers_summary=speakers_summary,
                overlap_stats=overlap_stats,
            )
        else:
            pdf_path = str(target.resolve())
        results["pdf"] = pdf_path

    if "html" in formats:
        from .summaries.html_summary_generator import HTMLSummaryGenerator

        target = base_outdir / "summary.html"
        if force or not target.exists():
            html_path = HTMLSummaryGenerator().render_to_html(
                out_dir=str(base_outdir),
                file_id=file_id,
                segments=segments,
                speakers_summary=speakers_summary,
                overlap_stats=overlap_stats,
            )
        else:
            html_path = str(target.resolve())
        results["html"] = html_path

    typer.echo(json.dumps(results, indent=2))


@vad_app.command("debug")
def vad_debug(
    input: Path = typer.Option(..., "--input", "-i", help="Audio file to analyse."),
    threshold: float = typer.Option(
        0.30, "--threshold", help="Silero probability threshold."
    ),
    min_speech: float = typer.Option(
        0.8, "--min-speech", help="Minimum detected speech duration."
    ),
    min_silence: float = typer.Option(
        0.8, "--min-silence", help="Minimum detected silence duration."
    ),
    pad: float = typer.Option(
        0.2, "--pad", help="Padding added around VAD speech regions."
    ),
    backend: str = typer.Option(
        "auto", "--backend", help="Backend to use: auto/torch/onnx."
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit machine-readable JSON."
    ),
) -> None:
    if not input.exists():
        raise typer.BadParameter(f"Input file '{input}' does not exist.")

    from .pipeline.speaker_diarization import DiarizationConfig, _SileroWrapper

    import librosa
    import numpy as np

    cfg = DiarizationConfig(
        target_sr=16000,
        vad_threshold=threshold,
        vad_min_speech_sec=min_speech,
        vad_min_silence_sec=min_silence,
        speech_pad_sec=pad,
        vad_backend=backend,
    )

    wav, sr = librosa.load(str(input), sr=cfg.target_sr, mono=True)
    wav = np.asarray(wav, dtype=np.float32)
    detector = _SileroWrapper(
        cfg.vad_threshold, cfg.speech_pad_sec, backend=cfg.vad_backend
    )
    speech_regions = detector.detect(
        wav, sr, cfg.vad_min_speech_sec, cfg.vad_min_silence_sec
    )

    segments = [
        {
            "index": idx + 1,
            "start": float(start),
            "end": float(end),
            "duration": float(max(0.0, end - start)),
        }
        for idx, (start, end) in enumerate(speech_regions)
    ]

    if json_output:
        typer.echo(json.dumps({"segments": segments}, indent=2))
        return

    if not segments:
        typer.echo("No speech detected by Silero VAD.")
        return

    total = sum(seg["duration"] for seg in segments)
    typer.echo(f"Detected {len(segments)} speech region(s), total {total:.2f}s")
    for seg in segments:
        typer.echo(
            f"#{seg['index']:02d} {seg['start']:.2f}s → {seg['end']:.2f}s "
            f"(dur {seg['duration']:.2f}s)"
        )


if __name__ == "__main__":  # pragma: no cover
    app()
