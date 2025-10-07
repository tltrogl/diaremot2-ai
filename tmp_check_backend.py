from types import SimpleNamespace

from diaremot.pipeline.cli_entry import _args_to_config
from diaremot.pipeline.orchestrator import AudioAnalysisPipelineV2

args = SimpleNamespace(
    registry_path="registry/speaker_registry.json",
    ahc_distance_threshold=0.02,
    speaker_limit=None,
    whisper_model="faster-whisper-tiny.en",
    asr_backend="auto",
    asr_compute_type="int8",
    asr_cpu_threads=1,
    language=None,
    language_mode="auto",
    ignore_tx_cache=False,
    quiet=False,
    disable_affect=False,
    affect_backend="auto",
    affect_text_model_dir=None,
    affect_intent_model_dir=None,
    beam_size=1,
    temperature=0.0,
    no_speech_threshold=0.5,
    noise_reduction=False,
    disable_sed=False,
    chunk_enabled=True,
    chunk_threshold_minutes=30.0,
    chunk_size_minutes=20.0,
    chunk_overlap_seconds=30.0,
    vad_threshold=0.3,
    vad_min_speech_sec=0.8,
    vad_min_silence_sec=0.8,
    vad_speech_pad_sec=0.2,
    vad_backend="auto",
    no_energy_fallback=False,
    energy_gate_db=-33.0,
    energy_hop_sec=0.01,
    asr_window_sec=480,
    asr_segment_timeout=300.0,
    asr_batch_timeout=1200.0,
    cpu_diarizer=False,
    clear_cache=False,
    verify_deps=False,
    strict_dependency_versions=False,
)
config = _args_to_config(args, ignore_tx_cache=False)
print("affect_backend config", config["affect_backend"])
pipe = AudioAnalysisPipelineV2(config)
print("pipeline affect backend", pipe.affect.affect_backend if pipe.affect else None)
