# High-Impact Module Refactor Assessment

## Selection Criteria
- Ranked Python modules in `src/diaremot` by lines of code and selected the largest seven for detailed review (`wc -l`).
- Focused on files that orchestrate core speech-processing stages or encapsulate dense feature extraction pipelines.
- Evaluated each module's callers (upstream) and dependencies (downstream) to determine architectural pressure points.

## Module Assessments

### `affect/paralinguistics.py`
**Role & Responsibilities.** Implements the full paralinguistic feature stack: text-derived disfluencies, pause analysis, pitch statistics, spectral metrics, and voice-quality via Parselmouth with extensive fallback handling.【F:src/diaremot/affect/paralinguistics.py†L14-L205】【F:src/diaremot/affect/paralinguistics.py†L1071-L1260】

**Upstream usage.** The pipeline orchestrator imports the module (aliased as `para`) to generate segment-level feature payloads during affect analysis.【F:src/diaremot/pipeline/orchestrator.py†L16-L63】

**Downstream dependencies.** Relies on heavy DSP libraries (librosa, scipy, parselmouth) and numerous helper routines defined in the same file (silence detection, pause stats, spectral features, SNR vetting).【F:src/diaremot/affect/paralinguistics.py†L20-L205】【F:src/diaremot/affect/paralinguistics.py†L360-L495】

**Refactor considerations.**
- *Pain points*: 2.5k-line monolith mixes configuration, caching layers, DSP kernels, and orchestration logic; testing each feature path in isolation is difficult. Shared state such as `ParalinguisticsConfig` and global caches complicate unit tests and future optimizations.
- *Recommendation*: Break into focused modules (`text_features`, `prosody`, `voice_quality`) with a coordinating façade class that consumes a config dataclass. Keep feature calculators pure to simplify benchmarking.

**Pros**
- Enables targeted unit tests per feature family and easier onboarding for speech scientists.
- Facilitates alternative implementations (e.g., GPU-optional pitch extractor) without touching unrelated code.

**Cons**
- Requires broad update of import sites and potential migration plan for serialized outputs.
- Short-term risk of regressions unless golden-audio regression suite exists (see [`golden_audio_regression_suite.md`](./golden_audio_regression_suite.md) for the recommended harness).

### `pipeline/transcription_module.py`
**Role & Responsibilities.** Provides async transcription orchestration, including backend discovery, audio resampling utilities, batching heuristics, asynchronous job execution, and factory functions for synchronous wrappers.【F:src/diaremot/pipeline/transcription_module.py†L1-L205】【F:src/diaremot/pipeline/transcription_module.py†L380-L610】【F:src/diaremot/pipeline/transcription_module.py†L1386-L1497】

**Upstream usage.** Instantiated by the orchestrator when running the transcription stage and by CLI tools that rely on `create_transcriber` or `AudioTranscriber` abstractions.【F:src/diaremot/pipeline/orchestrator.py†L16-L117】

**Downstream dependencies.** Talks to faster-whisper / Whisper backends, numpy, librosa, asyncio, ThreadPoolExecutor, and caching structures for SNR estimation.【F:src/diaremot/pipeline/transcription_module.py†L70-L205】【F:src/diaremot/pipeline/transcription_module.py†L1200-L1380】

**Refactor considerations.**
- *Pain points*: File couples environment setup, backend probing, batching policy, async runtime, and diagnostics. Complex concurrency (async + threads) makes reasoning about failure states hard.
- *Recommendation*: Extract backend management (`ModelManager`) and batching logic into separate modules; wrap CPU/GPU environment setup in a lightweight utility imported at entry points. Consider splitting synchronous wrapper into its own file.

**Pros**
- Clarifies responsibilities (backend selection vs. transcription scheduling) and enables reuse in tests.
- Simplifies targeted performance tuning (e.g., swapping batching strategies) without risking environment guard code.

**Cons**
- Fragmentation may increase number of files new contributors must touch for simple changes.
- Requires careful migration of shared caches and logging to avoid duplication.

### `affect/emotion_analyzer.py`
**Role & Responsibilities.** Houses text emotion, speech emotion, VAD emotion, and intent inference plus data serialization for affect outputs, all under a single analyzer class hierarchy.【F:src/diaremot/affect/emotion_analyzer.py†L34-L220】【F:src/diaremot/affect/emotion_analyzer.py†L960-L1120】

**Upstream usage.** Imported by the orchestrator as `EmotionIntentAnalyzer` for the affect stage, providing segment-level affect payloads and fallbacks.【F:src/diaremot/pipeline/orchestrator.py†L16-L117】【F:src/diaremot/affect/emotion_analyzer.py†L1011-L1094】

**Downstream dependencies.** Wraps ONNXRuntime sessions, Hugging Face tokenizers/pipelines, numpy preprocessing, and custom normalization utilities defined in-file.【F:src/diaremot/affect/emotion_analyzer.py†L34-L220】【F:src/diaremot/affect/emotion_analyzer.py†L1011-L1094】

**Refactor considerations.**
- *Pain points*: Mixed responsibilities (data prep, inference, fallback routing, serialization) hinder substituting alternative models. Intent logic is interleaved with emotion code, leading to long private methods with backend branching.
- *Recommendation*: Introduce dedicated providers (`TextEmotionService`, `SpeechEmotionService`, `IntentService`) composed by a thin orchestrator class. Relocate shared math utilities to a separate helpers module.

**Pros**
- Enables independent evolution of text vs. acoustic models and easier benchmarking.
- Reduces risk of regressions when swapping ONNX vs. Torch backends when paired with the golden-audio regression suite outlined in [`golden_audio_regression_suite.md`](./golden_audio_regression_suite.md).

**Cons**
- Requires API design to preserve backwards-compatible payload schema.
- Temporary duplication while migrating legacy helper methods.

### `pipeline/speaker_diarization.py`
**Role & Responsibilities.** Encapsulates VAD handling, embedding extraction, clustering, speaker registry integration, and fallback heuristics inside one file.【F:src/diaremot/pipeline/speaker_diarization.py†L1-L220】【F:src/diaremot/pipeline/speaker_diarization.py†L501-L760】

**Upstream usage.** The orchestrator imports `SpeakerDiarizer` and `DiarizationConfig` as aliases for stage execution.【F:src/diaremot/pipeline/orchestrator.py†L50-L55】

**Downstream dependencies.** Talks to ONNXRuntime for Silero/ECAPA models, sklearn clustering, librosa/scipy DSP, and registry persistence.【F:src/diaremot/pipeline/speaker_diarization.py†L1-L220】【F:src/diaremot/pipeline/speaker_diarization.py†L501-L760】

**Refactor considerations.**
- *Pain points*: VAD wrapper, embedding pipeline, and registry management each span hundreds of lines with shared logging. Hard to replace VAD or clustering logic independently.
- *Recommendation*: Split into submodules (`vad_wrappers`, `embedding_engine`, `speaker_registry`) with a cohesive diarizer class orchestrating them. Document explicit interfaces for injecting alternative models.

**Pros**
- Improves testability of each stage (e.g., simulate VAD outputs without loading ECAPA).
- Supports future multi-speaker features like overlap detection by swapping clustering module.

**Cons**
- Refactor touches critical path; regression risk unless diarization benchmarks exist.
- Additional module boundaries may slightly increase import time in constrained environments.

### `pipeline/audio_preprocessing.py`
**Role & Responsibilities.** Performs decoding, chunking, denoising, VAD gating, loudness normalization, QC metrics, and packaging results into structured dataclasses.【F:src/diaremot/pipeline/audio_preprocessing.py†L1-L176】【F:src/diaremot/pipeline/audio_preprocessing.py†L33-L134】

**Upstream usage.** Instantiated by the orchestrator (`AudioPreprocessor`) ahead of transcription and diarization to deliver normalized audio and health diagnostics.【F:src/diaremot/pipeline/orchestrator.py†L16-L117】

**Downstream dependencies.** Relies on ffmpeg, librosa, soundfile, scipy filters, and custom chunk metadata structures.【F:src/diaremot/pipeline/audio_preprocessing.py†L16-L176】【F:src/diaremot/pipeline/audio_preprocessing.py†L33-L134】

**Refactor considerations.**
- *Pain points*: File blends decoding I/O with DSP pipelines and QC analytics; chunk management shares state with preprocessing steps, complicating streaming scenarios.
- *Recommendation*: Separate decoding/IO utilities from signal conditioning; encapsulate QC metric computation in a reusable analyzer that can be invoked post-chunk merge.

**Pros**
- Enables reuse of decoding utilities for batch ingestion or testing harnesses.
- Simplifies future addition of GPU-accelerated denoisers by isolating DSP hooks.

**Cons**
- Splitting chunk management may introduce data marshaling overhead if not carefully designed.
- Requires new abstraction to keep backwards compatibility for `PreprocessResult.to_tuple()` consumers.

### `pipeline/pipeline_checkpoint_system.py`
**Role & Responsibilities.** Manages checkpoint metadata, persistence, hashing, and progress tracking across pipeline stages with locking for thread safety.【F:src/diaremot/pipeline/pipeline_checkpoint_system.py†L1-L190】

**Upstream usage.** The orchestrator constructs `PipelineCheckpointManager` to resume runs and manage checkpoints per stage.【F:src/diaremot/pipeline/orchestrator.py†L34-L117】

**Downstream dependencies.** Utilizes JSON/pickle serialization, threading primitives, and shared hashing utilities.【F:src/diaremot/pipeline/pipeline_checkpoint_system.py†L1-L190】

**Refactor considerations.**
- *Pain points*: Mixing persistence concerns (JSON vs. pickle), progress math, and file-hash caching in one class. Lacks abstraction for alternative storage backends.
- *Recommendation*: Extract persistence adapters (filesystem vs. cloud), encapsulate progress math into a helper class, and enforce typed metadata schemas.

**Pros**
- Unlocks remote checkpoint backends or database persistence without rewriting progress tracking.
- Improves unit-test coverage by mocking persistence adapters.

**Cons**
- Adds interface complexity when only filesystem storage is currently required.
- Needs migration tooling for existing checkpoint files if schema changes.

## Overall Guidance
- Prioritize refactoring `paralinguistics` and `transcription_module` first; they sit directly on the hot path and show the highest cognitive load per LOC.
- Stage refactors behind feature flags or environment toggles to preserve current pipeline stability during the transition.
