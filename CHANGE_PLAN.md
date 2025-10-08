# Change Plan

## Overview
- Refactor the audio preprocessing stack to expose structured results and clearer processing stages.
- Ensure downstream pipeline stages (preprocess stage runner, diarization, ASR) continue to receive correct data and cache metadata.
- Update supporting utilities/tests to align with the new abstractions.

## Planned Changes

### 1. Audio Preprocessing Refactor (High Priority)
- Introduce a `PreprocessResult` dataclass encapsulating audio array, sample rate, health metrics, and chunk metadata.
- Split `AudioPreprocessor.process_array` into composable private methods for high-pass, denoise, gain, compression, loudness, and metrics to improve readability/testability.
- Ensure chunked processing returns aggregated metadata in the result (duration, chunk info) and reuses helper methods.

### 2. Pipeline Stage Integration (High Priority)
- Update `stages/preprocess.run_preprocess` to consume `PreprocessResult`, populating `PipelineState` fields (audio, sr, health, duration, caches) from structured data.
- Preserve resume/cache logic and StageGuard accounting with the refactored output.

### 3. Tests and Stubs (Medium Priority)
- Adjust test stubs/mocks (e.g., `tests/test_stageguard.py`) to accommodate the new result type.
- Add focused unit coverage for `PreprocessResult` generation if feasible (e.g., verifying duration metadata for synthetic audio).

### 4. Documentation & CLI Example (Low Priority)
- Update CLI usage within `audio_preprocessing.py` to reflect the new return type.
- Document the new dataclass in module docstring or README section if needed.

## Verification
- Run targeted unit tests (`pytest tests/test_stageguard.py` and additional new tests) to ensure pipeline stage compatibility.
- (If time permits) run a smoke check on preprocessing CLI for sample audio.
