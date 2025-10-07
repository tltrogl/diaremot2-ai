# DiaRemot — Comprehensive Documentation

## Summary

**Status: In progress.** Key factual errors have been corrected using ground truth code review, but the full documentation set still needs expansion and polish.

## Files Updated

1. **README.md** - User-facing documentation
2. **CLAUDE.md** - AI assistant comprehensive instructions  
3. **AGENTS.md** - Agent/Codex instructions
4. **AI_INDEX.yaml** - Structured project index

## Major Corrections Made

### 1. Pipeline Stage Count
- **OLD:** Claimed 12 stages with `auto_tune` as stage 3
- **NEW:** 11 stages (verified in `stages/__init__.py::PIPELINE_STAGES`)
- **Finding:** `auto_tune.py` module exists but is NOT in PIPELINE_STAGES list

### 2. VAD Parameter Values
- **OLD:** Claimed adaptive tuning applies 0.22/0.40/0.40/0.15
- **NEW:** Orchestrator actually applies 0.35/0.80/0.80/0.10
- **Source:** `orchestrator.py::AudioAnalysisPipelineV2._init_components()` lines 234-244
- **Context:** These override CLI defaults when user doesn't set flags

### 3. ASR Compute Type Default
- **OLD:** Claimed int8 default
- **NEW:** Main CLI defaults to float32
- **Source:** `cli.py` line 217 (`asr_compute_type: str = typer.Option("float32", ...)`)

### 4. AHC Distance Threshold
- **OLD:** Claimed 0.02 or various other values
- **NEW:** CLI default 0.12, orchestrator overrides to 0.15
- **Source:** `cli.py` line 215, `orchestrator.py` line 239

## What Was Verified

All values verified against actual source code:
- `src/diaremot/pipeline/stages/__init__.py` - Stage registry
- `src/diaremot/pipeline/orchestrator.py` - Parameter overrides
- `src/diaremot/pipeline/outputs.py` - CSV schema (39 columns)
- `src/diaremot/cli.py` - CLI defaults

## Current State

The documentation rewrite remains **incomplete**. To finish the effort we still need to capture the following:

- Detailed stage parameter descriptions
- Model asset locations and fallback strategies
- ONNX conversion commands
- Testing strategies and examples
- Diagnostic procedures
- Development workflows
- Reporting templates

Next action: build out these sections while preserving the verified corrections above.
