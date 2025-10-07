# Documentation & Testing Updates - 2025-10-04

## Completed Work

### 1. Fixed Paralinguistics Fallback Bug
**File:** `src/diaremot/pipeline/orchestrator.py`
**Lines:** 549-578

**Issue:** Fallback path missing required CSV schema fields when `para.extract()` fails

**Added fields:**
- `duration_s`
- `words`
- `vq_jitter_pct`
- `vq_shimmer_db`
- `vq_hnr_db`
- `vq_cpps_db`

**Impact:** Prevents schema violations when Praat-Parselmouth unavailable

### 2. Standardized CLI Documentation & VAD Overrides
**Files:** `README.md`, `README_NEW.md`, `CLAUDE.md`, `AI_INDEX.yaml`, `AGENTS.md`

**Issue:** Documentation listed non-existent flags (`--asr-model`, `--speech-pad-sec`) and omitted orchestrator VAD auto-tuning values.

**Fixes:**
- Added README section detailing auto-tune defaults and explicit CLI overrides (`--vad-threshold`, `--vad-min-speech-sec`, `--vad-min-silence-sec`, `--vad-speech-pad-sec`).
- Updated CLI references to use canonical flags (`--whisper-model`, `--vad-speech-pad-sec`).
- Documented the 14 mandatory paralinguistics metrics in AGENTS instructions.

**Impact:** Eliminates conflicting guidance and ensures operators can reliably revert to CLI defaults.

## Outstanding Updates

### Test Coverage Follow-up
**Status:** Pending review

- Confirm `tests/test_orchestrator_para_fallback.py` remains aligned with the latest schema changes and run it in CI.
- Add integration coverage that forces the paralinguistics module to fail to validate CSV completeness end-to-end.

---

## Historical Context

### Added Unit Tests (2025-10-04)
**File:** `tests/test_orchestrator_para_fallback.py`

**Test coverage:**
- `test_fallback_populates_all_fields()` - Verifies all 14 required fields present
- `test_fallback_handles_empty_text()` - Edge case: empty text segments
- `test_fallback_computes_loudness()` - Validates RMS calculation
- `test_fallback_multiple_segments()` - Multiple segment handling

**To run:**
```powershell
cd D:\diaremot\diaremot2-ai
.\.venv\Scripts\Activate.ps1
pytest tests/test_orchestrator_para_fallback.py -v
```

## Verification Checklist

- [x] Orchestrator fallback populates all SEGMENT_COLUMNS fields
- [x] README documents adaptive VAD overrides (0.35/0.8/0.8/0.10)
- [x] Unit tests written for fallback path
- [ ] **TODO:** Run pytest to verify tests pass
- [ ] **TODO:** Test actual fallback with broken para module
- [ ] **TODO:** Verify CSV schema completeness in integration test

---

## Next Steps

1. **Run tests locally:**
   ```powershell
   pytest tests/test_orchestrator_para_fallback.py -v
   pytest tests/ -k paralinguistics
   ```

2. **Integration test:**
   - Force para module failure
   - Run full pipeline
   - Verify CSV has all 39 columns with no nulls

3. **Consider:**
   - Add orchestrator override config flag: `disable_adaptive_vad_tuning`
   - Log adaptive tuning applications
   - Add test for VAD override behavior

---

## Historical Files Modified (to be reconfirmed)

1. `src/diaremot/pipeline/orchestrator.py` (+6 lines)
2. `README.md` (+7 lines)
3. `tests/test_orchestrator_para_fallback.py` (+209 lines, new file)
4. `AGENTS.md` (+3 fields, timestamp update)
