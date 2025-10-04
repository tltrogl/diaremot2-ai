# Documentation & Testing Updates - 2025-10-04

## Changes Made

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

---

### 2. Documented Adaptive VAD Tuning
**File:** `README.md`

**Added section:**
```markdown
- **Adaptive VAD tuning**: Pipeline automatically relaxes VAD thresholds for soft-speech scenarios:
  - `vad_threshold`: 0.22 (relaxed from CLI default 0.30)
  - `vad_min_speech_sec`: 0.40s (relaxed from 0.80s)
  - `vad_min_silence_sec`: 0.40s (relaxed from 0.80s)
  - `speech_pad_sec`: 0.15s (relaxed from 0.20s)
- Override adaptive tuning via CLI: `--vad-threshold 0.3 --vad-min-speech-sec 0.8`
```

**Rationale:** Users were unaware orchestrator overrides VAD defaults to prevent energy-VAD fallback

---

### 3. Added Unit Tests
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

---

### 4. Updated AGENTS.md
**Added missing paralinguistics metrics:**
- `duration_s`
- `words`
- `pause_ratio`

**Updated timestamp:** 2025-10-04

---

## Verification Checklist

- [x] Orchestrator fallback populates all SEGMENT_COLUMNS fields
- [x] README documents adaptive VAD tuning
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

## Files Modified

1. `src/diaremot/pipeline/orchestrator.py` (+6 lines)
2. `README.md` (+7 lines)
3. `tests/test_orchestrator_para_fallback.py` (+209 lines, new file)
4. `AGENTS.md` (+3 fields, timestamp update)
