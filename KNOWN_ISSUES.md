# Known Issues & Solutions

## Current Status: January 9, 2026

### Working Algorithms ✅
1. **CLAHE** (Low-light enhancement) - Enhancing images successfully
2. **White Balance** (Color correction) - Enhancing images successfully
3. **Exposure Correction** (Color correction) - Enhancing images successfully

### Failing Algorithms ❌

#### 1. Real-ESRGAN (Super-Resolution) - FIXED ✅
**Error**: Wrong import path
```
cannot import name 'RealESRGANer' from 'basicsr.models.realesrgan_model'
```
**Cause**: Importing from wrong module - should use `realesrgan` not `basicsr.models.realesrgan_model`
**Impact**: Using placeholder (no actual enhancement)
**Status**: FIXED - Changed to `from realesrgan import RealESRGANer`

#### 2. GFPGAN (Face Enhancement)
**Error**: NumPy version incompatibility (same as Real-ESRGAN)
**Impact**: Using placeholder (no actual enhancement)

#### 3. super-image (Super-Resolution)
**Error**: NumPy dtype inference issue
```
RuntimeError: Could not infer dtype of numpy.float32
```
**Cause**: `super-image` package incompatible with NumPy 2.x
**Impact**: Crashes during enhancement

#### 4. Face Enhancement - Analysis (FIXED ✅)
**Error**: `NameError: name 'cv2' is not defined`
**Cause**: Missing cv2 import
**Status**: FIXED - cv2 import added

## Root Cause

**NumPy Version Incompatibility**

The `basicsr` (for Real-ESRGAN), `gfpgan`, and `super-image` packages were compiled and tested with NumPy 1.x, but the system has NumPy 2.2.6 installed.

NumPy 2.0 introduced breaking changes that make older packages incompatible.

## Solutions

### Immediate Fix: Downgrade NumPy (RECOMMENDED)

This fixes ALL NumPy-related issues at once:

```bash
# Deactivate current venv
deactivate  # or close terminal

# Reactivate
.venv\Scripts\activate

# Downgrade NumPy to 1.x version
pip install "numpy<2.0"

# Verify version
pip list | grep numpy
# Should show: numpy 1.26.x (or similar 1.x version)
```

**Expected Results After NumPy Downgrade**:
- ✅ Real-ESRGAN loads and processes images (500-2000ms duration)
- ✅ GFPGAN loads and enhances faces (200-500ms duration)
- ✅ super-image works without dtype errors
- ✅ All 6 algorithms working

### Alternative: Package Upgrades (NOT RECOMMENDED)

Try upgrading packages to NumPy 2.x compatible versions:

```bash
# Check for NumPy 2.x compatible versions
pip install --upgrade basicsr --pre
pip install --upgrade gfpgan --pre
```

**Warning**: This may not work as packages may not have NumPy 2.x support yet.

## Testing After Fix

### Step 1: Verify NumPy Downgrade
```bash
python -c "import numpy; print(numpy.__version__)"
# Should output: 1.x.x (not 2.x)
```

### Step 2: Run Diagnostic Test
```bash
python -m scripts.diagnostic_test
```

### Expected Test Results (After NumPy Fix)

| Algorithm | Expected Status | Expected Duration |
|-----------|----------------|------------------|
| real_esrgan | PASS ✅ | 500-2000ms |
| super_image | PASS ✅ | 300-800ms |
| clahe | PASS ✅ | 50-400ms |
| white_balance | PASS ✅ | 50-150ms |
| exposure_correction | PASS ✅ | 50-200ms |
| face_enhancement | PASS ✅ | 200-500ms |

### Step 3: Run Full Test Suite
```bash
python -m scripts.test_runner
```

### Step 4: Review Results

Check `test_output/` for:
- Larger images (4x or 16x upscaling)
- Sharper details
- Better brightness (no dark images)
- Improved colors (white balance)
- Enhanced faces

## Current Test Summary

### Before NumPy Fix (Current State)

| Algorithm | Status | Notes |
|-----------|--------|-------|
| real_esrgan | PASS (placeholder) | No actual enhancement |
| super_image | FAIL | NumPy dtype error |
| clahe | PASS ✅ | Working |
| white_balance | PASS ✅ | Working |
| exposure_correction | PASS ✅ | Working |
| face_enhancement | FAIL | cv2 import fixed, GFPGAN fails |

**Pass Rate**: 4/6 (66%) with only placeholder algorithms passing

### After NumPy Fix (Expected)

| Algorithm | Expected Status | Enhancement |
|-----------|----------------|------------|
| real_esrgan | PASS ✅ | 4x upscaling |
| super_image | PASS ✅ | 4x upscaling |
| clahe | PASS ✅ | Low-light enhancement |
| white_balance | PASS ✅ | Color correction |
| exposure_correction | PASS ✅ | Brightness adjustment |
| face_enhancement | PASS ✅ | Face restoration |

**Expected Pass Rate**: 6/6 (100%)

## Dependencies Requiring NumPy 1.x

1. `basicsr` (Real-ESRGAN)
2. `gfpgan` (Face Enhancement)
3. `super-image` (Alternative Super-Resolution)

## Quick Fix Commands

```bash
# Single command to fix all NumPy issues
.venv\Scripts\activate && pip install "numpy<2.0" && python -m scripts.diagnostic_test
```

## Questions?

If issues persist after NumPy downgrade:

1. Check Python version: `python --version` (should be 3.8-3.11)
2. Check PyTorch version: `pip list | grep torch`
3. Clear pip cache: `pip cache purge`
4. Reinstall problem packages after NumPy downgrade

---

*Last Updated: January 9, 2026*
*Status: NumPy compatibility issue identified - Solution documented*
