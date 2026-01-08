# Known Issues & Solutions

## Current Status: January 9, 2026 - Evening

### CRITICAL ISSUE: No Visible Enhancement in test_runner ❌

**Issue Description**: When running `python -m scripts.test_runner`, images are not being enhanced. All 15 test images show:
- **No upscaling**: input_size == output_size (should be 4x larger)
- **No timing data**: All algorithms show 0.0ms duration (should be actual processing time)
- **User feedback**: "i couldnt see much changes" - images look the same

**Test Results Summary**:
```
Image 1: [1728, 2304, 3] -> [1728, 2304, 3]  ❌ Expected: [6912, 9216, 3]
Image 2: [2592, 4608, 3] -> [2592, 4608, 3]  ❌ Expected: [10368, 18432, 3]
...
All algorithms: Duration: 0.0ms  ❌ Expected: Real-ESRGAN ~100s, CLAHE ~162ms, etc.
```

### Root Causes Identified

#### 1. Timing Issue in manager.py (Lines 232-242)
```python
for name, enhancer in selected_algorithms:
    with log_operation(f"{name}"):  # Context manager handles timing
        start_time = time.time()  # ❌ Timing starts INSIDE context
        enhanced_image = enhancer.enhance(enhanced_image, **params)
        duration = (time.time() - start_time) * 1000  # ❌ Duration calculated INSIDE
```

**Problem**: The `log_operation()` context manager already handles timing. The manual timing calculation is redundant and results are being overwritten or ignored.

**Expected Fix**: Remove manual timing, rely on context manager's built-in timing.

#### 2. Parameter Not Used in real_esrgan.py (Lines 133-162)
```python
def enhance(self, image: np.ndarray, upscale_factor: int = 4, enhance_level: str = 'medium') -> np.ndarray:
    # ... validation code ...
    output, _ = self.model.enhance(img, outscale=self.upscale_factor)  # ❌ Using self.upscale_factor
```

**Problem**: The `upscale_factor` parameter is passed from manager.py but NEVER USED. The method always uses `self.upscale_factor` from __init__ (default=4).

**Expected Fix**: Use the passed `upscale_factor` parameter instead of `self.upscale_factor`.

#### 3. Incorrect Placeholder Check in real_esrgan.py (Line 146)
```python
if self.model == 'placeholder':  # ❌ This check is ALWAYS False for loaded models
    logger.warning(f"{self.name}: Using placeholder mode, returning original image")
    return image
```

**Problem**: This check compares a RealESRGANer object to the string 'placeholder'. For loaded models, this is always False, so the check never triggers.

**Expected Fix**: Check `if self.model is None` or use a separate placeholder flag.

### Why Diagnostic Test Works but test_runner Doesn't

**Diagnostic Test** (`scripts/diagnostic_test.py`):
- ✅ Real-ESRGAN: 99.8s, 4x upscale working
- ✅ super_image: 233.1s, 4x upscale working
- ✅ All 6 algorithms working correctly

**test_runner** (`scripts/test_runner.py`):
- ❌ All 15 images: no upscaling, 0.0ms duration

**Key Difference**:
- Diagnostic test calls `enhancer.enhance()` directly with hardcoded parameters
- test_runner calls through `manager.enhance_image()` which has parameter passing issues

### Previously Fixed Issues ✅

These issues have been resolved in earlier sessions:

#### 1. Real-ESRGAN Import - FIXED ✅
- **Error**: `cannot import name 'RealESRGANer' from 'basicsr.models.realesrgan_model'`
- **Fix**: Changed to `from realesrgan import RealESRGANer`

#### 2. NumPy Compatibility - FIXED ✅
- **Error**: RuntimeErrors with NumPy 2.x
- **Fix**: Downgraded NumPy from 2.2.6 to 1.26.4

#### 3. GFPGAN Loading - FIXED ✅
- **Error**: NumPy compatibility issues
- **Fix**: NumPy downgrade resolved all model loading issues

#### 4. Face Enhancement cv2 Import - FIXED ✅
- **Error**: `NameError: name 'cv2' is not defined`
- **Fix**: Added `import cv2` to face_enhancer.py

#### 5. Windows Emoji Encoding - FIXED ✅
- **Error**: Characters causing charmap errors on Windows
- **Fix**: Replaced all emojis with text equivalents ([OK], [ERROR], [WARNING])

#### 6. CLAHE tileGridSize Bug - FIXED ✅
- **Error**: Integer multiplication issue
- **Fix**: Fixed tile size calculation in CLAHE algorithm

#### 7. Windows Timeout Issue - FIXED ✅
- **Error**: SIGALRM not available on Windows
- **Fix**: Implemented threading-based timeout mechanism

## Action Items for Tomorrow

### Priority 1: Fix Enhancement Pipeline (CRITICAL)

#### Task 1.1: Fix Timing in manager.py
**File**: `algorithms/manager.py` (Lines 232-242)

**Current Code**:
```python
for name, enhancer in selected_algorithms:
    with log_operation(f"{name}"):
        start_time = time.time()
        enhanced_image = enhancer.enhance(enhanced_image, **params)
        duration = (time.time() - start_time) * 1000
```

**Fix**:
```python
for name, enhancer in selected_algorithms:
    with log_operation(f"{name}"):
        enhanced_image = enhancer.enhance(enhanced_image, **params)
        # log_operation handles timing automatically
```

#### Task 1.2: Fix Parameter Usage in real_esrgan.py
**File**: `algorithms/super_resolution/real_esrgan.py` (Lines 133-162)

**Current Code** (Line 158):
```python
output, _ = self.model.enhance(img, outscale=self.upscale_factor)
```

**Fix**:
```python
output, _ = self.model.enhance(img, outscale=upscale_factor)
```

#### Task 1.3: Fix Placeholder Check in real_esrgan.py
**File**: `algorithms/super_resolution/real_esrgan.py` (Line 146)

**Current Code**:
```python
if self.model == 'placeholder':
```

**Fix**:
```python
if self.model is None or self.placeholder_mode:
```

Add `self.placeholder_mode = False` to __init__ after model loading.

#### Task 1.4: Verify Other Algorithms
Check if other algorithms have similar issues:
- `super_image.py` - Does it use passed upscale_factor?
- `clahe.py` - Does it use passed enhance_level?
- `white_balance.py` - Does it use passed enhance_level?
- `exposure.py` - Does it use passed enhance_level?
- `face_enhancer.py` - Does it use passed enhance_level?

### Priority 2: Verify Fixes

1. **Run diagnostic test** to ensure individual algorithms still work:
   ```bash
   python -m scripts.diagnostic_test
   ```

2. **Run test_runner** to verify enhancement pipeline works:
   ```bash
   python -m scripts.test_runner
   ```

3. **Check output images** for actual upscaling:
   - File sizes should be 16x larger (4x width × 4x height)
   - Image dimensions should match expected upscale
   - Visual quality should show improvement

4. **Verify timing data** in logs:
   - Real-ESRGAN: ~100s (GPU)
   - super_image: ~233s (CPU)
   - CLAHE: ~162ms
   - white_balance: ~64ms
   - exposure_correction: ~58ms
   - face_enhancement: ~3s

### Priority 3: Documentation Updates

- Update KNOWN_ISSUES.md with fix status
- Update CONTEXT.md with current findings
- Create CONTRIBUTING.md with setup instructions
- Update README.md with known workarounds

## Expected Results After Fixes

### Test Output (After Fix)
```
Image 1: [1728, 2304, 3] -> [6912, 9216, 3]  [OK]
  - super_resolution | Duration: 99.8s | 4x upscale
  - low_light | Duration: 0.162s | CLAHE enhancement
  - color_correction | Duration: 0.064s | White balance
  - face_analysis | Duration: 3.0s | Face enhancement
```

### Success Criteria
- ✅ All images upscaled correctly (4x or 16x)
- ✅ All algorithms show realistic timing data
- ✅ Enhanced images show visible improvement
- ✅ File sizes reflect upscaling (16x increase)
- ✅ No "0.0ms" durations in logs

## Dependencies

### Currently Installed
```
torch==2.0.0+cu118
torchvision==0.15.0+cu118
numpy==1.26.4
basicsr==1.4.2
realesrgan==0.3.0
gfpgan==1.3.8
facexlib==0.3.0
super-image==0.2.0
opencv-python==4.12.0.88
```

### GPU Support
- ✅ NVIDIA GeForce GTX 1650 Ti
- ✅ CUDA 11.8
- ✅ Real-ESRGAN using GPU
- ✅ GFPGAN using GPU

## Quick Reference

### Files to Modify
1. `algorithms/manager.py` - Fix timing in enhancement loop
2. `algorithms/super_resolution/real_esrgan.py` - Use passed upscale_factor parameter
3. `algorithms/super_resolution/super_image.py` - Verify parameter usage
4. `algorithms/low_light/clahe.py` - Verify parameter usage
5. `algorithms/color_correction/white_balance.py` - Verify parameter usage
6. `algorithms/color_correction/exposure.py` - Verify parameter usage
7. `algorithms/face_analysis/face_enhancer.py` - Verify parameter usage

### Testing Commands
```bash
# Diagnostic test (individual algorithms)
python -m scripts.diagnostic_test

# Full batch test (enhancement pipeline)
python -m scripts.test_runner

# Check output dimensions
python -c "from PIL import Image; img = Image.open('test_output/image.jpg'); print(img.size)"
```

### Log Locations
- `logs/enhancement.log` - Enhancement operations
- `logs/analysis.log` - Image analysis results
- `logs/error.log` - Errors and warnings

---

## Questions?

If issues persist after implementing fixes:

1. Check if GPU is being used during test_runner (not just diagnostic test)
2. Verify models are actually loaded (not in placeholder mode)
3. Add debug logging to see what params are passed to each algorithm
4. Check if enhanced images are being saved correctly
5. Verify file sizes of output images vs input images

---

*Last Updated: January 9, 2026 (Evening)*
*Status: CRITICAL ISSUE - No enhancement in test_runner, fix plan documented*
*Next Action: Implement fixes in manager.py and real_esrgan.py*
