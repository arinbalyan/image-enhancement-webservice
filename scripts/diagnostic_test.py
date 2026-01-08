"""
Diagnostic test to check algorithm implementations
"""

import sys
import time
import gc
import threading
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from algorithms.manager import AlgorithmManager
from config.settings import get_settings


class TimeoutError(Exception):
    pass


class TimeoutException(Exception):
    pass


def timeout_handler():
    """Timeout handler for Windows-compatible timeout"""
    raise TimeoutException("Algorithm enhancement timed out")


def get_memory_usage_mb():
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0


def check_memory_and_warn():
    """Check memory usage and warn if too high."""
    mem_mb = get_memory_usage_mb()
    if mem_mb > 3000:  # 3GB threshold
        print(f"\n[WARNING] High memory usage ({mem_mb:.0f} MB)")
        print("Consider closing other applications or restarting the script")
        return False
    return True


def test_single_algorithm(manager, algorithm_name, test_image_path, timeout_seconds=60):
    """Test a single algorithm in isolation"""
    print(f"\n{'='*60}")
    print(f"Testing {algorithm_name}")
    print(f"{'='*60}")

    try:
        # Load algorithm
        print(f"[1/4] Loading {algorithm_name}...")
        manager.load_algorithm(algorithm_name)

        # Check if loaded
        if algorithm_name in manager.algorithms:
            enhancer = manager.algorithms[algorithm_name]
            print(f"  [OK] Algorithm loaded")
            print(f"  Model loaded: {enhancer.model_loaded}")
            print(f"  Model type: {type(enhancer.model)}")
        else:
            print(f"  [ERROR] Algorithm NOT loaded")
            return False

        # Load test image
        print(f"\n[2/4] Loading test image...")
        from utils.io import load_image
        image = load_image(test_image_path)
        print(f"  Image shape: {image.shape}")
        print(f"  Image dtype: {image.dtype}")

        # Analyze
        print(f"\n[3/4] Analyzing image...")
        analysis = enhancer.analyze(image)
        print(f"  Analysis result: {analysis}")

        # Enhance with timeout
        print(f"\n[4/4] Enhancing image... (timeout: {timeout_seconds}s)")
        start_time = time.time()

        # Windows-compatible timeout using threading
        timer = threading.Timer(timeout_seconds, timeout_handler)
        timer.start()

        try:
            enhanced_image = enhancer.enhance(image)
            timer.cancel()  # Cancel timer if enhancement completes
        except TimeoutException:
            timer.cancel()  # Cancel timer
            print(f"  [WARNING] TIMEOUT: Enhancement took longer than {timeout_seconds}s")
            print(f"  Skipping to next algorithm...")
            return False

        duration = (time.time() - start_time) * 1000
        print(f"  Duration: {duration:.2f}ms")
        print(f"  Output shape: {enhanced_image.shape}")
        print(f"  Output dtype: {enhanced_image.dtype}")

        # Check if image changed
        if isinstance(image, np.ndarray) and isinstance(enhanced_image, np.ndarray):
            if image.shape == enhanced_image.shape:
                diff = np.abs(image.astype(float) - enhanced_image.astype(float))
                mean_diff = np.mean(diff)
                max_diff = np.max(diff)

                print(f"\n  Image comparison:")
                print(f"    Mean difference: {mean_diff:.2f}")
                print(f"    Max difference: {max_diff:.2f}")

                if mean_diff < 1.0:
                    print(f"    [WARNING] Image barely changed (mean diff: {mean_diff:.2f})")
                    print(f"    [WARNING] Algorithm might not be working correctly!")
                else:
                    print(f"    [OK] Image changed significantly")
            else:
                print(f"  [OK] Image shape changed (upscale): {image.shape} -> {enhanced_image.shape}")

        return True

    except Exception as e:
        print(f"\n  [ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print("="*60)
    print("Algorithm Diagnostic Test")
    print("="*60)

    # Initial memory check
    initial_mem = get_memory_usage_mb()
    print(f"Initial memory usage: {initial_mem:.0f} MB")

    settings = get_settings()
    manager = AlgorithmManager(settings)

    # Check test images
    test_images_dir = Path("test_images")
    if not test_images_dir.exists():
        print(f"\n[ERROR] test_images directory not found!")
        print("Please add test images first.")
        return

    test_images = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png"))
    if not test_images:
        print(f"\n[ERROR] No test images found!")
        return

    test_image = str(test_images[0])
    print(f"\nUsing test image: {test_image}")

    # Test each algorithm with specific timeouts
    # super_image and face_enhancement are slow on CPU, use shorter timeout
    algorithms_to_test = [
        ('real_esrgan', 120),
        ('super_image', 90),
        ('clahe', 30),
        ('white_balance', 30),
        ('exposure_correction', 30),
        ('face_enhancement', 90)
    ]

    results = {}

    for alg, timeout in algorithms_to_test:
        # Check memory before loading
        mem_before = get_memory_usage_mb()
        print(f"\n--- Memory before {alg}: {mem_before:.0f} MB ---")
        
        if mem_before > 4000:  # 4GB threshold - too high!
            print(f"\n[ERROR] Memory too high ({mem_before:.0f} MB) - stopping to prevent crash")
            print("Please restart the diagnostic script")
            break
            
        success = test_single_algorithm(manager, alg, test_image, timeout_seconds=timeout)
        results[alg] = success

        # Unload after test to free memory
        manager.unload_algorithm(alg)
        
        # Force garbage collection
        gc.collect()
        
        # Wait for memory to be freed
        time.sleep(2)
        
        # Check memory after unload
        mem_after = get_memory_usage_mb()
        print(f"--- Memory after {alg}: {mem_after:.0f} MB ---")

        if mem_after > mem_before + 500:
            print(f"[WARNING] Memory not freed properly ({mem_after - mem_before:.0f} MB increase)")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    for alg, success in results.items():
        status = "[OK] PASS" if success else "[ERROR] FAIL"
        print(f"{status}: {alg}")

    # Final memory check
    final_mem = get_memory_usage_mb()
    print(f"\nFinal memory usage: {final_mem:.0f} MB")
    print(f"Memory change: {final_mem - initial_mem:.0f} MB")

    # Cleanup
    manager.cleanup()

    print("\nDiagnostic test completed!")


if __name__ == "__main__":
    main()
