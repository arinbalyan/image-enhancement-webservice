"""
Diagnostic test to check algorithm implementations
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from algorithms.manager import AlgorithmManager
from config.settings import get_settings


def test_single_algorithm(manager, algorithm_name, test_image_path):
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
            print(f"  ✅ Algorithm loaded")
            print(f"  Model loaded: {enhancer.model_loaded}")
            print(f"  Model type: {type(enhancer.model)}")
        else:
            print(f"  ❌ Algorithm NOT loaded")
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

        # Enhance
        print(f"\n[4/4] Enhancing image...")
        start_time = time.time()

        enhanced_image = enhancer.enhance(image)

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
                    print(f"    ⚠️  WARNING: Image barely changed (mean diff: {mean_diff:.2f})")
                    print(f"    ⚠️  Algorithm might not be working correctly!")
                else:
                    print(f"    ✅ Image changed significantly")
            else:
                print(f"  ❌ Image shape changed unexpectedly")

        return True

    except Exception as e:
        print(f"\n  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print("="*60)
    print("Algorithm Diagnostic Test")
    print("="*60)

    settings = get_settings()
    manager = AlgorithmManager(settings)

    # Check test images
    test_images_dir = Path("test_images")
    if not test_images_dir.exists():
        print(f"\n❌ ERROR: test_images directory not found!")
        print("Please add test images first.")
        return

    test_images = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png"))
    if not test_images:
        print(f"\n❌ ERROR: No test images found!")
        return

    test_image = str(test_images[0])
    print(f"\nUsing test image: {test_image}")

    # Test each algorithm
    algorithms_to_test = [
        'real_esrgan',
        'super_image',
        'clahe',
        'white_balance',
        'exposure_correction',
        'face_enhancement'
    ]

    results = {}

    for alg in algorithms_to_test:
        success = test_single_algorithm(manager, alg, test_image)
        results[alg] = success

        # Unload after test to free memory
        manager.unload_algorithm(alg)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    for alg, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {alg}")

    # Cleanup
    manager.cleanup()

    print("\nDiagnostic test completed!")


if __name__ == "__main__":
    main()
