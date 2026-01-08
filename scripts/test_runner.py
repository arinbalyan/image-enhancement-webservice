"""
Test runner for image enhancement.

Processes images from test_images/ and outputs to test_output/.
"""

import sys
import time
from pathlib import Path
from config.settings import get_settings
from algorithms.manager import AlgorithmManager
from utils.io import list_images, ensure_directory
from utils.logger import get_logger


def run_tests(input_dir: str = 'test_images', output_dir: str = 'test_output'):
    """
    Run enhancement tests on all images in test_images/.

    Args:
        input_dir: Input directory with test images
        output_dir: Output directory for enhanced images

    Returns:
        Dictionary with test results
    """
    print("=" * 60)
    print("Image Enhancement Test Runner")
    print("=" * 60)

    # Initialize
    settings = get_settings()
    manager = AlgorithmManager(settings)
    logger = get_logger(settings)

    # Ensure directories exist
    ensure_directory(input_dir, clear=False)
    ensure_directory(output_dir, clear=False)

    # List test images
    images = list_images(input_dir, recursive=False)

    if not images:
        print(f"\n[WARNING] No test images found in: {input_dir}")
        print(f"Please add test images to: {input_dir}/")
        return {
            'total': 0,
            'success': 0,
            'failed': 0,
            'success_rate': 0.0
        }

    print(f"\n[DIR] Test directory: {input_dir}")
    print(f"[DIR] Output directory: {output_dir}")
    print(f"[IMAGE] Found {len(images)} test image(s)")
    print("=" * 60)

    # Process each image
    results = {
        'total': len(images),
        'success': 0,
        'failed': 0,
        'errors': [],
        'timings': []
    }

    overall_start = time.time()

    for i, image_path in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] Processing: {Path(image_path).name}")

        try:
            # Log workflow start
            logger.log_enhancement(
                image_name=Path(image_path).name,
                algorithm='workflow_start',
                duration_ms=0,
                details={'input_path': image_path, 'output_dir': output_dir}
            )

            # Analyze image first
            analysis = manager.image_analyzer.analyze(image_path)
            logger.log_analysis(Path(image_path).name, analysis)

            # Print recommendations to console
            print(f"  Recommendations: {len(analysis['recommendations'])} algorithm(s)")
            for rec in analysis['recommendations']:
                print(f"    - {rec['algorithm']} (confidence: {rec['confidence']:.2f})")

            # Enhance image
            print("  Enhancing...")
            start_time = time.time()

            output_path = manager.enhance_image(
                image_path=image_path,
                output_dir=output_dir,
                algorithms=None,  # Auto-detect
                preserve_original=True
            )

            duration = time.time() - start_time
            results['timings'].append({
                'image': Path(image_path).name,
                'duration': duration
            })

            print(f"  [OK] Success: {Path(output_path).name}")
            print(f"  [TIME] Duration: {duration:.2f}s")

            # Update results
            results['success'] += 1

        except Exception as e:
            print(f"  [ERROR] Failed: {e}")
            results['failed'] += 1
            results['errors'].append({
                'image': Path(image_path).name,
                'error': str(e)
            })

    overall_duration = time.time() - overall_start

    # Print summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    print(f"Total images: {results['total']}")
    print(f"Success: {results['success']}")
    print(f"Failed: {results['failed']}")

    if results['total'] > 0:
        success_rate = (results['success'] / results['total']) * 100
        print(f"Success rate: {success_rate:.1f}%")

    if results['timings']:
        avg_duration = sum(t['duration'] for t in results['timings']) / len(results['timings'])
        min_duration = min(t['duration'] for t in results['timings'])
        max_duration = max(t['duration'] for t in results['timings'])
        print(f"Average time per image: {avg_duration:.2f}s")
        print(f"Fastest image: {min_duration:.2f}s")
        print(f"Slowest image: {max_duration:.2f}s")

    print(f"Total processing time: {overall_duration:.2f}s")
    print(f"\nEnhanced images saved to: {output_dir}")

    # Print errors if any
    if results['errors']:
        print("\n" + "=" * 60)
        print("Errors")
        print("=" * 60)
        for error in results['errors']:
            print(f"  {error['image']}: {error['error']}")

    print("\n" + "=" * 60)
    print("Test runner completed!")
    print("=" * 60)

    # Cleanup
    manager.cleanup()

    return results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Run image enhancement tests'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='test_images',
        help='Input directory with test images (default: test_images)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='test_output',
        help='Output directory for enhanced images (default: test_output)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()

    # Check if input directory exists
    if not Path(args.input).exists():
        print(f"Error: Input directory does not exist: {args.input}")
        print(f"Creating directory: {args.input_dir}")
        Path(args.input).mkdir(parents=True, exist_ok=True)
        print(f"Please add test images to: {args.input}/")
        print("\nTest runner waiting for images...")
        return

    # Run tests
    results = run_tests(
        input_dir=args.input,
        output_dir=args.output
    )

    # Exit with appropriate code
    sys.exit(0 if results['failed'] == 0 else 1)


if __name__ == "__main__":
    main()
