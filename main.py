"""
Main entry point for Image Enhancement Web Service.

Provides CLI interface for processing images.
"""

import argparse
from pathlib import Path
from config.settings import get_settings, create_default_env_file
from algorithms.manager import AlgorithmManager
from utils.io import list_images, ensure_directory
from utils.logger import get_logger


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Image Enhancement Web Service - Enhance images with AI algorithms'
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup configuration')
    setup_parser.add_argument(
        '--create-env',
        action='store_true',
        help='Create default .env file'
    )

    # Process command
    process_parser = subparsers.add_parser('process', help='Process images')
    process_parser.add_argument(
        'input',
        type=str,
        help='Input image or directory'
    )
    process_parser.add_argument(
        '-o', '--output',
        type=str,
        default='test_output',
        help='Output directory (default: test_output)'
    )
    process_parser.add_argument(
        '-a', '--algorithms',
        type=str,
        nargs='+',
        help='Specific algorithms to use (default: auto)'
    )
    process_parser.add_argument(
        '--preserve-original',
        action='store_true',
        help='Preserve original filename'
    )
    process_parser.add_argument(
        '--recursive',
        action='store_true',
        help='Process images recursively'
    )

    # Test command
    test_parser = subparsers.add_parser('test', help='Run tests on test_images/')
    test_parser.add_argument(
        '--input',
        type=str,
        default='test_images',
        help='Test input directory (default: test_images)'
    )
    test_parser.add_argument(
        '--output',
        type=str,
        default='test_output',
        help='Test output directory (default: test_output)'
    )

    # Status command
    status_parser = subparsers.add_parser('status', help='Show system status')

    args = parser.parse_args()

    # Handle setup command
    if args.command == 'setup':
        handle_setup(args)

    # Handle process command
    elif args.command == 'process':
        handle_process(args)

    # Handle test command
    elif args.command == 'test':
        handle_test(args)

    # Handle status command
    elif args.command == 'status':
        handle_status(args)

    else:
        parser.print_help()


def handle_setup(args):
    """Handle setup command."""
    print("Setting up Image Enhancement Web Service...")

    if args.create_env:
        create_default_env_file()
        print("Created default .env file")
        print("Please edit .env to configure the service")
    else:
        print("Use --create-env to create .env file")


def handle_process(args):
    """Handle process command."""
    print(f"Processing images from: {args.input}")

    # Initialize manager
    settings = get_settings()
    manager = AlgorithmManager(settings)
    logger = get_logger(settings)

    # Check if input is file or directory
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: Input path does not exist: {args.input}")
        return

    # Collect images to process
    if input_path.is_file():
        images = [str(input_path)]
    else:
        images = list_images(
            str(input_path),
            recursive=args.recursive
        )

    if not images:
        print("No images found to process")
        return

    print(f"Found {len(images)} image(s) to process")

    # Ensure output directory exists
    ensure_directory(args.output)

    # Process each image
    success_count = 0
    failure_count = 0

    for i, image_path in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] Processing: {Path(image_path).name}")

        try:
            output_path = manager.enhance_image(
                image_path=image_path,
                output_dir=args.output,
                algorithms=args.algorithms,
                preserve_original=not args.preserve_original
            )

            print(f"  -> Enhanced: {output_path}")
            success_count += 1

        except Exception as e:
            print(f"  -> Failed: {e}")
            logger.log_error(
                image_name=Path(image_path).name,
                error=str(e)
            )
            failure_count += 1

    # Print summary
    print(f"\n=== Summary ===")
    print(f"Processed: {len(images)} images")
    print(f"Success: {success_count}")
    print(f"Failed: {failure_count}")
    print(f"Output directory: {args.output}")


def handle_test(args):
    """Handle test command."""
    print("Running image enhancement tests...")

    # Initialize manager
    settings = get_settings()
    manager = AlgorithmManager(settings)

    # Ensure test directories exist
    ensure_directory(args.input, clear=False)
    ensure_directory(args.output, clear=False)

    # List test images
    images = list_images(args.input, recursive=False)

    if not images:
        print(f"No test images found in: {args.input}")
        print(f"Please add test images to {args.input}/")
        return

    print(f"Found {len(images)} test image(s)")

    # Process all test images
    success_count = 0
    failure_count = 0

    for i, image_path in enumerate(images, 1):
        image_name = Path(image_path).name
        print(f"\n[{i}/{len(images)}] Testing: {image_name}")

        try:
            output_path = manager.enhance_image(
                image_path=image_path,
                output_dir=args.output,
                algorithms=None,  # Auto-detect
                preserve_original=True
            )

            print(f"  -> Success: {Path(output_path).name}")
            success_count += 1

        except Exception as e:
            print(f"  -> Failed: {e}")
            failure_count += 1

    # Print summary
    print(f"\n=== Test Results ===")
    print(f"Total images: {len(images)}")
    print(f"Success: {success_count}")
    print(f"Failed: {failure_count}")
    print(f"Success rate: {success_count/len(images)*100:.1f}%")
    print(f"\nEnhanced images saved to: {args.output}")


def handle_status(args):
    """Handle status command."""
    print("Image Enhancement Web Service Status\n")

    # Get settings
    settings = get_settings()

    # Display configuration
    print("=== Configuration ===")
    print(f"Project root: {settings.project_root}")
    print(f"API: {settings.api.host}:{settings.api.port}")
    print(f"Debug mode: {settings.api.debug}")
    print(f"GPU enabled: {settings.processing.enable_gpu}")
    print(f"GPU device: {settings.processing.gpu_device}")
    print(f"Log level: {settings.logging.log_level}")

    # Get algorithm manager status
    manager = AlgorithmManager(settings)
    status = manager.get_status()

    print("\n=== Algorithms ===")
    print(f"Available: {len(status['available_algorithms'])}")
    print(f"Loaded: {len(status['loaded_algorithms'])}")
    print(f"Device: {status['device']}")

    print("\n=== Available Algorithms ===")
    for alg in status['available_algorithms']:
        config = get_algorithm_config(settings, alg)
        print(f"  {alg}")
        print(f"    Enabled: {config.enabled}")
        print(f"    Priority: {config.priority}")


if __name__ == "__main__":
    main()
