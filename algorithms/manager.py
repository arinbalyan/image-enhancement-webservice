"""
Algorithm manager for orchestrating image enhancement.

Selects and executes appropriate enhancement algorithms based on image analysis.
"""

import time
from typing import Dict, Any, List, Optional
from pathlib import Path

from config.settings import get_settings, get_algorithm_config
from utils.image_analyzer import ImageAnalyzer
from utils.logger import get_logger
from utils.io import load_image, save_image, get_unique_filename
from algorithms.base_enhancer import BaseEnhancer


class AlgorithmManager:
    """Orchestrates enhancement algorithms."""

    def __init__(self, settings=None):
        """
        Initialize algorithm manager.

        Args:
            settings: Optional settings object
        """
        if settings is None:
            settings = get_settings()

        self.settings = settings
        self.logger = get_logger(settings)
        self.image_analyzer = ImageAnalyzer(settings)

        # Store loaded algorithms
        self.algorithms: Dict[str, BaseEnhancer] = {}
        self.loaded_algorithms: set = set()

    def load_algorithm(self, algorithm_name: str) -> None:
        """
        Load a specific algorithm.

        Args:
            algorithm_name: Name of algorithm to load
        """
        if algorithm_name in self.loaded_algorithms:
            return

        self.logger.log_enhancement(
            image_name='system',
            algorithm=f'load_{algorithm_name}',
            duration_ms=0,
            details={'action': 'loading_model'}
        )

        try:
            # Import and instantiate algorithm
            enhancer = self._get_enhancer_instance(algorithm_name)
            enhancer.load_model()
            self.algorithms[algorithm_name] = enhancer
            self.loaded_algorithms.add(algorithm_name)

            self.logger.log_enhancement(
                image_name='system',
                algorithm=f'load_{algorithm_name}',
                duration_ms=0,
                details={'action': 'model_loaded', 'success': True}
            )

        except Exception as e:
            self.logger.log_error(
                image_name='system',
                error=f'Failed to load {algorithm_name}: {e}',
                details={'algorithm': algorithm_name}
            )

    def unload_algorithm(self, algorithm_name: str) -> None:
        """
        Unload a specific algorithm to free memory.

        Args:
            algorithm_name: Name of algorithm to unload
        """
        if algorithm_name not in self.loaded_algorithms:
            return

        try:
            enhancer = self.algorithms.get(algorithm_name)
            if enhancer:
                enhancer.unload_model()

            del self.algorithms[algorithm_name]
            self.loaded_algorithms.discard(algorithm_name)

            self.logger.log_enhancement(
                image_name='system',
                algorithm=f'unload_{algorithm_name}',
                duration_ms=0,
                details={'action': 'model_unloaded', 'success': True}
            )

        except Exception as e:
            self.logger.log_error(
                image_name='system',
                error=f'Failed to unload {algorithm_name}: {e}',
                details={'algorithm': algorithm_name}
            )

    def select_algorithms(
        self,
        analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Select appropriate algorithms based on image analysis.

        Args:
            analysis: Image analysis results

        Returns:
            List of algorithm configurations to apply
        """
        # Get recommendations from analysis
        recommendations = analysis.get('recommendations', [])

        # Filter enabled algorithms
        selected = []
        for rec in recommendations:
            if rec.get('enabled', False):
                selected.append(rec)

        return selected

    def enhance_image(
        self,
        image_path: str,
        output_dir: str,
        algorithms: Optional[List[str]] = None,
        preserve_original: bool = True
    ) -> str:
        """
        Enhance an image with selected algorithms.

        Args:
            image_path: Path to input image
            output_dir: Directory to save enhanced image
            algorithms: List of algorithms to use (None for auto)
            preserve_original: If True, add suffix to filename

        Returns:
            str: Path to enhanced image
        """
        start_time = time.time()

        # Get image filename
        input_path = Path(image_path)
        image_name = input_path.name

        self.logger.log_enhancement(
            image_name=image_name,
            algorithm='workflow_start',
            duration_ms=0,
            details={'input_path': image_path, 'output_dir': output_dir}
        )

        try:
            # Load image
            image = load_image(image_path)

            # Analyze image if algorithms not specified
            if algorithms is None:
                self.logger.log_analysis(image_name, analysis={'status': 'starting'})

                analysis = self.image_analyzer.analyze(image_path)
                self.logger.log_analysis(image_name, analysis)

                # Select algorithms
                selected_algorithms = self.select_algorithms(analysis)
            else:
                # Use specified algorithms
                selected_algorithms = [
                    {
                        'algorithm': alg,
                        'sub_algorithm': alg,
                        'confidence': 1.0,
                        'priority': 1,
                        'parameters': {},
                        'enabled': True
                    }
                    for alg in algorithms
                ]

            # Log selected algorithms
            algorithms_used = [alg['algorithm'] for alg in selected_algorithms]
            self.logger.log_enhancement(
                image_name=image_name,
                algorithm='algorithm_selection',
                duration_ms=0,
                details={'algorithms': algorithms_used, 'count': len(algorithms_used)}
            )

            # Load required algorithms
            for alg_config in selected_algorithms:
                alg_name = alg_config['sub_algorithm']
                if alg_name not in self.loaded_algorithms:
                    self.load_algorithm(alg_name)

            # Apply enhancements sequentially (for now)
            enhanced_image = image.copy() if hasattr(image, 'copy') else image

            for alg_config in selected_algorithms:
                alg_name = alg_config['sub_algorithm']

                if alg_name not in self.algorithms:
                    self.logger.log_error(
                        image_name=image_name,
                        error=f'Algorithm {alg_name} not loaded',
                        details={'algorithm': alg_name}
                    )
                    continue

                enhancer = self.algorithms[alg_name]

                # Get parameters from algorithm config
                alg_params = alg_config.get('parameters', {})
                upscale_factor = alg_params.get('scale', 4)
                enhance_level = alg_params.get('enhance_level', 'medium')

                # Apply enhancement
                with self.logger.log_operation(
                    operation_name=alg_name,
                    image_name=image_name,
                    algorithm=alg_name,
                    details=alg_params
                ) as (log_success, log_error):
                    try:
                        # Start timing
                        alg_start = time.time()

                        # Apply enhancement with parameters
                        enhanced_image = enhancer.enhance(enhanced_image, upscale_factor=upscale_factor, enhance_level=enhance_level)

                        # Log success
                        duration_ms = (time.time() - alg_start) * 1000
                        log_success({
                            'input_size': image.shape if hasattr(image, 'shape') else 'N/A',
                            'output_size': enhanced_image.shape if hasattr(enhanced_image, 'shape') else 'N/A',
                            'upscale_factor': upscale_factor,
                            'enhance_level': enhance_level
                        })

                    except Exception as e:
                        log_error(str(e))
                        raise

            # Generate output filename
            if preserve_original:
                stem = input_path.stem
                extension = input_path.suffix
                output_filename = f"{stem}_enhanced{extension}"
            else:
                output_filename = input_path.name

            output_path = Path(output_dir) / output_filename

            # Save enhanced image
            save_image(
                enhanced_image,
                str(output_path),
                quality=self.settings.processing.quality,
                format=self.settings.processing.output_format
            )

            # Calculate total duration
            total_duration_ms = (time.time() - start_time) * 1000

            # Log workflow completion
            self.logger.log_workflow(
                image_name=image_name,
                total_duration_ms=total_duration_ms,
                algorithms_used=algorithms_used
            )

            return str(output_path)

        except Exception as e:
            self.logger.log_error(
                image_name=image_name,
                error=f'Enhancement failed: {e}',
                details={'input_path': image_path, 'exception_type': type(e).__name__}
            )
            raise

    def _get_enhancer_instance(self, algorithm_name: str) -> BaseEnhancer:
        """
        Get enhancer instance for algorithm.

        Args:
            algorithm_name: Name of algorithm

        Returns:
            BaseEnhancer: Algorithm instance

        Raises:
            ValueError: If algorithm not found
        """
        # Map algorithm names to classes
        try:
            if algorithm_name == 'real_esrgan':
                from algorithms.super_resolution.real_esrgan import RealESRGANEnhancer
                return RealESRGANEnhancer(self._get_algorithm_config('super_resolution'))
            elif algorithm_name == 'super_image':
                from algorithms.super_resolution.super_image import SuperImageEnhancer
                return SuperImageEnhancer(self._get_algorithm_config('super_resolution'))
            elif algorithm_name == 'clahe':
                from algorithms.low_light.clahe import CLAHEEnhancer
                return CLAHEEnhancer(self._get_algorithm_config('low_light'))
            elif algorithm_name == 'white_balance':
                from algorithms.color_correction.white_balance import WhiteBalanceEnhancer
                return WhiteBalanceEnhancer(self._get_algorithm_config('color_correction'))
            elif algorithm_name == 'exposure_correction':
                from algorithms.color_correction.exposure import ExposureCorrectionEnhancer
                return ExposureCorrectionEnhancer(self._get_algorithm_config('color_correction'))
            elif algorithm_name == 'face_enhancement':
                from algorithms.face_analysis.face_enhancer import FaceEnhancer
                return FaceEnhancer(self._get_algorithm_config('face_analysis'))
            else:
                # Fallback to dummy enhancer for unimplemented algorithms
                from algorithms.base_enhancer import DummyEnhancer
                return DummyEnhancer({})
        except ImportError as e:
            print(f"Warning: Could not import {algorithm_name}: {e}")
            print(f"Using dummy enhancer as fallback")
            from algorithms.base_enhancer import DummyEnhancer
            return DummyEnhancer({})

    def _get_algorithm_config(self, algorithm_type: str) -> Dict[str, Any]:
        """
        Get configuration for algorithm type.

        Args:
            algorithm_type: Type of algorithm

        Returns:
            Dict: Configuration
        """
        from config.settings import get_algorithm_config
        config = get_algorithm_config(self.settings, algorithm_type)

        # Convert to dict
        return {
            'name': config.name,
            'enabled': config.enabled,
            'priority': config.priority,
            'parameters': config.parameters
        }

    def get_available_algorithms(self) -> List[str]:
        """
        Get list of available algorithms.

        Returns:
            List of algorithm names
        """
        return [
            'super_resolution',
            'low_light',
            'color_correction',
            'face_analysis'
        ]

    def get_status(self) -> Dict[str, Any]:
        """
        Get manager status.

        Returns:
            Dictionary with status information
        """
        return {
            'loaded_algorithms': list(self.loaded_algorithms),
            'available_algorithms': self.get_available_algorithms(),
            'total_algorithms': len(self.algorithms),
            'device': self.settings.processing.gpu_device if self.settings.processing.enable_gpu else 'cpu'
        }

    def cleanup(self) -> None:
        """Unload all algorithms and clean up resources."""
        for algorithm_name in list(self.loaded_algorithms):
            self.unload_algorithm(algorithm_name)


if __name__ == "__main__":
    # Test algorithm manager
    print("Testing Algorithm Manager...")

    manager = AlgorithmManager()

    # Test algorithm loading
    print("\n1. Testing algorithm loading...")
    manager.load_algorithm('super_resolution')
    print(f"   Loaded algorithms: {manager.get_status()['loaded_algorithms']}")

    # Test algorithm selection
    print("\n2. Testing algorithm selection...")
    test_analysis = {
        'resolution': {'is_small': True, 'recommended_upscale': 16},
        'brightness': {'is_low_light': True, 'mean': 0.2},
        'faces': {'has_faces': True, 'count': 2},
        'color': {'needs_white_balance': True},
        'noise': {'is_noisy': False},
        'recommendations': [
            {
                'algorithm': 'super_resolution',
                'sub_algorithm': 'real_esrgan',
                'confidence': 1.0,
                'priority': 1,
                'parameters': {},
                'enabled': True
            },
            {
                'algorithm': 'low_light',
                'sub_algorithm': 'clahe',
                'confidence': 0.9,
                'priority': 2,
                'parameters': {},
                'enabled': True
            }
        ]
    }

    selected = manager.select_algorithms(test_analysis)
    print(f"   Selected {len(selected)} algorithms:")
    for alg in selected:
        print(f"     - {alg['algorithm']} (confidence: {alg['confidence']:.2f})")

    # Test get status
    print("\n3. Testing status...")
    status = manager.get_status()
    print(f"   Available algorithms: {status['available_algorithms']}")
    print(f"   Loaded algorithms: {status['loaded_algorithms']}")
    print(f"   Device: {status['device']}")

    # Test cleanup
    print("\n4. Testing cleanup...")
    manager.cleanup()
    print("   All algorithms unloaded")

    print("\nAlgorithm manager test completed!")
