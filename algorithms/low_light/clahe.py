"""
CLAHE-based low-light image enhancement.

Based on Low-Light-Image-Enhancement-CLAHE-Based repository.
"""

from typing import Dict, Any
import cv2
import numpy as np
from PIL import Image

from algorithms.base_enhancer import BaseEnhancer


class CLAHEEnhancer(BaseEnhancer):
    """CLAHE low-light enhancement."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.clip_limit = config.get('clip_limit', 2.0)
        self.tile_grid_size = config.get('tile_grid_size', (8, 8))

    def enhance(self, image: Any, upscale_factor: int = 4, enhance_level: str = 'medium') -> Any:
        """
        Enhance low-light image using CLAHE.

        Args:
            image: Input image (numpy array or PIL Image)
            upscale_factor: Factor by which to upscale the image (default: 4)
            enhance_level: Enhancement level (default: 'medium')

        Returns:
            Enhanced image (numpy array or PIL Image)
        """
        # Convert to numpy
        if isinstance(image, Image.Image):
            image = np.array(image)
        elif isinstance(image, np.ndarray):
            pass
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Convert to LAB color space (better for lighting)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

        # Apply CLAHE to L channel
        l, a, b = cv2.split(lab)

        # Create CLAHE object
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size
        )

        # Apply CLAHE
        l_enhanced = clahe.apply(l)

        # Merge channels back
        lab_enhanced = cv2.merge([l_enhanced, a, b])

        # Convert back to RGB
        enhanced_image = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

        return enhanced_image

    def analyze(self, image: Any) -> Dict[str, Any]:
        """
        Analyze image to determine if CLAHE enhancement is needed.

        Args:
            image: Input image

        Returns:
            Dictionary with analysis results
        """
        # Convert to numpy
        if isinstance(image, Image.Image):
            image = np.array(image)
        elif isinstance(image, np.ndarray):
            pass
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Calculate brightness
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mean_brightness = np.mean(gray) / 255.0

        # Check if low-light
        threshold = self.config.get('brightness_threshold', 0.3)
        is_low_light = mean_brightness < threshold

        # Check contrast
        contrast = gray.std() / 255.0
        is_low_contrast = contrast < 0.1

        # Determine if enhancement is needed
        needs_enhancement = is_low_light or is_low_contrast
        confidence = 0.9 if is_low_light else 0.7

        return {
            'needs_enhancement': needs_enhancement,
            'confidence': confidence,
            'reason': f'Low light (brightness: {mean_brightness:.3f})' if is_low_light else (
                'Low contrast' if is_low_contrast else 'Normal lighting'
            ),
            'brightness': float(mean_brightness),
            'contrast': float(contrast),
            'is_low_light': is_low_light
        }

    def load_model(self) -> None:
        """Load model (CLAHE doesn't need model)."""
        self.model = 'clahe'
        self.model_loaded = True

    def unload_model(self) -> None:
        """Unload model (CLAHE doesn't need unloading)."""
        self.model = None
        self.model_loaded = False


if __name__ == "__main__":
    print("Testing CLAHE Enhancer...")

    config = {'clip_limit': 2.0, 'tile_grid_size': (8, 8)}
    enhancer = CLAHEEnhancer(config)

    print(f"Enhancer info: {enhancer.get_info()}")

    # Test analyze
    # Create test image
    test_image = np.random.randint(50, 150, (100, 100, 3), dtype=np.uint8)
    analysis = enhancer.analyze(test_image)
    print(f"\nAnalysis: {analysis}")

    # Test enhance
    enhanced = enhancer.enhance(test_image)
    print(f"\nEnhanced image shape: {enhanced.shape}")

    print("\nCLAHE enhancer test completed!")
