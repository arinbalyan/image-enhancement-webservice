"""
super-image library integration for super-resolution.

Uses pre-trained models from HuggingFace.
"""

from typing import Dict, Any
import torch
import numpy as np
from PIL import Image

from algorithms.base_enhancer import BaseEnhancer


class SuperImageEnhancer(BaseEnhancer):
    """Super-image super-resolution enhancer."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.upscale_factor = config.get('scale', 4)
        self.model_name = config.get('model_name', 'edsr')

    def load_model(self) -> None:
        """Load super-image model from HuggingFace."""
        try:
            from super_image import EdsrModel

            # Load pre-trained model
            self.model = EdsrModel.from_pretrained(
                f"eugenesiow/{self.model_name}",
                scale=self.upscale_factor
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True

            print(f"Loaded super-image model: {self.model_name} (scale: {self.upscale_factor}x)")

        except Exception as e:
            raise Exception(f"Failed to load super-image model: {e}")

    def enhance(self, image: Any) -> Any:
        """
        Enhance image using super-image model.

        Args:
            image: Input image (numpy array or PIL Image)

        Returns:
            Enhanced image (numpy array or PIL Image)
        """
        if not self.model_loaded:
            self.load_model()

        # Convert to PIL
        pil_image = self._convert_to_pil(image)

        # Convert to tensor (super-image expects specific format)
        # super-image uses ImageLoader internally
        from super_image import ImageLoader
        img_tensor = ImageLoader.load_image(pil_image)
        img_tensor = img_tensor.to(self.device)

        # Inference
        with torch.no_grad():
            output = self.model(img_tensor)

        # Get PIL Image from output
        if isinstance(output, list):
            enhanced_image = ImageLoader.save_image(output[0])
        else:
            enhanced_image = ImageLoader.save_image(output)

        return enhanced_image

    def analyze(self, image: Any) -> Dict[str, Any]:
        """
        Analyze image to determine if super-resolution is needed.

        Args:
            image: Input image

        Returns:
            Dictionary with analysis results
        """
        # Get image size
        if isinstance(image, np.ndarray):
            h, w = image.shape[:2]
        else:
            w, h = image.size

        shortest_edge = min(h, w)

        # Determine if super-resolution is needed
        threshold = self.config.get('threshold', 512)
        is_small = shortest_edge < threshold

        needs_enhancement = is_small
        confidence = 1.0 if is_small else 0.7

        return {
            'needs_enhancement': needs_enhancement,
            'confidence': confidence,
            'reason': f'Small image ({shortest_edge}px)' if is_small else 'Normal size',
            'shortest_edge': shortest_edge,
            'is_small': is_small
        }

    def unload_model(self) -> None:
        """Unload model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
            self.model_loaded = False

            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    print("Testing Super-Image Enhancer...")

    config = {'scale': 4, 'model_name': 'edsr-base'}
    enhancer = SuperImageEnhancer(config)

    print(f"Enhancer info: {enhancer.get_info()}")

    # Test analyze
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    analysis = enhancer.analyze(test_image)
    print(f"\nAnalysis: {analysis}")

    # Test enhance (would need actual model)
    # enhanced = enhancer.enhance(test_image)
    # print("\nSuper-image enhancer test completed!")
