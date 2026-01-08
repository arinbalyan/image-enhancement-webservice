"""
Real-ESRGAN super-resolution implementation.

Uses basicsr package and pre-trained models from Real-ESRGAN repo.
"""

from typing import Dict, Any, Optional
import torch
import numpy as np
from PIL import Image

from algorithms.base_enhancer import BaseEnhancer


class RealESRGANEnhancer(BaseEnhancer):
    """Real-ESRGAN super-resolution enhancer."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
        self.upscale_factor = config.get('scale', 4)
        self.model_name = config.get('model_name', 'RealESRGAN_x4plus')

    def load_model(self) -> None:
        """Load Real-ESRGAN model."""
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from basicsr.models.realesrgan_model import RealESRGANer

            model_path = self._download_model(self.model_name)

            if not Path(model_path).exists():
                print(f"Model file not found: {model_path}")
                print(f"Please download Real-ESRGAN model from:")
                print(f"  https://github.com/xinntao/Real-ESRGAN/releases")
                print(f"  Model: {self.model_name}.pth")
                print(f"\nFor now, using placeholder (no actual enhancement)")
                self.model = 'placeholder'
                self.model_loaded = True
                return

            # Load pre-trained weights
            print(f"Loading Real-ESRGAN model: {model_path}")

            self.model = RealESRGANer(
                scale=self.upscale_factor,
                model_path=model_path,
                model=self.model_name,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=False
            )

            self.model.device = self.device
            self.model.model.to(self.device)
            self.model.eval()
            self.model_loaded = True

            print(f"✅ Loaded Real-ESRGAN model successfully")

        except Exception as e:
            print(f"Warning: Failed to load Real-ESRGAN model: {e}")
            print("Using placeholder (no actual enhancement)")
            self.model = 'placeholder'
            self.model_loaded = True

    def _download_model(self, model_name: str) -> str:
        """
        Get path to downloaded model.

        Args:
            model_name: Name of the model

        Returns:
            str: Path to model file
        """
        from pathlib import Path
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)

        model_path = models_dir / f"{model_name}.pth"
        return str(model_path)

    def enhance(self, image: Any) -> Any:
        """
        Enhance image using Real-ESRGAN.

        Args:
            image: Input image (numpy array or PIL Image)

        Returns:
            Enhanced image (numpy array or PIL Image)
        """
        if not self.model_loaded:
            self.load_model()

        if self.model == 'placeholder':
            print("Warning: Real-ESRGAN using placeholder (no actual enhancement)")
            return image

        # Convert to numpy
        if isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image

        # Enhance using RealESRGANer
        try:
            output, _ = self.model.enhance(img, outscale=self.upscale_factor)
            return output
        except Exception as e:
            print(f"Real-ESRGAN enhancement error: {e}")
            return image

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
    print("Testing Real-ESRGAN Enhancer...")

    config = {'scale': 4, 'model_name': 'RealESRGAN_x4plus'}
    enhancer = RealESRGANEnhancer(config)

    print(f"Enhancer info: {enhancer.get_info()}")

    # Test analyze
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    analysis = enhancer.analyze(test_image)
    print(f"\nAnalysis: {analysis}")

    # Test enhance (would need actual model)
    # enhanced = enhancer.enhance(test_image)
    print("\nReal-ESRGAN enhancer test completed!")
