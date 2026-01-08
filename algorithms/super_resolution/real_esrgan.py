"""
Real-ESRGAN super-resolution implementation.

Uses basicsr package and pre-trained models from Real-ESRGAN repo.
"""

from typing import Dict, Any, Optional
from pathlib import Path
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
            from realesrgan import RealESRGANer
            import cv2
            from basicsr.utils import img2tensor
            import torch

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

            # Determine device
            if torch.cuda.is_available() and self.config.get('use_gpu', True):
                self.device = torch.device('cuda')
                print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device('cpu')
                print(f"Using CPU (CUDA not available)")

            # Load pre-trained weights
            print(f"Loading Real-ESRGAN model: {model_path}")

            # Load weights first to check structure
            loadnet = torch.load(model_path, map_location=self.device)

            # Create model architecture based on model name
            # RealESRGAN_x4plus uses RRDBNet architecture
            if self.model_name == 'RealESRGAN_x4plus':
                model = RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=23,
                    num_grow_ch=32,
                    scale=self.upscale_factor
                )
            else:
                # Default to RRDBNet for other models
                model = RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=23,
                    num_grow_ch=32,
                    scale=self.upscale_factor
                )

            # Load state dict from checkpoint
            # Real-ESRGAN checkpoints use 'params' or 'params_ema' keys
            if 'params_ema' in loadnet:
                model.load_state_dict(loadnet['params_ema'])
            elif 'params' in loadnet:
                model.load_state_dict(loadnet['params'])
            else:
                model.load_state_dict(loadnet)
            
            model = model.to(self.device)
            model.eval()
            
            # Create RealESRGANer
            self.model = RealESRGANer(
                scale=self.upscale_factor,
                model_path=model_path,
                model=model,
                tile=400,  # Tile processing for memory efficiency
                tile_pad=10,
                pre_pad=10,
                half=False
            )
            
            self.model_loaded = True
            print(f"[OK] Loaded Real-ESRGAN model successfully")

        except Exception as e:
            import traceback
            print(f"Warning: Failed to load Real-ESRGAN model: {e}")
            traceback.print_exc()
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
