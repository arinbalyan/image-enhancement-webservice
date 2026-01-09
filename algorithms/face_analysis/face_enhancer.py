"""
Face enhancement using GFPGAN.

Based on face enhancement in Real-ESRGAN repo.
"""

from typing import Dict, Any
import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path

from algorithms.base_enhancer import BaseEnhancer


class FaceEnhancer(BaseEnhancer):
    """Face enhancement using GFPGAN."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
        self.detection_threshold = config.get('detection_threshold', 0.7)

    def load_model(self) -> None:
        """Load GFPGAN model."""
        try:
            from gfpgan import GFPGANer

            # Check for model weights
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)

            model_path = models_dir / "GFPGANv1.3.pth"

            if not model_path.exists():
                print(f"GFPGAN model not found: {model_path}")
                print("Please download from: https://github.com/TencentARC/GFPGAN/releases")
                print(f"For now, using placeholder (no actual enhancement)")
                self.model = 'placeholder'
                self.model_loaded = True
                return

            # Load GFPGAN
            self.model = GFPGANer(
                model_path=str(model_path),
                upscale=1,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None,
                device=self.device
            )
            self.model_loaded = True

            print(f"[OK] Loaded GFPGAN model successfully")

        except Exception as e:
            print(f"Warning: Failed to load GFPGAN: {e}")
            print("Using placeholder face enhancement")
            self.model = 'placeholder'
            self.model_loaded = True

    def enhance(self, image: Any, upscale_factor: int = 4, enhance_level: str = 'medium') -> Any:
        """
        Enhance faces in image.

        Args:
            image: Input image (numpy array or PIL Image)
            upscale_factor: Factor by which to upscale the image (default: 4)
            enhance_level: Enhancement level (default: 'medium')

        Returns:
            Enhanced image (numpy array or PIL Image)
        """
        if not self.model_loaded:
            self.load_model()

        # Convert to numpy
        if isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image

        if self.model == 'placeholder':
            print("Warning: Face enhancer using placeholder (no actual enhancement)")
            return image

        try:
            # Use GFPGAN to enhance
            _, _, output = self.model.enhance(
                img,
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
                weight=0.5
            )
            return output
        except Exception as e:
            print(f"Face enhancement error: {e}")
            return image

    def analyze(self, image: Any) -> Dict[str, Any]:
        """
        Analyze image for faces.

        Args:
            image: Input image

        Returns:
            Dictionary with analysis results
        """
        # Convert to numpy for face detection
        if isinstance(image, Image.Image):
            image = np.array(image)
        elif not isinstance(image, np.ndarray):
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Use OpenCV face detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Simple face detection using haar cascade
        try:
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                maxSize=(500, 500)
            )

            has_faces = len(faces) > 0

            # Calculate confidence
            if has_faces:
                confidence = 1.0
                reason = f'{len(faces)} face(s) detected'
            else:
                confidence = 0.5
                reason = 'No faces detected'

            return {
                'needs_enhancement': has_faces,
                'confidence': confidence,
                'reason': reason,
                'face_count': len(faces),
                'has_faces': has_faces,
                'positions': faces.tolist() if has_faces else []
            }

        except Exception as e:
            print(f"Face detection error: {e}")
            return {
                'needs_enhancement': False,
                'confidence': 0.3,
                'reason': 'Face detection unavailable',
                'face_count': 0,
                'has_faces': False,
                'positions': []
            }

    def unload_model(self) -> None:
        """Unload model."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.model = None
        self.model_loaded = False


if __name__ == "__main__":
    print("Testing Face Enhancer...")

    config = {'detection_threshold': 0.7}
    enhancer = FaceEnhancer(config)

    print(f"Enhancer info: {enhancer.get_info()}")

    # Test analyze
    # Create test image with face
    test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    analysis = enhancer.analyze(test_image)
    print(f"\nAnalysis: {analysis}")

    # Test enhance (would need GFPGAN)
    # enhanced = enhancer.enhance(test_image)
    # print("\nFace enhancer test completed!")

    print("\nFace enhancer test completed!")
