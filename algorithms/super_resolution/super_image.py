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

    def enhance(self, image: Any, upscale_factor: int = 4, enhance_level: str = 'medium') -> Any:
        """
        Enhance image using super-image model.

        Args:
            image: Input image (numpy array or PIL Image)
            upscale_factor: Factor by which to upscale the image (default: 4)
            enhance_level: Enhancement level (default: 'medium')

        Returns:
            Enhanced image (numpy array or PIL Image)
        """
        if not self.model_loaded:
            self.load_model()

        # Use passed upscale_factor instead of self.upscale_factor
        actual_scale = upscale_factor if upscale_factor else self.upscale_factor

        # Convert to PIL Image if numpy
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        h, w = pil_image.size[1], pil_image.size[0]

        # For large images, use tile-based processing
        max_tile_size = 400  # Maximum tile size for memory efficiency

        if h > max_tile_size or w > max_tile_size:
            print(f"Using tile-based processing for large image ({h}x{w})")
            return self._enhance_tiled(np.array(pil_image), actual_scale)

        # For small images, process directly using model's built-in method
        try:
            from super_image import ImageLoader

            # Use built-in enhancement method
            with torch.no_grad():
                inputs = ImageLoader.load_image(pil_image)
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)

                # Convert to PIL Image using super-image's utilities
                from super_image.utils.image_processing import post_process
                outputs = post_process(outputs)

            return np.array(outputs)

        except Exception as e:
            print(f"Error in direct enhancement: {e}")
            # Fallback to manual conversion
            return self._enhance_direct(np.array(pil_image))

    def _enhance_direct(self, image: np.ndarray, scale: int = 4) -> np.ndarray:
        """Enhance small image directly (fallback method)."""
        from super_image import ImageLoader

        pil_image = Image.fromarray(image)

        # Convert to tensor
        img_tensor = ImageLoader.load_image(pil_image)
        img_tensor = img_tensor.to(self.device)

        # Inference
        with torch.no_grad():
            output = self.model(img_tensor)

        # Handle different output formats
        if isinstance(output, list):
            output = output[0]

        # Move to CPU and detach
        output = output.detach().cpu()

        # Remove batch dimension if present
        if output.dim() == 4:
            output = output.squeeze(0)

        # Rearrange from C,H,W to H,W,C
        if output.dim() == 3 and output.shape[0] == 3:
            output = output.permute(1, 2, 0)

        # Convert to numpy
        output_np = output.numpy()

        # Handle normalization
        if output_np.max() <= 1.0:
            output_np = (output_np * 255).astype(np.uint8)
        else:
            output_np = np.clip(output_np, 0, 255).astype(np.uint8)

        return output_np

    def _enhance_tiled(self, image: np.ndarray, scale: int = 4) -> np.ndarray:
        """Enhance large image using tile-based processing."""
        import cv2
        from super_image import ImageLoader
        from basicsr.utils import img2tensor
        
        h, w = image.shape[:2]
        scale = self.upscale_factor
        tile_size = 400
        tile_pad = 10
        
        # Calculate number of tiles
        h_out = h * scale
        w_out = w * scale
        
        # Create output image
        output = np.zeros((h_out, w_out, 3), dtype=np.uint8)
        
        # Process in tiles
        for y in range(0, h, tile_size - tile_pad * 2):
            for x in range(0, w, tile_size - tile_pad * 2):
                # Get tile coordinates
                y_start = max(0, y - tile_pad)
                y_end = min(h, y + tile_size + tile_pad)
                x_start = max(0, x - tile_pad)
                x_end = min(w, x + tile_size + tile_pad)
                
                # Extract tile
                tile = image[y_start:y_end, x_start:x_end]
                
                # Skip if tile is too small
                if tile.shape[0] < tile_pad * 2 or tile.shape[1] < tile_pad * 2:
                    continue
                
                # Enhance tile
                enhanced_tile = self._enhance_direct(tile)
                
                # Calculate output tile position
                out_y_start = y_start * scale
                out_x_start = x_start * scale
                out_y_end = out_y_start + (y_end - y_start) * scale
                out_x_end = out_x_start + (x_end - x_start) * scale
                
                # Calculate tile overlap
                overlap_y = tile_pad * scale
                overlap_x = tile_pad * scale
                
                # Copy tile to output (without edges to avoid seams)
                output[out_y_start + overlap_y:out_y_end - overlap_y,
                       out_x_start + overlap_x:out_x_end - overlap_x] = \
                    enhanced_tile[overlap_y:-overlap_y, overlap_x:-overlap_x]
        
        return output

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
