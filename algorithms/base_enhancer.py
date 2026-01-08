"""
Base enhancer class for all image enhancement algorithms.

Provides common interface and utility methods for all enhancement modules.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, Tuple
from pathlib import Path
import numpy as np
from PIL import Image
import torch


class BaseEnhancer(ABC):
    """Abstract base class for all enhancement algorithms."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize enhancer with configuration.

        Args:
            config: Configuration dictionary for this enhancer
        """
        self.config = config
        self.model = None
        self.device = self._get_device()
        self.model_loaded = False

    @abstractmethod
    def enhance(self, image: Union[np.ndarray, Image.Image]) -> Union[np.ndarray, Image.Image]:
        """
        Enhance an image.

        Args:
            image: Input image (numpy array or PIL Image)

        Returns:
            Enhanced image (numpy array or PIL Image)
        """
        pass

    @abstractmethod
    def analyze(self, image: Union[np.ndarray, Image.Image]) -> Dict[str, Any]:
        """
        Analyze image to determine if enhancement is needed.

        Args:
            image: Input image (numpy array or PIL Image)

        Returns:
            Dictionary with analysis results
        """
        pass

    @abstractmethod
    def load_model(self) -> None:
        """Load model into memory."""
        pass

    @abstractmethod
    def unload_model(self) -> None:
        """Unload model from memory."""
        pass

    def validate_image(self, image: Any) -> Tuple[bool, str]:
        """
        Validate input image.

        Args:
            image: Input to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if image is None:
            return False, "Image is None"

        if isinstance(image, np.ndarray):
            if len(image.shape) not in [2, 3]:
                return False, f"Invalid image shape: {image.shape}"
            if image.size == 0:
                return False, "Image is empty"

        elif isinstance(image, Image.Image):
            if image.size[0] == 0 or image.size[1] == 0:
                return False, "Image has zero dimensions"

        else:
            return False, f"Unsupported image type: {type(image)}"

        return True, ""

    def _get_device(self) -> str:
        """
        Get the device to use for processing.

        Returns:
            str: Device string ('cuda:0', 'cpu', etc.)
        """
        # Check if GPU is enabled and available
        from config.settings import get_settings
        settings = get_settings()

        if settings.processing.enable_gpu and torch.cuda.is_available():
            return f"cuda:{settings.processing.gpu_device}"
        else:
            return "cpu"

    def _convert_to_pil(self, image: Union[np.ndarray, Image.Image]) -> Image.Image:
        """
        Convert image to PIL Image.

        Args:
            image: Input image

        Returns:
            PIL Image
        """
        if isinstance(image, Image.Image):
            return image

        elif isinstance(image, np.ndarray):
            # Handle different numpy array formats
            if image.dtype == np.uint8:
                mode = 'L' if len(image.shape) == 2 else 'RGB'
            elif image.dtype == np.float32 or image.dtype == np.float64:
                # Normalize to 0-255
                image = (image * 255).astype(np.uint8)
                mode = 'L' if len(image.shape) == 2 else 'RGB'
            else:
                raise ValueError(f"Unsupported numpy dtype: {image.dtype}")

            # Handle grayscale vs RGB
            if len(image.shape) == 2:
                return Image.fromarray(image, mode=mode)
            else:
                # Remove extra dimensions
                if len(image.shape) == 3 and image.shape[0] in [1, 3]:
                    image = np.transpose(image, (1, 2, 0))
                if image.shape[-1] == 1:
                    image = image.squeeze(-1)
                    mode = 'L'
                elif image.shape[-1] == 3:
                    mode = 'RGB'
                elif image.shape[-1] == 4:
                    mode = 'RGBA'
                else:
                    raise ValueError(f"Unsupported number of channels: {image.shape[-1]}")

                return Image.fromarray(image, mode=mode)

        else:
            raise ValueError(f"Unsupported input type: {type(image)}")

    def _convert_to_numpy(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Convert image to numpy array.

        Args:
            image: Input image

        Returns:
            numpy array
        """
        if isinstance(image, np.ndarray):
            return image

        elif isinstance(image, Image.Image):
            # Convert to numpy
            image_array = np.array(image)

            # Ensure RGB mode
            if image.mode in ('L', 'P'):
                # Convert grayscale to RGB
                image = image.convert('RGB')
                image_array = np.array(image)
            elif image.mode == 'RGBA':
                # Convert RGBA to RGB
                image = image.convert('RGB')
                image_array = np.array(image)

            return image_array

        else:
            raise ValueError(f"Unsupported input type: {type(image)}")

    def _convert_to_tensor(
        self,
        image: Union[np.ndarray, Image.Image],
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Convert image to PyTorch tensor.

        Args:
            image: Input image
            normalize: Whether to normalize to [0, 1] range

        Returns:
            PyTorch tensor
        """
        # Convert to numpy first
        numpy_image = self._convert_to_numpy(image)

        # Handle grayscale
        if len(numpy_image.shape) == 2:
            numpy_image = np.expand_dims(numpy_image, axis=2)
            if numpy_image.shape[-1] == 1:
                numpy_image = np.repeat(numpy_image, 3, axis=2)

        # Convert to tensor (H, W, C) -> (C, H, W)
        tensor = torch.from_numpy(numpy_image).permute(2, 0, 1)

        # Convert to float32
        tensor = tensor.float()

        # Normalize
        if normalize:
            tensor = tensor / 255.0

        # Move to device
        tensor = tensor.to(self.device)

        return tensor

    def _convert_from_tensor(
        self,
        tensor: torch.Tensor,
        denormalize: bool = True
    ) -> np.ndarray:
        """
        Convert PyTorch tensor to numpy array.

        Args:
            tensor: Input tensor
            denormalize: Whether to denormalize from [0, 1] to [0, 255]

        Returns:
            numpy array
        """
        # Move to CPU
        tensor = tensor.cpu()

        # Denormalize
        if denormalize:
            tensor = tensor * 255.0

        # Clamp values
        tensor = torch.clamp(tensor, 0, 255)

        # Convert to numpy (C, H, W) -> (H, W, C)
        numpy_image = tensor.permute(1, 2, 0).numpy()

        # Convert to uint8
        numpy_image = numpy_image.astype(np.uint8)

        return numpy_image

    def _tile_image(
        self,
        image: np.ndarray,
        tile_size: int = 512,
        overlap: int = 32
    ) -> list[np.ndarray]:
        """
        Tile large image into smaller patches.

        Args:
            image: Input image
            tile_size: Size of each tile
            overlap: Overlap between tiles

        Returns:
            List of tiles
        """
        h, w = image.shape[:2]
        tiles = []

        for y in range(0, h - overlap, tile_size - overlap):
            for x in range(0, w - overlap, tile_size - overlap):
                # Extract tile
                tile = image[y:y+tile_size, x:x+tile_size]

                # Skip small tiles at edges
                if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                    continue

                tiles.append((tile, x, y))

        return tiles

    def _merge_tiles(
        self,
        tiles: list[tuple[np.ndarray, int, int]],
        image_size: tuple[int, int],
        tile_size: int = 512,
        overlap: int = 32
    ) -> np.ndarray:
        """
        Merge tiles back into a single image.

        Args:
            tiles: List of (tile, x, y) tuples
            image_size: Target image size (h, w)
            tile_size: Size of each tile
            overlap: Overlap between tiles

        Returns:
            Merged image
        """
        h, w = image_size
        result = np.zeros((h, w, 3), dtype=np.uint8)
        weight = np.zeros((h, w), dtype=np.float32)

        # Blend overlapping regions
        for tile, x, y in tiles:
            tile_h = min(tile.shape[0], h - y)
            tile_w = min(tile.shape[1], w - x)

            # Create weight map for blending
            tile_weight = np.ones((tile_h, tile_w), dtype=np.float32)

            # Fade edges for overlapping regions
            if y > 0 and y % (tile_size - overlap) != 0:
                top_overlap = overlap
                for i in range(top_overlap):
                    alpha = i / top_overlap
                    tile_weight[:top_overlap, :] *= alpha

            if x > 0 and x % (tile_size - overlap) != 0:
                left_overlap = overlap
                for i in range(left_overlap):
                    alpha = i / left_overlap
                    tile_weight[:, :left_overlap] *= alpha

            # Add weighted tile to result
            result[y:y+tile_h, x:x+tile_w] += (
                tile[:tile_h, :tile_w].astype(np.float32) * tile_weight[..., np.newaxis]
            ).astype(np.uint8)

            weight[y:y+tile_h, x:x+tile_w] += tile_weight

        # Normalize by weights
        weight = np.maximum(weight, 1)
        result = (result.astype(np.float32) / weight[..., np.newaxis]).astype(np.uint8)

        return result

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about this enhancer.

        Returns:
            Dictionary with enhancer information
        """
        return {
            'name': self.__class__.__name__,
            'config': self.config,
            'device': self.device,
            'model_loaded': self.model_loaded
        }


class DummyEnhancer(BaseEnhancer):
    """Dummy enhancer for testing purposes."""

    def enhance(self, image: Union[np.ndarray, Image.Image]) -> Union[np.ndarray, Image.Image]:
        """Return image unchanged."""
        return image

    def analyze(self, image: Union[np.ndarray, Image.Image]) -> Dict[str, Any]:
        """Return dummy analysis."""
        return {
            'needs_enhancement': False,
            'confidence': 1.0,
            'message': 'Dummy enhancer - no actual enhancement'
        }

    def load_model(self) -> None:
        """Load dummy model."""
        self.model = 'dummy'
        self.model_loaded = True

    def unload_model(self) -> None:
        """Unload dummy model."""
        self.model = None
        self.model_loaded = False


if __name__ == "__main__":
    # Test base enhancer functionality
    print("Testing Base Enhancer...")

    # Test device detection
    enhancer = DummyEnhancer({})
    print(f"Device: {enhancer.device}")

    # Test image conversions
    # Create dummy image
    dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    # Test numpy -> PIL
    pil_image = enhancer._convert_to_pil(dummy_image)
    print(f"PIL Image size: {pil_image.size}, mode: {pil_image.mode}")

    # Test PIL -> numpy
    numpy_image = enhancer._convert_to_numpy(pil_image)
    print(f"Numpy image shape: {numpy_image.shape}, dtype: {numpy_image.dtype}")

    # Test numpy -> tensor
    tensor_image = enhancer._convert_to_tensor(dummy_image)
    print(f"Tensor shape: {tensor_image.shape}, dtype: {tensor_image.dtype}, device: {tensor_image.device}")

    # Test tensor -> numpy
    numpy_from_tensor = enhancer._convert_from_tensor(tensor_image)
    print(f"Numpy from tensor shape: {numpy_from_tensor.shape}, dtype: {numpy_from_tensor.dtype}")

    # Test image validation
    valid, error = enhancer.validate_image(dummy_image)
    print(f"Image validation: valid={valid}, error='{error}'")

    # Test tiling
    large_image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
    tiles = enhancer._tile_image(large_image, tile_size=512, overlap=32)
    print(f"Tiling: {len(tiles)} tiles created from {large_image.shape} image")

    # Test merging
    merged = enhancer._merge_tiles(tiles, image_size=large_image.shape[:2], tile_size=512, overlap=32)
    print(f"Merge: merged image shape: {merged.shape}, matches original: {merged.shape == large_image.shape}")

    print("Base enhancer tests completed!")
