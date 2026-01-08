"""
I/O utilities for image enhancement.

Provides file I/O operations for images and files.
"""

import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple, Union, List
import shutil


def load_image(
    image_path: str,
    mode: str = 'RGB',
    as_numpy: bool = True
) -> Union[np.ndarray, Image.Image]:
    """
    Load image from file with automatic format detection.

    Args:
        image_path: Path to image file
        mode: Color mode for PIL ('RGB', 'L', 'RGBA')
        as_numpy: If True, return numpy array; if False, return PIL Image

    Returns:
        Image as numpy array or PIL Image

    Raises:
        FileNotFoundError: If image doesn't exist
        ValueError: If image format is not supported
    """
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {image_path}")

    # Get file extension
    extension = path.suffix.lower()

    # Check if supported format
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    if extension not in supported_formats:
        raise ValueError(f"Unsupported image format: {extension}")

    # Load with PIL for better format support
    try:
        pil_image = Image.open(image_path)

        # Convert to requested mode
        if mode:
            pil_image = pil_image.convert(mode)

        # Exif orientation fix
        if hasattr(pil_image, '_getexif'):
            pil_image = _fix_image_orientation(pil_image)

        if as_numpy:
            # Convert to numpy
            numpy_image = np.array(pil_image)

            # Ensure correct channel order (RGB not BGR)
            if len(numpy_image.shape) == 3 and numpy_image.shape[-1] == 3:
                numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

            return numpy_image
        else:
            return pil_image

    except Exception as e:
        raise ValueError(f"Failed to load image {image_path}: {e}")


def save_image(
    image: Union[np.ndarray, Image.Image],
    output_path: str,
    quality: int = 95,
    format: Optional[str] = None,
    overwrite: bool = False
) -> str:
    """
    Save image to file with format preservation.

    Args:
        image: Image to save (numpy array or PIL Image)
        output_path: Path to save image
        quality: JPEG quality (1-100)
        format: Output format (auto-detect if None)
        overwrite: If True, overwrite existing file

    Returns:
        str: Path where image was saved

    Raises:
        ValueError: If image type is not supported
        FileExistsError: If file exists and overwrite is False
    """
    path = Path(output_path)

    # Check if file exists
    if path.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {output_path}")

    # Create parent directory if needed
    path.parent.mkdir(parents=True, exist_ok=True)

    # Determine format
    if format is None:
        format = path.suffix.lower().replace('.', '')

    # Convert numpy to PIL if needed
    if isinstance(image, np.ndarray):
        # Check if BGR (OpenCV format)
        if len(image.shape) == 3 and image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pil_image = Image.fromarray(image)
    elif isinstance(image, Image.Image):
        pil_image = image
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    # Save based on format
    save_kwargs = {'quality': quality}

    # Format-specific options
    if format in ['jpg', 'jpeg']:
        pil_image.save(output_path, format='JPEG', **save_kwargs)
    elif format == 'png':
        pil_image.save(output_path, format='PNG', compress_level=6)
    elif format in ['bmp', 'webp']:
        pil_image.save(output_path, format=format.upper(), **save_kwargs)
    elif format in ['tiff', 'tif']:
        pil_image.save(output_path, format='TIFF', **save_kwargs)
    else:
        pil_image.save(output_path, format=format.upper(), **save_kwargs)

    return str(path)


def get_image_info(image_path: str) -> dict:
    """
    Get comprehensive information about an image.

    Args:
        image_path: Path to image file

    Returns:
        Dictionary with image information
    """
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load image
    pil_image = Image.open(image_path)

    # Get basic info
    info = {
        'path': str(path.absolute()),
        'filename': path.name,
        'extension': path.suffix.lower(),
        'size_bytes': path.stat().st_size,
        'width': pil_image.width,
        'height': pil_image.height,
        'mode': pil_image.mode,
        'format': pil_image.format,
        'aspect_ratio': pil_image.width / pil_image.height if pil_image.height > 0 else 0
    }

    # Get EXIF data if available
    if hasattr(pil_image, '_getexif'):
        try:
            exif = pil_image._getexif()
            if exif:
                info['has_exif'] = True
                # Extract common EXIF tags
                from PIL.ExifTags import TAGS
                exif_data = {}
                for tag_id, value in exif.items():
                    tag_name = TAGS.get(tag_id, tag_id)
                    exif_data[tag_name] = value
                info['exif'] = exif_data
            else:
                info['has_exif'] = False
        except:
            info['has_exif'] = False

    return info


def list_images(
    directory: str,
    recursive: bool = False,
    extensions: Optional[List[str]] = None
) -> List[str]:
    """
    List all image files in directory.

    Args:
        directory: Path to directory
        recursive: If True, search subdirectories
        extensions: List of extensions to include (None for all)

    Returns:
        List of image file paths
    """
    path = Path(directory)

    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    # Default extensions
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']

    # Collect files
    if recursive:
        image_files = []
        for ext in extensions:
            image_files.extend(path.rglob(f'*{ext}'))
    else:
        image_files = []
        for ext in extensions:
            image_files.extend(path.glob(f'*{ext}'))

    # Convert to strings and sort
    image_files = sorted([str(f) for f in image_files])

    return image_files


def ensure_directory(path: str, clear: bool = False) -> None:
    """
    Ensure directory exists.

    Args:
        path: Directory path
        clear: If True, clear existing directory contents
    """
    dir_path = Path(path)

    if clear and dir_path.exists():
        # Clear directory contents
        for item in dir_path.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

    # Create directory
    dir_path.mkdir(parents=True, exist_ok=True)


def get_unique_filename(
    base_path: str,
    suffix: str = '_enhanced',
    extension: str = '.jpg'
) -> str:
    """
    Generate unique filename if file exists.

    Args:
        base_path: Base path (without extension)
        suffix: Suffix to add
        extension: File extension

    Returns:
        str: Unique filename path
    """
    path = Path(base_path).with_suffix(extension)

    if suffix:
        stem = path.stem
        path = path.with_stem(f"{stem}{suffix}")

    # If file exists, add counter
    counter = 1
    while path.exists():
        stem = path.stem
        if f'_{counter}' in stem:
            stem = stem.rsplit(f'_{counter}', 1)[0]

        new_stem = f"{stem}_{counter}"
        path = path.with_stem(new_stem)
        counter += 1

    return str(path)


def validate_image_size(image_path: str, max_size_mb: int = 100) -> Tuple[bool, str]:
    """
    Validate image size.

    Args:
        image_path: Path to image file
        max_size_mb: Maximum allowed size in MB

    Returns:
        Tuple of (is_valid, error_message)
    """
    path = Path(image_path)

    if not path.exists():
        return False, f"File not found: {image_path}"

    # Get size in MB
    size_bytes = path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)

    if size_mb > max_size_mb:
        return False, f"Image too large: {size_mb:.2f}MB (max: {max_size_mb}MB)"

    return True, ""


def _fix_image_orientation(pil_image: Image.Image) -> Image.Image:
    """
    Fix image orientation based on EXIF data.

    Args:
        pil_image: PIL Image

    Returns:
        PIL Image with corrected orientation
    """
    from PIL.ExifTags import TAGS

    try:
        exif = pil_image._getexif()

        if exif is None:
            return pil_image

        # Get orientation tag
        orientation = exif.get(TAGS.get('Orientation', 274))

        if orientation is None:
            return pil_image

        # Rotate based on orientation
        if orientation == 3:
            return pil_image.rotate(180, expand=True)
        elif orientation == 6:
            return pil_image.rotate(270, expand=True)
        elif orientation == 8:
            return pil_image.rotate(90, expand=True)
        else:
            return pil_image

    except Exception:
        return pil_image


if __name__ == "__main__":
    # Test I/O utilities
    print("Testing I/O Utilities...")

    # Create test directory
    test_dir = Path("test_io")
    test_dir.mkdir(exist_ok=True)

    # Create test image
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    test_path = str(test_dir / "test.jpg")

    # Test save
    print("1. Testing save_image()...")
    saved_path = save_image(test_image, test_path, quality=95)
    print(f"   Saved to: {saved_path}")

    # Test load
    print("2. Testing load_image()...")
    loaded_image = load_image(test_path, as_numpy=True)
    print(f"   Loaded shape: {loaded_image.shape}")

    # Test get info
    print("3. Testing get_image_info()...")
    info = get_image_info(test_path)
    print(f"   Size: {info['width']}x{info['height']}")
    print(f"   Mode: {info['mode']}")
    print(f"   File size: {info['size_bytes']} bytes")

    # Test list images
    print("4. Testing list_images()...")
    images = list_images(str(test_dir))
    print(f"   Found {len(images)} image(s): {images}")

    # Test unique filename
    print("5. Testing get_unique_filename()...")
    unique_path = get_unique_filename(str(test_dir / "image"), suffix='_enhanced')
    print(f"   Unique path: {unique_path}")

    # Test validation
    print("6. Testing validate_image_size()...")
    is_valid, error = validate_image_size(test_path, max_size_mb=10)
    print(f"   Valid: {is_valid}, error: '{error}'")

    # Cleanup
    shutil.rmtree(test_dir)
    print("\nI/O utilities test completed!")
