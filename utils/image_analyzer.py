"""
Image analyzer for deep analysis of images.

Analyzes images to determine optimal enhancement strategies.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from config.settings import get_settings


class ImageAnalyzer:
    """Deep image analyzer for enhancement recommendation."""

    def __init__(self, settings=None):
        """
        Initialize image analyzer.

        Args:
            settings: Optional settings object
        """
        if settings is None:
            settings = get_settings()

        self.settings = settings
        self.face_cascade = None

        # Load face cascade
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                print("Warning: Failed to load face cascade")
        except Exception as e:
            print(f"Warning: Failed to initialize face detection: {e}")

    def analyze(self, image_path: str) -> Dict[str, Any]:
        """
        Perform comprehensive image analysis.

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with complete analysis results
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Convert to RGB for consistency
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get file info
        file_info = self._get_file_info(image_path)

        # Get resolution info
        resolution_info = self.get_resolution_info(image_rgb)

        # Analyze brightness
        brightness_info = self.analyze_brightness(image_rgb)

        # Detect faces
        face_info = self.detect_faces(image)

        # Analyze colors
        color_info = self.analyze_colors(image_rgb)

        # Estimate noise
        noise_info = self.estimate_noise(image_rgb)

        # Generate recommendations
        recommendations = self.get_recommendations({
            'resolution': resolution_info,
            'brightness': brightness_info,
            'faces': face_info,
            'color': color_info,
            'noise': noise_info
        })

        return {
            'image_path': image_path,
            'file_info': file_info,
            'resolution': resolution_info,
            'brightness': brightness_info,
            'faces': face_info,
            'color': color_info,
            'noise': noise_info,
            'recommendations': recommendations
        }

    def _get_file_info(self, image_path: str) -> Dict[str, Any]:
        """
        Get file information.

        Args:
            image_path: Path to image

        Returns:
            Dictionary with file info
        """
        path = Path(image_path)
        return {
            'filename': path.name,
            'size_bytes': path.stat().st_size if path.exists() else 0,
            'extension': path.suffix.lower()
        }

    def get_resolution_info(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Get image resolution and determine upscale factor.

        Args:
            image: Image as numpy array

        Returns:
            Dictionary with resolution information
        """
        height, width = image.shape[:2]
        shortest_edge = min(width, height)

        # Determine if image is small
        threshold = self.settings.super_resolution.small_image_threshold
        is_small = shortest_edge < threshold

        # Determine upscale factor
        if is_small:
            recommended_upscale = self.settings.super_resolution.small_image_scale
        else:
            recommended_upscale = self.settings.super_resolution.normal_image_scale

        return {
            'width': width,
            'height': height,
            'shortest_edge': shortest_edge,
            'is_small': is_small,
            'recommended_upscale': recommended_upscale,
            'aspect_ratio': width / height if height > 0 else 0
        }

    def analyze_brightness(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze image brightness.

        Args:
            image: Image as numpy array (RGB)

        Returns:
            Dictionary with brightness information
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Calculate statistics
        mean = np.mean(gray) / 255.0
        std = np.std(gray) / 255.0
        median = np.median(gray) / 255.0
        min_val = np.min(gray) / 255.0
        max_val = np.max(gray) / 255.0

        # Determine lighting condition
        threshold = self.settings.low_light.brightness_threshold
        is_low_light = mean < threshold
        is_overexposed = mean > (1.0 - threshold)
        contrast = max_val - min_val

        return {
            'mean': float(mean),
            'median': float(median),
            'std': float(std),
            'min': float(min_val),
            'max': float(max_val),
            'contrast': float(contrast),
            'is_low_light': is_low_light,
            'is_overexposed': is_overexposed,
            'lighting_condition': 'low_light' if is_low_light else ('overexposed' if is_overexposed else 'normal')
        }

    def detect_faces(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect faces in image.

        Args:
            image: Image as numpy array (BGR)

        Returns:
            Dictionary with face information
        """
        if self.face_cascade is None or self.face_cascade.empty():
            return {
                'count': 0,
                'positions': [],
                'has_faces': False,
                'confidence': 0.0,
                'message': 'Face detection not available'
            }

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            maxSize=(500, 500)
        )

        # Extract positions
        positions = [(int(x), int(y), int(x+w), int(y+h)) for x, y, w, h in faces]

        # Calculate confidence based on number of faces and quality
        has_faces = len(faces) > 0
        confidence = 1.0 if has_faces and len(faces) <= 5 else 0.5

        return {
            'count': len(faces),
            'positions': positions,
            'has_faces': has_faces,
            'confidence': confidence,
            'method': 'haar_cascade'
        }

    def analyze_colors(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze color distribution and detect color cast.

        Args:
            image: Image as numpy array (RGB)

        Returns:
            Dictionary with color information
        """
        # Calculate mean values per channel
        mean_r = np.mean(image[:, :, 0]) / 255.0
        mean_g = np.mean(image[:, :, 1]) / 255.0
        mean_b = np.mean(image[:, :, 2]) / 255.0

        # Calculate std per channel
        std_r = np.std(image[:, :, 0]) / 255.0
        std_g = np.std(image[:, :, 1]) / 255.0
        std_b = np.std(image[:, :, 2]) / 255.0

        # Detect color cast by comparing channel means
        mean_values = np.array([mean_r, mean_g, mean_b])
        mean_color = np.mean(mean_values)

        # Calculate deviation from gray
        r_deviation = abs(mean_r - mean_color)
        g_deviation = abs(mean_g - mean_color)
        b_deviation = abs(mean_b - mean_color)

        # Determine color cast
        max_deviation = max(r_deviation, g_deviation, b_deviation)

        # Check if deviation is significant (> 0.05)
        needs_white_balance = max_deviation > 0.05

        # Determine color cast type
        if needs_white_balance:
            if r_deviation > g_deviation and r_deviation > b_deviation:
                color_cast = 'red'
            elif g_deviation > r_deviation and g_deviation > b_deviation:
                color_cast = 'green'
            else:
                color_cast = 'blue'
        else:
            color_cast = None

        return {
            'mean_r': float(mean_r),
            'mean_g': float(mean_g),
            'mean_b': float(mean_b),
            'std_r': float(std_r),
            'std_g': float(std_g),
            'std_b': float(std_b),
            'mean_color': float(mean_color),
            'color_cast': color_cast,
            'needs_white_balance': needs_white_balance,
            'balance_score': float(max_deviation)
        }

    def estimate_noise(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Estimate noise level in image using Laplacian variance.

        Args:
            image: Image as numpy array (RGB)

        Returns:
            Dictionary with noise information
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Use Laplacian variance to estimate noise
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise_level = laplacian.var()

        # Normalize noise level (typical range 0-1000)
        noise_level_normalized = min(noise_level / 1000.0, 1.0)

        # Determine if image is noisy
        is_noisy = noise_level > 100.0

        return {
            'level': float(noise_level),
            'normalized_level': float(noise_level_normalized),
            'is_noisy': is_noisy
        }

    def get_recommendations(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate enhancement recommendations based on analysis.

        Args:
            analysis: Complete analysis results

        Returns:
            List of recommended enhancements
        """
        recommendations = []

        # Super-resolution recommendation
        resolution = analysis['resolution']
        recommendations.append({
            'algorithm': 'super_resolution',
            'sub_algorithm': self.settings.super_resolution.default_algorithm,
            'confidence': 1.0 if resolution['is_small'] else 0.8,
            'priority': 1,
            'parameters': {
                'scale': resolution['recommended_upscale'],
                'reason': f'Small image ({resolution["shortest_edge"]}px)' if resolution['is_small'] else 'Standard upscaling'
            },
            'enabled': True
        })

        # Low-light enhancement recommendation
        brightness = analysis['brightness']
        if brightness['is_low_light']:
            recommendations.append({
                'algorithm': 'low_light',
                'sub_algorithm': self.settings.low_light.default_algorithm,
                'confidence': 0.9,
                'priority': 2,
                'parameters': {
                    'current_brightness': brightness['mean'],
                    'threshold': self.settings.low_light.brightness_threshold,
                    'reason': 'Image is too dark'
                },
                'enabled': True
            })

        # White balance recommendation
        color = analysis['color']
        if color['needs_white_balance']:
            recommendations.append({
                'algorithm': 'color_correction',
                'sub_algorithm': 'white_balance',
                'confidence': 0.85,
                'priority': 3,
                'parameters': {
                    'color_cast': color['color_cast'],
                    'balance_score': color['balance_score'],
                    'reason': f'{color["color_cast"]} color cast detected'
                },
                'enabled': self.settings.color_correction.enable_white_balance
            })

        # Face enhancement recommendation
        faces = analysis['faces']
        if faces['has_faces'] and self.settings.face_analysis.enabled:
            recommendations.append({
                'algorithm': 'face_analysis',
                'sub_algorithm': 'face_enhancement',
                'confidence': faces['confidence'],
                'priority': 4,
                'parameters': {
                    'face_count': faces['count'],
                    'detection_threshold': self.settings.face_analysis.detection_threshold,
                    'reason': f'{faces["count"]} face(s) detected'
                },
                'enabled': self.settings.face_analysis.enabled
            })

        # Exposure correction recommendation
        if brightness['is_overexposed']:
            recommendations.append({
                'algorithm': 'color_correction',
                'sub_algorithm': 'exposure_correction',
                'confidence': 0.85,
                'priority': 3,
                'parameters': {
                    'current_brightness': brightness['mean'],
                    'reason': 'Image is overexposed'
                },
                'enabled': self.settings.color_correction.enable_exposure_correction
            })

        # Sort by priority and confidence
        recommendations.sort(key=lambda x: (x['priority'], -x['confidence']))

        return recommendations


if __name__ == "__main__":
    # Test image analyzer
    print("Testing Image Analyzer...")

    analyzer = ImageAnalyzer()

    # Create a test image
    test_image = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)

    # Save test image
    test_path = "test_image.jpg"
    cv2.imwrite(test_path, cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR))

    # Analyze
    analysis = analyzer.analyze(test_path)

    # Print results
    print("\n=== Image Analysis Results ===")
    print(f"Resolution: {analysis['resolution']['width']}x{analysis['resolution']['height']}")
    print(f"  Shortest edge: {analysis['resolution']['shortest_edge']}")
    print(f"  Is small: {analysis['resolution']['is_small']}")
    print(f"  Recommended upscale: {analysis['resolution']['recommended_upscale']}x")

    print(f"\nBrightness: {analysis['brightness']['mean']:.3f}")
    print(f"  Lighting: {analysis['brightness']['lighting_condition']}")
    print(f"  Contrast: {analysis['brightness']['contrast']:.3f}")

    print(f"\nColors: R={analysis['color']['mean_r']:.3f}, G={analysis['color']['mean_g']:.3f}, B={analysis['color']['mean_b']:.3f}")
    print(f"  Color cast: {analysis['color']['color_cast']}")
    print(f"  Needs white balance: {analysis['color']['needs_white_balance']}")

    print(f"\nFaces: {analysis['faces']['count']}")
    print(f"  Has faces: {analysis['faces']['has_faces']}")

    print(f"\nNoise level: {analysis['noise']['level']:.2f}")
    print(f"  Is noisy: {analysis['noise']['is_noisy']}")

    print(f"\nRecommendations:")
    for i, rec in enumerate(analysis['recommendations'], 1):
        print(f"  {i}. {rec['algorithm']} ({rec['sub_algorithm']})")
        print(f"     Confidence: {rec['confidence']:.2f}")
        print(f"     Priority: {rec['priority']}")
        print(f"     Enabled: {rec['enabled']}")

    # Cleanup
    import os
    if os.path.exists(test_path):
        os.remove(test_path)

    print("\nImage analyzer test completed!")
