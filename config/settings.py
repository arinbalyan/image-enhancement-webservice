"""
Configuration management for Image Enhancement Web Service.

Uses Pydantic for type-safe configuration with environment variable support.
"""

from typing import Optional, List
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
import os
from pathlib import Path


class AlgorithmConfig(BaseModel):
    """Configuration for individual algorithms."""

    name: str
    enabled: bool = True
    priority: int = 1
    parameters: dict = {}


class SuperResolutionConfig(BaseModel):
    """Super-resolution algorithm configuration."""

    default_algorithm: str = "real_esrgan"
    small_image_threshold: int = 512  # Shortest edge size for small images
    small_image_scale: int = 16
    normal_image_scale: int = 4

    @validator('default_algorithm')
    def validate_algorithm(cls, v):
        valid_algorithms = ['real_esrgan', 'super_image', 'realtime_sr',
                         '4x_sr_cnn', 'pcsr', 'fourk_agent']
        if v not in valid_algorithms:
            raise ValueError(f'Algorithm must be one of {valid_algorithms}')
        return v


class LowLightConfig(BaseModel):
    """Low-light enhancement configuration."""

    default_algorithm: str = "clahe"
    brightness_threshold: float = 0.3  # Below this is considered low-light
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple = (8, 8)

    @validator('default_algorithm')
    def validate_algorithm(cls, v):
        valid_algorithms = ['clahe', 'csec']
        if v not in valid_algorithms:
            raise ValueError(f'Algorithm must be one of {valid_algorithms}')
        return v


class ColorCorrectionConfig(BaseModel):
    """Color correction configuration."""

    default_algorithm: str = "white_balance"
    enable_white_balance: bool = True
    enable_exposure_correction: bool = True
    white_balance_algorithm: str = "deep"

    @validator('default_algorithm')
    def validate_algorithm(cls, v):
        valid_algorithms = ['white_balance', 'exposure']
        if v not in valid_algorithms:
            raise ValueError(f'Algorithm must be one of {valid_algorithms}')
        return v


class FaceAnalysisConfig(BaseModel):
    """Face analysis configuration."""

    enabled: bool = True
    detection_threshold: float = 0.7
    enhancement_strength: float = 0.5
    preserve_features: bool = True


class LoggingConfig(BaseModel):
    """Logging configuration."""

    log_level: str = "INFO"
    log_format: str = "ENHANCEMENT"
    log_dir: Path = Path("logs")
    enable_json_logging: bool = True
    separate_log_files: bool = True

    @validator('log_level')
    def validate_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of {valid_levels}')
        return v.upper()


class APIConfig(BaseModel):
    """API configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    debug: bool = False
    max_upload_size_mb: int = 50


class GoogleDriveConfig(BaseModel):
    """Google Drive configuration."""

    credentials_path: Path = Path("credentials.json")
    token_path: Path = Path("token.json")
    input_folder: str = "raw_images"
    output_folder: str = "image-enhancer"
    batch_size: int = 10
    enable_resume: bool = True


class ProcessingConfig(BaseModel):
    """General processing configuration."""

    enable_gpu: bool = True
    gpu_device: int = 0
    batch_size: int = 4
    num_workers: int = 4
    max_image_size_mb: int = 100
    preserve_original: bool = True
    output_format: str = "jpg"
    quality: int = 95


class Settings(BaseSettings):
    """Main settings class."""

    # API
    api: APIConfig = APIConfig()

    # Algorithms
    super_resolution: SuperResolutionConfig = SuperResolutionConfig()
    low_light: LowLightConfig = LowLightConfig()
    color_correction: ColorCorrectionConfig = ColorCorrectionConfig()
    face_analysis: FaceAnalysisConfig = FaceAnalysisConfig()

    # Processing
    processing: ProcessingConfig = ProcessingConfig()

    # Logging
    logging: LoggingConfig = LoggingConfig()

    # Google Drive
    google_drive: GoogleDriveConfig = GoogleDriveConfig()

    # Paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    algorithms_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "algorithms")
    utils_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "utils")
    config_dir: Path = Field(default_factory=lambda: Path(__file__).parent)
    test_images_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "test_images")
    test_output_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "test_output")

    class Config:
        env_prefix = "IMAGE_ENHANCEMENT_"
        env_file = ".env"
        case_sensitive = False


def get_settings() -> Settings:
    """
    Get settings instance.

    Returns:
        Settings: Configured settings instance
    """
    return Settings()


def get_algorithm_config(settings: Settings, algorithm_type: str) -> Optional[AlgorithmConfig]:
    """
    Get configuration for a specific algorithm type.

    Args:
        settings: Settings instance
        algorithm_type: Type of algorithm ('super_resolution', 'low_light', etc.)

    Returns:
        AlgorithmConfig: Configuration for the algorithm
    """
    config_map = {
        'super_resolution': AlgorithmConfig(
            name=settings.super_resolution.default_algorithm,
            enabled=True,
            priority=1,
            parameters={
                'small_scale': settings.super_resolution.small_image_scale,
                'normal_scale': settings.super_resolution.normal_image_scale,
                'threshold': settings.super_resolution.small_image_threshold
            }
        ),
        'low_light': AlgorithmConfig(
            name=settings.low_light.default_algorithm,
            enabled=True,
            priority=2,
            parameters={
                'brightness_threshold': settings.low_light.brightness_threshold,
                'clahe_clip_limit': settings.low_light.clahe_clip_limit
            }
        ),
        'color_correction': AlgorithmConfig(
            name=settings.color_correction.default_algorithm,
            enabled=True,
            priority=3,
            parameters={
                'enable_white_balance': settings.color_correction.enable_white_balance,
                'enable_exposure': settings.color_correction.enable_exposure_correction
            }
        ),
        'face_analysis': AlgorithmConfig(
            name="face_enhancement",
            enabled=settings.face_analysis.enabled,
            priority=4,
            parameters={
                'detection_threshold': settings.face_analysis.detection_threshold,
                'enhancement_strength': settings.face_analysis.enhancement_strength
            }
        )
    }

    return config_map.get(algorithm_type)


def create_default_env_file():
    """Create a default .env file if it doesn't exist."""
    env_path = Path(".env")
    if not env_path.exists():
        env_content = """# Image Enhancement Web Service Configuration

# API Settings
IMAGE_ENHANCEMENT_API__HOST=0.0.0.0
IMAGE_ENHANCEMENT_API__PORT=8000
IMAGE_ENHANCEMENT_API__DEBUG=False
IMAGE_ENHANCEMENT_API__MAX_UPLOAD_SIZE_MB=50

# Processing Settings
IMAGE_ENHANCEMENT_PROCESSING__ENABLE_GPU=True
IMAGE_ENHANCEMENT_PROCESSING__GPU_DEVICE=0
IMAGE_ENHANCEMENT_PROCESSING__BATCH_SIZE=4
IMAGE_ENHANCEMENT_PROCESSING__NUM_WORKERS=4
IMAGE_ENHANCEMENT_PROCESSING__MAX_IMAGE_SIZE_MB=100
IMAGE_ENHANCEMENT_PROCESSING__PRESERVE_ORIGINAL=True
IMAGE_ENHANCEMENT_PROCESSING__OUTPUT_FORMAT=jpg
IMAGE_ENHANCEMENT_PROCESSING__QUALITY=95

# Super-Resolution Settings
IMAGE_ENHANCEMENT_SUPER_RESOLUTION__DEFAULT_ALGORITHM=real_esrgan
IMAGE_ENHANCEMENT_SUPER_RESOLUTION__SMALL_IMAGE_THRESHOLD=512
IMAGE_ENHANCEMENT_SUPER_RESOLUTION__SMALL_IMAGE_SCALE=16
IMAGE_ENHANCEMENT_SUPER_RESOLUTION__NORMAL_IMAGE_SCALE=4

# Low-Light Settings
IMAGE_ENHANCEMENT_LOW_LIGHT__DEFAULT_ALGORITHM=clahe
IMAGE_ENHANCEMENT_LOW_LIGHT__BRIGHTNESS_THRESHOLD=0.3
IMAGE_ENHANCEMENT_LOW_LIGHT__CLIP_LIMIT=2.0

# Color Correction Settings
IMAGE_ENHANCEMENT_COLOR_CORRECTION__DEFAULT_ALGORITHM=white_balance
IMAGE_ENHANCEMENT_COLOR_CORRECTION__ENABLE_WHITE_BALANCE=True
IMAGE_ENHANCEMENT_COLOR_CORRECTION__ENABLE_EXPOSURE_CORRECTION=True

# Face Analysis Settings
IMAGE_ENHANCEMENT_FACE_ANALYSIS__ENABLED=True
IMAGE_ENHANCEMENT_FACE_ANALYSIS__DETECTION_THRESHOLD=0.7
IMAGE_ENHANCEMENT_FACE_ANALYSIS__ENHANCEMENT_STRENGTH=0.5

# Logging Settings
IMAGE_ENHANCEMENT_LOGGING__LOG_LEVEL=INFO
IMAGE_ENHANCEMENT_LOGGING__LOG_FORMAT=ENHANCEMENT
IMAGE_ENHANCEMENT_LOGGING__ENABLE_JSON_LOGGING=True
IMAGE_ENHANCEMENT_LOGGING__SEPARATE_LOG_FILES=True

# Google Drive Settings
IMAGE_ENHANCEMENT_GOOGLE_DRIVE__CREDENTIALS_PATH=credentials.json
IMAGE_ENHANCEMENT_GOOGLE_DRIVE__TOKEN_PATH=token.json
IMAGE_ENHANCEMENT_GOOGLE_DRIVE__INPUT_FOLDER=raw_images
IMAGE_ENHANCEMENT_GOOGLE_DRIVE__OUTPUT_FOLDER=image-enhancer
IMAGE_ENHANCEMENT_GOOGLE_DRIVE__BATCH_SIZE=10
IMAGE_ENHANCEMENT_GOOGLE_DRIVE__ENABLE_RESUME=True
"""
        with open(env_path, 'w') as f:
            f.write(env_content)
        print(f"Created default .env file at {env_path}")


if __name__ == "__main__":
    # Create default .env file
    create_default_env_file()

    # Test loading settings
    settings = get_settings()
    print("Settings loaded successfully:")
    print(f"  Project Root: {settings.project_root}")
    print(f"  API Host:Port: {settings.api.host}:{settings.api.port}")
    print(f"  GPU Enabled: {settings.processing.enable_gpu}")
    print(f"  Log Level: {settings.logging.log_level}")
