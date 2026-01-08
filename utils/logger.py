"""
Enhanced logging system for Image Enhancement Web Service.

Provides structured JSON logging with [ ENHANCEMENT ] format for all operations.
"""

import logging
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, List
from logging.handlers import RotatingFileHandler
from contextlib import contextmanager

from config.settings import get_settings


class EnhancementFormatter(logging.Formatter):
    """Custom formatter for [ ENHANCEMENT ] format."""

    def __init__(self, log_format: str = "ENHANCEMENT"):
        super().__init__()
        self.log_format = log_format

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record with [ ENHANCEMENT ] prefix.

        Args:
            record: Log record to format

        Returns:
            str: Formatted log message
        """
        timestamp = datetime.fromtimestamp(record.created).isoformat()

        # Extract enhancement details from record if available
        image_name = getattr(record, 'image_name', 'N/A')
        algorithm = getattr(record, 'algorithm', 'N/A')
        duration = getattr(record, 'duration_ms', 'N/A')
        details = getattr(record, 'details', {})

        # Format with [ ENHANCEMENT ] prefix
        if self.log_format == "ENHANCEMENT":
            formatted = f"[ {self.log_format} ] [{timestamp}] "
            formatted += f"Image: {image_name} | "
            formatted += f"Algorithm: {algorithm} | "
            if duration != 'N/A':
                formatted += f"Duration: {duration}ms | "
            if details:
                formatted += f"Details: {json.dumps(details)}"
            else:
                formatted = formatted.rstrip(" | ")
        else:
            formatted = super().format(record)

        return formatted


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.

        Args:
            record: Log record to format

        Returns:
            str: JSON-formatted log message
        """
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # Add enhancement-specific fields if available
        if hasattr(record, 'image_name'):
            log_entry['image_name'] = record.image_name
        if hasattr(record, 'algorithm'):
            log_entry['algorithm'] = record.algorithm
        if hasattr(record, 'duration_ms'):
            log_entry['duration_ms'] = record.duration_ms
        if hasattr(record, 'details'):
            log_entry['details'] = record.details
        if hasattr(record, 'confidence'):
            log_entry['confidence'] = record.confidence

        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


class EnhancementLogger:
    """Enhanced logging system with [ ENHANCEMENT ] format."""

    def __init__(self, settings=None):
        """
        Initialize logger.

        Args:
            settings: Optional settings object
        """
        if settings is None:
            settings = get_settings()

        self.settings = settings
        self.log_dir = self.settings.logging.log_dir
        self.log_format = self.settings.logging.log_format
        self.log_level = self.settings.logging.log_level
        self.enable_json = self.settings.logging.enable_json_logging
        self.separate_files = self.settings.logging.separate_log_files

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create loggers
        self.analysis_logger = self._create_logger('analysis', 'analysis.log')
        self.enhancement_logger = self._create_logger('enhancement', 'enhancement.log')
        self.error_logger = self._create_logger('error', 'error.log')
        self.main_logger = self._create_logger('main', 'main.log')

    def _create_logger(self, name: str, filename: str) -> logging.Logger:
        """
        Create a logger with file and console handlers.

        Args:
            name: Logger name
            filename: Log filename

        Returns:
            logging.Logger: Configured logger
        """
        logger = logging.getLogger(f'image_enhancement.{name}')
        logger.setLevel(getattr(logging, self.log_level))
        logger.handlers.clear()  # Clear existing handlers

        # File handler with rotation
        log_path = self.log_dir / filename
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding='utf-8'
        )

        # Choose formatter based on settings
        if self.enable_json:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
        logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        if name == 'error':
            console_handler = logging.StreamHandler(sys.stderr)

        if name in ['enhancement', 'analysis']:
            console_handler.setFormatter(EnhancementFormatter(self.log_format))
        else:
            console_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
        logger.addHandler(console_handler)

        return logger

    def log_enhancement(
        self,
        image_name: str,
        algorithm: str,
        duration_ms: float,
        details: Optional[Dict[str, Any]] = None,
        level: str = "INFO"
    ):
        """
        Log enhancement operation.

        Args:
            image_name: Name of the image being enhanced
            algorithm: Algorithm used
            duration_ms: Processing duration in milliseconds
            details: Additional details as dictionary
            level: Log level
        """
        log_func = getattr(self.enhancement_logger, level.lower())

        extra = {
            'image_name': image_name,
            'algorithm': algorithm,
            'duration_ms': duration_ms,
            'details': details or {}
        }

        message = f"Enhanced image with {algorithm}"
        log_func(message, extra=extra)

    def log_analysis(
        self,
        image_name: str,
        analysis: Dict[str, Any],
        level: str = "INFO"
    ):
        """
        Log image analysis results.

        Args:
            image_name: Name of the image analyzed
            analysis: Analysis results dictionary
            level: Log level
        """
        log_func = getattr(self.analysis_logger, level.lower())

        extra = {
            'image_name': image_name,
            'details': analysis
        }

        # Extract key metrics for message
        resolution = analysis.get('resolution', {})
        brightness = analysis.get('brightness', {})
        faces = analysis.get('faces', {})

        message = f"Analyzed image: {resolution.get('width', 0)}x{resolution.get('height', 0)}, "
        message += f"Brightness: {brightness.get('mean', 0):.2f}, "
        message += f"Faces: {faces.get('count', 0)}"

        log_func(message, extra=extra)

    def log_error(
        self,
        image_name: str,
        error: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Log error.

        Args:
            image_name: Name of the image where error occurred
            error: Error message
            details: Additional error details
        """
        extra = {
            'image_name': image_name,
            'details': details or {}
        }

        self.error_logger.error(error, extra=extra)

    def log_workflow(
        self,
        image_name: str,
        total_duration_ms: float,
        algorithms_used: List[str],
        level: str = "INFO"
    ):
        """
        Log complete workflow for an image.

        Args:
            image_name: Name of the image
            total_duration_ms: Total processing duration
            algorithms_used: List of algorithms applied
            level: Log level
        """
        log_func = getattr(self.main_logger, level.lower())

        extra = {
            'image_name': image_name,
            'duration_ms': total_duration_ms,
            'details': {'algorithms': algorithms_used}
        }

        message = f"Workflow completed for {image_name} in {total_duration_ms:.2f}ms"
        message += f" using {len(algorithms_used)} algorithms: {', '.join(algorithms_used)}"

        log_func(message, extra=extra)

    @contextmanager
    def log_operation(self, operation_name: str, **kwargs):
        """
        Context manager for logging operations with timing.

        Args:
            operation_name: Name of the operation
            **kwargs: Additional fields to log

        Yields:
            function: Function to call when done
        """
        import time
        start_time = time.time()

        def log_success(additional_details: Dict[str, Any] = None):
            """Log successful operation completion."""
            duration_ms = (time.time() - start_time) * 1000
            details = kwargs.copy()
            if additional_details:
                details.update(additional_details)

            self.log_enhancement(
                image_name=kwargs.get('image_name', 'N/A'),
                algorithm=operation_name,
                duration_ms=duration_ms,
                details=details
            )

        def log_error(error_message: str):
            """Log operation error."""
            self.log_error(
                image_name=kwargs.get('image_name', 'N/A'),
                error=error_message,
                details=kwargs
            )

        yield log_success, log_error


def get_logger(settings=None) -> EnhancementLogger:
    """
    Get or create EnhancementLogger instance.

    Args:
        settings: Optional settings object

    Returns:
        EnhancementLogger: Logger instance
    """
    return EnhancementLogger(settings)


if __name__ == "__main__":
    # Test the logging system
    logger = get_logger()

    # Test analysis logging
    test_analysis = {
        'resolution': {'width': 1920, 'height': 1080, 'is_small': False},
        'brightness': {'mean': 0.45, 'std': 0.2, 'is_low_light': False},
        'faces': {'count': 2, 'has_faces': True}
    }
    logger.log_analysis('test_image.jpg', test_analysis)

    # Test enhancement logging
    logger.log_enhancement(
        image_name='test_image.jpg',
        algorithm='real_esrgan',
        duration_ms=1500.5,
        details={'scale': 4, 'output_size': '7680x4320'}
    )

    # Test workflow logging
    logger.log_workflow(
        image_name='test_image.jpg',
        total_duration_ms=2500.75,
        algorithms_used=['real_esrgan', 'white_balance', 'face_enhancement']
    )

    # Test error logging
    logger.log_error(
        image_name='test_image.jpg',
        error='Failed to load model',
        details={'model_path': '/path/to/model.pth', 'error_code': 404}
    )

    print("Logging system test completed!")
