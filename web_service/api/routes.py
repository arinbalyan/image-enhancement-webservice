"""
API Routes for Image Enhancement Web Service.

Provides endpoints for image enhancement, batch processing, and system status.
"""

from fastapi import APIRouter, UploadFile, HTTPException, BackgroundTasks, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import base64
import io
import tempfile
import os
from typing import Dict, Any, List, Optional
import uuid
from datetime import datetime
from pathlib import Path

# Import project modules
from algorithms.manager import AlgorithmManager
from utils.logger import get_logger
from config.settings import get_settings


# ============================================================================
# Pydantic Models (inline to avoid import conflicts)
# ============================================================================

class AlgorithmInfo(BaseModel):
    """Information about an available algorithm."""
    name: str
    category: str
    description: str
    parameters: Dict[str, Any]
    enabled: bool


class EnhancementParameters(BaseModel):
    """Parameters for image enhancement."""
    upscale_factor: str = Field(default="auto", description="Scale factor: 'auto', 2, 4, 8, 16")
    quality: str = Field(default="high", description="Quality setting: 'low', 'medium', 'high'")
    preserve_original: bool = Field(default=True, description="Preserve original image metadata")


class EnhanceRequest(BaseModel):
    """Request to enhance a single image."""
    image: str = Field(..., description="Base64 encoded image data or image URL")
    algorithms: List[str] = Field(default=["auto"], description="List of algorithms to apply")
    parameters: Optional[EnhancementParameters] = None


class BatchImage(BaseModel):
    """Single image in a batch request."""
    filename: str
    data: str = Field(..., description="Base64 encoded image data")


class BatchEnhanceRequest(BaseModel):
    """Request to enhance multiple images."""
    images: List[BatchImage] = Field(..., min_length=1, max_length=10)
    algorithms: List[str] = Field(default=["auto"], description="List of algorithms to apply")
    parameters: Optional[EnhancementParameters] = None


class EnhancementStep(BaseModel):
    """Information about a single enhancement step."""
    algorithm: str
    duration_ms: float
    parameters: Optional[Dict[str, Any]] = None


class SystemStatus(BaseModel):
    """System status information."""
    status: str
    version: str = "1.0.0"
    uptime_seconds: float
    device: str
    models_loaded: List[str]
    memory_usage_mb: float


class EnhancedImageData(BaseModel):
    """Enhanced image data."""
    width: int
    height: int
    size_bytes: int
    url: Optional[str] = None
    base64: Optional[str] = None


class EnhanceResponse(BaseModel):
    """Response from image enhancement."""
    status: str
    image_id: str
    original_image: Optional[Dict[str, Any]] = None
    enhanced_image: EnhancedImageData
    enhancements_applied: List[EnhancementStep]
    total_duration_ms: float
    logs: List[str]


class BatchEnhanceResponse(BaseModel):
    """Response from batch enhancement request."""
    batch_id: str
    status: str
    message: str
    status_url: str


class BatchStatusResponse(BaseModel):
    """Status of a batch processing job."""
    batch_id: str
    status: str
    total_images: int
    completed: int
    failed: int
    in_progress: int
    progress_percentage: float
    estimated_remaining_time_seconds: float
    started_at: str
    estimated_completion_at: str
    images: List[Dict[str, Any]]


class ErrorResponse(BaseModel):
    """Error response."""
    status: str = "error"
    error: str
    detail: Optional[Dict[str, Any]] = None


# ============================================================================
# Router Setup
# ============================================================================

router = APIRouter(prefix="/api/v1")
logger = get_logger(__name__)
settings = get_settings()

# Global algorithm manager instance
_algorithm_manager: Optional[AlgorithmManager] = None

# Batch job storage (in-memory for now)
_batch_jobs: Dict[str, Dict[str, Any]] = {}


async def get_algorithm_manager() -> AlgorithmManager:
    """Get or create algorithm manager instance."""
    global _algorithm_manager
    if _algorithm_manager is None:
        _algorithm_manager = AlgorithmManager()
        logger.info("Algorithm manager initialized")
    return _algorithm_manager


# ============================================================================
# API Endpoints
# ============================================================================

@router.get("/algorithms", response_model=List[AlgorithmInfo])
async def list_algorithms() -> List[AlgorithmInfo]:
    """List all available enhancement algorithms."""
    algorithms_info = [
        AlgorithmInfo(
            name="real_esrgan",
            category="super_resolution",
            description="Real-ESRGAN super-resolution model for high-quality upscaling",
            parameters={"scale": {"type": "int", "default": 4, "min": 2, "max": 16}},
            enabled=True
        ),
        AlgorithmInfo(
            name="super_image",
            category="super_resolution",
            description="Super-image library for alternative upscaling",
            parameters={"scale": {"type": "int", "default": 4, "min": 2, "max": 4}},
            enabled=True
        ),
        AlgorithmInfo(
            name="clahe",
            category="low_light",
            description="CLAHE-based low-light enhancement",
            parameters={"clip_limit": {"type": "float", "default": 2.0, "min": 1.0, "max": 10.0}},
            enabled=True
        ),
        AlgorithmInfo(
            name="white_balance",
            category="color_correction",
            description="Deep white balance correction",
            parameters={},
            enabled=True
        ),
        AlgorithmInfo(
            name="exposure_correction",
            category="color_correction",
            description="Exposure and brightness correction",
            parameters={},
            enabled=True
        ),
        AlgorithmInfo(
            name="face_enhancement",
            category="face_analysis",
            description="Face detection and enhancement using GFPGAN",
            parameters={"strength": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0}},
            enabled=True
        )
    ]
    
    logger.info(f"Listed {len(algorithms_info)} algorithms")
    return algorithms_info


@router.get("/status", response_model=SystemStatus)
async def get_status() -> SystemStatus:
    """Get system status and health."""
    import torch
    
    try:
        import psutil
        mem = psutil.virtual_memory()
        memory_usage_mb = round(mem.used / (1024 * 1024), 2)
    except ImportError:
        memory_usage_mb = 0.0
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    manager = await get_algorithm_manager()
    models_loaded = list(manager.loaded_algorithms) if hasattr(manager, 'loaded_algorithms') else []
    
    status = SystemStatus(
        status="healthy",
        version="1.0.0",
        uptime_seconds=0.0,
        device=device,
        models_loaded=models_loaded,
        memory_usage_mb=memory_usage_mb
    )
    
    logger.info(f"System status: {status.status}, device: {device}")
    return status


@router.post("/enhance", response_model=EnhanceResponse)
async def enhance_image(request: EnhanceRequest) -> EnhanceResponse:
    """
    Enhance a single image.
    
    Accepts base64 encoded image data or image URL.
    """
    logger.info("Received enhance request")
    
    try:
        from PIL import Image
        import numpy as np
        
        manager = await get_algorithm_manager()
        
        # Parse image data
        if request.image.startswith('data:image/'):
            # Base64 with data URI
            header, data = request.image.split(',', 1)
            image_bytes = base64.b64decode(data)
            image = Image.open(io.BytesIO(image_bytes))
        elif request.image.startswith(('http://', 'https://')):
            # URL
            import requests
            response = requests.get(request.image, timeout=30)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))
        else:
            # Assume raw base64
            try:
                image_bytes = base64.b64decode(request.image)
                image = Image.open(io.BytesIO(image_bytes))
            except Exception:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid image format. Provide base64 data or URL."
                )
        
        # Convert to numpy array
        image_array = np.array(image)
        original_shape = image_array.shape
        logger.info(f"Image loaded: {original_shape}")
        
        # Save to temp file for processing
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_input:
            image.save(tmp_input.name)
            input_path = tmp_input.name
        
        # Create temp output directory
        with tempfile.TemporaryDirectory() as tmp_output_dir:
            start_time = datetime.now()
            
            # Determine algorithms to use
            algorithms_to_use = None
            if request.algorithms and request.algorithms != ["auto"]:
                algorithms_to_use = request.algorithms
            
            # Enhance image
            enhanced_path = manager.enhance_image(
                image_path=input_path,
                output_dir=tmp_output_dir,
                algorithms=algorithms_to_use,
                preserve_original=True
            )
            
            end_time = datetime.now()
            total_duration_ms = (end_time - start_time).total_seconds() * 1000
            
            # Load enhanced image
            enhanced_image = Image.open(enhanced_path)
            enhanced_array = np.array(enhanced_image)
            
            # Convert to base64
            img_byte_arr = io.BytesIO()
            enhanced_image.save(img_byte_arr, format='JPEG', quality=95)
            img_byte_arr.seek(0)
            enhanced_base64 = base64.b64encode(img_byte_arr.getvalue()).decode()
        
        # Clean up temp input file
        try:
            os.unlink(input_path)
        except Exception:
            pass
        
        # Build response
        response = EnhanceResponse(
            status="success",
            image_id=str(uuid.uuid4()),
            original_image={
                "width": original_shape[1],
                "height": original_shape[0],
                "size_bytes": image_array.nbytes
            },
            enhanced_image=EnhancedImageData(
                width=enhanced_array.shape[1],
                height=enhanced_array.shape[0],
                size_bytes=enhanced_array.nbytes,
                base64=enhanced_base64
            ),
            enhancements_applied=[],
            total_duration_ms=total_duration_ms,
            logs=[]
        )
        
        logger.info(f"Image enhanced successfully in {total_duration_ms:.2f}ms")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error enhancing image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/enhance/batch", response_model=BatchEnhanceResponse)
async def enhance_batch(
    request: BatchEnhanceRequest,
    background_tasks: BackgroundTasks
) -> BatchEnhanceResponse:
    """
    Enhance multiple images in batch.
    
    Returns immediately with batch ID; processing happens in background.
    """
    batch_id = str(uuid.uuid4())
    logger.info(f"Received batch enhance request: {batch_id}")
    
    # Initialize batch job
    _batch_jobs[batch_id] = {
        "status": "started",
        "total_images": len(request.images),
        "completed": 0,
        "failed": 0,
        "in_progress": len(request.images),
        "started_at": datetime.utcnow().isoformat(),
        "images": []
    }
    
    # Add background task for processing
    background_tasks.add_task(process_batch, batch_id, request)
    
    response = BatchEnhanceResponse(
        batch_id=batch_id,
        status="started",
        message=f"Batch processing started for {len(request.images)} images",
        status_url=f"/api/v1/batch/{batch_id}"
    )
    
    logger.info(f"Batch {batch_id} started")
    return response


async def process_batch(batch_id: str, request: BatchEnhanceRequest):
    """Background task to process batch images."""
    from PIL import Image
    import numpy as np
    
    manager = await get_algorithm_manager()
    
    for idx, batch_image in enumerate(request.images):
        try:
            # Decode image
            image_bytes = base64.b64decode(batch_image.data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                image.save(tmp.name)
                input_path = tmp.name
            
            with tempfile.TemporaryDirectory() as tmp_output_dir:
                start_time = datetime.now()
                
                enhanced_path = manager.enhance_image(
                    image_path=input_path,
                    output_dir=tmp_output_dir,
                    algorithms=None if request.algorithms == ["auto"] else request.algorithms,
                    preserve_original=True
                )
                
                duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # Clean up
            try:
                os.unlink(input_path)
            except Exception:
                pass
            
            # Update batch status
            _batch_jobs[batch_id]["completed"] += 1
            _batch_jobs[batch_id]["in_progress"] -= 1
            _batch_jobs[batch_id]["images"].append({
                "filename": batch_image.filename,
                "status": "completed",
                "duration_ms": duration_ms
            })
            
        except Exception as e:
            logger.error(f"Batch {batch_id}: Failed {batch_image.filename}: {e}")
            _batch_jobs[batch_id]["failed"] += 1
            _batch_jobs[batch_id]["in_progress"] -= 1
            _batch_jobs[batch_id]["images"].append({
                "filename": batch_image.filename,
                "status": "failed",
                "error": str(e)
            })
    
    # Mark batch as complete
    _batch_jobs[batch_id]["status"] = "completed"
    logger.info(f"Batch {batch_id} completed")


@router.get("/batch/{batch_id}", response_model=BatchStatusResponse)
async def get_batch_status(batch_id: str) -> BatchStatusResponse:
    """Get batch processing status."""
    logger.info(f"Checking batch status: {batch_id}")
    
    if batch_id not in _batch_jobs:
        raise HTTPException(status_code=404, detail=f"Batch job {batch_id} not found")
    
    job = _batch_jobs[batch_id]
    total = job["total_images"]
    completed = job["completed"]
    
    response = BatchStatusResponse(
        batch_id=batch_id,
        status=job["status"],
        total_images=total,
        completed=completed,
        failed=job["failed"],
        in_progress=job["in_progress"],
        progress_percentage=(completed / total * 100) if total > 0 else 0.0,
        estimated_remaining_time_seconds=0,
        started_at=job["started_at"],
        estimated_completion_at=job.get("completed_at", datetime.utcnow().isoformat()),
        images=job["images"]
    )
    
    return response


@router.post("/enhance/file")
async def enhance_file(file: UploadFile = File(...)) -> EnhanceResponse:
    """
    Enhance an uploaded image file.
    
    Alternative endpoint that accepts file upload instead of base64.
    """
    logger.info(f"Received file upload: {file.filename}")
    
    try:
        from PIL import Image
        import numpy as np
        
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_array = np.array(image)
        original_shape = image_array.shape
        
        manager = await get_algorithm_manager()
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_input:
            image.save(tmp_input.name)
            input_path = tmp_input.name
        
        with tempfile.TemporaryDirectory() as tmp_output_dir:
            start_time = datetime.now()
            
            enhanced_path = manager.enhance_image(
                image_path=input_path,
                output_dir=tmp_output_dir,
                algorithms=None,
                preserve_original=True
            )
            
            total_duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # Load enhanced image
            enhanced_image = Image.open(enhanced_path)
            enhanced_array = np.array(enhanced_image)
            
            # Convert to base64
            img_byte_arr = io.BytesIO()
            enhanced_image.save(img_byte_arr, format='JPEG', quality=95)
            img_byte_arr.seek(0)
            enhanced_base64 = base64.b64encode(img_byte_arr.getvalue()).decode()
        
        # Clean up
        try:
            os.unlink(input_path)
        except Exception:
            pass
        
        response = EnhanceResponse(
            status="success",
            image_id=str(uuid.uuid4()),
            original_image={
                "width": original_shape[1],
                "height": original_shape[0],
                "size_bytes": image_array.nbytes
            },
            enhanced_image=EnhancedImageData(
                width=enhanced_array.shape[1],
                height=enhanced_array.shape[0],
                size_bytes=enhanced_array.nbytes,
                base64=enhanced_base64
            ),
            enhancements_applied=[],
            total_duration_ms=total_duration_ms,
            logs=[]
        )
        
        logger.info(f"File {file.filename} enhanced in {total_duration_ms:.2f}ms")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error enhancing file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
