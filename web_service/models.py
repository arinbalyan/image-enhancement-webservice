from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Dict, Any
from uuid import uuid4


class AlgorithmInfo(BaseModel):
    name: str
    category: str
    description: str
    parameters: Dict[str, Any]
    enabled: bool


class EnhancementParameters(BaseModel):
    upscale_factor: str = Field(default="auto", description="Scale factor: 'auto', 2, 4, 8, 16")
    quality: str = Field(default="high", description="Quality setting: 'low', 'medium', 'high'")
    preserve_original: bool = Field(default=True, description="Preserve original image metadata")


class EnhanceRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image data or image URL")
    algorithms: List[str] = Field(default=["auto"], description="List of algorithms to apply")
    parameters: EnhancementParameters = Field(default_factory=EnhancementParameters)


class BatchImage(BaseModel):
    filename: str
    data: str = Field(..., description="Base64 encoded image data")


class BatchEnhanceRequest(BaseModel):
    images: List[BatchImage] = Field(..., min_items=1, max_items=10)
    algorithms: List[str] = Field(default=["auto"], description="List of algorithms to apply")
    parameters: EnhancementParameters = Field(default_factory=EnhancementParameters)


class EnhancementStep(BaseModel):
    algorithm: str
    duration_ms: float
    parameters: Optional[Dict[str, Any]] = None


class SystemStatus(BaseModel):
    status: str
    version: str = "1.0.0"
    uptime_seconds: float
    device: str
    models_loaded: List[str]
    memory_usage_mb: float


class EnhancedImageData(BaseModel):
    width: int
    height: int
    size_bytes: int
    url: Optional[HttpUrl] = None
    base64: Optional[str] = None


class EnhanceResponse(BaseModel):
    status: str
    image_id: str
    original_image: Optional[Dict[str, Any]] = None
    enhanced_image: EnhancedImageData
    enhancements_applied: List[EnhancementStep]
    total_duration_ms: float
    logs: List[str]


class BatchEnhanceResponse(BaseModel):
    batch_id: str
    status: str
    message: str
    status_url: str


class BatchStatusResponse(BaseModel):
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
    status: str = "error"
    error: str
    detail: Optional[Dict[str, Any]] = None
