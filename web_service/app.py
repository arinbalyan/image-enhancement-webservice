"""
Image Enhancement Web Service - FastAPI Application

This module provides the main FastAPI application for the image enhancement service.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import router from routes
from web_service.api.routes import router

# Import utilities
from utils.logger import get_logger
from config.settings import get_settings

logger = get_logger(__name__)
settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title="Image Enhancement Web Service",
    description="AI-powered image enhancement service with super-resolution, low-light enhancement, color correction, and face restoration",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)


@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Image Enhancement Web Service",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "enhance": "/api/v1/enhance",
            "enhance_file": "/api/v1/enhance/file",
            "enhance_batch": "/api/v1/enhance/batch",
            "batch_status": "/api/v1/batch/{batch_id}",
            "algorithms": "/api/v1/algorithms",
            "status": "/api/v1/status",
            "docs": "/docs",
            "redoc": "/redoc"
        },
        "documentation": "/docs",
        "health": "/api/v1/status"
    }


@app.get("/health", response_class=JSONResponse)
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "image-enhancement-api"
    }


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting Image Enhancement Web Service...")
    gpu_enabled = settings.processing.enable_gpu
    max_size = settings.processing.max_image_size_mb
    logger.info(f"Settings: GPU={gpu_enabled}, Max Image Size={max_size}MB")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Image Enhancement Web Service...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "web_service.app:app",
        host="0.0.0.0",
        port=7860,
        log_level="info"
    )
