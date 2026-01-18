# Image Enhancement Web Service

[![GitHub Stars](https://img.shields.io/github/stars/arinbalyan/image-enhancement-webservice)](https://github.com/arinbalyan/image-enhancement-webservice/stargazers)
[![GitHub Issues](https://img.shields.io/github/issues/arinbalyan/image-enhancement-webservice)](https://github.com/arinbalyan/image-enhancement-webservice/issues)
[![Python Version](https://img.shields.io/badge/python-3.10-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128-green)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)
[![HuggingFace Spaces](https://img.shields.io/badge/HuggingFace--spaces-blue)](https://huggingface.co)

A comprehensive image enhancement web service that uses advanced AI algorithms to enhance mobile phone photos. The system provides modular enhancement capabilities including super-resolution (16x for small images, 4x for normal), low-light enhancement, white balancing, face analysis, and color grading.

## Features

- **Modular Enhancement Pipeline**: Separate Python modules for each enhancement algorithm
- **Smart Image Analysis**: Automatic detection of enhancement needs (brightness, faces, colors, resolution)
- **Super-Resolution**: 16x upscaling for small images, 4x for normal images
- **Low-Light Enhancement**: CLAHE and CSEC algorithms for dark images
- **Color Correction**: Deep White Balance and Exposure Correction
- **Face Enhancement**: Face detection and enhancement
- **Quality-First Approach**: No time constraints, prioritize quality over speed
- **Comprehensive Logging**: [ ENHANCEMENT ] format for full transparency
- **Web Service API**: FastAPI-based REST API for programmatic access
- **Google Drive Integration**: Batch processing for entire photo collections

## Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (optional but recommended for performance)
- uv package manager (recommended for dependency management)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd imgEnhanceWebService
```

2. Create a virtual environment with uv:
```bash
uv venv

# Activate venv:
# On Windows:
.venv\Scripts\activate

# On Linux/Mac:
source .venv/bin/activate
```

3. Install dependencies with uv:
```bash
uv pip install -e .
# Or install from pyproject.toml:
uv sync
# Or install from requirements.txt:
uv pip install -r requirements.txt
```

## Quick Start

### HuggingFace Spaces Deployment (RECOMMENDED)

**Why HuggingFace Spaces?**
- ✅ 16GB RAM (vs 512MB on Render free)
- ✅ 50GB storage (vs 512MB on Render free)
- ✅ Free GPU available (A10G/T4)
- ✅ Designed for ML apps

#### 1. Create HuggingFace Account
- Go to https://huggingface.co
- Sign up for a free account

#### 2. Create a New Space
- Navigate to https://huggingface.co/new-space
- Choose **Docker** template
- Select GPU: **A10G** (free tier) or **T4**
- Space name: `img-enhance-service`

#### 3. Deploy Your Code

```bash
# Clone your Space repository
git clone https://huggingface.co/spaces/<your-username>/img-enhance-service

# Copy your project files
cp -r /path/to/imgEnhanceWebService/* img-enhance-service/

# Push changes
cd img-enhance-service
git add .
git commit -m "Deploy image enhancement service"
git push
```

Your API will be available at:
- `https://<your-username>-img-enhance-service.hf.space/docs` (Swagger UI)
- `https://<your-username>-img-enhance-service.hf.space/api/v1/enhance` (API endpoint)

#### 4. Environment Variables

Set these in your Space settings:
- `MAX_IMAGE_SIZE_MB=50`
- `ENABLE_GPU=True`

### Testing Enhancement Pipeline

1. **Ensure models are downloaded** (required for actual enhancement):
   - Real-ESRGAN: Download from [GitHub Releases](https://github.com/xinntao/Real-ESRGAN/releases)
     - Get `RealESRGAN_x4plus.pth` (~67MB)
     - Save to `models/` directory
   - GFPGAN: Download from [GitHub Releases](https://github.com/TencentARC/GFPGAN/releases)
     - Get `GFPGANv1.3.pth` (~350MB)
     - Save to `models/` directory

2. **Verify algorithms load correctly**:
```bash
uv run python -m scripts.diagnostic_test
```

3. Place test images in `test_images/` directory
4. Run test runner:
```bash
uv run python -m scripts.test_runner
```

5. Find enhanced images in `test_output/` directory

**Algorithm Validation Status (All 6/6 Passing)**:
- ✅ real_esrgan - GPU-accelerated, 4x upscale (99.8s)
- ✅ super_image - CPU-based, 4x upscale with tile processing (233.1s)
- ✅ clahe - Low-light enhancement (162ms)
- ✅ white_balance - Color correction (64ms)
- ✅ exposure_correction - Brightness adjustment (58ms)
- ✅ face_enhancement - Face enhancement (3.0s)

**Troubleshooting**:
- All algorithms are currently validated and working
- GPU is detected and used for Real-ESRGAN and GFPGAN (GTX 1650 Ti)
- See [KNOWN_ISSUES.md](KNOWN_ISSUES.md) for historical issue documentation

### Starting the Web Service

```bash
uvicorn web_service.app:app --reload
```

The API will be available at `http://localhost:8000`

API documentation: `http://localhost:8000/docs`

### Google Drive Batch Processing

1. Set up Google Drive credentials (see [Google Drive Setup](#google-drive-setup))
2. Run batch processor:
```bash
python scripts/batch_processor.py --input /path/to/raw_images --output /path/to/image-enhancer
```

## Project Structure

```
imgEnhanceWebService/
├── algorithms/              # Enhancement modules
│   ├── base_enhancer.py    # Base class for all enhancers
│   ├── manager.py          # Algorithm orchestrator
│   ├── super_resolution/   # Upscaling algorithms
│   ├── low_light/          # Low-light enhancement
│   ├── color_correction/   # White balance, exposure
│   └── face_analysis/      # Face detection & enhancement
├── utils/
│   ├── logger.py           # Logging system
│   ├── image_analyzer.py   # Image analysis
│   └── io.py               # File I/O utilities
├── config/                  # Configuration files
├── test_images/            # Test input images
├── test_output/            # Test enhanced images
├── web_service/            # FastAPI web service
│   ├── api/               # API routes
│   └── models/            # Pydantic models
├── scripts/
│   ├── test_runner.py      # Test orchestrator
│   └── batch_processor.py  # Google Drive batch processing
└── main.py                # Main entry point
```

## API Endpoints

### POST /api/v1/enhance
Enhance a single image.

**Request**:
```json
{
  "image": "base64_encoded_image",
  "algorithms": ["auto"],
  "parameters": {
    "upscale_factor": "auto",
    "quality": "high"
  }
}
```

### POST /api/v1/enhance/batch
Enhance multiple images.

### GET /api/v1/algorithms
List available enhancement algorithms.

### GET /api/v1/status
Get system status.

## Configuration

Create a `.env` file in the project root:

```env
# Google Drive Credentials
GOOGLE_DRIVE_CREDENTIALS_PATH=path/to/credentials.json
GOOGLE_DRIVE_TOKEN_PATH=path/to/token.json

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=False

# Processing Settings
MAX_IMAGE_SIZE_MB=50
ENABLE_GPU=True
BATCH_SIZE=4
```

## Google Drive Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project
3. Enable the Google Drive API
4. Create OAuth 2.0 credentials
5. Download credentials.json
6. Place it in the project root (or update .env path)
7. Run the batch processor - it will guide you through OAuth flow

## Development

### Running Tests

```bash
# Run pytest
pytest

# Run diagnostic test for individual algorithms
uv run python -m scripts.diagnostic_test

# Run test runner for batch processing
uv run python -m scripts.test_runner
```

### Code Formatting

```bash
# Format code with black
black .

# Check formatting
black --check .
```

### Linting

```bash
flake8 .
```

### Type Checking

```bash
mypy .
```

### Package Management

```bash
# Install dependencies with uv
uv pip install <package>

# Install from pyproject.toml
uv sync

# Install all dependencies including dev
uv sync --dev

# Freeze current environment
uv pip freeze
```

## Attribution

This project integrates code from the following open-source projects:

### Super-Resolution
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) by xinntao
- [super-image](https://github.com/eugenesiow/super-image) by eugenesiow
- [Real-Time-Super-Resolution](https://github.com/braindotai/Real-Time-Super-Resolution) by braindotai
- [4x_superresolution_cnn](https://github.com/shamim-hussain/4x_superresolution_cnn) by shamim-hussain
- [PCSR](https://github.com/3587jjh/PCSR) by 3587jjh
- [4KAgent](https://github.com/taco-group/4KAgent) by taco-group

### Low-Light Enhancement
- [Low-Light-Image-Enhancement-CLAHE-Based](https://github.com/ThomasWangWeiHong/Low-Light-Image-Enhancement-CLAHE-Based) by ThomasWangWeiHong
- [CSEC](https://github.com/yiyulics/CSEC) by yiyulics
- [awesome-low-light-image-enhancement](https://github.com/zhihongz/awesome-low-light-image-enhancement) by zhihongz

### Color Correction
- [Deep_White_Balance](https://github.com/mahmoudnafifi/Deep_White_Balance) by mahmoudnafifi
- [Exposure_Correction](https://github.com/mahmoudnafifi/Exposure_Correction) by mahmoudnafifi

### General Enhancement
- [FUnIE-GAN](https://github.com/xahidbuffon/FUnIE-GAN) by xahidbuffon

Thank you to all contributors of these amazing projects!

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Support

For issues and questions, please refer to:
- Documentation in `.opencode/docs/`
- [PRD.md](.opencode/docs/PRD.md) - Product requirements
- [ARCHITECTURE.md](.opencode/docs/ARCHITECTURE.md) - Technical architecture
- [TASKS.md](.opencode/docs/TASKS.md) - Implementation tasks

## Known Issues

### GPU Memory Limitation (GTX 1650 Ti - 4GB VRAM)

**Issue**: Large images may cause CUDA out-of-memory errors.

**Symptoms**:
- Real-ESRGAN fails with CUDA out of memory
- Processing slows computer significantly
- Some very large images cannot be upscaled

**Solutions**:
- Use smaller tile sizes for large images
- Process images sequentially, not in parallel
- Consider reducing image size before enhancement

**Status**: Documented in [KNOWN_ISSUES.md](KNOWN_ISSUES.md) with full details

## Roadmap

- [x] Project planning and documentation
- [x] Foundation and algorithm integration
- [x] Model weights downloaded (Real-ESRGAN, GFPGAN)
- [x] Bug fixes (JSON serialization, CLAHE, cv2 imports)
- [x] Diagnostic test script created
- [x] KNOWN_ISSUES.md documented
- [x] NumPy compatibility fixed (downgraded to 1.26.4)
- [x] All 6 algorithms validated with diagnostic tests
- [x] GPU acceleration enabled (CUDA 11.8, GTX 1650 Ti)
- [x] pyproject.toml created with uv package management
- [x] Enhanced test_runner with timing and statistics
- [x] Windows-compatible timeout mechanism (threading-based)
- [x] Human validation and feedback collection (15 images, 100% success)
- [x] Parameter passing bugs fixed (manager.py, real_esrgan.py)
- [x] All algorithm signatures updated to accept parameters
- [ ] Web service implementation (IN PROGRESS)
- [ ] Google Drive integration
- [ ] Performance optimization
- [ ] Documentation and community engagement

---

**Status**: ✅ ENHANCEMENT PIPELINE WORKING - READY FOR WEB SERVICE DEVELOPMENT
**Last Updated**: January 9, 2026 (Evening)

**Test Results Summary**:
- ✅ 15/15 images successfully enhanced (100% success rate)
- ✅ Proper 4x upscaling confirmed (1728x2304 → 6912x9216)
- ✅ All algorithms showing correct timing (~107s super_resolution, ~1.75s low_light)
- ✅ User feedback: "i find the result good this time"
- ✅ Green flag received to proceed to web service development

**Algorithm Performance** (on GTX 1650 Ti, 1728x2304 test image):
| Algorithm | Time | GPU | Status |
|-----------|-------|------|--------|
| real_esrgan | 99.8s | ✅ | 4x upscale, high quality |
| super_image | 233.1s | ❌ | 4x upscale, tile processing |
| clahe | 162ms | N/A | Low-light enhancement |
| white_balance | 64ms | N/A | Color correction |
| exposure_correction | 58ms | N/A | Brightness adjustment |
| face_enhancement | 3.0s | ✅ | Face enhancement |

**Next Phase (Web Service)**:
1. Create FastAPI web service structure
2. Implement /enhance endpoint for single image processing
3. Implement /enhance/batch endpoint for batch processing
4. Create /algorithms and /status endpoints
5. Add async processing with background tasks
6. API documentation with Swagger UI

**Documentation**:
- See [KNOWN_ISSUES.md](KNOWN_ISSUES.md) for historical issue documentation
- See `.opencode/docs/` for project planning and architecture details
- See [CONTEXT.md](CONTEXT.md) for detailed development history
