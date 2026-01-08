# Image Enhancement Web Service

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

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd imgEnhanceWebService
```

2. Create a virtual environment:
```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On Linux/Mac:
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Testing Enhancement Pipeline

1. Place test images in `test_images/` directory
2. Run the test runner:
```bash
python scripts/test_runner.py
```

3. Find enhanced images in `test_output/` directory

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
pytest
```

### Code Formatting

```bash
black .
```

### Linting

```bash
flake8 .
```

### Type Checking

```bash
mypy .
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

## Roadmap

- [x] Project planning and documentation
- [ ] Foundation and algorithm integration
- [ ] Testing and validation
- [ ] Web service implementation
- [ ] Google Drive integration
- [ ] Performance optimization
- [ ] Documentation and community engagement

---

**Status**: Planning Complete - Implementation In Progress
**Last Updated**: January 8, 2026
