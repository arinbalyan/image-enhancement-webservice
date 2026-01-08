# Project Context

## Project Summary
**Name**: Image Enhancement Web Service
**Status**: Foundation & Setup Complete
**Start Date**: January 8, 2026
**Estimated Duration**: 10+ weeks

## Quick Overview

A comprehensive image enhancement web service that uses advanced AI algorithms to enhance mobile phone photos. The system provides modular enhancement capabilities including super-resolution (16x for small images, 4x for normal), low-light enhancement, white balancing, face analysis, and color grading.

### Key Features
- **Modular Pipeline**: Separate Python modules for each enhancement algorithm
- **Smart Analysis**: Deep image analysis to automatically select appropriate enhancements
- **Quality-First**: No time constraints, prioritize quality over speed
- **Phased Development**: Algorithms → Testing → Web Service → Google Drive Integration
- **Comprehensive Logging**: [ ENHANCEMENT ] format for full transparency

### Tech Stack
- **Framework**: Python 3.10+, PyTorch 2.0+, FastAPI
- **Image Processing**: OpenCV, Pillow, NumPy
- **Google Drive**: google-api-python-client, OAuth2
- **Logging**: Python logging with JSON structure

## Architecture Highlights

### Directory Structure
```
imgEnhanceWebService/
├── .opencode/docs/          # Documentation (PRD, ARCHITECTURE, TASKS)
├── algorithms/              # Enhancement modules
│   ├── base_enhancer.py    # Base class
│   ├── manager.py          # Algorithm orchestrator
│   ├── super_resolution/   # Upscaling algorithms
│   ├── low_light/          # Low-light enhancement
│   ├── color_correction/   # White balance, exposure
│   └── face_analysis/      # Face detection & enhancement
├── utils/
│   ├── logger.py           # Logging system
│   ├── image_analyzer.py   # Image analysis
│   └── io.py               # File I/O
├── config/                  # Configuration
├── test_images/            # Testing input
├── test_output/            # Testing output
├── web_service/            # FastAPI web service
├── scripts/
│   ├── test_runner.py      # Test orchestrator
│   └── batch_processor.py  # Google Drive batch processing
└── main.py                 # Main entry point
```

### Component Interaction
1. **Input Image** → Image Analyzer (detects brightness, faces, colors, resolution)
2. **Analyzer** → Algorithm Manager (selects algorithms)
3. **Manager** → Enhancement Pipeline (parallel/sequential execution)
4. **Pipeline** → Enhanced Image Output

## Development Phases

### Phase 1: Foundation & Algorithms (Weeks 1-3)
- Project setup and base infrastructure
- Image analyzer implementation
- Integration of all enhancement algorithms:
  - Super-resolution: Real-ESRGAN, super-image, Real-Time-SR, 4x_SR_CNN, PCSR, 4KAgent
  - Low-light: CLAHE, CSEC
  - Color correction: Deep White Balance, Exposure Correction
  - Face analysis: Face detection and enhancement
- Algorithm manager creation
- Logging system implementation

### Phase 2: Testing & Validation (Weeks 4-5)
- Individual algorithm testing
- Combined enhancement testing
- Human validation with user's test images
- Iterative improvement based on feedback
- Performance optimization

### Phase 3: Web Service (Weeks 6-7)
- FastAPI web service creation
- API endpoints: /enhance, /enhance/batch, /algorithms, /status
- Async processing with background tasks
- API documentation with Swagger UI
- Authentication (if needed)
- Deployment and monitoring

### Phase 4: Google Drive Integration (Weeks 8-9)
- Google Drive API integration with OAuth2
- Batch processing script
- Folder structure preservation
- Progress tracking and resume capability
- Error handling and recovery

### Phase 5: Polish & Launch (Weeks 10+)
- Documentation (README, API docs, user guides)
- Attribution for all integrated algorithms
- Performance optimization
- Monitoring and metrics
- Community engagement

## Key Decisions

| Decision | Rationale |
|----------|-----------|
| Modular architecture | Easy to add/remove algorithms, maintainable code |
| Quality over speed | User prioritizes quality, no time constraints |
| Phased approach | Validate algorithms before building webservice |
| 16x/4x upscaling | User requirement for small vs normal images |
| FastAPI | Modern, async support, automatic documentation |
| PyTorch | Industry standard, extensive model ecosystem |
| Structured logging | Machine-readable, comprehensive debugging |
| Resume capability | Handle large photo collections gracefully |

## Open-Source Integrations

### Super-Resolution
- Real-ESRGAN: https://github.com/xinntao/Real-ESRGAN
- super-image: https://github.com/eugenesiow/super-image
- Real-Time-SR: https://github.com/braindotai/Real-Time-Super-Resolution
- 4x_SR_CNN: https://github.com/shamim-hussain/4x_superresolution_cnn
- PCSR: https://github.com/3587jjh/PCSR
- 4KAgent: https://github.com/taco-group/4KAgent

### Low-Light Enhancement
- CLAHE: https://github.com/ThomasWangWeiHong/Low-Light-Image-Enhancement-CLAHE-Based
- CSEC: https://github.com/yiyulics/CSEC
- Collection: https://github.com/zhihongz/awesome-low-light-image-enhancement

### Color Correction
- Deep White Balance: https://github.com/mahmoudnafifi/Deep_White_Balance
- Exposure Correction: https://github.com/mahmoudnafifi/Exposure_Correction

### General Enhancement
- FUnIE-GAN: https://github.com/xahidbuffon/FUnIE-GAN

## Current Status

**Status**: Foundation & Setup In Progress

**Completed - Planning Phase**:
- ✅ Project planning with sequential thinking
- ✅ Research of FastAPI and PyTorch best practices
- ✅ Decision storage in memory MCP
- ✅ PRD.md created with comprehensive requirements
- ✅ ARCHITECTURE.md created with technical design
- ✅ TASKS.md created with 62 detailed tasks

**Completed - Foundation & Setup**:
- ✅ Directory structure created (Task 1.1.1)
  - algorithms/ (with subdirectories)
  - utils/, config/
  - test_images/, test_output/
  - web_service/ (with api/ and models/)
  - scripts/
- ✅ Virtual environment created (Task 1.1.2)
- ✅ .gitignore created (Task 1.1.3)
- ✅ requirements.txt created (Task 1.1.4)
- ✅ README.md created (Task 1.1.5)
- ✅ Configuration system implemented (Task 1.1.6-1.1.8)
  - config/settings.py with Pydantic models
  - Environment variable support
  - Algorithm-specific configs
  - get_settings() and get_algorithm_config() functions
- ✅ Logging system implemented (Task 1.1.9-1.1.13)
  - utils/logger.py with EnhancementLogger class
  - [ ENHANCEMENT ] format support
  - JSON logging
  - Log rotation
  - Separate log files (analysis, enhancement, error)
- ✅ Base enhancer class created (Task 1.1.14-1.1.17)
  - algorithms/base_enhancer.py with abstract base class
  - Image format conversion methods (PIL, numpy, tensor)
  - Device management (CPU/CUDA)
  - Model loading/unloading methods
  - Tiling/merging for large images
- ✅ Image analyzer implemented (Task 1.2.1-1.2.7)
  - utils/image_analyzer.py with ImageAnalyzer class
  - Resolution detection and upscale factor logic
  - Brightness and contrast analysis
  - Face detection (using OpenCV's haar cascades)
  - Color analysis and cast detection
  - Noise level estimation
  - Enhancement recommendation engine
- ✅ Utility I/O functions created (Task 1.3.13-1.3.15)
  - utils/io.py with comprehensive I/O utilities
  - Image loading with format detection
  - Image saving with format preservation
  - File listing and validation utilities
- ✅ Algorithm manager created (Task 1.3.6-1.3.12)
  - algorithms/manager.py with full orchestration logic
  - Algorithm loading and discovery
  - Algorithm selection based on image analysis
  - Parallel execution support (structure ready)
  - Model memory management
  - Unified enhance_image method
- ✅ Main.py entry point created (Task 1.4.1-1.4.4)
  - CLI interface with setup, process, test, status commands
  - Integration with algorithm manager
  - Support for batch processing
- ✅ Test runner script created (Task 1.4.1-1.4.4)
  - scripts/test_runner.py with test orchestration
  - Progress tracking
  - Result comparison (before/after)

**Next Steps**:
1. Integrate actual enhancement algorithms (Task 1.2.8-1.3.5)
2. Run tests with sample images in test_images/
3. Human validation and feedback
4. Begin Phase 3: Web Service (after green flag)

## Important Notes

### User Requirements
- "step 1 create a proper pipeline for modular enhancement"
- "seemlessly connected by there parent algo manager python script"
- "test them properly then only we will move forward"
- "human testing i would put images in the test_images folder"
- "get all teh enhanced images in the test_output folder"
- "give you the green flag to create the proper webservice"
- "take the gdrive folder enhance all the images"
- "save those images in a different folder...image-enhancer"
- "proper 16x upscale for very small images and 4x enhancement for normal sized"
- "white balancing and proper face analysis and otehr color grading"
- "make sure that the images dont look very dark"
- "analyse each image very deeply"
- "proper logs for this like this [ ENHANCEMENT ]"
- "time taken by the entire workflow is not restricted we need good results not fast image processing"

### Testing Protocol
1. Create test_images/ folder with sample images
2. Run test_runner.py to process images
3. Review enhanced images in test_output/
4. Provide feedback on quality
5. Iterate improvements
6. Get "green flag" before proceeding to webservice

### Google Drive Integration
- Input folder: raw_images/ (or user-specified)
- Output folder: image-enhancer/
- Preserve folder structure
- Add "_enhanced" suffix to filenames
- Support batch processing of all images
- Resume capability for interrupted jobs

## Success Criteria

### Quality
- User rating ≥ 4/5 stars for enhanced images
- Natural appearance, no artifacts
- Proper brightness and colors
- Sharp details after upscaling

### Technical
- Algorithm integration success rate ≥ 95%
- Error rate for batch processing ≤ 2%
- No memory leaks
- Proper logging of all operations

### User Experience
- Clear, comprehensive documentation
- Easy to use configuration
- Helpful error messages
- Reliable batch processing

## Resources

### Documentation
- PRD: `.opencode/docs/PRD.md` - Product requirements and specifications
- ARCHITECTURE: `.opencode/docs/ARCHITECTURE.md` - Technical design and architecture
- TASKS: `.opencode/docs/TASKS.md` - Detailed implementation tasks (62 tasks)
- CONTEXT: `.opencode/docs/context.md` - This file

### Memory (MCP)
Key decisions and component information stored in memory MCP for future reference:
- Image Enhancement Web Service (project overview)
- Algorithm Manager (orchestration)
- Image Analyzer (analysis logic)
- Logging System (logging format)
- Google Drive Integration (batch processing)

### External References
- FastAPI docs: https://fastapi.tiangolo.com
- PyTorch docs: https://pytorch.org/docs
- Google Drive API: https://developers.google.com/drive/api

## Quick Start Commands (Future)

```bash
# Setup
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# Testing
python scripts/test_runner.py

# Web Service
uvicorn web_service.app:app --reload

# Google Drive Batch Processing
python scripts/batch_processor.py --input /path/to/raw_images --output /path/to/image-enhancer
```

## Contact & Support

For questions or issues, refer to:
- This context document for overview
- ARCHITECTURE.md for technical details
- TASKS.md for implementation tasks
- PRD.md for requirements and user stories

---

*Generated by OpenCode Kiro Planning*
*Last Updated: Thu Jan 08 2026*
*Documentation stored in: .opencode/docs/*
