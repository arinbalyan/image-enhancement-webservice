# Contributing Guide

Thank you for your interest in contributing to the Image Enhancement Web Service! This guide will help you get started.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Project Structure](#project-structure)
- [Adding New Algorithms](#adding-new-algorithms)
- [Testing](#testing)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

## Prerequisites

- **Python**: 3.10 or higher
- **Git**: For version control
- **CUDA 11.8** (optional but recommended for GPU acceleration)
- **NVIDIA GPU** with CUDA support (optional but recommended)

## Getting Started

### 1. Clone the Repository

```bash
git clone <repository-url>
cd imgEnhanceWebService
```

### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/macOS:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

**Using uv (Recommended):**
```bash
uv pip install -r requirements.txt
```

**Using pip:**
```bash
pip install -r requirements.txt
```

### 4. Download Pre-trained Models

The project requires pre-trained models for certain algorithms. Download them to the `models/` directory:

```bash
# Create models directory if it doesn't exist
mkdir models

# Download Real-ESRGAN model (67MB)
# https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
# Save to: models/RealESRGAN_x4plus.pth

# Download GFPGAN model (350MB)
# https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.3.pth
# Save to: models/GFPGANv1.3.pth
```

### 5. Verify Installation

Run the diagnostic test to verify all algorithms load correctly:

```bash
python -m scripts.diagnostic_test
```

Expected output:
```
[DIAGNOSTIC] Testing algorithm: real_esrgan
[DIAGNOSTIC]   Model loaded: RealESRGANer
[DIAGNOSTIC]   Using device: cuda
[DIAGNOSTIC]   Enhancement successful
...
[DIAGNOSTIC] All algorithms tested successfully
```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

Follow the project structure and coding standards described below.

### 3. Test Your Changes

```bash
# Run diagnostic tests
python -m scripts.diagnostic_test

# Run full test suite (if available)
pytest tests/

# Run linter
flake8 algorithms/ utils/ scripts/

# Run type checker
mypy algorithms/ utils/ scripts/
```

### 4. Commit Your Changes

```bash
git add .
git commit -m "feat: add your feature description"
```

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with a clear description of your changes.

## Project Structure

```
imgEnhanceWebService/
├── .opencode/docs/          # Documentation (PRD, ARCHITECTURE, TASKS)
├── algorithms/              # Enhancement modules
│   ├── base_enhancer.py    # Base class for all algorithms
│   ├── manager.py          # Algorithm orchestrator
│   ├── super_resolution/   # Upscaling algorithms
│   │   ├── real_esrgan.py
│   │   └── super_image.py
│   ├── low_light/          # Low-light enhancement
│   │   └── clahe.py
│   ├── color_correction/   # White balance, exposure
│   │   ├── white_balance.py
│   │   └── exposure.py
│   └── face_analysis/      # Face detection & enhancement
│       └── face_enhancer.py
├── utils/
│   ├── logger.py           # Logging system
│   ├── image_analyzer.py   # Image analysis
│   └── io.py               # File I/O utilities
├── config/                  # Configuration files
│   └── settings.py         # Pydantic settings
├── web_service/            # FastAPI web service
│   ├── api/                # API endpoints
│   └── app.py              # FastAPI application
├── scripts/
│   ├── test_runner.py      # Test orchestrator
│   ├── diagnostic_test.py  # Algorithm diagnostic tool
│   └── batch_processor.py  # Google Drive batch processing
├── models/                  # Pre-trained model storage
├── test_images/            # Testing input images
├── test_output/            # Testing output images
├── logs/                   # Log files
├── main.py                 # Main entry point
├── requirements.txt        # Python dependencies
├── pyproject.toml         # Python package configuration
└── README.md              # Project README
```

## Adding New Algorithms

### Step 1: Create Algorithm File

Create a new file in the appropriate subdirectory:

```bash
# Example: Add new super-resolution algorithm
touch algorithms/super_resolution/your_algorithm.py
```

### Step 2: Implement Algorithm Class

Inherit from `BaseEnhancer` and implement required methods:

```python
from algorithms.base_enhancer import BaseEnhancer
import numpy as np
from typing import Optional, Dict, Any

class YourAlgorithm(BaseEnhancer):
    def __init__(self, device: str = 'cuda', config: Optional[Dict] = None):
        super().__init__(
            name="your_algorithm",
            device=device,
            config=config
        )
        self.model = None
        self.load_model()

    def load_model(self) -> None:
        """Load the pre-trained model."""
        # Your model loading logic here
        pass

    def enhance(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Enhance the input image.

        Args:
            image: Input image as numpy array (RGB format)
            **kwargs: Algorithm-specific parameters

        Returns:
            Enhanced image as numpy array (RGB format)
        """
        # Your enhancement logic here
        return enhanced_image

    def unload_model(self) -> None:
        """Unload the model and free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

### Step 3: Register Algorithm in Manager

Add your algorithm to the algorithm registry in `algorithms/manager.py`:

```python
from algorithms.super_resolution.your_algorithm import YourAlgorithm

# Add to ALGORITHMS dictionary
ALGORITHMS = {
    # ... existing algorithms ...
    'your_algorithm': {
        'class': YourAlgorithm,
        'category': 'super_resolution',
        'description': 'Your algorithm description',
        'requirements': ['gpu'],  # or ['cpu'] or ['gpu', 'cpu']
    }
}
```

### Step 4: Update Configuration

Add algorithm-specific configuration to `config/settings.py`:

```python
class YourAlgorithmSettings(BaseSettings):
    """Configuration for YourAlgorithm."""

    model_path: str = Field(
        default="models/your_model.pth",
        description="Path to pre-trained model"
    )

    param1: float = Field(
        default=1.0,
        description="Description of parameter"
    )

    class Config:
        env_prefix = "YOUR_ALGORITHM_"
```

### Step 5: Test Your Algorithm

Create test images and run the diagnostic test:

```bash
# Add test images to test_images/
python -m scripts.diagnostic_test
```

### Step 6: Update Documentation

Update the following files:
- `README.md`: Add algorithm to the list
- `.opencode/docs/PRD.md`: Update feature list
- `.opencode/docs/ARCHITECTURE.md`: Update algorithm catalog

## Testing

### Running Diagnostic Tests

```bash
# Test all algorithms individually
python -m scripts.diagnostic_test

# Test specific algorithm
python -m scripts.diagnostic_test --algorithm real_esrgan
```

### Running Test Runner

```bash
# Process all test images
python -m scripts.test_runner

# Process specific images
python -m scripts.test_runner --input test_images/*.jpg

# Use specific algorithms
python -m scripts.test_runner --algorithms real_esrgan,clahe
```

### Running Unit Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_algorithms.py

# Run with coverage
pytest --cov=algorithms --cov=utils
```

## Code Style

### Python Style Guide

We follow PEP 8 guidelines. Use the following tools to ensure code quality:

**Black (Code Formatter):**
```bash
# Format all Python files
black .

# Format specific file
black algorithms/manager.py
```

**Flake8 (Linter):**
```bash
# Lint all Python files
flake8 algorithms/ utils/ scripts/

# Lint specific file
flake8 algorithms/manager.py
```

**MyPy (Type Checker):**
```bash
# Type check all Python files
mypy algorithms/ utils/ scripts/

# Type check specific file
mypy algorithms/manager.py
```

### Naming Conventions

- **Classes**: PascalCase (e.g., `ImageEnhancer`)
- **Functions/Methods**: snake_case (e.g., `enhance_image`)
- **Variables**: snake_case (e.g., `upscale_factor`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `DEFAULT_UPSCALE_FACTOR`)
- **Private methods**: _leading_underscore (e.g., `_internal_method`)

### Documentation Style

**Docstrings**: Use Google style docstrings

```python
def enhance(self, image: np.ndarray, upscale_factor: int = 4) -> np.ndarray:
    """Enhance the input image using super-resolution.

    Args:
        image: Input image as numpy array (RGB format)
        upscale_factor: Factor by which to upscale the image

    Returns:
        Enhanced image as numpy array (RGB format)

    Raises:
        ValueError: If image is invalid
        RuntimeError: If model is not loaded
    """
    pass
```

**Comments**: Add comments for complex logic

```python
# Convert image to BGR format for OpenCV processing
img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
```

## Logging

Use the centralized logging system:

```python
from utils.logger import get_logger

logger = get_logger(__name__)

# Log information
logger.info("Processing image: %s", image_path)

# Log warnings
logger.warning("Image is very small, using 16x upscale")

# Log errors
logger.error("Failed to load model: %s", str(e), exc_info=True)
```

**Log Format**: All logs should follow the [ ENHANCEMENT ] format for consistency.

## Submitting Changes

### Pull Request Checklist

Before submitting a pull request, ensure:

- [ ] Code follows project style guidelines
- [ ] All tests pass (`pytest`)
- [ ] Code is formatted (`black .`)
- [ ] No linting errors (`flake8`)
- [ ] Type checking passes (`mypy`)
- [ ] New features are documented
- [ ] Changes are committed with clear messages
- [ ] PR description explains the changes
- [ ] Related issues are referenced

### Commit Message Format

Use conventional commit format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(algorithms): add new super-resolution algorithm

Implement XYZ algorithm with support for GPU acceleration.

Closes #123
```

```
fix(manager): fix parameter passing to algorithms

The enhance() method was not passing upscale_factor correctly,
causing no upscaling in test_runner.

Fixes #456
```

## Reporting Issues

When reporting issues, please include:

1. **Issue Description**: Clear description of the problem
2. **Steps to Reproduce**: How to reproduce the issue
3. **Expected Behavior**: What should happen
4. **Actual Behavior**: What actually happens
5. **Environment Details**:
   - Python version
   - Operating system
   - GPU/CPU details
6. **Logs**: Relevant log files (logs/error.log, logs/enhancement.log)
7. **Images**: Sample images (if applicable)

**Issue Template:**

```markdown
## Issue Description
[Describe the issue]

## Steps to Reproduce
1. [Step 1]
2. [Step 2]
3. [Step 3]

## Expected Behavior
[What should happen]

## Actual Behavior
[What actually happens]

## Environment
- Python version: [e.g., 3.10.12]
- OS: [e.g., Windows 11]
- GPU: [e.g., NVIDIA GTX 1650 Ti]
- CUDA version: [e.g., 11.8]

## Logs
[Attach relevant logs]

## Additional Information
[Any other relevant information]
```

## Known Issues

See `KNOWN_ISSUES.md` for a list of known issues and their workarounds.

## Deployment: HuggingFace Spaces

This project is designed to deploy to **HuggingFace Spaces** for free hosting with GPU support.

### Why HuggingFace Spaces?

| Resource | Render Free | HuggingFace Free | Required |
|----------|-------------|------------------|----------|
| RAM | 512 MB | 16 GB | 1.5 GB |
| Storage | 512 MB | 50 GB | 700 MB |
| GPU | None | A10G/T4 free | Recommended |

### Deploying to HuggingFace Spaces

#### 1. Create a HuggingFace Account
- Go to https://huggingface.co
- Sign up for a free account

#### 2. Create a New Space
- Navigate to https://huggingface.co/new-space
- Choose **Docker** template (for FastAPI)
- Select GPU: **A10G** (free tier) or **T4**
- Give your space a name (e.g., `img-enhance-service`)

#### 3. Deploy Your Application
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

#### 4. Configuration for HuggingFace

Create a `Dockerfile` for your Space:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 7860

# Run the application
CMD ["uvicorn", "web_service.app:app", "--host", "0.0.0.0", "--port", "7860"]
```

Create a `packages.txt` for system dependencies:
```txt
build-essential
```

#### 5. Environment Variables

Set these in your Space settings:
- `MAX_IMAGE_SIZE_MB=50`
- `ENABLE_GPU=True`
- `HF_TOKEN=your_huggingface_token` (if needed)

#### 6. Test Your Deployment

After deployment, your API will be available at:
- `https://<your-username>-img-enhance-service.hf.space/docs` (Swagger UI)
- `https://<your-username>-img-enhance-service.hf.space/api/v1/enhance` (API endpoint)

### Local Development vs HuggingFace

**Local Development:**
```bash
# Run locally for testing
uvicorn web_service.app:app --reload
```

**HuggingFace Spaces:**
- Automatic deployment on push
- GPU available for inference
- Public API access
- Sleep after inactivity (configurable)

### Troubleshooting Deployment

**Memory Issues:**
- Reduce `MAX_IMAGE_SIZE_MB` in settings
- Use CPU-only mode if GPU unavailable
- Implement image tiling for very large images

**Cold Starts:**
- HuggingFace may sleep after inactivity
- First request after sleep may be slow
- Keep-alive pings can prevent sleeping

**GPU Not Available:**
- Check Space GPU settings
- Verify model fits in GPU memory
- Fall back to CPU processing if needed

## Getting Help

If you need help:

1. Check the documentation in `.opencode/docs/`
2. Review `KNOWN_ISSUES.md` for known problems
3. Search existing issues and pull requests
4. Ask questions in issues or discussions

## Code of Conduct

Be respectful and constructive in all interactions. Treat others as you would like to be treated.

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Attribution

This project integrates several open-source algorithms. Please attribute them properly when using or modifying this code:

- **Real-ESRGAN**: https://github.com/xinntao/Real-ESRGAN
- **GFPGAN**: https://github.com/TencentARC/GFPGAN
- **super-image**: https://github.com/eugenesiow/super-image
- **CLAHE**: https://github.com/ThomasWangWeiHong/Low-Light-Image-Enhancement-CLAHE-Based
- **Deep White Balance**: https://github.com/mahmoudnafifi/Deep_White_Balance
- **Exposure Correction**: https://github.com/mahmoudnafifi/Exposure_Correction

---

Thank you for contributing to the Image Enhancement Web Service!
