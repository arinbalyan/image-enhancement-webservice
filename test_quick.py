#!/usr/bin/env python
"""Quick test script to verify web service imports and basic functionality."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test all critical imports."""
    print("Testing imports...")
    
    try:
        from config.settings import get_settings
        print("  [OK] config.settings")
    except Exception as e:
        print(f"  [FAIL] config.settings: {e}")
        return False
    
    try:
        from utils.logger import get_logger
        print("  [OK] utils.logger")
    except Exception as e:
        print(f"  [FAIL] utils.logger: {e}")
        return False
    
    try:
        from utils.io import load_image, save_image
        print("  [OK] utils.io")
    except Exception as e:
        print(f"  [FAIL] utils.io: {e}")
        return False
    
    try:
        from utils.image_analyzer import ImageAnalyzer
        print("  [OK] utils.image_analyzer")
    except Exception as e:
        print(f"  [FAIL] utils.image_analyzer: {e}")
        return False
    
    try:
        from algorithms.manager import AlgorithmManager
        print("  [OK] algorithms.manager")
    except Exception as e:
        print(f"  [FAIL] algorithms.manager: {e}")
        return False
    
    try:
        from web_service.api.routes import router
        print("  [OK] web_service.api.routes")
        print(f"      Endpoints: {[r.path for r in router.routes]}")
    except Exception as e:
        print(f"  [FAIL] web_service.api.routes: {e}")
        return False
    
    try:
        from web_service.app import app
        print("  [OK] web_service.app")
    except Exception as e:
        print(f"  [FAIL] web_service.app: {e}")
        return False
    
    return True


def test_algorithm_manager():
    """Test algorithm manager initialization."""
    print("\nTesting AlgorithmManager...")
    
    try:
        from algorithms.manager import AlgorithmManager
        manager = AlgorithmManager()
        print("  [OK] AlgorithmManager initialized")
        
        available = manager.get_available_algorithms()
        print(f"  [OK] Available algorithms: {available}")
        
        status = manager.get_status()
        print(f"  [OK] Status: device={status['device']}")
        
        return True
    except Exception as e:
        print(f"  [FAIL] AlgorithmManager: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fastapi_app():
    """Test FastAPI app creation."""
    print("\nTesting FastAPI app...")
    
    try:
        from web_service.app import app
        print(f"  [OK] App title: {app.title}")
        print(f"  [OK] App version: {app.version}")
        
        routes = [route.path for route in app.routes]
        print(f"  [OK] Routes: {routes}")
        
        return True
    except Exception as e:
        print(f"  [FAIL] FastAPI app: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Image Enhancement Web Service - Quick Test")
    print("=" * 60)
    
    all_passed = True
    
    if not test_imports():
        all_passed = False
    
    if not test_algorithm_manager():
        all_passed = False
    
    if not test_fastapi_app():
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("=" * 60)
    
    sys.exit(0 if all_passed else 1)
