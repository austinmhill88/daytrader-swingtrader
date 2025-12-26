"""
System Setup Verification Script
Checks that all dependencies and configurations are properly installed.
"""
import sys
from pathlib import Path
from loguru import logger

def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor}.{version.micro} (requires 3.10+)")
        return False

def check_package(package_name, import_name=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"  ✓ {package_name}")
        return True
    except ImportError:
        print(f"  ✗ {package_name} (missing)")
        return False

def check_dependencies():
    """Check all required dependencies."""
    print("\nChecking core dependencies...")
    core = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("scikit-learn", "sklearn"),
        ("FastAPI", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("httpx", "httpx"),
        ("PyYAML", "yaml"),
        ("loguru", "loguru"),
    ]
    
    results = [check_package(pkg, imp) for pkg, imp in core]
    
    print("\nChecking AI dependencies...")
    ai_deps = [
        ("llama-cpp-python", "llama_cpp"),
    ]
    
    ai_results = [check_package(pkg, imp) for pkg, imp in ai_deps]
    
    print("\nChecking GUI dependencies...")
    gui_deps = [
        ("PyQt6", "PyQt6"),
        ("pyqtgraph", "pyqtgraph"),
        ("matplotlib", "matplotlib"),
    ]
    
    gui_results = [check_package(pkg, imp) for pkg, imp in gui_deps]
    
    return all(results), all(ai_results), all(gui_results)

def check_cuda():
    """Check CUDA availability."""
    print("\nChecking CUDA/GPU support...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"    CUDA version: {torch.version.cuda}")
            return True
        else:
            print("  ⚠ CUDA not available (CPU mode)")
            return False
    except ImportError:
        print("  ℹ PyTorch not installed (optional for CUDA check)")
        # Try checking with llama-cpp
        try:
            from llama_cpp import Llama
            print("  ✓ llama-cpp-python installed (GPU support depends on build)")
        except ImportError:
            pass
        return None

def check_config_files():
    """Check configuration files exist."""
    print("\nChecking configuration files...")
    
    configs = [
        "config/config.yaml",
        "config/ai-coder.yaml",
    ]
    
    results = []
    for config in configs:
        path = Path(config)
        if path.exists():
            print(f"  ✓ {config}")
            results.append(True)
        else:
            print(f"  ✗ {config} (missing)")
            results.append(False)
    
    return all(results)

def check_directories():
    """Check required directories exist."""
    print("\nChecking directory structure...")
    
    dirs = [
        "server",
        "gui",
        "src",
        "config",
        "logs",
        "data",
    ]
    
    results = []
    for dir_name in dirs:
        path = Path(dir_name)
        if path.exists() and path.is_dir():
            print(f"  ✓ {dir_name}/")
            results.append(True)
        else:
            print(f"  ✗ {dir_name}/ (missing)")
            results.append(False)
    
    return all(results)

def check_environment():
    """Check environment variables."""
    print("\nChecking environment variables...")
    import os
    
    vars_to_check = [
        ("APCA_API_KEY_ID", False, "Alpaca API key"),
        ("APCA_API_SECRET_KEY", False, "Alpaca secret key"),
        ("MODELS_DIR", True, "AI models directory"),
        ("DAYTRADER_ROOT", True, "Trading app root"),
    ]
    
    for var_name, optional, description in vars_to_check:
        value = os.environ.get(var_name)
        if value:
            print(f"  ✓ {var_name} ({description})")
        elif optional:
            print(f"  ⚠ {var_name} ({description}) - optional, not set")
        else:
            print(f"  ✗ {var_name} ({description}) - required, not set")

def main():
    """Main verification routine."""
    print("=" * 60)
    print("Trading System Setup Verification")
    print("=" * 60)
    
    # Check Python version
    py_ok = check_python_version()
    
    # Check dependencies
    core_ok, ai_ok, gui_ok = check_dependencies()
    
    # Check CUDA
    cuda_ok = check_cuda()
    
    # Check configs
    config_ok = check_config_files()
    
    # Check directories
    dirs_ok = check_directories()
    
    # Check environment
    check_environment()
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    if not py_ok:
        print("✗ Python version: FAILED (upgrade to 3.10+)")
    else:
        print("✓ Python version: OK")
    
    if not core_ok:
        print("✗ Core dependencies: FAILED (run: pip install -r requirements.txt)")
    else:
        print("✓ Core dependencies: OK")
    
    if not ai_ok:
        print("⚠ AI dependencies: INCOMPLETE")
        print("  Install: pip install llama-cpp-python")
        print("  For GPU: pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121")
    else:
        print("✓ AI dependencies: OK")
    
    if not gui_ok:
        print("⚠ GUI dependencies: INCOMPLETE (run: pip install PyQt6 pyqtgraph matplotlib)")
    else:
        print("✓ GUI dependencies: OK")
    
    if cuda_ok is True:
        print("✓ GPU support: Available")
    elif cuda_ok is False:
        print("⚠ GPU support: Not available (will use CPU)")
    else:
        print("ℹ GPU support: Unknown (install PyTorch to verify)")
    
    if not config_ok:
        print("✗ Configuration: INCOMPLETE (create missing config files)")
    else:
        print("✓ Configuration: OK")
    
    if not dirs_ok:
        print("✗ Directory structure: INCOMPLETE")
    else:
        print("✓ Directory structure: OK")
    
    print("\n" + "=" * 60)
    
    if py_ok and core_ok and config_ok and dirs_ok:
        print("✓ Core system ready!")
        if ai_ok:
            print("✓ AI features ready!")
        else:
            print("⚠ AI features incomplete (optional)")
        if gui_ok:
            print("✓ GUI ready!")
        else:
            print("⚠ GUI incomplete (optional)")
    else:
        print("✗ System not ready - fix issues above")
        return 1
    
    print("\nNext steps:")
    print("1. Configure .env with Alpaca credentials")
    print("2. Download AI models (optional)")
    print("3. Run: python launcher.py --paper")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
