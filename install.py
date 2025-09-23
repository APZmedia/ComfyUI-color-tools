#!/usr/bin/env python3
"""
Install script for ComfyUI Color Tools

This script handles optional dependency installation and setup.
Runs after pip install -r requirements.txt
"""

import os
import sys
import subprocess
import importlib.util

def check_dependency(package_name, install_name=None):
    """Check if a package is available, return True if available"""
    if install_name is None:
        install_name = package_name
    
    print(f"[ComfyUI Color Tools] Checking for dependency: {package_name}")
    try:
        spec = importlib.util.find_spec(package_name)
        if spec is not None:
            print(f"[ComfyUI Color Tools] ✅ Found {package_name}")
            return True
        else:
            print(f"[ComfyUI Color Tools] ❌ {package_name} not found")
            return False
    except ImportError:
        print(f"[ComfyUI Color Tools] ❌ ImportError when checking for {package_name}")
        return False

def install_optional_dependencies():
    """Install optional dependencies for advanced color tools"""
    optional_deps = [
        ("numpy", "numpy>=1.21.0"),
        ("cv2", "opencv-python>=4.5.0"),
        ("skimage", "scikit-image>=0.18.0"),
        ("matplotlib", "matplotlib>=3.5.0"),
        ("colorspacious", "colorspacious>=1.1.0"),
        ("colour", "colour-science>=0.3.16"),
        ("scipy", "scipy>=1.7.0"),
        ("sklearn", "scikit-learn>=1.0.0"),
        ("PyOpenColorIO", "opencolorio>=2.0.0"),
    ]
    
    missing_deps = []
    for package, install_cmd in optional_deps:
        if not check_dependency(package):
            missing_deps.append(install_cmd)
    
    if missing_deps:
        print(f"[ComfyUI Color Tools] Installing {len(missing_deps)} optional dependencies...")
        if any("opencolorio" in dep for dep in missing_deps):
            print("[ComfyUI Color Tools] 📷 OCIO support will be installed")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install"
            ] + missing_deps)
            print("[ComfyUI Color Tools] ✅ Optional dependencies installed successfully")
            if any("opencolorio" in dep for dep in missing_deps):
                print("[ComfyUI Color Tools] 🎉 OCIO Color Space Converter and related nodes are now available")
        except subprocess.CalledProcessError as e:
            print(f"[ComfyUI Color Tools] ⚠️  Failed to install optional dependencies: {e}")
            print("[ComfyUI Color Tools] 💡 You can install them manually later if needed")
            if any("opencolorio" in dep for dep in missing_deps):
                print("[ComfyUI Color Tools] 📷 Note: OCIO nodes will not be available without PyOpenColorIO")
    else:
        print("[ComfyUI Color Tools] ✅ All optional dependencies already available")

def main():
    """Main installation routine"""
    print("[ComfyUI Color Tools] Running post-install setup...")
    
    # Check if we're in a ComfyUI environment by checking for the root main.py
    comfy_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if not os.path.isfile(os.path.join(comfy_root_dir, "main.py")):
        print(f"[ComfyUI Color Tools] ⚠️  Could not find ComfyUI root at '{comfy_root_dir}', skipping optional dependency installation.")
        return
    
    # Install optional dependencies
    install_optional_dependencies()
    
    # Verify core dependencies
    try:
        import PIL
        print(f"[ComfyUI Color Tools] ✅ Core dependency Pillow {PIL.__version__} available")
    except ImportError:
        print("[ComfyUI Color Tools] ❌ Core dependency Pillow not found!")
        return
    
    # Check for ComfyUI
    try:
        import comfy
        print("[ComfyUI Color Tools] ✅ ComfyUI environment detected")
    except ImportError:
        print("[ComfyUI Color Tools] ⚠️  ComfyUI not detected in current environment")
    
    print("[ComfyUI Color Tools] 🎉 Installation completed!")
    print("[ComfyUI Color Tools] 💡 Restart ComfyUI to load the nodes")

if __name__ == "__main__":
    main()
