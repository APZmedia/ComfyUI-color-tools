"""
ComfyUI Color Tools

A comprehensive collection of color manipulation and analysis nodes for ComfyUI workflows.
This package provides advanced color processing capabilities including color space conversions,
color grading, palette extraction, color analysis tools, and color profile reading.
"""

import os
import subprocess
import sys

# Run the installation script before trying to import any nodes
install_script_path = os.path.join(os.path.dirname(__file__), "install.py")
try:
    print("[ComfyUI Color Tools]  initiator: Running installation script...")
    subprocess.check_call([sys.executable, install_script_path])
except (subprocess.CalledProcessError, FileNotFoundError) as e:
    print(f"[ComfyUI Color Tools] ⚠️  Initiator: Failed to run install script: {e}")

# Node imports
try:
    # File-only nodes (profile reading)
    from .nodes.color_profile_reader import ColorProfileReader, GammaCompare
    from .nodes.color_profile_convert_simple import ColorProfileConvert
    from .nodes.color_converter_advanced import ColorConverterAdvanced
    
    # LittleCMS color profile conversion nodes
    from .nodes.littlecms_converter import LittleCMSColorProfileConverter
    from .nodes.quick_color_fix import QuickColorSpaceFix
    
    # Dual input conversion nodes
    from .nodes.color_conversion import ColorSpaceConverter, ColorTemperature, ColorSpaceAnalyzer
    
    # Dual input grading nodes
    from .nodes.color_grading import ColorBalance, BrightnessContrast, Saturation, HueShift, GammaCorrection
    
    # Dual input analysis nodes
    from .nodes.color_analysis import DominantColors, ColorHistogram, ColorPalette, ColorSimilarity, ColorHarmony
    
    # Vector scope node
    from .nodes.vector_scope import VectorScopeNode
    
    # OCIO nodes
    from .nodes.ocio_tools import OCIOColorSpaceConverter, OCIOConfigInfo, TestPatternGenerator
    from .nodes.ocio_advanced import AdvancedOcioColorTransform

    NODE_CLASS_MAPPINGS = {
        # File-only nodes (profile reading)
        "ColorProfileReader": ColorProfileReader,
        "GammaCompare": GammaCompare,
        "ColorProfileConvert": ColorProfileConvert,
        "ColorConverterAdvanced": ColorConverterAdvanced,
        
        # LittleCMS color profile conversion nodes
        "LittleCMSColorProfileConverter": LittleCMSColorProfileConverter,
        "QuickColorSpaceFix": QuickColorSpaceFix,
        
        # Dual input conversion nodes
        "ColorSpaceConverter": ColorSpaceConverter,
        "ColorTemperature": ColorTemperature,
        "ColorSpaceAnalyzer": ColorSpaceAnalyzer,
        
        # Dual input grading nodes
        "ColorBalance": ColorBalance,
        "BrightnessContrast": BrightnessContrast,
        "Saturation": Saturation,
        "HueShift": HueShift,
        "GammaCorrection": GammaCorrection,
        
        # Dual input analysis nodes
        "DominantColors": DominantColors,
        "ColorHistogram": ColorHistogram,
        "ColorPalette": ColorPalette,
        "ColorSimilarity": ColorSimilarity,
        "ColorHarmony": ColorHarmony,
        
        # Vector scope node
        "VectorScope": VectorScopeNode,
        
        # OCIO nodes
        "OCIOColorSpaceConverter": OCIOColorSpaceConverter,
        "OCIOConfigInfo": OCIOConfigInfo,
        "TestPatternGenerator": TestPatternGenerator,
        "AdvancedOcioColorTransform": AdvancedOcioColorTransform,
    }
except ImportError as e:
    print(f"[ComfyUI Color Tools] ❌ Failed to import nodes: {e}")
    print("[ComfyUI Color Tools] 💡 This can happen if dependencies are missing. Please check the console for installation errors.")
    NODE_CLASS_MAPPINGS = {}

# Display names for all potential nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    # File-only nodes (profile reading)
    "ColorProfileReader": "Read Image Color Profile",
    "GammaCompare": "Compare Image Gamma Values",
    "ColorProfileConvert": "Convert Image Color Space",
    "ColorConverterAdvanced": "Advanced Color Converter",
    
    # LittleCMS color profile conversion nodes
    "LittleCMSColorProfileConverter": "LittleCMS Color Profile Converter",
    "QuickColorSpaceFix": "Quick Color Space Fix",
    
    # Dual input conversion nodes
    "ColorSpaceConverter": "Color Space Converter",
    "ColorTemperature": "Color Temperature",
    "ColorSpaceAnalyzer": "Color Space Analyzer",
    
    # Dual input grading nodes
    "ColorBalance": "Color Balance",
    "BrightnessContrast": "Brightness/Contrast",
    "Saturation": "Saturation",
    "HueShift": "Hue Shift",
    "GammaCorrection": "Gamma Correction",
    
    # Dual input analysis nodes
    "DominantColors": "Dominant Colors",
    "ColorHistogram": "Color Histogram",
    "ColorPalette": "Color Palette",
    "ColorSimilarity": "Color Similarity",
    "ColorHarmony": "Color Harmony",
    
    # Vector scope node
    "VectorScope": "Vector Scope",
    
    # OCIO nodes
    "OCIOColorSpaceConverter": "OCIO Color Space Converter",
    "OCIOConfigInfo": "OCIO Config Info",
    "TestPatternGenerator": "Test Pattern Generator",
    "AdvancedOcioColorTransform": "Advanced OCIO Color Transform",
}

# Filter display names to only those that were successfully loaded
NODE_DISPLAY_NAME_MAPPINGS = {
    key: value for key, value in NODE_DISPLAY_NAME_MAPPINGS.items() if key in NODE_CLASS_MAPPINGS
}

print(f"[ComfyUI Color Tools] --- Registration ---")
print(f"[ComfyUI Color Tools] ✅ Registered {len(NODE_CLASS_MAPPINGS)} nodes.")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
