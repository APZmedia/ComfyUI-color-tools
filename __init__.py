"""
ComfyUI Color Tools

A comprehensive collection of color manipulation and analysis nodes for ComfyUI workflows.
This package provides advanced color processing capabilities including color space conversions,
color grading, palette extraction, color analysis tools, and color profile reading.
"""

import sys
import traceback

__version__ = "1.0.0"
__author__ = "Pablo Apiolazza"
__email__ = "me@apzmedia.com"
__license__ = "MIT"

# Console logging for debugging
print(f"[ComfyUI Color Tools] Loading version {__version__} by {__author__}")
print("[ComfyUI Color Tools] Starting node registration...")

# Import color profile reader nodes
try:
    print("[ComfyUI Color Tools] Importing color profile reader nodes...")
    from .nodes.color_profile_reader import ColorProfileReader, GammaCompare
    print("[ComfyUI Color Tools] ✅ Color Profile Reader nodes imported successfully")
except ImportError as e:
    print(f"[ComfyUI Color Tools] ⚠️  Relative import failed, trying absolute import: {e}")
    try:
        from nodes.color_profile_reader import ColorProfileReader, GammaCompare
        print("[ComfyUI Color Tools] ✅ Color Profile Reader nodes imported via absolute import")
    except ImportError as e2:
        print(f"[ComfyUI Color Tools] ❌ Failed to import Color Profile Reader nodes: {e2}")
        raise e2

# Import color profile convert node
try:
    print("[ComfyUI Color Tools] Importing color profile convert node...")
    from .nodes.color_profile_convert import ColorProfileConvert
    print("[ComfyUI Color Tools] ✅ Color Profile Convert node imported successfully")
except ImportError as e:
    print(f"[ComfyUI Color Tools] ⚠️  Relative import failed, trying absolute import: {e}")
    try:
        from nodes.color_profile_convert import ColorProfileConvert
        print("[ComfyUI Color Tools] ✅ Color Profile Convert node imported via absolute import")
    except ImportError as e2:
        print(f"[ComfyUI Color Tools] ❌ Failed to import Color Profile Convert node: {e2}")
        raise e2

# Import all color tools nodes
try:
    print("[ComfyUI Color Tools] Importing color tools nodes...")
    
    print("[ComfyUI Color Tools] Importing color conversion nodes...")
    from .nodes.color_conversion import (
        ColorSpaceConverter,
        ColorTemperature,
        ColorSpaceAnalyzer,
    )

    print("[ComfyUI Color Tools] Importing color grading nodes...")
    from .nodes.color_grading import (
        ColorBalance,
        BrightnessContrast,
        Saturation,
        HueShift,
        GammaCorrection,
    )

    print("[ComfyUI Color Tools] Importing color analysis nodes...")
    from .nodes.color_analysis import (
        DominantColors,
        ColorHistogram,
        ColorPalette,
        ColorSimilarity,
        ColorHarmony,
    )

    print("[ComfyUI Color Tools] Importing advanced tools nodes...")
    from .nodes.advanced_tools import (
        ColorMatcher,
        ColorQuantizer,
        GamutMapper,
        ColorBlindSim,
    )
    
    print("[ComfyUI Color Tools] ✅ All color tools nodes imported successfully")
    
except ImportError as e:
    print(f"[ComfyUI Color Tools] ⚠️  Relative import failed, trying absolute import: {e}")
    try:
        print("[ComfyUI Color Tools] Retrying with absolute imports...")
        from nodes.color_conversion import (
            ColorSpaceConverter,
            ColorTemperature,
            ColorSpaceAnalyzer,
        )

        from nodes.color_grading import (
            ColorBalance,
            BrightnessContrast,
            Saturation,
            HueShift,
            GammaCorrection,
        )

        from nodes.color_analysis import (
            DominantColors,
            ColorHistogram,
            ColorPalette,
            ColorSimilarity,
            ColorHarmony,
        )

        from nodes.advanced_tools import (
            ColorMatcher,
            ColorQuantizer,
            GamutMapper,
            ColorBlindSim,
        )
        print("[ComfyUI Color Tools] ✅ All color tools nodes imported via absolute import")
    except ImportError as e2:
        print(f"[ComfyUI Color Tools] ❌ Failed to import color tools nodes: {e2}")
        print(f"[ComfyUI Color Tools] Traceback: {traceback.format_exc()}")
        raise e2

# Define the node mapping for ComfyUI
print("[ComfyUI Color Tools] Registering nodes with ComfyUI...")

NODE_CLASS_MAPPINGS = {
    # Color Profile Reader Nodes
    "ColorProfileReader": ColorProfileReader,
    "GammaCompare": GammaCompare,
    "ColorProfileConvert": ColorProfileConvert,
    
    # Color Conversion Nodes
    "ColorSpaceConverter": ColorSpaceConverter,
    "ColorTemperature": ColorTemperature,
    "ColorSpaceAnalyzer": ColorSpaceAnalyzer,
    
    # Color Grading Nodes
    "ColorBalance": ColorBalance,
    "BrightnessContrast": BrightnessContrast,
    "Saturation": Saturation,
    "HueShift": HueShift,
    "GammaCorrection": GammaCorrection,
    
    # Color Analysis Nodes
    "DominantColors": DominantColors,
    "ColorHistogram": ColorHistogram,
    "ColorPalette": ColorPalette,
    "ColorSimilarity": ColorSimilarity,
    "ColorHarmony": ColorHarmony,
    
    # Advanced Tools Nodes
    "ColorMatcher": ColorMatcher,
    "ColorQuantizer": ColorQuantizer,
    "GamutMapper": GamutMapper,
    "ColorBlindSim": ColorBlindSim,
}

print(f"[ComfyUI Color Tools] ✅ Registered {len(NODE_CLASS_MAPPINGS)} nodes:")
for node_name in NODE_CLASS_MAPPINGS.keys():
    print(f"[ComfyUI Color Tools]   - {node_name}")

# Define the display names for the ComfyUI interface
NODE_DISPLAY_NAME_MAPPINGS = {
    # Color Profile Reader Nodes
    "ColorProfileReader": "Color Profile Reader",
    "GammaCompare": "Gamma Compare",
    "ColorProfileConvert": "Color Profile → sRGB / Linear",
    
    # Color Conversion Nodes
    "ColorSpaceConverter": "Color Space Converter",
    "ColorTemperature": "Color Temperature",
    "ColorSpaceAnalyzer": "Color Space Analyzer",
    
    # Color Grading Nodes
    "ColorBalance": "Color Balance",
    "BrightnessContrast": "Brightness/Contrast",
    "Saturation": "Saturation",
    "HueShift": "Hue Shift",
    "GammaCorrection": "Gamma Correction",
    
    # Color Analysis Nodes
    "DominantColors": "Dominant Colors",
    "ColorHistogram": "Color Histogram",
    "ColorPalette": "Color Palette",
    "ColorSimilarity": "Color Similarity",
    "ColorHarmony": "Color Harmony",
    
    # Advanced Tools Nodes
    "ColorMatcher": "Color Matcher",
    "ColorQuantizer": "Color Quantizer",
    "GamutMapper": "Gamut Mapper",
    "ColorBlindSim": "Color Blind Simulator",
}

print(f"[ComfyUI Color Tools] ✅ Display names configured for {len(NODE_DISPLAY_NAME_MAPPINGS)} nodes")
print("[ComfyUI Color Tools] 🎉 Node registration completed successfully!")
print("=" * 60)

# Export the mappings for ComfyUI
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]