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
        # Try importing from the nodes package
        try:
            from .nodes import ColorProfileReader, GammaCompare
            print("[ComfyUI Color Tools] ✅ Color Profile Reader nodes imported via nodes package")
        except ImportError as e3:
            print(f"[ComfyUI Color Tools] ❌ All import methods failed: {e3}")
            raise e3

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
        # Try importing from the nodes package
        try:
            from .nodes import ColorProfileConvert
            print("[ComfyUI Color Tools] ✅ Color Profile Convert node imported via nodes package")
        except ImportError as e3:
            print(f"[ComfyUI Color Tools] ❌ All import methods failed: {e3}")
            raise e3

# Import all color tools nodes (with dependency handling)
color_tools_nodes = {}
missing_dependencies = []

try:
    print("[ComfyUI Color Tools] Importing color tools nodes...")
    
    # Try to import color conversion nodes
    try:
        print("[ComfyUI Color Tools] Importing color conversion nodes...")
        from .nodes.color_conversion import (
            ColorSpaceConverter,
            ColorTemperature,
            ColorSpaceAnalyzer,
        )
        color_tools_nodes.update({
            "ColorSpaceConverter": ColorSpaceConverter,
            "ColorTemperature": ColorTemperature,
            "ColorSpaceAnalyzer": ColorSpaceAnalyzer,
        })
        print("[ComfyUI Color Tools] ✅ Color conversion nodes imported")
    except ImportError as e:
        print(f"[ComfyUI Color Tools] ⚠️  Color conversion nodes require additional dependencies: {e}")
        missing_dependencies.append("color_conversion")

    # Try to import color grading nodes
    try:
        print("[ComfyUI Color Tools] Importing color grading nodes...")
        from .nodes.color_grading import (
            ColorBalance,
            BrightnessContrast,
            Saturation,
            HueShift,
            GammaCorrection,
        )
        color_tools_nodes.update({
            "ColorBalance": ColorBalance,
            "BrightnessContrast": BrightnessContrast,
            "Saturation": Saturation,
            "HueShift": HueShift,
            "GammaCorrection": GammaCorrection,
        })
        print("[ComfyUI Color Tools] ✅ Color grading nodes imported")
    except ImportError as e:
        print(f"[ComfyUI Color Tools] ⚠️  Color grading nodes require additional dependencies: {e}")
        missing_dependencies.append("color_grading")

    # Try to import color analysis nodes
    try:
        print("[ComfyUI Color Tools] Importing color analysis nodes...")
        from .nodes.color_analysis import (
            DominantColors,
            ColorHistogram,
            ColorPalette,
            ColorSimilarity,
            ColorHarmony,
        )
        color_tools_nodes.update({
            "DominantColors": DominantColors,
            "ColorHistogram": ColorHistogram,
            "ColorPalette": ColorPalette,
            "ColorSimilarity": ColorSimilarity,
            "ColorHarmony": ColorHarmony,
        })
        print("[ComfyUI Color Tools] ✅ Color analysis nodes imported")
    except ImportError as e:
        print(f"[ComfyUI Color Tools] ⚠️  Color analysis nodes require additional dependencies: {e}")
        missing_dependencies.append("color_analysis")

    # Try to import advanced tools nodes
    try:
        print("[ComfyUI Color Tools] Importing advanced tools nodes...")
        from .nodes.advanced_tools import (
            ColorMatcher,
            ColorQuantizer,
            GamutMapper,
            ColorBlindSim,
        )
        color_tools_nodes.update({
            "ColorMatcher": ColorMatcher,
            "ColorQuantizer": ColorQuantizer,
            "GamutMapper": GamutMapper,
            "ColorBlindSim": ColorBlindSim,
        })
        print("[ComfyUI Color Tools] ✅ Advanced tools nodes imported")
    except ImportError as e:
        print(f"[ComfyUI Color Tools] ⚠️  Advanced tools nodes require additional dependencies: {e}")
        missing_dependencies.append("advanced_tools")
    
    if missing_dependencies:
        print(f"[ComfyUI Color Tools] ⚠️  Some nodes require additional dependencies: {missing_dependencies}")
        print("[ComfyUI Color Tools] 💡 Install missing dependencies with: pip install -r requirements.txt")
    else:
        print("[ComfyUI Color Tools] ✅ All color tools nodes imported successfully")
    
except Exception as e:
    print(f"[ComfyUI Color Tools] ❌ Unexpected error importing color tools: {e}")
    print(f"[ComfyUI Color Tools] Traceback: {traceback.format_exc()}")

# Define the node mapping for ComfyUI
print("[ComfyUI Color Tools] Registering nodes with ComfyUI...")

# Build the node mappings dynamically
NODE_CLASS_MAPPINGS = {
    # Color Profile Reader Nodes (always available)
    "ColorProfileReader": ColorProfileReader,
    "GammaCompare": GammaCompare,
    "ColorProfileConvert": ColorProfileConvert,
}

# Add color tools nodes if they were successfully imported
NODE_CLASS_MAPPINGS.update(color_tools_nodes)

print(f"[ComfyUI Color Tools] ✅ Registered {len(NODE_CLASS_MAPPINGS)} nodes:")
for node_name in NODE_CLASS_MAPPINGS.keys():
    print(f"[ComfyUI Color Tools]   - {node_name}")

# Define the display names for the ComfyUI interface
# Build display name mappings dynamically
NODE_DISPLAY_NAME_MAPPINGS = {
    # Color Profile Reader Nodes (always available)
    "ColorProfileReader": "Color Profile Reader",
    "GammaCompare": "Gamma Compare",
    "ColorProfileConvert": "Color Profile → sRGB / Linear",
}

# Add display names for color tools nodes if they were successfully imported
display_names = {
    "ColorSpaceConverter": "Color Space Converter",
    "ColorTemperature": "Color Temperature",
    "ColorSpaceAnalyzer": "Color Space Analyzer",
    "ColorBalance": "Color Balance",
    "BrightnessContrast": "Brightness/Contrast",
    "Saturation": "Saturation",
    "HueShift": "Hue Shift",
    "GammaCorrection": "Gamma Correction",
    "DominantColors": "Dominant Colors",
    "ColorHistogram": "Color Histogram",
    "ColorPalette": "Color Palette",
    "ColorSimilarity": "Color Similarity",
    "ColorHarmony": "Color Harmony",
    "ColorMatcher": "Color Matcher",
    "ColorQuantizer": "Color Quantizer",
    "GamutMapper": "Gamut Mapper",
    "ColorBlindSim": "Color Blind Simulator",
}

# Only add display names for nodes that were successfully imported
for node_name in color_tools_nodes.keys():
    if node_name in display_names:
        NODE_DISPLAY_NAME_MAPPINGS[node_name] = display_names[node_name]

print(f"[ComfyUI Color Tools] ✅ Display names configured for {len(NODE_DISPLAY_NAME_MAPPINGS)} nodes")
print("[ComfyUI Color Tools] 🎉 Node registration completed successfully!")
print("=" * 60)

# Export the mappings for ComfyUI
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]