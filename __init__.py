"""
ComfyUI Color Tools

A comprehensive collection of color manipulation and analysis nodes for ComfyUI workflows.
This package provides advanced color processing capabilities including color space conversions,
color grading, palette extraction, color analysis tools, and color profile reading.
"""

__version__ = "1.0.0"
__author__ = "Pablo Apiolazza"
__email__ = "me@apzmedia.com"
__license__ = "MIT"

# Import color profile reader nodes
try:
    from .color_profile_reader import ColorProfileReader, GammaCompare
except ImportError:
    from color_profile_reader import ColorProfileReader, GammaCompare

# Import all color tools nodes
try:
    from .nodes.color_conversion import (
        ColorSpaceConverter,
        ColorTemperature,
        ColorSpaceAnalyzer,
    )

    from .nodes.color_grading import (
        ColorBalance,
        BrightnessContrast,
        Saturation,
        HueShift,
        GammaCorrection,
    )

    from .nodes.color_analysis import (
        DominantColors,
        ColorHistogram,
        ColorPalette,
        ColorSimilarity,
        ColorHarmony,
    )

    from .nodes.advanced_tools import (
        ColorMatcher,
        ColorQuantizer,
        GamutMapper,
        ColorBlindSim,
    )
except ImportError:
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

# Define the node mapping for ComfyUI
NODE_CLASS_MAPPINGS = {
    # Color Profile Reader Nodes
    "ColorProfileReader": ColorProfileReader,
    "GammaCompare": GammaCompare,
    
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

# Define the display names for the ComfyUI interface
NODE_DISPLAY_NAME_MAPPINGS = {
    # Color Profile Reader Nodes
    "ColorProfileReader": "Color Profile Reader",
    "GammaCompare": "Gamma Compare",
    
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

# Export the mappings for ComfyUI
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]