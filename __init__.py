"""
ComfyUI Color Tools

A comprehensive collection of color manipulation and analysis nodes for ComfyUI workflows.
This package provides advanced color processing capabilities including color space conversions,
color grading, palette extraction, color analysis tools, and color profile reading.
"""

# Import only the core nodes that have minimal dependencies
try:
    # Try relative import first (works in ComfyUI)
    from .nodes.color_profile_reader import ColorProfileReader, GammaCompare
    from .nodes.color_profile_convert_simple import ColorProfileConvert
except (ImportError, ValueError):
    # Fallback to absolute import (works when running directly)
    from nodes.color_profile_reader import ColorProfileReader, GammaCompare
    from nodes.color_profile_convert_simple import ColorProfileConvert

# Core nodes (always available - minimal dependencies)
NODE_CLASS_MAPPINGS = {
    "ColorProfileReader": ColorProfileReader,
    "GammaCompare": GammaCompare,
    "ColorProfileConvert": ColorProfileConvert,
}

# Display names for the ComfyUI interface
NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorProfileReader": "Color Profile Reader",
    "GammaCompare": "Gamma Compare", 
    "ColorProfileConvert": "Color Profile → sRGB / Linear",
}

# Export the mappings for ComfyUI
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]