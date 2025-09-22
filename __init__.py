"""
ComfyUI Color Profile Reader

A ComfyUI custom node for reading color profiles and color space information from image files.
Supports ICC profiles, PNG color space chunks, and various image formats.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"
__license__ = "MIT"

# Import the node class
from .color_profile_reader import ColorProfileReader

# Define the node mapping for ComfyUI
NODE_CLASS_MAPPINGS = {
    "ColorProfileReader": ColorProfileReader,
}

# Define the display names for the ComfyUI interface
NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorProfileReader": "Color Profile Reader",
}

# Export the mappings for ComfyUI
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]