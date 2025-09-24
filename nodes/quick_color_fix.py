"""
Quick Color Space Fix

Quick color space fix for common Photoshop misalignments.
Simplified interface for common cases.
"""

import torch
import numpy as np
import json
from typing import Tuple
from .littlecms_converter import LittleCMSColorProfileConverter


class QuickColorSpaceFix:
    """
    Quick color space fix for common Photoshop misalignments.
    Simplified interface for common cases.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_mode": (["file", "tensor"], {"default": "tensor"}),
                "fix_type": (["Adobe RGB to sRGB", "sRGB to Adobe RGB", "Linearize sRGB", "sRGB to Linear"], 
                           {"default": "Adobe RGB to sRGB"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image_path": ("STRING", {"default": "", "multiline": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("fixed_image", "fix_info")
    FUNCTION = "fix_color_space"
    CATEGORY = "Color Tools/Quick Fix"
    
    def fix_color_space(self, input_mode: str, fix_type: str, 
                       image: torch.Tensor = None, image_path: str = "") -> Tuple[torch.Tensor, str]:
        """
        Quick fix for common color space issues.
        """
        # Map fix types to profile conversions
        fix_mapping = {
            "Adobe RGB to sRGB": ("Adobe RGB", "sRGB", "Relative Colorimetric"),
            "sRGB to Adobe RGB": ("sRGB", "Adobe RGB", "Relative Colorimetric"),
            "Linearize sRGB": ("sRGB", "Linear sRGB", "Relative Colorimetric"),
            "sRGB to Linear": ("sRGB", "Linear sRGB", "Relative Colorimetric"),
        }
        
        source_profile, target_profile, rendering_intent = fix_mapping[fix_type]
        
        # Use the main converter
        converter = LittleCMSColorProfileConverter()
        converted_image, conversion_info, profile_info = converter.convert_color_profile(
            input_mode, source_profile, target_profile, rendering_intent, True,
            image, image_path
        )
        
        # Create fix info
        fix_info = {
            "fix_type": fix_type,
            "description": self._get_fix_description(fix_type),
            "conversion_info": json.loads(conversion_info)
        }
        
        return converted_image, json.dumps(fix_info, indent=2)
    
    def _get_fix_description(self, fix_type: str) -> str:
        """Get description for fix type"""
        descriptions = {
            "Adobe RGB to sRGB": "Fixes Adobe RGB images that appear oversaturated when displayed as sRGB",
            "sRGB to Adobe RGB": "Converts sRGB images to Adobe RGB for wide gamut workflows",
            "Linearize sRGB": "Converts sRGB to linear space for compositing and HDR workflows",
            "sRGB to Linear": "Converts sRGB to linear space for compositing and HDR workflows",
        }
        return descriptions.get(fix_type, "Color space conversion")


# ComfyUI node registry
NODE_CLASS_MAPPINGS = {
    "QuickColorSpaceFix": QuickColorSpaceFix
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "QuickColorSpaceFix": "Quick Color Space Fix"
}
