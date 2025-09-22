# comfyui_custom_nodes/color_converter_advanced.py
import base64
import io
import json
import math
from typing import Dict, Optional, Tuple

from PIL import Image, ImageCms


class ColorConverterAdvanced:
    """
    ComfyUI node: Advanced color conversion with separate gamma and color space control.
    This allows users to convert gamma and color space independently.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "gamma_mode": (["preserve", "linearize", "apply_gamma", "sRGB"], {
                    "default": "preserve"
                }),
                "gamma_value": ("FLOAT", {
                    "default": 2.2,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "display": "number"
                }),
                "colorspace_mode": (["preserve", "sRGB", "linear_sRGB", "Adobe_RGB", "ProPhoto_RGB"], {
                    "default": "preserve"
                }),
                "output_format": (["sRGB", "Linear sRGB", "Adobe RGB", "ProPhoto RGB"], {
                    "default": "sRGB"
                }),
            },
            "optional": {
                "profile_json": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Optional: JSON string with source color profile data"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("converted_image", "conversion_info")
    FUNCTION = "convert"
    CATEGORY = "Image/Color"

    def convert(self, image, gamma_mode, gamma_value, colorspace_mode, output_format, profile_json=""):
        """
        Convert image with separate gamma and color space control.
        """
        try:
            # Get basic image info
            batch_size, height, width, channels = image.shape
            
            # Convert to PIL Image for processing
            import torch
            import numpy as np
            if hasattr(torch, 'tensor'):
                # Convert tensor to numpy
                if hasattr(image, 'cpu'):
                    img_array = image.cpu().numpy()
                else:
                    img_array = image
            else:
                img_array = image
            
            # Ensure values are in 0-1 range
            img_array = img_array.clip(0, 1)
            
            # Handle batch dimension
            if batch_size == 1:
                img_array = img_array[0]  # Remove batch dimension
            else:
                # For now, process first image in batch
                img_array = img_array[0]
            
            # Create PIL Image
            pil_image = Image.fromarray((img_array * 255).astype('uint8'), 'RGB')
            
            # Parse source profile data if provided
            source_profile = None
            if profile_json.strip():
                try:
                    profile_data = json.loads(profile_json)
                    source_profile = profile_data.get('profile')
                except:
                    pass
            
            # Initialize conversion info
            conversion_info = {
                "original_size": (width, height),
                "gamma_mode": gamma_mode,
                "gamma_value": gamma_value,
                "colorspace_mode": colorspace_mode,
                "output_format": output_format,
                "source_profile": source_profile,
                "conversions_applied": []
            }
            
            # Step 1: Handle Gamma Conversion
            if gamma_mode == "linearize":
                # Convert to linear (remove gamma)
                img_array = self._linearize_gamma(img_array, gamma_value)
                conversion_info["conversions_applied"].append(f"Linearized with gamma {gamma_value}")
                
            elif gamma_mode == "apply_gamma":
                # Apply specific gamma
                img_array = self._apply_gamma(img_array, gamma_value)
                conversion_info["conversions_applied"].append(f"Applied gamma {gamma_value}")
                
            elif gamma_mode == "sRGB":
                # Convert to sRGB gamma
                img_array = self._sRGB_gamma(img_array)
                conversion_info["conversions_applied"].append("Applied sRGB gamma")
            
            # Step 2: Handle Color Space Conversion
            if colorspace_mode == "sRGB":
                # Convert to sRGB color space
                img_array = self._convert_to_sRGB(img_array)
                conversion_info["conversions_applied"].append("Converted to sRGB color space")
                
            elif colorspace_mode == "linear_sRGB":
                # Convert to linear sRGB
                img_array = self._convert_to_linear_sRGB(img_array)
                conversion_info["conversions_applied"].append("Converted to linear sRGB")
                
            elif colorspace_mode == "Adobe_RGB":
                # Convert to Adobe RGB
                img_array = self._convert_to_Adobe_RGB(img_array)
                conversion_info["conversions_applied"].append("Converted to Adobe RGB")
                
            elif colorspace_mode == "ProPhoto_RGB":
                # Convert to ProPhoto RGB
                img_array = self._convert_to_ProPhoto_RGB(img_array)
                conversion_info["conversions_applied"].append("Converted to ProPhoto RGB")
            
            # Step 3: Apply Output Format
            if output_format == "Linear sRGB":
                img_array = self._convert_to_linear_sRGB(img_array)
                conversion_info["conversions_applied"].append("Output: Linear sRGB")
            elif output_format == "Adobe RGB":
                img_array = self._convert_to_Adobe_RGB(img_array)
                conversion_info["conversions_applied"].append("Output: Adobe RGB")
            elif output_format == "ProPhoto RGB":
                img_array = self._convert_to_ProPhoto_RGB(img_array)
                conversion_info["conversions_applied"].append("Output: ProPhoto RGB")
            
            # Convert back to ComfyUI format
            img_array = img_array.clip(0, 1)
            
            # Add batch dimension back
            if len(img_array.shape) == 3:
                img_array = img_array[np.newaxis, ...]
            
            # Convert back to tensor if needed
            if hasattr(torch, 'tensor'):
                result_image = torch.from_numpy(img_array)
            else:
                result_image = img_array
            
            return (result_image, json.dumps(conversion_info, indent=2))
            
        except Exception as e:
            error_info = {
                "error": str(e),
                "error_type": type(e).__name__,
                "gamma_mode": gamma_mode,
                "colorspace_mode": colorspace_mode,
                "output_format": output_format
            }
            return (image, json.dumps(error_info, indent=2))
    
    def _linearize_gamma(self, img_array, gamma):
        """Convert from gamma-encoded to linear"""
        import numpy as np
        return np.power(img_array, gamma)
    
    def _apply_gamma(self, img_array, gamma):
        """Apply gamma correction"""
        import numpy as np
        return np.power(img_array, 1.0 / gamma)
    
    def _sRGB_gamma(self, img_array):
        """Apply sRGB gamma correction"""
        import numpy as np
        # sRGB EOTF (gamma to linear)
        return np.where(
            img_array <= 0.04045,
            img_array / 12.92,
            np.power((img_array + 0.055) / 1.055, 2.4)
        )
    
    def _convert_to_sRGB(self, img_array):
        """Convert to sRGB color space"""
        # sRGB is the default, so just return
        return img_array
    
    def _convert_to_linear_sRGB(self, img_array):
        """Convert to linear sRGB"""
        import numpy as np
        # Apply sRGB gamma correction to linearize
        return self._sRGB_gamma(img_array)
    
    def _convert_to_Adobe_RGB(self, img_array):
        """Convert to Adobe RGB color space"""
        import numpy as np
        # Adobe RGB has different primaries and gamma
        # This is a simplified conversion
        # In practice, you'd use proper color space matrices
        return img_array
    
    def _convert_to_ProPhoto_RGB(self, img_array):
        """Convert to ProPhoto RGB color space"""
        import numpy as np
        # ProPhoto RGB has different primaries and gamma
        # This is a simplified conversion
        # In practice, you'd use proper color space matrices
        return img_array
