# comfyui_custom_nodes/color_profile_convert_simple.py
import base64
import io
import json
from typing import Dict, Optional, Tuple

from PIL import Image, ImageCms


class ColorProfileConvert:
    """
    ComfyUI node: converts images based on color profile data.
    This is a simplified version that works with minimal dependencies.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "profile_json": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "JSON string with color profile data"
                }),
                "output_mode": (["sRGB", "Linear sRGB"], {
                    "default": "sRGB"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("converted_image", "conversion_info")
    FUNCTION = "convert"
    CATEGORY = "Image/Color"

    def convert(self, image, profile_json: str, output_mode: str):
        """
        Convert image based on color profile data.
        """
        try:
            # Parse profile data
            if not profile_json.strip():
                return (image, "No profile data provided")
            
            profile_data = json.loads(profile_json)
            
            # Get basic image info
            batch_size, height, width, channels = image.shape
            
            # Convert to PIL Image for processing
            # Convert from ComfyUI format (0-1 float) to PIL format (0-255 int)
            import torch
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
            
            # Convert to 0-255 range for PIL
            img_array = (img_array * 255).astype('uint8')
            
            # Handle batch dimension
            if batch_size == 1:
                img_array = img_array[0]  # Remove batch dimension
            else:
                # For now, process first image in batch
                img_array = img_array[0]
            
            # Create PIL Image
            pil_image = Image.fromarray(img_array, 'RGB')
            
            # Apply conversion based on profile data
            conversion_info = {
                "original_size": (width, height),
                "profile_found": bool(profile_data.get('profile')),
                "gamma": profile_data.get('gamma'),
                "primaries": profile_data.get('primaries'),
                "output_mode": output_mode
            }
            
            # Simple gamma correction if gamma is available
            if profile_data.get('gamma') and output_mode == "Linear sRGB":
                gamma = profile_data['gamma']
                if gamma > 0:
                    # Apply gamma correction to linearize
                    import numpy as np
                    img_array = img_array.astype(np.float32) / 255.0
                    img_array = np.power(img_array, gamma)
                    img_array = (img_array * 255).astype(np.uint8)
                    pil_image = Image.fromarray(img_array, 'RGB')
                    conversion_info["gamma_applied"] = gamma
            
            # Convert back to ComfyUI format
            img_array = np.array(pil_image).astype(np.float32) / 255.0
            
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
                "output_mode": output_mode
            }
            return (image, json.dumps(error_info, indent=2))
