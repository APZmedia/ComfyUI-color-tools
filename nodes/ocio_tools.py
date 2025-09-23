"""
OCIO Color Tools Nodes

This module contains OpenColorIO-based color management nodes for professional-grade
color space conversions and color processing.
"""

import torch
import numpy as np
try:
    import PyOpenColorIO as ocio
except ImportError:
    print("PyOpenColorIO not available. OCIO nodes will not work.")
    ocio = None

from typing import Tuple, Optional


class OCIOColorSpaceConverter:
    """
    Professional color space conversion using OpenColorIO.
    Supports any color spaces defined in the OCIO configuration.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "ocio_config_path": ("STRING", {"default": "", "multiline": False}),
                "source_colorspace": ("STRING", {"default": "sRGB", "multiline": False}),
                "target_colorspace": ("STRING", {"default": "Linear sRGB", "multiline": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "conversion_info")
    FUNCTION = "convert_colorspace"
    CATEGORY = "Color Tools/OCIO"

    def convert_colorspace(self, image: torch.Tensor, ocio_config_path: str,
                          source_colorspace: str, target_colorspace: str) -> Tuple[torch.Tensor, str]:
        """
        Convert image between color spaces using OCIO.
        """
        if ocio is None:
            raise ImportError("PyOpenColorIO is required for OCIO color space conversion")

        # Convert tensor to numpy (handle batch and non-batch)
        if len(image.shape) == 4:
            batch_size, h, w, c = image.shape
            img_np = image[0].cpu().numpy()
        else:
            h, w, c = image.shape
            img_np = image.cpu().numpy()

        # Normalize to 0-1 if necessary
        if img_np.max() > 1.0:
            img_np = img_np / 255.0

        # Handle alpha channel
        if c == 4:
            alpha = img_np[:, :, 3:4].copy()
            img_rgb = img_np[:, :, :3]
        else:
            img_rgb = img_np
            alpha = None

        # Load OCIO configuration
        try:
            if ocio_config_path.strip():
                config = ocio.Config.CreateFromFile(ocio_config_path)
            else:
                config = ocio.GetCurrentConfig()
        except Exception as e:
            return image, f"Failed to load OCIO config: {str(e)}"

        # Validate color spaces
        try:
            if not config.getColorSpace(source_colorspace):
                return image, f"Source color space '{source_colorspace}' not found in config"
            if not config.getColorSpace(target_colorspace):
                return image, f"Target color space '{target_colorspace}' not found in config"
        except Exception as e:
            return image, f"Error accessing color spaces: {str(e)}"

        # Create processor for color space conversion
        try:
            processor = config.getProcessor(source_colorspace, target_colorspace).getDefaultCPUProcessor()
        except Exception as e:
            return image, f"Failed to create processor: {str(e)}"

        # Convert to OCIO-friendly format
        # OCIO expects float32, RGB, row-major
        img_flat = img_rgb.astype(np.float32).flatten()
        pixel_count = len(img_flat) // 3

        # Apply OCIO transform (process RGB channels)
        for i in range(pixel_count):
            rgb_pixel = img_flat[i*3:(i+1)*3]
            processed_rgb = processor.applyRGB(rgb_pixel)
            img_flat[i*3:(i+1)*3] = processed_rgb

        # Reshape back
        result_rgb = img_flat.reshape((h, w, 3))

        # Reattach alpha if present
        if alpha is not None:
            result_img = np.concatenate([result_rgb, alpha], axis=2)
        else:
            result_img = result_rgb

        # Convert back to tensor
        result_tensor = torch.from_numpy(result_img).float()
        if len(image.shape) == 4:
            result_tensor = result_tensor.unsqueeze(0)

        # Conversion info
        info = {
            "source_colorspace": source_colorspace,
            "target_colorspace": target_colorspace,
            "ocio_config": ocio_config_path if ocio_config_path else "Default",
            "image_shape": f"{h}x{w}",
            "alpha_preserved": alpha is not None
        }

        return result_tensor, str(info)


class OCIOConfigInfo:
    """
    Display information about an OCIO configuration.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ocio_config_path": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("config_info",)
    FUNCTION = "get_config_info"
    CATEGORY = "Color Tools/OCIO"

    def get_config_info(self, ocio_config_path: str) -> Tuple[str]:
        """
        Get information about the OCIO configuration.
        """
        if ocio is None:
            return ["PyOpenColorIO not available"]

        try:
            if ocio_config_path.strip():
                config = ocio.Config.CreateFromFile(ocio_config_path)
            else:
                config = ocio.GetCurrentConfig()
        except Exception as e:
            return [f"Failed to load config: {str(e)}"]

        # Get color spaces
        color_spaces = []
        for cs in config.getColorSpaces():
            color_spaces.append(f"- {cs.getName()} ({cs.getFamily()})")

        # Get displays and views
        displays_views = []
        for display in config.getDisplays():
            for view in config.getViews(display):
                displays_views.append(f"- {display}: {view}")

        info = f"""OCIO Configuration Info:
Config File: {ocio_config_path if ocio_config_path else 'Default'}

Working Color Space: {config.getColorSpace(config.getCanonicalName(ocio.REFERENCE_SPACE_SCENE))}

Color Spaces ({len(color_spaces)}):
{chr(10).join(color_spaces)}

Displays/Views ({len(displays_views)}):
{chr(10).join(displays_views)}

Roles:
- color_picking: {config.getColorSpace(config.getCanonicalName(ocio.REFERENCE_SPACE_DISPLAY)) if config.getCanonicalName(ocio.REFERENCE_SPACE_DISPLAY) else 'Not defined'}
- scene_linear: {config.getColorSpace(config.getCanonicalName(ocio.REFERENCE_SPACE_SCENE)) if config.getCanonicalName(ocio.REFERENCE_SPACE_SCENE) else 'Not defined'}
"""

        return [info]
