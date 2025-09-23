"""
Advanced OCIO Color Tools Node

This module contains the AdvancedOcioColorTransform node, giving granular control
over OCIO color conversions with a staged pipeline.
"""

import torch
import numpy as np
from typing import Tuple, List
import os

try:
    import PyOpenColorIO as ocio
except ImportError:
    print("[Color Tools] ❌ PyOpenColorIO not available. Advanced OCIO node will not work.")
    ocio = None

def get_ocio_config_path():
    """Gets the path to the bundled OCIO config."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ocio", "config.ocio")

def get_ocio_spaces():
    """Dynamically get the color spaces from the current OCIO config."""
    if ocio is None:
        return ["sRGB", "Linear", "raw"]
    try:
        # First, try to load the bundled config
        config_path = get_ocio_config_path()
        if os.path.exists(config_path):
            config = ocio.Config.CreateFromFile(config_path)
        else:
            # Fallback to the system's default config
            config = ocio.GetCurrentConfig()
        return [cs.getName() for cs in config.getColorSpaces()]
    except ocio.Exception:
        # If all else fails, return a safe list
        return ["sRGB", "Linear", "raw"]

# Mapping from display spaces to their linear equivalents
# This is now a fallback, as the code will try to dynamically find a linear space
DISPLAY_TO_LINEAR_MAP = {
    "sRGB": "Linear",
    "Rec.709": "Linear",
    "Rec.2020": "Linear",
    "P3-D65": "Linear P3-D65", # Example for a more complex config
}

def get_linear_equivalent(space_name: str, config) -> str:
    """Finds the linear version of a display space, or returns it if already linear."""
    if "linear" in space_name.lower():
        return space_name

    # Try to find a space with the same name but "Linear" in it
    for cs in config.getColorSpaces():
        if f"linear {space_name.lower()}" in cs.getName().lower():
            return cs.getName()
            
    # Fallback to the hardcoded map
    return DISPLAY_TO_LINEAR_MAP.get(space_name, space_name)


class AdvancedOcioColorTransform:
    """
    An advanced OCIO node for fine-grained control over color space conversions.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "source_space": (get_ocio_spaces(), {"default": "sRGB"}),
                "dest_space": (get_ocio_spaces(), {"default": "sRGB"}),
                "fix_transfer": ("BOOLEAN", {"default": True}),
                "fix_gamut": ("BOOLEAN", {"default": True}),
                "gamut_compress": ("BOOLEAN", {"default": False}),
                "gc_threshold": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01}),
                "gc_power": ("FLOAT", {"default": 1.15, "min": 0.0, "max": 2.0, "step": 0.01}),
                "gc_scale": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "alpha_mode": (["passthrough", "premultiply"], {"default": "passthrough"}),
                "clip_after": ("BOOLEAN", {"default": True}),
                "ocio_config_path": ("STRING", {"default": get_ocio_config_path(), "multiline": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "transform_info")
    FUNCTION = "transform_image"
    CATEGORY = "Color Tools/OCIO"

    def transform_image(self, image: torch.Tensor, source_space: str, dest_space: str,
                        fix_transfer: bool, fix_gamut: bool, gamut_compress: bool,
                        gc_threshold: float, gc_power: float, gc_scale: float,
                        alpha_mode: str, clip_after: bool, ocio_config_path: str) -> Tuple[torch.Tensor, str]:

        if ocio is None:
            raise ImportError("PyOpenColorIO is required for this node to work.")

        # Load OCIO configuration
        try:
            if ocio_config_path and os.path.exists(ocio_config_path):
                config = ocio.Config.CreateFromFile(ocio_config_path)
            else:
                config = ocio.GetCurrentConfig()
        except Exception as e:
            return image, f"Failed to load OCIO config: {str(e)}"

        # Prepare image
        h, w, c = image.shape[-3:]
        img_np = image[0].cpu().numpy() if len(image.shape) == 4 else image.cpu().numpy()
        
        alpha = img_np[:, :, 3:4] if c == 4 else None
        if alpha_mode == 'premultiply' and alpha is not None:
            img_np[:, :, :3] *= alpha # Premultiply
            
        img_rgb = img_np[:, :, :3].astype(np.float32)

        # Build the transform pipeline
        transforms = []
        current_space = source_space
        
        source_linear = get_linear_equivalent(source_space, config)
        dest_linear = get_linear_equivalent(dest_space, config)

        # Stage A: Decode transfer
        if fix_transfer and current_space != source_linear:
            transforms.append(ocio.ColorSpaceTransform(src=current_space, dst=source_linear))
            current_space = source_linear

        # Stage B: Change primaries
        if fix_gamut and source_linear != dest_linear:
            transforms.append(ocio.ColorSpaceTransform(src=current_space, dst=dest_linear))
            current_space = dest_linear
            
        # Optional: Gamut compression
        if gamut_compress:
            compress_transform = ocio.GamutCompressTransform(
                threshold=gc_threshold, power=gc_power, scale=gc_scale)
            transforms.append(compress_transform)

        # Stage C: Encode transfer
        if fix_transfer and current_space != dest_space:
            transforms.append(ocio.ColorSpaceTransform(src=current_space, dst=dest_space))
            current_space = dest_space

        if not transforms:
            return image, "No conversion needed. All toggles were off or spaces matched."

        # Create the full processor
        group_transform = ocio.GroupTransform(transforms)
        processor = config.getProcessor(group_transform)
        cpu_processor = processor.getDefaultCPUProcessor()

        # Apply the transform
        cpu_processor.applyRGB(img_rgb)
        
        if clip_after:
            img_rgb = np.clip(img_rgb, 0.0, 1.0)
            
        # Re-handle alpha
        if alpha is not None:
            if alpha_mode == 'premultiply':
                img_rgb /= (alpha + 1e-8) # Unpremultiply, avoiding division by zero
            final_img = np.concatenate([img_rgb, alpha], axis=2)
        else:
            final_img = img_rgb

        # Convert back to tensor
        result_tensor = torch.from_numpy(final_img).float().unsqueeze(0)
        
        info = {
            "source": source_space,
            "destination": dest_space,
            "final_space": current_space,
            "stages": [t.__class__.__name__ for t in transforms],
            "gamut_compression": "on" if gamut_compress else "off"
        }
        
        return result_tensor, str(info)
