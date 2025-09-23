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


class TestPatternGenerator:
    """
    Generate test patterns for color space validation and calibration.
    Creates color bars, tone ramps, and other patterns useful for testing color transforms.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pattern_type": (["Color Bars", "Tone Ramp", "Gray Ramp", "SMPTE Color Bars", "ColorChecker"], {"default": "Color Bars"}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 64}),
                "height": ("INT", {"default": 256, "min": 64, "max": 4096, "step": 64}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("test_pattern", "pattern_info")
    FUNCTION = "generate_test_pattern"
    CATEGORY = "Color Tools/OCIO"

    def generate_test_pattern(self, pattern_type: str, width: int, height: int) -> Tuple[torch.Tensor, str]:
        """
        Generate a test pattern for color space validation.
        """
        # Create empty image
        pattern = np.zeros((height, width, 3), dtype=np.float32)

        if pattern_type == "Color Bars":
            self._generate_color_bars(pattern)
            info = "Color Bars: Primary colors (R,G,B), secondary (C,M,Y), and gray steps"
        elif pattern_type == "Tone Ramp":
            self._generate_tone_ramp(pattern)
            info = "Tone Ramp: Linear ramp from black to white"
        elif pattern_type == "Gray Ramp":
            self._generate_gray_ramp(pattern)
            info = "Gray Ramp: From pure dark to pure white through grays"
        elif pattern_type == "SMPTE Color Bars":
            self._generate_smpte_bars(pattern)
            info = "SMPTE Color Bars: Standard broadcast test pattern"
        elif pattern_type == "ColorChecker":
            self._generate_color_checker(pattern)
            info = "ColorChecker: 24 standard color patches"
        else:
            # Default to color bars
            self._generate_color_bars(pattern)
            info = "Color Bars: Primary colors test pattern"

        # Convert to tensor and add batch dimension
        pattern_tensor = torch.from_numpy(pattern).float().unsqueeze(0)

        return pattern_tensor, info

    def _generate_color_bars(self, pattern: np.ndarray) -> None:
        """Generate basic color bars."""
        h, w = pattern.shape[:2]
        bar_width = w // 8

        # Red
        pattern[:, :bar_width, 0] = 1.0
        # Green
        pattern[:, bar_width:2*bar_width, 1] = 1.0
        # Blue
        pattern[:, 2*bar_width:3*bar_width, 2] = 1.0
        # Cyan
        pattern[:, 3*bar_width:4*bar_width, 1:] = 1.0
        # Magenta
        pattern[:, 4*bar_width:5*bar_width, [0, 2]] = 1.0
        # Yellow
        pattern[:, 5*bar_width:6*bar_width, [0, 1]] = 1.0
        # White
        pattern[:, 6*bar_width:7*bar_width, :] = 1.0
        # Black
        pattern[:, 7*bar_width:, :] = 0.0

    def _generate_tone_ramp(self, pattern: np.ndarray) -> None:
        """Generate a linear tone ramp."""
        h, w = pattern.shape[:2]
        for x in range(w):
            intensity = x / (w - 1)
            pattern[:, x, :] = intensity

    def _generate_gray_ramp(self, pattern: np.ndarray) -> None:
        """Generate a gray ramp."""
        h, w = pattern.shape[:2]
        for x in range(w):
            gray = x / (w - 1)
            pattern[:, x, :] = gray

    def _generate_smpte_bars(self, pattern: np.ndarray) -> None:
        """Generate SMPTE color bars."""
        h, w = pattern.shape[:2]
        bar_width = w // 7

        # SMPTE color bars pattern
        bars = [
            (255, 255, 255),  # White
            (255, 255, 0),   # Yellow
            (0, 255, 255),   # Cyan
            (0, 255, 0),     # Green
            (255, 0, 255),   # Magenta
            (255, 0, 0),     # Red
            (0, 0, 255),     # Blue
        ]

        for i, (r, g, b) in enumerate(bars):
            if i < len(bars):
                x_start = i * bar_width
                x_end = min((i + 1) * bar_width, w)
                pattern[:, x_start:x_end, 0] = r / 255.0
                pattern[:, x_start:x_end, 1] = g / 255.0
                pattern[:, x_start:x_end, 2] = b / 255.0

    def _generate_color_checker(self, pattern: np.ndarray) -> None:
        """Generate a simplified ColorChecker pattern."""
        h, w = pattern.shape[:2]
        patch_height = h // 4
        patch_width = w // 6

        # Simplified ColorChecker colors (24 patches in 4x6 grid)
        colors = [
            [0.42, 0.31, 0.28],  # Dark Skin
            [0.62, 0.44, 0.38],  # Light Skin
            [0.31, 0.33, 0.35],  # Blue Sky
            [0.15, 0.20, 0.24],  # Foliage
            [0.50, 0.23, 0.17],  # Blue Flower
            [0.14, 0.14, 0.14],  # Bluish Gray
            [0.43, 0.34, 0.22],  # Orange
            [0.19, 0.21, 0.05],  # Purplish Blue
            [0.35, 0.37, 0.36],  # Moderate Red
            [0.39, 0.27, 0.21],  # Purple
            [0.53, 0.48, 0.45],  # Yellow Green
            [0.25, 0.25, 0.25],  # Orange Yellow
            [0.59, 0.35, 0.33],  # Blue
            [0.35, 0.35, 0.35],  # Green
            [0.19, 0.20, 0.18],  # Red
            [0.62, 0.62, 0.62],  # Yellow
            [0.19, 0.28, 0.35],  # Magenta
            [0.14, 0.14, 0.14],  # Cyan
            [0.85, 0.85, 0.85],  # White
            [0.58, 0.58, 0.58],  # Neutral 8
            [0.35, 0.35, 0.35],  # Neutral 6.5
            [0.19, 0.19, 0.19],  # Neutral 5
            [0.12, 0.12, 0.12],  # Neutral 3.5
            [0.06, 0.06, 0.06],  # Black
        ]

        for i in range(4):
            for j in range(6):
                if i * 6 + j < len(colors):
                    y_start = i * patch_height
                    y_end = (i + 1) * patch_height
                    x_start = j * patch_width
                    x_end = (j + 1) * patch_width

                    color = colors[i * 6 + j]
                    pattern[y_start:y_end, x_start:x_end, 0] = color[0]
                    pattern[y_start:y_end, x_start:x_end, 1] = color[1]
                    pattern[y_start:y_end, x_start:x_end, 2] = color[2]
