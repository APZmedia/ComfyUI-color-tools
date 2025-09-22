"""
Color Conversion Nodes

This module contains nodes for converting between different color spaces
and analyzing color space properties.
"""

import torch
import numpy as np
import cv2
from PIL import Image
import colorspacious as cs
from typing import Tuple, Dict, Any, Optional

class ColorSpaceConverter:
    """
    Convert images between different color spaces.
    Supports RGB, HSV, HSL, LAB, XYZ, CMYK conversions.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "source_space": (["RGB", "HSV", "HSL", "LAB", "XYZ", "CMYK"],),
                "target_space": (["RGB", "HSV", "HSL", "LAB", "XYZ", "CMYK"],),
                "preserve_alpha": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "gamma_correction": ("FLOAT", {"default": 2.2, "min": 0.1, "max": 5.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "conversion_info")
    FUNCTION = "convert_color_space"
    CATEGORY = "Color Tools/Conversion"
    
    def convert_color_space(self, image: torch.Tensor, source_space: str, target_space: str, 
                          preserve_alpha: bool, gamma_correction: float = 2.2) -> Tuple[torch.Tensor, str]:
        """
        Convert image between color spaces.
        """
        # Convert tensor to numpy
        if len(image.shape) == 4:
            batch_size, height, width, channels = image.shape
            img_np = image[0].numpy()
        else:
            height, width, channels = image.shape
            img_np = image.numpy()
        
        # Ensure image is in [0, 1] range
        if img_np.max() > 1.0:
            img_np = img_np / 255.0
        
        # Handle alpha channel
        has_alpha = channels == 4
        if has_alpha and preserve_alpha:
            alpha = img_np[:, :, 3:4]
            rgb_img = img_np[:, :, :3]
        else:
            rgb_img = img_np[:, :, :3]
            alpha = None
        
        # Convert to target color space
        converted_img = self._convert_space(rgb_img, source_space, target_space, gamma_correction)
        
        # Reconstruct image with alpha if needed
        if has_alpha and preserve_alpha:
            if converted_img.shape[2] == 3:
                converted_img = np.concatenate([converted_img, alpha], axis=2)
        
        # Convert back to tensor
        converted_tensor = torch.from_numpy(converted_img).float()
        if len(image.shape) == 4:
            converted_tensor = converted_tensor.unsqueeze(0)
        
        # Create conversion info
        info = f"Converted from {source_space} to {target_space}"
        if has_alpha and preserve_alpha:
            info += " (alpha preserved)"
        
        return converted_tensor, info
    
    def _convert_space(self, img: np.ndarray, source: str, target: str, gamma: float) -> np.ndarray:
        """Internal method to handle color space conversion."""
        if source == target:
            return img
        
        # Convert to RGB first if needed
        if source != "RGB":
            img = self._to_rgb(img, source, gamma)
        
        # Convert from RGB to target
        if target == "RGB":
            return img
        else:
            return self._from_rgb(img, target, gamma)
    
    def _to_rgb(self, img: np.ndarray, source_space: str, gamma: float) -> np.ndarray:
        """Convert from source space to RGB."""
        if source_space == "HSV":
            return cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_HSV2RGB) / 255.0
        elif source_space == "HSL":
            return cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_HLS2RGB) / 255.0
        elif source_space == "LAB":
            return cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_LAB2RGB) / 255.0
        elif source_space == "XYZ":
            return cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_XYZ2RGB) / 255.0
        elif source_space == "CMYK":
            # CMYK to RGB conversion
            c, m, y, k = img[:, :, 0], img[:, :, 1], img[:, :, 2], img[:, :, 3]
            r = (1 - c) * (1 - k)
            g = (1 - m) * (1 - k)
            b = (1 - y) * (1 - k)
            return np.stack([r, g, b], axis=2)
        return img
    
    def _from_rgb(self, img: np.ndarray, target_space: str, gamma: float) -> np.ndarray:
        """Convert from RGB to target space."""
        if target_space == "HSV":
            return cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV) / 255.0
        elif target_space == "HSL":
            return cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HLS) / 255.0
        elif target_space == "LAB":
            return cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB) / 255.0
        elif target_space == "XYZ":
            return cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2XYZ) / 255.0
        elif target_space == "CMYK":
            # RGB to CMYK conversion
            r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            k = 1 - np.maximum(np.maximum(r, g), b)
            c = (1 - r - k) / (1 - k + 1e-8)
            m = (1 - g - k) / (1 - k + 1e-8)
            y = (1 - b - k) / (1 - k + 1e-8)
            return np.stack([c, m, y, k], axis=2)
        return img


class ColorTemperature:
    """
    Adjust color temperature and tint of images.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "temperature": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "tint": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "adjust_temperature"
    CATEGORY = "Color Tools/Conversion"
    
    def adjust_temperature(self, image: torch.Tensor, temperature: float, tint: float) -> torch.Tensor:
        """
        Adjust color temperature and tint.
        """
        # Convert tensor to numpy
        if len(image.shape) == 4:
            img_np = image[0].numpy()
        else:
            img_np = image.numpy()
        
        # Ensure image is in [0, 1] range
        if img_np.max() > 1.0:
            img_np = img_np / 255.0
        
        # Apply temperature adjustment
        adjusted_img = self._apply_temperature_tint(img_np, temperature, tint)
        
        # Convert back to tensor
        adjusted_tensor = torch.from_numpy(adjusted_img).float()
        if len(image.shape) == 4:
            adjusted_tensor = adjusted_tensor.unsqueeze(0)
        
        return adjusted_tensor
    
    def _apply_temperature_tint(self, img: np.ndarray, temp: float, tint: float) -> np.ndarray:
        """Apply temperature and tint adjustments."""
        # Temperature adjustment (blue/amber)
        temp_factor = temp / 100.0
        if temp > 0:  # Warmer (more amber)
            img[:, :, 0] = np.clip(img[:, :, 0] + temp_factor * 0.1, 0, 1)  # Red
            img[:, :, 2] = np.clip(img[:, :, 2] - temp_factor * 0.1, 0, 1)  # Blue
        else:  # Cooler (more blue)
            img[:, :, 0] = np.clip(img[:, :, 0] - abs(temp_factor) * 0.1, 0, 1)  # Red
            img[:, :, 2] = np.clip(img[:, :, 2] + abs(temp_factor) * 0.1, 0, 1)  # Blue
        
        # Tint adjustment (green/magenta)
        tint_factor = tint / 100.0
        if tint > 0:  # More magenta
            img[:, :, 0] = np.clip(img[:, :, 0] + tint_factor * 0.05, 0, 1)  # Red
            img[:, :, 2] = np.clip(img[:, :, 2] + tint_factor * 0.05, 0, 1)  # Blue
        else:  # More green
            img[:, :, 1] = np.clip(img[:, :, 1] + abs(tint_factor) * 0.1, 0, 1)  # Green
        
        return img


class ColorSpaceAnalyzer:
    """
    Analyze color space properties and provide information about the image.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("color_space_info", "color_stats", "recommendations")
    FUNCTION = "analyze_color_space"
    CATEGORY = "Color Tools/Conversion"
    
    def analyze_color_space(self, image: torch.Tensor) -> Tuple[str, str, str]:
        """
        Analyze the color space properties of the image.
        """
        # Convert tensor to numpy
        if len(image.shape) == 4:
            img_np = image[0].numpy()
        else:
            img_np = image.numpy()
        
        # Ensure image is in [0, 1] range
        if img_np.max() > 1.0:
            img_np = img_np / 255.0
        
        # Analyze color space
        info = self._get_color_space_info(img_np)
        stats = self._get_color_statistics(img_np)
        recommendations = self._get_recommendations(img_np)
        
        return info, stats, recommendations
    
    def _get_color_space_info(self, img: np.ndarray) -> str:
        """Get basic color space information."""
        height, width, channels = img.shape
        info = f"Image dimensions: {width}x{height}\n"
        info += f"Channels: {channels}\n"
        info += f"Data type: {img.dtype}\n"
        info += f"Value range: [{img.min():.3f}, {img.max():.3f}]\n"
        
        if channels == 3:
            info += "Color space: RGB\n"
        elif channels == 4:
            info += "Color space: RGBA\n"
        else:
            info += f"Color space: {channels} channels\n"
        
        return info
    
    def _get_color_statistics(self, img: np.ndarray) -> str:
        """Get color statistics."""
        stats = "Color Statistics:\n"
        
        for i, channel in enumerate(['Red', 'Green', 'Blue']):
            if i < img.shape[2]:
                channel_data = img[:, :, i]
                stats += f"{channel}: mean={channel_data.mean():.3f}, "
                stats += f"std={channel_data.std():.3f}, "
                stats += f"min={channel_data.min():.3f}, "
                stats += f"max={channel_data.max():.3f}\n"
        
        return stats
    
    def _get_recommendations(self, img: np.ndarray) -> str:
        """Get color space recommendations."""
        recommendations = "Recommendations:\n"
        
        # Check for potential issues
        if img.max() > 1.0:
            recommendations += "- Image values exceed 1.0, consider normalizing\n"
        
        if img.min() < 0.0:
            recommendations += "- Image has negative values, check color space\n"
        
        # Check for color balance
        r_mean = img[:, :, 0].mean()
        g_mean = img[:, :, 1].mean()
        b_mean = img[:, :, 2].mean()
        
        if abs(r_mean - g_mean) > 0.1 or abs(g_mean - b_mean) > 0.1:
            recommendations += "- Color channels are unbalanced, consider color correction\n"
        
        # Check for saturation
        hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1].mean() / 255.0
        
        if saturation < 0.3:
            recommendations += "- Low saturation detected, consider saturation boost\n"
        elif saturation > 0.8:
            recommendations += "- High saturation detected, consider saturation reduction\n"
        
        return recommendations
