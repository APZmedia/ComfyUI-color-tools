"""
Color Grading Nodes

This module contains nodes for color correction and grading operations
including brightness, contrast, saturation, and color balance adjustments.
"""

import torch
import numpy as np
import cv2
from typing import Tuple, Dict, Any, Optional

class ColorBalance:
    """
    Adjust color balance for shadows, midtones, and highlights.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "shadow_red": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "shadow_green": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "shadow_blue": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "midtone_red": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "midtone_green": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "midtone_blue": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "highlight_red": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "highlight_green": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "highlight_blue": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "adjust_color_balance"
    CATEGORY = "Color Tools/Grading"
    
    def adjust_color_balance(self, image: torch.Tensor, shadow_red: float, shadow_green: float, 
                           shadow_blue: float, midtone_red: float, midtone_green: float, 
                           midtone_blue: float, highlight_red: float, highlight_green: float, 
                           highlight_blue: float) -> torch.Tensor:
        """
        Adjust color balance for different tonal ranges.
        """
        # Convert tensor to numpy
        if len(image.shape) == 4:
            img_np = image[0].numpy()
        else:
            img_np = image.numpy()
        
        # Ensure image is in [0, 1] range
        if img_np.max() > 1.0:
            img_np = img_np / 255.0
        
        # Apply color balance
        adjusted_img = self._apply_color_balance(
            img_np, 
            (shadow_red, shadow_green, shadow_blue),
            (midtone_red, midtone_green, midtone_blue),
            (highlight_red, highlight_green, highlight_blue)
        )
        
        # Convert back to tensor
        adjusted_tensor = torch.from_numpy(adjusted_img).float()
        if len(image.shape) == 4:
            adjusted_tensor = adjusted_tensor.unsqueeze(0)
        
        return adjusted_tensor
    
    def _apply_color_balance(self, img: np.ndarray, shadow: Tuple[float, float, float], 
                           midtone: Tuple[float, float, float], 
                           highlight: Tuple[float, float, float]) -> np.ndarray:
        """Apply color balance adjustments."""
        result = img.copy()
        
        # Calculate luminance to determine tonal ranges
        luminance = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        
        # Create masks for shadows, midtones, and highlights
        shadow_mask = np.where(luminance < 0.33, 1.0, 0.0)
        midtone_mask = np.where((luminance >= 0.33) & (luminance <= 0.66), 1.0, 0.0)
        highlight_mask = np.where(luminance > 0.66, 1.0, 0.0)
        
        # Apply shadow adjustments
        for i, (r, g, b) in enumerate([shadow, midtone, highlight]):
            mask = [shadow_mask, midtone_mask, highlight_mask][i]
            result[:, :, 0] += r * mask  # Red
            result[:, :, 1] += g * mask  # Green
            result[:, :, 2] += b * mask  # Blue
        
        # Clamp values to [0, 1] range
        result = np.clip(result, 0, 1)
        
        return result


class BrightnessContrast:
    """
    Adjust brightness and contrast of images.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "brightness": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "adjust_brightness_contrast"
    CATEGORY = "Color Tools/Grading"
    
    def adjust_brightness_contrast(self, image: torch.Tensor, brightness: float, contrast: float) -> torch.Tensor:
        """
        Adjust brightness and contrast.
        """
        # Convert tensor to numpy
        if len(image.shape) == 4:
            img_np = image[0].numpy()
        else:
            img_np = image.numpy()
        
        # Ensure image is in [0, 1] range
        if img_np.max() > 1.0:
            img_np = img_np / 255.0
        
        # Apply brightness and contrast
        adjusted_img = self._apply_brightness_contrast(img_np, brightness, contrast)
        
        # Convert back to tensor
        adjusted_tensor = torch.from_numpy(adjusted_img).float()
        if len(image.shape) == 4:
            adjusted_tensor = adjusted_tensor.unsqueeze(0)
        
        return adjusted_tensor
    
    def _apply_brightness_contrast(self, img: np.ndarray, brightness: float, contrast: float) -> np.ndarray:
        """Apply brightness and contrast adjustments."""
        # Apply contrast first
        result = (img - 0.5) * contrast + 0.5
        
        # Then apply brightness
        result = result + brightness
        
        # Clamp values to [0, 1] range
        result = np.clip(result, 0, 1)
        
        return result


class Saturation:
    """
    Adjust color saturation of images.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "preserve_luminance": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "adjust_saturation"
    CATEGORY = "Color Tools/Grading"
    
    def adjust_saturation(self, image: torch.Tensor, saturation: float, preserve_luminance: bool) -> torch.Tensor:
        """
        Adjust color saturation.
        """
        # Convert tensor to numpy
        if len(image.shape) == 4:
            img_np = image[0].numpy()
        else:
            img_np = image.numpy()
        
        # Ensure image is in [0, 1] range
        if img_np.max() > 1.0:
            img_np = img_np / 255.0
        
        # Apply saturation
        adjusted_img = self._apply_saturation(img_np, saturation, preserve_luminance)
        
        # Convert back to tensor
        adjusted_tensor = torch.from_numpy(adjusted_img).float()
        if len(image.shape) == 4:
            adjusted_tensor = adjusted_tensor.unsqueeze(0)
        
        return adjusted_tensor
    
    def _apply_saturation(self, img: np.ndarray, saturation: float, preserve_luminance: bool) -> np.ndarray:
        """Apply saturation adjustment."""
        if preserve_luminance:
            # Convert to HSV, adjust saturation, convert back
            hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB) / 255.0
        else:
            # Simple multiplication approach
            # Calculate grayscale
            gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
            gray = np.stack([gray, gray, gray], axis=2)
            
            # Apply saturation
            result = gray + saturation * (img - gray)
        
        # Clamp values to [0, 1] range
        result = np.clip(result, 0, 1)
        
        return result


class HueShift:
    """
    Shift hue values of images.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "hue_shift": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "shift_hue"
    CATEGORY = "Color Tools/Grading"
    
    def shift_hue(self, image: torch.Tensor, hue_shift: float) -> torch.Tensor:
        """
        Shift hue values.
        """
        # Convert tensor to numpy
        if len(image.shape) == 4:
            img_np = image[0].numpy()
        else:
            img_np = image.numpy()
        
        # Ensure image is in [0, 1] range
        if img_np.max() > 1.0:
            img_np = img_np / 255.0
        
        # Apply hue shift
        adjusted_img = self._apply_hue_shift(img_np, hue_shift)
        
        # Convert back to tensor
        adjusted_tensor = torch.from_numpy(adjusted_img).float()
        if len(image.shape) == 4:
            adjusted_tensor = adjusted_tensor.unsqueeze(0)
        
        return adjusted_tensor
    
    def _apply_hue_shift(self, img: np.ndarray, hue_shift: float) -> np.ndarray:
        """Apply hue shift."""
        # Convert to HSV
        hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        
        # Apply hue shift
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        
        # Convert back to RGB
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB) / 255.0
        
        return result


class GammaCorrection:
    """
    Apply gamma correction to images.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_gamma_correction"
    CATEGORY = "Color Tools/Grading"
    
    def apply_gamma_correction(self, image: torch.Tensor, gamma: float) -> torch.Tensor:
        """
        Apply gamma correction.
        """
        # Convert tensor to numpy
        if len(image.shape) == 4:
            img_np = image[0].numpy()
        else:
            img_np = image.numpy()
        
        # Ensure image is in [0, 1] range
        if img_np.max() > 1.0:
            img_np = img_np / 255.0
        
        # Apply gamma correction
        adjusted_img = self._apply_gamma(img_np, gamma)
        
        # Convert back to tensor
        adjusted_tensor = torch.from_numpy(adjusted_img).float()
        if len(image.shape) == 4:
            adjusted_tensor = adjusted_tensor.unsqueeze(0)
        
        return adjusted_tensor
    
    def _apply_gamma(self, img: np.ndarray, gamma: float) -> np.ndarray:
        """Apply gamma correction."""
        # Apply gamma correction
        result = np.power(img, 1.0 / gamma)
        
        # Clamp values to [0, 1] range
        result = np.clip(result, 0, 1)
        
        return result
