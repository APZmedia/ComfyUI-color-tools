"""
Advanced Color Tools Nodes

This module contains advanced color processing nodes including color matching,
quantization, gamut mapping, and color blindness simulation.
"""

import torch
import numpy as np
import cv2
from sklearn.cluster import KMeans
import json
from typing import Tuple, Dict, Any, List, Optional

class ColorMatcher:
    """
    Match and replace colors in images.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "source_color": ("STRING", {"default": "#FF0000"}),
                "target_color": ("STRING", {"default": "#00FF00"}),
                "tolerance": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "replace_mode": (["Exact", "Similar", "Hue Only"], {"default": "Similar"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "replacement_info")
    FUNCTION = "match_and_replace_colors"
    CATEGORY = "Color Tools/Advanced"
    
    def match_and_replace_colors(self, image: torch.Tensor, source_color: str, target_color: str,
                               tolerance: float, replace_mode: str) -> Tuple[torch.Tensor, str]:
        """
        Match and replace colors in the image.
        """
        # Convert tensor to numpy
        if len(image.shape) == 4:
            img_np = image[0].numpy()
        else:
            img_np = image.numpy()
        
        # Ensure image is in [0, 1] range
        if img_np.max() > 1.0:
            img_np = img_np / 255.0
        
        # Parse colors
        source_rgb = self._parse_color(source_color)
        target_rgb = self._parse_color(target_color)
        
        # Match and replace colors
        result_img, info = self._match_and_replace(img_np, source_rgb, target_rgb, tolerance, replace_mode)
        
        # Convert back to tensor
        result_tensor = torch.from_numpy(result_img).float()
        if len(image.shape) == 4:
            result_tensor = result_tensor.unsqueeze(0)
        
        return result_tensor, info
    
    def _parse_color(self, color_str: str) -> np.ndarray:
        """Parse color string to RGB values."""
        if color_str.startswith("#"):
            # Hex color
            hex_color = color_str[1:]
            r = int(hex_color[0:2], 16) / 255.0
            g = int(hex_color[2:4], 16) / 255.0
            b = int(hex_color[4:6], 16) / 255.0
            return np.array([r, g, b])
        else:
            # Assume RGB tuple
            try:
                rgb = eval(color_str)
                return np.array(rgb) / 255.0
            except:
                return np.array([1.0, 0.0, 0.0])  # Default to red
    
    def _match_and_replace(self, img: np.ndarray, source_rgb: np.ndarray, target_rgb: np.ndarray,
                          tolerance: float, replace_mode: str) -> Tuple[np.ndarray, str]:
        """Match and replace colors based on the specified mode."""
        result = img.copy()
        
        if replace_mode == "Exact":
            # Exact color matching
            mask = np.all(np.abs(img - source_rgb) < tolerance, axis=2)
            result[mask] = target_rgb
            
        elif replace_mode == "Similar":
            # Similar color matching using Euclidean distance
            distances = np.sqrt(np.sum((img - source_rgb) ** 2, axis=2))
            mask = distances < tolerance
            result[mask] = target_rgb
            
        elif replace_mode == "Hue Only":
            # Match only hue, preserve saturation and value
            img_hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
            source_hsv = cv2.cvtColor((source_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
            target_hsv = cv2.cvtColor((target_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
            
            # Match hue
            hue_diff = np.abs(img_hsv[:, :, 0].astype(float) - source_hsv[0, 0, 0])
            hue_diff = np.minimum(hue_diff, 180 - hue_diff)  # Handle hue wraparound
            mask = hue_diff < (tolerance * 180)
            
            # Replace hue while preserving saturation and value
            result_hsv = img_hsv.copy()
            result_hsv[mask, 0] = target_hsv[0, 0, 0]
            result = cv2.cvtColor(result_hsv, cv2.COLOR_HSV2RGB) / 255.0
        
        # Calculate replacement statistics
        total_pixels = img.shape[0] * img.shape[1]
        replaced_pixels = np.sum(mask)
        replacement_percentage = (replaced_pixels / total_pixels) * 100
        
        info = {
            "source_color": source_rgb.tolist(),
            "target_color": target_rgb.tolist(),
            "replace_mode": replace_mode,
            "tolerance": tolerance,
            "replaced_pixels": int(replaced_pixels),
            "total_pixels": int(total_pixels),
            "replacement_percentage": float(replacement_percentage)
        }
        
        return result, json.dumps(info)


class ColorQuantizer:
    """
    Reduce the number of colors in an image using various quantization methods.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "num_colors": ("INT", {"default": 16, "min": 2, "max": 256, "step": 1}),
                "quantization_method": (["K-means", "Median Cut", "Octree", "Uniform"], 
                                     {"default": "K-means"}),
                "dithering": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "quantization_info")
    FUNCTION = "quantize_colors"
    CATEGORY = "Color Tools/Advanced"
    
    def quantize_colors(self, image: torch.Tensor, num_colors: int, quantization_method: str,
                       dithering: bool) -> Tuple[torch.Tensor, str]:
        """
        Quantize colors in the image.
        """
        # Convert tensor to numpy
        if len(image.shape) == 4:
            img_np = image[0].numpy()
        else:
            img_np = image.numpy()
        
        # Ensure image is in [0, 1] range
        if img_np.max() > 1.0:
            img_np = img_np / 255.0
        
        # Quantize colors
        quantized_img, info = self._quantize_colors(img_np, num_colors, quantization_method, dithering)
        
        # Convert back to tensor
        quantized_tensor = torch.from_numpy(quantized_img).float()
        if len(image.shape) == 4:
            quantized_tensor = quantized_tensor.unsqueeze(0)
        
        return quantized_tensor, info
    
    def _quantize_colors(self, img: np.ndarray, num_colors: int, method: str, dithering: bool) -> Tuple[np.ndarray, str]:
        """Quantize colors using the specified method."""
        if method == "K-means":
            result = self._kmeans_quantization(img, num_colors)
        elif method == "Median Cut":
            result = self._median_cut_quantization(img, num_colors)
        elif method == "Octree":
            result = self._octree_quantization(img, num_colors)
        elif method == "Uniform":
            result = self._uniform_quantization(img, num_colors)
        else:
            result = self._kmeans_quantization(img, num_colors)
        
        # Apply dithering if requested
        if dithering:
            result = self._apply_dithering(img, result)
        
        # Calculate quantization statistics
        original_colors = len(np.unique(img.reshape(-1, 3), axis=0))
        quantized_colors = len(np.unique(result.reshape(-1, 3), axis=0))
        
        info = {
            "method": method,
            "target_colors": num_colors,
            "original_colors": int(original_colors),
            "quantized_colors": int(quantized_colors),
            "dithering": dithering,
            "compression_ratio": float(original_colors / quantized_colors)
        }
        
        return result, json.dumps(info)
    
    def _kmeans_quantization(self, img: np.ndarray, num_colors: int) -> np.ndarray:
        """K-means color quantization."""
        pixels = img.reshape(-1, 3)
        kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)
        quantized_pixels = kmeans.cluster_centers_[labels]
        return quantized_pixels.reshape(img.shape)
    
    def _median_cut_quantization(self, img: np.ndarray, num_colors: int) -> np.ndarray:
        """Median cut color quantization."""
        # Simplified median cut implementation
        pixels = img.reshape(-1, 3)
        
        def median_cut(pixels, depth):
            if depth == 0 or len(pixels) == 0:
                return np.mean(pixels, axis=0)
            
            # Find the channel with the greatest range
            ranges = np.max(pixels, axis=0) - np.min(pixels, axis=0)
            channel = np.argmax(ranges)
            
            # Sort by the channel with greatest range
            pixels_sorted = pixels[np.argsort(pixels[:, channel])]
            
            # Split at median
            median = len(pixels_sorted) // 2
            
            # Recursively process both halves
            left = median_cut(pixels_sorted[:median], depth - 1)
            right = median_cut(pixels_sorted[median:], depth - 1)
            
            return [left, right]
        
        # Calculate depth needed
        depth = int(np.log2(num_colors))
        colors = median_cut(pixels, depth)
        
        # Flatten the color tree
        def flatten_colors(colors):
            if isinstance(colors, list):
                result = []
                for color in colors:
                    result.extend(flatten_colors(color))
                return result
            else:
                return [colors]
        
        flat_colors = flatten_colors(colors)[:num_colors]
        
        # Assign each pixel to the closest color
        result = np.zeros_like(pixels)
        for i, pixel in enumerate(pixels):
            distances = [np.linalg.norm(pixel - color) for color in flat_colors]
            closest_color = flat_colors[np.argmin(distances)]
            result[i] = closest_color
        
        return result.reshape(img.shape)
    
    def _octree_quantization(self, img: np.ndarray, num_colors: int) -> np.ndarray:
        """Octree color quantization (simplified)."""
        # For now, fall back to K-means
        return self._kmeans_quantization(img, num_colors)
    
    def _uniform_quantization(self, img: np.ndarray, num_colors: int) -> np.ndarray:
        """Uniform color quantization."""
        # Calculate quantization levels
        levels = int(np.ceil(num_colors ** (1/3)))
        
        # Quantize each channel
        quantized = np.zeros_like(img)
        for i in range(3):
            channel = img[:, :, i]
            quantized[:, :, i] = np.round(channel * (levels - 1)) / (levels - 1)
        
        return quantized
    
    def _apply_dithering(self, original: np.ndarray, quantized: np.ndarray) -> np.ndarray:
        """Apply Floyd-Steinberg dithering."""
        result = quantized.copy()
        error = original - quantized
        
        height, width = original.shape[:2]
        
        for y in range(height - 1):
            for x in range(width - 1):
                # Floyd-Steinberg error distribution
                if x < width - 1:
                    result[y, x + 1] += error[y, x] * 7/16
                if y < height - 1:
                    result[y + 1, x] += error[y, x] * 5/16
                    if x > 0:
                        result[y + 1, x - 1] += error[y, x] * 3/16
                    if x < width - 1:
                        result[y + 1, x + 1] += error[y, x] * 1/16
        
        return np.clip(result, 0, 1)


class GamutMapper:
    """
    Map colors between different color gamuts.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "source_gamut": (["sRGB", "Adobe RGB", "DCI-P3", "Rec. 2020"], {"default": "sRGB"}),
                "target_gamut": (["sRGB", "Adobe RGB", "DCI-P3", "Rec. 2020"], {"default": "Adobe RGB"}),
                "mapping_method": (["Perceptual", "Relative", "Saturation", "Absolute"], {"default": "Perceptual"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "mapping_info")
    FUNCTION = "map_gamut"
    CATEGORY = "Color Tools/Advanced"
    
    def map_gamut(self, image: torch.Tensor, source_gamut: str, target_gamut: str,
                 mapping_method: str) -> Tuple[torch.Tensor, str]:
        """
        Map colors between different gamuts.
        """
        # Convert tensor to numpy
        if len(image.shape) == 4:
            img_np = image[0].numpy()
        else:
            img_np = image.numpy()
        
        # Ensure image is in [0, 1] range
        if img_np.max() > 1.0:
            img_np = img_np / 255.0
        
        # Map gamut
        mapped_img, info = self._map_gamut(img_np, source_gamut, target_gamut, mapping_method)
        
        # Convert back to tensor
        mapped_tensor = torch.from_numpy(mapped_img).float()
        if len(image.shape) == 4:
            mapped_tensor = mapped_tensor.unsqueeze(0)
        
        return mapped_tensor, info
    
    def _map_gamut(self, img: np.ndarray, source_gamut: str, target_gamut: str, 
                  mapping_method: str) -> Tuple[np.ndarray, str]:
        """Map colors between gamuts."""
        if source_gamut == target_gamut:
            return img, json.dumps({"message": "Source and target gamuts are the same"})
        
        # Define gamut primaries (simplified)
        gamuts = {
            "sRGB": {
                "red": [0.64, 0.33],
                "green": [0.30, 0.60],
                "blue": [0.15, 0.06],
                "white": [0.3127, 0.3290]
            },
            "Adobe RGB": {
                "red": [0.64, 0.33],
                "green": [0.21, 0.71],
                "blue": [0.15, 0.06],
                "white": [0.3127, 0.3290]
            },
            "DCI-P3": {
                "red": [0.68, 0.32],
                "green": [0.265, 0.69],
                "blue": [0.15, 0.06],
                "white": [0.3127, 0.3290]
            },
            "Rec. 2020": {
                "red": [0.708, 0.292],
                "green": [0.170, 0.797],
                "blue": [0.131, 0.046],
                "white": [0.3127, 0.3290]
            }
        }
        
        # For now, implement a simplified gamut mapping
        # In a real implementation, you would use proper color space conversion matrices
        
        if mapping_method == "Perceptual":
            # Perceptual mapping preserves visual appearance
            result = self._perceptual_mapping(img, source_gamut, target_gamut)
        elif mapping_method == "Relative":
            # Relative mapping preserves relative color relationships
            result = self._relative_mapping(img, source_gamut, target_gamut)
        elif mapping_method == "Saturation":
            # Saturation mapping preserves saturation
            result = self._saturation_mapping(img, source_gamut, target_gamut)
        else:  # Absolute
            # Absolute mapping preserves absolute color values
            result = self._absolute_mapping(img, source_gamut, target_gamut)
        
        info = {
            "source_gamut": source_gamut,
            "target_gamut": target_gamut,
            "mapping_method": mapping_method,
            "mapped_pixels": int(np.prod(img.shape[:2]))
        }
        
        return result, json.dumps(info)
    
    def _perceptual_mapping(self, img: np.ndarray, source: str, target: str) -> np.ndarray:
        """Perceptual gamut mapping."""
        # Simplified implementation - in reality, this would use proper color space conversion
        # For now, apply a simple scaling based on gamut size
        gamut_ratios = {"sRGB": 1.0, "Adobe RGB": 1.1, "DCI-P3": 1.2, "Rec. 2020": 1.3}
        
        source_ratio = gamut_ratios.get(source, 1.0)
        target_ratio = gamut_ratios.get(target, 1.0)
        
        scale_factor = target_ratio / source_ratio
        result = img * scale_factor
        
        return np.clip(result, 0, 1)
    
    def _relative_mapping(self, img: np.ndarray, source: str, target: str) -> np.ndarray:
        """Relative gamut mapping."""
        # Preserve relative color relationships
        return self._perceptual_mapping(img, source, target)
    
    def _saturation_mapping(self, img: np.ndarray, source: str, target: str) -> np.ndarray:
        """Saturation-preserving gamut mapping."""
        # Convert to HSV, adjust saturation, convert back
        hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        
        # Adjust saturation based on gamut
        gamut_ratios = {"sRGB": 1.0, "Adobe RGB": 1.1, "DCI-P3": 1.2, "Rec. 2020": 1.3}
        source_ratio = gamut_ratios.get(source, 1.0)
        target_ratio = gamut_ratios.get(target, 1.0)
        
        scale_factor = target_ratio / source_ratio
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * scale_factor, 0, 255)
        
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB) / 255.0
        return np.clip(result, 0, 1)
    
    def _absolute_mapping(self, img: np.ndarray, source: str, target: str) -> np.ndarray:
        """Absolute gamut mapping."""
        # Preserve absolute color values as much as possible
        return self._perceptual_mapping(img, source, target)


class ColorBlindSim:
    """
    Simulate different types of color blindness.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "color_blindness_type": (["Protanopia", "Deuteranopia", "Tritanopia", "Protanomaly", 
                                        "Deuteranomaly", "Tritanomaly", "Monochromacy"], 
                                       {"default": "Protanopia"}),
                "severity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "simulation_info")
    FUNCTION = "simulate_color_blindness"
    CATEGORY = "Color Tools/Advanced"
    
    def simulate_color_blindness(self, image: torch.Tensor, color_blindness_type: str, 
                                severity: float) -> Tuple[torch.Tensor, str]:
        """
        Simulate color blindness.
        """
        # Convert tensor to numpy
        if len(image.shape) == 4:
            img_np = image[0].numpy()
        else:
            img_np = image.numpy()
        
        # Ensure image is in [0, 1] range
        if img_np.max() > 1.0:
            img_np = img_np / 255.0
        
        # Simulate color blindness
        simulated_img, info = self._simulate_color_blindness(img_np, color_blindness_type, severity)
        
        # Convert back to tensor
        simulated_tensor = torch.from_numpy(simulated_img).float()
        if len(image.shape) == 4:
            simulated_tensor = simulated_tensor.unsqueeze(0)
        
        return simulated_tensor, info
    
    def _simulate_color_blindness(self, img: np.ndarray, blindness_type: str, severity: float) -> Tuple[np.ndarray, str]:
        """Simulate color blindness using transformation matrices."""
        # Define transformation matrices for different types of color blindness
        matrices = {
            "Protanopia": np.array([
                [0.567, 0.433, 0.0],
                [0.558, 0.442, 0.0],
                [0.0, 0.242, 0.758]
            ]),
            "Deuteranopia": np.array([
                [0.625, 0.375, 0.0],
                [0.7, 0.3, 0.0],
                [0.0, 0.3, 0.7]
            ]),
            "Tritanopia": np.array([
                [0.95, 0.05, 0.0],
                [0.0, 0.433, 0.567],
                [0.0, 0.475, 0.525]
            ]),
            "Protanomaly": np.array([
                [0.817, 0.183, 0.0],
                [0.333, 0.667, 0.0],
                [0.0, 0.125, 0.875]
            ]),
            "Deuteranomaly": np.array([
                [0.8, 0.2, 0.0],
                [0.258, 0.742, 0.0],
                [0.0, 0.142, 0.858]
            ]),
            "Tritanomaly": np.array([
                [0.967, 0.033, 0.0],
                [0.0, 0.733, 0.267],
                [0.0, 0.183, 0.817]
            ]),
            "Monochromacy": np.array([
                [0.299, 0.587, 0.114],
                [0.299, 0.587, 0.114],
                [0.299, 0.587, 0.114]
            ])
        }
        
        # Get transformation matrix
        transform_matrix = matrices.get(blindness_type, matrices["Protanopia"])
        
        # Apply severity (blend with identity matrix)
        identity = np.eye(3)
        transform_matrix = (1 - severity) * identity + severity * transform_matrix
        
        # Apply transformation
        result = np.zeros_like(img)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                pixel = img[i, j, :3]
                result[i, j, :3] = np.dot(transform_matrix, pixel)
        
        # Preserve alpha channel if present
        if img.shape[2] == 4:
            result[:, :, 3] = img[:, :, 3]
        
        # Clamp values
        result = np.clip(result, 0, 1)
        
        info = {
            "blindness_type": blindness_type,
            "severity": severity,
            "transformation_matrix": transform_matrix.tolist(),
            "description": f"Simulated {blindness_type} with {severity:.2f} severity"
        }
        
        return result, json.dumps(info)
