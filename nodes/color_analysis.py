"""
Color Analysis Nodes

This module contains nodes for analyzing colors in images, extracting
dominant colors, generating palettes, and performing color similarity analysis.
"""

import torch
import numpy as np
import cv2
from sklearn.cluster import KMeans
from collections import Counter
import json
from typing import Tuple, Dict, Any, List, Optional

class DominantColors:
    """
    Extract dominant colors from an image using K-means clustering.
    Works with both file paths and image tensors.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_mode": (["file", "tensor"], {"default": "tensor"}),
                "num_colors": ("INT", {"default": 5, "min": 1, "max": 20, "step": 1}),
                "color_format": (["RGB", "HSV", "HEX"], {"default": "RGB"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image_path": ("STRING", {"default": "", "multiline": False}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("dominant_colors", "color_percentages")
    FUNCTION = "extract_dominant_colors"
    CATEGORY = "Color Tools/Analysis"
    
    def extract_dominant_colors(self, input_mode: str, num_colors: int, color_format: str, 
                               image: torch.Tensor = None, image_path: str = "") -> Tuple[str, str]:
        """
        Extract dominant colors from the image.
        Supports both file paths and image tensors.
        """
        if input_mode == "file":
            return self._extract_from_file(image_path, num_colors, color_format)
        else:
            return self._extract_from_tensor(image, num_colors, color_format)
    
    def _extract_from_file(self, image_path: str, num_colors: int, color_format: str) -> Tuple[str, str]:
        """Extract colors from file"""
        if not image_path.strip():
            raise ValueError("Image path required when input_mode is 'file'")
        
        # Load image from file
        img_array = self._load_image_from_path(image_path)
        return self._extract_colors(img_array, num_colors, color_format)
    
    def _extract_from_tensor(self, image: torch.Tensor, num_colors: int, color_format: str) -> Tuple[str, str]:
        """Extract colors from tensor"""
        if image is None:
            raise ValueError("Image tensor required when input_mode is 'tensor'")
        
        # Convert tensor to numpy
        img_array = self._tensor_to_array(image)
        return self._extract_colors(img_array, num_colors, color_format)
    
    def _load_image_from_path(self, image_path: str) -> np.ndarray:
        """Load image from file path"""
        from PIL import Image
        pil_image = Image.open(image_path)
        img_array = np.array(pil_image) / 255.0
        return img_array
    
    def _tensor_to_array(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert ComfyUI tensor to numpy array"""
        if len(tensor.shape) == 4:
            return tensor[0].cpu().numpy()
        else:
            return tensor.cpu().numpy()
    
    def _extract_colors(self, img_array: np.ndarray, num_colors: int, color_format: str) -> Tuple[str, str]:
        """Core color extraction logic"""
        # Ensure image is in [0, 1] range
        if img_array.max() > 1.0:
            img_array = img_array / 255.0
        
        # Extract dominant colors
        colors, percentages = self._extract_colors_internal(img_array, num_colors, color_format)
        
        return colors, percentages
    
    def _extract_colors_internal(self, img: np.ndarray, num_colors: int, color_format: str) -> Tuple[str, str]:
        """Extract dominant colors using K-means clustering."""
        # Reshape image to list of pixels
        pixels = img.reshape(-1, 3)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get cluster centers and labels
        colors = kmeans.cluster_centers_
        labels = kmeans.labels_
        
        # Calculate percentages
        label_counts = Counter(labels)
        total_pixels = len(labels)
        percentages = [count / total_pixels for count in label_counts.values()]
        
        # Convert colors to desired format
        if color_format == "RGB":
            colors_str = json.dumps(colors.tolist())
        elif color_format == "HSV":
            hsv_colors = []
            for color in colors:
                hsv = cv2.cvtColor(np.uint8([[color * 255]]), cv2.COLOR_RGB2HSV)[0][0]
                hsv_colors.append(hsv.tolist())
            colors_str = json.dumps(hsv_colors)
        elif color_format == "HEX":
            hex_colors = []
            for color in colors:
                r, g, b = (color * 255).astype(int)
                hex_color = f"#{r:02x}{g:02x}{b:02x}"
                hex_colors.append(hex_color)
            colors_str = json.dumps(hex_colors)
        
        percentages_str = json.dumps(percentages)
        
        return colors_str, percentages_str


class ColorHistogram:
    """
    Generate color histograms for analysis.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "bins": ("INT", {"default": 256, "min": 32, "max": 512, "step": 32}),
                "histogram_type": (["RGB", "HSV", "LAB"], {"default": "RGB"}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("histogram_data", "statistics")
    FUNCTION = "generate_histogram"
    CATEGORY = "Color Tools/Analysis"
    
    def generate_histogram(self, image: torch.Tensor, bins: int, histogram_type: str) -> Tuple[str, str]:
        """
        Generate color histogram data.
        """
        # Convert tensor to numpy
        if len(image.shape) == 4:
            img_np = image[0].numpy()
        else:
            img_np = image.numpy()
        
        # Ensure image is in [0, 1] range
        if img_np.max() > 1.0:
            img_np = img_np / 255.0
        
        # Generate histogram
        histogram_data, statistics = self._generate_histogram(img_np, bins, histogram_type)
        
        return histogram_data, statistics
    
    def _generate_histogram(self, img: np.ndarray, bins: int, hist_type: str) -> Tuple[str, str]:
        """Generate color histogram."""
        # Convert to appropriate color space
        if hist_type == "RGB":
            channels = [img[:, :, 0], img[:, :, 1], img[:, :, 2]]
            channel_names = ["Red", "Green", "Blue"]
        elif hist_type == "HSV":
            hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
            channels = [hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]]
            channel_names = ["Hue", "Saturation", "Value"]
        elif hist_type == "LAB":
            lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
            channels = [lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]]
            channel_names = ["L", "A", "B"]
        
        # Calculate histograms
        histograms = []
        statistics = {}
        
        for i, (channel, name) in enumerate(zip(channels, channel_names)):
            if hist_type == "RGB":
                hist, _ = np.histogram(channel, bins=bins, range=(0, 1))
            else:
                hist, _ = np.histogram(channel, bins=bins, range=(0, 255))
            
            histograms.append(hist.tolist())
            
            # Calculate statistics
            statistics[name] = {
                "mean": float(np.mean(channel)),
                "std": float(np.std(channel)),
                "min": float(np.min(channel)),
                "max": float(np.max(channel)),
                "median": float(np.median(channel))
            }
        
        histogram_data = json.dumps({
            "channels": channel_names,
            "histograms": histograms,
            "bins": bins
        })
        
        statistics_str = json.dumps(statistics)
        
        return histogram_data, statistics_str


class ColorPalette:
    """
    Generate comprehensive color palettes from images.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "palette_size": ("INT", {"default": 8, "min": 3, "max": 32, "step": 1}),
                "palette_type": (["K-means", "Median Cut", "Octree"], {"default": "K-means"}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("palette", "palette_info")
    FUNCTION = "generate_palette"
    CATEGORY = "Color Tools/Analysis"
    
    def generate_palette(self, image: torch.Tensor, palette_size: int, palette_type: str) -> Tuple[str, str]:
        """
        Generate color palette from image.
        """
        # Convert tensor to numpy
        if len(image.shape) == 4:
            img_np = image[0].numpy()
        else:
            img_np = image.numpy()
        
        # Ensure image is in [0, 1] range
        if img_np.max() > 1.0:
            img_np = img_np / 255.0
        
        # Generate palette
        palette, palette_info = self._generate_palette(img_np, palette_size, palette_type)
        
        return palette, palette_info
    
    def _generate_palette(self, img: np.ndarray, palette_size: int, palette_type: str) -> Tuple[str, str]:
        """Generate color palette using specified method."""
        # Reshape image to list of pixels
        pixels = img.reshape(-1, 3)
        
        if palette_type == "K-means":
            palette_colors = self._kmeans_palette(pixels, palette_size)
        elif palette_type == "Median Cut":
            palette_colors = self._median_cut_palette(pixels, palette_size)
        elif palette_type == "Octree":
            palette_colors = self._octree_palette(pixels, palette_size)
        else:
            palette_colors = self._kmeans_palette(pixels, palette_size)
        
        # Create palette data
        palette_data = {
            "colors": palette_colors.tolist(),
            "size": palette_size,
            "method": palette_type,
            "hex_colors": [f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" 
                          for r, g, b in palette_colors]
        }
        
        # Create palette info
        info = {
            "total_colors": len(palette_colors),
            "method": palette_type,
            "color_diversity": float(np.std(palette_colors)),
            "brightness_range": [float(np.min(palette_colors)), float(np.max(palette_colors))]
        }
        
        return json.dumps(palette_data), json.dumps(info)
    
    def _kmeans_palette(self, pixels: np.ndarray, n_colors: int) -> np.ndarray:
        """Generate palette using K-means clustering."""
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        return kmeans.cluster_centers_
    
    def _median_cut_palette(self, pixels: np.ndarray, n_colors: int) -> np.ndarray:
        """Generate palette using median cut algorithm."""
        # Simplified median cut implementation
        def median_cut(pixels, depth):
            if depth == 0 or len(pixels) == 0:
                return [np.mean(pixels, axis=0)]
            
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
            
            return left + right
        
        # Calculate depth needed
        depth = int(np.log2(n_colors))
        colors = median_cut(pixels, depth)
        
        # Limit to requested number of colors
        return np.array(colors[:n_colors])
    
    def _octree_palette(self, pixels: np.ndarray, n_colors: int) -> np.ndarray:
        """Generate palette using octree quantization."""
        # Simplified octree implementation
        # For now, fall back to K-means
        return self._kmeans_palette(pixels, n_colors)


class ColorSimilarity:
    """
    Find similar colors in an image based on color distance.
    Works with both file paths and image tensors.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_mode": (["file", "tensor"], {"default": "tensor"}),
                "target_color": ("STRING", {"default": "#FF0000"}),
                "similarity_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "color_space": (["RGB", "HSV", "LAB"], {"default": "LAB"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image_path": ("STRING", {"default": "", "multiline": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("mask", "similarity_info")
    FUNCTION = "find_similar_colors"
    CATEGORY = "Color Tools/Analysis"
    
    def find_similar_colors(self, input_mode: str, target_color: str, 
                           similarity_threshold: float, color_space: str, 
                           image: torch.Tensor = None, image_path: str = "") -> Tuple[torch.Tensor, str]:
        """
        Find colors similar to the target color.
        Supports both file paths and image tensors.
        """
        if input_mode == "file":
            return self._find_from_file(image_path, target_color, similarity_threshold, color_space)
        else:
            return self._find_from_tensor(image, target_color, similarity_threshold, color_space)
    
    def _find_from_file(self, image_path: str, target_color: str, 
                       similarity_threshold: float, color_space: str) -> Tuple[torch.Tensor, str]:
        """Find similar colors from file"""
        if not image_path.strip():
            raise ValueError("Image path required when input_mode is 'file'")
        
        # Load image from file
        img_array = self._load_image_from_path(image_path)
        mask, info = self._find_similar_colors(img_array, target_color, similarity_threshold, color_space)
        
        # Convert mask back to tensor
        return self._array_to_tensor(mask), info
    
    def _find_from_tensor(self, image: torch.Tensor, target_color: str, 
                         similarity_threshold: float, color_space: str) -> Tuple[torch.Tensor, str]:
        """Find similar colors from tensor"""
        if image is None:
            raise ValueError("Image tensor required when input_mode is 'tensor'")
        
        # Convert tensor to numpy
        img_array = self._tensor_to_array(image)
        mask, info = self._find_similar_colors(img_array, target_color, similarity_threshold, color_space)
        
        # Convert mask back to tensor
        return self._array_to_tensor(mask), info
    
    def _load_image_from_path(self, image_path: str) -> np.ndarray:
        """Load image from file path"""
        from PIL import Image
        pil_image = Image.open(image_path)
        img_array = np.array(pil_image) / 255.0
        return img_array
    
    def _tensor_to_array(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert ComfyUI tensor to numpy array"""
        if len(tensor.shape) == 4:
            return tensor[0].cpu().numpy()
        else:
            return tensor.cpu().numpy()
    
    def _array_to_tensor(self, array: np.ndarray) -> torch.Tensor:
        """Convert numpy array to ComfyUI tensor"""
        if len(array.shape) == 3:
            array = array[np.newaxis, ...]
        return torch.from_numpy(array).float()
    
    def _find_similar_colors(self, img_array: np.ndarray, target_color: str, 
                           similarity_threshold: float, color_space: str) -> Tuple[np.ndarray, str]:
        """Core similarity finding logic"""
        # Ensure image is in [0, 1] range
        if img_array.max() > 1.0:
            img_array = img_array / 255.0
        
        # Parse target color
        target_rgb = self._parse_color(target_color)
        
        # Find similar colors
        mask, info = self._find_similar_colors_internal(img_array, target_rgb, similarity_threshold, color_space)
        
        return mask, info
    
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
    
    def _find_similar_colors(self, img: np.ndarray, target_rgb: np.ndarray, 
                           threshold: float, color_space: str) -> Tuple[np.ndarray, str]:
        """Find similar colors in the image."""
        # Convert to target color space
        if color_space == "RGB":
            img_space = img
            target_space = target_rgb
        elif color_space == "HSV":
            img_space = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV) / 255.0
            target_space = cv2.cvtColor((target_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2HSV) / 255.0
        elif color_space == "LAB":
            img_space = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB) / 255.0
            target_space = cv2.cvtColor((target_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2LAB) / 255.0
        
        # Calculate color distances
        distances = np.sqrt(np.sum((img_space - target_space) ** 2, axis=2))
        
        # Create similarity mask
        similarity_mask = distances <= threshold
        
        # Calculate statistics
        total_pixels = img.shape[0] * img.shape[1]
        similar_pixels = np.sum(similarity_mask)
        similarity_percentage = (similar_pixels / total_pixels) * 100
        
        info = {
            "target_color": target_rgb.tolist(),
            "color_space": color_space,
            "threshold": threshold,
            "similar_pixels": int(similar_pixels),
            "total_pixels": int(total_pixels),
            "similarity_percentage": float(similarity_percentage)
        }
        
        return similarity_mask.astype(np.float32), json.dumps(info)


class ColorHarmony:
    """
    Analyze color harmony and relationships in images.
    Works with both file paths and image tensors.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_mode": (["file", "tensor"], {"default": "tensor"}),
                "harmony_type": (["Complementary", "Triadic", "Analogous", "Split-Complementary"], 
                               {"default": "Complementary"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image_path": ("STRING", {"default": "", "multiline": False}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("harmony_analysis", "color_relationships")
    FUNCTION = "analyze_color_harmony"
    CATEGORY = "Color Tools/Analysis"
    
    def analyze_color_harmony(self, input_mode: str, harmony_type: str, 
                            image: torch.Tensor = None, image_path: str = "") -> Tuple[str, str]:
        """
        Analyze color harmony in the image.
        Supports both file paths and image tensors.
        """
        if input_mode == "file":
            return self._analyze_from_file(image_path, harmony_type)
        else:
            return self._analyze_from_tensor(image, harmony_type)
    
    def _analyze_from_file(self, image_path: str, harmony_type: str) -> Tuple[str, str]:
        """Analyze harmony from file"""
        if not image_path.strip():
            raise ValueError("Image path required when input_mode is 'file'")
        
        # Load image from file
        img_array = self._load_image_from_path(image_path)
        return self._analyze_harmony(img_array, harmony_type)
    
    def _analyze_from_tensor(self, image: torch.Tensor, harmony_type: str) -> Tuple[str, str]:
        """Analyze harmony from tensor"""
        if image is None:
            raise ValueError("Image tensor required when input_mode is 'tensor'")
        
        # Convert tensor to numpy
        img_array = self._tensor_to_array(image)
        return self._analyze_harmony(img_array, harmony_type)
    
    def _load_image_from_path(self, image_path: str) -> np.ndarray:
        """Load image from file path"""
        from PIL import Image
        pil_image = Image.open(image_path)
        img_array = np.array(pil_image) / 255.0
        return img_array
    
    def _tensor_to_array(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert ComfyUI tensor to numpy array"""
        if len(tensor.shape) == 4:
            return tensor[0].cpu().numpy()
        else:
            return tensor.cpu().numpy()
    
    def _analyze_harmony(self, img_array: np.ndarray, harmony_type: str) -> Tuple[str, str]:
        """Core harmony analysis logic"""
        # Ensure image is in [0, 1] range
        if img_array.max() > 1.0:
            img_array = img_array / 255.0
        
        # Analyze color harmony
        harmony_analysis, color_relationships = self._analyze_harmony_internal(img_array, harmony_type)
        
        return harmony_analysis, color_relationships
    
    def _analyze_harmony_internal(self, img: np.ndarray, harmony_type: str) -> Tuple[str, str]:
        """Analyze color harmony."""
        # Convert to HSV for hue analysis
        hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        hues = hsv[:, :, 0]
        
        # Calculate hue distribution
        hue_hist, _ = np.histogram(hues, bins=36, range=(0, 180))
        
        # Find dominant hues
        dominant_hues = np.argsort(hue_hist)[-3:][::-1] * 5  # Convert to actual hue values
        
        # Analyze harmony based on type
        if harmony_type == "Complementary":
            analysis = self._analyze_complementary(dominant_hues)
        elif harmony_type == "Triadic":
            analysis = self._analyze_triadic(dominant_hues)
        elif harmony_type == "Analogous":
            analysis = self._analyze_analogous(dominant_hues)
        elif harmony_type == "Split-Complementary":
            analysis = self._analyze_split_complementary(dominant_hues)
        else:
            analysis = {"type": "Unknown", "score": 0.0}
        
        # Create color relationships
        relationships = {
            "dominant_hues": dominant_hues.tolist(),
            "hue_distribution": hue_hist.tolist(),
            "harmony_type": harmony_type,
            "analysis": analysis
        }
        
        return json.dumps(analysis), json.dumps(relationships)
    
    def _analyze_complementary(self, hues: np.ndarray) -> Dict[str, Any]:
        """Analyze complementary color harmony."""
        if len(hues) < 2:
            return {"type": "Complementary", "score": 0.0, "description": "Insufficient color data"}
        
        # Check for complementary colors (180 degrees apart)
        hue_diff = abs(hues[0] - hues[1])
        if hue_diff > 90:
            hue_diff = 180 - hue_diff
        
        score = 1.0 - (hue_diff / 90.0)
        
        return {
            "type": "Complementary",
            "score": float(score),
            "description": f"Complementary harmony score: {score:.2f}"
        }
    
    def _analyze_triadic(self, hues: np.ndarray) -> Dict[str, Any]:
        """Analyze triadic color harmony."""
        if len(hues) < 3:
            return {"type": "Triadic", "score": 0.0, "description": "Insufficient color data"}
        
        # Check for triadic colors (120 degrees apart)
        hue_diffs = []
        for i in range(len(hues)):
            for j in range(i + 1, len(hues)):
                diff = abs(hues[i] - hues[j])
                if diff > 60:
                    diff = 120 - diff
                hue_diffs.append(diff)
        
        avg_diff = np.mean(hue_diffs)
        score = 1.0 - (avg_diff / 60.0)
        
        return {
            "type": "Triadic",
            "score": float(score),
            "description": f"Triadic harmony score: {score:.2f}"
        }
    
    def _analyze_analogous(self, hues: np.ndarray) -> Dict[str, Any]:
        """Analyze analogous color harmony."""
        if len(hues) < 2:
            return {"type": "Analogous", "score": 0.0, "description": "Insufficient color data"}
        
        # Check for analogous colors (close together on color wheel)
        hue_diffs = []
        for i in range(len(hues)):
            for j in range(i + 1, len(hues)):
                diff = abs(hues[i] - hues[j])
                if diff > 90:
                    diff = 180 - diff
                hue_diffs.append(diff)
        
        avg_diff = np.mean(hue_diffs)
        score = 1.0 - (avg_diff / 30.0)  # Analogous colors should be within 30 degrees
        
        return {
            "type": "Analogous",
            "score": float(score),
            "description": f"Analogous harmony score: {score:.2f}"
        }
    
    def _analyze_split_complementary(self, hues: np.ndarray) -> Dict[str, Any]:
        """Analyze split-complementary color harmony."""
        if len(hues) < 3:
            return {"type": "Split-Complementary", "score": 0.0, "description": "Insufficient color data"}
        
        # Check for split-complementary (one color and two colors adjacent to its complement)
        # This is a simplified analysis
        score = 0.5  # Placeholder for more complex analysis
        
        return {
            "type": "Split-Complementary",
            "score": float(score),
            "description": f"Split-complementary harmony score: {score:.2f}"
        }
