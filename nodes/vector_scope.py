"""
Vector Scope Node for ComfyUI Color Tools

This node generates an NTSC vector scope visualization from an input image tensor.
Based on the implementation from https://github.com/delphinus1024/vector_scope
"""

import torch
import numpy as np
import cv2
import math
from typing import Tuple

print("[Color Tools] 🎯 Initializing Vector Scope node...")
print("[Color Tools] ✅ Vector Scope node initialized successfully")


class VectorScope:
    """
    Generate an NTSC vector scope visualization from an input image tensor.
    """
    
    def __init__(self):
        # Scope size
        self.cols = 512
        self.rows = 512
        
        # Misc constant values
        self.margin = 10
        self.outerlinewidth = 2
        self.outerlinecolor = (128, 128, 128)
        self.small_tick_ratio = 0.98
        self.large_tick_ratio = 0.95
        self.tick_width = 2
        self.dot_radius = 2
        self.vector_len = 8
        self.al_out_angle = 4.
        self.al_out_rad = 0.05
        self.al_thick = 1
        
        # I/Q angle
        self.rad_iq = 33. * math.pi / 180.
        
        # NTSC constants
        self.a_r = 0.701
        self.b_r = -0.587
        self.c_r = -0.114
        
        self.a_b = -0.299
        self.b_b = -0.587
        self.c_b = 0.886
        
        self.maxval = 0
        self._calc_maxval()
    
    def _calc_maxval(self):
        """Calculate maximum value for normalization."""
        ry = self.a_r
        by = self.c_b
        self.maxval = self._calc_ec(ry, by)
    
    def _calc_ryby(self, r: float, g: float, b: float) -> Tuple[float, float]:
        """Calculate R-Y and B-Y values."""
        ry = self.a_r * r + self.b_r * g + self.c_r * b
        by = self.a_b * r + self.b_b * g + self.c_b * b
        return ry, by
    
    def _calc_ec(self, ry: float, by: float) -> float:
        """Calculate eccentricity."""
        ec = math.sqrt(((ry / 1.14) ** 2) + ((by / 2.03) ** 2))
        return ec
    
    def _calc_theta(self, ry: float, by: float) -> float:
        """Calculate theta angle."""
        theta = math.atan2(ry / 1.14, by / 2.03)
        return theta
    
    def _calc_transform(self, r: float, g: float, b: float) -> Tuple[float, float]:
        """Calculate transform values."""
        ry, by = self._calc_ryby(r, g, b)
        ec = self._calc_ec(ry, by)
        theta = self._calc_theta(ry, by)
        return ec, theta
    
    def _pole2cart(self, center_x: int, center_y: int, theta: float, radius: float) -> Tuple[int, int]:
        """Convert polar coordinates to cartesian."""
        x = np.float64(center_x) + np.float64(radius) * math.cos(theta)
        y = np.float64(center_y) - np.float64(radius) * math.sin(theta)
        return int(x), int(y)
    
    def _rgb2cart(self, center_x: int, center_y: int, radius: float, r: float, g: float, b: float, angle_delta: float) -> Tuple[int, int]:
        """Convert RGB to cartesian coordinates."""
        ec, theta = self._calc_transform(r, g, b)
        x, y = self._pole2cart(center_x, center_y, theta + (angle_delta * math.pi / 180.), ec / self.maxval * radius)
        return x, y
    
    def _draw_allowance(self, result_img: np.ndarray, center_x: int, center_y: int, radius: int, v: list, c: str, font_type):
        """Draw allowance lines for color vectors."""
        # Inner lines
        x, y = self._rgb2cart(center_x, center_y, radius * (1. - 0.0474), v[0], v[1], v[2], 0.)
        x2, y2 = self._rgb2cart(center_x, center_y, radius * (1. + 0.0474), v[0], v[1], v[2], 0.)
        cv2.line(result_img, (x, y), (x2, y2), self.outerlinecolor, self.al_thick)
        
        x, y = self._rgb2cart(center_x, center_y, radius * (1. - 0.0474), v[0], v[1], v[2], 2.5)
        x2, y2 = self._rgb2cart(center_x, center_y, radius * (1. + 0.0474), v[0], v[1], v[2], 2.5)
        cv2.line(result_img, (x, y), (x2, y2), self.outerlinecolor, self.al_thick)
        
        x, y = self._rgb2cart(center_x, center_y, radius * (1. - 0.0474), v[0], v[1], v[2], -2.5)
        x2, y2 = self._rgb2cart(center_x, center_y, radius * (1. + 0.0474), v[0], v[1], v[2], -2.5)
        cv2.line(result_img, (x, y), (x2, y2), self.outerlinecolor, self.al_thick)
        
        x, y = self._rgb2cart(center_x, center_y, radius * (1.), v[0], v[1], v[2], -2.5)
        x2, y2 = self._rgb2cart(center_x, center_y, radius * (1.), v[0], v[1], v[2], 2.5)
        cv2.line(result_img, (x, y), (x2, y2), self.outerlinecolor, self.al_thick)
        
        x, y = self._rgb2cart(center_x, center_y, radius * (1. - 0.0474), v[0], v[1], v[2], -2.5)
        x2, y2 = self._rgb2cart(center_x, center_y, radius * (1. - 0.0474), v[0], v[1], v[2], 2.5)
        cv2.line(result_img, (x, y), (x2, y2), self.outerlinecolor, self.al_thick)
        
        x, y = self._rgb2cart(center_x, center_y, radius * (1. + 0.0474), v[0], v[1], v[2], -2.5)
        x2, y2 = self._rgb2cart(center_x, center_y, radius * (1. + 0.0474), v[0], v[1], v[2], 2.5)
        cv2.line(result_img, (x, y), (x2, y2), self.outerlinecolor, self.al_thick)
        
        # Outer lines
        x, y = self._rgb2cart(center_x, center_y, radius * (1. - 0.2), v[0], v[1], v[2], -10.)
        x2, y2 = self._rgb2cart(center_x, center_y, radius * (1. - 0.2), v[0], v[1], v[2], -10. + self.al_out_angle)
        cv2.line(result_img, (x, y), (x2, y2), self.outerlinecolor, self.al_thick)
        
        x, y = self._rgb2cart(center_x, center_y, radius * (1. - 0.2), v[0], v[1], v[2], -10.)
        x2, y2 = self._rgb2cart(center_x, center_y, radius * (1. - 0.2 + self.al_out_rad), v[0], v[1], v[2], -10.)
        cv2.line(result_img, (x, y), (x2, y2), self.outerlinecolor, self.al_thick)
        
        x, y = self._rgb2cart(center_x, center_y, radius * (1. - 0.2), v[0], v[1], v[2], 10.)
        x2, y2 = self._rgb2cart(center_x, center_y, radius * (1. - 0.2), v[0], v[1], v[2], 10. - self.al_out_angle)
        cv2.line(result_img, (x, y), (x2, y2), self.outerlinecolor, self.al_thick)
        
        x, y = self._rgb2cart(center_x, center_y, radius * (1. - 0.2), v[0], v[1], v[2], 10.)
        x2, y2 = self._rgb2cart(center_x, center_y, radius * (1. - 0.2 + self.al_out_rad), v[0], v[1], v[2], 10.)
        cv2.line(result_img, (x, y), (x2, y2), self.outerlinecolor, self.al_thick)
        
        x, y = self._rgb2cart(center_x, center_y, radius * (1. + 0.2), v[0], v[1], v[2], -10.)
        x2, y2 = self._rgb2cart(center_x, center_y, radius * (1. + 0.2), v[0], v[1], v[2], -10. + self.al_out_angle)
        cv2.line(result_img, (x, y), (x2, y2), self.outerlinecolor, self.al_thick)
        
        x, y = self._rgb2cart(center_x, center_y, radius * (1. + 0.2), v[0], v[1], v[2], -10.)
        x2, y2 = self._rgb2cart(center_x, center_y, radius * (1. + 0.2 - self.al_out_rad), v[0], v[1], v[2], -10.)
        cv2.line(result_img, (x, y), (x2, y2), self.outerlinecolor, self.al_thick)
        
        x, y = self._rgb2cart(center_x, center_y, radius * (1. + 0.2), v[0], v[1], v[2], 10.)
        x2, y2 = self._rgb2cart(center_x, center_y, radius * (1. + 0.2), v[0], v[1], v[2], 10. - self.al_out_angle)
        cv2.line(result_img, (x, y), (x2, y2), self.outerlinecolor, self.al_thick)
        
        x, y = self._rgb2cart(center_x, center_y, radius * (1. + 0.2), v[0], v[1], v[2], 10.)
        x2, y2 = self._rgb2cart(center_x, center_y, radius * (1. + 0.2 - self.al_out_rad), v[0], v[1], v[2], 10.)
        cv2.line(result_img, (x, y), (x2, y2), self.outerlinecolor, self.al_thick)
        
        x, y = self._rgb2cart(center_x, center_y, radius, v[0], v[1], v[2], 0.)
        cv2.putText(result_img, c, (x + self.vector_len, y - self.vector_len), font_type, 1, self.outerlinecolor, 1, 16)
    
    def _draw_background(self, result_img: np.ndarray, center_x: int, center_y: int, radius: int):
        """Draw the background grid and reference lines."""
        # Outer circle and tick
        cv2.circle(result_img, (center_x, center_y), radius, self.outerlinecolor, self.outerlinewidth, 8)
        
        radius_div = radius / 5.
        for i in range(1, 5):
            cv2.circle(result_img, (center_x, center_y), int(float(i) * radius_div), self.outerlinecolor, 1)
        
        cv2.line(result_img, (center_x - radius, center_y), (center_x + radius, center_y), self.outerlinecolor, 1)
        cv2.line(result_img, (center_x, center_y - radius), (center_x, center_y + radius), self.outerlinecolor, 1)
        
        for i in range(0, 360, 2):
            theta = np.float64(i) / 180 * math.pi
            if (i % 10) == 0:
                r_s = np.float64(radius) * self.large_tick_ratio
            else:
                r_s = np.float64(radius) * self.small_tick_ratio
            
            xs, ys = self._pole2cart(center_x, center_y, theta, radius)
            xe, ye = self._pole2cart(center_x, center_y, theta, r_s)
            cv2.line(result_img, (xs, ys), (xe, ye), self.outerlinecolor, self.tick_width)
        
        # I/Q lines
        xs, ys = self._pole2cart(center_x, center_y, self.rad_iq, radius)
        xe, ye = self._pole2cart(center_x, center_y, math.pi + self.rad_iq, radius)
        cv2.line(result_img, (xs, ys), (xe, ye), self.outerlinecolor, 1)
        
        xs, ys = self._pole2cart(center_x, center_y, math.pi * 0.5 + self.rad_iq, radius)
        xe, ye = self._pole2cart(center_x, center_y, math.pi * 1.5 + self.rad_iq, radius)
        cv2.line(result_img, (xs, ys), (xe, ye), self.outerlinecolor, 1)
        
        # Draw vectors
        vec = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0],
               [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
        col_name = ["B", "G", "CY", "R", "MG", "YL"]
        
        font_type = cv2.FONT_HERSHEY_SIMPLEX
        
        for v, c in zip(vec, col_name):
            self._draw_allowance(result_img, center_x, center_y, radius, v, c, font_type)
    
    def _draw_pixel(self, result_img: np.ndarray, center_x: int, center_y: int, radius: int, bgr: np.ndarray):
        """Draw a single pixel on the vectorscope."""
        col = (int(bgr[0] * 255.), int(bgr[1] * 255.), int(bgr[2] * 255.))
        x, y = self._rgb2cart(center_x, center_y, radius, bgr[2], bgr[1], bgr[0], 0.)
        cv2.circle(result_img, (x, y), self.dot_radius, col, -1)
    
    def generate_vectorscope(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Generate vector scope visualization from image tensor.
        
        Args:
            image_tensor: Input image tensor with shape [B, H, W, C] or [H, W, C]
        
        Returns:
            Vector scope image as tensor with shape [1, H, W, C]
        """
        # Convert tensor to numpy array
        if len(image_tensor.shape) == 4:
            # Remove batch dimension if present
            image_array = image_tensor[0].cpu().numpy()
        else:
            image_array = image_tensor.cpu().numpy()
        
        # Ensure values are in [0, 1] range
        if image_array.max() > 1.0:
            image_array = image_array / 255.0
        
        # Convert to BGR for OpenCV
        if image_array.shape[2] == 3:
            # Assume RGB, convert to BGR
            image_array = image_array[:, :, [2, 1, 0]]
        
        height, width, depth = image_array.shape
        
        if depth != 3:
            raise ValueError("Image must have 3 channels (RGB/BGR)")
        
        # Create result image with black background
        center_x = int(self.cols / 2)
        center_y = int(self.rows / 2)
        radius = int(self.rows / 2 - self.margin)
        result_img = np.zeros((self.rows, self.cols, 3), np.uint8)
        
        # Reshape to 1D for processing
        lin_image = (np.reshape(image_array, (height * width, 3))).astype(np.float64)
        
        # Plot all pixels
        for bgr in lin_image:
            self._draw_pixel(result_img, center_x, center_y, radius, bgr)
        
        # Draw background
        self._draw_background(result_img, center_x, center_y, radius)
        
        # Convert back to tensor
        result_tensor = torch.from_numpy(result_img).float() / 255.0
        result_tensor = result_tensor.unsqueeze(0)  # Add batch dimension
        
        return result_tensor


# ComfyUI Node Implementation
class VectorScopeNode:
    """
    ComfyUI node for generating NTSC vector scope visualization.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "scope_size": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 64}),
                "dot_radius": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("vectorscope",)
    FUNCTION = "generate_vectorscope"
    CATEGORY = "Image/Color"
    
    def generate_vectorscope(self, image: torch.Tensor, scope_size: int = 512, dot_radius: int = 2):
        """
        Generate vector scope visualization from input image.
        
        Args:
            image: Input image tensor
            scope_size: Size of the vectorscope image (square)
            dot_radius: Radius of dots in the vectorscope
        
        Returns:
            Vector scope image tensor
        """
        print(f"[Color Tools] 🎯 Generating vector scope: {image.shape} -> {scope_size}x{scope_size}")
        
        # Create vectorscope generator
        vs = VectorScope()
        vs.cols = scope_size
        vs.rows = scope_size
        vs.dot_radius = dot_radius
        vs.margin = scope_size // 50  # Scale margin with size
        vs._calc_maxval()  # Recalculate maxval for new size
        
        # Generate vectorscope
        result = vs.generate_vectorscope(image)
        
        print(f"[Color Tools] ✅ Vector scope generated successfully")
        return (result,)


# ComfyUI node registry
NODE_CLASS_MAPPINGS = {
    "VectorScope": VectorScopeNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VectorScope": "Vector Scope"
}
