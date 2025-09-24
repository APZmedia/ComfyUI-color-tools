"""
Color Tools Utility Functions

Common utility functions for handling dual input (file/tensor) nodes.
This module provides reusable functions to reduce code duplication.
"""

import torch
import numpy as np
from PIL import Image
from typing import Tuple, Union


class ColorInputHandler:
    """
    Utility class for handling dual input (file/tensor) operations.
    """
    
    @staticmethod
    def load_image_from_path(image_path: str) -> np.ndarray:
        """Load image from file path and return as numpy array."""
        if not image_path.strip():
            raise ValueError("Image path is required")
        
        pil_image = Image.open(image_path)
        img_array = np.array(pil_image) / 255.0
        return img_array
    
    @staticmethod
    def tensor_to_array(tensor: torch.Tensor) -> np.ndarray:
        """Convert ComfyUI tensor to numpy array."""
        if len(tensor.shape) == 4:
            return tensor[0].cpu().numpy()
        else:
            return tensor.cpu().numpy()
    
    @staticmethod
    def array_to_tensor(array: np.ndarray) -> torch.Tensor:
        """Convert numpy array to ComfyUI tensor."""
        if len(array.shape) == 3:
            array = array[np.newaxis, ...]
        return torch.from_numpy(array).float()
    
    @staticmethod
    def normalize_image_array(img_array: np.ndarray) -> np.ndarray:
        """Ensure image array is in [0, 1] range."""
        if img_array.max() > 1.0:
            img_array = img_array / 255.0
        return img_array
    
    @staticmethod
    def process_dual_input(input_mode: str, image: torch.Tensor = None, 
                          image_path: str = "", process_func=None, *args, **kwargs):
        """
        Generic function to handle dual input processing.
        
        Args:
            input_mode: "file" or "tensor"
            image: Image tensor (for tensor mode)
            image_path: Image file path (for file mode)
            process_func: Function to process the image array
            *args, **kwargs: Additional arguments for process_func
        
        Returns:
            Result from process_func
        """
        if input_mode == "file":
            if not image_path.strip():
                raise ValueError("Image path required when input_mode is 'file'")
            img_array = ColorInputHandler.load_image_from_path(image_path)
        else:
            if image is None:
                raise ValueError("Image tensor required when input_mode is 'tensor'")
            img_array = ColorInputHandler.tensor_to_array(image)
        
        # Normalize image array
        img_array = ColorInputHandler.normalize_image_array(img_array)
        
        # Process the image
        return process_func(img_array, *args, **kwargs)
    
    @staticmethod
    def process_dual_input_with_tensor_output(input_mode: str, image: torch.Tensor = None, 
                                            image_path: str = "", process_func=None, *args, **kwargs):
        """
        Generic function to handle dual input processing with tensor output.
        
        Args:
            input_mode: "file" or "tensor"
            image: Image tensor (for tensor mode)
            image_path: Image file path (for file mode)
            process_func: Function to process the image array
            *args, **kwargs: Additional arguments for process_func
        
        Returns:
            Processed image as tensor
        """
        result_array = ColorInputHandler.process_dual_input(
            input_mode, image, image_path, process_func, *args, **kwargs
        )
        return ColorInputHandler.array_to_tensor(result_array)


def create_dual_input_types():
    """
    Create standard input types for dual input nodes.
    
    Returns:
        Dictionary with standard input type configuration
    """
    return {
        "required": {
            "input_mode": (["file", "tensor"], {"default": "tensor"}),
        },
        "optional": {
            "image": ("IMAGE",),
            "image_path": ("STRING", {"default": "", "multiline": False}),
        }
    }


def create_file_only_input_types():
    """
    Create input types for file-only nodes (profile reading).
    
    Returns:
        Dictionary with file-only input type configuration
    """
    return {
        "required": {
            "image_path": ("STRING", {
                "default": "",
                "multiline": False,
                "placeholder": "Path to image file"
            }),
        }
    }


def validate_dual_input(input_mode: str, image: torch.Tensor = None, image_path: str = ""):
    """
    Validate dual input parameters.
    
    Args:
        input_mode: "file" or "tensor"
        image: Image tensor (for tensor mode)
        image_path: Image file path (for file mode)
    
    Raises:
        ValueError: If input parameters are invalid
    """
    if input_mode == "file":
        if not image_path.strip():
            raise ValueError("Image path required when input_mode is 'file'")
    else:
        if image is None:
            raise ValueError("Image tensor required when input_mode is 'tensor'")


def get_standard_dual_input_params(input_mode: str, image: torch.Tensor = None, image_path: str = ""):
    """
    Get standardized parameters for dual input processing.
    
    Args:
        input_mode: "file" or "tensor"
        image: Image tensor (for tensor mode)
        image_path: Image file path (for file mode)
    
    Returns:
        Tuple of (img_array, input_source_info)
    """
    validate_dual_input(input_mode, image, image_path)
    
    if input_mode == "file":
        img_array = ColorInputHandler.load_image_from_path(image_path)
        source_info = f"file: {image_path}"
    else:
        img_array = ColorInputHandler.tensor_to_array(image)
        source_info = f"tensor: {image.shape}"
    
    # Normalize image array
    img_array = ColorInputHandler.normalize_image_array(img_array)
    
    return img_array, source_info
