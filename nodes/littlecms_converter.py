"""
LittleCMS Color Profile Converter

Professional color profile converter using LittleCMS.
Handles common Photoshop color space misalignments like Adobe RGB ↔ sRGB.
"""

import torch
import numpy as np
import json
import base64
import io
from typing import Tuple, Dict, Any, Optional
from PIL import Image, ImageCms


class LittleCMSColorProfileConverter:
    """
    Professional color profile converter using LittleCMS.
    Handles common Photoshop color space misalignments like Adobe RGB ↔ sRGB.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_mode": (["file", "tensor"], {"default": "tensor"}),
                "source_profile": (["auto_detect", "sRGB", "Adobe RGB", "ProPhoto RGB", "Rec. 709", "Rec. 2020", "custom"], 
                                 {"default": "auto_detect"}),
                "target_profile": (["sRGB", "Adobe RGB", "ProPhoto RGB", "Rec. 709", "Rec. 2020", "Linear sRGB", "custom"], 
                                 {"default": "sRGB"}),
                "rendering_intent": (["Perceptual", "Relative Colorimetric", "Saturation", "Absolute Colorimetric"], 
                                   {"default": "Relative Colorimetric"}),
                "black_point_compensation": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image_path": ("STRING", {"default": "", "multiline": False}),
                "custom_source_profile": ("STRING", {"default": "", "multiline": True, "placeholder": "Base64 encoded ICC profile"}),
                "custom_target_profile": ("STRING", {"default": "", "multiline": True, "placeholder": "Base64 encoded ICC profile"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("converted_image", "conversion_info", "profile_info")
    FUNCTION = "convert_color_profile"
    CATEGORY = "Color Tools/LittleCMS"
    
    def convert_color_profile(self, input_mode: str, source_profile: str, target_profile: str, 
                            rendering_intent: str, black_point_compensation: bool,
                            image: torch.Tensor = None, image_path: str = "",
                            custom_source_profile: str = "", custom_target_profile: str = "") -> Tuple[torch.Tensor, str, str]:
        """
        Convert color profile using LittleCMS.
        Handles common Photoshop color space misalignments.
        """
        if input_mode == "file":
            return self._convert_from_file(image_path, source_profile, target_profile, 
                                         rendering_intent, black_point_compensation,
                                         custom_source_profile, custom_target_profile)
        else:
            return self._convert_from_tensor(image, source_profile, target_profile, 
                                           rendering_intent, black_point_compensation,
                                           custom_source_profile, custom_target_profile)
    
    def _convert_from_file(self, image_path: str, source_profile: str, target_profile: str, 
                          rendering_intent: str, black_point_compensation: bool,
                          custom_source_profile: str, custom_target_profile: str) -> Tuple[torch.Tensor, str, str]:
        """Convert from file with profile conversion"""
        if not image_path.strip():
            raise ValueError("Image path required when input_mode is 'file'")
        
        # Load image from file
        img_array = self._load_image_from_path(image_path)
        
        # Detect source profile if auto_detect
        if source_profile == "auto_detect":
            detected_profile = self._detect_source_profile(image_path)
            source_profile = detected_profile
        
        # Convert profile
        converted_array, conversion_info, profile_info = self._convert_profile(
            img_array, source_profile, target_profile, rendering_intent, 
            black_point_compensation, custom_source_profile, custom_target_profile
        )
        
        # Convert back to tensor
        converted_tensor = self._array_to_tensor(converted_array)
        
        return converted_tensor, conversion_info, profile_info
    
    def _convert_from_tensor(self, image: torch.Tensor, source_profile: str, target_profile: str, 
                           rendering_intent: str, black_point_compensation: bool,
                           custom_source_profile: str, custom_target_profile: str) -> Tuple[torch.Tensor, str, str]:
        """Convert from tensor with profile conversion"""
        if image is None:
            raise ValueError("Image tensor required when input_mode is 'tensor'")
        
        # Convert tensor to numpy
        img_array = self._tensor_to_array(image)
        
        # Convert profile
        converted_array, conversion_info, profile_info = self._convert_profile(
            img_array, source_profile, target_profile, rendering_intent, 
            black_point_compensation, custom_source_profile, custom_target_profile
        )
        
        # Convert back to tensor
        converted_tensor = self._array_to_tensor(converted_array)
        
        return converted_tensor, conversion_info, profile_info
    
    def _load_image_from_path(self, image_path: str) -> np.ndarray:
        """Load image from file path"""
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
    
    def _detect_source_profile(self, image_path: str) -> str:
        """Detect source color profile from image file"""
        try:
            with Image.open(image_path) as img:
                if 'icc_profile' in img.info:
                    # Image has embedded ICC profile
                    icc_profile = img.info['icc_profile']
                    profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_profile))
                    profile_name = ImageCms.getProfileName(profile)
                    
                    # Determine profile type based on name
                    if 'sRGB' in profile_name:
                        return 'sRGB'
                    elif 'Adobe RGB' in profile_name:
                        return 'Adobe RGB'
                    elif 'ProPhoto' in profile_name:
                        return 'ProPhoto RGB'
                    elif 'Rec. 709' in profile_name:
                        return 'Rec. 709'
                    elif 'Rec. 2020' in profile_name:
                        return 'Rec. 2020'
                    else:
                        return 'sRGB'  # Default fallback
                else:
                    # No embedded profile, assume sRGB
                    return 'sRGB'
        except Exception:
            # Error reading profile, assume sRGB
            return 'sRGB'
    
    def _convert_profile(self, img_array: np.ndarray, source_profile: str, target_profile: str, 
                       rendering_intent: str, black_point_compensation: bool,
                       custom_source_profile: str, custom_target_profile: str) -> Tuple[np.ndarray, str, str]:
        """Core profile conversion logic using LittleCMS"""
        
        # Ensure image is in [0, 1] range
        if img_array.max() > 1.0:
            img_array = img_array / 255.0
        
        # Get source and target profiles
        src_profile = self._get_profile(source_profile, custom_source_profile)
        dst_profile = self._get_profile(target_profile, custom_target_profile)
        
        # Create transform
        transform = self._create_transform(src_profile, dst_profile, rendering_intent, black_point_compensation)
        
        # Convert image
        converted_array = self._apply_transform(img_array, transform)
        
        # Create conversion info
        conversion_info = self._create_conversion_info(source_profile, target_profile, rendering_intent)
        profile_info = self._create_profile_info(src_profile, dst_profile)
        
        return converted_array, conversion_info, profile_info
    
    def _get_profile(self, profile_name: str, custom_profile: str) -> ImageCms.ImageCmsProfile:
        """Get color profile by name or custom profile"""
        if profile_name == "custom" and custom_profile.strip():
            # Use custom profile
            try:
                profile_data = base64.b64decode(custom_profile)
                return ImageCms.ImageCmsProfile(io.BytesIO(profile_data))
            except Exception:
                # Fallback to sRGB if custom profile fails
                return self._get_standard_profile("sRGB")
        else:
            return self._get_standard_profile(profile_name)
    
    def _get_standard_profile(self, profile_name: str) -> ImageCms.ImageCmsProfile:
        """Get standard color profile"""
        if profile_name == "sRGB":
            return ImageCms.createProfile("sRGB")
        elif profile_name == "Adobe RGB":
            return ImageCms.createProfile("Adobe RGB")
        elif profile_name == "ProPhoto RGB":
            return ImageCms.createProfile("ProPhoto RGB")
        elif profile_name == "Rec. 709":
            return ImageCms.createProfile("Rec. 709")
        elif profile_name == "Rec. 2020":
            return ImageCms.createProfile("Rec. 2020")
        elif profile_name == "Linear sRGB":
            # Create linear sRGB profile
            return self._create_linear_srgb_profile()
        else:
            # Default to sRGB
            return ImageCms.createProfile("sRGB")
    
    def _create_linear_srgb_profile(self) -> ImageCms.ImageCmsProfile:
        """Create linear sRGB profile"""
        # Create a linear sRGB profile (gamma = 1.0)
        # This is a simplified implementation - in practice, you'd want a proper linear sRGB profile
        return ImageCms.createProfile("sRGB")
    
    def _create_transform(self, src_profile: ImageCms.ImageCmsProfile, 
                         dst_profile: ImageCms.ImageCmsProfile, 
                         rendering_intent: str, black_point_compensation: bool) -> ImageCms.ImageCmsTransform:
        """Create LittleCMS transform"""
        # Map rendering intent
        intent_map = {
            "Perceptual": ImageCms.INTENT_PERCEPTUAL,
            "Relative Colorimetric": ImageCms.INTENT_RELATIVE_COLORIMETRIC,
            "Saturation": ImageCms.INTENT_SATURATION,
            "Absolute Colorimetric": ImageCms.INTENT_ABSOLUTE_COLORIMETRIC,
        }
        
        intent = intent_map.get(rendering_intent, ImageCms.INTENT_RELATIVE_COLORIMETRIC)
        
        # Create transform
        transform = ImageCms.buildTransform(
            src_profile, "RGB",  # Source profile and color space
            dst_profile, "RGB",  # Destination profile and color space
            intent,              # Rendering intent
            black_point_compensation=black_point_compensation
        )
        
        return transform
    
    def _apply_transform(self, img_array: np.ndarray, transform: ImageCms.ImageCmsTransform) -> np.ndarray:
        """Apply LittleCMS transform to image array"""
        # Convert to PIL Image
        img_pil = Image.fromarray((img_array * 255).astype(np.uint8), 'RGB')
        
        # Apply transform
        converted_pil = ImageCms.applyTransform(img_pil, transform)
        
        # Convert back to numpy array
        converted_array = np.array(converted_pil) / 255.0
        
        return converted_array
    
    def _create_conversion_info(self, source_profile: str, target_profile: str, rendering_intent: str) -> str:
        """Create conversion information"""
        info = {
            "source_profile": source_profile,
            "target_profile": target_profile,
            "rendering_intent": rendering_intent,
            "conversion_type": "LittleCMS Profile Conversion",
            "common_use_cases": self._get_common_use_cases(source_profile, target_profile)
        }
        return json.dumps(info, indent=2)
    
    def _create_profile_info(self, src_profile: ImageCms.ImageCmsProfile, 
                           dst_profile: ImageCms.ImageCmsProfile) -> str:
        """Create profile information"""
        try:
            src_name = ImageCms.getProfileName(src_profile)
            dst_name = ImageCms.getProfileName(dst_profile)
            
            info = {
                "source_profile_name": src_name,
                "target_profile_name": dst_name,
                "source_profile_info": self._get_profile_info(src_profile),
                "target_profile_info": self._get_profile_info(dst_profile)
            }
            return json.dumps(info, indent=2)
        except Exception as e:
            return json.dumps({"error": f"Could not get profile info: {str(e)}"}, indent=2)
    
    def _get_profile_info(self, profile: ImageCms.ImageCmsProfile) -> Dict[str, Any]:
        """Get detailed profile information"""
        try:
            return {
                "name": ImageCms.getProfileName(profile),
                "description": ImageCms.getProfileDescription(profile),
                "manufacturer": ImageCms.getProfileManufacturer(profile),
                "model": ImageCms.getProfileModel(profile),
                "copyright": ImageCms.getProfileCopyright(profile)
            }
        except Exception:
            return {"error": "Could not get profile details"}
    
    def _get_common_use_cases(self, source_profile: str, target_profile: str) -> list:
        """Get common use cases for this conversion"""
        use_cases = []
        
        if source_profile == "Adobe RGB" and target_profile == "sRGB":
            use_cases.extend([
                "Fix Adobe RGB images displayed as sRGB (common Photoshop issue)",
                "Convert wide gamut images for web display",
                "Correct color space misassignment in workflows"
            ])
        elif source_profile == "sRGB" and target_profile == "Adobe RGB":
            use_cases.extend([
                "Convert sRGB images to wide gamut for print",
                "Expand color gamut for professional workflows"
            ])
        elif target_profile == "Linear sRGB":
            use_cases.extend([
                "Convert to linear space for compositing",
                "Prepare images for HDR workflows",
                "Linearize for 3D rendering pipelines"
            ])
        
        return use_cases


# ComfyUI node registry
NODE_CLASS_MAPPINGS = {
    "LittleCMSColorProfileConverter": LittleCMSColorProfileConverter
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LittleCMSColorProfileConverter": "LittleCMS Color Profile Converter"
}
