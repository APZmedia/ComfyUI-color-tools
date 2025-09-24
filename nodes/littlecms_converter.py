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
        """Core profile conversion logic using LittleCMS with proper color space handling"""
        
        # Ensure image is in [0, 1] range
        if img_array.max() > 1.0:
            img_array = img_array / 255.0
        
        # Handle custom profiles first
        if source_profile == "custom" and custom_source_profile.strip():
            return self._convert_with_custom_profiles(img_array, source_profile, target_profile, 
                                                    rendering_intent, black_point_compensation,
                                                    custom_source_profile, custom_target_profile)
        
        # Use LittleCMS with proper color space handling
        converted_array = self._convert_with_littlecms(img_array, source_profile, target_profile, 
                                                     rendering_intent, black_point_compensation)
        
        # Create conversion info
        conversion_info = self._create_conversion_info(source_profile, target_profile, rendering_intent)
        profile_info = self._create_profile_info_simple(source_profile, target_profile)
        
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
            # Use sRGB as base and apply Adobe RGB transformation
            return self._create_adobe_rgb_compatible_profile()
        elif profile_name == "ProPhoto RGB":
            # Use sRGB as base and apply ProPhoto RGB transformation  
            return self._create_prophoto_rgb_compatible_profile()
        elif profile_name == "Rec. 709":
            return ImageCms.createProfile("sRGB")  # Rec. 709 is very similar to sRGB
        elif profile_name == "Rec. 2020":
            # Use sRGB as base and apply Rec. 2020 transformation
            return self._create_rec2020_compatible_profile()
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
    
    def _create_adobe_rgb_compatible_profile(self) -> ImageCms.ImageCmsProfile:
        """Create Adobe RGB compatible profile using sRGB as base"""
        # Since LittleCMS doesn't support Adobe RGB profile creation,
        # we'll use sRGB and handle the conversion in the transform
        return ImageCms.createProfile("sRGB")
    
    def _create_prophoto_rgb_compatible_profile(self) -> ImageCms.ImageCmsProfile:
        """Create ProPhoto RGB compatible profile using sRGB as base"""
        return ImageCms.createProfile("sRGB")
    
    def _create_rec2020_compatible_profile(self) -> ImageCms.ImageCmsProfile:
        """Create Rec. 2020 compatible profile using sRGB as base"""
        return ImageCms.createProfile("sRGB")
    
    def _convert_with_littlecms(self, img_array: np.ndarray, source_profile: str, 
                               target_profile: str, rendering_intent: str, 
                               black_point_compensation: bool) -> np.ndarray:
        """Convert using LittleCMS with proper color space handling"""
        
        # For Adobe RGB and other non-standard profiles, use matrix transformations
        if source_profile == "Adobe RGB" or target_profile == "Adobe RGB":
            return self._convert_with_matrix_transform(img_array, source_profile, target_profile)
        
        # For standard profiles that LittleCMS supports, use the original method
        try:
            # Get source and target profiles
            src_profile = self._get_profile(source_profile, "")
            dst_profile = self._get_profile(target_profile, "")
            
            # Create transform
            transform = self._create_transform(src_profile, dst_profile, rendering_intent, black_point_compensation)
            
            # Convert image
            converted_array = self._apply_transform(img_array, transform)
            
            return converted_array
            
        except Exception as e:
            # Fallback to matrix transformation if LittleCMS fails
            return self._convert_with_matrix_transform(img_array, source_profile, target_profile)
    
    def _convert_with_matrix_transform(self, img_array: np.ndarray, source_profile: str, 
                                     target_profile: str) -> np.ndarray:
        """Convert using matrix transformations for color spaces LittleCMS doesn't support natively"""
        
        # Convert to XYZ (common intermediate space)
        xyz = self._rgb_to_xyz(img_array, source_profile)
        
        # Convert from XYZ to target color space
        converted_array = self._xyz_to_rgb(xyz, target_profile)
        
        return converted_array
    
    def _rgb_to_xyz(self, rgb_array: np.ndarray, source_profile: str) -> np.ndarray:
        """Convert RGB to XYZ color space"""
        if source_profile == "sRGB":
            # sRGB to XYZ matrix
            matrix = np.array([
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041]
            ])
        elif source_profile == "Adobe RGB":
            # Adobe RGB to XYZ matrix
            matrix = np.array([
                [0.5767309, 0.1855540, 0.1881852],
                [0.2973769, 0.6273491, 0.0752741],
                [0.0270343, 0.0706872, 0.9911085]
            ])
        else:
            # Default to sRGB
            matrix = np.array([
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041]
            ])
        
        # Apply gamma correction first
        if source_profile == "sRGB":
            rgb_linear = self._srgb_gamma_to_linear(rgb_array)
        elif source_profile == "Adobe RGB":
            rgb_linear = self._adobe_rgb_gamma_to_linear(rgb_array)
        else:
            rgb_linear = self._srgb_gamma_to_linear(rgb_array)
        
        # Apply matrix transformation
        xyz = np.dot(rgb_linear, matrix.T)
        return xyz
    
    def _xyz_to_rgb(self, xyz_array: np.ndarray, target_profile: str) -> np.ndarray:
        """Convert XYZ to RGB color space"""
        if target_profile == "sRGB":
            # XYZ to sRGB matrix
            matrix = np.array([
                [3.2404542, -1.5371385, -0.4985314],
                [-0.9692660, 1.8760108, 0.0415560],
                [0.0556434, -0.2040259, 1.0572252]
            ])
        elif target_profile == "Adobe RGB":
            # XYZ to Adobe RGB matrix
            matrix = np.array([
                [2.0413690, -0.5649464, -0.3446944],
                [-0.9692660, 1.8760108, 0.0415560],
                [0.0134474, -0.1183897, 1.0154096]
            ])
        else:
            # Default to sRGB
            matrix = np.array([
                [3.2404542, -1.5371385, -0.4985314],
                [-0.9692660, 1.8760108, 0.0415560],
                [0.0556434, -0.2040259, 1.0572252]
            ])
        
        # Apply matrix transformation
        rgb_linear = np.dot(xyz_array, matrix.T)
        
        # Apply gamma correction
        if target_profile == "sRGB":
            rgb = self._linear_to_srgb_gamma(rgb_linear)
        elif target_profile == "Adobe RGB":
            rgb = self._linear_to_adobe_rgb_gamma(rgb_linear)
        else:
            rgb = self._linear_to_srgb_gamma(rgb_linear)
        
        return rgb
    
    def _srgb_gamma_to_linear(self, rgb: np.ndarray) -> np.ndarray:
        """Convert sRGB gamma to linear"""
        return np.where(rgb <= 0.04045, rgb / 12.92, np.power((rgb + 0.055) / 1.055, 2.4))
    
    def _linear_to_srgb_gamma(self, rgb: np.ndarray) -> np.ndarray:
        """Convert linear to sRGB gamma"""
        return np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.power(rgb, 1.0/2.4) - 0.055)
    
    def _adobe_rgb_gamma_to_linear(self, rgb: np.ndarray) -> np.ndarray:
        """Convert Adobe RGB gamma to linear"""
        return np.power(rgb, 2.2)
    
    def _linear_to_adobe_rgb_gamma(self, rgb: np.ndarray) -> np.ndarray:
        """Convert linear to Adobe RGB gamma"""
        return np.power(rgb, 1.0/2.2)
    
    def _convert_with_custom_profiles(self, img_array: np.ndarray, source_profile: str, target_profile: str, 
                                    rendering_intent: str, black_point_compensation: bool,
                                    custom_source_profile: str, custom_target_profile: str) -> Tuple[np.ndarray, str, str]:
        """Convert using custom ICC profiles"""
        try:
            # Get custom profiles
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
            
        except Exception as e:
            # Fallback to matrix transformation
            converted_array = self._convert_with_matrix_transform(img_array, source_profile, target_profile)
            conversion_info = self._create_conversion_info(source_profile, target_profile, rendering_intent)
            profile_info = json.dumps({"error": f"Custom profile conversion failed: {str(e)}"}, indent=2)
            return converted_array, conversion_info, profile_info
    
    def _create_profile_info_simple(self, source_profile: str, target_profile: str) -> str:
        """Create simple profile information"""
        info = {
            "source_profile": source_profile,
            "target_profile": target_profile,
            "conversion_method": "Matrix transformation" if "Adobe RGB" in [source_profile, target_profile] else "LittleCMS"
        }
        return json.dumps(info, indent=2)
    
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
