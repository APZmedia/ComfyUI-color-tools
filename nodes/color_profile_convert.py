# comfyui_custom_nodes/color_profile_convert.py
import base64
import io
import json
import math
from typing import Dict, Optional, Tuple

from PIL import Image, ImageCms

# Lazy imports to avoid side effects at module load time
_torch = None
_numpy = None

def _get_torch():
    """Lazy import torch to avoid import side effects"""
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def _get_numpy():
    """Lazy import numpy to avoid import side effects"""
    global _numpy
    if _numpy is None:
        import numpy as np
        _numpy = np
    return _numpy


def _clamp01(t):
    torch = _get_torch()
    return t.clamp_(0.0, 1.0)


# ---- sRGB EOTF/OETF (gamma) ----

def srgb_to_linear_t(t):
    torch = _get_torch()
    a = 0.055
    return torch.where(
        t <= 0.04045,
        t / 12.92,
        torch.pow((t + a) / (1 + a), 2.4)
    )


def linear_to_srgb_t(t):
    torch = _get_torch()
    a = 0.055
    return torch.where(
        t <= 0.0031308,
        12.92 * t,
        (1 + a) * torch.pow(t, 1 / 2.4) - a
    )


# ---- Simple gamma (PNG gAMA) helpers ----

def gamma_to_linear_t(t, encoded_gamma: float):
    # PNG gAMA stores "image gamma". Typical gAMA=0.45455 -> gamma≈1/2.2
    # To get linear: raise to power (1 / gAMA).
    torch = _get_torch()
    if encoded_gamma is None or encoded_gamma <= 0:
        return t
    return torch.pow(t, 1.0 / encoded_gamma)


def linear_to_gamma_t(t: torch.Tensor, encoded_gamma: float) -> torch.Tensor:
    if encoded_gamma is None or encoded_gamma <= 0:
        return t
    return torch.pow(t, encoded_gamma)


# ---- Chromaticity (cHRM) matrix math ----

def xy_to_xyz(x: float, y: float) -> np.ndarray:
    if y == 0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float64)
    return np.array([x / y, 1.0, (1.0 - x - y) / y], dtype=np.float64)


def primaries_to_rgb2xyz(rxy, gxy, bxy, wxy) -> np.ndarray:
    Rx = xy_to_xyz(rxy[0], rxy[1])
    Gx = xy_to_xyz(gxy[0], gxy[1])
    Bx = xy_to_xyz(bxy[0], bxy[1])
    W  = xy_to_xyz(wxy[0], wxy[1])
    M  = np.stack([Rx, Gx, Bx], axis=1)  # 3x3
    S  = np.linalg.solve(M, W)
    return (M * S).astype(np.float64)  # scale columns


def build_matrix_cHRM_to_sRGB(chrm: Dict[str, float]) -> Optional[np.ndarray]:
    # sRGB / D65 reference primaries
    s_r = (0.6400, 0.3300)
    s_g = (0.3000, 0.6000)
    s_b = (0.1500, 0.0600)
    s_w = (0.3127, 0.3290)  # D65

    try:
        rxy = (float(chrm["rx"]), float(chrm["ry"]))
        gxy = (float(chrm["gx"]), float(chrm["gy"]))
        bxy = (float(chrm["bx"]), float(chrm["by"]))
        wxy = (float(chrm["wx"]), float(chrm["wy"]))
    except Exception:
        return None

    M_src = primaries_to_rgb2xyz(rxy, gxy, bxy, wxy)
    M_dst = primaries_to_rgb2xyz(s_r, s_g, s_b, s_w)
    try:
        M = np.matmul(np.linalg.inv(M_src), M_dst)  # RGB_src -> RGB_sRGB (both linear)
    except np.linalg.LinAlgError:
        return None
    return M


def apply_3x3(t_lin_rgb: torch.Tensor, M: np.ndarray) -> torch.Tensor:
    # t_lin_rgb: [B,H,W,C], C=3, linear
    B, H, W, C = t_lin_rgb.shape
    flat = t_lin_rgb.view(-1, 3)
    M_t = torch.from_numpy(M).to(t_lin_rgb.device, dtype=t_lin_rgb.dtype)  # 3x3
    out = torch.matmul(flat, M_t)  # (N,3) x (3,3)
    return out.view(B, H, W, 3)


# ---- ICC via Pillow/lcms (8-bit path) ----

def icc_convert_to_srgb_uint8(img_uint8: Image.Image, icc_bytes: bytes) -> Image.Image:
    src = ImageCms.ImageCmsProfile(io.BytesIO(icc_bytes))
    dst = ImageCms.createProfile("sRGB")
    # Use perceptual intent by default; adjust if you want to expose a UI parameter
    return ImageCms.profileToProfile(img_uint8, src, dst, outputMode="RGB", renderingIntent=0)


def tensor_to_pil_uint8(img: torch.Tensor) -> Image.Image:
    # img: [B,H,W,C] float 0..1, assume B==1
    if img.ndim != 4 or img.shape[0] != 1 or img.shape[-1] != 3:
        raise ValueError("Expected IMAGE tensor with shape [1,H,W,3].")
    arr = (img[0].cpu().numpy() * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def pil_to_tensor_uint8(img: Image.Image, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.uint8)
    t = torch.from_numpy(arr).to(device=device, dtype=dtype) / 255.0
    return t.unsqueeze(0)  # [1,H,W,3]


# ---- Main conversion strategy ----

def convert_any_to_srgb_or_linear(
    img: torch.Tensor,
    target_space: str,                   # "sRGB" or "sRGB_linear"
    icc_b64: Optional[str],
    srgb_intent: Optional[int],
    gamma_val: Optional[float],
    chroma_json: Optional[str],
) -> torch.Tensor:

    device, dtype = img.device, img.dtype
    icc_bytes = None
    if icc_b64:
        try:
            icc_bytes = base64.b64decode(icc_b64)
        except Exception:
            icc_bytes = None

    # 1) ICC path → display sRGB
    if icc_bytes:
        pil_in = tensor_to_pil_uint8(img)
        try:
            pil_srgb = icc_convert_to_srgb_uint8(pil_in, icc_bytes)
            t = pil_to_tensor_uint8(pil_srgb, device, dtype)
        except Exception:
            # fall back to identity if lcms fails
            t = img.clone()
    else:
        # 2) If PNG said sRGB, trust it → already sRGB
        if srgb_intent is not None:
            t = img.clone()
        else:
            # 3) Try cHRM + gAMA matrix conversion
            chrm = None
            if chroma_json:
                try:
                    chrm = json.loads(chroma_json)
                except Exception:
                    chrm = None

            if chrm and all(k in chrm for k in ("rx","ry","gx","gy","bx","by","wx","wy")):
                # Assume input is gamma-encoded by gAMA (if given), convert to linear
                t_lin = gamma_to_linear_t(img, gamma_val) if gamma_val else img
                M = build_matrix_cHRM_to_sRGB(chrm)
                if M is not None:
                    t_lin_srgb = apply_3x3(t_lin, M)
                    t = _clamp01(t_lin_srgb)
                    # If final target is display sRGB, encode; if linear, keep linear
                    if target_space == "sRGB":
                        t = linear_to_srgb_t(t)
                else:
                    # Matrix failed → assume sRGB
                    t = img.clone()
            else:
                # 4) No reliable info → assume it's already sRGB
                t = img.clone()

    # Final step: if target is linear, linearize sRGB
    if target_space == "sRGB_linear":
        t = _clamp01(srgb_to_linear_t(t))
    else:
        t = _clamp01(t)

    return t


# ---------- ComfyUI Node ----------

class ColorProfileConvert:
    """
    Convert an IMAGE tensor to sRGB or linear sRGB using ICC (preferred),
    then sRGB flag, then cHRM+gAMA matrix, else assume sRGB.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_space": (["sRGB", "sRGB_linear"], {"default": "sRGB"}),
            },
            "optional": {
                # Typically fed from the ColorProfileReader node
                "icc_profile_base64": ("STRING", {"default": "", "multiline": False}),
                "png_srgb_intent": ("INT", {"default": -1, "min": -1, "max": 3, "step": 1}),
                "png_gamma": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.00001}),
                "png_chromaticity_json": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": '{"wx":0.3127,"wy":0.3290,"rx":0.64,...}'
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "convert"
    CATEGORY = "Image/Color"

    def convert(
        self,
        image: torch.Tensor,
        target_space: str,
        icc_profile_base64: str = "",
        png_srgb_intent: int = -1,
        png_gamma: float = 0.0,
        png_chromaticity_json: str = "",
    ):
        srgb_intent = None if png_srgb_intent < 0 else int(png_srgb_intent)
        gamma_val = png_gamma if png_gamma > 0 else None

        out = convert_any_to_srgb_or_linear(
            image, target_space,
            icc_profile_base64 or None,
            srgb_intent,
            gamma_val,
            png_chromaticity_json or None,
        )
        return (out,)


# ComfyUI node registry
NODE_CLASS_MAPPINGS = {
    "ColorProfileConvert": ColorProfileConvert
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorProfileConvert": "Color Profile → sRGB / Linear"
}
