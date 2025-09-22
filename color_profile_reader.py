# comfyui_custom_nodes/color_profile_reader.py
import base64
import io
import json
import struct
from typing import Any, Dict, Optional

from PIL import Image, ImageCms  # Pillow + littleCMS via ImageCms


def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def _png_iter_chunks(f):
    # assumes file pointer after PNG signature
    while True:
        length_bytes = f.read(4)
        if len(length_bytes) < 4:
            return
        length = struct.unpack(">I", length_bytes)[0]
        ctype = f.read(4)
        data = f.read(length)
        _ = f.read(4)  # CRC
        yield ctype, data
        if ctype == b'IEND':
            return


def _parse_png_colorspace(path: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "container": "PNG",
        "icc_present": False,
        "icc_profile_name": None,
        "icc_base64": None,
        "srgb_chunk": None,           # rendering intent 0..3
        "gamma": None,                # display-exponent inverse (e.g., 0.45455)
        "chromaticity": None,         # {wx, wy, rx, ry, gx, gy, bx, by}
        "notes": [],
    }
    with open(path, "rb") as f:
        sig = f.read(8)
        if sig != b"\x89PNG\r\n\x1a\n":
            out["notes"].append("Not a PNG signature.")
            return out
        for ctype, data in _png_iter_chunks(f):
            if ctype == b'iCCP':
                # iCCP: profile name (latin-1, null-terminated), compression method (1 byte), zlib data
                try:
                    name_end = data.find(b"\x00")
                    prof_name = data[:name_end].decode("latin-1", "replace")
                    comp_method = data[name_end + 1]
                    comp_data = data[name_end + 2:]
                    if comp_method != 0:
                        out["notes"].append("Unknown iCCP compression method.")
                    import zlib
                    icc_bytes = zlib.decompress(comp_data)
                    out["icc_present"] = True
                    out["icc_profile_name"] = prof_name
                    out["icc_base64"] = base64.b64encode(icc_bytes).decode("ascii")
                except Exception as e:
                    out["notes"].append(f"iCCP parse error: {e}")
            elif ctype == b'sRGB':
                # 1 byte rendering intent (0=Perceptual,1=Relative,2=Saturation,3=Absolute)
                if len(data) == 1:
                    out["srgb_chunk"] = int(data[0])
            elif ctype == b'gAMA':
                # 4-byte unsigned int: image gamma * 100000
                if len(data) == 4:
                    g = struct.unpack(">I", data)[0] / 100000.0
                    out["gamma"] = g
            elif ctype == b'cHRM':
                # 8 unsigned ints: wx, wy, rx, ry, gx, gy, bx, by each * 100000
                if len(data) == 32:
                    vals = struct.unpack(">8I", data)
                    vals = [v / 100000.0 for v in vals]
                    out["chromaticity"] = {
                        "wx": vals[0], "wy": vals[1],
                        "rx": vals[2], "ry": vals[3],
                        "gx": vals[4], "gy": vals[5],
                        "bx": vals[6], "by": vals[7],
                    }
    return out


def _icc_summary(icc_bytes: bytes) -> Dict[str, Any]:
    """
    Minimal ICC header introspection for quick metadata without external deps.
    Reference: ICC.1:2010 specification. Header = 128 bytes.
    """
    if not icc_bytes or len(icc_bytes) < 128:
        return {}
    hdr = icc_bytes[:128]
    # bytes 36-39: PCS ('XYZ ' or 'Lab ')
    pcs = hdr[36:40].decode('ascii', 'replace')
    # bytes 16-19: profile/device class ('mntr','scnr','prtr','spac','link','abst','nmcl','cdev')
    pclass = hdr[12:16].decode('ascii', 'replace')
    # bytes 48-51: creation year, etc. are encoded in dateTimeNumber (36..? in older refs); skip
    # bytes 80-83: signature 'acsp'
    sig = hdr[36+44:36+44+4].decode('ascii', 'replace')  # not exact, just sanity
    result = {
        "profile_class": pclass,
        "pcs": pcs,
        "acsp_sanity": ("acsp" in icc_bytes[:200].decode('latin-1', 'ignore')),
    }
    return result


def _profile_from_pillow(path: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "container": None,
        "icc_present": False,
        "icc_profile_name": None,
        "icc_base64": None,
        "srgb_chunk": None,
        "gamma": None,
        "chromaticity": None,
        "pillow_mode": None,
        "rendering_intent": None,
        "icc_summary": None,
        "notes": [],
    }
    try:
        with Image.open(path) as im:
            out["container"] = im.format
            out["pillow_mode"] = im.mode
            info = getattr(im, "info", {}) or {}
            icc = info.get("icc_profile", None)
            if icc:
                if isinstance(icc, bytes):
                    out["icc_present"] = True
                    out["icc_base64"] = base64.b64encode(icc).decode("ascii")
                    try:
                        prof = ImageCms.ImageCmsProfile(io.BytesIO(icc))
                        # Try get name; fall back to None
                        try:
                            out["icc_profile_name"] = ImageCms.getProfileName(prof)
                        except Exception:
                            out["icc_profile_name"] = None
                        out["icc_summary"] = _icc_summary(icc)
                    except Exception as e:
                        out["notes"].append(f"ICC parse via ImageCms failed: {e}")
                else:
                    out["notes"].append("Unexpected icc_profile type in Pillow info.")

            # PNG specifics (only if PNG and no ICC extras parsed above)
            if im.format == "PNG":
                png_meta = _parse_png_colorspace(path)
                # Merge but keep ICC from Pillow if already present
                if not out["icc_present"] and png_meta.get("icc_present"):
                    out["icc_present"] = True
                    out["icc_base64"] = png_meta.get("icc_base64")
                    out["icc_profile_name"] = png_meta.get("icc_profile_name")
                    out["icc_summary"] = _icc_summary(base64.b64decode(out["icc_base64"]))
                out["srgb_chunk"] = png_meta.get("srgb_chunk")
                out["gamma"] = png_meta.get("gamma")
                out["chromaticity"] = png_meta.get("chromaticity")
                out["notes"] += png_meta.get("notes", [])

            # JPEG: Pillow exposes ICC in info if present (APP2)
            # WEBP/TIFF/HEIF: Pillow may expose ICC similarly when supported.

    except Exception as e:
        out["notes"].append(f"Open error: {e}")

    return out


class ColorProfileReader:
    """
    ComfyUI node: reads color profile / colorspace hints from an image file.
    Returns JSON blobs so you can branch with Conditionals or display in a Text node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Absolute or ComfyUI input path to image file"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("profile_json", "icc_base64", "primaries_json", "notes_json")
    FUNCTION = "read"
    CATEGORY = "Image/Color"

    def read(self, image_path: str):
        meta = _profile_from_pillow(image_path)

        # primaries/white from cHRM if available
        primaries = meta.get("chromaticity") or {}

        # Build a compact summary for general use
        summary = {
            "container": meta.get("container"),
            "pillow_mode": meta.get("pillow_mode"),
            "icc_present": meta.get("icc_present"),
            "icc_profile_name": meta.get("icc_profile_name"),
            "icc_summary": meta.get("icc_summary"),
            "srgb_chunk_intent": meta.get("srgb_chunk"),  # 0..3 or None
            "gamma": meta.get("gamma"),
            "chromaticity": primaries or None,
        }

        profile_json = json.dumps(summary, ensure_ascii=False)
        primaries_json = json.dumps(primaries or {}, ensure_ascii=False)
        notes_json = json.dumps(meta.get("notes", []), ensure_ascii=False)
        icc_b64 = meta.get("icc_base64") or ""

        return (profile_json, icc_b64, primaries_json, notes_json)


# ComfyUI node registry
NODE_CLASS_MAPPINGS = {
    "ColorProfileReader": ColorProfileReader
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorProfileReader": "Color Profile Reader"
}
