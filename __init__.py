"""
ComfyUI Color Tools

A comprehensive collection of color manipulation and analysis nodes for ComfyUI workflows.
This package provides advanced color processing capabilities including color space conversions,
color grading, palette extraction, color analysis tools, and color profile reading.
"""

print("--- ComfyUI Color Tools: Initializing ---")

# A dictionary to hold all loaded node classes
loaded_nodes = {}

# Import process with detailed logging
try:
    print("[Color Tools] Attempting to import core nodes...")
    from .nodes.color_profile_reader import ColorProfileReader, GammaCompare
    loaded_nodes["ColorProfileReader"] = ColorProfileReader
    loaded_nodes["GammaCompare"] = GammaCompare
    print("[Color Tools] ✅ Loaded: ColorProfileReader, GammaCompare")

    from .nodes.color_profile_convert_simple import ColorProfileConvert
    loaded_nodes["ColorProfileConvert"] = ColorProfileConvert
    print("[Color Tools] ✅ Loaded: ColorProfileConvert")

    from .nodes.color_converter_advanced import ColorConverterAdvanced
    loaded_nodes["ColorConverterAdvanced"] = ColorConverterAdvanced
    print("[Color Tools] ✅ Loaded: ColorConverterAdvanced")

    print("[Color Tools] Attempting to import OCIO nodes...")
    from .nodes.ocio_tools import OCIOColorSpaceConverter, OCIOConfigInfo, TestPatternGenerator
    loaded_nodes["OCIOColorSpaceConverter"] = OCIOColorSpaceConverter
    loaded_nodes["OCIOConfigInfo"] = OCIOConfigInfo
    loaded_nodes["TestPatternGenerator"] = TestPatternGenerator
    print("[Color Tools] ✅ Loaded: OCIOColorSpaceConverter, OCIOConfigInfo, TestPatternGenerator")

except (ImportError, ValueError) as e:
    print(f"[Color Tools] ⚠️  Import Error: {e}")
    print("[Color Tools] 💡 Some nodes may not be available. This can happen if a dependency is missing.")
    print("[Color Tools] 💡 OCIO nodes require the 'opencolorio' package. Check your environment.")

# Define mappings based on successfully loaded nodes
NODE_CLASS_MAPPINGS = loaded_nodes

# Display names for all potential nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorProfileReader": "Read Image Color Profile",
    "GammaCompare": "Compare Image Gamma Values",
    "ColorProfileConvert": "Convert Image Color Space",
    "ColorConverterAdvanced": "Advanced Color Converter",
    "OCIOColorSpaceConverter": "OCIO Color Space Converter",
    "OCIOConfigInfo": "OCIO Config Info",
    "TestPatternGenerator": "Test Pattern Generator",
}

print(f"[Color Tools] --- Registration ---")
print(f"[Color Tools] 📌 Registered {len(NODE_CLASS_MAPPINGS)} node classes:")
for name in NODE_CLASS_MAPPINGS.keys():
    display_name = NODE_DISPLAY_NAME_MAPPINGS.get(name, "Unknown")
    print(f"[Color Tools]   - {name} -> '{display_name}'")

# Filter display names to only those that were successfully loaded
NODE_DISPLAY_NAME_MAPPINGS = {
    key: value for key, value in NODE_DISPLAY_NAME_MAPPINGS.items() if key in NODE_CLASS_MAPPINGS
}

print("--- ComfyUI Color Tools: Initialization Complete ---")

# Export the mappings for ComfyUI
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
