# ComfyUI Color Profile Reader

A ComfyUI custom node for reading color profiles and color space information from image files. This node extracts ICC profiles, PNG color space chunks, and other color metadata to help you understand the color characteristics of your images.

## 🎨 Features

### **Color Profile Detection**
- **ICC Profile Extraction**: Reads embedded ICC color profiles from images
- **PNG Color Space Support**: Extracts sRGB, gamma, and chromaticity data from PNG files
- **Multi-format Support**: Works with JPEG, PNG, TIFF, WEBP, and other formats supported by Pillow
- **Profile Analysis**: Provides detailed information about color profiles and color spaces

### **Gamma Comparison**
- **Gamma Analysis**: Compares gamma values between two images
- **Standard Gamma Detection**: Identifies common gamma values (sRGB, Rec. 709, Rec. 2020, etc.)
- **Tolerance-based Comparison**: Configurable tolerance for gamma difference detection
- **Detailed Recommendations**: Provides workflow and technical recommendations for gamma mismatches

### **Output Information**
- **Profile JSON**: Complete color profile metadata in JSON format
- **ICC Base64**: Raw ICC profile data encoded in Base64
- **Primaries JSON**: Color primaries and white point information
- **Notes JSON**: Any warnings or additional information about the image

## 🚀 Installation

### Method 1: ComfyUI Manager (Recommended)

1. Open ComfyUI
2. Go to Manager → Install
3. Search for "ComfyUI Color Profile Reader"
4. Click Install

### Method 2: Manual Installation

1. Clone this repository to your ComfyUI `custom_nodes` directory:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/yourusername/ComfyUI-color-tools.git
   ```

2. Install dependencies:
   ```bash
   cd ComfyUI-color-tools
   pip install -r requirements.txt
   ```

3. Restart ComfyUI to load the new node

## 📖 Usage

### Basic Color Profile Reading

1. Add a "Color Profile Reader" node from the "Image/Color" category
2. Connect an image path (string) to the `image_path` input
3. The node will output:
   - `profile_json`: Complete profile information
   - `icc_base64`: Raw ICC profile data
   - `primaries_json`: Color primaries information
   - `notes_json`: Any additional notes or warnings

### Workflow Integration

The node outputs JSON strings that can be:
- Displayed in Text nodes for inspection
- Used with Conditional nodes for branching logic
- Processed by other nodes that accept string inputs
- Saved to files for further analysis

## 🛠️ Node Details

### **Color Profile Reader**

**Inputs:**
- `image_path` (STRING): Path to the image file (absolute or relative to ComfyUI)

**Outputs:**
- `profile_json` (STRING): Complete color profile metadata
- `icc_base64` (STRING): Raw ICC profile data in Base64 encoding
- `primaries_json` (STRING): Color primaries and white point data
- `notes_json` (STRING): Additional information and warnings

### **Gamma Compare**

**Inputs:**
- `image_path_1` (STRING): Path to the first image file
- `image_path_2` (STRING): Path to the second image file
- `tolerance` (FLOAT): Tolerance for gamma difference detection (0.001-0.1)

**Outputs:**
- `comparison_json` (STRING): Detailed comparison data between the two images
- `gamma_analysis` (STRING): In-depth gamma analysis and interpretation
- `recommendations` (STRING): Workflow and technical recommendations

### **Profile JSON Structure**

```json
{
  "container": "PNG",
  "pillow_mode": "RGB",
  "icc_present": true,
  "icc_profile_name": "sRGB IEC61966-2.1",
  "icc_summary": {
    "profile_class": "mntr",
    "pcs": "XYZ ",
    "acsp_sanity": true
  },
  "srgb_chunk_intent": 0,
  "gamma": 0.45455,
  "chromaticity": {
    "wx": 0.3127,
    "wy": 0.3290,
    "rx": 0.64,
    "ry": 0.33,
    "gx": 0.30,
    "gy": 0.60,
    "bx": 0.15,
    "by": 0.06
  }
}
```

## 📊 Use Cases

### **Color Management Workflows**
- Verify color profiles in source images
- Ensure proper color space handling in processing pipelines
- Detect color profile mismatches that could affect output quality
- Compare gamma values between images for consistency

### **Image Analysis**
- Analyze color characteristics of reference images
- Extract color space information for documentation
- Validate color profile compliance
- Detect gamma mismatches that could cause color shifts

### **Quality Control**
- Check for missing or incorrect color profiles
- Verify color space consistency across image sets
- Monitor color profile usage in production workflows
- Ensure gamma consistency in multi-image projects

### **Gamma Comparison Workflows**
- Compare gamma values between source and output images
- Detect gamma mismatches in color pipelines
- Analyze gamma characteristics for color space identification
- Generate recommendations for gamma correction

## 🔧 Dependencies

- `Pillow>=8.0.0`: Image processing and ICC profile support

## 📁 Project Structure

```
ComfyUI-color-tools/
├── README.md
├── LICENSE
├── requirements.txt
├── __init__.py
├── color_profile_reader.py
└── examples/
    └── color_profile_workflow.json
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

MIT License - See LICENSE file for details.

## 🆘 Support

For issues, feature requests, or questions:
- Open an issue on GitHub
- Check the documentation
- Review example workflows

## 🔄 Version History

- **v1.0.0**: Initial release with color profile reading functionality

## 🙏 Acknowledgments

- ComfyUI community for the excellent framework
- Pillow team for image processing capabilities
- ICC profile specification contributors

---

**Author**: Pablo Apiolazza  
**Repository**: [ComfyUI-color-tools](https://github.com/APZmedia/ComfyUI-color-tools)  
**Category**: Image/Color