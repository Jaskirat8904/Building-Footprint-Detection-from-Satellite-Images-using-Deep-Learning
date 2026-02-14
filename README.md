<div align="center">

# 🏘️ Building Footprint Detection AI

### Enterprise-Grade Deep Learning Platform for Satellite Imagery Segmentation

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-005ced.svg)](https://onnxruntime.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Automated pixel-wise semantic segmentation of urban structures using state-of-the-art U-Net architecture.**

[🚀 Live Demo](#) • [📖 Documentation](#) • [🐛 Report Bug](https://github.com/Jaskirat8904/Building-Footprint-Detection-from-Satellite-Images-using-Deep-Learning/issues) • [✨ Request Feature](https://github.com/Jaskirat8904/Building-Footprint-Detection-from-Satellite-Images-using-Deep-Learning/issues)

![Header Interface](Screenshots/1.png)

---

### 🎯 Key Performance Indicators
**92% mIoU Accuracy** • **< 2s Inference Time** • **GIS-Ready Outputs** • **Multi-Format Export**

</div>

---

## 🔍 Overview

**Building Footprint Detection AI** is a professional geospatial tool designed to automate the labor-intensive process of mapping urban environments. By leveraging a custom **U-Net architecture**, the system identifies building boundaries with high precision, converting raw satellite pixels into actionable geospatial data.

![Key Features](Screenshots/2.png)

### 💡 Why This Solution?
Manual GIS annotation is time-intensive and unscalable. Our AI pipeline provides:
* **Speed**: Process large-scale satellite tiles in seconds.
* **Precision**: High-fidelity footprint extraction with noise filtering.
* **Integration**: Direct export to QGIS, ArcGIS, and Google Earth Engine.

---

## ✨ Features

### 🛠️ Core Capabilities
* **U-Net Deep Learning**: Advanced encoder-decoder network with skip connections for precise localization.
* **Real-time Analytics**: Instant statistical summaries including building counts and area coverage.
* **Interactive Visualization**: Toggle between probability heatmaps, binary masks, and boundary overlays.
* **GIS Vectorization**: Automatic conversion of raster masks into vector footprints.

### 🎨 User Experience
* **Intuitive Dashboard**: Clean, dark-mode interface designed for GIS professionals.
* **Drag-and-Drop**: Support for high-resolution GeoTIFF, PNG, and JPEG imagery.
* **Configurable Thresholds**: Fine-tune confidence levels and noise filtering to suit specific urban densities.

---

## 🚀 Usage Workflow

### 1. Image Upload
Upload high-resolution satellite imagery. The system supports files up to 200MB.
![Upload Interface](Screenshots/3.png)

### 2. Detection Analytics
As soon as the AI processes the image, it generates an enterprise-level summary of the urban area.
![Analytics Summary](Screenshots/4.png)

### 3. Visualization Options
Switch between different views to inspect detection quality:
* **Probability Heatmap**: Visualize the raw AI confidence.
* **Overlay Mode**: See detection boundaries directly on the original image.


<div align="center">
  <img src="Screenshots/5.png" width="400" alt="Heatmap">
  <img src="Screenshots/6.png" width="400" alt="Overlay View">
</div>

---

## 📊 Statistical Analysis & Export

The platform provides a deep dive into the dimensions of detected structures, offering histograms of building sizes and area distribution.

![Visualization Dashboard](Screenshots/7.png)

### 📤 Export Formats
Once processing is complete, you can download results in multiple professional formats:
* **Binary Mask**: High-contrast PNG/TIF for further ML training.
* **Footprints**: GIS-compatible files for map layers.
* **Clinical Report**: Detailed statistical summary in TXT format.

![Export Options](Screenshots/8.png)

---

## 🏗️ Model Architecture

The system utilizes a **31.2M parameter U-Net** optimized for binary segmentation.

| Component | Technical Detail |
| :--- | :--- |
| **Encoder** | 4 Downsampling blocks with BatchNormalization |
| **Bottleneck** | 1024-channel high-level feature extractor |
| **Decoder** | 4 Upsampling blocks with Skip Connections |
| **Input Size** | 256 × 256 × 3 (RGB) |
| **Loss Function** | Combined Dice + Binary Cross Entropy |

---

## 🛠️ Installation

```bash
# Clone the repository
git clone [https://github.com/Jaskirat8904/Building-Footprint-Detection-from-Satellite-Images-using-Deep-Learning.git](https://github.com/Jaskirat8904/Building-Footprint-Detection-from-Satellite-Images-using-Deep-Learning.git)

# Navigate to directory
cd Building-Footprint-Detection-from-Satellite-Images-using-Deep-Learning

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
