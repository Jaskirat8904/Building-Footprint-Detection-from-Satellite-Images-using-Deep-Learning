# 🏘️ Building Footprint Detection from Satellite Images using U-Net

<div align="center">

**Deep learning pipeline for automated detection and segmentation of building footprints from satellite imagery**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

*Leveraging U-Net architecture for precise semantic segmentation of urban structures*

</div>

---

## 📋 Table of Contents
- [Problem Statement](#-problem-statement)
- [Key Features](#-key-features)
- [Applications](#-applications)
- [Model Architecture](#-model-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Dataset](#-dataset)
- [Contributing](#-contributing)

---

## 🎯 Problem Statement

Identifying building footprints from satellite imagery is crucial for urban planning, disaster response, and geospatial analytics. However, manual annotation is:
- ⏱️ **Time-intensive** - Hours of manual work per image
- 💰 **Expensive** - Requires trained GIS specialists
- 📏 **Not scalable** - Impractical for large geographical areas
- ❌ **Prone to inconsistency** - Varies between annotators

### 💡 Our Solution

An automated deep learning pipeline that performs **pixel-wise semantic segmentation** to classify each pixel as:
- 🏢 **Building** (foreground)
- 🌳 **Background** (non-building)

---

## ✨ Key Features

### 🔧 **End-to-End Pipeline**
- Automated preprocessing of satellite imagery and ground truth masks
- Custom data augmentation strategies for improved generalization
- Efficient training loop with checkpointing and early stopping

### 🧠 **U-Net Architecture**
- Encoder-decoder structure with skip connections
- Batch normalization for training stability
- Adaptive loss functions for class imbalance handling

### 📊 **Comprehensive Evaluation**
- **Metrics**: IoU (Intersection over Union), Pixel Accuracy, Precision, Recall, F1-Score
- **Visualization**: Predicted masks, boundary overlays, side-by-side comparisons
- **Analysis**: Per-class performance and confusion matrices

### 🗺️ **Multi-Format Output**
- Raster predictions (`.tif`) with georeferencing
- Vector polygons (`.geojson`) for GIS integration
- Direct compatibility with QGIS, ArcGIS, and Google Earth Engine

---

## 🚀 Applications

<table>
<tr>
<td width="50%">

### 🏗️ **Urban Planning**
- Infrastructure development mapping
- Zoning compliance monitoring
- Smart city analytics
- Population density estimation

</td>
<td width="50%">

### 🆘 **Disaster Management**
- Post-disaster damage assessment
- Emergency response routing
- Evacuation planning
- Recovery monitoring

</td>
</tr>
<tr>
<td width="50%">

### 🌍 **Geospatial Analytics**
- Automated map generation
- Change detection over time
- Land use classification
- Urban sprawl analysis

</td>
<td width="50%">

### 📡 **Remote Sensing**
- Multi-temporal analysis
- Cross-sensor validation
- Large-scale mapping projects
- Infrastructure monitoring

</td>
</tr>
</table>

---

## 🏗️ Model Architecture


### Architecture Details

| Layer Type | Input Size | Output Size | Parameters |
|------------|------------|-------------|------------|
| **Encoder Block 1** | 256×256×3 | 128×128×64 | ~38K |
| **Encoder Block 2** | 128×128×64 | 64×64×128 | ~221K |
| **Encoder Block 3** | 64×64×128 | 32×32×256 | ~885K |
| **Encoder Block 4** | 32×32×256 | 16×16×512 | ~3.5M |
| **Bottleneck** | 16×16×512 | 16×16×1024 | ~14M |
| **Decoder Block 1** | 16×16×1024 | 32×32×512 | ~14M |
| **Decoder Block 2** | 32×32×512 | 64×64×256 | ~3.5M |
| **Decoder Block 3** | 64×64×256 | 128×128×128 | ~885K |
| **Decoder Block 4** | 128×128×128 | 256×256×64 | ~221K |
| **Output Layer** | 256×256×64 | 256×256×1 | 65 |

**Total Parameters**: ~31.2M  
**Trainable Parameters**: ~31.2M

### Key Components

#### 🔽 **Contracting Path (Encoder)**
- Captures contextual information through progressive downsampling
- Each block: 2× (3×3 Conv + BatchNorm + ReLU) + 2×2 MaxPool
- Feature channels double at each level: 64 → 128 → 256 → 512

#### 🔄 **Bottleneck**
- Extracts high-level semantic features
- 1024 feature channels for maximum representation capacity
- Connects encoder and decoder paths

#### 🔼 **Expanding Path (Decoder)**
- Enables precise localization through upsampling
- Each block: 2×2 UpConv + Concatenation with encoder features + 2× (3×3 Conv + BatchNorm + ReLU)
- Feature channels halve at each level: 512 → 256 → 128 → 64

#### ⚡ **Skip Connections**
- Direct connections from encoder to decoder at corresponding levels
- Preserve fine-grained spatial information lost during downsampling
- Enable gradient flow and faster convergence

#### 🎯 **Output Layer**
- 1×1 convolution for pixel-wise classification
- Sigmoid activation for binary segmentation
- Produces probability map for building presence




