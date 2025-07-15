# Building-Footprint-Detection-from-Satellite-Images-using-Unet
Deep learning pipeline for detecting and segmenting building footprints from satellite imagery using a custom U-Net model. Includes preprocessing, training, evaluation, visualization, and GeoJSON vectorization of outputs.

Problem Statement

Accurately identifying building footprints from satellite imagery is a critical task in domains such as urban planning, disaster management, and geospatial analytics. Manual annotation of such features is labor-intensive, time-consuming, and not scalable across large geographical regions.

This project aims to build an automated pipeline for semantic segmentation of satellite images using a deep learning model (U-Net). The objective is to classify each pixel in a satellite image as either:

Building

Background

✨ Key Goals:
Preprocess raw satellite imagery and corresponding ground truth masks.

Train a U-Net model to detect building footprints.

Evaluate the model using metrics like IoU, Accuracy, Precision, and Recall.

Visualize results, including masks, boundaries, and overlays.

Export predictions as raster (.tif) and vector (.geojson) formats for downstream GIS applications.

This solution demonstrates the potential of deep learning to automate building detection with high precision, enabling scalable and reproducible mapping for urban analytics.
