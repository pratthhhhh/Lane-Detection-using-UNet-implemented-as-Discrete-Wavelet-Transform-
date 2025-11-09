# Lane Detection using UNet with Discrete Wavelet Transform

A lane detection system implemented using UNet architecture with Discrete Wavelet Transform for improved feature extraction and accuracy.

## Overview
This project uses a UNet neural network combined with Discrete Wavelet Transform (DWT) to accurately detect road lanes in images and video frames.

## Features
- UNet-based semantic segmentation
- Discrete Wavelet Transform for enhanced features
- High accuracy lane detection

## Installation
```bash
git clone <repository-url>
cd Lane-Detection-using-UNet-implemented-as-Discrete-Wavelet-Transform-
pip install -r requirements.txt
```

## Usage
```python
from model import LaneDetector

detector = LaneDetector()
result = detector.detect_lanes('image.jpg')
```
