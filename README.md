# Lane Detection using U-Net with Discrete Wavelet Transform (DWT)

This repository contains a deep learning pipeline for lane detection (semantic segmentation). The architecture leverages a custom U-Net model that incorporates 2D Haar Discrete Wavelet Transform (DWT) for downsampling in the encoder and Inverse Haar Wavelet Transform for upsampling in the decoder.

This approach allows the network to preserve high-frequency spatial information (such as edges and fine textures) that is often lost during standard max-pooling operations, resulting in more accurate and sharper lane segmentations.

## Features

- **Custom Wavelet Layers:** Implements `HaarWaveletLayer` for feature decomposition and `InverseHaarWaveletLayer` for reconstruction directly within TensorFlow/Keras.
- **U-Net Architecture:** Fully convolutional encoder-decoder structure enhanced by wavelet transforms.
- **Modular Codebase:** Training, data loading, preprocessing, and inference are neatly separated into distinct, easy-to-read scripts.
- **Ready-to-run Pipeline:** Includes image resizing, grayscale conversion, and mask label encoding.

## Project Structure

```text
.
├── imports.py         # Consolidates all necessary library imports
├── data_loader.py     # Discovers and loads images (X) and masks (y) from disk
├── preprocessing.py   # Handles label encoding, normalization, and train/test splits
├── model.py           # Defines the Haar DWT layers and the U-Net architecture
├── train.py           # Compiles the model, executes training, and saves weights
├── predict.py         # Loads the trained model to perform inference and plot results
├── requirements.txt   # Python package dependencies
└── train/             # (Expected) Directory containing the dataset
    ├── images/        # Input images for training
    └── masks.jpg/     # Corresponding segmentation masks
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Lane-Detection-using-UNet-implemented-as-Discrete-Wavelet-Transform-
   ```

2. **Install dependencies:**
   It is recommended to use a virtual environment. Install the required packages via:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Preparation

The code expects the dataset to be located in a `train/` directory at the project root. Inside, there should be an `images/` folder and a `masks.jpg/` folder containing the ground truth.

Example:
```text
train/
├── images/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
└── masks.jpg/
    ├── mask_001.jpg
    ├── mask_002.jpg
    └── ...
```

## Usage

Because the project is modular, you can run the files sequentially, or run the high-level scripts directly.

### 1. Train the Model
To start training the U-Net DWT model on your data, simply run:
```bash
python train.py
```

### 2. Predict and Visualize
Once the model is trained, you can run inference to see the predictions compared against the ground truth masks:
```bash
python predict.py
```
This script will draw a random sample from the dataset, predict the lane mask, and display a matplotlib figure showing the original image next to the model's prediction.

## Model Architecture Details

Standard U-Nets use MaxPooling for downsampling and Transposed Convolutions or UpSampling for the decoder. 
This network replaces those standard operations with Wavelet Transforms:
- **Encoder:** The `HaarWaveletLayer` splits incoming feature maps into Low-Low (LL), Low-High (LH), High-Low (HL), and High-High (HH) frequency components. 
- **Decoder:** The `InverseHaarWaveletLayer` takes these 4 frequency components and precisely reconstructs the higher-resolution spatial map. This prevents the "checkerboard" artifacts sometimes seen in standard upsampling and keeps lane edges incredibly sharp.
