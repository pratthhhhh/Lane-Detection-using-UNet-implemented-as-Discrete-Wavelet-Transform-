{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5bf9274a",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import tensorflow as tf\n",
    "\n",
    "import cv2\n",
    "import numpy as np\n",
    "from tqdm import tqdm\n",
    "from matplotlib import pyplot as plt \n",
    "import glob\n",
    "import keras\n",
    "from keras.utils import normalize\n",
    "from keras.metrics import MeanIoU"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c9ec3155",
   "metadata": {},
   "outputs": [],
   "source": [
    "SIZE_X = 640\n",
    "SIZE_Y = 640\n",
    "n_classes = 3 #Number of classes for segmentation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "340fdb9e",
   "metadata": {},
   "outputs": [],
   "source": [
    "xpath = \"./train/images\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "46d6ee8a",
   "metadata": {},
   "outputs": [],
   "source": [
    "xfiles = sorted(os.listdir(xpath))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3a49f636",
   "metadata": {},
   "outputs": [],
   "source": [
    "ypath = \"./train/masks.jpg\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a36fc90b",
   "metadata": {},
   "outputs": [],
   "source": [
    "xtrain = []\n",
    "\n",
    "for i in tqdm(xfiles):\n",
    "    img_path = os.path.join(xpath, i)\n",
    "    img = cv2.imread(img_path)\n",
    "    if img is None:\n",
    "        print(f\"Warning: Failed to load image {img_path}\")\n",
    "        continue  # Skip this image\n",
    "    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\n",
    "    img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)\n",
    "    xtrain.append(img)\n",
    "\n",
    "xtrain = np.array(xtrain)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4b48e27e",
   "metadata": {},
   "outputs": [],
   "source": [
    "np.shape(xtrain)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "528a4baa",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Get a sorted list of image paths\n",
    "img_paths = sorted(glob.glob(os.path.join(ypath, \"*.jpg\")))\n",
    "\n",
    "ytrain = []\n",
    "\n",
    "# Iterate through the sorted list of image paths\n",
    "for mask_path in img_paths:\n",
    "  mask = cv2.imread(mask_path, 0)\n",
    "  # mask = mask/255.0\n",
    "  # mask = mask.astype(np.float32)\n",
    "  mask = cv2.resize(mask, (SIZE_Y, SIZE_X), interpolation = cv2.INTER_NEAREST)  #Otherwise ground truth changes due to interpolation\n",
    "  ytrain.append(mask)\n",
    "\n",
    "#Convert list to array for machine learning processing\n",
    "ytrain = np.array(ytrain)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "61e39137",
   "metadata": {},
   "outputs": [],
   "source": [
    "np.shape(ytrain)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7c554249",
   "metadata": {},
   "outputs": [],
   "source": [
    "np.unique(ytrain)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cdfbc79e",
   "metadata": {},
   "outputs": [],
   "source": [
    "import random\n",
    "image_number = random.randint(0, 100)\n",
    "\n",
    "image = xtrain[image_number]\n",
    "mask = ytrain[image_number]\n",
    "\n",
    "fig, axes = plt.subplots(1, 2, figsize=(10, 5))\n",
    "\n",
    "axes[0].imshow(image, cmap = 'jet')\n",
    "axes[0].set_title('Image')\n",
    "axes[0].set_axis_off()\n",
    "\n",
    "axes[1].imshow(mask, cmap = 'gray')\n",
    "axes[1].set_title('Mask')\n",
    "axes[1].set_axis_off()\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3b2cdd01",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Encode labels... but multi dim array so need to flatten, encode and reshape\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "labelencoder = LabelEncoder()\n",
    "n, h, w = ytrain.shape\n",
    "train_masks_reshaped = ytrain.reshape(-1,1)\n",
    "train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)\n",
    "train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)\n",
    "\n",
    "np.unique(train_masks_encoded_original_shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7a4ec1ac",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)\n",
    "\n",
    "np.shape(xtrain), np.shape(train_masks_input)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "37bf61dd",
   "metadata": {},
   "outputs": [],
   "source": [
    "import random\n",
    "image_number = random.randint(0, 100)\n",
    "\n",
    "image = xtrain[image_number]\n",
    "mask = train_masks_input[image_number]\n",
    "\n",
    "fig, axes = plt.subplots(1, 2, figsize=(10, 5))\n",
    "\n",
    "axes[0].imshow(image, cmap = 'gray')\n",
    "axes[0].set_title('Image')\n",
    "axes[0].set_axis_off()\n",
    "\n",
    "axes[1].imshow(mask, cmap = 'gray')\n",
    "axes[1].set_title('Mask')\n",
    "axes[1].set_axis_off()\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1b61a0d5",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "X_train, X_do_not_use, y_train, y_do_not_use = train_test_split(xtrain, ytrain, test_size = 0.2, random_state = 0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ddc169d7",
   "metadata": {},
   "outputs": [],
   "source": [
    "len(X_train), len(X_do_not_use), len(y_train), len(y_do_not_use)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5c954366",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "\n",
    "# Normalize images by dividing by 255 (assuming they are in range [0, 255])\n",
    "X_train_normalized = X_train.astype('float32') / 255.0\n",
    "X_do_not_use_normalized = X_do_not_use.astype('float32') / 255.0\n",
    "\n",
    "# Optional: Normalize the labels if needed, assuming they are binary (0 or 1)\n",
    "# No normalization is usually needed for labels if they are already in [0, 1] range\n",
    "y_train_normalized = y_train.astype('float32')\n",
    "y_do_not_use_normalized = y_do_not_use.astype('float32')\n",
    "\n",
    "# Check the lengths\n",
    "print(len(X_train_normalized), len(y_train_normalized), len(X_do_not_use_normalized), len(y_do_not_use_normalized))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "120f5405",
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"Class values in the dataset are ... \", np.unique(y_train), np.unique(y_train).__len__())  # 0 is the background/few unlabeled"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "94adcecd",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Custom Haar Wavelet Layer (downsampling)\n",
    "class HaarWaveletLayer(layers.Layer):\n",
    "    def __init__(self, **kwargs):\n",
    "        super(HaarWaveletLayer, self).__init__(**kwargs)\n",
    "\n",
    "    def call(self, inputs):\n",
    "        # Assuming input shape is fixed and known, e.g., (batch_size, height, width, channels)\n",
    "        height = tf.shape(inputs)[1]\n",
    "        width = tf.shape(inputs)[2]\n",
    "\n",
    "        # Ensure that the height and width are divisible by 2\n",
    "        height = height // 2 * 2\n",
    "        width = width // 2 * 2\n",
    "\n",
    "        # Perform Haar wavelet transform (downsampling)\n",
    "        LL = (inputs[:, :height:2, :width:2, :] + inputs[:, 1:height:2, :width:2, :]) / 2\n",
    "        LH = (inputs[:, :height:2, :width:2, :] - inputs[:, 1:height:2, :width:2, :]) / 2\n",
    "        HL = (inputs[:, :height:2, :width:2, :] + inputs[:, :height:2, 1:width:2, :]) / 2\n",
    "        HH = (inputs[:, :height:2, :width:2, :] - inputs[:, :height:2, 1:width:2, :]) / 2\n",
    "        \n",
    "        # Concatenate the four components along the last axis (channels)\n",
    "        result = tf.concat([LL, LH, HL, HH], axis=-1)\n",
    "        return LL, LH, HL, HH\n",
    "\n",
    "# Custom Inverse Haar Wavelet Layer (upsampling)\n",
    "class InverseHaarWaveletLayer(layers.Layer):\n",
    "    def __init__(self, **kwargs):\n",
    "        super(InverseHaarWaveletLayer, self).__init__(**kwargs)\n",
    "\n",
    "    def call(self, inputs):\n",
    "        # Split the input tensor into four components\n",
    "        LL, LH, HL, HH = tf.split(inputs, 4, axis=-1)\n",
    "\n",
    "        # Reconstruct the image (Inverse Haar Transform)\n",
    "        x1 = (LL + LH) / 2\n",
    "        x2 = (LL - LH) / 2\n",
    "        x3 = (HL + HH) / 2\n",
    "        x4 = (HL - HH) / 2\n",
    "\n",
    "        # Concatenate the components back (reconstructing the original image)\n",
    "        reconstructed = tf.concat([x1, x2, x3, x4], axis=1)  # Concatenate along the second axis\n",
    "        return reconstructed\n",
    "\n",
    "# Convolution Block\n",
    "def conv_block(x, filters, kernel_size=30, activation='relu'):\n",
    "    x = layers.Conv2D(filters, kernel_size, padding='same', activation=activation)(x)\n",
    "    return x\n",
    "\n",
    "# U-Net Model Definition with Haar Wavelet Transform\n",
    "def unet(input_shape):\n",
    "    inputs = layers.Input(shape=input_shape)\n",
    "\n",
    "    # Encoder: Layer 1\n",
    "    x = inputs\n",
    "    LL1, LH1, HL1, HH1 = HaarWaveletLayer()(x)  \n",
    "    x11 = conv_block(LL1, 1)  \n",
    "\n",
    "    # Encoder: Layer 2\n",
    "    LL2, LH2, HL2, HH2 = HaarWaveletLayer()(x11)\n",
    "    x21 = conv_block(LL2, 1) \n",
    "\n",
    "    # Encoder: Layer 3\n",
    "    LL3, LH3, HL3, HH3 = HaarWaveletLayer()(x21)\n",
    "    x31 = conv_block(LL3, 1) \n",
    "\n",
    "    # Encoder: Layer 4\n",
    "    LL4, LH4, HL4, HH4 = HaarWaveletLayer()(x31)\n",
    "    x41 = conv_block(LL4, 1)  \n",
    "\n",
    "    # Bottleneck\n",
    "    bottleneck = conv_block(x41, 1)\n",
    "    bottleneck = conv_block(bottleneck, 1)\n",
    "    \n",
    "    # Decoder: Layer 4\n",
    "    concat14 = tf.concat([bottleneck, LH4, HL4, HH4], axis=-1)    \n",
    "    x14 = InverseHaarWaveletLayer()(concat14)\n",
    "    x14 = tf.image.resize(x14, size=(80, 80), method=tf.image.ResizeMethod.BILINEAR)\n",
    "    x14 = conv_block(x14, 1)\n",
    "    \n",
    "    # Decoder: Layer 3\n",
    "    concat13 = tf.concat([x14, LH3, HL3, HH3], axis=-1)\n",
    "    x13 = InverseHaarWaveletLayer()(concat13)\n",
    "    x13 = tf.image.resize(x13, size=(160, 160), method=tf.image.ResizeMethod.BILINEAR)\n",
    "    x13 = conv_block(x13, 1)\n",
    "    \n",
    "    # Decoder: Layer 2\n",
    "    concat12 = tf.concat([x13, LH2, HL2, HH2], axis=-1)\n",
    "    x12 = InverseHaarWaveletLayer()(concat12)\n",
    "    x12 = tf.image.resize(x12, size=(320, 320), method=tf.image.ResizeMethod.BILINEAR)\n",
    "    x12 = conv_block(x12, 1)\n",
    "        \n",
    "    # Decoder: Layer 1\n",
    "    concat11 = tf.concat([x12, LH1, HL1, HH1], axis=-1)\n",
    "    x1_ = InverseHaarWaveletLayer()(concat11)\n",
    "    x1_ = tf.image.resize(x1_, size=(640, 640), method=tf.image.ResizeMethod.BILINEAR)\n",
    "    x1_ = conv_block(x1_, 1)\n",
    "    \n",
    "    #print(x1_.shape)\n",
    "    output = layers.Conv2D(1, (1, 1), activation='linear')(x1_)\n",
    "\n",
    "    # Define the model\n",
    "    model = Model(inputs, output)\n",
    "\n",
    "    return model\n",
    "\n",
    "model = unet((640, 640, 1)) #640"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4dc6ce9e",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0fbc7a3a",
   "metadata": {},
   "outputs": [],
   "source": [
    "n_classes = 3\n",
    "activation = 'softmax'\n",
    "\n",
    "LR = 0.0001\n",
    "optim = keras.optimizers.Adam(LR)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3ceed210",
   "metadata": {},
   "outputs": [],
   "source": [
    "print(f\"xtrain shape: {xtrain.shape}\")  # Should be (num_samples, 640, 640, 1)\n",
    "print(f\"ytrain shape: {ytrain.shape}\")  # Should match xtrain\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2bcc686e",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile(optimizer='adam', loss='mse')\n",
    "\n",
    "batch_size = 8\n",
    "epochs = 10\n",
    "history = model.fit(xtrain, ytrain, batch_size=batch_size, epochs=epochs)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "12677ee7",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.save('unet_dwt_model.keras.h5')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c80c6a46",
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.plot(history.history['loss'])\n",
    "plt.grid()\n",
    "plt.xlabel('epochs')\n",
    "plt.ylabel('loss')\n",
    "plt.title('Error vs Epoch')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a3dff87c",
   "metadata": {},
   "outputs": [],
   "source": [
    "xtrain[0].shape, np.expand_dims(xtrain[0], axis=2).shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8a8e0354",
   "metadata": {},
   "outputs": [],
   "source": [
    "y = model.predict(xtrain[:10])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "28ed447d",
   "metadata": {},
   "outputs": [],
   "source": [
    "y[0].shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "11fba6be",
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.imshow(y[8], cmap='gray')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bf874698",
   "metadata": {},
   "outputs": [],
   "source": [
    "import random\n",
    "image_num = random.randint(0, 100)\n",
    "\n",
    "image1 = xtrain[image_num]\n",
    "mask1 = ytrain[image_num]\n",
    "\n",
    "fig, axes = plt.subplots(1, 2, figsize=(10, 5))\n",
    "\n",
    "axes[0].imshow(image, cmap = 'gray')\n",
    "axes[0].set_title('Image')\n",
    "axes[0].set_axis_off()\n",
    "\n",
    "axes[1].imshow(mask, cmap = 'gray')\n",
    "axes[1].set_title('Mask')\n",
    "axes[1].set_axis_off()\n",
    "\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "zayn",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.15"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
