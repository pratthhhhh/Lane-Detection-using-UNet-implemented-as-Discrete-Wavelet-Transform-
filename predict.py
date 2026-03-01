"""
predict.py
----------
Runs inference with a trained DWT U-Net model and visualises predictions.
"""

from imports import *
from data_loader import xtrain, ytrain
from model import unet

# ── Load or build model ───────────────────────────────────────────────────────
# To load a saved model:
# model = keras.models.load_model('unet_dwt_model.keras.h5',
#             custom_objects={'HaarWaveletLayer': HaarWaveletLayer,
#                             'InverseHaarWaveletLayer': InverseHaarWaveletLayer})
#
# Or rebuild + load weights (uncomment as needed):
model = unet((640, 640, 1))
model.load_weights('unet_dwt_model.keras.h5')

# ── Predict ───────────────────────────────────────────────────────────────────
y = model.predict(xtrain[:10])
print("Prediction output shape:", y[0].shape)

# ── Visualise one prediction ──────────────────────────────────────────────────
plt.imshow(y[8], cmap='gray')
plt.title('Prediction (sample 8)')
plt.show()

# ── Visualise image vs ground-truth mask ──────────────────────────────────────
image_num = random.randint(0, 100)
image1    = xtrain[image_num]
mask1     = ytrain[image_num]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(image1, cmap='gray'); axes[0].set_title('Image'); axes[0].set_axis_off()
axes[1].imshow(mask1,  cmap='gray'); axes[1].set_title('Mask');  axes[1].set_axis_off()
plt.show()
