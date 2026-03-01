"""
train.py
--------
Compiles and trains the DWT-based U-Net model, then saves the trained weights.
"""

from imports import *
from data_loader import xtrain, ytrain
from model import unet

# ── Hyper-parameters ──────────────────────────────────────────────────────────
n_classes  = 3
activation = 'softmax'
LR         = 0.0001
batch_size = 8
epochs     = 10

# ── Build & compile model ─────────────────────────────────────────────────────
model = unet((640, 640, 1))
model.summary()

print(f"xtrain shape: {xtrain.shape}")
print(f"ytrain shape: {ytrain.shape}")

optim = keras.optimizers.Adam(LR)
model.compile(optimizer='adam', loss='mse')

# ── Train ─────────────────────────────────────────────────────────────────────
history = model.fit(xtrain, ytrain, batch_size=batch_size, epochs=epochs)

# ── Save model ────────────────────────────────────────────────────────────────
model.save('unet_dwt_model.keras.h5')
print("Model saved to unet_dwt_model.keras.h5")

# ── Plot training loss ────────────────────────────────────────────────────────
plt.plot(history.history['loss'])
plt.grid()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('Error vs Epoch')
plt.show()
