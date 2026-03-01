from imports import *
from data_loader import xtrain, ytrain

#  Label-encode masks 
labelencoder = LabelEncoder()
n, h, w = ytrain.shape
train_masks_reshaped         = ytrain.reshape(-1, 1)
train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)

print("Unique encoded mask values:", np.unique(train_masks_encoded_original_shape))

train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)
print("xtrain shape:", np.shape(xtrain), "  train_masks_input shape:", np.shape(train_masks_input))

#  Train / validation split 
X_train, X_do_not_use, y_train, y_do_not_use = train_test_split(
    xtrain, ytrain, test_size=0.2, random_state=0
)
print(f"Split sizes — train: {len(X_train)}, hold-out: {len(X_do_not_use)}")

#  Normalise ─
X_train_normalized      = X_train.astype('float32')      / 255.0
X_do_not_use_normalized = X_do_not_use.astype('float32') / 255.0

y_train_normalized      = y_train.astype('float32')
y_do_not_use_normalized = y_do_not_use.astype('float32')

print(len(X_train_normalized), len(y_train_normalized),
      len(X_do_not_use_normalized), len(y_do_not_use_normalized))

print("Class values in the dataset are ...",
      np.unique(y_train), np.unique(y_train).__len__())

#  Visualise a sample after encoding 
image_number = random.randint(0, 100)
image = xtrain[image_number]
mask  = train_masks_input[image_number]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(image, cmap='gray'); axes[0].set_title('Image'); axes[0].set_axis_off()
axes[1].imshow(mask,  cmap='gray'); axes[1].set_title('Mask');  axes[1].set_axis_off()
plt.show()
