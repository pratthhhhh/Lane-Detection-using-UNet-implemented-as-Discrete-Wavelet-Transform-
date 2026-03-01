from imports import *

#  Config 
SIZE_X = 640
SIZE_Y = 640
n_classes = 3  # Number of classes for segmentation

xpath = "./train/images"
ypath = "./train/masks.jpg"

#  Load images 
xfiles = sorted(os.listdir(xpath))

xtrain = []
for i in tqdm(xfiles):
    img_path = os.path.join(xpath, i)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Failed to load image {img_path}")
        continue  # Skip this image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
    xtrain.append(img)

xtrain = np.array(xtrain)
print("xtrain shape:", np.shape(xtrain))

#  Load masks ─
img_paths = sorted(glob.glob(os.path.join(ypath, "*.jpg")))

ytrain = []
for mask_path in img_paths:
    mask = cv2.imread(mask_path, 0)
    mask = cv2.resize(mask, (SIZE_Y, SIZE_X), interpolation=cv2.INTER_NEAREST)
    ytrain.append(mask)

ytrain = np.array(ytrain)
print("ytrain shape:", np.shape(ytrain))
print("Unique mask values:", np.unique(ytrain))

#  Quick sanity-check visualisation ─
image_number = random.randint(0, 100)
image = xtrain[image_number]
mask  = ytrain[image_number]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(image, cmap='jet');  axes[0].set_title('Image');  axes[0].set_axis_off()
axes[1].imshow(mask,  cmap='gray'); axes[1].set_title('Mask');   axes[1].set_axis_off()
plt.show()
