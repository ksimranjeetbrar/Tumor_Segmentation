import random, pathlib, cv2, numpy as np
import matplotlib.pyplot as plt

img_dir = pathlib.Path("Processed_Data_2020/train/images")
mask_dir = pathlib.Path("Processed_Data_2020/train/masks")

files = list(img_dir.glob("*.png"))
f = random.choice(files)

img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
mask = cv2.imread(str(mask_dir / (f.stem + "_mask.png")), cv2.IMREAD_GRAYSCALE)

# binary mask: tumor = 1, background = 0
mask_bin = (mask > 0).astype(np.uint8)

# make a red overlay for tumor
overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
overlay[mask_bin == 1] = [255, 0, 0]

plt.figure(figsize=(9,3))
plt.subplot(1,3,1); plt.imshow(img, cmap="gray"); plt.title("MRI")
plt.subplot(1,3,2); plt.imshow(mask_bin, cmap="gray"); plt.title("Mask (binary)")
plt.subplot(1,3,3); plt.imshow(overlay); plt.title("Overlay")
plt.tight_layout()
plt.show()
