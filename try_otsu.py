"""Try Otsu thresholding on one image.   python try_otsu.py my_image.png"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters

path = sys.argv[1]
img = io.imread(path)
if img.ndim == 3:                      # colour or RGBA -> greyscale
    img = color.rgb2gray(img[..., :3])
img = img.astype(float)
img = (img - img.min()) / (img.max() - img.min())   # scale to 0..1

# Cut off the bottom 10% in case there is a scale bar / info banner. Set to 0 if there is none.
CROP_BOTTOM = 0.10
img = img[: int(img.shape[0] * (1 - CROP_BOTTOM))]

smooth = filters.gaussian(img, sigma=1)             # light denoise

t2 = filters.threshold_otsu(smooth)                 # one cut  -> 2 phases
t3 = filters.threshold_multiotsu(smooth, classes=3) # two cuts -> 3 phases
two = smooth > t2
three = np.digitize(smooth, t3)

print(f"image size: {img.shape}")
print(f"2-phase cut at {t2:.2f}: bright fraction = {two.mean():.3f}")
print(f"3-phase cuts at {t3[0]:.2f}, {t3[1]:.2f}: dark/mid/bright = "
      + ", ".join(f"{(three == i).mean():.3f}" for i in range(3)))

fig, ax = plt.subplots(2, 2, figsize=(12, 10))
ax[0, 0].imshow(img, cmap="gray"); ax[0, 0].set_title("original")
ax[0, 1].hist(smooth.ravel(), bins=256, color="gray")
ax[0, 1].axvline(t2, color="red", label="2-phase cut")
for t in t3: ax[0, 1].axvline(t, color="blue", ls="--")
ax[0, 1].set_title("brightness histogram (red = Otsu, blue = multi-Otsu)"); ax[0, 1].legend()
ax[1, 0].imshow(two, cmap="gray"); ax[1, 0].set_title("Otsu: 2 phases")
ax[1, 1].imshow(three, cmap="viridis"); ax[1, 1].set_title("multi-Otsu: 3 phases")
for a in (ax[0, 0], ax[1, 0], ax[1, 1]): a.axis("off")
plt.tight_layout()
out = path.rsplit(".", 1)[0] + "_otsu.png"
plt.savefig(out, dpi=120)
print("saved", out)