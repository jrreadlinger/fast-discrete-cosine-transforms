from PIL import Image
from dct.dct_naive import dct_naive, idct_naive, dct2_naive, idct2_naive, dct2_rgb, idct2_rgb
from dct.dct_fast import fast_dct_recursive, dct_lee, dct_lee_2d, idct_lee, idct_lee_2d, inverse_transform
from dct.utils import load_grayscale_image
import matplotlib.pyplot as plt
import numpy as np


img = Image.open("data/mary.jpeg")
#img_gs = load_grayscale_image(img)
img_gs = load_grayscale_image("data/mary.jpeg")


block = img_gs[:128, :128]
dct_block = dct_lee_2d(block)
reconstructed = idct_lee_2d(dct_block)

# print(block)
# print(dct_block)
# print(reconstructed)
# print(type(img_gs))

plt.figure(figsize=(12, 4))

# Plot original block
plt.subplot(1, 3, 1)
plt.imshow(block, cmap='gray')
plt.title("Original Block")
plt.axis('off')

# Plot DCT block (note: DCT coefficients can have large range → better to use abs())
plt.subplot(1, 3, 2)
plt.imshow(dct_block, cmap='gray')
plt.title("DCT Coefficients (abs)")
plt.axis('off')
dct_img = plt.imshow(np.abs(dct_block), cmap='gray')
plt.colorbar(dct_img)

# Plot reconstructed block
plt.subplot(1, 3, 3)
plt.imshow(reconstructed, cmap='gray')
plt.title("Reconstructed Block")
plt.axis('off')

plt.tight_layout()
plt.show()
