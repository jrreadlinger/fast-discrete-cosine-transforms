from PIL import Image
from dct.dct_naive import dct_naive, idct_naive, dct2_naive, idct2_naive, dct2_rgb, idct2_rgb
from dct.dct_fast import fast_dct_recursive
from dct.utils import load_grayscale_image
import numpy as np
import matplotlib.pyplot as plt

img = np.array(Image.open("data/mary.jpeg"))
# img = load_grayscale_image('data/mary.jpeg')

block = img[:32, :32]
dct_block = dct2_rgb(block)
reconstructed = idct2_rgb(dct_block)

print(block)
print(dct_block)
print(reconstructed)

plt.figure(figsize=(12, 4))

# Plot original block
plt.subplot(1, 3, 1)
plt.imshow(block, cmap='gray')
plt.title("Original Block")
plt.axis('off')

# Plot DCT coefficients for each RGB channel separately
fig, axs = plt.subplots(1, 4, figsize=(14, 4))

axs[0].imshow(dct_block.astype(np.uint8))
axs[0].set_title("Original RGB Block")
axs[0].axis('off')

channel_names = ['Red', 'Green', 'Blue']
for i in range(3):
    dct_channel = np.abs(dct_block[:, :, i])
    im = axs[i + 1].imshow(dct_channel, cmap='viridis')
    axs[i + 1].set_title(f"{channel_names[i]} DCT Coeffs")
    axs[i + 1].axis('off')
    fig.colorbar(im, ax=axs[i + 1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

# Plot reconstructed block
plt.subplot(1, 3, 3)
plt.imshow(reconstructed, cmap='gray')
plt.title("Reconstructed Block")
plt.axis('off')

plt.tight_layout()
plt.show()