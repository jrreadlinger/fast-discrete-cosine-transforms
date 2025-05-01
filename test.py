from PIL import Image
from dct.dct_naive import dct_naive, idct_naive, dct2_naive, idct2_naive, dct2_rgb, idct2_rgb
from dct.utils import load_grayscale_image

img = Image.open("steve-face.png")
img_gs = load_grayscale_image(img)

block = img_gs[:8, :8]
dct_block = dct2_naive(block)
reconstructed = idct2_naive(dct_block)

print(block)
print(dct_block)
print(reconstructed)