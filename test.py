from PIL import Image
import numpy as np
import os

from dct.utils import check_image_mode

# def check_image_mode(image_path):
#     with Image.open(image_path) as img:
#         print(f"File: {os.path.basename(image_path)}")
#         print(f" - Mode: {img.mode}")
#         img_array = np.array(img)
#         print(f" - Shape: {img_array.shape}")
#         if img.mode == 'L':
#             print(" → This is a grayscale image.")
#         elif img.mode == 'RGB':
#             print(" → This is an RGB image.")
#         else:
#             print(" → Other image mode (e.g., RGBA, CMYK).")
#         print()

# Example usage:



check_image_mode("data/checkerboard_small.png")
check_image_mode("data/checkerboard_large.jpg")