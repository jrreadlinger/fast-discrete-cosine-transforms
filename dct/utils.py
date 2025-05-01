from PIL import Image
import numpy as np
import os

def load_grayscale_image(image_path):
    """
    Loads an image from a given path and ensures it's in grayscale ('L') mode.
    Returns:
        A 2D numpy array of dtype uint8.
    """
    with Image.open(image_path) as img:
        if img.mode != 'L':
            img = img.convert('L')
        return np.array(img)
    
def check_image_mode(image_path):
    with Image.open(image_path) as img:
        print(f"File: {os.path.basename(image_path)}")
        print(f" - Mode: {img.mode}")
        img_array = np.array(img)
        print(f" - Shape: {img_array.shape}")
        if img.mode == 'L':
            print(" → This is a grayscale image.")
        elif img.mode == 'RGB':
            print(" → This is an RGB image.")
        else:
            print(" → Other image mode (e.g., RGBA, CMYK).")
        print()