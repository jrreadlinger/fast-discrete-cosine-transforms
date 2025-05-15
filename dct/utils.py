import numpy as np
from PIL import Image
import os
from skimage.metrics import structural_similarity as ssim


### IMAGE HELPERS ###
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
    """
    Print the mode and shape of the given image.
    Helps determine if it's grayscale, RGB, or another type.
    """
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


### BLOCK HELPERS ###
def split_into_blocks(img_array, block_size=8):
    """
    Split a grayscale or RGB image into non-overlapping blocks.
    Pads the image with zeros if necessary to match block size.

    Returns:
        Padded blocks and the original shape (to crop later if needed).
    """
    h, w = img_array.shape[:2]
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size

    if img_array.ndim == 2:
        padded = np.pad(img_array, ((0, pad_h), (0, pad_w)), mode='constant')
        blocks = padded.reshape(
            padded.shape[0] // block_size, block_size,
            padded.shape[1] // block_size, block_size
        ).swapaxes(1, 2)
    else:
        padded = np.pad(img_array, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
        blocks = padded.reshape(
            padded.shape[0] // block_size, block_size,
            padded.shape[1] // block_size, block_size, 3
        ).swapaxes(1, 2)

    return blocks, img_array.shape  # return original shape for later cropping

def combine_blocks(blocks, original_shape):
    """
    Reconstruct an image from non-overlapping blocks and crop to original size.

    Args:
        blocks: Output from split_into_blocks
        original_shape: (height, width) or (height, width, channels)

    Returns:
        Cropped reconstructed image
    """
    if blocks.ndim == 4:  # grayscale
        h_blocks, w_blocks, bs, _ = blocks.shape
        image = blocks.swapaxes(1, 2).reshape(h_blocks * bs, w_blocks * bs)
    else:  # RGB
        h_blocks, w_blocks, bs, _, c = blocks.shape
        image = blocks.swapaxes(1, 2).reshape(h_blocks * bs, w_blocks * bs, c)

    # Crop back to original size
    slices = tuple(slice(0, s) for s in original_shape)
    return image[slices]


### PERFORMANCE METRIC HELPERS ###
def compute_psnr(original, reconstructed, max_pixel=255.0):
    """
    Compute Peak Signal-to-Noise Ratio between two images.

    Args:
        original, reconstructed: 2D or 3D NumPy arrays
        max_pixel: maximum pixel intensity (usually 255)

    Returns:
        PSNR in dB
    """
    mse = np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def compute_ssim(original, reconstructed, multichannel=True):
    """
    Compute Structural Similarity Index (SSIM) between two images.

    Args:
        original, reconstructed: NumPy arrays (grayscale or RGB)
        multichannel: Set to True if input is color (RGB)

    Returns:
        SSIM score between 0 and 1
    """
    return ssim(original, reconstructed, data_range=255, channel_axis=-1 if multichannel else None)

def verify_inverse_consistency(original, reconstructed, tol=1e-6):
    """
    Checks if original and reconstructed signals match within a relative tolerance.

    Args:
        original: original input signal or image (1D, 2D, or 3D array)
        reconstructed: output after DCT + IDCT
        tol: relative error threshold (default 1e-6)

    Returns:
        True if consistent, else False
    """
    original = original.astype(np.float64)
    reconstructed = reconstructed.astype(np.float64)

    rel_error = np.linalg.norm(original - reconstructed) / np.linalg.norm(original)
    print(f"[Inverse Check] Relative error: {rel_error:.2e}")
    return rel_error < tol

def verify_parseval_identity(original, transformed, tol=1e-6):
    """
    Verifies Parseval's identity: energy in time domain ≈ energy in frequency domain.

    Args:
        original: original input signal or image (1D or 2D array)
        transformed: its DCT coefficients
        tol: allowed relative error (default 1e-6)

    Returns:
        True if identity holds within tolerance, else False
    """
    energy_time = np.sum(np.square(original.astype(np.float64)))
    energy_freq = np.sum(np.square(transformed.astype(np.float64)))

    rel_error = abs(energy_time - energy_freq) / energy_time
    print(f"[Parseval] Time: {energy_time:.4f}, Freq: {energy_freq:.4f}, Rel Error: {rel_error:.2e}")
    return rel_error < tol


### EXTRA HELPERS ###
def mask_dct_coefficients(dct_block, keep=8):
    """
    Zero out all but the top-left [keep x keep] DCT coefficients of a block.

    Args:
        dct_block: 2D DCT block
        keep: how many coefficients to retain from the top-left

    Returns:
        Masked DCT block
    """
    keep = int(keep)
    masked = np.zeros_like(dct_block)
    masked[:keep, :keep] = dct_block[:keep, :keep]
    return masked

# TODO: test this and give it a docstring
def zigzag_mask_dct(dct_block, keep_fraction):
    N = dct_block.shape[0]
    flat = []
    for i in range(2 * N - 1):
        for j in range(i + 1):
            x = j
            y = i - j
            if i % 2 == 0:
                x, y = y, x
            if x < N and y < N:
                flat.append((x, y))

    total = len(flat)
    keep = int(keep_fraction * total)
    mask = np.zeros_like(dct_block)
    for idx in flat[:keep]:
        mask[idx] = dct_block[idx]
    return mask

def normalize_image(img):
    """
    Normalize an image to 0–255 and convert to uint8.

    Args:
        img: 2D or 3D NumPy array

    Returns:
        uint8 image with values in [0, 255]
    """
    img = img - np.min(img)
    if np.max(img) == 0:
        return np.zeros_like(img, dtype=np.uint8)
    img = (img / np.max(img)) * 255
    return img.astype(np.uint8)