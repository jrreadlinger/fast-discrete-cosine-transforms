import numpy as np
from PIL import Image
import os
from skimage.metrics import structural_similarity as ssim


# Image Helpers
def load_grayscale_image(image_path):
    """
    load a grayscale image from image path
    """
    with Image.open(image_path) as img:
        if img.mode != 'L':
            img = img.convert('L')
        return np.array(img)

def check_image_mode(image_path):
    """
    return image mode from image path
    """
    with Image.open(image_path) as img:
        print(f"File: {os.path.basename(image_path)}")
        print(f"Mode: {img.mode}")
        img_array = np.array(img)
        print(f"Shape: {img_array.shape}")
        if img.mode == 'L':
            print("This is a grayscale image")
        elif img.mode == 'RGB':
            print("This is an RGB image")
        else:
            print("Unknown image mode")
        print()


# Block Helpers
def split_into_blocks(img_array, block_size=8):
    """
    split image into 8x8 blocks
    pad image if necessary to make dimensions divisible by block_size
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

    return blocks, img_array.shape

def combine_blocks(blocks, original_shape):
    """
    reconstruct image from 8x8 blocks
    crop the padding if necessary
    """
    if blocks.ndim == 4: # grayscale
        h_blocks, w_blocks, bs, _ = blocks.shape
        image = blocks.swapaxes(1, 2).reshape(h_blocks * bs, w_blocks * bs)
    else: # RGB
        h_blocks, w_blocks, bs, _, c = blocks.shape
        image = blocks.swapaxes(1, 2).reshape(h_blocks * bs, w_blocks * bs, c)

    # Crop back to original size
    slices = tuple(slice(0, s) for s in original_shape)
    return image[slices]


# Performance Metric Helpers
def compute_psnr(original, reconstructed, max_pixel=255.0):
    """
    compute Peak Signal-to-Noise Ratio between two images
    """
    mse = np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def compute_ssim(original, reconstructed, multichannel=True):
    """
    compute Structural Similarity Index (SSIM) between two images
    """
    return ssim(original, reconstructed, data_range=255, channel_axis=-1 if multichannel else None)

def verify_inverse_consistency(original, reconstructed, tol=1e-6):
    """
    checks if original and reconstructed signals match within a given tolerance
    """
    original = original.astype(np.float64)
    reconstructed = reconstructed.astype(np.float64)

    rel_error = np.linalg.norm(original - reconstructed) / np.linalg.norm(original)
    print(f"Relative error: {rel_error:.2e}")
    return rel_error < tol

def verify_parseval_identity(original, transformed, tol=1e-6):
    """
    verifies Parseval's Identity
    energy in the time domain matches energy in frequency domain
    """
    energy_time = np.sum(np.square(original.astype(np.float64)))
    energy_freq = np.sum(np.square(transformed.astype(np.float64)))

    rel_error = abs(energy_time - energy_freq) / energy_time
    print(f"[Parseval] Time: {energy_time:.4f}, Freq: {energy_freq:.4f}, Rel Error: {rel_error:.2e}")
    return rel_error < tol


# Extra Helpers
def mask_dct_coefficients(dct_block, keep=8):
    """
    zero out all but the top-left DCT coefficients of a block (square)
    """
    keep = int(keep)
    masked = np.zeros_like(dct_block)
    masked[:keep, :keep] = dct_block[:keep, :keep]
    return masked

# TODO: test this and give it a docstring
def zigzag_mask_dct(dct_block, keep_fraction):
    """
    zero out all but the top-left DCT coefficients of a block (triangle)
    """
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
    normalize an image to 0–255 pixel values
    """
    img = img - np.min(img)
    if np.max(img) == 0:
        return np.zeros_like(img, dtype=np.uint8)
    img = (img / np.max(img)) * 255
    return img.astype(np.uint8)