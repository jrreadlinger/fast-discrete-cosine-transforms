import numpy as np
from dct.utils import (
    split_into_blocks,
    combine_blocks,
    mask_dct_coefficients,
    normalize_image
)

def test_mask_dct_coefficients():
    block = np.arange(64).reshape(8, 8)
    masked = mask_dct_coefficients(block, keep=4)
    
    assert np.all(masked[4:, :] == 0)
    assert np.all(masked[:, 4:] == 0)

def test_normalize_image():
    test_img = np.array([[10, 20], [30, 40]], dtype=np.float32)
    norm_img = normalize_image(test_img)

    assert norm_img.min() == 0
    assert norm_img.max() == 255
    assert norm_img.dtype == np.uint8

def test_split_and_combine_blocks_grayscale():
    img = np.random.randint(0, 256, size=(18, 20), dtype=np.uint8)
    blocks, original_shape = split_into_blocks(img, block_size=8)
    img_reconstructed = combine_blocks(blocks, original_shape)

    assert img_reconstructed.shape == original_shape
    assert np.all(img_reconstructed[:img.shape[0], :img.shape[1]] == img)

def test_split_and_combine_blocks_rgb():
    img = np.random.randint(0, 256, size=(30, 34, 3), dtype=np.uint8)
    blocks, original_shape = split_into_blocks(img, block_size=8)
    img_reconstructed = combine_blocks(blocks, original_shape)

    assert img_reconstructed.shape == original_shape
    assert np.all(img_reconstructed[:img.shape[0], :img.shape[1], :] == img)

if __name__ == "__main__":
    test_mask_dct_coefficients()
    test_normalize_image()
    test_split_and_combine_blocks_grayscale()
    test_split_and_combine_blocks_rgb()