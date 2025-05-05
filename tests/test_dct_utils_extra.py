import numpy as np
from dct.utils import (
    split_into_blocks,
    combine_blocks,
    mask_dct_coefficients,
    normalize_image
)

def test_mask_dct_coefficients():
    print("\n--- Testing mask_dct_coefficients ---")
    block = np.arange(64).reshape(8, 8)
    masked = mask_dct_coefficients(block, keep=4)
    
    assert np.all(masked[4:, :] == 0)
    assert np.all(masked[:, 4:] == 0)
    print(" → Passed: only top-left 4×4 retained.")


def test_normalize_image():
    print("\n--- Testing normalize_image ---")
    test_img = np.array([[10, 20], [30, 40]], dtype=np.float32)
    norm_img = normalize_image(test_img)

    assert norm_img.min() == 0
    assert norm_img.max() == 255
    assert norm_img.dtype == np.uint8
    print(" → Passed: image scaled to 0–255.")


def test_split_and_combine_blocks_grayscale():
    print("\n--- Testing split/combine (grayscale) ---")
    img = np.random.randint(0, 256, size=(18, 20), dtype=np.uint8)
    blocks, original_shape = split_into_blocks(img, block_size=8)
    img_reconstructed = combine_blocks(blocks, original_shape)

    assert img_reconstructed.shape == original_shape
    assert np.all(img_reconstructed[:img.shape[0], :img.shape[1]] == img)
    print(" → Passed: grayscale split & recombine matches original.")


def test_split_and_combine_blocks_rgb():
    print("\n--- Testing split/combine (RGB) ---")
    img = np.random.randint(0, 256, size=(30, 34, 3), dtype=np.uint8)
    blocks, original_shape = split_into_blocks(img, block_size=8)
    img_reconstructed = combine_blocks(blocks, original_shape)

    assert img_reconstructed.shape == original_shape
    assert np.all(img_reconstructed[:img.shape[0], :img.shape[1], :] == img)
    print(" → Passed: RGB split & recombine matches original.")


if __name__ == "__main__":
    test_mask_dct_coefficients()
    test_normalize_image()
    test_split_and_combine_blocks_grayscale()
    test_split_and_combine_blocks_rgb()