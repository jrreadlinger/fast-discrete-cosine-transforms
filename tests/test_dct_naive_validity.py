from PIL import Image
import numpy as np
from scipy.fft import dct as scipy_dct, idct as scipy_idct
from dct.dct_naive import (
    dct1,
    idct1,
    dct2,
    idct2,
    dct2rgb,
    idct2rgb
)
from dct.utils import (
    load_grayscale_image,
    split_into_blocks,
    combine_blocks,
    compute_psnr,
    compute_ssim,
    verify_inverse_consistency,
    verify_parseval_identity,
    normalize_image
)

# === 1D Signal Tests ===
def test_1d_signals(dct_fn, idct_fn):
    print("\n--- Testing 1D Signals ---")
    signals = {
        "impulse": np.load("data/impulse_signal.npy"),
        "sine": np.load("data/sine_signal.npy"),
        "step": np.load("data/step_signal.npy")
    }

    for name, x in signals.items():
        print(f"\nSignal: {name}")
        X = dct_fn(x)
        x_rec = idct_fn(X)

        # Naive vs scipy reference
        X_ref = scipy_dct(x, type=2, norm='ortho')
        x_ref = scipy_idct(X, type=2, norm='ortho')

        assert np.allclose(X, X_ref, atol=1e-6), f"DCT mismatch for {name}"
        assert np.allclose(x_rec, x_ref, atol=1e-6), f"IDCT mismatch for {name}"

        assert verify_parseval_identity(x, X)
        assert verify_inverse_consistency(x, x_rec)


# === 2D Grayscale Image Tests ===
def test_grayscale_images(dct_fn, idct_fn):
    print("\n--- Testing Grayscale Images ---")
    images = ["data/nasir_ahmed.png"]
    for path in images:
        print(f"\nImage: {path}")
        img = normalize_image(load_grayscale_image(path))
        blocks, shape = split_into_blocks(img, block_size=8)

        dct_blocks = np.zeros_like(blocks, dtype=np.float64)
        total_time_energy = 0
        total_freq_energy = 0

        # DCT and energy computation per block
        for i in range(blocks.shape[0]):
            for j in range(blocks.shape[1]):
                block = blocks[i, j].astype(np.float64)
                dct_block = dct_fn(block)
                dct_blocks[i, j] = dct_block

                total_time_energy += np.sum(block ** 2)
                total_freq_energy += np.sum(dct_block ** 2)

        rel_error = abs(total_time_energy - total_freq_energy) / total_time_energy
        print(f"[Parseval] Time: {total_time_energy:.4f}, Freq: {total_freq_energy:.4f}, Rel Error: {rel_error:.2e}")
        assert rel_error < 1e-6

        # IDCT and reconstruction
        rec_blocks = np.zeros_like(dct_blocks)
        for i in range(dct_blocks.shape[0]):
            for j in range(dct_blocks.shape[1]):
                rec_blocks[i, j] = idct_fn(dct_blocks[i, j])

        img_rec = combine_blocks(rec_blocks, shape)

        assert verify_inverse_consistency(img, img_rec)
        psnr = compute_psnr(img, img_rec)
        ssim = compute_ssim(img, img_rec)
        print(f"[{path}] PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")


# === 2D RGB Image Tests ===
def test_rgb_images(dct_fn, idct_fn):
    print("\n--- Testing RGB Images ---")
    images = ["data/jack.jpg", "data/mary.jpg"]

    for path in images:
        print(f"\nImage: {path}")
        img = normalize_image(np.array(Image.open(path)))
        blocks, shape = split_into_blocks(img, block_size=8)

        dct_blocks = np.zeros_like(blocks, dtype=np.float64)
        total_time_energy = 0
        total_freq_energy = 0

        # Apply DCT to each block
        for i in range(blocks.shape[0]):
            for j in range(blocks.shape[1]):
                block = blocks[i, j].astype(np.float64)
                dct_block = dct_fn(block)
                dct_blocks[i, j] = dct_block

                total_time_energy += np.sum(block ** 2)
                total_freq_energy += np.sum(dct_block ** 2)

        rel_error = abs(total_time_energy - total_freq_energy) / total_time_energy
        print(f"[Parseval] Time: {total_time_energy:.4f}, Freq: {total_freq_energy:.4f}, Rel Error: {rel_error:.2e}")
        assert rel_error < 1e-6

        # Apply IDCT to each block
        # Reconstruct
        rec_blocks = np.zeros_like(dct_blocks)
        for i in range(dct_blocks.shape[0]):
            for j in range(dct_blocks.shape[1]):
                rec_blocks[i, j] = idct_fn(dct_blocks[i, j])

        img_rec = combine_blocks(rec_blocks, shape)

        # Now run the tests
        assert verify_inverse_consistency(img, img_rec)
        psnr = compute_psnr(img, img_rec)
        ssim = compute_ssim(img, img_rec)
        print(f"[{path}] PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")


if __name__ == "__main__":
    # Import your implementations here:
    test_1d_signals(dct1, idct1)
    test_grayscale_images(dct2, idct2)
    test_rgb_images(dct2rgb, idct2rgb)