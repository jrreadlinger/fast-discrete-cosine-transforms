import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from PIL import Image
from time import perf_counter
from dct.dct_naive import dct1, idct1, dct2, idct2, dct2rgb, idct2rgb
from dct.dct_fast import dct_lee, idct_lee, dct_lee_2d, idct_lee_2d, dct_lee_2d_rgb, idct_lee_2d_rgb
from dct.utils import (
    load_grayscale_image,
    split_into_blocks,
    combine_blocks,
    compute_psnr,
    compute_ssim,
    normalize_image,
    mask_dct_coefficients,
    zigzag_mask_dct
)

os.makedirs("plots", exist_ok=True)

# 1d reconstruction
def plot_1d_reconstruction(name, signal):
    X = dct1(signal)
    signal_rec = idct1(X)

    plt.figure()
    plt.plot(signal, label="Original", marker='o')
    plt.plot(signal_rec, label="Reconstructed", marker='x')
    plt.title(f"1D Signal Reconstruction: {name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plots/1d_recon_{name}.png")
    plt.close()

    plt.figure()
    plt.stem(X)
    plt.title(f"DCT Coefficients: {name}")
    plt.grid(True)
    plt.savefig(f"plots/1d_coeff_{name}.png")
    plt.close()


# image reconstruction with blocks
# def plot_image_reconstruction(image_path, color=False):
#     name = os.path.splitext(os.path.basename(image_path))[0]
#     block_size = 8

#     if color:
#         img = normalize_image(np.array(Image.open(image_path)))
#         blocks, shape = split_into_blocks(img, block_size)
#         dct_blocks = np.zeros_like(blocks, dtype=np.float64)
#         for i in range(blocks.shape[0]):
#             for j in range(blocks.shape[1]):
#                 dct_blocks[i, j] = dct2rgb(blocks[i, j])
#         rec_blocks = np.zeros_like(dct_blocks)
#         for i in range(dct_blocks.shape[0]):
#             for j in range(dct_blocks.shape[1]):
#                 rec_blocks[i, j] = idct2rgb(dct_blocks[i, j])
#         img_rec = combine_blocks(rec_blocks, shape)
#     else:
#         img = normalize_image(load_grayscale_image(image_path))
#         blocks, shape = split_into_blocks(img, block_size)
#         dct_blocks = np.zeros_like(blocks, dtype=np.float64)
#         for i in range(blocks.shape[0]):
#             for j in range(blocks.shape[1]):
#                 dct_blocks[i, j] = dct2(blocks[i, j])
#         rec_blocks = np.zeros_like(dct_blocks)
#         for i in range(dct_blocks.shape[0]):
#             for j in range(dct_blocks.shape[1]):
#                 rec_blocks[i, j] = idct2(dct_blocks[i, j])
#         img_rec = combine_blocks(rec_blocks, shape)

#     psnr = compute_psnr(img, img_rec)
#     ssim = compute_ssim(img, img_rec)

#     fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#     axs[0].imshow(img.astype(np.uint8) if color else img, cmap=None if color else 'gray')
#     axs[0].set_title("Original")
#     axs[0].axis('off')
#     axs[1].imshow(img_rec.astype(np.uint8) if color else img_rec, cmap=None if color else 'gray')
#     axs[1].set_title(f"Reconstructed\nPSNR: {psnr:.2f}, SSIM: {ssim:.4f}")
#     axs[1].axis('off')
#     plt.tight_layout()
#     path = f"plots/image_recon_{name}"
#     if color:
#         path += "_color.png"
#     else:
#         path += ".png"
#     plt.savefig(path)
#     plt.close()

# def plot_image_reconstruction(image_path, color=False):
#     name = os.path.splitext(os.path.basename(image_path))[0]
#     block_size = 8

#     if color:
#         img = normalize_image(np.array(Image.open(image_path)))
#         raise NotImplementedError("DCT coefficient heatmap not implemented for RGB yet.")
#     else:
#         img = normalize_image(load_grayscale_image(image_path))
#         blocks, shape = split_into_blocks(img, block_size)
#         dct_blocks = np.zeros_like(blocks, dtype=np.float64)
#         for i in range(blocks.shape[0]):
#             for j in range(blocks.shape[1]):
#                 dct_blocks[i, j] = dct2(blocks[i, j])
#         rec_blocks = np.zeros_like(dct_blocks)
#         for i in range(dct_blocks.shape[0]):
#             for j in range(dct_blocks.shape[1]):
#                 rec_blocks[i, j] = idct2(dct_blocks[i, j])
#         img_rec = combine_blocks(rec_blocks, shape)

#         # Visualize DCT coefficient magnitudes (log scale for contrast)
#         abs_dct = np.abs(combine_blocks(dct_blocks, shape))
#         log_dct = np.log1p(abs_dct)  # log(1 + x) to compress dynamic range

#         psnr = compute_psnr(img, img_rec)
#         ssim = compute_ssim(img, img_rec)

#         fig, axs = plt.subplots(1, 3, figsize=(15, 5))
#         axs[0].imshow(img, cmap='gray')
#         axs[0].set_title("Original")
#         axs[0].axis('off')

#         im = axs[1].imshow(log_dct, cmap='viridis')
#         axs[1].set_title("DCT Coefficient Heatmap (log scale)")
#         axs[1].axis('off')
#         fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)

#         axs[2].imshow(img_rec, cmap='gray')
#         axs[2].set_title(f"Reconstructed\nPSNR: {psnr:.2f}, SSIM: {ssim:.4f}")
#         axs[2].axis('off')

#         plt.tight_layout()
#         plt.savefig(f"plots/image_recon_{name}.png")
#         plt.close()

def plot_image_reconstruction(image_path, color=False):
    name = os.path.splitext(os.path.basename(image_path))[0]
    block_size = 8

    if color:
        img = normalize_image(np.array(Image.open(image_path)))
        blocks, shape = split_into_blocks(img, block_size)
        dct_blocks = np.zeros_like(blocks, dtype=np.float64)
        for i in range(blocks.shape[0]):
            for j in range(blocks.shape[1]):
                dct_blocks[i, j] = dct2rgb(blocks[i, j])
        rec_blocks = np.zeros_like(dct_blocks)
        for i in range(dct_blocks.shape[0]):
            for j in range(dct_blocks.shape[1]):
                rec_blocks[i, j] = idct2rgb(dct_blocks[i, j])
        img_rec = combine_blocks(rec_blocks, shape)
    else:
        img = normalize_image(load_grayscale_image(image_path))
        blocks, shape = split_into_blocks(img, block_size)
        dct_blocks = np.zeros_like(blocks, dtype=np.float64)
        for i in range(blocks.shape[0]):
            for j in range(blocks.shape[1]):
                dct_blocks[i, j] = dct2(blocks[i, j])
        rec_blocks = np.zeros_like(dct_blocks)
        for i in range(dct_blocks.shape[0]):
            for j in range(dct_blocks.shape[1]):
                rec_blocks[i, j] = idct2(dct_blocks[i, j])
        img_rec = combine_blocks(rec_blocks, shape)

    psnr = compute_psnr(img, img_rec)
    ssim = compute_ssim(img, img_rec)

    if color:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(img.astype(np.uint8))
        axs[0].set_title("Original")
        axs[0].axis('off')
        axs[1].imshow(img_rec.astype(np.uint8))
        axs[1].set_title(f"Reconstructed\nPSNR: {psnr:.2f}, SSIM: {ssim:.4f}")
        axs[1].axis('off')
    else:
        avg_mag = np.mean(np.abs(dct_blocks), axis=(0, 1))

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(img, cmap='gray')
        axs[0].set_title("Original")
        axs[0].axis('off')

        im = axs[1].imshow(avg_mag, cmap='plasma', norm=LogNorm(vmin=1e-2, vmax=avg_mag.max()))
        axs[1].set_title("Average DCT Coefficient Magnitudes")
        axs[1].axis('off')
        fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)

        axs[2].imshow(img_rec, cmap='gray')
        axs[2].set_title(f"Reconstructed\nPSNR: {psnr:.2f}, SSIM: {ssim:.4f}")
        axs[2].axis('off')

    plt.tight_layout()
    suffix = "_color.png" if color else ".png"
    plt.savefig(f"plots/image_recon_{name}{suffix}")
    plt.close()


# runtime comparison
def plot_runtime_comparison(algorithms):
    sizes = [8, 16, 32, 64, 128]
    timings = {label: [] for label in algorithms}

    for N in sizes:
        data = np.random.rand(N, N)
        for label, func in algorithms.items():
            start = perf_counter()
            func(data)
            timings[label].append(perf_counter() - start)

    plt.figure()
    for label, times in timings.items():
        plt.plot(sizes, times, label=label, marker='o')
    plt.title("2D DCT Runtime Comparison")
    plt.xlabel("Image Size (NxN)")
    plt.ylabel("Time (s, log scale)")
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.savefig("plots/runtime_comparison.png")
    plt.close()


# compression demo
def plot_compression_curve_and_examples(image_path, percentages=[1, 5, 10, 25, 50, 75, 100], show_examples=True):
    name = os.path.splitext(os.path.basename(image_path))[0]
    img = normalize_image(load_grayscale_image(image_path))
    block_size = 8
    blocks, shape = split_into_blocks(img, block_size)
    dct_blocks = np.zeros_like(blocks, dtype=np.float64)
    for i in range(blocks.shape[0]):
        for j in range(blocks.shape[1]):
            dct_blocks[i, j] = dct2(blocks[i, j])

    psnrs_zigzag = []
    ssims_zigzag = []

    if show_examples:
        fig, axs = plt.subplots(1, len(percentages) + 1, figsize=(4 * (len(percentages) + 1), 5))
        axs[0].imshow(img, cmap='gray')
        axs[0].set_title("Original")
        axs[0].axis('off')

    for idx, pct in enumerate(percentages):
        p = pct / 100.0
        rec_blocks = np.zeros_like(dct_blocks)
        for i in range(dct_blocks.shape[0]):
            for j in range(dct_blocks.shape[1]):
                masked = zigzag_mask_dct(dct_blocks[i, j], int(block_size * block_size * p))
                rec_blocks[i, j] = idct2(masked)
        img_rec = combine_blocks(rec_blocks, shape)

        psnr = compute_psnr(img, img_rec)
        ssim = compute_ssim(img, img_rec)
        psnrs_zigzag.append(psnr)
        ssims_zigzag.append(ssim)

        if show_examples:
            axs[idx + 1].imshow(img_rec, cmap='gray')
            axs[idx + 1].set_title(f"{pct}% Kept\nPSNR: {psnr:.1f}, SSIM: {ssim:.2f}")
            axs[idx + 1].axis('off')

    if show_examples:
        plt.tight_layout()
        plt.savefig(f"plots/compression_examples_{name}.png")
        plt.close()

    plt.figure()
    plt.plot(percentages, psnrs_zigzag, marker='x', label="Zigzag Mask")
    plt.title("PSNR vs % Coefficients Kept")
    plt.xlabel("% Coefficients")
    plt.ylabel("PSNR (dB)")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/psnr_comparison.png")
    plt.close()

    plt.figure()
    plt.plot(percentages, ssims_zigzag, marker='x', label="Zigzag Mask")
    plt.title("SSIM vs % Coefficients Kept")
    plt.xlabel("% Coefficients")
    plt.ylabel("SSIM")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/ssim_comparison.png")
    plt.close()

def plot_compression_curve(image_path):
    block_size = 8
    img = normalize_image(load_grayscale_image(image_path))
    blocks, shape = split_into_blocks(img, block_size)
    dct_blocks = np.zeros_like(blocks, dtype=np.float64)
    for i in range(blocks.shape[0]):
        for j in range(blocks.shape[1]):
            dct_blocks[i, j] = dct2(blocks[i, j])

    percentages = [1, 5, 10, 25, 50, 75, 100]
    psnrs_square = []
    psnrs_zigzag = []
    ssims_square = []
    ssims_zigzag = []

    for pct in percentages:
        p = pct / 100.0

        masked_sq = np.zeros_like(dct_blocks)
        masked_zz = np.zeros_like(dct_blocks)

        for i in range(dct_blocks.shape[0]):
            for j in range(dct_blocks.shape[1]):
                masked_sq[i, j] = mask_dct_coefficients(dct_blocks[i, j], int(block_size * p))
                masked_zz[i, j] = zigzag_mask_dct(dct_blocks[i, j], p)

        rec_sq = np.zeros_like(masked_sq)
        rec_zz = np.zeros_like(masked_zz)
        for i in range(masked_sq.shape[0]):
            for j in range(masked_sq.shape[1]):
                rec_sq[i, j] = idct2(masked_sq[i, j])
                rec_zz[i, j] = idct2(masked_zz[i, j])

        img_sq = combine_blocks(rec_sq, shape)
        img_zz = combine_blocks(rec_zz, shape)

        psnrs_square.append(compute_psnr(img, img_sq))
        ssims_square.append(compute_ssim(img, img_sq))
        psnrs_zigzag.append(compute_psnr(img, img_zz))
        ssims_zigzag.append(compute_ssim(img, img_zz))

    plt.figure()
    plt.plot(percentages, psnrs_square, marker='o', label="Square Mask")
    plt.plot(percentages, psnrs_zigzag, marker='x', label="Zigzag Mask")
    plt.title("PSNR vs % Coefficients Kept")
    plt.xlabel("% Coefficients")
    plt.ylabel("PSNR (dB)")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/psnr_comparison.png")
    plt.close()

    plt.figure()
    plt.plot(percentages, ssims_square, marker='o', label="Square Mask")
    plt.plot(percentages, ssims_zigzag, marker='x', label="Zigzag Mask")
    plt.title("SSIM vs % Coefficients Kept")
    plt.xlabel("% Coefficients")
    plt.ylabel("SSIM")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/ssim_comparison.png")
    plt.close()


# cumulative energy plot
def plot_cumulative_energy(signal, name):
    X = dct1(signal)
    sorted_energy = np.sort(np.abs(X)**2)[::-1]
    cumulative = np.cumsum(sorted_energy)
    cumulative /= cumulative[-1]

    plt.figure()
    plt.plot(np.linspace(0, 100, len(cumulative)), cumulative * 100, marker='.')
    plt.title(f"Cumulative Energy Curve: {name}")
    plt.xlabel("% of Coefficients")
    plt.ylabel("% of Total Energy")
    plt.grid(True)
    plt.savefig(f"plots/cumulative_energy_{name}.png")
    plt.close()


if __name__ == "__main__":
    # signals = {
    #     "impulse": np.load("data/impulse_signal.npy"),
    #     "sine": np.load("data/sine_signal.npy"),
    #     "step": np.load("data/step_signal.npy")
    # }
    # for name, x in signals.items():
    #     plot_1d_reconstruction(name, x)
    #     plot_cumulative_energy(x, name)
    # for path in [
    #     "data/checkerboard_large.jpg",
    #     "data/checkerboard_small.png",
    #     "data/dome.jpg"
    # ]:
    #     plot_image_reconstruction(path, color=False)
    # for path in [
    #     "data/creeper-face.png",
    #     "data/jack.jpg",
    #     "data/mary_art.jpg",
    #     "data/roller_rink_carpet.jpg",
    #     "data/steve-face.png"
    # ]:
    #     plot_image_reconstruction(path, color=True)
    # plot_image_reconstruction("data/nasir_ahmed.png", color=False)
    # plot_image_reconstruction("data/mary.jpg", color=True)
    # plot_runtime_comparison({
    #     "Naive DCT": dct2,
    #     "Fast DCT (Lee)": dct_lee_2d
    # })
    plot_compression_curve("data/nasir_ahmed.png")

    print("Plots generated in /plots")
