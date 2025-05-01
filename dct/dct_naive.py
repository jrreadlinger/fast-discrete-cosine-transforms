from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dct

# 1d implementations
def dct_naive(x):
    N = len(x)
    X = np.zeros(N)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.cos(np.pi * (n + 0.5) * k / N)
        X[k] *= 2  # scale by 2 for energy preservation
    return X


def idct_naive(X):
    N = len(X)
    x_rec = np.zeros(N)
    for n in range(N):
        sum_val = 0.5 * X[0]
        for k in range(1, N):
            sum_val += X[k] * np.cos(np.pi * k * (n + 0.5) / N)
        x_rec[n] = sum_val / N  # normalize by N
    return x_rec


# grayscale implementations
# def dct2_naive(block):
#     N, M = block.shape
#     result = np.zeros((N, M))
#     for u in range(N):
#         for v in range(M):
#             sum_val = 0
#             for i in range(N):
#                 for j in range(M):
#                     sum_val += block[i, j] * np.cos(np.pi * (i + 0.5) * u / N) * np.cos(np.pi * (j + 0.5) * v / M)
#             result[u, v] = 4 * sum_val
#     return result

def dct2_naive(block):
    N, M = block.shape
    X = np.zeros((N, M))
    for u in range(N):
        for v in range(M):
            alpha_u = np.sqrt(1 / N) if u == 0 else np.sqrt(2 / N)
            alpha_v = np.sqrt(1 / M) if v == 0 else np.sqrt(2 / M)
            sum_val = 0
            for i in range(N):
                for j in range(M):
                    sum_val += block[i, j] * np.cos(np.pi * (i + 0.5) * u / N) * np.cos(np.pi * (j + 0.5) * v / M)
            X[u, v] = alpha_u * alpha_v * sum_val
    return X


# def idct2_naive(block):
#     N, M = block.shape
#     result = np.zeros((N, M))
#     for i in range(N):
#         for j in range(M):
#             sum_val = 0.5 * block[0, 0]
#             for u in range(N):
#                 for v in range(M):
#                     alpha_u = 1 if u == 0 else 2
#                     alpha_v = 1 if v == 0 else 2
#                     sum_val += (block[u, v] *
#                                 np.cos(np.pi * (i + 0.5) * u / N) *
#                                 np.cos(np.pi * (j + 0.5) * v / M) *
#                                 alpha_u * alpha_v / 4)
#             result[i, j] = sum_val / (N * M)
#     return result

def idct2_naive(X):
    N, M = X.shape
    x = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            sum_val = 0
            for u in range(N):
                for v in range(M):
                    alpha_u = np.sqrt(1 / N) if u == 0 else np.sqrt(2 / N)
                    alpha_v = np.sqrt(1 / M) if v == 0 else np.sqrt(2 / M)
                    sum_val += alpha_u * alpha_v * X[u, v] * \
                               np.cos(np.pi * (i + 0.5) * u / N) * \
                               np.cos(np.pi * (j + 0.5) * v / M)
            x[i, j] = sum_val
    return x

# rgb implementations
def dct2_rgb(image_rgb):
    """Apply DCT to each RGB channel separately."""
    channels = []
    for i in range(3):
        channels.append(dct2_naive(image_rgb[:, :, i]))
    return np.stack(channels, axis=2)


def idct2_rgb(dct_rgb):
    """Apply inverse DCT to each RGB channel."""
    channels = []
    for i in range(3):
        channels.append(idct2_naive(dct_rgb[:, :, i]))
    return np.stack(channels, axis=2)


# # Create input signal
# N = 32
# x = np.sin(2 * np.pi * np.arange(N) / N) + 0.5 * np.sin(4 * np.pi * np.arange(N) / N)


# X = dct_naive(x)
# X_builtin = dct(x, type=2, norm=None)

# # Reconstruct the signal from DCT coefficients
# x_rec = idct_naive(X)

# # Plot original vs reconstructed
# plt.figure(figsize=(12, 5))

# plt.plot(x, label="Original", marker='o')
# plt.plot(x_rec, label="Reconstructed", marker='x')
# plt.title("Signal Reconstruction via Naive DCT and IDCT")
# plt.xlabel("n")
# plt.ylabel("Signal Value")
# plt.legend()
# plt.grid(True)
# plt.show()



# plt.figure(figsize=(12, 5))

# # Input signal
# plt.subplot(1, 2, 1)
# plt.plot(x, marker='o')
# plt.title("Original Signal")
# plt.xlabel("n")
# plt.ylabel("x[n]")

# # DCT output
# plt.subplot(1, 2, 2)
# plt.stem(X, basefmt=" ", use_line_collection=True)
# plt.title("DCT-II Coefficients")
# plt.xlabel("k")
# plt.ylabel("X[k]")

# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(12, 5))
# plt.stem(X_builtin, basefmt=" ", use_line_collection=True)
# plt.show()

# rel_error = np.linalg.norm(x - x_rec) / np.linalg.norm(x)
# print(f"Relative error: {rel_error:.2e}")