import numpy as np

# === 1D DCT/IDCT ===
def dct1(x):
    """
    Compute the 1D DCT-II of a 1D signal using a naive implementation.

    Args:
        x (np.ndarray): 1D input signal (real-valued)

    Returns:
        np.ndarray: DCT coefficients (float64)
    """
    N = len(x)
    X = np.zeros(N, dtype=np.float64)
    for k in range(N):
        alpha = np.sqrt(1 / N) if k == 0 else np.sqrt(2 / N)
        for n in range(N):
            X[k] += x[n] * np.cos(np.pi * (n + 0.5) * k / N)
        X[k] *= alpha
    return X

def idct1(X):
    """
    Compute the 1D Inverse DCT (DCT-III) of a coefficient array.

    Args:
        X (np.ndarray): DCT coefficients

    Returns:
        np.ndarray: Reconstructed time-domain signal (float64)
    """
    N = len(X)
    x_rec = np.zeros(N, dtype=np.float64)
    for n in range(N):
        for k in range(N):
            alpha = np.sqrt(1 / N) if k == 0 else np.sqrt(2 / N)
            x_rec[n] += alpha * X[k] * np.cos(np.pi * k * (n + 0.5) / N)
    return x_rec


# === 2D DCT/IDCT ===
def dct2(block):
    """
    Compute the 2D DCT-II of a 2D image block.

    Args:
        block (np.ndarray): 2D array (e.g., 8x8 block)

    Returns:
        np.ndarray: 2D DCT coefficients
    """
    N, M = block.shape
    X = np.zeros((N, M), dtype=np.float64)
    for u in range(N):
        for v in range(M):
            alpha_u = np.sqrt(1 / N) if u == 0 else np.sqrt(2 / N)
            alpha_v = np.sqrt(1 / M) if v == 0 else np.sqrt(2 / M)
            sum_val = 0
            for i in range(N):
                for j in range(M):
                    sum_val += block[i, j] * \
                               np.cos(np.pi * (i + 0.5) * u / N) * \
                               np.cos(np.pi * (j + 0.5) * v / M)
            X[u, v] = alpha_u * alpha_v * sum_val
    return X

def idct2(X):
    """
    Compute the 2D Inverse DCT (DCT-III) of a coefficient block.

    Args:
        X (np.ndarray): 2D DCT coefficients

    Returns:
        np.ndarray: Reconstructed spatial-domain block
    """
    N, M = X.shape
    x = np.zeros((N, M), dtype=np.float64)
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


# === RGB Extensions ===
def dct2rgb(image_rgb):
    """
    Apply 2D DCT to each RGB channel independently.

    Args:
        image_rgb (np.ndarray): 3D RGB image array (H x W x 3)

    Returns:
        np.ndarray: DCT coefficients for each channel
    """
    channels = []
    for i in range(3):
        channels.append(dct2(image_rgb[:, :, i]))
    return np.stack(channels, axis=2)

def idct2rgb(dct_rgb):
    """
    Apply 2D IDCT to each RGB channel independently.

    Args:
        dct_rgb (np.ndarray): 3D DCT coefficients (H x W x 3)

    Returns:
        np.ndarray: Reconstructed RGB image (float64)
    """
    channels = []
    for i in range(3):
        channel = idct2(dct_rgb[:, :, i])
        channels.append(channel)
    return np.stack(channels, axis=2)