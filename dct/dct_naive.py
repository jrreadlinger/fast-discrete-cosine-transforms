import numpy as np

def dct1(x):
    """
    1 dimensional DCT-II
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
    1 dimensional DCT-III
    """
    N = len(X)
    x_rec = np.zeros(N, dtype=np.float64)
    for n in range(N):
        for k in range(N):
            alpha = np.sqrt(1 / N) if k == 0 else np.sqrt(2 / N)
            x_rec[n] += alpha * X[k] * np.cos(np.pi * k * (n + 0.5) / N)
    return x_rec

def dct2(block):
    """
    2 dimensional DCT-II
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
    2 dimensional DCT-III
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

def dct2rgb(image_rgb):
    """
    3 channel (RGB) 2 dimensional DCT-II
    """
    channels = []
    for i in range(3):
        channels.append(dct2(image_rgb[:, :, i]))
    return np.stack(channels, axis=2)

def idct2rgb(dct_rgb):
    """
    3 channel (RGB) 2 dimensional DCT-III
    """
    channels = []
    for i in range(3):
        channel = idct2(dct_rgb[:, :, i])
        channels.append(channel)
    return np.stack(channels, axis=2)