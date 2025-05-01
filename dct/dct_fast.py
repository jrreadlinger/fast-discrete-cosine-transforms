from dct.utils import load_grayscale_image

import numpy as np

def fast_dct_recursive(x):
    N = len(x)
    x = np.asarray(x, dtype=float)

    if N == 1:
        return x.copy()

    if N % 2 != 0:
        raise ValueError("Input length must be a power of 2")

    # Step 1: Split into even and odd indices
    x_even = x[::2]
    x_odd = x[1::2]

    # Step 2: Compute DCTs of each half recursively
    X_even = fast_dct_recursive(x_even)
    X_odd = fast_dct_recursive(x_odd)

    # Step 3: Combine
    X = np.zeros(N)
    for k in range(N // 2):
        angle = np.pi * (2 * k + 1) / (2 * N)
        cos_val = np.cos(angle)
        sin_val = np.sin(angle)

        X[k] = X_even[k] + cos_val * x_odd[k] + sin_val * x_odd[::-1][k]
        X[N - 1 - k] = X_even[k] - cos_val * x_odd[k] - sin_val * x_odd[::-1][k]

    return X

N = 32
x = np.sin(2 * np.pi * np.arange(N) / N) + 0.5 * np.sin(4 * np.pi * np.arange(N) / N)

X_naive = dct_naive(x)
X_fast = fast_dct_recursive(x)

print("Relative error:", np.linalg.norm(X_naive - X_fast) / np.linalg.norm(X_naive))

plt.plot(X_naive, 'o-', label='Naive DCT')
plt.plot(X_fast, 'x--', label='Fast DCT (recursive)')
plt.legend()
plt.title("Comparison of Naive and Fast DCT-II")
plt.show()
