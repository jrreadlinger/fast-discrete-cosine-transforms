import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dct

def dct_naive(x):
    N = len(x)
    X = np.zeros(N)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.cos(np.pi * (n + 0.5) * k / N)
    return X

# Create input signal
N = 32
x = np.sin(2 * np.pi * np.arange(N) / N) + 0.5 * np.sin(4 * np.pi * np.arange(N) / N)


X = dct_naive(x)
X_builtin = dct(x, type=2, norm=None)


plt.figure(figsize=(12, 5))

# Input signal
plt.subplot(1, 2, 1)
plt.plot(x, marker='o')
plt.title("Original Signal")
plt.xlabel("n")
plt.ylabel("x[n]")

# DCT output
plt.subplot(1, 2, 2)
plt.stem(X, basefmt=" ", use_line_collection=True)
plt.title("DCT-II Coefficients")
plt.xlabel("k")
plt.ylabel("X[k]")

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
plt.stem(X_builtin, basefmt=" ", use_line_collection=True)
plt.show()
