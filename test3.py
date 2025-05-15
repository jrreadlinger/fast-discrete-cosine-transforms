from PIL import Image
from dct.dct_naive import dct1, idct1
from dct.dct_fast import dct_lee, idct_lee, fast_dct, fast_idct
from dct.utils import load_grayscale_image
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import dct


# # Generate random signal
# N = 64
# # x = np.ones(N)
# x = np.random.rand(N)
# # N = 32
# # x = np.sin(2 * np.pi * np.arange(N) / N) + 0.5 * np.sin(4 * np.pi * np.arange(N) / N)


# # Compute DCT using naive implementation
# X_naive = dct1(x)

# # Compute DCT using Lee's fast implementation
# X_lee = dct_lee(x)



# # Normalize both for fair comparison (optional but recommended)
# def orthonormalize_dct(X):
#     N = len(X)
#     scale = np.sqrt(2 / N) * np.ones(N)
#     scale[0] = np.sqrt(1 / N)
#     return X * scale

# X_naive_norm = orthonormalize_dct(X_naive)
# X_lee_norm = orthonormalize_dct(X_lee)

# X_recon_naive = idct1(X_naive)

# X_recon_lee = idct_lee(X_lee)

# # Plotting the DCT coefficients
# plt.figure(figsize=(10, 5))

# plt.plot(X_naive, 'o-', label="lee DCT", alpha=0.7)
# plt.plot(X_lee, 'x--', label="naive DCT", alpha=0.7)
# # plt.plot(x, label="og")
# plt.xlabel("Coefficient Index (k)")
# plt.ylabel("DCT Coefficient Value")
# plt.title("DCT Coefficients: Naive vs Lee")
# plt.legend()
# plt.grid(True)
# plt.show()

# # Optionally print relative error
# error = np.linalg.norm(X_recon_lee - X_recon_naive) / np.linalg.norm(X_recon_naive)
# print(f"Relative error between naive and Lee DCT: {error:.2e}")


# Sample input
x = np.random.rand(16)

# N = 32
# x = np.sin(2 * np.pi * np.arange(N) / N) + 0.5 * np.sin(4 * np.pi * np.arange(N) / N)


# Compute DCT using Lee's algorithm
X_fast = fast_dct(x)
X_lee = dct_lee(x)
X_naive = dct1(x)
X_fast_recon = fast_idct(X_fast)
X_lee_recon = dct_lee(X_lee)
X_naive_recon = idct1(X_naive)

# Compute DCT using SciPy
X_scipy = dct(x, type=2, norm='ortho')

# Plotting the DCT coefficients
plt.figure(figsize=(10, 5))

plt.plot(X_lee, 'o-', label="fast", alpha=0.7)
plt.plot(X_naive, 'x--', label="naive", alpha=0.7)
plt.plot(X_scipy, label="Scipy built in")
# plt.plot(x, 'o-', label="original input", alpha=0.7)
# plt.plot(X_fast, 'o-', label="fast", alpha=0.7)
# plt.plot(X_naive, 'x--', label="naive", alpha=0.7)
# plt.plot(X_naive, label="naive")
plt.title("DCT random input coefficients comparison")
plt.xlabel("nth array entry")
plt.ylabel("Value")
plt.legend()
plt.show()

# plt.plot(X_lee_recon, 'o-', label="Lee's", alpha=0.7)
# plt.plot(X_naive_recon, 'x--', label="naive", alpha=0.7)
# plt.plot(x, 'o-', label="original input", alpha=0.7)
# # plt.plot(X_fast, 'o-', label="fast", alpha=0.7)
# # plt.plot(X_naive, 'x--', label="naive", alpha=0.7)
# # plt.plot(X_naive, label="naive")
# plt.title("DCT random input reconstruction: Lee's algorithm vs. Naive")
# plt.xlabel("nth array entry")
# plt.ylabel("Value")
# plt.legend()
# plt.show()

# Compare results
error = np.linalg.norm(X_lee - X_scipy)
print(f"Error between implementations: {error}")
