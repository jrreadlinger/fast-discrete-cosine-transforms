#from dct.utils import load_grayscale_image

import numpy as np
import matplotlib.pyplot as plt
import time
import math 

def orthonormalize_dct(X):
    N = len(X)
    scale = np.sqrt(2 / N) * np.ones(N)
    scale[0] = np.sqrt(1 / N)
    return X * scale

def fit_power_law(N_vals, times):
    logs_N = np.log2(N_vals)
    logs_T = np.log2(times)
    slope, intercept = np.polyfit(logs_N, logs_T, 1)
    return slope  # a in O(N^a)



def dct_naive(x):
    N = len(x)
    X = np.zeros(N)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.cos(np.pi * (n + 0.5) * k / N)
        X[k] *= 2  # scale by 2 for energy preservation
    return X


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

def inverse_transform(vector, root=True):
    vector = np.asarray(vector, dtype=float)
    n = len(vector)

    if root:
        vector[0] /= 2  # Apply scaling for the root call (corresponds to DCT scaling)

    if n == 1:
        return vector.copy()
    elif n == 0 or n % 2 != 0:
        raise ValueError("Length must be a power of 2")
    else:
        half = n // 2

        # Split into alpha and beta parts
        alpha = [vector[0]]
        beta = [vector[1]]

        for i in range(2, n, 2):
            alpha.append(vector[i])
            beta.append(vector[i - 1] + vector[i + 1])

        # Recursively invert alpha and beta
        alpha = inverse_transform(alpha, False)
        beta = inverse_transform(beta, False)

        # Combine alpha and beta to recover original vector
        result = np.zeros(n)

        for i in range(half):
            angle = math.pi * (i + 0.5) / n
            cos_val = math.cos(angle)
            y = beta[i] / (2 * cos_val)

            result[i] = alpha[i] + y
            result[-(i + 1)] = alpha[i] - y

        return result
    

def dct_lee(vector):
	if vector.ndim != 1:
		raise ValueError()
	n = vector.size
	if n == 1:
		return vector.copy()
	elif n == 0 or n % 2 != 0:
		raise ValueError()
	else:
		half = n // 2
		gamma = vector[ : half]
		delta = vector[n - 1 : half - 1 : -1]
		alpha = dct_lee(gamma + delta)
		beta  = dct_lee((gamma - delta) / (np.cos(np.arange(0.5, half + 0.5) * (np.pi / n)) * 2.0))
		result = np.zeros_like(vector)
		result[0 : : 2] = alpha
		result[1 : : 2] = beta
		result[1 : n - 1 : 2] += beta[1 : ]
		return result

def dct_lee_2d(matrix):
    matrix = np.asarray(matrix, dtype=float)
    
    if matrix.ndim != 2:
        raise ValueError("Input must be 2D")

    # Step 1: Apply DCT on rows
    temp = np.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        temp[i, :] = dct_lee(matrix[i, :])

    # Step 2: Apply DCT on columns
    result = np.zeros_like(matrix)
    for j in range(matrix.shape[1]):
        result[:, j] = dct_lee(temp[:, j])

    return result

def idct_lee(vector):
    vector = np.asarray(vector, dtype=float)
    if vector.ndim != 1:
        raise ValueError()

    result = inverse_transform(vector.copy())
    return np.array(result)

def fast_dct(x):
    """
    Compute the DCT of a 1D array using a simplified version of Algorithm 749.
    """
    N = len(x)
    X = np.zeros(N)
    for k in range(N):
        sum_val = 0
        for n in range(N):
            sum_val += x[n] * np.cos(np.pi * k * (2*n + 1) / (2 * N))
        X[k] = sum_val
    X[0] = X[0] / np.sqrt(N)
    X[1:] = X[1:] * np.sqrt(2 / N)
    return X

def idct_lee_2d(matrix):
    matrix = np.asarray(matrix, dtype=float)

    if matrix.ndim != 2:
        raise ValueError("Input must be 2D")

    # Step 1: Apply IDCT on columns
    temp = np.zeros_like(matrix)
    for j in range(matrix.shape[1]):
        temp[:, j] = idct_lee(matrix[:, j])

    # Step 2: Apply IDCT on rows
    result = np.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        result[i, :] = idct_lee(temp[i, :])

    return result



N = 32
x = np.sin(2 * np.pi * np.arange(N) / N) + 0.5 * np.sin(4 * np.pi * np.arange(N) / N)

#x = np.ones(8)

# X_naive = dct_naive(x)
# X_fast = fast_dct_recursive(x)
# X_naive_norm = X_naive / np.linalg.norm(X_naive)
# X_fast_norm = X_fast / np.linalg.norm(X_fast)

# Compute unnormalized DCTs
X_naive = dct_naive(x)
X_fast = fast_dct_recursive(x)
X_lee = dct_lee(x)
X_fast = fast_dct(x)

# Apply orthonormal scaling
X_naive_norm = orthonormalize_dct(X_naive)
X_fast_norm = orthonormalize_dct(X_fast)
X_lee_norm = orthonormalize_dct(X_lee)

plt.figure(figsize=(10, 5))
plt.plot(X_naive_norm, 'o-', label="Naive DCT (orthonormal)")
#plt.plot(X_fast_norm, 'x--', label="Fast DCT Recursive")
plt.plot(X_lee_norm, 's-.', label="Lee DCT")
plt.legend()
plt.title("Comparison of Orthonormal DCT-II Implementations")
plt.xlabel("k")
plt.ylabel("Coefficient value")
plt.grid(True)
plt.show()


err_fast = np.linalg.norm(X_naive_norm - X_fast_norm) / np.linalg.norm(X_naive_norm)
err_lee = np.linalg.norm(X_naive_norm - X_lee_norm) / np.linalg.norm(X_naive_norm)

print(f"Relative error (Fast Recursive vs Naive): {err_fast:.2e}")
print(f"Relative error (Lee vs Naive): {err_lee:.2e}")

# plt.plot(X_naive, 'o-', label='Naive DCT')
# plt.plot(X_fast, 'x--', label='Fast DCT (recursive)')
# plt.legend()
# plt.title("Comparison of Naive and Fast DCT-II")
# plt.show()


sizes = [2**i for i in range(4, 11)]  # N = 16 to 1024
naive_times = []
lee_times = []
faster = []

for N in sizes:
    x = np.random.rand(N)

    # Time naive DCT
    start = time.perf_counter()
    dct_naive(x)
    naive_times.append(time.perf_counter() - start)

    # Time Lee DCT
    start = time.perf_counter()
    dct_lee(x)
    lee_times.append(time.perf_counter() - start)

    # Time Lee DCT
    start = time.perf_counter()
    fast_dct(x)
    faster.append(time.perf_counter() - start)

plt.figure(figsize=(8, 5))
plt.loglog(sizes, naive_times, 'o-', label='Naive DCT')
plt.loglog(sizes, lee_times, 's--', label='Lee DCT (fast)')
plt.loglog(sizes, faster, 's--', label='faster DCT (fast)')
plt.xlabel("Input size N")
plt.ylabel("Execution time (seconds)")
plt.title("DCT Runtime Scaling")
plt.grid(True, which="both", ls=":")
plt.legend()
plt.show()


slope_naive = fit_power_law(sizes, naive_times)
slope_lee = fit_power_law(sizes, lee_times)

print(f"Naive DCT empirical complexity: O(N^{slope_naive:.2f})")
print(f"Lee DCT empirical complexity: O(N^{slope_lee:.2f})")
