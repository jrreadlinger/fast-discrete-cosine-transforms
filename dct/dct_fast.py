import numpy as np
import math

### === 1D DCT-II ===
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


### === 1D DCT-III ===
def idct_lee(vector):
    vector = np.asarray(vector, dtype=float)
    if vector.ndim != 1:
        raise ValueError()

    result = inverse_transform(vector.copy())
    return np.array(result)


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


### === 2D DCT & IDCT ===
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


### === RGB Extensions ===
def dct_lee_2d_rgb(image_rgb):
    """
    Apply 2D DCT-II to each RGB channel independently.

    Args:
        image_rgb (np.ndarray): H x W x 3 RGB image

    Returns:
        np.ndarray: DCT coefficients, shape H x W x 3
    """
    return np.stack([dct_lee_2d(image_rgb[:, :, c]) for c in range(3)], axis=2)


def idct_lee_2d_rgb(dct_rgb):
    """
    Apply 2D inverse DCT to each RGB channel.

    Args:
        dct_rgb (np.ndarray): H x W x 3 DCT coefficients

    Returns:
        np.ndarray: Reconstructed RGB image (float64)
    """
    return np.stack([idct_lee_2d(dct_rgb[:, :, c]) for c in range(3)], axis=2)