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
        scale = np.sqrt(2/n) * np.ones(n)
        scale[0] = np.sqrt(1/n)
        return result*scale
        # return result

# "Butterfly algorithm": In this algorithm we hard code a size 8 1D vector to decrease the complexity from O(N^2) to O(NlogN)
# Constants for the transform
cos_vals = [math.cos(math.pi / 16 * i) for i in range(8)]
scale_factors = [1 / (4 * c) for c in cos_vals]
scale_factors[0] = 1 / (2 * math.sqrt(2))
transform_consts = [
    None,
    cos_vals[4],
    cos_vals[2] - cos_vals[6],
    cos_vals[4],
    cos_vals[6] + cos_vals[2],
    cos_vals[6],
]

def fast_dct(input_vec): # input vector must be size 8
    # Stage 1: pairwise sum/diff
    sum07 = input_vec[0] + input_vec[7]
    sum16 = input_vec[1] + input_vec[6]
    sum25 = input_vec[2] + input_vec[5]
    sum34 = input_vec[3] + input_vec[4]
    
    diff34 = input_vec[3] - input_vec[4]
    diff25 = input_vec[2] - input_vec[5]
    diff16 = input_vec[1] - input_vec[6]
    diff07 = input_vec[0] - input_vec[7]
    
    # Stage 2
    stage2_a = sum07 + sum34
    stage2_b = sum16 + sum25
    stage2_c = sum16 - sum25
    stage2_d = sum07 - sum34
    
    stage2_e = -diff34 - diff25
    stage2_f = (diff25 + diff16) * transform_consts[3]
    stage2_g = diff16 + diff07

    # Stage 3
    stage3_a = stage2_a + stage2_b
    stage3_b = stage2_a - stage2_b
    stage3_c = (stage2_c + stage2_d) * transform_consts[1]
    stage3_d = (stage2_e + stage2_g) * transform_consts[5]

    # Stage 4
    stage4_a = -stage2_e * transform_consts[2] - stage3_d
    stage4_b = stage2_g * transform_consts[4] - stage3_d

    # Stage 5
    stage5_a = stage3_c + stage2_d
    stage5_b = stage2_d - stage3_c
    stage5_c = stage2_f + diff07
    stage5_d = diff07 - stage2_f

    # Final recombination
    out0 = scale_factors[0] * stage3_a
    out1 = scale_factors[1] * (stage5_c + stage4_b)
    out2 = scale_factors[2] * stage5_a
    out3 = scale_factors[3] * (stage5_d - stage4_a)
    out4 = scale_factors[4] * stage3_b
    out5 = scale_factors[5] * (stage5_d + stage4_a)
    out6 = scale_factors[6] * stage5_b
    out7 = scale_factors[7] * (stage5_c - stage4_b)

    return [out0, out1, out2, out3, out4, out5, out6, out7]

def inverse_transform(input_vec):
    # Rescale
    v0 = input_vec[0] / scale_factors[0]
    v1 = input_vec[1] / scale_factors[1]
    v2 = input_vec[2] / scale_factors[2]
    v3 = input_vec[3] / scale_factors[3]
    v4 = input_vec[4] / scale_factors[4]
    v5 = input_vec[5] / scale_factors[5]
    v6 = input_vec[6] / scale_factors[6]
    v7 = input_vec[7] / scale_factors[7]

    # Reverse recombination
    half_sum_19_28 = (v5 - v3) / 2
    half_sum_20_27 = (v1 - v7) / 2
    avg_23 = (v1 + v7) / 2
    avg_24 = (v5 + v3) / 2

    approx07 = (avg_23 + avg_24) / 2
    approx11 = (v2 + v6) / 2
    approx13 = (avg_23 - avg_24) / 2
    approx17 = (v2 - v6) / 2

    sum015 = (v0 + v4) / 2
    diff015 = (v0 - v4) / 2

    # Solve linear system for internal vars
    denom = (transform_consts[2] * transform_consts[5]
             - transform_consts[2] * transform_consts[4]
             - transform_consts[4] * transform_consts[5])
    
    dct18 = (half_sum_19_28 - half_sum_20_27) * transform_consts[5]
    dct12 = (half_sum_19_28 * transform_consts[4] - dct18) / denom
    dct14 = (dct18 - half_sum_20_27 * transform_consts[2]) / denom

    # Final reconstruction
    v6 = dct14 - approx07
    v5 = approx13 / transform_consts[3] - v6
    v4 = -v5 - dct12
    v10 = approx17 / transform_consts[1] - approx11

    x0 = (sum015 + approx11) / 2
    x1 = (diff015 + v10) / 2
    x2 = (diff015 - v10) / 2
    x3 = (sum015 - approx11) / 2

    return [
        (x0 + approx07) / 2,
        (x1 + v6) / 2,
        (x2 + v5) / 2,
        (x3 + v4) / 2,
        (x3 - v4) / 2,
        (x2 - v5) / 2,
        (x1 - v6) / 2,
        (x0 - approx07) / 2,
    ]

def fast_idct(X):
    N = len(X)
    x = np.zeros(N)

    for n in range(N):
        sum_val = 0.0
        for k in range(N):
            coeff = X[k]
            if k == 0:
                coeff *= np.sqrt(1 / N)
            else:
                coeff *= np.sqrt(2 / N)

            sum_val += coeff * np.cos(np.pi * k * (2 * n + 1) / (2 * N))

        x[n] = sum_val

    return x




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