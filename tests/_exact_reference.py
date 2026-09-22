"""Platform-independent exact references for numerical test oracles.

``np.longdouble`` is float64 on Windows and macOS ARM, so it cannot serve as a
higher-precision reference there. These helpers use only float64 error-free
transformations, whose exactness is a property of IEEE binary64 itself:

* TwoSum (Knuth) and TwoProduct (Dekker's product with Veltkamp's split), as
  Algorithms 3.1-3.3 of Ogita, Rump and Oishi, "Accurate Sum and Dot Product",
  SIAM J. Sci. Comput. 26(6), 2005;
* the error-free splitting of a matrix product into a sum of slice products
  that BLAS evaluates without rounding, from Ozaki, Ogita, Oishi and Rump,
  "Error-free transformations of matrix multiplication by using fast routines
  of matrix multiplication and its applications", Numer. Algorithms 59, 2012;
* ``math.fsum``, which rounds the exact sum of its terms correctly
  (Shewchuk, "Adaptive Precision Floating-Point Arithmetic and Fast Robust
  Geometric Predicates", Discrete Comput. Geom. 18, 1997).

Caller contract: every nonzero input magnitude lies in [2**-450, 2**450] and
inner dimensions stay below 2**30. Then no split overflows and no slice product
underflows, so every product below is exact and each result is the correctly
rounded exact value.
"""

import math

import numpy as np

_SPLITTER = 2.0**27 + 1
_RANGE = (2.0**-450, 2.0**450)


def _require_representable(*arrays):
    magnitudes = np.abs(np.concatenate([np.ravel(array) for array in arrays]))
    nonzero = magnitudes[magnitudes != 0]
    if not np.all((nonzero >= _RANGE[0]) & (nonzero <= _RANGE[1])):
        raise ValueError("exact reference inputs leave the error-free exponent range")


def two_sum(a, b):
    """Return ``(s, e)`` with ``s = fl(a + b)`` and ``s + e == a + b`` exactly."""
    total = a + b
    shifted = total - a
    return total, (a - (total - shifted)) + (b - shifted)


def _veltkamp(a):
    scaled = _SPLITTER * a
    high = scaled - (scaled - a)
    return high, a - high


def two_product(a, b):
    """Return ``(p, e)`` with ``p = fl(a * b)`` and ``p + e == a * b`` exactly."""
    a, b = np.broadcast_arrays(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))
    _require_representable(a, b)
    product = a * b
    (a_high, a_low), (b_high, b_low) = _veltkamp(a), _veltkamp(b)
    error = a_low * b_low - (((product - a_high * b_high) - a_low * b_high) - a_high * b_low)
    return product, error


def _row_slices(matrix, shift):
    # Each slice holds at most 53 - shift significant bits per row, aligned to
    # the row maximum, so two slices' products sum exactly over the inner axis.
    slices = []
    rest = matrix
    while True:
        _, exponent = np.frexp(np.max(np.abs(rest), axis=1, keepdims=True))
        sigma = np.ldexp(1.0, exponent + shift)
        head = (rest + sigma) - sigma
        slices.append(head)
        rest = rest - head
        if not rest.any():
            return np.stack(slices)


def _slice_products(left, right):
    left = np.atleast_2d(np.asarray(left, dtype=np.float64))
    right = np.asarray(right, dtype=np.float64).reshape(left.shape[1], -1)
    _require_representable(left, right)
    shift = math.ceil((53 + math.ceil(math.log2(max(left.shape[1], 2)))) / 2)
    left_slices = _row_slices(left, shift)
    right_slices = np.swapaxes(_row_slices(right.T, shift), 1, 2)
    products = np.matmul(left_slices[:, None], right_slices[None])
    return products.reshape(-1, *products.shape[2:])


def exact_matmul(*pairs):
    """Correctly rounded ``sum(left @ right for left, right in pairs)``.

    Each ``left`` is a vector or matrix and each ``right`` a vector or matrix
    with a matching inner dimension; all pairs share one result shape.
    """
    shape = np.shape(pairs[0][0])[:-1] + np.shape(pairs[0][1])[1:]
    terms = np.concatenate([_slice_products(left, right) for left, right in pairs])
    return np.apply_along_axis(math.fsum, 0, terms).reshape(shape)


def exact_weighted_gram(left, right, weights):
    """Correctly rounded ``left.T @ (weights[:, None] * right)``."""
    head, tail = two_product(np.asarray(weights)[:, None], right)
    return exact_matmul((left.T, head), (left.T, tail))


def exact_sum(values, axis=0):
    """Correctly rounded sum of float64 ``values`` along ``axis``."""
    return np.apply_along_axis(math.fsum, axis, np.asarray(values, dtype=np.float64))
