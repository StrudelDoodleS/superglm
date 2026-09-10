"""Native value loop for the separately enclosed float64 Dot2 action."""

from __future__ import annotations

import math

import numpy as np
from numba import njit  # type: ignore[import-untyped]

_SPLITTER = float(2**27 + 1)
_TINY = np.finfo(np.float64).tiny


@njit(cache=True, fastmath=False, inline="always")
def _two_product_split(left, right):
    """Return a normal-range TwoProduct or request the caller's fallback."""
    if not math.isfinite(left) or not math.isfinite(right):
        return 0.0, 0.0, False
    if (left != 0.0 and abs(left) < _TINY) or (right != 0.0 and abs(right) < _TINY):
        return 0.0, 0.0, False
    a, b = _SPLITTER * left, _SPLITTER * right
    if not math.isfinite(a) or not math.isfinite(b):
        return 0.0, 0.0, False
    left_high, right_high = a - (a - left), b - (b - right)
    left_low, right_low = left - left_high, right - right_high
    product = left * right
    high_high = left_high * right_high
    low_high = left_low * right_high
    high_low = left_high * right_low
    low_low = left_low * right_low
    products = (product, high_high, low_high, high_low, low_low)
    left_parts = (left, left_high, left_low, left_high, left_low)
    right_parts = (right, right_high, right_high, right_low, right_low)
    for index in range(5):
        part = products[index]
        if not math.isfinite(part):
            return 0.0, 0.0, False
        if part == 0.0:
            if left_parts[index] != 0.0 and right_parts[index] != 0.0:
                return 0.0, 0.0, False
        elif abs(part) < _TINY:
            return 0.0, 0.0, False
    error = low_low - (((product - high_high) - low_high) - high_low)
    return product, error, math.isfinite(error)


@njit(cache=True, fastmath=False)
def _dot2_value(x, y):
    """Evaluate float64 vectors in the caller's existing Dot2 order.

    Ogita, Rump and Oishi (2005), Algorithms 3.1/3.3 and 5.3. Each operation
    must retain IEEE rounding: reassociation and multiply-add contraction
    invalidate the error-free transformations. The caller owns the operand
    split and Proposition 5.5 enclosure. False requests its existing FMA or
    Split fallback, without changing the supported numerical domain.

    Inputs are already one-dimensional float64 arrays. No array scratch is
    allocated. Zero is valid; nonzero subnormal operands or products and
    nonfinite intermediates are deferred to the fallback.
    """
    if len(x) != len(y):
        return 0.0, False
    if not len(x):
        return 0.0, True
    product, correction, success = _two_product_split(x[0], y[0])
    if not success:
        return 0.0, False
    for index in range(1, len(x)):
        term, term_error, success = _two_product_split(x[index], y[index])
        if not success:
            return 0.0, False
        updated = product + term
        recovered = updated - product
        addition_error = (product - (updated - recovered)) + (term - recovered)
        correction += addition_error + term_error
        product = updated
        if not math.isfinite(product) or not math.isfinite(correction):
            return 0.0, False
    value = product + correction
    return value, math.isfinite(value)
