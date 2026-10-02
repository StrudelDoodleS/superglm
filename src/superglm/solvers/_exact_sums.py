"""Compiled exact weighted sums for ``mode_score.compensated_weighted_mean``.

Each sum streams over the rows twice and holds only Shewchuk's partials, a
fixed array of 64 floats: no per-row scratch, so the memory does not grow with
the rows.  Every product is formed exactly on the operands' ``frexp``
mantissas with its binary exponent kept apart (Dekker's TwoProduct with
Veltkamp's split; Ogita, Rump & Oishi 2005, Algorithms 3.2 and 3.3), every
piece is scaled by the sum's largest power of two, and the partials are
accumulated and rounded exactly as ``math.fsum`` does (Shewchuk 1997;
CPython's ``fsum``), so the result is the correctly rounded sum of the scaled
pieces.  IEEE rounding must hold for every operation: ``fastmath`` would
break the error-free transformations.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit  # type: ignore[import-untyped]

_SPLITTER = float(2**27 + 1)
# Non-overlapping partials of binary64 values number at most about
# (2098 + 53) / 53 = 41; a full array asks the caller to fall back.
_PARTIALS = 64


@njit(cache=True, fastmath=False, inline="always")
def _mantissa_product(left, right):
    """TwoProduct of two ``frexp`` mantissas, ``left * right = product + error`` exactly.

    Mantissas lie in ``[0.5, 1)`` in magnitude or are zero, so neither the
    split nor any partial product over- or underflows.
    """
    product = left * right
    scaled = _SPLITTER * left
    left_high = scaled - (scaled - left)
    left_low = left - left_high
    scaled = _SPLITTER * right
    right_high = scaled - (scaled - right)
    right_low = right - right_high
    error = (
        (left_high * right_high - product) + left_high * right_low + left_low * right_high
    ) + left_low * right_low
    return product, error


@njit(cache=True, fastmath=False, inline="always")
def _add_partial(partials, count, value):
    """Shewchuk's grow-expansion step, as ``math.fsum`` takes it: the partials stay exact.

    Returns the new count, or -1 when the fixed array is full.
    """
    kept = 0
    for index in range(count):
        other = partials[index]
        if abs(value) < abs(other):
            value, other = other, value
        high = value + other
        low = other - (high - value)
        if low != 0.0:
            partials[kept] = low
            kept += 1
        value = high
    if value != 0.0:
        if kept >= partials.shape[0]:
            return -1
        partials[kept] = value
        kept += 1
    return kept


@njit(cache=True, fastmath=False)
def _round_partials(partials, count):
    """The correctly rounded sum of non-overlapping partials, as ``math.fsum`` rounds it."""
    if count == 0:
        return 0.0
    index = count - 1
    high = partials[index]
    low = 0.0
    while index > 0:
        value = high
        index -= 1
        other = partials[index]
        high = value + other
        low = other - (high - value)
        if low != 0.0:
            break
    if index > 0 and (
        (low < 0.0 and partials[index - 1] < 0.0) or (low > 0.0 and partials[index - 1] > 0.0)
    ):
        doubled = low * 2.0
        candidate = high + doubled
        if doubled == candidate - high:
            high = candidate
    return high


@njit(cache=True, fastmath=False, inline="always")
def _row_pieces(weight, value, shift, mode):
    """A row's exact contribution as up to four mantissa products and their exponents.

    ``mode`` 0: ``w``; 1: ``w v``; 2: ``w (v - shift)``, with ``v - shift = h +
    e`` split exactly by TwoSum.  Returns the four (head, tail) pieces, their
    two exponents and whether the residual was finite.
    """
    weight_mantissa, weight_exponent = math.frexp(weight)
    if mode == 0:
        return weight_mantissa, 0.0, 0.0, 0.0, weight_exponent, 0, True
    if mode == 1:
        value_mantissa, value_exponent = math.frexp(value)
        head, tail = _mantissa_product(weight_mantissa, value_mantissa)
        return head, tail, 0.0, 0.0, weight_exponent + value_exponent, 0, True
    residual = value - shift
    recovered = residual - value
    error = (value - (residual - recovered)) + (-shift - recovered)
    if not math.isfinite(residual):
        return 0.0, 0.0, 0.0, 0.0, 0, 0, False
    residual_mantissa, residual_exponent = math.frexp(residual)
    error_mantissa, error_exponent = math.frexp(error)
    head, tail = _mantissa_product(weight_mantissa, residual_mantissa)
    error_head, error_tail = _mantissa_product(weight_mantissa, error_mantissa)
    return (
        head,
        tail,
        error_head,
        error_tail,
        weight_exponent + residual_exponent,
        weight_exponent + error_exponent,
        True,
    )


@njit(cache=True, fastmath=False)
def scaled_exact_sum(weights, values, shift, mode):
    """``(S, K, ok)``: the rows' exact contributions summed, ``S 2^K``, up to the scaling loss.

    Pass one finds ``K``, the largest exponent of a nonzero piece; pass two
    adds every piece, scaled by ``2^-K``, to the partials.  A piece pushed
    below the normal range loses at most ``2^-1075`` of ``2^K``, and ``2^K <= 4
    max |piece value|``.  ``ok`` is False when a residual is not finite or the
    partials overflow their array; the caller then falls back.
    """
    top = -(2**31)
    for row in range(weights.shape[0]):
        head, tail, error_head, error_tail, exponent, error_exponent, finite = _row_pieces(
            weights[row], values[row], shift, mode
        )
        if not finite:
            return 0.0, 0, False
        if head != 0.0 and exponent > top:
            top = exponent
        if error_head != 0.0 and error_exponent > top:
            top = error_exponent
    if top == -(2**31):
        return 0.0, 0, True
    partials = np.empty(_PARTIALS, dtype=np.float64)
    count = 0
    for row in range(weights.shape[0]):
        head, tail, error_head, error_tail, exponent, error_exponent, finite = _row_pieces(
            weights[row], values[row], shift, mode
        )
        for piece, piece_exponent in (
            (head, exponent),
            (tail, exponent),
            (error_head, error_exponent),
            (error_tail, error_exponent),
        ):
            if piece != 0.0:
                count = _add_partial(partials, count, math.ldexp(piece, piece_exponent - top))
                if count < 0:
                    return 0.0, 0, False
    return _round_partials(partials, count), top, True


def native_operand(values) -> np.ndarray:
    """``values`` as the read-only C-contiguous float64 vector the kernel is compiled for.

    Contiguous input is viewed, not copied; one layout keeps one specialisation.
    """
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64).reshape(-1)).view()
    array.flags.writeable = False
    return array


def _warmup_exact_sums() -> None:
    """Compile ``scaled_exact_sum`` for the operands its caller passes."""
    vector = native_operand(np.ones(2))
    for mode in (0, 1, 2):
        scaled_exact_sum(vector, vector, 0.5, mode)
