"""Compiled exact weighted sums for ``mode_score``'s compensated mean and intercept remainder.

Every sum streams over the rows and holds only Shewchuk's partials, fixed
arrays of 64 floats: no per-row scratch, so the memory does not grow with the
rows.  Each product is split exactly into two floats (Dekker's TwoProduct with
Veltkamp's split; Ogita, Rump & Oishi 2005, Algorithms 3.2 and 3.3), and the
partials are accumulated and rounded exactly as ``math.fsum`` does (Shewchuk
1997; CPython's ``fsum``).

The unscaled kernels add the products as they are, in one pass, so no
contribution underflows before the large terms cancel (``[1e150, -1e150,
1e-200]`` keeps its ``1e-200``).  TwoProduct is exact only while its error
term stays in the normal range, so a product below ``2^-969`` is instead
formed on the operands' ``frexp`` mantissas and carried, exactly, in a second
set of partials scaled up by ``2^1126``; the two sets are merged at the end,
at whichever scale the result needs.  The only losses are a few half-units of
``2^-1074`` in that merge and the tails of products below ``2^-2148``.  If a
split, product or partial overflows, the kernel reports it and the caller
falls back to ``scaled_exact_sum``, which scales every piece by the sum's
largest power of two and streams the rows twice: a floating-point overflow
fallback, not a value gate.  IEEE rounding must hold for every operation:
``fastmath`` would break the error-free transformations.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit  # type: ignore[import-untyped]

_SPLITTER = float(2**27 + 1)
# Non-overlapping partials of binary64 values number at most about
# (2098 + 53) / 53 = 41; a full array asks the caller to fall back.
_PARTIALS = 64
# Below this a TwoProduct's error term can leave the normal range.
_TINY_PRODUCT = 2.0**-969
# Carries every product below _TINY_PRODUCT into the normal range: the
# smallest, two subnormal mantissas at 2^-2148, lands at 2^-1022, the largest
# at 2^157.
_TINY_SHIFT = 1126
_MERGE_FLOOR = 2.0**-968


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
def _raw_product(left, right):
    """TwoProduct of two finite floats; a non-finite part flags overflow."""
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
    return product, error, math.isfinite(product) and math.isfinite(error)


@njit(cache=True, fastmath=False, inline="always")
def _two_sum(left, right):
    """Knuth's TwoSum: ``left + right = total + error`` exactly for finite operands."""
    total = left + right
    recovered = total - left
    return total, (left - (total - recovered)) + (right - recovered)


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


@njit(cache=True, fastmath=False)
def _partials_finite(partials, count):
    for index in range(count):
        if not math.isfinite(partials[index]):
            return False
    return True


@njit(cache=True, fastmath=False, inline="always")
def _add_value(big, big_count, tiny, tiny_count, value):
    """Add one float exactly: to ``big`` as it is, or to ``tiny`` scaled up by ``2^1126``."""
    if value == 0.0:
        return big_count, tiny_count
    if abs(value) >= _TINY_PRODUCT:
        return _add_partial(big, big_count, value), tiny_count
    return big_count, _add_partial(tiny, tiny_count, math.ldexp(value, _TINY_SHIFT))


@njit(cache=True, fastmath=False, inline="always")
def _add_product(big, big_count, tiny, tiny_count, left, right):
    """Add ``left * right`` exactly; ``ok`` False on overflow or a full partials array."""
    if left == 0.0 or right == 0.0:
        return big_count, tiny_count, True
    if abs(left * right) >= _TINY_PRODUCT:
        head, tail, finite = _raw_product(left, right)
        if not finite:
            return big_count, tiny_count, False
        big_count = _add_partial(big, big_count, head)
        if big_count >= 0 and tail != 0.0:
            big_count = _add_partial(big, big_count, tail)
        return big_count, tiny_count, big_count >= 0
    left_mantissa, left_exponent = math.frexp(left)
    right_mantissa, right_exponent = math.frexp(right)
    head, tail = _mantissa_product(left_mantissa, right_mantissa)
    exponent = left_exponent + right_exponent + _TINY_SHIFT
    tiny_count = _add_partial(tiny, tiny_count, math.ldexp(head, exponent))
    if tiny_count >= 0 and tail != 0.0:
        tiny_count = _add_partial(tiny, tiny_count, math.ldexp(tail, exponent))
    return big_count, tiny_count, tiny_count >= 0


@njit(cache=True, fastmath=False)
def _merge(big, big_count, tiny, tiny_count):
    """``(S, K, ok)``: both sets of partials as one sum ``S 2^K``, at the scale it needs.

    A result at least ``2^-968`` takes the tiny partials scaled back down
    (each may round below the normal range, half a unit of ``2^-1074`` at
    most) at ``K = 0``; a smaller one, whose big partials are then all below
    ``2^-967`` (they do not overlap), takes the big partials scaled up exactly,
    at ``K = -1126``.
    """
    if not (_partials_finite(big, big_count) and _partials_finite(tiny, tiny_count)):
        return 0.0, 0, False
    big_value = _round_partials(big, big_count)
    if not math.isfinite(big_value):
        return 0.0, 0, False
    if abs(big_value) >= _MERGE_FLOOR:
        for index in range(tiny_count):
            big_count = _add_partial(big, big_count, math.ldexp(tiny[index], -_TINY_SHIFT))
            if big_count < 0:
                return 0.0, 0, False
        total = _round_partials(big, big_count)
        return total, 0, math.isfinite(total)
    for index in range(big_count):
        tiny_count = _add_partial(tiny, tiny_count, math.ldexp(big[index], _TINY_SHIFT))
        if tiny_count < 0:
            return 0.0, 0, False
    return _round_partials(tiny, tiny_count), -_TINY_SHIFT, True


@njit(cache=True, fastmath=False)
def unscaled_exact_sum(weights, values, shift, mode):
    """``(S, K, ok)``: the rows' exact contributions summed and rounded once, in one pass.

    ``mode`` 0: ``sum w``; 1: ``sum w v``; 2: ``sum w (v - shift)``, with ``v -
    shift = h + e`` split exactly by TwoSum.  ``ok`` is False when a split,
    product, partial or residual overflows, or the partials fill their arrays.
    """
    big = np.empty(_PARTIALS, dtype=np.float64)
    tiny = np.empty(_PARTIALS, dtype=np.float64)
    big_count = 0
    tiny_count = 0
    for row in range(weights.shape[0]):
        weight = weights[row]
        if mode == 0:
            big_count, tiny_count = _add_value(big, big_count, tiny, tiny_count, weight)
            if big_count < 0 or tiny_count < 0:
                return 0.0, 0, False
            continue
        if mode == 1:
            big_count, tiny_count, ok = _add_product(
                big, big_count, tiny, tiny_count, weight, values[row]
            )
            if not ok:
                return 0.0, 0, False
            continue
        residual, residual_error = _two_sum(values[row], -shift)
        if not math.isfinite(residual):
            return 0.0, 0, False
        for part in (residual, residual_error):
            big_count, tiny_count, ok = _add_product(big, big_count, tiny, tiny_count, weight, part)
            if not ok:
                return 0.0, 0, False
    return _merge(big, big_count, tiny, tiny_count)


@njit(cache=True, fastmath=False)
def weighted_residual_sum(weights, response, offset, has_offset, alpha, contribution):
    """``(S, K, ok)``: ``sum w (y - o - alpha - t)`` with every residual split exactly, ``S 2^K``.

    Each row's residual is carried as four floats by TwoSum (``y - o``, then
    ``- alpha``, then ``- t``), never added back into its head before the
    reduction, and each is weighted exactly into the partials.  ``ok`` is
    False on overflow, as ``unscaled_exact_sum``.
    """
    big = np.empty(_PARTIALS, dtype=np.float64)
    tiny = np.empty(_PARTIALS, dtype=np.float64)
    big_count = 0
    tiny_count = 0
    for row in range(weights.shape[0]):
        weight = weights[row]
        if weight == 0.0:
            continue
        value = response[row]
        offset_error = 0.0
        if has_offset:
            value, offset_error = _two_sum(value, -offset[row])
        value, alpha_error = _two_sum(value, -alpha)
        value, contribution_error = _two_sum(value, -contribution[row])
        if not math.isfinite(value):
            return 0.0, 0, False
        for part in (value, contribution_error, alpha_error, offset_error):
            big_count, tiny_count, ok = _add_product(big, big_count, tiny, tiny_count, weight, part)
            if not ok:
                return 0.0, 0, False
    return _merge(big, big_count, tiny, tiny_count)


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
    residual, error = _two_sum(value, -shift)
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
    """``(S, K, ok)``: the overflow fallback, every piece scaled by the sum's largest power of two.

    Pass one finds ``K``, the largest exponent of a nonzero piece; pass two
    adds every piece, scaled by ``2^-K``, to the partials.  A piece pushed
    below the normal range loses at most ``2^-1075`` of ``2^K``, and ``2^K <= 4
    max |piece value|``.  ``ok`` is False when a residual is not finite or the
    partials overflow their array.
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
    """``values`` as the read-only C-contiguous float64 vector the kernels are compiled for.

    Contiguous input is viewed, not copied; one layout keeps one specialisation.
    """
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64).reshape(-1)).view()
    array.flags.writeable = False
    return array


def _warmup_exact_sums() -> None:
    """Compile the kernels for the operands their callers pass."""
    vector = native_operand(np.ones(2))
    for mode in (0, 1, 2):
        unscaled_exact_sum(vector, vector, 0.5, mode)
        scaled_exact_sum(vector, vector, 0.5, mode)
    weighted_residual_sum(vector, vector, vector, True, 0.5, vector)
