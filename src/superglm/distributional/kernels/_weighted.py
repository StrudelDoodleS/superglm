"""Frequency scaling of natural channels with bounded exponent recovery."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.kernels.gamma import _binary_product_divide

_MIN_NORMAL = np.finfo(np.float64).tiny


def weighted_natural_channel(
    unit: NDArray,
    multiplier: NDArray,
    numerators: tuple[NDArray | float, ...],
    denominators: tuple[NDArray | float, ...] = (),
) -> NDArray[np.float64]:
    """Scale a row channel, including its mass before exceptional rounding.

    ``numerators/denominators`` are its original finite analytic factors;
    the caller owns their validation and any additions forming a factor.
    Recovery is selected from the precomputed ``unit`` alone; normal unit
    channels retain ordinary frequency multiplication without inspecting
    their factors. For a subnormal/nonfinite unit candidate, bounded binary
    exponent composition recovers the weighted product. Each mantissa operation
    rounds; a final subnormal rounds once at ldexp, and true overflow stays infinite.
    """
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = multiplier * unit
    unsafe = ~np.isfinite(unit) | (np.abs(unit) < _MIN_NORMAL)
    if not np.any(unsafe):
        return result
    tops = tuple(np.broadcast_to(value, unit.shape) for value in numerators)
    bottoms = tuple(np.broadcast_to(value, unit.shape) for value in denominators)
    zero = multiplier == 0.0
    for value in tops:
        zero |= value == 0.0
    result[zero & ~np.isfinite(unit)] = 0.0
    unsafe &= ~zero
    for index in np.flatnonzero(unsafe):
        numerator = (float(multiplier[index]), *(float(value[index]) for value in tops))
        denominator = tuple(float(value[index]) for value in bottoms)
        try:
            result[index] = _binary_product_divide(numerator, denominator)
        except ValueError:
            factors = (*numerator, *denominator)
            if any(math.isnan(value) for value in factors):
                result[index] = np.nan
            else:
                sign = math.prod(math.copysign(1.0, value) for value in factors)
                result[index] = math.copysign(math.inf, sign)
    return result
