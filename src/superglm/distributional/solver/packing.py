"""Canonical symmetric-channel packing for distributional derivatives."""

from __future__ import annotations

import operator


def _positive_parameter_count(k_parameters: int) -> int:
    if isinstance(k_parameters, bool):
        raise TypeError("k_parameters must be an integer")
    try:
        count = operator.index(k_parameters)
    except TypeError as exc:
        raise TypeError("k_parameters must be an integer") from exc
    if count < 1:
        raise ValueError("k_parameters must be at least one")
    return count


def packed_pairs(k_parameters: int) -> tuple[tuple[int, int], ...]:
    """Return canonical upper-triangular parameter pairs."""
    count = _positive_parameter_count(k_parameters)
    return tuple((left, right) for left in range(count) for right in range(left, count))
