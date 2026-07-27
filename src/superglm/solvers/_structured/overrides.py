"""Shared structural validation for authoritative penalty overrides."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

_STRUCTURAL_ROUNDOFF_FACTOR = 64.0 * np.finfo(np.float64).eps


def _has_structural_zero_mass(
    values: NDArray,
    row_diagonal: NDArray,
    column_diagonal: NDArray,
) -> bool:
    """Return whether an expected-zero block contains more than local roundoff."""
    if values.size == 0:
        return False
    if not (
        np.all(np.isfinite(values))
        and np.all(np.isfinite(row_diagonal))
        and np.all(np.isfinite(column_diagonal))
    ):
        return True
    row_scale = np.sqrt(np.abs(row_diagonal))
    column_scale = np.sqrt(np.abs(column_diagonal))
    participating_scale = row_scale[:, None] * column_scale[None, :]
    tolerance = _STRUCTURAL_ROUNDOFF_FACTOR * np.maximum(
        participating_scale,
        np.finfo(np.float64).tiny,
    )
    return bool(np.any(np.abs(values) > tolerance))


def _has_cross_block_mass(
    penalty: NDArray,
    left_indices: NDArray,
    right_indices: NDArray,
) -> bool:
    """Check both orientations of an expected-zero coefficient cross block."""
    diagonal = np.diag(penalty)
    left_diagonal = diagonal[left_indices]
    right_diagonal = diagonal[right_indices]
    return _has_structural_zero_mass(
        penalty[np.ix_(left_indices, right_indices)],
        left_diagonal,
        right_diagonal,
    ) or _has_structural_zero_mass(
        penalty[np.ix_(right_indices, left_indices)],
        right_diagonal,
        left_diagonal,
    )


def _has_noncanonical_sum_to_zero_mass(
    public_penalty: NDArray,
    expected: NDArray,
) -> bool:
    """Return whether SZ public geometry differs beyond entry-local roundoff."""
    diagonal = np.diag(public_penalty)
    root_scale = np.sqrt(np.abs(diagonal))
    participating_scale = root_scale[:, None] * root_scale[None, :]
    comparison_scale = np.maximum(np.abs(expected), participating_scale)
    tolerance = _STRUCTURAL_ROUNDOFF_FACTOR * np.maximum(
        comparison_scale,
        np.finfo(np.float64).tiny,
    )
    difference = public_penalty - expected
    return not np.all(np.isfinite(difference)) or bool(np.any(np.abs(difference) > tolerance))


def _structured_override_incompatibility(
    penalty: NDArray,
    *,
    small_indices: NDArray,
    structured_indices: NDArray,
    geometry: Literal["random_effect", "factor_smooth", "sum_to_zero"],
) -> str | None:
    """Return why an authoritative override cannot use the compact geometry."""
    flat_structured = structured_indices.ravel()
    if _has_cross_block_mass(penalty, flat_structured, small_indices):
        if geometry == "sum_to_zero":
            return "S_override couples the SZ and dense-small blocks."
        return "S_override couples the dominant and dense-small blocks."

    public = penalty[np.ix_(flat_structured, flat_structured)]
    if geometry == "random_effect":
        residual = np.array(public, copy=True)
        np.fill_diagonal(residual, 0.0)
        diagonal = np.diag(public)
        if _has_structural_zero_mass(residual, diagonal, diagonal):
            return "S_override for the dominant RandomEffect block must be diagonal."
        return None

    if structured_indices.ndim != 2:
        raise ValueError("Factor-smooth structured indices must be two-dimensional.")
    n_blocks, block_size = structured_indices.shape
    if geometry == "factor_smooth":
        residual = np.array(public, copy=True)
        for level in range(n_blocks):
            local = slice(level * block_size, (level + 1) * block_size)
            residual[local, local] = 0.0
        diagonal = np.diag(public)
        if _has_structural_zero_mass(residual, diagonal, diagonal):
            return "S_override couples distinct factor-smooth levels."
        return None

    blocks = public.reshape(n_blocks, block_size, n_blocks, block_size)
    local = 0.5 * blocks[0, :, 0, :] if n_blocks == 1 else blocks[0, :, 1, :]
    expected_blocks = np.empty_like(blocks)
    for left in range(n_blocks):
        for right in range(n_blocks):
            expected_blocks[left, :, right, :] = (2.0 if left == right else 1.0) * local
    expected = expected_blocks.reshape(public.shape)
    if _has_noncanonical_sum_to_zero_mass(public, expected):
        return "S_override has noncanonical sum-to-zero penalty geometry."
    return None


def _factor_smooth_override_local_blocks(
    penalty: NDArray,
    structured_indices: NDArray,
    *,
    sum_to_zero: bool,
) -> NDArray:
    """Extract the all-level local penalties represented by a valid override."""
    if structured_indices.ndim != 2:
        raise ValueError("Factor-smooth structured indices must be two-dimensional.")
    public_levels, block_size = structured_indices.shape
    flat_structured = structured_indices.ravel()
    public = penalty[np.ix_(flat_structured, flat_structured)]
    blocks = public.reshape(
        public_levels,
        block_size,
        public_levels,
        block_size,
    )
    if sum_to_zero:
        local = 0.5 * blocks[0, :, 0, :] if public_levels == 1 else blocks[0, :, 1, :]
        return np.repeat(local[None, :, :], public_levels + 1, axis=0)
    return np.stack(
        [blocks[level, :, level, :] for level in range(public_levels)],
        axis=0,
    )
