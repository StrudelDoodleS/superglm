"""Shared coefficient geometry for sum-to-zero factor smooths."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def sum_to_zero_contrast(n_levels: int) -> NDArray[np.float64]:
    """Return ``[I; -1]`` for ``n_levels`` symmetric factor deviations."""
    if isinstance(n_levels, bool) or not isinstance(n_levels, int):
        raise TypeError("n_levels must be an integer")
    if n_levels < 2:
        raise ValueError("sum-to-zero factor geometry requires at least two levels")
    return np.vstack(
        (
            np.eye(n_levels - 1, dtype=np.float64),
            -np.ones((1, n_levels - 1), dtype=np.float64),
        )
    )


def expand_sum_to_zero_blocks(values: NDArray) -> NDArray[np.float64]:
    """Expand ``K-1`` free coefficient blocks to ``K`` blocks summing to zero."""
    free = np.asarray(values, dtype=np.float64)
    if free.ndim not in (2, 3) or free.shape[0] < 1:
        raise ValueError("free blocks must have shape (K-1, k[, m])")
    return np.concatenate(
        (free, -np.sum(free, axis=0, keepdims=True)),
        axis=0,
    )


def adjoint_sum_to_zero_blocks(values: NDArray) -> NDArray[np.float64]:
    """Apply the transpose of ``[I; -1]`` to raw level blocks."""
    raw = np.asarray(values, dtype=np.float64)
    if raw.ndim not in (2, 3) or raw.shape[0] < 2:
        raise ValueError("raw blocks must have shape (K, k[, m])")
    return raw[:-1] - raw[-1:]


def sum_to_zero_penalty(local: NDArray, n_levels: int) -> NDArray[np.float64]:
    """Materialize ``(C.T @ C) kron local`` for dense reference paths."""
    marginal = np.asarray(local, dtype=np.float64)
    if marginal.ndim != 2 or marginal.shape[0] != marginal.shape[1]:
        raise ValueError("local penalty must be a square matrix")
    contrast = sum_to_zero_contrast(n_levels)
    return np.kron(contrast.T @ contrast, marginal)


__all__ = [
    "adjoint_sum_to_zero_blocks",
    "expand_sum_to_zero_blocks",
    "sum_to_zero_contrast",
    "sum_to_zero_penalty",
]
