"""Stable centered weighted products for grouped design matrices."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _compensated_add(total: NDArray, compensation: NDArray, value: NDArray) -> None:
    corrected = value - compensation
    updated = total + corrected
    compensation[...] = (updated - total) - corrected
    total[...] = updated


def centered_gram_rhs(
    *,
    dm,
    W: NDArray,
    mean_x: NDArray,
    z_centered: NDArray,
    chunk_size: int = 8192,
) -> tuple[NDArray, NDArray]:
    """Return centered ``X'WX`` and ``X'Wz`` without raw-moment subtraction.

    Rows are materialized only in bounded chunks. Centering happens before
    multiplication, so large feature offsets cannot cancel two raw moments.
    Group-specific ``row_subset`` implementations preserve sparse/discretized
    storage and avoid materializing the full training design.
    """
    n, p = dm.shape
    W = np.asarray(W, dtype=float)
    mean_x = np.asarray(mean_x, dtype=float)
    z_centered = np.asarray(z_centered, dtype=float)
    if W.shape != (n,) or z_centered.shape != (n,):
        raise ValueError("W and z_centered must match the design row count")
    if mean_x.shape != (p,):
        raise ValueError("mean_x must match the design column count")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    gram = np.zeros((p, p), dtype=float)
    gram_compensation = np.zeros_like(gram)
    rhs = np.zeros(p, dtype=float)
    rhs_compensation = np.zeros_like(rhs)

    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        rows = np.arange(start, stop)
        block = np.asarray(dm.row_subset(rows).toarray(), dtype=float)
        block -= mean_x
        W_block = W[start:stop]
        gram_block = block.T @ (W_block[:, None] * block)
        rhs_block = block.T @ (W_block * z_centered[start:stop])
        _compensated_add(gram, gram_compensation, gram_block)
        _compensated_add(rhs, rhs_compensation, rhs_block)

    gram = 0.5 * (gram + gram.T)
    return gram, rhs
