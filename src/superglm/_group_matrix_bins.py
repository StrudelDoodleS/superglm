"""Private discretization helpers for group-matrix construction."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def discretize_column(x: NDArray, n_bins: int = 256) -> tuple[NDArray, NDArray]:
    """Compress a continuous variable to exact support or equal-width bins."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    unique_vals, inverse = np.unique(x, return_inverse=True)
    if len(unique_vals) <= n_bins:
        return unique_vals.astype(np.float64), inverse.astype(np.intp)

    lo, hi = float(x.min()), float(x.max())
    if lo == hi:
        return np.array([lo]), np.zeros(len(x), dtype=np.intp)
    edges = np.linspace(lo, hi, n_bins + 1)
    bin_idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, n_bins - 1)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    return bin_centers, bin_idx


__all__ = ["discretize_column"]
