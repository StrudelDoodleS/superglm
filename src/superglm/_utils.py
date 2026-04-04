"""Shared private array helpers used across superglm submodules."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _ensure_array(a, dtype=np.float64) -> NDArray:
    """Convert input to a 1-D float64 array."""
    return np.asarray(a, dtype=dtype)


def _ensure_1d_float(x) -> NDArray:
    """Convert to 1-D float64, raveling if needed."""
    return np.asarray(x, dtype=np.float64).ravel()


def _default_weights(w, n: int) -> NDArray:
    """Return uniform weights if *w* is None, else ensure array."""
    if w is None:
        return np.ones(n, dtype=np.float64)
    return np.asarray(w, dtype=np.float64)
