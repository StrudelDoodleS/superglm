"""Numeric feature: single column, group of size 1."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm.types import GroupInfo


class Numeric:
    """A single continuous feature used as-is.

    Group lasso on a size-1 group = standard L1 soft-thresholding.

    Parameters
    ----------
    standardize : bool
        Center and scale before fitting. Back-transformed in reconstruct().
    """

    def __init__(self, standardize: bool = True):
        self.standardize = standardize
        self._mean: float = 0.0
        self._std: float = 1.0

    def build(
        self,
        x: NDArray[np.floating],
        exposure: NDArray[np.floating] | None = None,
    ) -> GroupInfo:
        x = np.asarray(x, dtype=np.float64).ravel()
        if self.standardize:
            self._mean = float(np.mean(x))
            self._std = max(float(np.std(x)), 1e-12)
            col = ((x - self._mean) / self._std).reshape(-1, 1)
        else:
            col = x.reshape(-1, 1)
        return GroupInfo(columns=col, n_cols=1)

    def reconstruct(self, beta: NDArray[np.floating]) -> dict[str, Any]:
        b = float(beta[0])
        if self.standardize:
            b_orig = b / self._std
        else:
            b_orig = b
        return {
            "coef_transformed": b,
            "coef_original": b_orig,
            "relativity_per_unit": float(np.exp(b_orig)),
        }
