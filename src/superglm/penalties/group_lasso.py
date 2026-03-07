"""Group lasso penalty.

For group size 1, this reduces to standard L1 soft-thresholding (lasso).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from superglm.penalties.base import Flavor
from superglm.types import GroupSlice


class GroupLasso:
    """Group lasso penalty: lambda1 * sum_g(w_g * ||beta_g||_2).

    Parameters
    ----------
    lambda1 : float or None
        Regularisation strength. If None, auto-calibrated at fit time
        to 10% of lambda_max.
    flavor : Flavor or None
        Optional modifier (e.g. Adaptive) that adjusts group weights
        based on an initial estimate.
    """

    def __init__(self, lambda1: float | None = None, flavor: Flavor | None = None):
        self.lambda1 = lambda1
        self.flavor = flavor

    def prox_group(self, bg: NDArray, group: GroupSlice, step: float) -> NDArray:
        """Block soft-thresholding for a single group."""
        if not group.penalized:
            return bg
        norm_g = np.linalg.norm(bg)
        thr = step * self.lambda1 * group.weight
        if norm_g <= thr:
            return np.zeros_like(bg)
        return bg * (1.0 - thr / norm_g)

    def prox(self, beta: NDArray, groups: list[GroupSlice], step: float) -> NDArray:
        """Block soft-thresholding proximal operator."""
        beta = beta.copy()
        for g in groups:
            beta[g.sl] = self.prox_group(beta[g.sl], g, step)
        return beta

    def eval(self, beta: NDArray, groups: list[GroupSlice]) -> float:
        """Penalty value: lambda1 * sum_g(w_g * ||beta_g||_2)."""
        val = 0.0
        for g in groups:
            if g.penalized:
                val += g.weight * np.linalg.norm(beta[g.sl])
        return self.lambda1 * val
