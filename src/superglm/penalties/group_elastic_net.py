"""Group elastic net penalty.

Combines group lasso (feature selection) with ridge (shrinkage stabilisation
for correlated groups). Zou & Hastie (2005) showed that elastic net fixes the
lasso's instability with correlated predictors; the same benefit applies at
the group level.

Proximal operator decomposes into sequential ridge then group soft-threshold
(Parikh & Boyd, 2014, §2.2).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from superglm.penalties.base import Flavor
from superglm.types import GroupSlice


class GroupElasticNet:
    """Group elastic net: lambda1 * [alpha * group_L2 + (1-alpha)/2 * L2^2].

    alpha=1.0 → pure group lasso, alpha=0.0 → pure ridge.

    Parameters
    ----------
    lambda1 : float or None
        Regularisation strength. If None, auto-calibrated at fit time
        to 10% of lambda_max.
    alpha : float
        Mixing parameter in [0, 1]. 1 = pure group lasso, 0 = pure ridge.
    flavor : Flavor or None
        Optional modifier (e.g. Adaptive) that adjusts group weights
        based on an initial estimate.
    """

    def __init__(
        self,
        lambda1: float | None = None,
        alpha: float = 0.5,
        flavor: Flavor | None = None,
    ):
        self.lambda1 = lambda1
        self.alpha = alpha
        self.flavor = flavor

    def prox_group(self, bg: NDArray, group: GroupSlice, step: float) -> NDArray:
        """Two-step proximal operator for a single group."""
        if not group.penalized:
            return bg
        # Step 1: ridge shrinkage
        bg = bg / (1.0 + step * self.lambda1 * (1.0 - self.alpha))
        # Step 2: group soft-threshold
        norm_g = np.linalg.norm(bg)
        thr = step * self.lambda1 * self.alpha * group.weight
        if norm_g <= thr:
            return np.zeros_like(bg)
        return bg * (1.0 - thr / norm_g)

    def prox(self, beta: NDArray, groups: list[GroupSlice], step: float) -> NDArray:
        """Proximal operator: ridge shrinkage then group soft-threshold."""
        beta = beta.copy()
        for g in groups:
            beta[g.sl] = self.prox_group(beta[g.sl], g, step)
        return beta

    def eval(self, beta: NDArray, groups: list[GroupSlice]) -> float:
        """Penalty value: lambda1 * [alpha * group_L2 + (1-alpha)/2 * L2^2]."""
        grp_val = 0.0
        ridge_val = 0.0
        for g in groups:
            if g.penalized:
                grp_val += g.weight * np.linalg.norm(beta[g.sl])
                ridge_val += np.dot(beta[g.sl], beta[g.sl])
        return self.lambda1 * (self.alpha * grp_val + (1.0 - self.alpha) * ridge_val / 2.0)
