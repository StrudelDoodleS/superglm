"""Sparse group lasso penalty.

Combines group L2 (group selection) with elementwise L1 (within-group
sparsity). At alpha=0 this is pure group lasso; at alpha=1 pure lasso.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from superglm.penalties.base import Flavor, normalize_penalty_features, penalty_targets_group
from superglm.types import GroupSlice


class SparseGroupLasso:
    """Sparse group lasso: lambda1 * [(1-alpha) * group_L2 + alpha * L1].

    Parameters
    ----------
    lambda1 : float or None
        Regularisation strength. If None, auto-calibrated at fit time.
    alpha : float
        Mixing parameter in [0, 1]. 0 = pure group lasso, 1 = pure L1.
    flavor : Flavor or None
        Optional modifier (e.g. Adaptive).
    """

    def __init__(
        self,
        lambda1: float | None = None,
        alpha: float = 0.5,
        flavor: Flavor | None = None,
        features: str | list[str] | None = None,
    ):
        self.lambda1 = lambda1
        self.alpha = alpha
        self.flavor = flavor
        self.features = normalize_penalty_features(features)

    def prox_group(self, bg: NDArray, group: GroupSlice, step: float) -> NDArray:
        """L1 soft-threshold then group L2 prox for a single group."""
        if not penalty_targets_group(self, group):
            return bg
        lam = step * self.lambda1

        # Elementwise L1 soft-threshold
        thr_l1 = lam * self.alpha
        bg = np.sign(bg) * np.maximum(np.abs(bg) - thr_l1, 0.0)

        # Group L2 block soft-threshold
        norm_g = np.linalg.norm(bg)
        thr_grp = lam * (1.0 - self.alpha) * group.weight
        if norm_g <= thr_grp:
            return np.zeros_like(bg)
        return bg * (1.0 - thr_grp / norm_g)

    def prox(self, beta: NDArray, groups: list[GroupSlice], step: float) -> NDArray:
        """Proximal operator via decomposition: L1 soft-threshold then group L2 prox."""
        beta = beta.copy()
        for g in groups:
            beta[g.sl] = self.prox_group(beta[g.sl], g, step)
        return beta

    def eval(self, beta: NDArray, groups: list[GroupSlice]) -> float:
        """Penalty value."""
        grp_val = 0.0
        l1_val = 0.0
        for g in groups:
            if penalty_targets_group(self, g):
                grp_val += g.weight * np.linalg.norm(beta[g.sl])
                l1_val += np.sum(np.abs(beta[g.sl]))
        return self.lambda1 * ((1.0 - self.alpha) * grp_val + self.alpha * l1_val)
