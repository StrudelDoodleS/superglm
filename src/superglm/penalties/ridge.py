"""Ridge (L2) penalty.

Pure shrinkage, no variable selection. Useful as a baseline or when
you want all features retained.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from superglm.penalties.base import normalize_penalty_features, penalty_targets_group
from superglm.types import GroupSlice


class Ridge:
    """Ridge penalty: lambda1 * ||beta||_2^2 / 2.

    Parameters
    ----------
    lambda1 : float or None
        Regularisation strength. If None, auto-calibrated at fit time.
    """

    def __init__(self, lambda1: float | None = None, features: str | list[str] | None = None):
        self.lambda1 = lambda1
        self.flavor = None  # Ridge doesn't support flavors
        self.features = normalize_penalty_features(features)

    def prox_group(self, bg: NDArray, group: GroupSlice, step: float) -> NDArray:
        """Closed-form proximal operator for a single group."""
        if not penalty_targets_group(self, group):
            return bg
        return bg / (1.0 + step * self.lambda1)

    def prox(self, beta: NDArray, groups: list[GroupSlice], step: float) -> NDArray:
        """Closed-form proximal operator: beta / (1 + step * lambda1)."""
        beta = beta.copy()
        for g in groups:
            if penalty_targets_group(self, g):
                beta[g.sl] = beta[g.sl] / (1.0 + step * self.lambda1)
        return beta

    def eval(self, beta: NDArray, groups: list[GroupSlice]) -> float:
        """Penalty value: lambda1 * ||beta||_2^2 / 2."""
        val = 0.0
        for g in groups:
            if penalty_targets_group(self, g):
                val += np.dot(beta[g.sl], beta[g.sl])
        return self.lambda1 * val / 2.0
