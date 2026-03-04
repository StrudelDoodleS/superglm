"""Ridge (L2) penalty.

Pure shrinkage, no variable selection. Useful as a baseline or when
you want all features retained.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from superglm.types import GroupSlice


class Ridge:
    """Ridge penalty: lambda1 * ||beta||_2^2 / 2.

    Parameters
    ----------
    lambda1 : float or None
        Regularisation strength. If None, auto-calibrated at fit time.
    """

    def __init__(self, lambda1: float | None = None):
        self.lambda1 = lambda1
        self.flavor = None  # Ridge doesn't support flavors

    def prox_group(self, bg: NDArray, group: GroupSlice, step: float) -> NDArray:
        """Closed-form proximal operator for a single group."""
        return bg / (1.0 + step * self.lambda1)

    def prox(self, beta: NDArray, groups: list[GroupSlice], step: float) -> NDArray:
        """Closed-form proximal operator: beta / (1 + step * lambda1)."""
        return beta / (1.0 + step * self.lambda1)

    def eval(self, beta: NDArray, groups: list[GroupSlice]) -> float:
        """Penalty value: lambda1 * ||beta||_2^2 / 2."""
        return self.lambda1 * np.dot(beta, beta) / 2.0
