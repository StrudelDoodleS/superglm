"""Penalty flavors (modifiers).

Flavors adjust group weights based on an initial estimate. They don't
change the proximal operator — they just change the weights it uses.
"""

from __future__ import annotations

import copy

import numpy as np
from numpy.typing import NDArray

from superglm.types import GroupSlice


class Adaptive:
    """Adaptive weighting flavor.

    Computes per-group weights inversely proportional to the initial
    estimate's group norms. Groups with large initial coefficients get
    smaller penalties (kept more easily); groups with small coefficients
    get larger penalties (zeroed more aggressively).

    This gives the adaptive group lasso better oracle properties than
    the plain group lasso (Zou, 2006; Wang & Leng, 2008).

    Parameters
    ----------
    expon : float
        Exponent for the inverse weighting. Higher values increase
        the contrast between large and small groups.
    eps : float
        Small constant to avoid division by zero for initially-zeroed groups.
    """

    def __init__(self, expon: float = 1.0, eps: float = 1e-6):
        self.expon = expon
        self.eps = eps

    def adjust_weights(
        self,
        groups: list[GroupSlice],
        beta_init: NDArray,
        group_matrices: list | None = None,
    ) -> list[GroupSlice]:
        """Return new GroupSlice list with adaptive weights.

        When ``group_matrices`` is provided, uses fitted-value norms
        ``||X_g beta_g|| / sqrt(n)`` (RMS contribution to eta) instead
        of raw coefficient norms. This is scale-invariant across groups
        with different reparametrizations (e.g. SSP splines).

        Without ``group_matrices``, falls back to coefficient norms:
        ``new_weight_g = sqrt(p_g) / (||beta_init_g|| + eps)^expon``
        """
        new_groups = []
        for i, g in enumerate(groups):
            if group_matrices is not None:
                fitted = group_matrices[i].matvec(beta_init[g.sl])
                norm_g = np.linalg.norm(fitted) / np.sqrt(len(fitted))
            else:
                norm_g = np.linalg.norm(beta_init[g.sl])
            adaptive_w = np.sqrt(g.size) / (norm_g + self.eps) ** self.expon
            new_g = copy.copy(g)
            new_g.weight = adaptive_w
            new_groups.append(new_g)
        return new_groups
