"""REML smoothing parameter estimation.

Estimates per-term smoothing parameters (lambda_j) from the data using
a fixed-point iteration on the Restricted Maximum Likelihood (REML)
criterion. Wraps the existing PIRLS solver: each REML iteration fixes
lambda_j, warm-starts PIRLS, then updates lambda_j via the Wood (2011)
fixed-point formula.

Coexists with group lasso: REML controls within-group smoothness
(per-term lambda_j), group lasso controls between-group selection
(lambda1). They are orthogonal.

References
----------
- Wood (2011): Fast stable restricted maximum likelihood and marginal
  likelihood estimation of semiparametric generalized linear models.
  JRSS-B 73(1), 3-36.
- Wood (2017): Generalized Additive Models, 2nd ed., Ch 6.2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from superglm.group_matrix import DiscretizedSSPGroupMatrix, SparseSSPGroupMatrix


@dataclass
class REMLResult:
    """Result of REML smoothing parameter estimation."""

    lambdas: dict[str, float]  # group_name -> estimated lambda_j
    pirls_result: object  # PIRLSResult from final iteration
    n_reml_iter: int
    converged: bool
    lambda_history: list[dict[str, float]] = field(default_factory=list)


def _map_beta_between_bases(
    beta: NDArray,
    old_gms: list,
    new_gms: list,
    groups: list,
) -> NDArray:
    """Map coefficient vector from old SSP basis to new when R_inv changes.

    For SSP groups, coefficients are in the reparametrised space:
    beta_bspline = R_inv_old @ beta_old. When R_inv changes (due to a new
    lambda), we solve for the new beta: beta_new = R_inv_new^{-1} @ beta_bspline.

    Non-SSP groups are copied unchanged.
    """
    beta_new = beta.copy()
    for gm_old, gm_new, g in zip(old_gms, new_gms, groups):
        if isinstance(gm_old, SparseSSPGroupMatrix | DiscretizedSSPGroupMatrix) and isinstance(
            gm_new, SparseSSPGroupMatrix | DiscretizedSSPGroupMatrix
        ):
            # Map through B-spline space: old_R_inv @ beta_old = new_R_inv @ beta_new
            beta_bspline = gm_old.R_inv @ beta_new[g.sl]
            beta_new[g.sl] = np.linalg.lstsq(gm_new.R_inv, beta_bspline, rcond=None)[0]
    return beta_new
