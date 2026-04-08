"""REML result types and basis-mapping utilities."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from superglm.group_matrix import DiscretizedSSPGroupMatrix, SparseSSPGroupMatrix


@dataclass
class PenaltyCache:
    """Pre-computed per-group penalty eigenstructure for REML optimization.

    Computed once at ``fit_reml()`` entry and reused across all Newton /
    fixed-point iterations, avoiding redundant eigendecompositions of Ω.
    """

    omega_ssp: NDArray  # (p_g, p_g) = R_inv.T @ omega @ R_inv
    log_det_omega_plus: float  # log|Ω|₊ (constant across lambda iterations)
    rank: float  # rank(Ω) = r_j
    eigvals_omega: NDArray  # positive eigenvalues of Ω_ssp


@dataclass
class REMLResult:
    """Result of REML smoothing parameter estimation."""

    lambdas: dict[str, float]  # group_name -> estimated lambda_j
    pirls_result: object  # PIRLSResult from final iteration
    n_reml_iter: int
    converged: bool
    lambda_history: list[dict[str, float]] = field(default_factory=list)
    objective: float | None = None
    reml_penalties: list | None = None  # merged SSP + SCOP PenaltyComponents
    scop_states: dict | None = None  # converged SCOP state for objective reproduction
    inner_iter_history: list[int] | None = None  # PIRLS iters per outer EFS step
    objective_history: list[float] | None = None  # REML objective per outer step
    scop_step_norms: list[dict[str, float]] | None = None  # per-group Newton step_norm per step
    scop_fisher_fallbacks: int = 0  # total Fisher-fallback count
    scop_halving_count: int = 0  # total step-halving count


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
