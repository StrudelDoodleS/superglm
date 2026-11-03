"""REML result types and basis-mapping utilities."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from superglm.group_matrix import (
    DiscretizedSplineCategoricalGroupMatrix,
    DiscretizedSSPGroupMatrix,
    SparseSSPGroupMatrix,
    SplineCategoricalGroupMatrix,
)

_SSP_LIKE = (
    SparseSSPGroupMatrix
    | SplineCategoricalGroupMatrix
    | DiscretizedSplineCategoricalGroupMatrix
    | DiscretizedSSPGroupMatrix
)


@dataclass
class PenaltyCache:
    """Pre-computed per-group penalty eigenstructure for REML optimization.

    Computed once at ``fit_reml()`` entry and reused across all Newton /
    fixed-point iterations, avoiding redundant eigendecompositions of Ω.
    """

    omega_ssp: NDArray | None  # (p_g, p_g) = R_inv.T @ omega @ R_inv
    log_det_omega_plus: float  # log|Ω|₊ (constant across lambda iterations)
    rank: float  # rank(Ω) = r_j
    eigvals_omega: NDArray | None  # positive eigenvalues of Ω_ssp


@dataclass
class REMLResult:
    """Result of REML smoothing parameter estimation.

    A result a model fitted on the retired SCOP engine (0.41.0 or earlier)
    also carries that engine's fields, such as ``scop_newton_fallback``,
    which the diagnostics read with ``getattr``.
    """

    lambdas: dict[str, float]  # group_name -> estimated lambda_j
    pirls_result: object  # PIRLSResult from final iteration
    n_reml_iter: int
    converged: bool
    lambda_history: list[dict[str, float]] = field(default_factory=list)
    objective: float | None = None
    reml_penalties: list | None = None  # merged PenaltyComponents
    objective_history: list[float] | None = None  # REML objective per outer step
    curvature_source: str | None = None  # retained coefficient-Hessian geometry
    termination_reason: str | None = None  # outer-optimizer terminal condition
    # The final refit's termination reason when it missed its convergence
    # certificate (``converged`` is then False), else None.
    terminal_refit_termination: str | None = None
    # Smoothing parameters whose penalized maximum sat on the mean-space
    # boundary (a binomial/log probability of 1) and that ``reml.direct``
    # restored to an interior start.
    mean_space_restorations: int = 0
    # The search's fit-invariant Tweedie saturated-density cache, reused by the
    # terminal REML evaluations. A memo, so excluded from equality and repr.
    tweedie_scale_data: object | None = field(default=None, repr=False, compare=False)


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
        if isinstance(gm_old, _SSP_LIKE) and isinstance(gm_new, _SSP_LIKE):
            if gm_old is gm_new or gm_old.R_inv is gm_new.R_inv:
                continue
            # Map through B-spline space: old_R_inv @ beta_old = new_R_inv @ beta_new
            beta_bspline = gm_old.R_inv @ beta_new[g.sl]
            beta_new[g.sl] = np.linalg.lstsq(gm_new.R_inv, beta_bspline, rcond=None)[0]
    return beta_new


def _map_centred_state_between_bases(
    centred: tuple[float, NDArray] | None,
    beta: NDArray,
    old_gms: Sequence,
    new_gms: Sequence,
    groups: Sequence,
) -> tuple[float, NDArray] | None:
    """A warm centred state ``(alpha, c)`` read with ``_map_beta_between_bases``' coefficients.

    A rebuilt block has new columns, so its ``c' beta`` at the old ``beta``
    moves into ``alpha`` and its centre becomes zero. A reused block keeps
    its centre and coefficients bit for bit.
    """
    if centred is None:
        return None
    alpha, centre = centred
    moved = np.zeros(len(centre), dtype=bool)
    for gm_old, gm_new, g in zip(old_gms, new_gms, groups):
        if gm_old is not gm_new:
            moved[g.sl] = True
    if not np.any(moved & (centre != 0.0)):
        return alpha, centre
    shift = math.fsum(centre[moved] * np.asarray(beta, dtype=np.float64)[moved])
    return float(alpha) - shift, np.where(moved, 0.0, centre)
