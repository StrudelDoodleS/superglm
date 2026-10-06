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

    Cleanup histories, when present, are accepted post-update snapshots from
    each outer REML iteration.
    """

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
    curvature_source: str | None = None  # retained coefficient-Hessian geometry
    termination_reason: str | None = None  # outer-optimizer terminal condition
    scop_step_norms: list[dict[str, float]] | None = None  # per-group Newton step_norm per step
    scop_fisher_fallbacks: int = 0  # total Fisher-fallback count
    managed_cleanup_names: list[str] | None = None  # SCOP names handled by managed cleanup
    managed_cleanup_frozen_names: list[str] | None = None  # final managed frozen set
    # First 1-based outer iteration where the managed frozen set changed
    # from the previous accepted iteration.
    managed_cleanup_freeze_iter: int | None = None
    # Accepted post-update active managed names per outer step.
    managed_cleanup_active_history: list[list[str]] | None = None
    # Accepted post-update frozen managed names per outer step.
    managed_cleanup_frozen_history: list[list[str]] | None = None
    # SCOP components a suppression hold covered at the published mode: those at
    # a flat end of the criterion, where the residual EDF is under 0.05, or
    # (a penalty other penalties cover) d log|S|+ / d rho_j is under 0.05
    # with the slope asking for a decrease under 0.05 as well
    # (``scop_efs._scop_suppression_holds``). There the criterion cannot say
    # which way the new data's optimum lies, so these are left out of warm
    # starts.
    flat_components: list[str] | None = None
    # The SCOP outer step taken at each iteration ("newton", "efs",
    # "efs_fisher": an EFS step at an iterate where a block's inner solve or
    # the joint geometry fell back to Fisher curvature, which leaves the
    # Newton Jacobian without its reparameterisation terms, or
    # "efs_uncorrected": an EFS step at an observed iterate where those terms
    # could not be formed, not finite or under a map other than exp), and why and at
    # which iteration (0: before the first) the run handed itself to EFS, or
    # None when Newton ran throughout. Reasons: "requested",
    # "multi_scop_cleanup", "scale_profile", "newton_system", and three for a
    # Newton step none of whose forward trials the LAML accepted:
    # "line_search_first_iteration" (EFS took over in place at the first
    # iteration), "line_search" (the search restarted from the bootstrap with
    # EFS steps at that iteration, and the histories hold the Newton
    # iterations followed by the restarted search's), and "line_search_at_cap"
    # (no iteration was left to restart in: the run stopped there on
    # ``max_reml_iter``).
    scop_outer_steps: list[str] | None = None
    scop_newton_fallback: str | None = None
    scop_newton_fallback_iter: int | None = None
    # The final coefficient refit's termination reason when that refit, at the
    # selected smoothing parameters, did not meet its convergence certificate
    # (``converged`` is then False whatever ``termination_reason`` says), or
    # None.
    terminal_refit_termination: str | None = None
    # Fit-invariant Tweedie saturated-density state built by the optimizer.
    # Carried so the terminal REML evaluations in finalize re-enter the SAME
    # per-fit phi cache the search filled, instead of rebuilding a cold one and
    # re-solving an already-solved (Dp, Mp).  Excluded from equality and repr:
    # it is a memo, not part of the result's identity.
    tweedie_scale_data: object | None = field(default=None, repr=False, compare=False)
    # The components the smoothing search started from their warm values
    # (``lambda2_init``), sorted: the SCOP engine empties it when its bootstrap
    # had no certified mode at the warm start and retried at Hessian-scaled
    # values. None where the engine does not record it.
    warm_start_components: list[str] | None = None


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

    A rebuilt block (a new matrix: a new reparametrization) has new columns,
    so its centre no longer applies: its ``c' beta`` at the old coefficients
    ``beta`` moves into ``alpha`` and its centre becomes zero, where the
    mapped coefficients read the same function on the new columns.  A block
    the rebuild reuses (every non-spline block, every dense column) keeps its
    centre and coefficients bit for bit, so no offset is ever cancelled.
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
