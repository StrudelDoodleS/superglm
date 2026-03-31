"""REML smoothing parameter estimation.

Estimates per-term smoothing parameters (lambda_j) from the data. The
direct ``lambda1=0`` path optimizes a Laplace-approximate REML/LAML
criterion over log-lambdas, while the mixed penalized-selection path
retains the Wood (2011) fixed-point update around PIRLS.

Coexists with group lasso: REML controls within-group smoothness
(per-term lambda_j), group lasso controls between-group selection
(lambda1). They are orthogonal.

References
----------
- Wood (2011): Fast stable restricted maximum likelihood and marginal
  likelihood estimation of semiparametric generalized linear models.
  JRSS-B 73(1), 3-36.
- Wood (2017): Generalized Additive Models, 2nd ed., Ch 6.2.
- Wood & Fasiolo (2017): A generalized Fellner-Schall method for smoothing
  parameter optimization. Biometrics 73(4), 1071-1081.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from superglm.group_matrix import DiscretizedSSPGroupMatrix, SparseSSPGroupMatrix
from superglm.types import PenaltyComponent


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


def build_penalty_caches(
    group_matrices: list,
    reml_groups: list[tuple[int, object]],
) -> dict[str, PenaltyCache]:
    """Build PenaltyCache for each REML-eligible group.

    Wood (2011) Section 3.1: pre-compute eigenstructure of each Ω_j in
    SSP coordinates so that log|S|₊ and rank(Ω_j) are O(1) per Newton step.

    Parameters
    ----------
    group_matrices : list of GroupMatrix
    reml_groups : list of (group_index, GroupSlice) tuples

    Returns
    -------
    dict mapping group name to PenaltyCache.
    """
    caches: dict[str, PenaltyCache] = {}
    for idx, g in reml_groups:
        gm = group_matrices[idx]
        omega_ssp = gm.R_inv.T @ gm.omega @ gm.R_inv
        eigvals = np.linalg.eigvalsh(omega_ssp)
        # Adaptive rank threshold: eps^{2/3} balances between the old fixed
        # 1e-8 and the Higham-suggested eps*p.  Too tight (eps*p ~ 1e-15)
        # includes numerical-zero eigenvalues; too loose (1e-8) may miss
        # genuine small eigenvalues in ill-conditioned penalties.
        thresh = np.finfo(float).eps ** (2 / 3) * max(eigvals.max(), 1e-12)
        pos_eigvals = eigvals[eigvals > thresh]
        rank = float(len(pos_eigvals))
        log_det = float(np.sum(np.log(pos_eigvals))) if pos_eigvals.size else 0.0
        caches[g.name] = PenaltyCache(
            omega_ssp=omega_ssp,
            log_det_omega_plus=log_det,
            rank=rank,
            eigvals_omega=pos_eigvals,
        )
    return caches


def cached_logdet_s_plus(
    lambdas: dict[str, float],
    penalty_caches: dict[str, PenaltyCache],
) -> float:
    """Compute log|S|₊ from cached penalty eigenstructure.

    Wood (2011) Section 3.1: stable block-diagonal identity
    log|S|₊ = Σ_j (r_j · log(λ_j) + log|Ω_j|₊), avoiding repeated
    eigendecompositions of the full penalty matrix.
    """
    total = 0.0
    for name, cache in penalty_caches.items():
        lam = lambdas.get(name, 1.0)
        if lam > 0 and cache.rank > 0:
            total += cache.rank * np.log(lam) + cache.log_det_omega_plus
    return total


def build_penalty_components(
    group_matrices: list,
    reml_groups: list[tuple[int, object]],
) -> list[PenaltyComponent]:
    """Build PenaltyComponent list from the current single-penalty group structure.

    This is the bridge between the existing single-penalty-per-group architecture
    and the future multi-penalty path. Currently produces one PenaltyComponent per
    REML-eligible group (same cardinality as reml_groups). Multi-penalty terms
    would produce multiple PenaltyComponents per group.

    The PenaltyComponent list defines the REML lambda dimensions — one smoothing
    parameter per component. This separates term structure (GroupSlice) from
    penalty structure (PenaltyComponent).

    Parameters
    ----------
    group_matrices : list of GroupMatrix
    reml_groups : list of (group_index, GroupSlice) tuples

    Returns
    -------
    list of PenaltyComponent, one per smoothing parameter.
    """
    components: list[PenaltyComponent] = []
    eps_thresh = np.finfo(float).eps ** (2 / 3)
    for idx, g in reml_groups:
        gm = group_matrices[idx]
        omega_ssp = gm.R_inv.T @ gm.omega @ gm.R_inv
        eigvals = np.linalg.eigvalsh(omega_ssp)
        thresh = eps_thresh * max(eigvals.max(), 1e-12)
        pos_eigvals = eigvals[eigvals > thresh]
        rank = float(len(pos_eigvals))
        log_det = float(np.sum(np.log(pos_eigvals))) if pos_eigvals.size else 0.0
        components.append(
            PenaltyComponent(
                name=g.name,
                group_name=g.name,
                group_index=idx,
                omega_raw=gm.omega,
                omega_ssp=omega_ssp,
                rank=rank,
                log_det_omega_plus=log_det,
                eigvals_omega=pos_eigvals,
            )
        )
    return components


@dataclass
class REMLResult:
    """Result of REML smoothing parameter estimation."""

    lambdas: dict[str, float]  # group_name -> estimated lambda_j
    pirls_result: object  # PIRLSResult from final iteration
    n_reml_iter: int
    converged: bool
    lambda_history: list[dict[str, float]] = field(default_factory=list)
    objective: float | None = None


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
