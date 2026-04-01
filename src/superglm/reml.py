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


def build_penalty_components(
    group_matrices: list,
    reml_groups: list[tuple[int, object]],
) -> list[PenaltyComponent]:
    """Build PenaltyComponent list — the single source of penalty eigenstructure.

    Wood (2011) Section 3.1: pre-compute eigenstructure of each Ω_j in
    SSP coordinates so that log|S|₊ and rank(Ω_j) are O(1) per Newton step.

    Currently produces one PenaltyComponent per REML-eligible group (single
    penalty per term). Multi-penalty terms would produce multiple components
    per group, each with its own lambda optimized by REML.

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

        if getattr(gm, "omega_components", None) is not None:
            # Multi-penalty path: N components share this coefficient block.
            for suffix, omega_j in gm.omega_components:
                omega_ssp_j = gm.R_inv.T @ omega_j @ gm.R_inv
                eigvals = np.linalg.eigvalsh(omega_ssp_j)
                thresh = eps_thresh * max(eigvals.max(), 1e-12)
                pos_eigvals = eigvals[eigvals > thresh]
                rank = float(len(pos_eigvals))
                log_det = float(np.sum(np.log(pos_eigvals))) if pos_eigvals.size else 0.0
                components.append(
                    PenaltyComponent(
                        name=f"{g.name}:{suffix}",
                        group_name=g.name,
                        group_index=idx,
                        group_sl=g.sl,
                        omega_raw=omega_j,
                        omega_ssp=omega_ssp_j,
                        rank=rank,
                        log_det_omega_plus=log_det,
                        eigvals_omega=pos_eigvals,
                    )
                )
        else:
            # Single-penalty path (unchanged).
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
                    group_sl=g.sl,
                    omega_raw=gm.omega,
                    omega_ssp=omega_ssp,
                    rank=rank,
                    log_det_omega_plus=log_det,
                    eigvals_omega=pos_eigvals,
                )
            )
    return components


def build_penalty_caches(
    group_matrices: list,
    reml_groups: list[tuple[int, object]],
) -> dict[str, PenaltyCache]:
    """Build PenaltyCache dict — thin wrapper over build_penalty_components.

    Retained for backward compatibility. New code should prefer
    build_penalty_components directly.
    """
    components = build_penalty_components(group_matrices, reml_groups)
    return {
        c.name: PenaltyCache(
            omega_ssp=c.omega_ssp,
            log_det_omega_plus=c.log_det_omega_plus,
            rank=c.rank,
            eigvals_omega=c.eigvals_omega,
        )
        for c in components
    }


def cached_logdet_s_plus(
    lambdas: dict[str, float],
    penalty_caches: dict[str, PenaltyCache],
) -> float:
    """Compute log|S|₊ from cached penalty eigenstructure.

    Wood (2011) Section 3.1: stable block-diagonal identity
    log|S|₊ = Σ_j (r_j · log(λ_j) + log|Ω_j|₊), avoiding repeated
    eigendecompositions of the full penalty matrix.

    NOTE: This is the single-penalty-per-block shortcut. For multi-penalty
    groups sharing a coefficient block, use ``compute_logdet_s_plus``
    which correctly computes the joint log-determinant.
    """
    total = 0.0
    for name, cache in penalty_caches.items():
        lam = lambdas.get(name, 1.0)
        if lam > 0 and cache.rank > 0:
            total += cache.rank * np.log(lam) + cache.log_det_omega_plus
    return total


def compute_total_penalty_rank(penalties: list[PenaltyComponent]) -> float:
    """Compute total penalty rank, correctly handling shared-block groups.

    For single-component groups, uses the component rank.
    For multi-component groups sharing a coefficient block, computes
    rank(Σ Ω_j) which is <= sum of individual ranks due to overlap.
    """
    total = 0.0
    eps_thresh = np.finfo(float).eps ** (2 / 3)
    for group_name, indices in _group_penalties(penalties).items():
        if len(indices) == 1:
            total += penalties[indices[0]].rank
        else:
            omega_sum = sum(penalties[i].omega_ssp for i in indices)
            eigvals = np.linalg.eigvalsh(omega_sum)
            thresh = eps_thresh * max(eigvals.max(), 1e-12)
            total += float(np.sum(eigvals > thresh))
    return total


def _group_penalties(penalties: list[PenaltyComponent]) -> dict[str, list[int]]:
    """Group penalty component indices by group_name."""
    groups: dict[str, list[int]] = {}
    for i, pc in enumerate(penalties):
        groups.setdefault(pc.group_name, []).append(i)
    return groups


def compute_logdet_s_plus(
    lambdas: dict[str, float],
    penalties: list[PenaltyComponent],
) -> float:
    """Compute log|S|₊ correctly for both single and multi-penalty groups.

    For single-component groups, uses the fast additive formula
    r_j · log(λ_j) + log|Ω_j|₊. For multi-component groups sharing
    a coefficient block, calls similarity_transform_logdet to compute
    log|Σ λ_j Ω_j|₊ correctly.
    """
    from superglm.multi_penalty import similarity_transform_logdet

    total = 0.0
    for group_name, indices in _group_penalties(penalties).items():
        if len(indices) == 1:
            pc = penalties[indices[0]]
            lam = lambdas.get(pc.name, 1.0)
            if lam > 0 and pc.rank > 0:
                total += pc.rank * np.log(lam) + pc.log_det_omega_plus
        else:
            comp_omegas = [penalties[i].omega_ssp for i in indices]
            comp_lambdas = np.array([lambdas.get(penalties[i].name, 1.0) for i in indices])
            result = similarity_transform_logdet(comp_omegas, comp_lambdas)
            total += result.logdet_s_plus
    return total


def compute_logdet_s_derivatives(
    lambdas: dict[str, float],
    penalties: list[PenaltyComponent],
) -> tuple[dict[str, float], dict[tuple[str, str], float]]:
    """Compute ∂log|S|₊/∂ρ_j and ∂²log|S|₊/(∂ρ_i ∂ρ_j) for multi-penalty groups.

    Returns (r_dict, hess_dict) where:
    - r_dict[pc.name] = the effective r_j for gradient: λ_j tr(S⁻¹ S_j)
    - hess_dict[(name_i, name_j)] = the log-det Hessian contribution

    For single-component groups, r_j = rank(Ω_j) (the fast shortcut).
    For multi-component groups, uses logdet_s_gradient / logdet_s_hessian.
    """
    from superglm.multi_penalty import (
        logdet_s_gradient,
        logdet_s_hessian,
        similarity_transform_logdet,
    )

    r_dict: dict[str, float] = {}
    hess_dict: dict[tuple[str, str], float] = {}

    for group_name, indices in _group_penalties(penalties).items():
        if len(indices) == 1:
            pc = penalties[indices[0]]
            r_dict[pc.name] = pc.rank
            hess_dict[(pc.name, pc.name)] = pc.rank
        else:
            comp_omegas = [penalties[i].omega_ssp for i in indices]
            comp_lambdas = np.array([lambdas.get(penalties[i].name, 1.0) for i in indices])
            result = similarity_transform_logdet(comp_omegas, comp_lambdas)
            grad = logdet_s_gradient(result, comp_omegas, comp_lambdas)
            hess = logdet_s_hessian(result, comp_omegas, comp_lambdas)
            for local_i, global_i in enumerate(indices):
                name_i = penalties[global_i].name
                r_dict[name_i] = float(grad[local_i])
                for local_j, global_j in enumerate(indices):
                    name_j = penalties[global_j].name
                    hess_dict[(name_i, name_j)] = float(hess[local_i, local_j])
    return r_dict, hess_dict


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
