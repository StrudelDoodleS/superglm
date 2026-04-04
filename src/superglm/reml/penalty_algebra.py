"""REML penalty eigenstructure and log-determinant algebra.

Pre-computes per-term penalty eigenstructure (Wood 2011 Section 3.1) and
provides log|S|₊ / ∂log|S|₊ / ∂²log|S|₊ for both single and multi-penalty
groups.

References
----------
- Wood (2011): Fast stable restricted maximum likelihood and marginal
  likelihood estimation of semiparametric generalized linear models.
  JRSS-B 73(1), 3-36.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from superglm.reml.result import PenaltyCache
from superglm.types import PenaltyComponent


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

    def _rank_and_logdet(omega_raw: NDArray, omega_ssp: NDArray) -> tuple[float, float, NDArray]:
        """Compute basis-invariant rank from raw penalty, log|Ω|₊ from SSP.

        Rank is computed from the raw (basis-invariant) penalty to avoid
        SSP-congruence sensitivity: R_inv.T @ Ω @ R_inv can shift near-null
        eigenvalues above/below the threshold depending on which valid R_inv
        was used. But log|Ω|₊ must stay in SSP coordinates because it enters
        the REML objective as log|S|₊ - log|H|, where log|H| is also in SSP
        coordinates. The 2*log|R_inv| factors cancel only when both terms
        use the same basis.
        """
        # Rank from raw penalty (basis-invariant)
        raw_eigvals = np.linalg.eigvalsh(omega_raw)
        raw_thresh = eps_thresh * max(raw_eigvals.max(), 1e-12)
        rank = float(np.sum(raw_eigvals > raw_thresh))

        # log|Ω|₊ and eigvals from SSP penalty (same basis as log|H|)
        ssp_eigvals = np.linalg.eigvalsh(omega_ssp)
        # Use the raw-basis rank to select the top eigenvalues from SSP
        n_pos = int(rank)
        if n_pos > 0:
            sorted_ssp = np.sort(ssp_eigvals)[::-1]
            pos_eigvals = sorted_ssp[:n_pos]
            log_det = float(np.sum(np.log(np.maximum(pos_eigvals, 1e-300))))
        else:
            pos_eigvals = np.array([])
            log_det = 0.0

        return rank, log_det, pos_eigvals

    for idx, g in reml_groups:
        gm = group_matrices[idx]

        if getattr(gm, "omega_components", None) is not None:
            # Multi-penalty path: N components share this coefficient block.
            ct_map = getattr(gm, "component_types", None) or {}
            for suffix, omega_j in gm.omega_components:
                omega_ssp_j = gm.R_inv.T @ omega_j @ gm.R_inv
                rank, log_det, pos_eigvals = _rank_and_logdet(omega_j, omega_ssp_j)
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
                        component_type=ct_map.get(suffix),
                    )
                )
        else:
            # Single-penalty path.
            omega_ssp = gm.R_inv.T @ gm.omega @ gm.R_inv
            rank, log_det, pos_eigvals = _rank_and_logdet(gm.omega, omega_ssp)
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
    from superglm.reml.multi_penalty import similarity_transform_logdet

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
    from superglm.reml.multi_penalty import (
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
            # Second derivative of log|λΩ|₊ = r·log(λ) + const w.r.t. ρ is 0.
            hess_dict[(pc.name, pc.name)] = 0.0
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
