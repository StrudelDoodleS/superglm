"""REML gradient and Hessian w.r.t. log-lambdas.

Wood (2011) Appendix B / Eq 6.2.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm.reml.penalty_algebra import (
    coerce_reml_penalties,
    compute_logdet_s_derivatives,
    compute_total_penalty_rank,
)
from superglm.solvers.pirls import PIRLSResult
from superglm.types import PenaltyComponent


def reml_direct_gradient(
    group_matrices: list,
    result: PIRLSResult,
    XtWX_S_inv: NDArray,
    lambdas: dict[str, float],
    reml_groups=None,
    penalty_ranks: dict[str, float] | None = None,
    phi_hat: float = 1.0,
    *,
    reml_penalties: list[PenaltyComponent] | None = None,
    penalty_caches: dict | None = None,
) -> NDArray:
    """Partial gradient of the LAML objective w.r.t. log-lambdas (fixed W)."""
    penalties = coerce_reml_penalties(
        reml_groups=reml_groups,
        reml_penalties=reml_penalties,
        group_matrices=group_matrices,
        penalty_caches=penalty_caches,
    )

    # Pre-compute log-det derivatives for multi-penalty groups.
    # For single-penalty groups, r_j = rank(Omega_j) (fast shortcut).
    # For multi-penalty groups sharing a block, r_j = lambda_j tr(S^{-1} S_j).
    r_dict, _ = compute_logdet_s_derivatives(lambdas, penalties)

    grad = np.zeros(len(penalties), dtype=np.float64)
    inv_phi = 1.0 / max(phi_hat, 1e-10)
    for i, pc in enumerate(penalties):
        gm = group_matrices[pc.group_index]
        omega_ssp = pc.omega_ssp if pc.omega_ssp is not None else gm.R_inv.T @ gm.omega @ gm.R_inv
        beta_g = result.beta[pc.group_sl]
        quad = float(beta_g @ omega_ssp @ beta_g)
        H_inv_jj = XtWX_S_inv[pc.group_sl, pc.group_sl]
        trace_term = float(np.trace(H_inv_jj @ omega_ssp))
        lam = float(lambdas[pc.name])
        r_j = r_dict.get(pc.name, pc.rank)
        if r_j <= 0 and penalty_ranks is not None:
            r_j = penalty_ranks.get(pc.name, 0.0)
        grad[i] = 0.5 * (lam * (inv_phi * quad + trace_term) - r_j)
    return grad


def reml_direct_hessian(
    group_matrices: list,
    distribution: Any,
    XtWX_S_inv: NDArray,
    lambdas: dict[str, float],
    reml_groups=None,
    gradient: NDArray | None = None,
    penalty_ranks: dict[str, float] | None = None,
    penalty_caches: dict | None = None,
    pirls_result: object | None = None,
    n_obs: int = 0,
    phi_hat: float = 1.0,
    dH_extra: dict[int, NDArray] | None = None,
    dH2_cross: NDArray | None = None,
    *,
    reml_penalties: list[PenaltyComponent] | None = None,
) -> NDArray:
    """Outer Hessian of the REML criterion w.r.t. log-lambdas.

    Wood (2011) Appendix B / Eq 6.2.  Uses implicit-differentiation
    Jacobian (outer products of H^{-1} dH_j) rather than explicit K/T
    matrices from Appendix B.

    Parameters
    ----------
    dH2_cross : NDArray or None
        Second-order W(rho) Hessian correction from ``reml_w_correction``
        with ``w_correction_order=2``.  An (m, m) matrix of
        0.5 * tr(H^{-1} X'diag(d2w/(drho_j drho_k))X) values,
        added directly to the Hessian.
    """
    penalties = coerce_reml_penalties(
        reml_groups=reml_groups,
        reml_penalties=reml_penalties,
        group_matrices=group_matrices,
        penalty_caches=penalty_caches,
    )
    m = len(penalties)
    p = XtWX_S_inv.shape[0]
    hess = np.zeros((m, m))

    # Pre-compute log-det derivatives for multi-penalty groups.
    # r_logdet: first derivative d(log|S|_+)/drho_i
    # h_logdet: second derivative d2(log|S|_+)/(drho_i drho_j)
    # For single-penalty, r_i = rank and h_ii = rank (they're equal).
    # For shared-block multi-penalty, h_ij is non-trivial and needed
    # to correct the Hessian curvature for the anisotropy directions.
    r_logdet, h_logdet = compute_logdet_s_derivatives(lambdas, penalties)

    full_HdHj: dict[int, NDArray] = {}
    quad_per_group: list[float] = []
    s_beta_list: list[NDArray] = []
    for i, pc in enumerate(penalties):
        omega_ssp = pc.omega_ssp
        if omega_ssp is None:
            if penalty_caches is not None and pc.name in penalty_caches:
                omega_ssp = penalty_caches[pc.name].omega_ssp
            else:
                gm = group_matrices[pc.group_index]
                omega_ssp = gm.R_inv.T @ gm.omega @ gm.R_inv
        lam = lambdas[pc.name]
        F = np.zeros((p, p))
        F[:, pc.group_sl] = XtWX_S_inv[:, pc.group_sl] @ (lam * omega_ssp)

        if dH_extra is not None and i in dH_extra:
            F = F + XtWX_S_inv @ dH_extra[i]

        full_HdHj[i] = F

        if pirls_result is not None:
            beta_g = pirls_result.beta[pc.group_sl]
            quad_per_group.append(lam * float(beta_g @ omega_ssp @ beta_g))
            v = np.zeros(p)
            v[pc.group_sl] = lam * (omega_ssp @ beta_g)
            s_beta_list.append(v)
        else:
            quad_per_group.append(0.0)
            s_beta_list.append(np.zeros(p))

    for i in range(m):
        for j in range(i, m):
            h = -0.5 * float(np.sum(full_HdHj[i] * full_HdHj[j].T))
            hess[i, j] = h
            hess[j, i] = h

    # Wood (2011) Eq 6.2: diagonal includes g_i + 0.5 * r_i.
    for i in range(m):
        r_i = r_logdet.get(penalties[i].name, penalties[i].rank)
        if r_i <= 0 and penalty_ranks is not None:
            r_i = penalty_ranks.get(penalties[i].name, 0.0)
        hess[i, i] += gradient[i] + 0.5 * r_i

    # Shared-block log|S|_+ Hessian correction: -0.5 * d2(log|S|_+)/(drho_i drho_j).
    # For single-penalty, h_logdet = 0 (log|lambda*Omega|_+ is linear in rho), so this
    # is a no-op. For shared-block multi-penalty, the non-zero cross-terms
    # give the Newton step proper curvature for the anisotropy directions.
    for (name_i, name_j), h_ij in h_logdet.items():
        if h_ij == 0.0:
            continue
        i = next(k for k, pc in enumerate(penalties) if pc.name == name_i)
        j = next(k for k, pc in enumerate(penalties) if pc.name == name_j)
        hess[i, j] -= 0.5 * h_ij

    if pirls_result is not None:
        inv_phi = 1.0 / max(phi_hat, 1e-10)
        S_beta = np.column_stack(s_beta_list)
        HinvSbeta = XtWX_S_inv @ S_beta
        hess -= inv_phi * (S_beta.T @ HinvSbeta)

    scale_known = getattr(distribution, "scale_known", True)
    if not scale_known and pirls_result is not None and n_obs > 0:
        M_p = compute_total_penalty_rank(penalties)
        if M_p <= 0 and penalty_ranks is not None:
            M_p = sum(penalty_ranks[pc.name] for pc in penalties)
        pq_total = sum(quad_per_group)
        d_plus_pq = max(pirls_result.deviance + pq_total, 1e-300)
        q = np.array(quad_per_group)
        hess -= 0.5 * max(n_obs - M_p, 1.0) * np.outer(q, q) / d_plus_pq**2

    # Second-order W(rho) cross-term from d2W/(drho_j drho_k)
    if dH2_cross is not None:
        hess += dH2_cross

    return hess
