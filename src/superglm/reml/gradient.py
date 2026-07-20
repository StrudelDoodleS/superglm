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
    compute_penalty_nullity,
    compute_total_penalty_rank,
)
from superglm.solvers.pirls import PIRLSResult
from superglm.types import PenaltyComponent


def _penalty_block_trace(
    XtWX_S_inv: NDArray,
    sl_i: slice,
    weighted_omega_i: NDArray,
    sl_j: slice,
    weighted_omega_j: NDArray,
) -> float:
    """Return tr(H^-1 dS_i H^-1 dS_j) without materialising p x p blocks."""
    H_ji = XtWX_S_inv[sl_j, :][:, sl_i]
    H_ij = XtWX_S_inv[sl_i, :][:, sl_j]
    return float(np.trace(H_ji @ weighted_omega_i @ H_ij @ weighted_omega_j))


def _same_slice(sl_i: slice, sl_j: slice) -> bool:
    return bool(sl_i.start == sl_j.start and sl_i.stop == sl_j.stop and sl_i.step == sl_j.step)


def reml_direct_gradient(
    group_matrices: list,
    result: PIRLSResult,
    XtWX_S_inv: NDArray,
    lambdas: dict[str, float],
    reml_groups=None,
    penalty_ranks: dict[str, float] | None = None,
    phi_hat: float = 1.0,
    *,
    inverse_phi: float | None = None,
    reml_penalties: list[PenaltyComponent] | None = None,
    penalty_caches: dict | None = None,
    tensor_pair_evaluations: dict | None = None,
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
    r_dict, _ = compute_logdet_s_derivatives(
        lambdas,
        penalties,
        tensor_pair_evaluations=tensor_pair_evaluations,
    )

    grad = np.zeros(len(penalties), dtype=np.float64)
    inv_phi = 1.0 / max(phi_hat, 1e-10) if inverse_phi is None else float(inverse_phi)
    if not np.isfinite(inv_phi) or inv_phi <= 0.0:
        raise ValueError("inverse_phi must be positive and finite")
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
    pirls_result: PIRLSResult | None = None,
    n_obs: int = 0,
    phi_hat: float = 1.0,
    penalty_nullity: float | None = None,
    dH_extra: dict[int, NDArray] | None = None,
    dH2_cross: NDArray | None = None,
    *,
    inverse_phi: float | None = None,
    d_inverse_phi_d_penalized_deviance: float | None = None,
    reml_penalties: list[PenaltyComponent] | None = None,
    tensor_pair_evaluations: dict | None = None,
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
    if gradient is None:
        raise ValueError("gradient is required for the direct REML Hessian")
    m = len(penalties)
    p = XtWX_S_inv.shape[0]
    hess = np.zeros((m, m))
    use_compact_trace = dH_extra is None

    # Pre-compute log-det derivatives for multi-penalty groups.
    # r_logdet: first derivative d(log|S|_+)/drho_i
    # h_logdet: second derivative d2(log|S|_+)/(drho_i drho_j)
    # For single-penalty, r_i = rank and h_ii = rank (they're equal).
    # For shared-block multi-penalty, h_ij is non-trivial and needed
    # to correct the Hessian curvature for the anisotropy directions.
    r_logdet, h_logdet = compute_logdet_s_derivatives(
        lambdas,
        penalties,
        tensor_pair_evaluations=tensor_pair_evaluations,
    )

    full_HdHj: dict[int, NDArray] = {}
    compact_dS: list[tuple[slice, NDArray]] = []
    same_slice_H_blocks: dict[tuple[int | None, int | None, int | None], NDArray] = {}
    same_slice_products: dict[int, NDArray] = {}
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
        weighted_omega = lam * omega_ssp
        if use_compact_trace:
            compact_dS.append((pc.group_sl, weighted_omega))
        else:
            F = np.zeros((p, p))
            F[:, pc.group_sl] = XtWX_S_inv[:, pc.group_sl] @ weighted_omega
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
            if use_compact_trace:
                sl_i, weighted_omega_i = compact_dS[i]
                sl_j, weighted_omega_j = compact_dS[j]
                if _same_slice(sl_i, sl_j):
                    key_i = (sl_i.start, sl_i.stop, sl_i.step)
                    H_block = same_slice_H_blocks.get(key_i)
                    if H_block is None:
                        H_block = XtWX_S_inv[sl_i, sl_i]
                        same_slice_H_blocks[key_i] = H_block
                    A_i = same_slice_products.get(i)
                    if A_i is None:
                        A_i = H_block @ weighted_omega_i
                        same_slice_products[i] = A_i
                    A_j = same_slice_products.get(j)
                    if A_j is None:
                        A_j = H_block @ weighted_omega_j
                        same_slice_products[j] = A_j
                    h = -0.5 * float(np.sum(A_i * A_j.T))
                else:
                    h = -0.5 * _penalty_block_trace(
                        XtWX_S_inv,
                        sl_i,
                        weighted_omega_i,
                        sl_j,
                        weighted_omega_j,
                    )
            else:
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
        inv_phi = 1.0 / max(phi_hat, 1e-10) if inverse_phi is None else float(inverse_phi)
        if not np.isfinite(inv_phi) or inv_phi <= 0.0:
            raise ValueError("inverse_phi must be positive and finite")
        S_beta = np.column_stack(s_beta_list)
        HinvSbeta = XtWX_S_inv @ S_beta
        hess -= inv_phi * (S_beta.T @ HinvSbeta)

    scale_known = getattr(distribution, "scale_known", True)
    if d_inverse_phi_d_penalized_deviance is not None and pirls_result is not None:
        inverse_phi_derivative = float(d_inverse_phi_d_penalized_deviance)
        if not np.isfinite(inverse_phi_derivative):
            raise ValueError("profiled inverse-scale derivative must be finite")
        q = np.asarray(quad_per_group, dtype=np.float64)
        hess += 0.5 * inverse_phi_derivative * np.outer(q, q)
    elif not scale_known and pirls_result is not None and n_obs > 0:
        if penalty_nullity is None:
            hessian_rank = getattr(pirls_result, "reml_hessian_rank", None)
            if hessian_rank is not None:
                M_p = compute_penalty_nullity(
                    hessian_rank=hessian_rank,
                    penalties=penalties,
                    lambdas=lambdas,
                    coefficient_width=p,
                )
            else:
                # Compatibility for synthetic/non-direct callers that do not
                # carry identified full-H metadata.
                M_p = compute_total_penalty_rank(
                    penalties,
                    tensor_pair_evaluations=tensor_pair_evaluations,
                )
                if M_p <= 0 and penalty_ranks is not None:
                    M_p = sum(penalty_ranks[pc.name] for pc in penalties)
        else:
            M_p = penalty_nullity
        pq_total = sum(quad_per_group)
        d_plus_pq = max(pirls_result.deviance + pq_total, 1e-300)
        q = np.array(quad_per_group)
        hess -= 0.5 * max(n_obs - M_p, 1.0) * np.outer(q, q) / d_plus_pq**2

    # Second-order W(rho) cross-term from d2W/(drho_j drho_k)
    if dH2_cross is not None:
        hess += dH2_cross

    return hess
