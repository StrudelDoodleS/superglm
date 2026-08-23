"""REML gradient and Hessian w.r.t. log-lambdas.

Wood (2011) Appendix B / Eq 6.2.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm.reml.penalty_algebra import (
    coerce_reml_penalties,
    compute_logdet_s_derivatives,
    compute_penalty_nullity,
    compute_total_penalty_rank,
    penalty_component_dense_matrix,
    penalty_component_matvec,
    penalty_component_quadratic,
)
from superglm.solvers.hessian_factor import (
    DenseHessianFactor,
    HessianFactor,
    as_hessian_factor,
)
from superglm.solvers.pirls import PIRLSResult
from superglm.solvers.structured import CompactSymmetricOperator
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
    group_matrices: Sequence,
    result: PIRLSResult,
    XtWX_S_inv: NDArray | HessianFactor,
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
    factor = as_hessian_factor(XtWX_S_inv)
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
        gm = group_matrices[pc.group_index] if 0 <= pc.group_index < len(group_matrices) else None
        beta_g = result.beta[pc.group_sl]
        quad = penalty_component_quadratic(pc, beta_g, gm)
        trace_term = factor.trace_inverse_penalty(pc)
        lam = float(lambdas[pc.name])
        r_j = r_dict.get(pc.name, pc.rank)
        if r_j <= 0 and penalty_ranks is not None:
            r_j = penalty_ranks.get(pc.name, 0.0)
        grad[i] = 0.5 * (lam * (inv_phi * quad + trace_term) - r_j)
    return grad


def reml_direct_hessian(
    group_matrices: Sequence,
    distribution: Any,
    XtWX_S_inv: NDArray | HessianFactor,
    lambdas: dict[str, float],
    reml_groups=None,
    gradient: NDArray | None = None,
    penalty_ranks: dict[str, float] | None = None,
    penalty_caches: dict | None = None,
    pirls_result: PIRLSResult | None = None,
    # The declared weight contract's likelihood size, not the physical row
    # count -- `sum(w)` under `"frequency"`, the positive-row count under
    # `"prior"`. It is the `n` in this Hessian's `0.5 * (n - M_p) * log(D)`
    # scale term, and it must match the `n` the OBJECTIVE uses for that same
    # term or the Newton step is inconsistent with the surface it steps on.
    # Float, because `sum(w)` is not an integer.
    n_obs: float = 0.0,
    phi_hat: float = 1.0,
    penalty_nullity: float | None = None,
    dH_extra: Mapping[int, NDArray | CompactSymmetricOperator] | None = None,
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
    factor = as_hessian_factor(XtWX_S_inv)
    dense_inverse = factor.inverse if isinstance(factor, DenseHessianFactor) else None
    m = len(penalties)
    p = factor.shape[0]
    hess = np.zeros((m, m))
    use_compact_trace = dH_extra is None or dense_inverse is None

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
    compact_dS: list[tuple[PenaltyComponent, float, Any]] = []
    same_slice_H_blocks: dict[tuple[int | None, int | None, int | None], NDArray] = {}
    same_slice_products: dict[int, NDArray] = {}
    quad_per_group: list[float] = []
    s_beta_list: list[NDArray] = []
    for i, pc in enumerate(penalties):
        gm = group_matrices[pc.group_index] if 0 <= pc.group_index < len(group_matrices) else None
        lam = lambdas[pc.name]
        if use_compact_trace:
            compact_dS.append((pc, lam, gm))
        else:
            F = np.zeros((p, p))
            if dense_inverse is None:  # pragma: no cover - branch invariant
                raise RuntimeError("Dense derivative branch has no dense inverse.")
            inverse_columns = dense_inverse[:, pc.group_sl]
            F[:, pc.group_sl] = (
                lam * inverse_columns
                if pc.penalty_kind == "identity"
                else inverse_columns @ (lam * penalty_component_dense_matrix(pc, gm))
            )
            if dH_extra is not None and i in dH_extra:
                F = F + factor.solve(dH_extra[i])
            full_HdHj[i] = F

        if pirls_result is not None:
            beta_g = pirls_result.beta[pc.group_sl]
            quad_per_group.append(lam * penalty_component_quadratic(pc, beta_g, gm))
            v = np.zeros(p)
            v[pc.group_sl] = lam * penalty_component_matvec(pc, beta_g, gm)
            s_beta_list.append(v)
        else:
            quad_per_group.append(0.0)
            s_beta_list.append(np.zeros(p))

    for i in range(m):
        for j in range(i, m):
            if use_compact_trace:
                pc_i, lam_i, gm_i = compact_dS[i]
                pc_j, lam_j, gm_j = compact_dS[j]
                if _same_slice(pc_i.group_sl, pc_j.group_sl):
                    key_i = (
                        pc_i.group_sl.start,
                        pc_i.group_sl.stop,
                        pc_i.group_sl.step,
                    )
                    if dense_inverse is not None:
                        H_block = same_slice_H_blocks.get(key_i)
                        if H_block is None:
                            H_block = dense_inverse[pc_i.group_sl, pc_i.group_sl]
                            same_slice_H_blocks[key_i] = H_block
                        A_i = same_slice_products.get(i)
                        if A_i is None:
                            if pc_i.penalty_kind == "identity":
                                A_i = lam_i * H_block
                            else:
                                A_i = H_block @ (lam_i * penalty_component_dense_matrix(pc_i, gm_i))
                            same_slice_products[i] = A_i
                        A_j = same_slice_products.get(j)
                        if A_j is None:
                            if pc_j.penalty_kind == "identity":
                                A_j = lam_j * H_block
                            else:
                                A_j = H_block @ (lam_j * penalty_component_dense_matrix(pc_j, gm_j))
                            same_slice_products[j] = A_j
                        trace_value = float(np.sum(A_i * A_j.T))
                    else:
                        trace_value = factor.penalty_cross_trace(
                            pc_i,
                            pc_j,
                            lam_i,
                            lam_j,
                        )
                    if dH_extra is not None:
                        left_extra = dH_extra.get(i)
                        right_extra = dH_extra.get(j)
                        if left_extra is not None:
                            if isinstance(left_extra, np.ndarray):
                                raise TypeError(
                                    "Structured Hessian correction received a dense operator."
                                )
                            trace_value += factor.penalty_operator_cross_trace(
                                pc_j,
                                lam_j,
                                left_extra,
                            )
                        if right_extra is not None:
                            if isinstance(right_extra, np.ndarray):
                                raise TypeError(
                                    "Structured Hessian correction received a dense operator."
                                )
                            trace_value += factor.penalty_operator_cross_trace(
                                pc_i,
                                lam_i,
                                right_extra,
                            )
                        if left_extra is not None and right_extra is not None:
                            assert not isinstance(left_extra, np.ndarray)
                            assert not isinstance(right_extra, np.ndarray)
                            trace_value += factor.operator_cross_trace(
                                left_extra,
                                right_extra,
                            )
                    h = -0.5 * trace_value
                else:
                    trace_value = factor.penalty_cross_trace(
                        pc_i,
                        pc_j,
                        lam_i,
                        lam_j,
                    )
                    if dH_extra is not None:
                        left_extra = dH_extra.get(i)
                        right_extra = dH_extra.get(j)
                        if left_extra is not None:
                            if isinstance(left_extra, np.ndarray):
                                raise TypeError(
                                    "Structured Hessian correction received a dense operator."
                                )
                            trace_value += factor.penalty_operator_cross_trace(
                                pc_j,
                                lam_j,
                                left_extra,
                            )
                        if right_extra is not None:
                            if isinstance(right_extra, np.ndarray):
                                raise TypeError(
                                    "Structured Hessian correction received a dense operator."
                                )
                            trace_value += factor.penalty_operator_cross_trace(
                                pc_i,
                                lam_i,
                                right_extra,
                            )
                        if left_extra is not None and right_extra is not None:
                            assert not isinstance(left_extra, np.ndarray)
                            assert not isinstance(right_extra, np.ndarray)
                            trace_value += factor.operator_cross_trace(
                                left_extra,
                                right_extra,
                            )
                    h = -0.5 * trace_value
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
        HinvSbeta = factor.solve(S_beta)
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
