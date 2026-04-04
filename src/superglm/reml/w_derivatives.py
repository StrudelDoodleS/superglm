"""W(rho) derivatives and correction for REML.

Computes dW/deta, d2W/deta2 (analytic + FD fallback), and the
W(rho) correction terms for the REML gradient and Hessian.

References
----------
- Wood (2011) Section 3.4 / Appendix C.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm.distributions import _VARIANCE_FLOOR, clip_mu
from superglm.group_matrix import (
    DesignMatrix,
    _block_xtwx_signed,
)
from superglm.links import stabilize_eta
from superglm.reml.runner import _coerce_reml_penalties
from superglm.solvers.pirls import PIRLSResult
from superglm.types import GroupSlice, PenaltyComponent


def compute_dW_deta(
    link: Any,
    distribution: Any,
    mu: NDArray,
    eta: NDArray,
    sample_weight: NDArray,
) -> NDArray | None:
    """Derivative of IRLS weights w.r.t. the linear predictor.

    W_i = exposure_i * (dmu/deta)^2 / V(mu)

    dW_i/deta = exposure_i * (dmu/deta / V(mu)) * [2(d2mu/deta2) - (dmu/deta)^2 V'(mu)/V(mu)]

    For log link: dW/deta = W*(2 - mu V'(mu)/V(mu)).
    Poisson/log: dW/deta = W. Gamma/log: dW/deta = 0 identically.

    Returns None if the link or distribution does not provide the
    required second-order methods (deriv2_inverse, variance_derivative),
    which skips the W(rho) correction for custom objects.
    """
    if not hasattr(link, "deriv2_inverse") or not hasattr(distribution, "variance_derivative"):
        return None
    g1 = link.deriv_inverse(eta)  # dmu/deta
    g2 = link.deriv2_inverse(eta)  # d2mu/deta2
    V = np.maximum(distribution.variance(mu), _VARIANCE_FLOOR)
    Vp = distribution.variance_derivative(mu)
    return sample_weight * (g1 / V) * (2.0 * g2 - g1**2 * Vp / V)


def compute_d2W_deta2(
    link: Any,
    distribution: Any,
    mu: NDArray,
    eta: NDArray,
    sample_weight: NDArray,
) -> NDArray | None:
    """Second derivative of IRLS weights w.r.t. the linear predictor.

    Analytic formula from differentiating the Appendix D expression
    dW/deta = sw * (g1/V) * [2g2 - g1^2 Vp/V].

    Let A = g1/V,  B = 2g2 - g1^2 Vp/V.
    Then dW/deta = sw * A * B, and
    d2W/deta2 = sw * (A' B + A B').

    Requires ``link.deriv3_inverse`` (d3mu/deta3) and
    ``distribution.variance_second_derivative`` (V''(mu)).
    Falls back to central finite differences of ``compute_dW_deta``
    when those methods are absent.
    """
    has_analytic = hasattr(link, "deriv3_inverse") and hasattr(
        distribution, "variance_second_derivative"
    )
    if has_analytic:
        return _compute_d2W_deta2_analytic(link, distribution, mu, eta, sample_weight)
    return _compute_d2W_deta2_fd(link, distribution, eta, sample_weight)


def _compute_d2W_deta2_analytic(
    link: Any,
    distribution: Any,
    mu: NDArray,
    eta: NDArray,
    sample_weight: NDArray,
) -> NDArray:
    """Analytic d2W/deta2 using third-order link and second-order variance."""
    g1 = link.deriv_inverse(eta)  # dmu/deta
    g2 = link.deriv2_inverse(eta)  # d2mu/deta2
    g3 = link.deriv3_inverse(eta)  # d3mu/deta3
    V = np.maximum(distribution.variance(mu), _VARIANCE_FLOOR)
    Vp = distribution.variance_derivative(mu)
    Vpp = distribution.variance_second_derivative(mu)

    # A = g1 / V
    # A' = dA/deta = (g2 V - g1 Vp g1) / V^2  = (g2 - g1^2 Vp/V) / V
    #    using chain rule: dV/deta = Vp * g1
    inv_V = 1.0 / V
    A = g1 * inv_V
    A_prime = (g2 - g1**2 * Vp * inv_V) * inv_V

    # B = 2g2 - g1^2 Vp / V
    # B' = dB/deta = 2g3 - d/deta[g1^2 Vp / V]
    #    d/deta[g1^2 Vp / V] = (2g1 g2 Vp + g1^2 Vpp g1) / V - g1^2 Vp * Vp g1 / V^2
    #                          = g1 (2g2 Vp + g1^2 Vpp) / V - g1^3 Vp^2 / V^2
    B = 2.0 * g2 - g1**2 * Vp * inv_V
    d_g1sq_Vp_over_V = g1 * (2.0 * g2 * Vp + g1**2 * Vpp) * inv_V - g1**3 * Vp**2 * inv_V**2
    B_prime = 2.0 * g3 - d_g1sq_Vp_over_V

    return sample_weight * (A_prime * B + A * B_prime)


def _compute_d2W_deta2_fd(
    link: Any,
    distribution: Any,
    eta: NDArray,
    sample_weight: NDArray,
) -> NDArray | None:
    """Finite-difference fallback for d2W/deta2.

    Central FD of ``compute_dW_deta``, used when the link or distribution
    does not provide ``deriv3_inverse`` or ``variance_second_derivative``.
    """
    eps = 1e-5
    mu_base = clip_mu(link.inverse(eta), distribution)
    dW_base = compute_dW_deta(link, distribution, mu_base, eta, sample_weight)
    if dW_base is None:
        return None

    eta_plus = eta + eps
    mu_plus = clip_mu(link.inverse(eta_plus), distribution)
    dW_plus = compute_dW_deta(link, distribution, mu_plus, eta_plus, sample_weight)

    eta_minus = eta - eps
    mu_minus = clip_mu(link.inverse(eta_minus), distribution)
    dW_minus = compute_dW_deta(link, distribution, mu_minus, eta_minus, sample_weight)

    if dW_plus is None or dW_minus is None:
        return None

    return (dW_plus - dW_minus) / (2.0 * eps)


def reml_w_correction(
    dm: DesignMatrix,
    link: Any,
    groups: list[GroupSlice],
    pirls_result: PIRLSResult,
    XtWX_S_inv: NDArray,
    lambdas: dict[str, float],
    reml_groups=None,
    penalty_caches: dict | None = None,
    sample_weight: NDArray | None = None,
    offset_arr: NDArray | None = None,
    distribution: Any = None,
    w_correction_order: int = 1,
    *,
    reml_penalties: list[PenaltyComponent] | None = None,
) -> tuple[NDArray, dict[int, NDArray]] | tuple[NDArray, dict[int, NDArray], NDArray | None] | None:
    """W(rho) correction for REML derivatives (first- or second-order).

    Wood (2011) Section 3.4 / Appendix C: implicit differentiation of beta_hat(rho)
    through W(eta(rho)) using the chain dbeta_hat/drho = -H^{-1} S_j beta_hat (IFT on the
    PIRLS stationarity condition).

    Computes the contribution from d(X'WX)/drho_j = X'diag(dW/drho_j)X
    which the fixed-W Laplace approximation drops.  The gradient
    correction is exact to first order; the Hessian C_j matrices are
    first-order (d2W/drho2 terms are dropped) unless ``w_correction_order=2``.

    When ``w_correction_order=2``, the full second-order Hessian correction
    from Section 3.5.1 is computed::

        d2w/(drho_j drho_k) = (d2w/deta2)*(deta/drho_j)*(deta/drho_k) + (dw/deta)*(d2eta/(drho_j drho_k))

    where d2eta/(drho_j drho_k) = X * d2beta_hat/(drho_j drho_k) from Section 3.4.

    Parameters
    ----------
    w_correction_order : int, default 1
        1 = first-order only (backward compatible).
        2 = include second-order Hessian cross-terms (Wood 2011 Section 3.5.1).

    Returns ``(grad_correction, dH_extra)`` when ``w_correction_order=1``
    (backward compatible 2-tuple), or
    ``(grad_correction, dH_extra, dH2_cross)`` when
    ``w_correction_order=2`` (3-tuple).  Returns None if the correction
    vanishes (e.g. Gamma with log link where dW/deta = 0 identically) or
    if the link/distribution does not provide the required methods.

    dH2_cross is an (m, m) array of second-order Hessian corrections:
    ``dH2_cross[j,k] = 0.5 * tr(H^{-1} X' diag(d2w/(drho_j drho_k)) X)``.
    """
    penalties = _coerce_reml_penalties(
        reml_groups=reml_groups,
        reml_penalties=reml_penalties,
        group_matrices=dm.group_matrices,
        penalty_caches=penalty_caches,
    )
    eta = stabilize_eta(
        dm.matvec(pirls_result.beta) + pirls_result.intercept + offset_arr,
        link,
    )
    mu = clip_mu(link.inverse(eta), distribution)
    dW_deta = compute_dW_deta(link, distribution, mu, eta, sample_weight)

    if dW_deta is None:
        return None  # Custom link/distribution w/o 2nd-order

    if np.max(np.abs(dW_deta)) < 1e-12:
        return None  # No correction (e.g. Gamma/log)

    p = XtWX_S_inv.shape[0]
    m = len(penalties)
    grad_correction = np.zeros(m)
    dH_extra: dict[int, NDArray] = {}

    gms = dm.group_matrices

    # Pre-compute d2W/deta2 for second-order path
    d2W_deta2: NDArray | None = None
    if w_correction_order >= 2:
        d2W_deta2 = compute_d2W_deta2(link, distribution, mu, eta, sample_weight)

    # Store per-group quantities for second-order cross-terms
    deta_vectors: list[NDArray] = []
    dbeta_vectors: list[NDArray] = []
    omega_ssp_list: list[NDArray] = []
    lam_list: list[float] = []

    for i, pc in enumerate(penalties):
        omega_ssp = pc.omega_ssp
        if omega_ssp is None:
            if penalty_caches is not None and pc.name in penalty_caches:
                omega_ssp = penalty_caches[pc.name].omega_ssp
            else:
                gm = gms[pc.group_index]
                omega_ssp = gm.R_inv.T @ gm.omega @ gm.R_inv
        lam = lambdas[pc.name]
        beta_g = pirls_result.beta[pc.group_sl]

        # S_j beta (p-vector, nonzero only in pc.group_sl block)
        s_beta = np.zeros(p)
        s_beta[pc.group_sl] = lam * (omega_ssp @ beta_g)

        # dbeta/drho_j = -H^{-1} S_j beta  (IFT)
        dbeta_j = -(XtWX_S_inv @ s_beta)

        # deta/drho_j = X dbeta/drho_j
        deta_j = dm.matvec(dbeta_j)

        # a_j = (dW/deta) * deta_j  -- weight changes per obs
        a_j = dW_deta * deta_j

        # C_j = X'diag(a_j)X -- dW contribution to dH/drho_j
        C_j = _block_xtwx_signed(gms, groups, a_j, tabmat_split=dm.tabmat_split)

        # Gradient correction: 0.5 tr(H^{-1} C_j)
        grad_correction[i] = 0.5 * float(np.sum(XtWX_S_inv * C_j))

        dH_extra[i] = C_j

        if w_correction_order >= 2:
            deta_vectors.append(deta_j)
            dbeta_vectors.append(dbeta_j)
            omega_ssp_list.append(omega_ssp)
            lam_list.append(lam)

    # -- Second-order Hessian cross-terms (Wood 2011, Section 3.5.1) --
    #
    # Cost: O(m^2 * n * p^2) per Newton iteration -- m^2/2 gram operations via
    # _block_xtwx_signed, plus m^2/2 rmatvec + matvec calls.  For typical
    # m=4, p=30 this is ~10 grams at ~30ms each (MTPL2 678k).
    dH2_cross: NDArray | None = None
    if w_correction_order >= 2 and d2W_deta2 is not None:
        dH2_cross = np.zeros((m, m))
        for i in range(m):
            pc_i = penalties[i]
            for j in range(i, m):
                pc_j = penalties[j]

                # f^{jk} vector (Section 3.4, eq for d2beta)
                f_jk = 0.5 * deta_vectors[i] * deta_vectors[j] * dW_deta

                # X^T f^{jk}
                Xt_f = dm.rmatvec(f_jk)

                # lam_i S_i dbeta/drho_j  (nonzero in pc_i block)
                lam_i_S_i_dbeta_j = np.zeros(p)
                lam_i_S_i_dbeta_j[pc_i.group_sl] = lam_list[i] * (
                    omega_ssp_list[i] @ dbeta_vectors[j][pc_i.group_sl]
                )

                # lam_j S_j dbeta/drho_i  (nonzero in pc_j block)
                lam_j_S_j_dbeta_i = np.zeros(p)
                lam_j_S_j_dbeta_i[pc_j.group_sl] = lam_list[j] * (
                    omega_ssp_list[j] @ dbeta_vectors[i][pc_j.group_sl]
                )

                # rhs = X^T f^{jk} + lam_i S_i dbeta_j + lam_j S_j dbeta_i
                rhs = Xt_f + lam_i_S_i_dbeta_j + lam_j_S_j_dbeta_i

                # d2beta_hat/(drho_i drho_j) = delta_ij * dbeta_hat/drho_j - H^{-1} rhs
                d2beta_ij = -(XtWX_S_inv @ rhs)
                if i == j:
                    d2beta_ij += dbeta_vectors[j]

                # d2eta/(drho_i drho_j) = X * d2beta_hat/(drho_i drho_j)
                d2eta_ij = dm.matvec(d2beta_ij)

                # Full d2w/(drho_i drho_j) (Section 3.5.1 T_{jk} derivation):
                d2w_drho_ij = d2W_deta2 * deta_vectors[i] * deta_vectors[j] + dW_deta * d2eta_ij

                # Hessian correction: 0.5 * tr(H^{-1} X' diag(d2w_drho_ij) X)
                C_ij = _block_xtwx_signed(gms, groups, d2w_drho_ij, tabmat_split=dm.tabmat_split)
                val = 0.5 * float(np.sum(XtWX_S_inv * C_ij))
                dH2_cross[i, j] = val
                dH2_cross[j, i] = val

    if w_correction_order >= 2:
        return grad_correction, dH_extra, dH2_cross
    return grad_correction, dH_extra
