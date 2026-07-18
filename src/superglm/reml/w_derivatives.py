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

from superglm._group_matrix._group_matrix_centered import (
    _raw_centering_well_scaled,
    centered_gram_rhs,
)
from superglm.distributions import _VARIANCE_FLOOR, Gamma, clip_mu
from superglm.group_matrix import DesignMatrix
from superglm.links import LogLink, stabilize_eta
from superglm.reml.observed_geometry import ObservedREMLGeometry
from superglm.reml.penalty_algebra import coerce_reml_penalties
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
    if isinstance(distribution, Gamma) and isinstance(link, LogLink):
        return np.zeros_like(mu, dtype=np.float64)
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
    if isinstance(distribution, Gamma) and isinstance(link, LogLink):
        return np.zeros_like(mu, dtype=np.float64)
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
    geometry: ObservedREMLGeometry | None = None,
) -> tuple[NDArray, dict[int, NDArray]] | tuple[NDArray, dict[int, NDArray], NDArray | None] | None:
    """W(rho) correction for REML derivatives (first- or second-order).

    Wood (2011) Section 3.4 / Appendix C: implicit differentiation of beta_hat(rho)
    through W(eta(rho)) using the chain dbeta_hat/drho = -H^{-1} S_j beta_hat (IFT on the
    PIRLS stationarity condition).

    Computes the contribution from
    ``d(X_c' W X_c)/drho_j = X_c' diag(dW/drho_j) X_c`` and from the
    profiled scalar determinant ``log(sum(W))``, which the fixed-W Laplace
    approximation drops.  The gradient correction is exact to first order;
    second derivatives are dropped unless ``w_correction_order=2``.

    When ``w_correction_order=2``, the full second-order Hessian correction
    from Section 3.5.1 is computed::

        d2w/(drho_j drho_k) = (d2w/deta2)*(deta/drho_j)*(deta/drho_k) + (dw/deta)*(d2eta/(drho_j drho_k))

    In the profiled-intercept coordinates used here,
    ``d2eta_jk = X_c d2beta_jk - dmean_x_k' dbeta_j``.  Differentiating
    ``X_c' W X_c`` also contributes the two weighted-mean outer products.

    Parameters
    ----------
    w_correction_order : int, default 1
        1 = first-order only (backward compatible).
        2 = include second-order Hessian cross-terms (Wood 2011 Section 3.5.1).

    Returns ``(grad_correction, dH_extra)`` when ``w_correction_order=1``
    (backward compatible 2-tuple), or
    ``(grad_correction, dH_extra, dH2_cross)`` when
    ``w_correction_order=2`` (3-tuple).  Returns None if the correction
    vanishes under the selected geometry or
    if the link/distribution does not provide the required methods.

    ``geometry`` supplies the observed-information rows and profiled-intercept
    metadata required by Wood's LAML criterion.  When it is omitted, the
    historical Fisher working geometry is retained for callers that optimize
    a working-model criterion.  The inverse passed as ``XtWX_S_inv`` must
    correspond to the same geometry.

    ``dH2_cross`` is an ``(m, m)`` array of second-order Hessian
    corrections.  It includes the second derivative of the profiled data
    Hessian ``X_c' W X_c`` (including both weighted-mean outer terms) and the
    scalar curvature of ``0.5 * log(sum(W))``.
    """
    w_correction_order = validate_w_correction_order(w_correction_order)
    penalties = coerce_reml_penalties(
        reml_groups=reml_groups,
        reml_penalties=reml_penalties,
        group_matrices=dm.group_matrices,
        penalty_caches=penalty_caches,
    )
    if geometry is None:
        eta = stabilize_eta(
            dm.matvec(pirls_result.beta) + pirls_result.intercept + offset_arr,
            link,
        )
        mu = clip_mu(link.inverse(eta), distribution)
        dW_deta = compute_dW_deta(link, distribution, mu, eta, sample_weight)
    else:
        if geometry.eta.shape != (dm.n,) or geometry.mu.shape != (dm.n,):
            raise ValueError("observed REML geometry does not match the design rows")
        if geometry.hessian_inverse is None:
            raise ValueError("observed REML geometry omitted the slope inverse")
        if geometry.weight_derivative is None:
            raise ValueError("observed REML geometry omitted first weight derivatives")
        eta = geometry.eta
        mu = geometry.mu
        dW_deta = geometry.weight_derivative
        # Geometry is an atomic determinant/inverse/centering payload.  Never
        # permit a separately supplied Fisher inverse to create a hybrid that
        # does not differentiate any coherent LAML criterion.
        XtWX_S_inv = geometry.hessian_inverse

    if dW_deta is None:
        return None  # Custom link/distribution w/o 2nd-order

    if not np.any(dW_deta):
        return None  # Structurally constant working curvature.

    p = XtWX_S_inv.shape[0]
    m = len(penalties)
    grad_correction = np.zeros(m)
    dH_extra: dict[int, NDArray] = {}

    gms = dm.group_matrices
    if geometry is not None:
        mean_x = np.asarray(geometry.mean_x, dtype=np.float64)
        sum_w: float | None = float(geometry.sum_w)
        if mean_x.shape != (p,):
            raise ValueError("observed REML geometry does not match the coefficient space")
        centered_diagonal = np.diag(geometry.centered_data_gram)
        with np.errstate(invalid="ignore", divide="ignore"):
            centered_scale = np.sqrt(np.abs(centered_diagonal) / sum_w)
        use_stable_signed_gram = not np.all(
            np.isfinite(centered_scale)
        ) or not _raw_centering_well_scaled(mean_x, centered_scale)
    elif pirls_result.rank_info is None:
        mean_x = np.zeros(p)
        sum_w = None
        use_stable_signed_gram = False
    else:
        mean_x = np.asarray(pirls_result.rank_info.mean_x, dtype=np.float64)
        if mean_x.shape != (p,):
            raise ValueError("PIRLS rank metadata does not match the REML coefficient space")
        sum_w = float(pirls_result.rank_info.sum_w)
        if not np.isfinite(sum_w) or sum_w <= 0.0:
            raise ValueError("PIRLS rank metadata has an invalid working-weight sum")
        use_stable_signed_gram = False
        rank_data = getattr(pirls_result.rank_info, "data", None)
        column_scale = getattr(rank_data, "column_scale", None)
        if column_scale is not None:
            centered_scale = np.asarray(column_scale, dtype=np.float64) / np.sqrt(sum_w)
            if centered_scale.shape == mean_x.shape:
                use_stable_signed_gram = not _raw_centering_well_scaled(
                    mean_x,
                    centered_scale,
                )
    stable_gram_rhs = np.zeros(dm.n, dtype=np.float64) if use_stable_signed_gram else None

    def centered_matvec(values: NDArray) -> NDArray:
        """Apply the profiled-intercept design ``X - 1 mean_x'``."""
        return dm.matvec(values) - float(mean_x @ values)

    def centered_rmatvec(values: NDArray) -> NDArray:
        """Apply the transpose of the profiled-intercept design."""
        return dm.rmatvec(values) - mean_x * float(np.sum(values, dtype=np.float64))

    def centered_signed_gram(row_weights: NDArray) -> NDArray:
        """Return ``X_c' diag(row_weights) X_c`` for fixed ``mean_x``."""
        if stable_gram_rhs is not None:
            result, _ = centered_gram_rhs(
                dm=dm,
                W=row_weights,
                mean_x=mean_x,
                z_centered=stable_gram_rhs,
            )
            return result
        moments = dm.execution_plan.moments(
            row_weights,
            include_xtw=True,
            signed=True,
        )
        if moments.xtw is None:
            raise RuntimeError("centered signed moments omitted X'W1")
        sum_weights = float(np.sum(row_weights, dtype=np.float64))
        result = moments.gram
        result = result - np.outer(moments.xtw, mean_x)
        result = result - np.outer(mean_x, moments.xtw)
        result = result + sum_weights * np.outer(mean_x, mean_x)
        return 0.5 * (result + result.T)

    # Pre-compute d2W/deta2 for second-order path
    d2W_deta2: NDArray | None = None
    if w_correction_order >= 2:
        if geometry is not None:
            d2W_deta2 = geometry.weight_second_derivative
            if d2W_deta2 is None:
                raise ValueError("observed REML geometry omitted second weight derivatives")
        else:
            d2W_deta2 = compute_d2W_deta2(link, distribution, mu, eta, sample_weight)

    # Store per-group quantities for second-order cross-terms
    deta_vectors: list[NDArray] = []
    dbeta_vectors: list[NDArray] = []
    dmean_vectors: list[NDArray] = []
    dsum_w_values: list[float] = []
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

        # The intercept is unpenalized and profiled out, so
        # dalpha/drho_j = -mean_x' dbeta/drho_j and hence
        # deta/drho_j = (X - 1 mean_x') dbeta/drho_j.
        deta_j = centered_matvec(dbeta_j)

        # a_j = (dW/deta) * deta_j  -- weight changes per obs
        a_j = dW_deta * deta_j

        # dm/drho_j = X_c' (dw/drho_j) / sum(W).  The first derivative
        # of the centered Gram does not need dm/drho because X_c'W1=0,
        # but both mean-derivative outer products enter at second order.
        dmean_j = centered_rmatvec(a_j) / sum_w if sum_w is not None else np.zeros_like(mean_x)

        # C_j = X_c'diag(a_j)X_c -- dW contribution to the
        # profiled-intercept Hessian.
        C_j = centered_signed_gram(a_j)

        # Profiled-determinant gradient: centered-Hessian trace plus the
        # scalar 0.5 * log(sum(W)) derivative below.
        grad_correction[i] = 0.5 * float(np.sum(XtWX_S_inv * C_j))
        dsum_w_j = float(np.sum(a_j, dtype=np.float64))
        if sum_w is not None:
            # The fitted determinant is log(sum(W)) + log|H_c| after
            # profiling the intercept, so its scalar factor varies with W.
            grad_correction[i] += 0.5 * dsum_w_j / sum_w

        dH_extra[i] = C_j

        if w_correction_order >= 2:
            deta_vectors.append(deta_j)
            dbeta_vectors.append(dbeta_j)
            dmean_vectors.append(dmean_j)
            dsum_w_values.append(dsum_w_j)
            omega_ssp_list.append(omega_ssp)
            lam_list.append(lam)

    # -- Second-order Hessian cross-terms (Wood 2011, Section 3.5.1) --
    #
    # Cost: m^2/2 signed centered-Gram operations plus m^2/2 rmatvec +
    # matvec calls.  Well-scaled designs keep the execution-plan kernels;
    # only unsafe location/scale geometry uses the stable chunked primitive.
    dH2_cross: NDArray | None = None
    if w_correction_order >= 2 and d2W_deta2 is not None:
        dH2_cross = np.zeros((m, m))
        for i in range(m):
            pc_i = penalties[i]
            for j in range(i, m):
                pc_j = penalties[j]

                # Differentiating H dbeta_i = -S_i beta contributes
                # X_c'[(dW/deta) deta_j deta_i] through dH_j dbeta_i.
                f_jk = deta_vectors[i] * deta_vectors[j] * dW_deta

                # X_c^T f^{jk}
                Xt_f = centered_rmatvec(f_jk)

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

                # rhs = X_c^T f^{jk} + lam_i S_i dbeta_j + lam_j S_j dbeta_i
                rhs = Xt_f + lam_i_S_i_dbeta_j + lam_j_S_j_dbeta_i

                # d2beta_hat/(drho_i drho_j) = delta_ij * dbeta_hat/drho_j - H^{-1} rhs
                d2beta_ij = -(XtWX_S_inv @ rhs)
                if i == j:
                    d2beta_ij += dbeta_vectors[j]

                # Differentiating eta_i = X_c dbeta_i also differentiates
                # the weighted center: d2eta_ij = X_c d2beta_ij - dm_j' dbeta_i.
                d2eta_ij = centered_matvec(d2beta_ij) - float(dmean_vectors[j] @ dbeta_vectors[i])

                # Full d2w/(drho_i drho_j) (Section 3.5.1 T_{jk} derivation):
                d2w_drho_ij = d2W_deta2 * deta_vectors[i] * deta_vectors[j] + dW_deta * d2eta_ij

                # Differentiate X_c' diag(dw_i) X_c: the centered d2w Gram
                # plus the two negative weighted-mean outer products.
                C_ij = centered_signed_gram(d2w_drho_ij)
                if sum_w is not None:
                    C_ij -= sum_w * (
                        np.outer(dmean_vectors[i], dmean_vectors[j])
                        + np.outer(dmean_vectors[j], dmean_vectors[i])
                    )
                val = 0.5 * float(np.sum(XtWX_S_inv * C_ij))
                if sum_w is not None:
                    d2sum_w_ij = float(np.sum(d2w_drho_ij, dtype=np.float64))
                    val += 0.5 * (
                        d2sum_w_ij / sum_w - dsum_w_values[i] * dsum_w_values[j] / sum_w**2
                    )
                dH2_cross[i, j] = val
                dH2_cross[j, i] = val

    if w_correction_order >= 2:
        return grad_correction, dH_extra, dH2_cross
    return grad_correction, dH_extra


def validate_w_correction_order(value: object) -> int:
    """Return a supported Wood derivative order or fail before fitting."""
    if type(value) is not int or value not in (1, 2):
        raise ValueError("w_correction_order must be 1 or 2")
    return value
