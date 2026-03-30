"""REML optimizer internals.

Contains the private REML helper functions extracted from SuperGLM.model:
gradient, Hessian, W(ρ) correction, Laplace objective, and the three
optimizer kernels (exact Newton, cached-W discrete, BCD fixed-point).

All functions take explicit state (design matrix, distribution, link,
groups) rather than accessing ``self``, making them independently testable
and keeping model.py focused on orchestration.

References
----------
- Wood (2011): Fast stable restricted maximum likelihood and marginal
  likelihood estimation of semiparametric generalized linear models.
  JRSS-B 73(1), 3-36.
- Wood (2017): Generalized Additive Models, 2nd ed., Ch 6.2.
"""

from __future__ import annotations

import time as _time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm.distributions import _VARIANCE_FLOOR, clip_mu
from superglm.group_matrix import (
    DesignMatrix,
    _block_xtwx,
    _block_xtwx_signed,
)
from superglm.links import stabilize_eta
from superglm.reml import (
    REMLResult,
    _map_beta_between_bases,
    build_penalty_caches,
    cached_logdet_s_plus,
)
from superglm.solvers.irls_direct import (
    _build_penalty_matrix,
    _invert_xtwx_plus_penalty,
    _safe_decompose_H,
    fit_irls_direct,
)
from superglm.solvers.pirls import PIRLSResult, fit_pirls
from superglm.types import GroupSlice

# ═══════════════════════════════════════════════════════════════════
# Weight derivative
# ═══════════════════════════════════════════════════════════════════


def compute_dW_deta(
    link: Any,
    distribution: Any,
    mu: NDArray,
    eta: NDArray,
    sample_weight: NDArray,
) -> NDArray | None:
    """Derivative of IRLS weights w.r.t. the linear predictor.

    Wood (2011) Appendix D: dW/dη needed for the implicit-differentiation
    chain in the W(ρ) correction (Section 3.4).

    W_i = exposure_i · (dμ/dη)² / V(μ)

    dW_i/dη = exposure_i · (dμ/dη / V(μ)) · [2(d²μ/dη²) − (dμ/dη)² V'(μ)/V(μ)]

    For log link: dW/dη = W·(2 − μV'(μ)/V(μ)).
    Poisson/log: dW/dη = W. Gamma/log: dW/dη = 0 identically.

    Returns None if the link or distribution does not provide the
    required second-order methods (deriv2_inverse, variance_derivative),
    which skips the W(ρ) correction for custom objects.
    """
    if not hasattr(link, "deriv2_inverse") or not hasattr(distribution, "variance_derivative"):
        return None
    g1 = link.deriv_inverse(eta)  # dμ/dη
    g2 = link.deriv2_inverse(eta)  # d²μ/dη²
    V = np.maximum(distribution.variance(mu), _VARIANCE_FLOOR)
    Vp = distribution.variance_derivative(mu)
    return sample_weight * (g1 / V) * (2.0 * g2 - g1**2 * Vp / V)


# ═══════════════════════════════════════════════════════════════════
# W(ρ) correction
# ═══════════════════════════════════════════════════════════════════


def _compute_d2W_deta2(
    link: Any,
    distribution: Any,
    eta: NDArray,
    sample_weight: NDArray,
) -> NDArray | None:
    """Second derivative of IRLS weights w.r.t. the linear predictor.

    Computed via central finite differences of ``compute_dW_deta``.
    This avoids requiring third-order link derivatives (d³μ/dη³)
    which most link objects do not provide.

    Returns None when ``compute_dW_deta`` itself returns None (missing
    second-order methods on the link or distribution).
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
    reml_groups: list[tuple[int, GroupSlice]],
    penalty_caches: dict | None,
    sample_weight: NDArray,
    offset_arr: NDArray,
    distribution: Any,
    w_correction_order: int = 1,
) -> tuple[NDArray, dict[int, NDArray]] | tuple[NDArray, dict[int, NDArray], NDArray | None] | None:
    """W(ρ) correction for REML derivatives (first- or second-order).

    Wood (2011) Section 3.4 / Appendix C: implicit differentiation of β̂(ρ)
    through W(η(ρ)) using the chain dβ̂/dρ = −H⁻¹ S_j β̂ (IFT on the
    PIRLS stationarity condition).

    Computes the contribution from d(X'WX)/dρ_j = X'diag(dW/dρ_j)X
    which the fixed-W Laplace approximation drops.  The gradient
    correction is exact to first order; the Hessian C_j matrices are
    first-order (d²W/dρ² terms are dropped) unless ``w_correction_order=2``.

    When ``w_correction_order=2``, the second-order cross-term Hessian
    correction from Appendix C is also computed::

        dH2_cross[j,k] = 0.5 * tr(H⁻¹ X' diag(d²W/dη² · dη_j · dη_k) X)

    This accounts for the curvature of W w.r.t. ρ through the product
    of per-group linear predictor perturbations.

    Parameters
    ----------
    w_correction_order : int, default 1
        1 = first-order only (current default, backward compatible).
        2 = include second-order Hessian cross-terms from d²W/dη².

    Returns ``(grad_correction, dH_extra)`` when ``w_correction_order=1``
    (backward compatible 2-tuple), or
    ``(grad_correction, dH_extra, dH2_cross)`` when
    ``w_correction_order=2`` (3-tuple).  Returns None if the correction
    vanishes (e.g. Gamma with log link where dW/dη = 0 identically) or
    if the link/distribution does not provide the required methods.

    dH2_cross is an (m, m) array of second-order Hessian corrections.
    """
    eta = stabilize_eta(
        dm.matvec(pirls_result.beta) + pirls_result.intercept + offset_arr,
        link,
    )
    mu = clip_mu(link.inverse(eta), distribution)
    dW_deta = compute_dW_deta(link, distribution, mu, eta, sample_weight)

    if dW_deta is None:
        return None  # Custom link/distribution without second-order methods

    if np.max(np.abs(dW_deta)) < 1e-12:
        return None  # No correction (e.g. Gamma/log)

    p = XtWX_S_inv.shape[0]
    m = len(reml_groups)
    grad_correction = np.zeros(m)
    dH_extra: dict[int, NDArray] = {}

    gms = dm.group_matrices

    # Pre-compute d²W/dη² for second-order path
    d2W_deta2: NDArray | None = None
    if w_correction_order >= 2:
        d2W_deta2 = _compute_d2W_deta2(link, distribution, eta, sample_weight)

    # Store deta_j vectors for second-order cross-terms
    deta_vectors: list[NDArray] = []

    for i, (idx, g) in enumerate(reml_groups):
        if penalty_caches is not None:
            omega_ssp = penalty_caches[g.name].omega_ssp
        else:
            gm = gms[idx]
            omega_ssp = gm.R_inv.T @ gm.omega @ gm.R_inv
        lam = lambdas[g.name]
        beta_g = pirls_result.beta[g.sl]

        # S_j β̂ (p-vector, nonzero only in g.sl block)
        s_beta = np.zeros(p)
        s_beta[g.sl] = lam * (omega_ssp @ beta_g)

        # dβ̂/dρ_j = -H⁻¹ S_j β̂  (IFT)
        dbeta_j = -(XtWX_S_inv @ s_beta)

        # dη/dρ_j = X dβ̂/dρ_j
        deta_j = dm.matvec(dbeta_j)

        # a_j = (dW/dη) ⊙ dη_j  — weights change per observation
        a_j = dW_deta * deta_j

        # C_j = X'diag(a_j)X — the dW contribution to dH/dρ_j
        C_j = _block_xtwx_signed(gms, groups, a_j, tabmat_split=dm.tabmat_split)

        # Gradient correction: ½ tr(H⁻¹ C_j)
        grad_correction[i] = 0.5 * float(np.sum(XtWX_S_inv * C_j))

        dH_extra[i] = C_j

        if w_correction_order >= 2:
            deta_vectors.append(deta_j)

    # Second-order Hessian cross-terms
    dH2_cross: NDArray | None = None
    if w_correction_order >= 2 and d2W_deta2 is not None:
        dH2_cross = np.zeros((m, m))
        for i in range(m):
            for j in range(i, m):
                # d²W/dη² · dη_i · dη_j  (elementwise product)
                a_ij = d2W_deta2 * deta_vectors[i] * deta_vectors[j]
                C_ij = _block_xtwx_signed(gms, groups, a_ij, tabmat_split=dm.tabmat_split)
                # 0.5 * tr(H⁻¹ C_ij)
                val = 0.5 * float(np.sum(XtWX_S_inv * C_ij))
                dH2_cross[i, j] = val
                dH2_cross[j, i] = val

    if w_correction_order >= 2:
        return grad_correction, dH_extra, dH2_cross
    return grad_correction, dH_extra


# ═══════════════════════════════════════════════════════════════════
# REML objective
# ═══════════════════════════════════════════════════════════════════


def reml_laml_objective(
    dm: DesignMatrix,
    distribution: Any,
    link: Any,
    groups: list[GroupSlice],
    y: NDArray,
    result: PIRLSResult,
    lambdas: dict[str, float],
    sample_weight: NDArray,
    offset_arr: NDArray,
    XtWX: NDArray | None = None,
    penalty_caches: dict | None = None,
) -> float:
    """Laplace REML/LAML objective up to additive constants.

    Wood (2011) Section 2, Eqs (4)-(5): V(ρ) = -ℓ(β̂) + ½β̂'Sβ̂ +
    ½log|H| - ½log|S|₊. Known-scale: nll + ½(penalty_quad + logdet_m -
    logdet_s). Estimated-scale: φ profiled out → ½(n-Mp)·log(D+β̂'Sβ̂)
    replaces the nll + ½ penalty_quad terms.

    Handles both known-scale families (Poisson, NB2 where φ=1) and
    estimated-scale families (Gamma, Tweedie) via φ-profiled REML.
    """
    eta = stabilize_eta(dm.matvec(result.beta) + result.intercept + offset_arr, link)
    mu = clip_mu(link.inverse(eta), distribution)
    if XtWX is None:
        V = distribution.variance(mu)
        dmu_deta = link.deriv_inverse(eta)
        W = sample_weight * dmu_deta**2 / np.maximum(V, _VARIANCE_FLOOR)
        XtWX = _block_xtwx(dm.group_matrices, groups, W, tabmat_split=dm.tabmat_split)

    p = XtWX.shape[0]
    S = _build_penalty_matrix(dm.group_matrices, groups, lambdas, p)
    penalty_quad = float(result.beta @ S @ result.beta)

    # log|S|₊
    if penalty_caches is not None:
        logdet_s = cached_logdet_s_plus(lambdas, penalty_caches)
    else:
        eigvals_s = np.linalg.eigvalsh(S)
        thresh_s = 1e-10 * max(eigvals_s.max(), 1e-12)
        pos_s = eigvals_s[eigvals_s > thresh_s]
        logdet_s = float(np.sum(np.log(pos_s))) if pos_s.size else 0.0

    # log|H| = log|X'WX + S|
    M = XtWX + S
    eigvals_m = np.linalg.eigvalsh(M)
    thresh_m = 1e-10 * max(eigvals_m.max(), 1e-12)
    pos_m = eigvals_m[eigvals_m > thresh_m]
    logdet_m = float(np.sum(np.log(pos_m))) if pos_m.size else 0.0

    # φ-profiled REML for estimated-scale families
    scale_known = getattr(distribution, "scale_known", True)
    if not scale_known:
        n = len(y)
        if penalty_caches is not None:
            M_p = sum(c.rank for c in penalty_caches.values())
        else:
            M_p = float(len(pos_s))
        d_plus_pq = max(result.deviance + penalty_quad, 1e-300)
        scale_term = 0.5 * max(n - M_p, 1.0) * np.log(d_plus_pq)
        return float(0.5 * (logdet_m - logdet_s) + scale_term)

    nll = -distribution.log_likelihood(y, mu, sample_weight, phi=1.0)
    return float(nll + 0.5 * (penalty_quad + logdet_m - logdet_s))


# ═══════════════════════════════════════════════════════════════════
# REML gradient and Hessian
# ═══════════════════════════════════════════════════════════════════


def reml_direct_gradient(
    group_matrices: list,
    result: PIRLSResult,
    XtWX_S_inv: NDArray,
    lambdas: dict[str, float],
    reml_groups: list[tuple[int, GroupSlice]],
    penalty_ranks: dict[str, float],
    phi_hat: float = 1.0,
) -> NDArray:
    """Partial gradient of the LAML objective w.r.t. log-lambdas (fixed W).

    Wood (2011) Section 3.4: ∂V/∂ρ_j at fixed W. Chain rule ∂/∂ρ = λ ∂/∂λ
    converts from λ to log-λ (ρ) parameterization, giving the λ factor.
    g_j = ½(λ_j(β̂'Ω_jβ̂/φ + tr(H⁻¹ λ_jΩ_j)) - r_j).
    """
    grad = np.zeros(len(reml_groups), dtype=np.float64)
    inv_phi = 1.0 / max(phi_hat, 1e-10)
    for i, (idx, g) in enumerate(reml_groups):
        gm = group_matrices[idx]
        omega_ssp = gm.R_inv.T @ gm.omega @ gm.R_inv
        beta_g = result.beta[g.sl]
        quad = float(beta_g @ omega_ssp @ beta_g)
        H_inv_jj = XtWX_S_inv[g.sl, g.sl]
        trace_term = float(np.trace(H_inv_jj @ omega_ssp))
        lam = float(lambdas[g.name])
        grad[i] = 0.5 * (lam * (inv_phi * quad + trace_term) - penalty_ranks[g.name])
    return grad


def reml_direct_hessian(
    group_matrices: list,
    distribution: Any,
    XtWX_S_inv: NDArray,
    lambdas: dict[str, float],
    reml_groups: list[tuple[int, GroupSlice]],
    gradient: NDArray,
    penalty_ranks: dict[str, float],
    penalty_caches: dict | None = None,
    pirls_result: object | None = None,
    n_obs: int = 0,
    phi_hat: float = 1.0,
    dH_extra: dict[int, NDArray] | None = None,
    dH2_cross: NDArray | None = None,
) -> NDArray:
    """Outer Hessian of the REML criterion w.r.t. log-lambdas.

    Wood (2011) Appendix B / Eq 6.2. Uses implicit-differentiation
    Jacobian (outer products of H⁻¹ dH_j) rather than explicit K/T
    matrices from Appendix B — both equivalent. Terms:
      -0.5 tr(F_i F_j^T)    — direct differentiation of log|H|
      -S_beta^T H⁻¹ S_beta  — implicit differentiation of β̂'Sβ̂/φ
      -outer(q,q)/(D+pq)²   — profiled-scale correction (estimated-φ)
      g_i + 0.5·r_j on diag — ρ-chain-rule diagonal correction (Eq 6.2)

    Parameters
    ----------
    dH2_cross : NDArray or None
        Second-order W(ρ) Hessian correction from ``reml_w_correction``
        with ``w_correction_order=2``.  An (m, m) matrix of
        0.5 * tr(H⁻¹ X'diag(d²W/dη² · dη_i · dη_j)X) values,
        added directly to the Hessian.
    """
    m = len(reml_groups)
    p = XtWX_S_inv.shape[0]
    hess = np.zeros((m, m))

    full_HdHj: dict[int, NDArray] = {}
    quad_per_group: list[float] = []
    s_beta_list: list[NDArray] = []
    for i, (idx, g) in enumerate(reml_groups):
        if penalty_caches is not None:
            omega_ssp = penalty_caches[g.name].omega_ssp
        else:
            gm = group_matrices[idx]
            omega_ssp = gm.R_inv.T @ gm.omega @ gm.R_inv
        lam = lambdas[g.name]
        F = np.zeros((p, p))
        F[:, g.sl] = XtWX_S_inv[:, g.sl] @ (lam * omega_ssp)

        if dH_extra is not None and i in dH_extra:
            F = F + XtWX_S_inv @ dH_extra[i]

        full_HdHj[i] = F

        if pirls_result is not None:
            beta_g = pirls_result.beta[g.sl]
            quad_per_group.append(lam * float(beta_g @ omega_ssp @ beta_g))
            v = np.zeros(p)
            v[g.sl] = lam * (omega_ssp @ beta_g)
            s_beta_list.append(v)
        else:
            quad_per_group.append(0.0)
            s_beta_list.append(np.zeros(p))

    for i in range(m):
        for j in range(i, m):
            h = -0.5 * float(np.sum(full_HdHj[i] * full_HdHj[j].T))
            hess[i, j] = h
            hess[j, i] = h
        name_i = reml_groups[i][1].name
        hess[i, i] += gradient[i] + 0.5 * penalty_ranks[name_i]

    if pirls_result is not None:
        inv_phi = 1.0 / max(phi_hat, 1e-10)
        S_beta = np.column_stack(s_beta_list)
        HinvSbeta = XtWX_S_inv @ S_beta
        hess -= inv_phi * (S_beta.T @ HinvSbeta)

    scale_known = getattr(distribution, "scale_known", True)
    if not scale_known and pirls_result is not None and n_obs > 0:
        M_p = sum(penalty_ranks[g.name] for _, g in reml_groups)
        pq_total = sum(quad_per_group)
        d_plus_pq = max(pirls_result.deviance + pq_total, 1e-300)
        q = np.array(quad_per_group)
        hess -= 0.5 * max(n_obs - M_p, 1.0) * np.outer(q, q) / d_plus_pq**2

    # Second-order W(ρ) cross-term from d²W/dη²
    if dH2_cross is not None:
        hess += dH2_cross

    return hess


# ═══════════════════════════════════════════════════════════════════
# Direct REML Newton optimizer (exact path)
# ═══════════════════════════════════════════════════════════════════


def optimize_direct_reml(
    dm: DesignMatrix,
    distribution: Any,
    link: Any,
    groups: list[GroupSlice],
    discrete: bool,
    y: NDArray,
    sample_weight: NDArray,
    offset_arr: NDArray,
    reml_groups: list[tuple[int, GroupSlice]],
    penalty_ranks: dict[str, float],
    lambdas: dict[str, float],
    *,
    max_reml_iter: int,
    reml_tol: float,
    verbose: bool,
    penalty_caches: dict | None = None,
    profile: dict | None = None,
    max_analytical_per_w: int = 30,
    select_snap: bool = True,
    direct_solve: str = "auto",
    w_correction_order: int = 1,
) -> REMLResult:
    """Optimize the direct REML objective via damped Newton (Wood 2011).

    Two algorithm variants depending on ``discrete``:

    **Exact path** (``discrete=False``):
        W(ρ)-corrected direct REML with gradient, Hessian, line search.

    **Discrete path** (``discrete=True``):
        Cached-W fREML optimizer (fewer data passes), delegated to
        ``optimize_discrete_reml_cached_w``.
    """
    if discrete:
        return optimize_discrete_reml_cached_w(
            dm,
            distribution,
            link,
            groups,
            y,
            sample_weight,
            offset_arr,
            reml_groups,
            penalty_ranks,
            lambdas,
            max_reml_iter=max_reml_iter,
            reml_tol=reml_tol,
            verbose=verbose,
            penalty_caches=penalty_caches,
            profile=profile,
            max_analytical_per_w=max_analytical_per_w,
            select_snap=select_snap,
            direct_solve=direct_solve,
        )

    scale_known = getattr(distribution, "scale_known", True)
    group_names = [g.name for _, g in reml_groups]
    m = len(group_names)
    log_lo, log_hi = np.log(1e-6), np.log(1e10)
    max_newton_step = 5.0
    _eps = np.finfo(float).eps
    # Relative convergence tolerance.  reml_tol was originally an absolute
    # gradient threshold; as a relative tolerance scaled by score_scale,
    # 1e-6 is the tightest sensible value (Wood 2011 default).
    _tol = min(reml_tol, 1e-6)

    lambda_history: list[dict[str, float]] = [lambdas.copy()]
    warm_beta: NDArray | None = None
    warm_intercept: float | None = None

    best_obj = np.inf
    best_lambdas = lambdas.copy()
    best_pirls = None
    converged = False
    n_iter = 0

    _t_reml_start = _time.perf_counter()
    _t_pirls = 0.0
    _t_objective = 0.0
    _t_gradient = 0.0
    _t_hessian = 0.0
    _t_w_correction = 0.0
    _t_linesearch = 0.0
    _n_linesearch_fits = 0

    # === Bootstrap: one FP step from minimal penalty ===
    boot_lambdas = {name: 1e-4 for name in lambdas}
    _t0 = _time.perf_counter()
    boot_result, boot_inv, boot_xtwx = fit_irls_direct(
        X=dm,
        y=y,
        weights=sample_weight,
        family=distribution,
        link=link,
        groups=groups,
        lambda2=boot_lambdas,
        offset=offset_arr,
        return_xtwx=True,
        profile=profile,
        direct_solve=direct_solve,
    )
    _t_pirls += _time.perf_counter() - _t0
    warm_beta = boot_result.beta.copy()
    warm_intercept = float(boot_result.intercept)

    boot_phi = 1.0
    if not scale_known and penalty_caches is not None:
        p_dim = boot_xtwx.shape[0]
        S_boot = _build_penalty_matrix(dm.group_matrices, groups, boot_lambdas, p_dim)
        pq_boot = float(boot_result.beta @ S_boot @ boot_result.beta)
        M_p = sum(c.rank for c in penalty_caches.values())
        boot_phi = max((boot_result.deviance + pq_boot) / max(len(y) - M_p, 1.0), 1e-10)
    boot_inv_phi = 1.0 / max(boot_phi, 1e-10)

    rho = np.zeros(m, dtype=np.float64)
    for i, (idx, g) in enumerate(reml_groups):
        gm = dm.group_matrices[idx]
        if penalty_caches is not None:
            omega_ssp = penalty_caches[g.name].omega_ssp
        else:
            omega_ssp = gm.R_inv.T @ gm.omega @ gm.R_inv
        beta_g = boot_result.beta[g.sl]
        quad = float(beta_g @ omega_ssp @ beta_g)
        H_inv_jj = boot_inv[g.sl, g.sl]
        trace_term = float(np.trace(H_inv_jj @ omega_ssp))
        r_j = penalty_ranks[g.name]
        denom = boot_inv_phi * quad + trace_term
        lam_fp = r_j / denom if denom > 1e-12 else 1.0
        # Snap degenerate select=True groups to upper bound.
        # When quad << trace, the FP update is degenerate (any lambda
        # is approximately a fixed point).  Snap breaks the degeneracy.
        if (
            select_snap
            and g.subgroup_type is not None
            and trace_term > 1e-12
            and boot_inv_phi * quad < 0.1 * trace_term
        ):
            lam_fp = np.exp(log_hi)
        rho[i] = np.clip(np.log(max(lam_fp, 1e-6)), log_lo, log_hi)

    prev_obj = np.inf

    if verbose:
        boot_lam_str = ", ".join(
            f"{name}={np.exp(rho[i]):.4g}" for i, name in enumerate(group_names)
        )
        print(f"  REML bootstrap: lambdas=[{boot_lam_str}]")

    for outer in range(max_reml_iter):
        n_iter = outer + 1
        rho_clipped = np.clip(rho, log_lo, log_hi)

        cand_lambdas = lambdas.copy()
        for name, val in zip(group_names, np.exp(rho_clipped), strict=False):
            cand_lambdas[name] = float(np.clip(val, 1e-6, 1e10))

        _t0 = _time.perf_counter()
        pirls_result, XtWX_S_inv, XtWX = fit_irls_direct(
            X=dm,
            y=y,
            weights=sample_weight,
            family=distribution,
            link=link,
            groups=groups,
            lambda2=cand_lambdas,
            offset=offset_arr,
            beta_init=warm_beta,
            intercept_init=warm_intercept,
            return_xtwx=True,
            profile=profile,
            direct_solve=direct_solve,
        )
        _t_pirls += _time.perf_counter() - _t0
        warm_beta = pirls_result.beta.copy()
        warm_intercept = float(pirls_result.intercept)

        _t0 = _time.perf_counter()
        obj = reml_laml_objective(
            dm,
            distribution,
            link,
            groups,
            y,
            pirls_result,
            cand_lambdas,
            sample_weight,
            offset_arr,
            XtWX=XtWX,
            penalty_caches=penalty_caches,
        )

        phi_hat = 1.0
        if not scale_known and penalty_caches is not None:
            p_dim = XtWX.shape[0]
            S_eval = _build_penalty_matrix(dm.group_matrices, groups, cand_lambdas, p_dim)
            pq = float(pirls_result.beta @ S_eval @ pirls_result.beta)
            M_p = sum(c.rank for c in penalty_caches.values())
            phi_hat = max((pirls_result.deviance + pq) / max(len(y) - M_p, 1.0), 1e-10)
        _t_objective += _time.perf_counter() - _t0

        _t0 = _time.perf_counter()
        grad_partial = reml_direct_gradient(
            dm.group_matrices,
            pirls_result,
            XtWX_S_inv,
            cand_lambdas,
            reml_groups,
            penalty_ranks,
            phi_hat=phi_hat,
        )
        _t_gradient += _time.perf_counter() - _t0

        # W(ρ) correction
        _t0 = _time.perf_counter()
        if not discrete:
            w_corr = reml_w_correction(
                dm,
                link,
                groups,
                pirls_result,
                XtWX_S_inv,
                cand_lambdas,
                reml_groups,
                penalty_caches,
                sample_weight,
                offset_arr,
                distribution,
                w_correction_order=w_correction_order,
            )
        else:
            w_corr = None
        _t_w_correction += _time.perf_counter() - _t0
        if w_corr is not None:
            grad_w_correction = w_corr[0]
            dH_extra = w_corr[1]
            dH2_cross = w_corr[2] if len(w_corr) > 2 else None
            grad = grad_partial + grad_w_correction
        else:
            grad = grad_partial.copy()
            dH_extra = None
            dH2_cross = None

        if obj < best_obj:
            best_obj = obj
            best_lambdas = cand_lambdas.copy()
            best_pirls = pirls_result

        lambda_history.append(cand_lambdas.copy())

        proj_grad = grad.copy()
        for i in range(m):
            if rho_clipped[i] >= log_hi - 0.01 and grad[i] < 0:
                proj_grad[i] = 0.0
            elif rho_clipped[i] <= log_lo + 0.01 and grad[i] > 0:
                proj_grad[i] = 0.0
        proj_grad_norm = float(np.max(np.abs(proj_grad)))

        # Compound convergence criterion (Wood 2011, Section 6.2):
        # max(|g_j|) < eps * (1 + |V_r|), both gradient and objective change.
        score_scale = max(1.0 + abs(obj), 1.0)
        obj_change = abs(obj - prev_obj) if outer > 0 else np.inf

        if verbose:
            lam_str = ", ".join(f"{name}={cand_lambdas[name]:.4g}" for name in group_names)
            print(
                f"  REML Newton iter={n_iter}  obj={obj:.4f}  "
                f"|∇|={proj_grad_norm:.6f}  Δobj={obj_change:.6g}  "
                f"lambdas=[{lam_str}]"
            )

        prev_obj = obj

        # Require at least 2 iterations before checking convergence
        if outer >= 1:
            grad_converged = proj_grad_norm < _tol * score_scale
            obj_converged = obj_change < _tol * score_scale
            if grad_converged and obj_converged:
                converged = True
                break

        # Newton with exact outer Hessian
        # Wood (2011) eq 6.2: diagonal correction H[i,i] += g_i + 0.5*r_j
        # must use the *total* gradient (partial + W(rho) correction), not
        # the fixed-W partial gradient alone.
        _t0 = _time.perf_counter()
        hess = reml_direct_hessian(
            dm.group_matrices,
            distribution,
            XtWX_S_inv,
            cand_lambdas,
            reml_groups,
            grad,
            penalty_ranks,
            penalty_caches=penalty_caches,
            pirls_result=pirls_result,
            n_obs=len(y),
            phi_hat=phi_hat,
            dH_extra=dH_extra,
            dH2_cross=dH2_cross,
        )

        # Active-set: freeze components with negligible gradient and Hessian
        freeze_tol = 0.1 * _tol
        frozen = np.zeros(m, dtype=bool)
        for i in range(m):
            if (
                abs(proj_grad[i]) < freeze_tol * score_scale
                and abs(hess[i, i]) < freeze_tol * score_scale
            ):
                frozen[i] = True
        active_idx = np.where(~frozen)[0]

        if active_idx.size == 0:
            # All components frozen — converged
            _t_hessian += _time.perf_counter() - _t0
            converged = True
            break

        # Modified Newton: eigendecompose, flip negatives, floor small eigenvalues
        if active_idx.size < m:
            hess_sub = hess[np.ix_(active_idx, active_idx)]
            grad_sub = grad[active_idx]
        else:
            hess_sub = hess
            grad_sub = grad

        eigvals_h, eigvecs_h = np.linalg.eigh(hess_sub)
        max_eig = max(abs(eigvals_h).max(), 1e-12)
        eig_floor = max_eig * _eps**0.7
        eigvals_pd = np.maximum(np.abs(eigvals_h), eig_floor)

        hess_pd = (eigvecs_h * eigvals_pd) @ eigvecs_h.T
        delta_sub = -np.linalg.solve(hess_pd, grad_sub)

        # Scatter back to full delta
        delta = np.zeros(m)
        delta[active_idx] = delta_sub

        # Proportional step cap: scale entire vector if any component > max_step
        max_delta = float(np.max(np.abs(delta)))
        if max_delta > max_newton_step:
            delta *= max_newton_step / max_delta
        _t_hessian += _time.perf_counter() - _t0

        # Step-halving line search with Armijo condition
        _t0 = _time.perf_counter()
        max_ls = 8
        step = 1.0
        armijo_c = 1e-4
        descent = float(grad @ delta)
        accepted = False
        for _ls in range(max_ls):
            rho_trial = np.clip(rho_clipped + step * delta, log_lo, log_hi)
            trial_lambdas = lambdas.copy()
            for name, val in zip(group_names, np.exp(rho_trial), strict=False):
                trial_lambdas[name] = float(np.clip(val, 1e-6, 1e10))

            _n_linesearch_fits += 1
            trial_result, trial_inv, trial_xtwx = fit_irls_direct(
                X=dm,
                y=y,
                weights=sample_weight,
                family=distribution,
                link=link,
                groups=groups,
                lambda2=trial_lambdas,
                offset=offset_arr,
                beta_init=warm_beta,
                intercept_init=warm_intercept,
                return_xtwx=True,
                profile=profile,
                direct_solve=direct_solve,
            )

            trial_obj = reml_laml_objective(
                dm,
                distribution,
                link,
                groups,
                y,
                trial_result,
                trial_lambdas,
                sample_weight,
                offset_arr,
                XtWX=trial_xtwx,
                penalty_caches=penalty_caches,
            )

            if trial_obj <= obj + armijo_c * step * descent:
                rho = rho_trial
                warm_beta = trial_result.beta.copy()
                warm_intercept = float(trial_result.intercept)
                accepted = True
                break
            step *= 0.5

        if not accepted:
            # Steepest descent fallback: unit-length in infinity norm
            grad_max = float(np.max(np.abs(grad)))
            if grad_max > 1e-12:
                rho = np.clip(
                    rho_clipped - grad / grad_max,
                    log_lo,
                    log_hi,
                )
            else:
                rho = rho_clipped
        _t_linesearch += _time.perf_counter() - _t0

    if best_pirls is None:
        raise RuntimeError("Direct REML Newton did not evaluate any candidates")

    if profile is not None:
        profile["reml_optimizer_s"] = _time.perf_counter() - _t_reml_start
        profile["reml_pirls_s"] = _t_pirls
        profile["reml_objective_s"] = _t_objective
        profile["reml_gradient_s"] = _t_gradient
        profile["reml_w_correction_s"] = _t_w_correction
        profile["reml_hessian_newton_s"] = _t_hessian
        profile["reml_linesearch_s"] = _t_linesearch
        profile["reml_n_linesearch_fits"] = _n_linesearch_fits
        profile["reml_n_outer_iter"] = n_iter

    return REMLResult(
        lambdas=best_lambdas,
        pirls_result=best_pirls,
        n_reml_iter=n_iter,
        converged=converged,
        lambda_history=lambda_history,
        objective=float(best_obj),
    )


# ═══════════════════════════════════════════════════════════════════
# Cached-W fREML optimizer (discrete path)
# ═══════════════════════════════════════════════════════════════════


def _solve_cached_augmented(
    XtWX: NDArray,
    S: NDArray,
    XtWz: NDArray,
    XtW1: NDArray,
    sum_W: float,
    sum_Wz: float,
) -> tuple[NDArray, float]:
    """Solve the augmented weighted LS system from cached gram quantities.

    Returns (beta, intercept) without any data passes — just O(p³) Cholesky.
    """
    import scipy.linalg

    p = XtWX.shape[0]
    M_aug = np.empty((p + 1, p + 1))
    M_aug[0, 0] = sum_W
    M_aug[0, 1:] = XtW1
    M_aug[1:, 0] = XtW1
    M_aug[1:, 1:] = XtWX + S

    rhs = np.empty(p + 1)
    rhs[0] = sum_Wz
    rhs[1:] = XtWz

    try:
        L = scipy.linalg.cholesky(M_aug, lower=True, check_finite=False)
        beta_aug = scipy.linalg.cho_solve((L, True), rhs)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(M_aug)
        threshold = 1e-10 * max(eigvals.max(), 1e-12)
        with np.errstate(divide="ignore"):
            inv_eig = np.where(eigvals > threshold, 1.0 / eigvals, 0.0)
        beta_aug = (eigvecs * inv_eig[None, :]) @ eigvecs.T @ rhs

    return beta_aug[1:], float(beta_aug[0])


def optimize_discrete_reml_cached_w(
    dm: DesignMatrix,
    distribution: Any,
    link: Any,
    groups: list[GroupSlice],
    y: NDArray,
    sample_weight: NDArray,
    offset_arr: NDArray,
    reml_groups: list[tuple[int, GroupSlice]],
    penalty_ranks: dict[str, float],
    lambdas: dict[str, float],
    *,
    max_reml_iter: int,
    reml_tol: float,
    verbose: bool,
    penalty_caches: dict | None = None,
    profile: dict | None = None,
    direct_solve: str = "auto",
    # Legacy kwargs accepted but ignored (removed in POI rewrite)
    max_analytical_per_w: int = 30,
    select_snap: bool = True,
) -> REMLResult:
    """POI fREML optimizer for the discrete path.

    Performance Oriented Iteration (mgcv bam-style): interleaves one
    PIRLS step (W update) with one Newton lambda step on the working
    model's REML criterion.  Line search re-solves the cached augmented
    system analytically (O(p³), no data pass) for each trial lambda.

    Typically converges in 5-15 total iterations instead of the old
    nested architecture's 200+ analytical iterations.

    Note: this is a faster approximate optimizer.  On models with many
    noise features (p >> n_signal), Newton-POI may converge to a
    slightly different REML stationary point than the old Fellner-Schall
    fixed-point path.  The REML surface is flat in noise-feature
    directions, and Newton settles at a nearby minimum where noise
    lambdas are large but not maximally penalized.  Deviance drift is
    typically <0.1% relative (guarded by test_wide_poisson_poi_quality).
    """
    scale_known = getattr(distribution, "scale_known", True)
    group_names = [g.name for _, g in reml_groups]
    m = len(group_names)
    log_lo, log_hi = np.log(1e-6), np.log(1e10)
    p = dm.p

    lambda_history: list[dict[str, float]] = [lambdas.copy()]
    warm_beta: NDArray | None = None
    warm_intercept: float | None = None
    max_newton_step = 5.0
    max_halving = 25
    _eps = np.finfo(float).eps
    _tol = min(reml_tol, 1e-6)

    best_obj = np.inf
    best_lambdas = lambdas.copy()
    best_pirls = None
    converged = False

    _t_reml_start = _time.perf_counter()
    _t_pirls = 0.0
    _t_objective = 0.0
    _t_newton = 0.0
    _t_linesearch = 0.0
    _n_pirls_steps = 0
    _n_newton_steps = 0
    _n_linesearch_evals = 0

    # === Bootstrap: one FP step from minimal penalty ===
    boot_lambdas = {name: 1e-4 for name in lambdas}
    _t0 = _time.perf_counter()
    cache: dict = {}
    boot_result, boot_inv, boot_xtwx = fit_irls_direct(
        X=dm,
        y=y,
        weights=sample_weight,
        family=distribution,
        link=link,
        groups=groups,
        lambda2=boot_lambdas,
        offset=offset_arr,
        return_xtwx=True,
        profile=profile,
        cache_out=cache,
        direct_solve=direct_solve,
    )
    _t_pirls += _time.perf_counter() - _t0
    _n_pirls_steps += boot_result.n_iter
    warm_beta = boot_result.beta.copy()
    warm_intercept = float(boot_result.intercept)

    # Bootstrap FP step for initial rho
    boot_phi = 1.0
    if not scale_known and penalty_caches is not None:
        S_boot = _build_penalty_matrix(dm.group_matrices, groups, boot_lambdas, p)
        pq_boot = float(boot_result.beta @ S_boot @ boot_result.beta)
        M_p = sum(c.rank for c in penalty_caches.values())
        boot_phi = max((boot_result.deviance + pq_boot) / max(len(y) - M_p, 1.0), 1e-10)
    boot_inv_phi = 1.0 / max(boot_phi, 1e-10)

    rho = np.zeros(m, dtype=np.float64)
    for i, (idx, g) in enumerate(reml_groups):
        if penalty_caches is not None:
            omega_ssp = penalty_caches[g.name].omega_ssp
        else:
            gm = dm.group_matrices[idx]
            omega_ssp = gm.R_inv.T @ gm.omega @ gm.R_inv
        beta_g = boot_result.beta[g.sl]
        quad = float(beta_g @ omega_ssp @ beta_g)
        H_inv_jj = boot_inv[g.sl, g.sl]
        trace_term = float(np.trace(H_inv_jj @ omega_ssp))
        r_j = penalty_ranks[g.name]
        denom = boot_inv_phi * quad + trace_term
        lam_fp = r_j / denom if denom > 1e-12 else 1.0
        # Snap degenerate select=True groups to upper bound.
        if (
            select_snap
            and g.subgroup_type is not None
            and trace_term > 1e-12
            and boot_inv_phi * quad < 0.1 * trace_term
        ):
            lam_fp = np.exp(log_hi)
        rho[i] = np.clip(np.log(max(lam_fp, 1e-6)), log_lo, log_hi)

    if verbose:
        boot_lam_str = ", ".join(
            f"{name}={np.exp(rho[i]):.4g}" for i, name in enumerate(group_names)
        )
        print(f"  REML bootstrap: lambdas=[{boot_lam_str}]")

    # === POI loop: one PIRLS step + one Newton lambda step ===
    prev_obj = np.inf
    for poi_iter in range(max_reml_iter):
        rho_clipped = np.clip(rho, log_lo, log_hi)
        cand_lambdas = lambdas.copy()
        for name, val in zip(group_names, np.exp(rho_clipped), strict=False):
            cand_lambdas[name] = float(np.clip(val, 1e-6, 1e10))

        # --- Step 1: One PIRLS step (W update) ---
        _t0 = _time.perf_counter()
        cache = {}
        pirls_result, XtWX_S_inv, XtWX = fit_irls_direct(
            X=dm,
            y=y,
            weights=sample_weight,
            family=distribution,
            link=link,
            groups=groups,
            lambda2=cand_lambdas,
            offset=offset_arr,
            beta_init=warm_beta,
            intercept_init=warm_intercept,
            max_iter=1,
            return_xtwx=True,
            profile=profile,
            cache_out=cache,
            direct_solve=direct_solve,
        )
        _t_pirls += _time.perf_counter() - _t0
        _n_pirls_steps += 1
        warm_beta = pirls_result.beta.copy()
        warm_intercept = float(pirls_result.intercept)

        c_XtWz = cache["XtWz"]
        c_XtW1 = cache["XtW1"]
        c_sum_W = cache["sum_W"]
        c_sum_Wz = cache["sum_Wz"]

        # Evaluate REML objective
        _t0 = _time.perf_counter()
        obj = reml_laml_objective(
            dm,
            distribution,
            link,
            groups,
            y,
            pirls_result,
            cand_lambdas,
            sample_weight,
            offset_arr,
            XtWX=XtWX,
            penalty_caches=penalty_caches,
        )

        phi_hat = 1.0
        if not scale_known and penalty_caches is not None:
            S_eval = _build_penalty_matrix(dm.group_matrices, groups, cand_lambdas, p)
            pq = float(pirls_result.beta @ S_eval @ pirls_result.beta)
            M_p = sum(c.rank for c in penalty_caches.values())
            phi_hat = max((pirls_result.deviance + pq) / max(len(y) - M_p, 1.0), 1e-10)
        _t_objective += _time.perf_counter() - _t0

        if obj < best_obj:
            best_obj = obj
            best_lambdas = cand_lambdas.copy()
            best_pirls = pirls_result
        lambda_history.append(cand_lambdas.copy())

        # --- Step 2: Newton step on lambda ---
        _t0 = _time.perf_counter()
        grad = reml_direct_gradient(
            dm.group_matrices,
            pirls_result,
            XtWX_S_inv,
            cand_lambdas,
            reml_groups,
            penalty_ranks,
            phi_hat=phi_hat,
        )
        hess = reml_direct_hessian(
            dm.group_matrices,
            distribution,
            XtWX_S_inv,
            cand_lambdas,
            reml_groups,
            grad,
            penalty_ranks,
            penalty_caches=penalty_caches,
            pirls_result=pirls_result,
            n_obs=len(y),
            phi_hat=phi_hat,
        )

        # Active-set: freeze components with negligible gradient and Hessian
        # Wood (2011) Section 6.2: score_scale = 1 + |V_r|
        score_scale_d = max(1.0 + abs(obj), 1.0)
        freeze_tol_d = 0.1 * _tol

        proj_grad_d = grad.copy()
        for i in range(m):
            if rho_clipped[i] >= log_hi - 0.01 and grad[i] < 0:
                proj_grad_d[i] = 0.0
            elif rho_clipped[i] <= log_lo + 0.01 and grad[i] > 0:
                proj_grad_d[i] = 0.0

        frozen_d = np.zeros(m, dtype=bool)
        for i in range(m):
            if (
                abs(proj_grad_d[i]) < freeze_tol_d * score_scale_d
                and abs(hess[i, i]) < freeze_tol_d * score_scale_d
            ):
                frozen_d[i] = True
        active_idx_d = np.where(~frozen_d)[0]

        # Modified Newton: eigendecompose, flip negatives, floor small eigenvalues
        if active_idx_d.size == 0:
            delta = np.zeros(m)
        else:
            if active_idx_d.size < m:
                hess_sub_d = hess[np.ix_(active_idx_d, active_idx_d)]
                grad_sub_d = grad[active_idx_d]
            else:
                hess_sub_d = hess
                grad_sub_d = grad

            eigvals_h, eigvecs_h = np.linalg.eigh(hess_sub_d)
            max_eig_d = max(abs(eigvals_h).max(), 1e-12)
            eig_floor_d = max_eig_d * _eps**0.7
            eigvals_pd = np.maximum(np.abs(eigvals_h), eig_floor_d)
            delta_sub_d = -(eigvecs_h * (1.0 / eigvals_pd)) @ (eigvecs_h.T @ grad_sub_d)
            delta = np.zeros(m)
            delta[active_idx_d] = delta_sub_d

        # Step capping
        max_delta = float(np.max(np.abs(delta)))
        if max_delta > max_newton_step:
            delta *= max_newton_step / max_delta
        _t_newton += _time.perf_counter() - _t0
        _n_newton_steps += 1

        # --- Step 3: Line search (step halving on working-model REML) ---
        _t0 = _time.perf_counter()
        accepted = False
        step = 1.0
        for _ls in range(max_halving):
            rho_trial = np.clip(rho + step * delta, log_lo, log_hi)
            trial_lambdas = lambdas.copy()
            for name, val in zip(group_names, np.exp(rho_trial), strict=False):
                trial_lambdas[name] = float(np.clip(val, 1e-6, 1e10))

            # Solve augmented system analytically (O(p³), no data pass)
            S_trial = _build_penalty_matrix(dm.group_matrices, groups, trial_lambdas, p)
            beta_trial, intercept_trial = _solve_cached_augmented(
                XtWX,
                S_trial,
                c_XtWz,
                c_XtW1,
                c_sum_W,
                c_sum_Wz,
            )

            # Evaluate REML at trial point.  Must compute deviance here
            # because reml_laml_objective reads result.deviance for
            # estimated-scale families (Gamma, Tweedie).
            eta_trial = stabilize_eta(dm.matvec(beta_trial) + intercept_trial + offset_arr, link)
            mu_trial = clip_mu(link.inverse(eta_trial), distribution)
            dev_trial = float(np.sum(sample_weight * distribution.deviance_unit(y, mu_trial)))
            trial_pirls = PIRLSResult(
                beta=beta_trial,
                intercept=intercept_trial,
                deviance=dev_trial,
                n_iter=0,
                converged=True,
                phi=phi_hat,
                effective_df=0.0,
            )
            trial_obj = reml_laml_objective(
                dm,
                distribution,
                link,
                groups,
                y,
                trial_pirls,
                trial_lambdas,
                sample_weight,
                offset_arr,
                XtWX=XtWX,
                penalty_caches=penalty_caches,
            )

            _n_linesearch_evals += 1
            if trial_obj < obj:
                rho = rho_trial
                warm_beta = beta_trial.copy()
                warm_intercept = intercept_trial
                accepted = True
                break

            step *= 0.5
        _t_linesearch += _time.perf_counter() - _t0

        if not accepted:
            # Steepest descent fallback: unit-length in infinity norm
            grad_max_d = float(np.max(np.abs(grad)))
            if grad_max_d > 1e-12:
                rho = np.clip(
                    rho - grad / grad_max_d,
                    log_lo,
                    log_hi,
                )
            # else: keep rho unchanged

        # Convergence check — compound criterion with score_scale
        proj_grad_norm = float(np.max(np.abs(proj_grad_d)))

        if verbose:
            lam_str = ", ".join(f"{name}={cand_lambdas[name]:.4g}" for name in group_names)
            obj_change_d = abs(obj - prev_obj) if poi_iter > 0 else np.inf
            print(
                f"  POI iter {poi_iter + 1}  obj={obj:.4f}  "
                f"|∇|={proj_grad_norm:.6f}  Δobj={obj_change_d:.6g}  [{lam_str}]"
            )

        obj_change = abs(obj - prev_obj) if poi_iter > 0 else np.inf
        prev_obj = obj
        if poi_iter >= 1:
            grad_converged_d = proj_grad_norm < _tol * score_scale_d
            obj_converged_d = obj_change < _tol * score_scale_d
            if grad_converged_d and obj_converged_d:
                converged = True
                break

    # === Final full IRLS refit at converged lambdas ===
    rho_clipped = np.clip(rho, log_lo, log_hi)
    final_lambdas = lambdas.copy()
    for name, val in zip(group_names, np.exp(rho_clipped), strict=False):
        final_lambdas[name] = float(np.clip(val, 1e-6, 1e10))
    _t0 = _time.perf_counter()
    final_result, final_inv, final_xtwx = fit_irls_direct(
        X=dm,
        y=y,
        weights=sample_weight,
        family=distribution,
        link=link,
        groups=groups,
        lambda2=final_lambdas,
        offset=offset_arr,
        beta_init=warm_beta,
        intercept_init=warm_intercept,
        return_xtwx=True,
        profile=profile,
        direct_solve=direct_solve,
    )
    _t_pirls += _time.perf_counter() - _t0
    _t0 = _time.perf_counter()
    final_obj = reml_laml_objective(
        dm,
        distribution,
        link,
        groups,
        y,
        final_result,
        final_lambdas,
        sample_weight,
        offset_arr,
        XtWX=final_xtwx,
        penalty_caches=penalty_caches,
    )
    _t_objective += _time.perf_counter() - _t0
    # Always use the final refit — it is the authoritative result from
    # full IRLS convergence at the converged lambdas.  The working-model
    # surrogates from the POI loop (n_iter=0) must not leak out.
    best_obj = final_obj
    best_lambdas = final_lambdas.copy()
    best_pirls = final_result
    lambda_history.append(final_lambdas.copy())

    if profile is not None:
        profile["reml_optimizer_s"] = _time.perf_counter() - _t_reml_start
        profile["reml_pirls_s"] = _t_pirls
        profile["reml_objective_s"] = _t_objective
        profile["reml_gradient_s"] = 0.0
        profile["reml_w_correction_s"] = 0.0
        profile["reml_hessian_newton_s"] = _t_newton
        profile["reml_linesearch_s"] = _t_linesearch
        profile["reml_fp_update_s"] = 0.0
        profile["reml_n_linesearch_fits"] = _n_linesearch_evals
        profile["reml_n_outer_iter"] = poi_iter + 1
        profile["reml_n_analytical_iters"] = _n_newton_steps

    return REMLResult(
        lambdas=best_lambdas,
        pirls_result=best_pirls,
        n_reml_iter=poi_iter + 1,
        converged=converged,
        lambda_history=lambda_history,
        objective=float(best_obj),
    )


# ═══════════════════════════════════════════════════════════════════
# EFS REML optimizer (Wood & Fasiolo 2017)
# ═══════════════════════════════════════════════════════════════════


def optimize_efs_reml(
    dm: DesignMatrix,
    distribution: Any,
    link: Any,
    groups: list[GroupSlice],
    penalty: Any,
    active_set: bool,
    y: NDArray,
    sample_weight: NDArray,
    offset_arr: NDArray,
    reml_groups: list[tuple[int, GroupSlice]],
    penalty_ranks: dict[str, float],
    lambdas: dict[str, float],
    *,
    max_reml_iter: int,
    reml_tol: float,
    verbose: bool,
    penalty_caches: dict | None = None,
    rebuild_dm: Any = None,
) -> tuple[REMLResult, DesignMatrix]:
    """EFS (generalized Fellner-Schall) REML optimizer for the BCD path.

    Implements Wood & Fasiolo (2017) fixed-point iteration with:
    - X'WX caching for O(p³) cheap iterations (no data pass)
    - Scalar group support (rank-1 penalties estimated, not skipped)
    - Two-tier iteration: DM rebuild + full PIRLS / cheap re-inversion
    - Anderson(1) acceleration on log-lambda scale

    Used when lambda1 > 0 (group lasso + REML smoothing).

    References
    ----------
    Wood & Fasiolo (2017). A generalized Fellner-Schall method for smoothing
    parameter optimization with application to shape constrained regression.
    Biometrics 73(4), 1071-1081.
    """
    scale_known = getattr(distribution, "scale_known", True)
    reml_update_names = [g.name for _, g in reml_groups]
    n = len(y)

    # ── Bootstrap: one PIRLS with minimal penalty → one EFS step ──
    # Analogous to direct REML bootstrap (optimize_direct_reml lines 417-467).
    # Gives data-informed initial lambdas regardless of lambda2_init.
    boot_lambdas = {name: 1e-4 for name in lambdas}
    boot_result = fit_pirls(
        X=dm,
        y=y,
        weights=sample_weight,
        family=distribution,
        link=link,
        groups=groups,
        penalty=penalty,
        offset=offset_arr,
        active_set=active_set,
        lambda2=boot_lambdas,
    )

    # Compute W and X'WX from bootstrap fit
    boot_eta = stabilize_eta(
        dm.matvec(boot_result.beta) + boot_result.intercept + offset_arr,
        link,
    )
    boot_mu = clip_mu(link.inverse(boot_eta), distribution)
    boot_V = distribution.variance(boot_mu)
    boot_dmu = link.deriv_inverse(boot_eta)
    boot_W = sample_weight * boot_dmu**2 / np.maximum(boot_V, _VARIANCE_FLOOR)
    boot_xtwx = _block_xtwx(dm.group_matrices, groups, boot_W, tabmat_split=dm.tabmat_split)

    # Estimate phi for estimated-scale families
    boot_inv_phi = 1.0
    if not scale_known and penalty_caches is not None:
        p_dim = boot_xtwx.shape[0]
        S_boot = _build_penalty_matrix(dm.group_matrices, groups, boot_lambdas, p_dim)
        pq_boot = float(boot_result.beta @ S_boot @ boot_result.beta)
        M_p = sum(c.rank for c in penalty_caches.values())
        boot_phi = max((boot_result.deviance + pq_boot) / max(n - M_p, 1.0), 1e-10)
        boot_inv_phi = 1.0 / boot_phi

    # One EFS fixed-point step on bootstrap beta
    S_boot = _build_penalty_matrix(dm.group_matrices, groups, boot_lambdas, dm.p)
    H_boot = boot_xtwx + S_boot
    H_boot_inv, _, _ = _safe_decompose_H(H_boot)

    for _idx, g in reml_groups:
        beta_g = boot_result.beta[g.sl]
        if np.linalg.norm(beta_g) < 1e-12:
            continue
        omega_ssp = penalty_caches[g.name].omega_ssp
        quad = float(beta_g @ omega_ssp @ beta_g)
        trace_term = float(np.trace(H_boot_inv[g.sl, g.sl] @ omega_ssp))
        r_j = penalty_ranks[g.name]
        denom = boot_inv_phi * quad + trace_term
        lam_fp = r_j / denom if denom > 1e-12 else 1.0
        lambdas[g.name] = float(np.clip(lam_fp, 1e-6, 1e10))

    # Rebuild DM with bootstrapped lambdas
    old_gms = dm.group_matrices
    dm = rebuild_dm(lambdas, sample_weight)
    penalty_caches = build_penalty_caches(dm.group_matrices, reml_groups)
    penalty_ranks = {n_: c.rank for n_, c in penalty_caches.items()}
    warm_beta = _map_beta_between_bases(boot_result.beta, old_gms, dm.group_matrices, groups)
    warm_intercept = float(boot_result.intercept)

    if verbose:
        lam_str = ", ".join(f"{g.name}={lambdas[g.name]:.4g}" for _, g in reml_groups)
        print(f"  REML bootstrap: lambdas=[{lam_str}]")

    # ── Main EFS loop ─────────────────────────────────────────────
    lambda_history: list[dict[str, float]] = [lambdas.copy()]
    converged = False
    n_reml_iter = 0
    cheap_iter = False
    cached_xtwx: NDArray | None = None
    last_pirls_iters = 0

    # Anderson(1) acceleration state
    aa_prev_log_x: NDArray | None = None
    aa_prev_log_gx: NDArray | None = None

    # Threshold for cheap iterations (re-invert cached X'WX only, no data pass)
    # R_inv depends on lambda, so DM must always be rebuilt when lambdas change
    # significantly — there is no valid "PIRLS without DM rebuild" tier.
    cheap_threshold = 0.01

    for reml_iter in range(max_reml_iter):
        n_reml_iter = reml_iter + 1

        # ── Tier 1 & 2: Full PIRLS solve ──────────────────────────
        if not cheap_iter:
            pirls_result = fit_pirls(
                X=dm,
                y=y,
                weights=sample_weight,
                family=distribution,
                link=link,
                groups=groups,
                penalty=penalty,
                offset=offset_arr,
                beta_init=warm_beta,
                intercept_init=warm_intercept,
                active_set=active_set,
                lambda2=lambdas,
            )
            beta = pirls_result.beta
            intercept = pirls_result.intercept
            last_pirls_iters = pirls_result.n_iter

            # Compute IRLS weights and cache X'WX
            eta = stabilize_eta(dm.matvec(beta) + intercept + offset_arr, link)
            mu = clip_mu(link.inverse(eta), distribution)
            V = distribution.variance(mu)
            dmu_deta = link.deriv_inverse(eta)
            W = sample_weight * dmu_deta**2 / np.maximum(V, _VARIANCE_FLOOR)

            cached_xtwx = _block_xtwx(dm.group_matrices, groups, W, tabmat_split=dm.tabmat_split)

        # ── Compute H⁻¹ = (X'WX + S)⁻¹ ──────────────────────────
        p = dm.p
        S = _build_penalty_matrix(dm.group_matrices, groups, lambdas, p)
        H = cached_xtwx + S
        H_inv, _, _ = _safe_decompose_H(H)

        # ── Estimate phi for estimated-scale families ─────────────
        inv_phi = 1.0
        if not scale_known and penalty_caches is not None:
            pq = float(beta @ S @ beta)
            M_p = sum(c.rank for c in penalty_caches.values())
            phi_hat = max((pirls_result.deviance + pq) / max(n - M_p, 1.0), 1e-10)
            inv_phi = 1.0 / phi_hat

        # ── EFS lambda update ─────────────────────────────────────
        lambdas_new = lambdas.copy()
        for _idx, g in reml_groups:
            beta_g = beta[g.sl]

            # Skip zeroed groups (L1 penalty killed them)
            if np.linalg.norm(beta_g) < 1e-12:
                continue

            omega_ssp = penalty_caches[g.name].omega_ssp
            quad = float(beta_g @ omega_ssp @ beta_g)
            trace_term = float(np.trace(H_inv[g.sl, g.sl] @ omega_ssp))

            r_j = penalty_ranks[g.name]
            denom = inv_phi * quad + trace_term

            if denom > 1e-12:
                lam_new = r_j / denom
            else:
                lam_new = lambdas[g.name]

            # Clamp log-lambda step to prevent wild jumps
            log_step = np.log(max(lam_new, 1e-10)) - np.log(max(lambdas[g.name], 1e-10))
            log_step = np.clip(log_step, -5.0, 5.0)
            lam_new = lambdas[g.name] * np.exp(log_step)

            lambdas_new[g.name] = float(np.clip(lam_new, 1e-6, 1e10))

        # ── Anderson(1) acceleration on log-lambda ────────────────
        if aa_prev_log_x is not None and len(reml_update_names) > 0:
            log_x = np.array([np.log(lambdas[n_]) for n_ in reml_update_names])
            log_gx = np.array([np.log(lambdas_new[n_]) for n_ in reml_update_names])
            f_curr = log_gx - log_x
            f_prev = aa_prev_log_gx - aa_prev_log_x
            df = f_curr - f_prev
            df_sq = float(np.dot(df, df))
            if df_sq > 1e-20:
                theta = float(-np.dot(f_curr, df) / df_sq)
                theta = max(-0.5, min(theta, 2.0))
                log_acc = (1.0 + theta) * log_gx - theta * aa_prev_log_gx
                for i, name in enumerate(reml_update_names):
                    lambdas_new[name] = float(np.clip(np.exp(log_acc[i]), 1e-6, 1e10))

        if len(reml_update_names) > 0:
            aa_prev_log_x = np.array([np.log(lambdas[n_]) for n_ in reml_update_names])
            aa_prev_log_gx = np.array([np.log(lambdas_new[n_]) for n_ in reml_update_names])

        # ── Stale-basis uphill-step guard (heuristic) ─────────────
        # EFS is not guaranteed to decrease the REML objective (Wood &
        # Fasiolo 2017).  We evaluate the objective at lambdas_new using
        # the *current* dm, pirls_result, and cached_xtwx — all of which
        # are stale w.r.t. the proposed lambdas.  After this check, the
        # DM/R_inv may be rebuilt (~line 1526), changing the true
        # objective surface.  So this guard can: (a) damp a step that
        # would actually improve the post-rebuild objective, or (b) miss
        # a step that goes uphill after rebuild.  It is a heuristic
        # safeguard against gross uphill moves, not a monotonicity fix.
        if cached_xtwx is not None:
            obj_curr = reml_laml_objective(
                dm,
                distribution,
                link,
                groups,
                y,
                pirls_result,
                lambdas,
                sample_weight,
                offset_arr,
                XtWX=cached_xtwx,
                penalty_caches=penalty_caches,
            )
            obj_trial = reml_laml_objective(
                dm,
                distribution,
                link,
                groups,
                y,
                pirls_result,
                lambdas_new,
                sample_weight,
                offset_arr,
                XtWX=cached_xtwx,
                penalty_caches=penalty_caches,
            )
            if obj_trial > obj_curr + 1e-8 * max(abs(obj_curr), 1.0):
                # Geometric mean interpolation in log-lambda space
                for _, g in reml_groups:
                    log_old = np.log(max(lambdas[g.name], 1e-10))
                    log_new = np.log(max(lambdas_new[g.name], 1e-10))
                    lambdas_new[g.name] = float(
                        np.clip(np.exp(0.5 * (log_old + log_new)), 1e-6, 1e10)
                    )
                # Update Anderson state for the dampened step
                if len(reml_update_names) > 0:
                    aa_prev_log_gx = np.array([np.log(lambdas_new[n_]) for n_ in reml_update_names])

        # ── Convergence check ─────────────────────────────────────
        changes = [
            abs(np.log(lambdas_new[g.name]) - np.log(lambdas[g.name]))
            for _, g in reml_groups
            if lambdas[g.name] > 0 and lambdas_new[g.name] > 0
        ]
        max_change = max(changes) if changes else 0.0

        if verbose:
            lam_str = ", ".join(f"{g.name}={lambdas_new[g.name]:.4g}" for _, g in reml_groups)
            mode = "cheap" if cheap_iter else f"pirls={last_pirls_iters}"
            print(
                f"  REML iter={n_reml_iter}  max_change={max_change:.6f}  "
                f"({mode})  lambdas=[{lam_str}]"
            )

        lambda_history.append(lambdas_new.copy())

        if max_change < reml_tol:
            converged = True
            lambdas = lambdas_new
            break

        # ── Decide next iteration tier ────────────────────────────
        if max_change > cheap_threshold:
            # Full: rebuild DM (R_inv depends on lambda) + PIRLS
            old_gms = dm.group_matrices
            dm = rebuild_dm(lambdas_new, sample_weight)
            warm_beta = _map_beta_between_bases(beta, old_gms, dm.group_matrices, groups)
            warm_intercept = intercept
            # R_inv changed → recompute penalty caches (omega_ssp etc.)
            penalty_caches = build_penalty_caches(dm.group_matrices, reml_groups)
            penalty_ranks = {n_: c.rank for n_, c in penalty_caches.items()}
            cheap_iter = False
        else:
            # Cheap: re-invert cached X'WX + S only (O(p³), no data pass)
            cheap_iter = True

        lambdas = lambdas_new

    # ── Final refit ───────────────────────────────────────────────
    if cheap_iter and converged:
        dm = rebuild_dm(lambdas, sample_weight)
        # R_inv changed → refresh caches for the objective computation
        penalty_caches = build_penalty_caches(dm.group_matrices, reml_groups)

    final_result = fit_pirls(
        X=dm,
        y=y,
        weights=sample_weight,
        family=distribution,
        link=link,
        groups=groups,
        penalty=penalty,
        offset=offset_arr,
        beta_init=warm_beta,
        intercept_init=warm_intercept,
        active_set=active_set,
        lambda2=lambdas,
    )

    reml_result = REMLResult(
        lambdas=lambdas,
        pirls_result=final_result,
        n_reml_iter=n_reml_iter,
        converged=converged,
        lambda_history=lambda_history,
        objective=reml_laml_objective(
            dm,
            distribution,
            link,
            groups,
            y,
            final_result,
            lambdas,
            sample_weight,
            offset_arr,
            penalty_caches=penalty_caches,
        ),
    )
    return reml_result, dm


# ═══════════════════════════════════════════════════════════════════
# BCD REML fixed-point optimizer (legacy)
# ═══════════════════════════════════════════════════════════════════


def run_reml_once(
    dm: DesignMatrix,
    distribution: Any,
    link: Any,
    groups: list[GroupSlice],
    penalty: Any,
    active_set: bool,
    y: NDArray,
    sample_weight: NDArray,
    offset_arr: NDArray,
    reml_groups: list[tuple[int, GroupSlice]],
    penalty_ranks: dict[str, float],
    lambdas: dict[str, float],
    *,
    max_reml_iter: int,
    reml_tol: float,
    verbose: bool,
    use_direct: bool,
    penalty_caches: dict | None = None,
    rebuild_dm: Any = None,
    direct_solve: str = "auto",
) -> tuple[REMLResult, DesignMatrix]:
    """Run a single REML fixed-point outer loop from a chosen initial lambda scale.

    Returns (REMLResult, DesignMatrix) — the dm may be updated on the BCD path.
    """
    from superglm.metrics import _penalised_xtwx_inv_gram

    scale_known = getattr(distribution, "scale_known", True)

    if use_direct:
        reml_update_names = [g.name for _, g in reml_groups]
    else:
        reml_update_names = [g.name for _, g in reml_groups if penalty_ranks[g.name] > 1]

    warm_beta = None
    warm_intercept = None
    lambda_history: list[dict[str, float]] = [lambdas.copy()]
    converged = False
    n_reml_iter = 0
    aa_prev_log_x: NDArray | None = None
    aa_prev_log_gx: NDArray | None = None
    cheap_iter = False
    cached_direct_xtwx: NDArray | None = None
    last_pirls_iters = 0
    direct_has_scalar_groups = any(penalty_ranks[g.name] <= 1 for _, g in reml_groups)
    direct_cheap_threshold = 0.01 if direct_has_scalar_groups else 0.2
    bcd_cheap_threshold = 0.01

    for reml_iter in range(max_reml_iter):
        n_reml_iter = reml_iter + 1

        if use_direct and not cheap_iter:
            pirls_result, XtWX_S_inv_full, XtWX_full = fit_irls_direct(
                X=dm,
                y=y,
                weights=sample_weight,
                family=distribution,
                link=link,
                groups=groups,
                lambda2=lambdas,
                offset=offset_arr,
                beta_init=warm_beta,
                intercept_init=warm_intercept,
                return_xtwx=True,
                direct_solve=direct_solve,
            )
            beta = pirls_result.beta
            intercept = pirls_result.intercept
            last_pirls_iters = pirls_result.n_iter
            cached_direct_xtwx = XtWX_full

            eta = stabilize_eta(dm.matvec(beta) + intercept + offset_arr, link)
            mu = clip_mu(link.inverse(eta), distribution)
            V = distribution.variance(mu)
            dmu_deta = link.deriv_inverse(eta)
            W = sample_weight * dmu_deta**2 / np.maximum(V, _VARIANCE_FLOOR)

            active_groups = list(groups)
            XtWX_S_inv = XtWX_S_inv_full
        elif not use_direct and not cheap_iter:
            pirls_result = fit_pirls(
                X=dm,
                y=y,
                weights=sample_weight,
                family=distribution,
                link=link,
                groups=groups,
                penalty=penalty,
                offset=offset_arr,
                beta_init=warm_beta,
                intercept_init=warm_intercept,
                active_set=active_set,
                lambda2=lambdas,
            )
            beta = pirls_result.beta
            intercept = pirls_result.intercept
            last_pirls_iters = pirls_result.n_iter

            eta = stabilize_eta(dm.matvec(beta) + intercept + offset_arr, link)
            mu = clip_mu(link.inverse(eta), distribution)
            V = distribution.variance(mu)
            dmu_deta = link.deriv_inverse(eta)
            W = sample_weight * dmu_deta**2 / np.maximum(V, _VARIANCE_FLOOR)

        if use_direct and cheap_iter:
            if cached_direct_xtwx is None:
                raise RuntimeError("REML cheap iteration missing cached direct XtWX")
            XtWX_S_inv = _invert_xtwx_plus_penalty(
                cached_direct_xtwx, dm.group_matrices, groups, lambdas
            )
            active_groups = list(groups)
        elif not use_direct:
            XtWX_S_inv, _, active_groups, _, _ = _penalised_xtwx_inv_gram(
                beta, W, dm.group_matrices, groups, lambdas
            )

        inv_phi = 1.0
        if not scale_known and penalty_caches is not None:
            p_dim = dm.p
            S_fp = _build_penalty_matrix(dm.group_matrices, groups, lambdas, p_dim)
            pq = float(beta @ S_fp @ beta)
            M_p = sum(c.rank for c in penalty_caches.values())
            phi_hat = max((pirls_result.deviance + pq) / max(len(y) - M_p, 1.0), 1e-10)
            inv_phi = 1.0 / phi_hat

        lambdas_new = lambdas.copy()
        for idx, g in reml_groups:
            if not use_direct and penalty_ranks[g.name] <= 1:
                continue

            gm = dm.group_matrices[idx]
            beta_g = beta[g.sl]
            if np.linalg.norm(beta_g) < 1e-12:
                continue

            omega_ssp = gm.R_inv.T @ gm.omega @ gm.R_inv
            quad = float(beta_g @ omega_ssp @ beta_g)

            ag = next((a for a in active_groups if a.name == g.name), None)
            if ag is None:
                continue

            H_inv_jj = XtWX_S_inv[ag.sl, ag.sl]
            trace_term = float(np.trace(H_inv_jj @ omega_ssp))

            r_j = penalty_ranks[g.name]
            denom = inv_phi * quad + trace_term
            lam_new = r_j / denom if denom > 1e-12 else lambdas[g.name]
            lambdas_new[g.name] = float(np.clip(lam_new, 1e-6, 1e10))

        if aa_prev_log_x is not None and len(reml_update_names) > 0:
            log_x = np.array([np.log(lambdas[n]) for n in reml_update_names])
            log_gx = np.array([np.log(lambdas_new[n]) for n in reml_update_names])
            f_curr = log_gx - log_x
            f_prev = aa_prev_log_gx - aa_prev_log_x
            df = f_curr - f_prev
            df_sq = float(np.dot(df, df))
            if df_sq > 1e-20:
                theta = float(-np.dot(f_curr, df) / df_sq)
                theta = max(-0.5, min(theta, 2.0))
                log_acc = (1.0 + theta) * log_gx - theta * aa_prev_log_gx
                for i, name in enumerate(reml_update_names):
                    lambdas_new[name] = float(np.clip(np.exp(log_acc[i]), 1e-6, 1e10))

        if len(reml_update_names) > 0:
            aa_prev_log_x = np.array([np.log(lambdas[n]) for n in reml_update_names])
            aa_prev_log_gx = np.array([np.log(lambdas_new[n]) for n in reml_update_names])

        if use_direct:
            changes = [
                abs(np.log(lambdas_new[g.name]) - np.log(lambdas[g.name]))
                for _, g in reml_groups
                if lambdas[g.name] > 0 and lambdas_new[g.name] > 0
            ]
        else:
            changes = [
                abs(np.log(lambdas_new[g.name]) - np.log(lambdas[g.name]))
                for _, g in reml_groups
                if lambdas[g.name] > 0 and lambdas_new[g.name] > 0 and penalty_ranks[g.name] > 1
            ]
            if not changes:
                changes = [
                    abs(np.log(lambdas_new[g.name]) - np.log(lambdas[g.name]))
                    for _, g in reml_groups
                    if lambdas[g.name] > 0 and lambdas_new[g.name] > 0
                ]
        max_change = max(changes) if changes else 0.0

        if verbose:
            lam_str = ", ".join(f"{g.name}={lambdas_new[g.name]:.4g}" for _, g in reml_groups)
            mode = "cheap" if cheap_iter else f"pirls={last_pirls_iters}"
            print(
                f"  REML iter={n_reml_iter}  max_change={max_change:.6f}  "
                f"({mode})  lambdas=[{lam_str}]"
            )

        lambda_history.append(lambdas_new.copy())

        if max_change < reml_tol:
            converged = True
            lambdas = lambdas_new
            break

        if use_direct:
            warm_beta = beta
            warm_intercept = intercept
            cheap_iter = max_change <= direct_cheap_threshold
        elif max_change > bcd_cheap_threshold:
            old_gms = dm.group_matrices
            dm = rebuild_dm(lambdas_new, sample_weight)
            warm_beta = _map_beta_between_bases(beta, old_gms, dm.group_matrices, groups)
            warm_intercept = intercept
            cheap_iter = False
        else:
            cheap_iter = True

        lambdas = lambdas_new

    if cheap_iter and converged and not use_direct:
        dm = rebuild_dm(lambdas, sample_weight)

    if use_direct:
        final_result, _ = fit_irls_direct(
            X=dm,
            y=y,
            weights=sample_weight,
            family=distribution,
            link=link,
            groups=groups,
            lambda2=lambdas,
            offset=offset_arr,
            beta_init=warm_beta,
            intercept_init=warm_intercept,
            direct_solve=direct_solve,
        )
    else:
        final_result = fit_pirls(
            X=dm,
            y=y,
            weights=sample_weight,
            family=distribution,
            link=link,
            groups=groups,
            penalty=penalty,
            offset=offset_arr,
            beta_init=warm_beta,
            intercept_init=warm_intercept,
            active_set=active_set,
            lambda2=lambdas,
        )

    final_caches = penalty_caches if use_direct else None
    reml_result = REMLResult(
        lambdas=lambdas,
        pirls_result=final_result,
        n_reml_iter=n_reml_iter,
        converged=converged,
        lambda_history=lambda_history,
        objective=reml_laml_objective(
            dm,
            distribution,
            link,
            groups,
            y,
            final_result,
            lambdas,
            sample_weight,
            offset_arr,
            penalty_caches=final_caches,
        ),
    )
    return reml_result, dm
