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

from superglm.distributions import clip_mu
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
    V = np.maximum(distribution.variance(mu), 1e-10)
    Vp = distribution.variance_derivative(mu)
    return sample_weight * (g1 / V) * (2.0 * g2 - g1**2 * Vp / V)


# ═══════════════════════════════════════════════════════════════════
# W(ρ) correction
# ═══════════════════════════════════════════════════════════════════


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
) -> tuple[NDArray, dict[int, NDArray]] | None:
    """First-order W(ρ) correction for REML derivatives.

    Computes the contribution from d(X'WX)/dρ_j = X'diag(dW/dρ_j)X
    which the fixed-W Laplace approximation drops.  The gradient
    correction is exact to first order; the Hessian C_j matrices are
    first-order (d²W/dρ² terms are dropped).

    Returns (grad_correction, dH_extra) or None if the correction vanishes
    (e.g. Gamma with log link where dW/dη = 0 identically) or if the
    link/distribution does not provide the required second-order methods.
    """
    eta = stabilize_eta(dm.matvec(pirls_result.beta) + pirls_result.intercept + offset_arr, link)
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

    Handles both known-scale families (Poisson, NB2 where φ=1) and
    estimated-scale families (Gamma, Tweedie) via φ-profiled REML.
    """
    eta = stabilize_eta(dm.matvec(result.beta) + result.intercept + offset_arr, link)
    mu = clip_mu(link.inverse(eta), distribution)
    if XtWX is None:
        V = distribution.variance(mu)
        dmu_deta = link.deriv_inverse(eta)
        W = sample_weight * dmu_deta**2 / np.maximum(V, 1e-10)
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
    """Partial gradient of the LAML objective w.r.t. log-lambdas (fixed W)."""
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
) -> NDArray:
    """Outer Hessian of the REML criterion w.r.t. log-lambdas."""
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

    lambda_history: list[dict[str, float]] = [lambdas.copy()]
    warm_beta: NDArray | None = None
    warm_intercept: float | None = None
    grad_tol = max(reml_tol, 1e-6)

    best_obj = np.inf
    best_lambdas = lambdas.copy()
    best_pirls = None
    best_grad: NDArray | None = None
    converged = False
    n_iter = 0
    n_warmup = 3

    _t_reml_start = _time.perf_counter()
    _t_pirls = 0.0
    _t_objective = 0.0
    _t_gradient = 0.0
    _t_hessian = 0.0
    _t_w_correction = 0.0
    _t_linesearch = 0.0
    _t_fp_update = 0.0
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
        rho[i] = np.clip(np.log(max(lam_fp, 1e-6)), log_lo, log_hi)

    rho_prev = rho.copy()

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
        inv_phi = 1.0 / max(phi_hat, 1e-10)
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

        # W(ρ) correction: skip during FP warmup and on discrete path
        _t0 = _time.perf_counter()
        if outer >= n_warmup and not discrete:
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
            )
        else:
            w_corr = None
        _t_w_correction += _time.perf_counter() - _t0
        if w_corr is not None:
            grad_w_correction, dH_extra = w_corr
            grad = grad_partial + grad_w_correction
        else:
            grad = grad_partial.copy()
            dH_extra = None

        if obj < best_obj:
            best_obj = obj
            best_lambdas = cand_lambdas.copy()
            best_pirls = pirls_result
            best_grad = grad.copy()

        lambda_history.append(cand_lambdas.copy())

        proj_grad = grad.copy()
        for i in range(m):
            if rho_clipped[i] >= log_hi - 0.01 and grad[i] < 0:
                proj_grad[i] = 0.0
            elif rho_clipped[i] <= log_lo + 0.01 and grad[i] > 0:
                proj_grad[i] = 0.0
        proj_grad_norm = float(np.max(np.abs(proj_grad)))
        rho_change = float(np.max(np.abs(rho_clipped - rho_prev)))

        if verbose:
            phase = "FP" if outer < n_warmup else "Newton"
            lam_str = ", ".join(f"{name}={cand_lambdas[name]:.4g}" for name in group_names)
            print(
                f"  REML {phase} iter={n_iter}  obj={obj:.4f}  "
                f"|∇|={proj_grad_norm:.6f}  Δρ={rho_change:.4f}  "
                f"lambdas=[{lam_str}]"
            )

        rho_prev = rho_clipped.copy()

        if outer >= n_warmup and proj_grad_norm < max(reml_tol, 5e-3):
            converged = True
            break

        # Phase 1: Fixed-point warm-up
        if outer < n_warmup:
            _t0 = _time.perf_counter()
            for i, (idx, g) in enumerate(reml_groups):
                gm = dm.group_matrices[idx]
                if penalty_caches is not None:
                    omega_ssp = penalty_caches[g.name].omega_ssp
                else:
                    omega_ssp = gm.R_inv.T @ gm.omega @ gm.R_inv
                beta_g = pirls_result.beta[g.sl]
                quad = float(beta_g @ omega_ssp @ beta_g)
                H_inv_jj = XtWX_S_inv[g.sl, g.sl]
                trace_term = float(np.trace(H_inv_jj @ omega_ssp))
                r_j = penalty_ranks[g.name]
                denom = inv_phi * quad + trace_term
                lam_new = r_j / denom if denom > 1e-12 else cand_lambdas[g.name]
                # Snap degenerate select=True groups to upper bound.
                # When quad << trace, the FP update is degenerate (any lambda
                # is approximately a fixed point).  Snap breaks the degeneracy.
                if (
                    select_snap
                    and g.subgroup_type is not None
                    and trace_term > 1e-12
                    and inv_phi * quad < 0.1 * trace_term
                ):
                    lam_new = np.exp(log_hi)
                rho[i] = np.clip(np.log(max(lam_new, 1e-6)), log_lo, log_hi)
            _t_fp_update += _time.perf_counter() - _t0
            continue

        # Phase 2: Newton with exact outer Hessian
        _t0 = _time.perf_counter()
        hess = reml_direct_hessian(
            dm.group_matrices,
            distribution,
            XtWX_S_inv,
            cand_lambdas,
            reml_groups,
            grad_partial,
            penalty_ranks,
            penalty_caches=penalty_caches,
            pirls_result=pirls_result,
            n_obs=len(y),
            phi_hat=phi_hat,
            dH_extra=dH_extra,
        )

        eigvals_h, eigvecs_h = np.linalg.eigh(hess)
        max_eig = max(eigvals_h.max(), 1e-12)
        eigvals_pd = np.maximum(eigvals_h, 1e-6 * max_eig)

        if eigvals_h.min() < -0.1 * max_eig:
            step_scale = min(1.0, max_newton_step / max(np.linalg.norm(grad), 1e-8))
            delta = -grad * step_scale
        else:
            hess_pd = (eigvecs_h * eigvals_pd) @ eigvecs_h.T
            delta = -np.linalg.solve(hess_pd, grad)
            delta = np.clip(delta, -max_newton_step, max_newton_step)
        _t_hessian += _time.perf_counter() - _t0

        # Step-halving line search with Armijo condition
        _t0 = _time.perf_counter()
        max_ls = 2 if discrete else 8
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
            rho = np.clip(
                rho_clipped - 0.1 * grad / max(np.linalg.norm(grad), 1e-8),
                log_lo,
                log_hi,
            )
        _t_linesearch += _time.perf_counter() - _t0

    if best_pirls is None:
        raise RuntimeError("Direct REML Newton did not evaluate any candidates")

    grad_norm = float(np.max(np.abs(best_grad))) if best_grad is not None else np.inf
    converged = converged or grad_norm <= grad_tol

    if profile is not None:
        profile["reml_optimizer_s"] = _time.perf_counter() - _t_reml_start
        profile["reml_pirls_s"] = _t_pirls
        profile["reml_objective_s"] = _t_objective
        profile["reml_gradient_s"] = _t_gradient
        profile["reml_w_correction_s"] = _t_w_correction
        profile["reml_hessian_newton_s"] = _t_hessian
        profile["reml_linesearch_s"] = _t_linesearch
        profile["reml_fp_update_s"] = _t_fp_update
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
    grad_tol = max(reml_tol, 5e-3)
    max_newton_step = 5.0
    max_halving = 25

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

        # Newton direction with PD Hessian fix (mgcv-style: flip neg eigs)
        eigvals_h, eigvecs_h = np.linalg.eigh(hess)
        eigvals_pd = np.abs(eigvals_h)
        thresh = max(eigvals_pd.max(), 1e-12) * np.finfo(float).eps ** 0.5
        eigvals_pd = np.maximum(eigvals_pd, thresh)
        delta = -(eigvecs_h * (1.0 / eigvals_pd)) @ (eigvecs_h.T @ grad)

        # Step capping
        max_delta = float(np.max(np.abs(delta)))
        if max_delta > max_newton_step:
            delta *= max_newton_step / max_delta
        _t_newton += _time.perf_counter() - _t0
        _n_newton_steps += 1

        # --- Step 3: Line search (step halving on working-model REML) ---
        _t0 = _time.perf_counter()
        rho_prev = rho.copy()
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
            # Tiny gradient step as last resort
            rho = np.clip(
                rho - 0.1 * grad / max(np.linalg.norm(grad), 1e-8),
                log_lo,
                log_hi,
            )

        # Convergence check
        rho_change = float(np.max(np.abs(rho - rho_prev)))

        # Project gradient onto feasible set
        proj_grad = grad.copy()
        for i in range(m):
            if rho_clipped[i] >= log_hi - 0.01 and grad[i] < 0:
                proj_grad[i] = 0.0
            elif rho_clipped[i] <= log_lo + 0.01 and grad[i] > 0:
                proj_grad[i] = 0.0
        proj_grad_norm = float(np.max(np.abs(proj_grad)))

        if verbose:
            lam_str = ", ".join(f"{name}={cand_lambdas[name]:.4g}" for name in group_names)
            print(
                f"  POI iter {poi_iter + 1}  obj={obj:.4f}  "
                f"|∇|={proj_grad_norm:.6f}  Δρ={rho_change:.4f}  [{lam_str}]"
            )

        obj_change = abs(obj - prev_obj) if poi_iter > 0 else np.inf
        obj_scale = max(abs(obj), 1.0)
        prev_obj = obj
        if poi_iter >= 2 and proj_grad_norm < grad_tol and obj_change < reml_tol * obj_scale:
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
    boot_W = sample_weight * boot_dmu**2 / np.maximum(boot_V, 1e-10)
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
            W = sample_weight * dmu_deta**2 / np.maximum(V, 1e-10)

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
            W = sample_weight * dmu_deta**2 / np.maximum(V, 1e-10)

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
            W = sample_weight * dmu_deta**2 / np.maximum(V, 1e-10)

        if use_direct and cheap_iter:
            if cached_direct_xtwx is None:
                raise RuntimeError("REML cheap iteration missing cached direct XtWX")
            XtWX_S_inv = _invert_xtwx_plus_penalty(
                cached_direct_xtwx, dm.group_matrices, groups, lambdas
            )
            active_groups = list(groups)
        elif not use_direct:
            XtWX_S_inv, active_groups = _penalised_xtwx_inv_gram(
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
