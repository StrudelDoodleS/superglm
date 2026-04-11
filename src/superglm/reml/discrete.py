"""Cached-W fREML optimizer (discrete path).

Performance Oriented Iteration (mgcv bam-style): interleaves one
PIRLS step (W update) with one Newton lambda step on the working
model's REML criterion.

References
----------
- Wood (2011) Section 6.2.
"""

from __future__ import annotations

import time as _time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm.distributions import clip_mu
from superglm.group_matrix import DesignMatrix
from superglm.links import stabilize_eta
from superglm.reml.gradient import reml_direct_gradient, reml_direct_hessian
from superglm.reml.objective import reml_laml_objective
from superglm.reml.penalty_algebra import (
    build_penalty_matrix,
    coerce_reml_penalties,
    compute_total_penalty_rank,
)
from superglm.reml.result import REMLResult
from superglm.solvers.irls_direct import fit_irls_direct
from superglm.solvers.pirls import PIRLSResult
from superglm.types import GroupSlice, PenaltyComponent


def _solve_cached_augmented(
    XtWX: NDArray,
    S: NDArray,
    XtWz: NDArray,
    XtW1: NDArray,
    sum_W: float,
    sum_Wz: float,
) -> tuple[NDArray, float]:
    """Solve the augmented weighted LS system from cached gram quantities.

    Returns (beta, intercept) without any data passes -- just O(p^3) Cholesky.
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
    reml_penalties: list[PenaltyComponent] | None = None,
    estimated_names: set[str] | None = None,
) -> REMLResult:
    """POI fREML optimizer for the discrete path.

    Performance Oriented Iteration (mgcv bam-style): interleaves one
    PIRLS step (W update) with one Newton lambda step on the working
    model's REML criterion.  Line search re-solves the cached augmented
    system analytically (O(p^3), no data pass) for each trial lambda.

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
    penalties = coerce_reml_penalties(
        reml_groups=reml_groups,
        reml_penalties=reml_penalties,
        group_matrices=dm.group_matrices,
        penalty_caches=penalty_caches,
    )
    scale_known = getattr(distribution, "scale_known", True)
    group_names = [pc.name for pc in penalties]
    m = len(group_names)
    # estimated_mask[i] = True  => component i is free to be optimized
    #                     False => component i has a fixed lambda (policy)
    if estimated_names is not None:
        estimated_mask = np.array([pc.name in estimated_names for pc in penalties])
    else:
        estimated_mask = np.ones(m, dtype=bool)
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
    S_boot = build_penalty_matrix(
        dm.group_matrices, groups, boot_lambdas, p, reml_penalties=penalties
    )
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
        S_override=S_boot,
    )
    _t_pirls += _time.perf_counter() - _t0
    _n_pirls_steps += boot_result.n_iter
    warm_beta = boot_result.beta.copy()
    warm_intercept = float(boot_result.intercept)

    # Bootstrap FP step for initial rho
    boot_phi = 1.0
    if not scale_known and penalty_caches is not None:
        pq_boot = float(boot_result.beta @ S_boot @ boot_result.beta)
        M_p = compute_total_penalty_rank(penalties)
        boot_phi = max((boot_result.deviance + pq_boot) / max(len(y) - M_p, 1.0), 1e-10)
    boot_inv_phi = 1.0 / max(boot_phi, 1e-10)

    # Store original fixed lambda values for exact restoration after exp->clip
    fixed_lambdas: dict[str, float] = {}
    for i, pc in enumerate(penalties):
        if not estimated_mask[i]:
            fixed_lambdas[pc.name] = float(lambdas[pc.name])

    rho = np.zeros(m, dtype=np.float64)
    for i, pc in enumerate(penalties):
        if not estimated_mask[i]:
            fixed_val = fixed_lambdas[pc.name]
            rho[i] = np.clip(np.log(max(fixed_val, 1e-6)), log_lo, log_hi)
            continue
        omega_ssp = pc.omega_ssp
        if omega_ssp is None:
            gm = dm.group_matrices[pc.group_index]
            omega_ssp = gm.R_inv.T @ gm.omega @ gm.R_inv
        beta_g = boot_result.beta[pc.group_sl]
        quad = float(beta_g @ omega_ssp @ beta_g)
        H_inv_jj = boot_inv[pc.group_sl, pc.group_sl]
        trace_term = float(np.trace(H_inv_jj @ omega_ssp))
        r_j = pc.rank if pc.rank > 0 else (penalty_ranks[pc.name] if penalty_ranks else 0.0)
        denom = boot_inv_phi * quad + trace_term
        lam_fp = r_j / denom if denom > 1e-12 else 1.0
        # Snap degenerate select=True null-space penalties to upper bound.
        pc_i = penalties[i]
        if (
            select_snap
            and pc_i.component_type == "selection"
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
        cand_lambdas.update(fixed_lambdas)

        # --- Step 1: One PIRLS step (W update) ---
        # Pre-build S once for this candidate
        S_cand = build_penalty_matrix(
            dm.group_matrices,
            groups,
            cand_lambdas,
            p,
            reml_penalties=penalties,
        )

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
            S_override=S_cand,
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
            S_override=S_cand,
            reml_penalties=penalties,
        )

        phi_hat = 1.0
        if not scale_known and penalty_caches is not None:
            pq = float(pirls_result.beta @ S_cand @ pirls_result.beta)
            M_p = compute_total_penalty_rank(penalties)
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
            reml_penalties=penalties,
            phi_hat=phi_hat,
        )
        hess = reml_direct_hessian(
            dm.group_matrices,
            distribution,
            XtWX_S_inv,
            cand_lambdas,
            gradient=grad,
            penalty_caches=penalty_caches,
            pirls_result=pirls_result,
            n_obs=len(y),
            phi_hat=phi_hat,
            reml_penalties=penalties,
        )

        # Active-set: freeze components with negligible gradient and Hessian
        # Wood (2011) Section 6.2: score_scale = 1 + |V_r|
        score_scale_d = max(1.0 + abs(obj), 1.0)
        freeze_tol_d = 0.1 * _tol

        proj_grad_d = grad.copy()
        for i in range(m):
            if not estimated_mask[i]:
                # Fixed lambda — always zero out gradient contribution
                proj_grad_d[i] = 0.0
            elif rho_clipped[i] >= log_hi - 0.01 and grad[i] < 0:
                proj_grad_d[i] = 0.0
            elif rho_clipped[i] <= log_lo + 0.01 and grad[i] > 0:
                proj_grad_d[i] = 0.0

        frozen_d = np.zeros(m, dtype=bool)
        for i in range(m):
            if not estimated_mask[i]:
                # Fixed lambda — always freeze
                frozen_d[i] = True
            elif (
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
            trial_lambdas.update(fixed_lambdas)

            # Solve augmented system analytically (O(p^3), no data pass)
            S_trial = build_penalty_matrix(
                dm.group_matrices,
                groups,
                trial_lambdas,
                p,
                reml_penalties=penalties,
            )
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
                S_override=S_trial,
                reml_penalties=penalties,
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
            # Use proj_grad_d so that fixed components are not moved.
            grad_max_d = float(np.max(np.abs(proj_grad_d)))
            if grad_max_d > 1e-12:
                rho = np.clip(
                    rho - proj_grad_d / grad_max_d,
                    log_lo,
                    log_hi,
                )
            # else: keep rho unchanged

        # Convergence check -- compound criterion with score_scale
        proj_grad_norm = float(np.max(np.abs(proj_grad_d)))

        if verbose:
            lam_str = ", ".join(f"{name}={cand_lambdas[name]:.4g}" for name in group_names)
            obj_change_d = abs(obj - prev_obj) if poi_iter > 0 else np.inf
            print(
                f"  POI iter {poi_iter + 1}  obj={obj:.4f}  "
                f"|grad|={proj_grad_norm:.6f}  delta_obj={obj_change_d:.6g}  [{lam_str}]"
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
    final_lambdas.update(fixed_lambdas)
    S_final = build_penalty_matrix(
        dm.group_matrices, groups, final_lambdas, dm.p, reml_penalties=penalties
    )
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
        S_override=S_final,
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
        S_override=S_final,
        reml_penalties=penalties,
    )
    _t_objective += _time.perf_counter() - _t0
    # Always use the final refit -- it is the authoritative result from
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
