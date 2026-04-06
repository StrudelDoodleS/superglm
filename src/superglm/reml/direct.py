"""Direct REML Newton optimizer (exact path).

Damped Newton optimization of the REML/LAML objective with
W(rho) correction, gradient, Hessian, and Armijo line search.

References
----------
- Wood (2011) Section 6.2.
"""

from __future__ import annotations

import time as _time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm.group_matrix import DesignMatrix
from superglm.reml.discrete import optimize_discrete_reml_cached_w
from superglm.reml.gradient import reml_direct_gradient, reml_direct_hessian
from superglm.reml.objective import reml_laml_objective
from superglm.reml.penalty_algebra import compute_total_penalty_rank
from superglm.reml.result import REMLResult
from superglm.reml.runner import _coerce_reml_penalties
from superglm.reml.w_derivatives import reml_w_correction
from superglm.solvers.irls_direct import (
    _build_penalty_matrix,
    fit_irls_direct,
)
from superglm.types import GroupSlice, PenaltyComponent


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
    reml_penalties: list[PenaltyComponent] | None = None,
    estimated_names: set[str] | None = None,
) -> REMLResult:
    """Optimize the direct REML objective via damped Newton (Wood 2011).

    Two algorithm variants depending on ``discrete``:

    **Exact path** (``discrete=False``):
        W(rho)-corrected direct REML with gradient, Hessian, line search.

    **Discrete path** (``discrete=True``):
        Cached-W fREML optimizer (fewer data passes), delegated to
        ``optimize_discrete_reml_cached_w``.
    """
    penalties = _coerce_reml_penalties(
        reml_groups=reml_groups,
        reml_penalties=reml_penalties,
        group_matrices=dm.group_matrices,
        penalty_caches=penalty_caches,
    )
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
            reml_penalties=penalties,
            estimated_names=estimated_names,
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
    max_newton_step = 5.0
    _eps = np.finfo(float).eps
    _tol = reml_tol

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
    S_boot = _build_penalty_matrix(
        dm.group_matrices, groups, boot_lambdas, dm.p, reml_penalties=penalties
    )
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
        S_override=S_boot,
    )
    _t_pirls += _time.perf_counter() - _t0
    warm_beta = boot_result.beta.copy()
    warm_intercept = float(boot_result.intercept)

    boot_phi = 1.0
    if not scale_known and penalty_caches is not None:
        pq_boot = float(boot_result.beta @ S_boot @ boot_result.beta)
        M_p = compute_total_penalty_rank(penalties)
        boot_phi = max((boot_result.deviance + pq_boot) / max(len(y) - M_p, 1.0), 1e-10)
    boot_inv_phi = 1.0 / max(boot_phi, 1e-10)

    # Store original fixed lambda values so they can be restored exactly
    # after the exp(rho)->clip round-trip (which would clamp 0.0 to 1e-6).
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
        gm = dm.group_matrices[pc.group_index]
        omega_ssp = pc.omega_ssp if pc.omega_ssp is not None else gm.R_inv.T @ gm.omega @ gm.R_inv
        beta_g = boot_result.beta[pc.group_sl]
        quad = float(beta_g @ omega_ssp @ beta_g)
        H_inv_jj = boot_inv[pc.group_sl, pc.group_sl]
        trace_term = float(np.trace(H_inv_jj @ omega_ssp))
        r_j = pc.rank if pc.rank > 0 else (penalty_ranks[pc.name] if penalty_ranks else 0.0)
        denom = boot_inv_phi * quad + trace_term
        lam_fp = r_j / denom if denom > 1e-12 else 1.0
        # Snap degenerate select=True null-space penalties to upper bound.
        # When quad << trace, the FP update is degenerate
        # (any lambda is approx a fixed point).  Snap breaks it.
        if (
            select_snap
            and pc.component_type == "selection"
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
        # Restore exact fixed values (exp->clip would clamp 0.0 to 1e-6)
        cand_lambdas.update(fixed_lambdas)

        # Pre-build penalty matrix S once for this lambda candidate
        S_cand = _build_penalty_matrix(
            dm.group_matrices,
            groups,
            cand_lambdas,
            dm.p,
            reml_penalties=penalties,
        )

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
            S_override=S_cand,
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
            log_det_H=pirls_result.log_det_H,
            S_override=S_cand,
            reml_penalties=penalties,
        )

        phi_hat = 1.0
        if not scale_known and penalty_caches is not None:
            pq = float(pirls_result.beta @ S_cand @ pirls_result.beta)
            M_p = compute_total_penalty_rank(penalties)
            phi_hat = max((pirls_result.deviance + pq) / max(len(y) - M_p, 1.0), 1e-10)
        _t_objective += _time.perf_counter() - _t0

        _t0 = _time.perf_counter()
        grad_partial = reml_direct_gradient(
            dm.group_matrices,
            pirls_result,
            XtWX_S_inv,
            cand_lambdas,
            reml_penalties=penalties,
            phi_hat=phi_hat,
        )
        _t_gradient += _time.perf_counter() - _t0

        # W(rho) correction
        _t0 = _time.perf_counter()
        if not discrete:
            w_corr = reml_w_correction(
                dm,
                link,
                groups,
                pirls_result,
                XtWX_S_inv,
                cand_lambdas,
                penalty_caches=penalty_caches,
                sample_weight=sample_weight,
                offset_arr=offset_arr,
                distribution=distribution,
                w_correction_order=w_correction_order,
                reml_penalties=penalties,
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
            if not estimated_mask[i]:
                # Fixed lambda — always zero out gradient contribution
                proj_grad[i] = 0.0
            elif rho_clipped[i] >= log_hi - 0.01 and grad[i] < 0:
                proj_grad[i] = 0.0
            elif rho_clipped[i] <= log_lo + 0.01 and grad[i] > 0:
                proj_grad[i] = 0.0
        proj_grad_norm = float(np.max(np.abs(proj_grad)))

        # Compound convergence criterion (Wood 2011):
        # Wood (2011) Section 6.2: max(|g_j|) < eps * (1 + |V_r|).
        score_scale = max(1.0 + abs(obj), 1.0)
        obj_change = abs(obj - prev_obj) if outer > 0 else np.inf

        if verbose:
            lam_str = ", ".join(f"{name}={cand_lambdas[name]:.4g}" for name in group_names)
            print(
                f"  REML Newton iter={n_iter}  obj={obj:.4f}  "
                f"|grad|={proj_grad_norm:.6f}  delta_obj={obj_change:.6g}  "
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
            gradient=grad,
            penalty_caches=penalty_caches,
            pirls_result=pirls_result,
            n_obs=len(y),
            phi_hat=phi_hat,
            dH_extra=dH_extra,
            dH2_cross=dH2_cross,
            reml_penalties=penalties,
        )

        # Active-set: freeze components with negligible gradient and Hessian
        freeze_tol = 0.1 * _tol
        frozen = np.zeros(m, dtype=bool)
        for i in range(m):
            if not estimated_mask[i]:
                # Fixed lambda — always freeze
                frozen[i] = True
            elif (
                abs(proj_grad[i]) < freeze_tol * score_scale
                and abs(hess[i, i]) < freeze_tol * score_scale
            ):
                frozen[i] = True
        active_idx = np.where(~frozen)[0]

        if active_idx.size == 0:
            # All components frozen -- converged
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
            trial_lambdas.update(fixed_lambdas)

            _n_linesearch_fits += 1
            S_trial = _build_penalty_matrix(
                dm.group_matrices,
                groups,
                trial_lambdas,
                dm.p,
                reml_penalties=penalties,
            )
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
                S_override=S_trial,
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
                log_det_H=trial_result.log_det_H,
                S_override=S_trial,
                reml_penalties=penalties,
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
            # Use proj_grad so that fixed components are not moved.
            grad_max = float(np.max(np.abs(proj_grad)))
            if grad_max > 1e-12:
                rho = np.clip(
                    rho_clipped - proj_grad / grad_max,
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
