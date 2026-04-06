"""EFS REML optimizer (Wood & Fasiolo 2017).

Generalized Fellner-Schall fixed-point iteration with X'WX caching,
Anderson(1) acceleration, and two-tier iteration.

Used when lambda1 > 0 (group lasso + REML smoothing).

References
----------
Wood & Fasiolo (2017). A generalized Fellner-Schall method for smoothing
parameter optimization with application to shape constrained regression.
Biometrics 73(4), 1071-1081.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm.distributions import _VARIANCE_FLOOR, clip_mu
from superglm.group_matrix import (
    DesignMatrix,
    _block_xtwx,
)
from superglm.links import stabilize_eta
from superglm.reml.objective import reml_laml_objective
from superglm.reml.penalty_algebra import (
    build_penalty_caches,
    build_penalty_components,
    compute_total_penalty_rank,
)
from superglm.reml.result import REMLResult, _map_beta_between_bases
from superglm.reml.runner import _coerce_reml_penalties
from superglm.solvers.irls_direct import (
    _build_penalty_matrix,
    _safe_decompose_H,
)
from superglm.solvers.pirls import fit_pirls
from superglm.types import GroupSlice, PenaltyComponent


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
    reml_penalties: list[PenaltyComponent] | None = None,
    estimated_names: set[str] | None = None,
    pirls_tol: float = 1e-6,
    max_pirls_iter: int = 100,
) -> tuple[REMLResult, DesignMatrix]:
    """EFS (generalized Fellner-Schall) REML optimizer for the BCD path.

    Implements Wood & Fasiolo (2017) fixed-point iteration with:
    - X'WX caching for O(p^3) cheap iterations (no data pass)
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
    penalties = _coerce_reml_penalties(
        reml_groups=reml_groups,
        reml_penalties=reml_penalties,
        group_matrices=dm.group_matrices,
        penalty_caches=penalty_caches,
    )
    reml_update_names = [
        pc.name for pc in penalties if estimated_names is None or pc.name in estimated_names
    ]
    n = len(y)

    # -- Bootstrap: one PIRLS with minimal penalty -> one EFS step --
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
        tol=pirls_tol,
        max_iter_outer=max_pirls_iter,
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
    S_boot = _build_penalty_matrix(
        dm.group_matrices, groups, boot_lambdas, dm.p, reml_penalties=penalties
    )
    if not scale_known and penalty_caches is not None:
        pq_boot = float(boot_result.beta @ S_boot @ boot_result.beta)
        M_p = compute_total_penalty_rank(penalties)
        boot_phi = max((boot_result.deviance + pq_boot) / max(n - M_p, 1.0), 1e-10)
        boot_inv_phi = 1.0 / boot_phi

    # One EFS fixed-point step on bootstrap beta
    H_boot = boot_xtwx + S_boot
    H_boot_inv, _, _ = _safe_decompose_H(H_boot)

    for pc in penalties:
        # Skip fixed-lambda components — never update their value
        if estimated_names is not None and pc.name not in estimated_names:
            continue
        beta_g = boot_result.beta[pc.group_sl]
        if np.linalg.norm(beta_g) < 1e-12:
            continue
        omega_ssp = pc.omega_ssp if pc.omega_ssp is not None else penalty_caches[pc.name].omega_ssp
        quad = float(beta_g @ omega_ssp @ beta_g)
        trace_term = float(np.trace(H_boot_inv[pc.group_sl, pc.group_sl] @ omega_ssp))
        r_j = pc.rank if pc.rank > 0 else penalty_ranks[pc.name]
        denom = boot_inv_phi * quad + trace_term
        lam_fp = r_j / denom if denom > 1e-12 else 1.0
        lambdas[pc.name] = float(np.clip(lam_fp, 1e-6, 1e10))

    # Rebuild DM with bootstrapped lambdas -- refresh penalties + caches
    old_gms = dm.group_matrices
    dm = rebuild_dm(lambdas, sample_weight)
    penalty_caches = build_penalty_caches(dm.group_matrices, reml_groups)
    penalty_ranks = {n_: c.rank for n_, c in penalty_caches.items()}
    penalties = build_penalty_components(dm.group_matrices, reml_groups)
    warm_beta = _map_beta_between_bases(boot_result.beta, old_gms, dm.group_matrices, groups)
    warm_intercept = float(boot_result.intercept)

    if verbose:
        lam_str = ", ".join(f"{pc.name}={lambdas[pc.name]:.4g}" for pc in penalties)
        print(f"  REML bootstrap: lambdas=[{lam_str}]")

    # -- Main EFS loop --
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
    # significantly -- there is no valid "PIRLS without DM rebuild" tier.
    cheap_threshold = 0.01

    for reml_iter in range(max_reml_iter):
        n_reml_iter = reml_iter + 1

        # -- Tier 1 & 2: Full PIRLS solve --
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
                tol=pirls_tol,
                max_iter_outer=max_pirls_iter,
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

        # -- Compute H^{-1} = (X'WX + S)^{-1} --
        p = dm.p
        S = _build_penalty_matrix(dm.group_matrices, groups, lambdas, p, reml_penalties=penalties)
        H = cached_xtwx + S
        H_inv, _, _ = _safe_decompose_H(H)

        # -- Estimate phi for estimated-scale families --
        inv_phi = 1.0
        if not scale_known and penalty_caches is not None:
            pq = float(beta @ S @ beta)
            M_p = compute_total_penalty_rank(penalties)
            phi_hat = max((pirls_result.deviance + pq) / max(n - M_p, 1.0), 1e-10)
            inv_phi = 1.0 / phi_hat

        # -- EFS lambda update --
        lambdas_new = lambdas.copy()
        for pc in penalties:
            # Skip fixed-lambda components — never update their value
            if estimated_names is not None and pc.name not in estimated_names:
                continue

            beta_g = beta[pc.group_sl]

            # Skip zeroed groups (L1 penalty killed them)
            if np.linalg.norm(beta_g) < 1e-12:
                continue

            omega_ssp = (
                pc.omega_ssp if pc.omega_ssp is not None else penalty_caches[pc.name].omega_ssp
            )
            quad = float(beta_g @ omega_ssp @ beta_g)
            trace_term = float(np.trace(H_inv[pc.group_sl, pc.group_sl] @ omega_ssp))

            r_j = pc.rank if pc.rank > 0 else penalty_ranks[pc.name]
            denom = inv_phi * quad + trace_term

            if denom > 1e-12:
                lam_new = r_j / denom
            else:
                lam_new = lambdas[pc.name]

            # Clamp log-lambda step to prevent wild jumps
            log_step = np.log(max(lam_new, 1e-10)) - np.log(max(lambdas[pc.name], 1e-10))
            log_step = np.clip(log_step, -5.0, 5.0)
            lam_new = lambdas[pc.name] * np.exp(log_step)

            lambdas_new[pc.name] = float(np.clip(lam_new, 1e-6, 1e10))

        # -- Anderson(1) acceleration on log-lambda --
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

        # -- Stale-basis uphill-step guard (heuristic) --
        # EFS is not guaranteed to decrease the REML objective (Wood &
        # Fasiolo 2017).  We evaluate the objective at lambdas_new using
        # the *current* dm, pirls_result, and cached_xtwx -- all of which
        # are stale w.r.t. the proposed lambdas.  After this check, the
        # DM/R_inv may be rebuilt, changing the true objective surface.
        # So this guard can: (a) damp a step that would actually improve
        # the post-rebuild objective, or (b) miss a step that goes uphill
        # after rebuild.  It is a heuristic safeguard against gross uphill
        # moves, not a monotonicity fix.
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
                reml_penalties=penalties,
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
                reml_penalties=penalties,
            )
            if obj_trial > obj_curr + 1e-8 * max(abs(obj_curr), 1.0):
                for pc in penalties:
                    if estimated_names is not None and pc.name not in estimated_names:
                        continue
                    log_old = np.log(max(lambdas[pc.name], 1e-10))
                    log_new = np.log(max(lambdas_new[pc.name], 1e-10))
                    lambdas_new[pc.name] = float(
                        np.clip(np.exp(0.5 * (log_old + log_new)), 1e-6, 1e10)
                    )
                if len(reml_update_names) > 0:
                    aa_prev_log_gx = np.array([np.log(lambdas_new[n_]) for n_ in reml_update_names])

        # -- Convergence check --
        changes = [
            abs(np.log(lambdas_new[pc.name]) - np.log(lambdas[pc.name]))
            for pc in penalties
            if lambdas[pc.name] > 0 and lambdas_new[pc.name] > 0
        ]
        max_change = max(changes) if changes else 0.0

        if verbose:
            lam_str = ", ".join(f"{pc.name}={lambdas_new[pc.name]:.4g}" for pc in penalties)
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

        # -- Decide next iteration tier --
        if max_change > cheap_threshold:
            # Full: rebuild DM (R_inv depends on lambda) + PIRLS
            old_gms = dm.group_matrices
            dm = rebuild_dm(lambdas_new, sample_weight)
            warm_beta = _map_beta_between_bases(beta, old_gms, dm.group_matrices, groups)
            warm_intercept = intercept
            # R_inv changed -> recompute penalties + caches (basis-dependent)
            penalty_caches = build_penalty_caches(dm.group_matrices, reml_groups)
            penalty_ranks = {n_: c.rank for n_, c in penalty_caches.items()}
            penalties = build_penalty_components(dm.group_matrices, reml_groups)
            cheap_iter = False
        else:
            # Cheap: re-invert cached X'WX + S only (O(p^3), no data pass)
            cheap_iter = True

        lambdas = lambdas_new

    # -- Final refit --
    if cheap_iter and converged:
        dm = rebuild_dm(lambdas, sample_weight)
        # R_inv changed -> refresh caches and penalties for the objective computation
        penalty_caches = build_penalty_caches(dm.group_matrices, reml_groups)
        penalties = build_penalty_components(dm.group_matrices, reml_groups)

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
        tol=pirls_tol,
        max_iter_outer=max_pirls_iter,
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
            reml_penalties=penalties,
        ),
    )
    return reml_result, dm
