"""REML optimizer internals.

Contains the private REML helper functions extracted from SuperGLM.model:
gradient, Hessian, W(rho) correction, Laplace objective, and the three
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

from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm.distributions import _VARIANCE_FLOOR, clip_mu
from superglm.group_matrix import DesignMatrix
from superglm.links import stabilize_eta
from superglm.reml.penalty_algebra import (
    build_penalty_caches,
    build_penalty_components,
    compute_total_penalty_rank,
)
from superglm.reml.result import REMLResult, _map_beta_between_bases
from superglm.solvers.irls_direct import (
    _build_penalty_matrix,
    _invert_xtwx_plus_penalty,
    fit_irls_direct,
)
from superglm.solvers.pirls import fit_pirls
from superglm.types import GroupSlice, PenaltyComponent


def _coerce_reml_penalties(
    reml_groups=None,
    reml_penalties=None,
    group_matrices=None,
    penalty_caches=None,
):
    """Coerce legacy reml_groups to reml_penalties list.

    Accepts either reml_penalties (preferred) or reml_groups (legacy).
    When given reml_groups, builds PenaltyComponent objects from the
    group matrices and penalty_caches (if available).
    """
    if reml_penalties is not None:
        return reml_penalties
    if reml_groups is None:
        raise ValueError("Either reml_penalties or reml_groups must be provided")
    # Build from legacy reml_groups
    components = []
    for idx, g in reml_groups:
        gm = group_matrices[idx] if group_matrices is not None else None
        omega_ssp = None
        rank = 0.0
        log_det = 0.0
        eigvals = None
        omega_raw = None
        if penalty_caches is not None and g.name in penalty_caches:
            cache = penalty_caches[g.name]
            omega_ssp = cache.omega_ssp
            rank = cache.rank
            log_det = cache.log_det_omega_plus
            eigvals = cache.eigvals_omega
        elif (
            gm is not None
            and hasattr(gm, "R_inv")
            and hasattr(gm, "omega")
            and gm.omega is not None
        ):
            omega_ssp = gm.R_inv.T @ gm.omega @ gm.R_inv
        if gm is not None and hasattr(gm, "omega"):
            omega_raw = gm.omega
        components.append(
            PenaltyComponent(
                name=g.name,
                group_name=g.name,
                group_index=idx,
                group_sl=g.sl,
                omega_raw=omega_raw,
                omega_ssp=omega_ssp,
                rank=rank,
                log_det_omega_plus=log_det,
                eigvals_omega=eigvals,
            )
        )
    return components


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
    reml_penalties: list[PenaltyComponent] | None = None,
    estimated_names: set[str] | None = None,
    pirls_tol: float = 1e-6,
    max_pirls_iter: int = 100,
) -> tuple[REMLResult, DesignMatrix]:
    """Run a single REML fixed-point outer loop from a chosen initial lambda scale.

    Returns (REMLResult, DesignMatrix) -- the dm may be updated on the BCD path.
    """
    from superglm.inference.covariance import _penalised_xtwx_inv_gram
    from superglm.reml.objective import reml_laml_objective

    scale_known = getattr(distribution, "scale_known", True)

    if reml_penalties is not None:
        penalties_rro = reml_penalties
    else:
        penalties_rro = _coerce_reml_penalties(
            reml_groups=reml_groups,
            group_matrices=dm.group_matrices,
            penalty_caches=penalty_caches,
        )
    if use_direct:
        reml_update_names = [
            pc.name for pc in penalties_rro if estimated_names is None or pc.name in estimated_names
        ]
    else:
        reml_update_names = [
            pc.name
            for pc in penalties_rro
            if pc.rank > 1 and (estimated_names is None or pc.name in estimated_names)
        ]

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
    direct_has_scalar_groups = any(pc.rank <= 1 for pc in penalties_rro)
    direct_cheap_threshold = 0.01 if direct_has_scalar_groups else 0.2
    bcd_cheap_threshold = 0.01

    for reml_iter in range(max_reml_iter):
        n_reml_iter = reml_iter + 1

        if use_direct and not cheap_iter:
            S_iter = _build_penalty_matrix(
                dm.group_matrices, groups, lambdas, dm.p, reml_penalties=penalties_rro
            )
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
                S_override=S_iter,
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
                tol=pirls_tol,
                max_iter_outer=max_pirls_iter,
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
                cached_direct_xtwx,
                dm.group_matrices,
                groups,
                lambdas,
                reml_penalties=penalties_rro,
            )
            active_groups = list(groups)
        elif not use_direct:
            S_rro = _build_penalty_matrix(
                dm.group_matrices,
                groups,
                lambdas,
                dm.p,
                reml_penalties=penalties_rro,
            )
            XtWX_S_inv, _, active_groups, _, _ = _penalised_xtwx_inv_gram(
                beta, W, dm.group_matrices, groups, lambdas, S_override=S_rro
            )

        inv_phi = 1.0
        if not scale_known and penalty_caches is not None:
            p_dim = dm.p
            S_fp = _build_penalty_matrix(
                dm.group_matrices,
                groups,
                lambdas,
                p_dim,
                reml_penalties=penalties_rro,
            )
            pq = float(beta @ S_fp @ beta)
            M_p = compute_total_penalty_rank(penalties_rro)
            phi_hat = max((pirls_result.deviance + pq) / max(len(y) - M_p, 1.0), 1e-10)
            inv_phi = 1.0 / phi_hat

        lambdas_new = lambdas.copy()
        for pc in penalties_rro:
            # Skip fixed-lambda components — never update their value
            if estimated_names is not None and pc.name not in estimated_names:
                continue
            if not use_direct and pc.rank <= 1:
                continue

            gm = dm.group_matrices[pc.group_index]
            beta_g = beta[pc.group_sl]
            if np.linalg.norm(beta_g) < 1e-12:
                continue

            omega_ssp = (
                pc.omega_ssp if pc.omega_ssp is not None else gm.R_inv.T @ gm.omega @ gm.R_inv
            )
            quad = float(beta_g @ omega_ssp @ beta_g)

            ag = next((a for a in active_groups if a.name == pc.group_name), None)
            if ag is None:
                continue

            H_inv_jj = XtWX_S_inv[ag.sl, ag.sl]
            trace_term = float(np.trace(H_inv_jj @ omega_ssp))

            r_j = pc.rank if pc.rank > 0 else penalty_ranks.get(pc.name, 0.0)
            denom = inv_phi * quad + trace_term
            lam_new = r_j / denom if denom > 1e-12 else lambdas[pc.name]
            lambdas_new[pc.name] = float(np.clip(lam_new, 1e-6, 1e10))

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
                abs(np.log(lambdas_new[pc.name]) - np.log(lambdas[pc.name]))
                for pc in penalties_rro
                if lambdas[pc.name] > 0 and lambdas_new[pc.name] > 0
            ]
        else:
            changes = [
                abs(np.log(lambdas_new[pc.name]) - np.log(lambdas[pc.name]))
                for pc in penalties_rro
                if lambdas[pc.name] > 0 and lambdas_new[pc.name] > 0 and pc.rank > 1
            ]
            if not changes:
                changes = [
                    abs(np.log(lambdas_new[pc.name]) - np.log(lambdas[pc.name]))
                    for pc in penalties_rro
                    if lambdas[pc.name] > 0 and lambdas_new[pc.name] > 0
                ]
        max_change = max(changes) if changes else 0.0

        if verbose:
            lam_str = ", ".join(f"{pc.name}={lambdas_new[pc.name]:.4g}" for pc in penalties_rro)
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
            # R_inv changed -> refresh penalties + caches (basis-dependent)
            penalty_caches = build_penalty_caches(dm.group_matrices, reml_groups)
            penalty_ranks = {n_: c.rank for n_, c in penalty_caches.items()}
            penalties_rro = build_penalty_components(dm.group_matrices, reml_groups)
            cheap_iter = False
        else:
            cheap_iter = True

        lambdas = lambdas_new

    if cheap_iter and converged and not use_direct:
        dm = rebuild_dm(lambdas, sample_weight)
        # R_inv changed -> refresh penalties so S_final_rro uses current basis
        penalties_rro = build_penalty_components(dm.group_matrices, reml_groups)

    S_final_rro = _build_penalty_matrix(
        dm.group_matrices, groups, lambdas, dm.p, reml_penalties=penalties_rro
    )
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
            S_override=S_final_rro,
        )
    else:
        # BCD/PIRLS path -- S_override not yet supported (Cut 2B)
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
            S_override=S_final_rro if use_direct else None,
            reml_penalties=penalties_rro,
        ),
    )
    return reml_result, dm
