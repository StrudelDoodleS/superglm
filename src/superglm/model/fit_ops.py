"""Fitting logic: fit(), fit_path(), fit_cv(), fit_reml(), and REML helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from superglm.distributions import Distribution, NegativeBinomial, clip_mu
from superglm.group_matrix import (
    DiscretizedSSPGroupMatrix,
    SparseSSPGroupMatrix,
)
from superglm.links import Link, stabilize_eta
from superglm.reml_optimizer import (
    compute_dW_deta,
    optimize_direct_reml,
    optimize_efs_reml,
    reml_direct_gradient,
    reml_direct_hessian,
    reml_laml_objective,
    reml_w_correction,
    run_reml_once,
)
from superglm.solvers.irls_direct import (
    _build_penalty_matrix,
    fit_irls_direct,
)
from superglm.solvers.pirls import PIRLSResult, fit_pirls
from superglm.types import FitStats, GroupSlice

logger = logging.getLogger(__name__)


@dataclass
class PathResult:
    """Container for regularization path results."""

    lambda_seq: NDArray  # shape (n_lambda,)
    coef_path: NDArray  # shape (n_lambda, p)
    intercept_path: NDArray  # shape (n_lambda,)
    deviance_path: NDArray  # shape (n_lambda,)
    n_iter_path: NDArray  # shape (n_lambda,) — PIRLS iters per lambda
    converged_path: NDArray  # shape (n_lambda,) — bool
    edf_path: NDArray | None = None  # shape (n_lambda,) — effective df


def _compute_fit_stats(
    y: NDArray,
    mu: NDArray,
    weights: NDArray,
    offset: NDArray | None,
    distribution: Distribution,
    link: Link,
    phi: float,
) -> FitStats:
    """Compute scalar fit statistics from training arrays."""
    from superglm.distributions import Binomial, Gaussian, clip_mu

    ll = distribution.log_likelihood(y, mu, weights, phi)

    # Null model (intercept-only MLE), offset-aware.
    # For the null model, use the actual weighted mean (the true MLE for
    # canonical links), clipped into the valid range for the link function.
    y_bar = float(np.average(y, weights=weights))
    if isinstance(distribution, Binomial):
        y_bar = np.clip(y_bar, 1e-3, 1 - 1e-3)
    elif isinstance(distribution, Gaussian):
        y_bar = float(y_bar)
    else:
        y_bar = max(y_bar, 1e-10)

    if offset is None or np.all(offset == 0):
        null_mu = np.full(len(y), y_bar)
    else:
        b0 = float(link.link(np.atleast_1d(y_bar))[0]) - float(np.average(offset, weights=weights))
        for _ in range(25):
            eta_null = stabilize_eta(b0 + offset, link)
            mu_null = clip_mu(link.inverse(eta_null), distribution)
            dmu = link.deriv_inverse(eta_null)
            V = distribution.variance(mu_null)
            score = np.sum(weights * (y - mu_null) * dmu / V)
            info = np.sum(weights * dmu**2 / V)
            step = score / max(info, 1e-10)
            b0 += step
            if abs(step) < 1e-8:
                break
        eta_null = stabilize_eta(b0 + offset, link)
        null_mu = clip_mu(link.inverse(eta_null), distribution)

    null_ll = distribution.log_likelihood(y, null_mu, weights, phi)
    null_dev = float(np.sum(weights * distribution.deviance_unit(y, null_mu)))
    dev = float(np.sum(weights * distribution.deviance_unit(y, mu)))
    expl_dev = 1.0 - dev / null_dev if null_dev > 0 else 0.0

    V = distribution.variance(mu)
    pearson = float(np.sum(weights * (y - mu) ** 2 / V))

    return FitStats(
        log_likelihood=ll,
        null_log_likelihood=null_ll,
        null_deviance=null_dev,
        explained_deviance=expl_dev,
        pearson_chi2=pearson,
        n_obs=len(y),
    )


def fit(model, X, y, exposure=None, offset=None, *, sample_weight=None):
    """Fit the model to data."""
    from superglm.model.base import resolve_sample_weight_alias

    exposure = resolve_sample_weight_alias(exposure, sample_weight, method_name="fit()")
    if model._splines is not None and not model._specs:
        from superglm.model.base import auto_detect

        auto_detect(model, X, exposure)

    # Auto-estimate NB theta if requested
    if isinstance(model.family, NegativeBinomial) and model.family.theta == "auto":
        from superglm.nb_profile import estimate_nb_theta

        nb_result = estimate_nb_theta(model, X, y, exposure=exposure, offset=offset)
        model.family = NegativeBinomial(theta=nb_result.theta_hat)
        model._nb_profile_result = nb_result
        logger.info(f"NB theta estimated: {nb_result.theta_hat:.4f}")

    from superglm.model.base import (
        compute_lambda_max,
        model_build_design_matrix,
        model_has_lambda1_targets,
    )

    y, exposure, offset = model_build_design_matrix(model, X, y, exposure, offset)

    # Validate response for the resolved distribution
    from superglm.distributions import validate_response

    validate_response(y, model._distribution)

    model._fit_weights = np.array(exposure)
    model._fit_offset = np.array(offset) if offset is not None else None
    exposure = model._fit_weights
    offset = model._fit_offset

    # Auto-calibrate lambda1 if not set
    if model.penalty.lambda1 is None:
        model.penalty.lambda1 = compute_lambda_max(model, y, exposure) * 0.1
    has_lambda1_targets = model_has_lambda1_targets(model)

    # Invalidate cached properties from previous fit
    model.__dict__.pop("_coef_covariance", None)
    model.__dict__.pop("_fit_active_info", None)
    model.__dict__.pop("_group_edf", None)

    # Direct IRLS when lambda1=0 (no L1 penalty → no BCD needed)
    if model.penalty.lambda1 is not None and (
        model.penalty.lambda1 == 0 or not has_lambda1_targets
    ):
        model._result, _ = fit_irls_direct(
            X=model._dm,
            y=y,
            weights=exposure,
            family=model._distribution,
            link=model._link,
            groups=model._groups,
            lambda2=model.lambda2,
            offset=offset,
        )
    else:
        model._result = fit_pirls(
            X=model._dm,
            y=y,
            weights=exposure,
            family=model._distribution,
            link=model._link,
            groups=model._groups,
            penalty=model.penalty,
            offset=offset,
            active_set=model._active_set,
            lambda2=model.lambda2,
        )

    # Fix phi for known-scale families (Poisson): phi is always 1.0.
    scale_known = getattr(model._distribution, "scale_known", True)
    if scale_known and model._result.phi != 1.0:
        model._result = PIRLSResult(
            beta=model._result.beta,
            intercept=model._result.intercept,
            n_iter=model._result.n_iter,
            deviance=model._result.deviance,
            converged=model._result.converged,
            phi=1.0,
            effective_df=model._result.effective_df,
        )

    eta = model._dm.matvec(model._result.beta) + model._result.intercept
    if offset is not None:
        eta = eta + offset
    eta = stabilize_eta(eta, model._link)
    mu = clip_mu(model._link.inverse(eta), model._distribution)

    model._fit_stats = _compute_fit_stats(
        y, mu, exposure, offset, model._distribution, model._link, model._result.phi
    )

    model._last_fit_meta = {"method": "fit", "discrete": model._discrete}
    return model


def fit_path(
    model,
    X,
    y,
    exposure=None,
    offset=None,
    *,
    sample_weight=None,
    n_lambda=50,
    lambda_ratio=1e-3,
    lambda_seq=None,
):
    """Fit a regularization path from lambda_max down to lambda_min."""
    from superglm.model.base import (
        compute_lambda_max,
        model_build_design_matrix,
        model_has_lambda1_targets,
        resolve_sample_weight_alias,
    )

    exposure = resolve_sample_weight_alias(exposure, sample_weight, method_name="fit_path()")
    y, exposure, offset = model_build_design_matrix(model, X, y, exposure, offset)
    model._fit_weights = np.array(exposure)
    model._fit_offset = np.array(offset) if offset is not None else None
    exposure = model._fit_weights
    offset = model._fit_offset
    model.__dict__.pop("_coef_covariance", None)
    model.__dict__.pop("_fit_active_info", None)
    model.__dict__.pop("_group_edf", None)
    if not model_has_lambda1_targets(model):
        raise ValueError(
            "fit_path() requires at least one group targeted by the penalty. "
            "Adjust penalty.features or use fit() / fit_reml() instead."
        )
    lambda_max = compute_lambda_max(model, y, exposure)

    if lambda_seq is None:
        lambda_seq = np.geomspace(lambda_max, lambda_max * lambda_ratio, n_lambda)
    else:
        lambda_seq = np.asarray(lambda_seq, dtype=np.float64)
        n_lambda = len(lambda_seq)

    p = model._dm.p
    coef_path = np.zeros((n_lambda, p))
    intercept_path = np.zeros(n_lambda)
    deviance_path = np.zeros(n_lambda)
    edf_path = np.zeros(n_lambda)
    n_iter_path = np.zeros(n_lambda, dtype=int)
    converged_path = np.zeros(n_lambda, dtype=bool)

    beta_warm = None
    intercept_warm = None

    for i, lam in enumerate(lambda_seq):
        model.penalty.lambda1 = lam
        result = fit_pirls(
            X=model._dm,
            y=y,
            weights=exposure,
            family=model._distribution,
            link=model._link,
            groups=model._groups,
            penalty=model.penalty,
            offset=offset,
            beta_init=beta_warm,
            intercept_init=intercept_warm,
            active_set=model._active_set,
            lambda2=model.lambda2,
        )
        coef_path[i] = result.beta
        intercept_path[i] = result.intercept
        deviance_path[i] = result.deviance
        edf_path[i] = result.effective_df
        n_iter_path[i] = result.n_iter
        converged_path[i] = result.converged
        beta_warm = result.beta
        intercept_warm = result.intercept

    # Set model state to the last (least-regularized) fit
    model._result = result

    return PathResult(
        lambda_seq=lambda_seq,
        coef_path=coef_path,
        intercept_path=intercept_path,
        deviance_path=deviance_path,
        n_iter_path=n_iter_path,
        converged_path=converged_path,
        edf_path=edf_path,
    )


def fit_cv(
    model,
    X,
    y,
    exposure=None,
    offset=None,
    *,
    sample_weight=None,
    n_folds=5,
    n_lambda=50,
    lambda_ratio=1e-3,
    lambda_seq=None,
    rule="1se",
    refit=True,
    random_state=None,
):
    """Select lambda by K-fold cross-validation."""
    from superglm.cv import CVResult, _fit_cv_folds, _select_lambda
    from superglm.model.base import (
        auto_detect,
        compute_lambda_max,
        model_build_design_matrix,
        model_has_lambda1_targets,
        resolve_sample_weight_alias,
    )

    exposure = resolve_sample_weight_alias(exposure, sample_weight, method_name="fit_cv()")

    if model._splines is not None and not model._specs:
        auto_detect(model, X, exposure)

    if isinstance(model.family, NegativeBinomial) and model.family.theta == "auto":
        from superglm.nb_profile import estimate_nb_theta

        nb_result = estimate_nb_theta(model, X, y, exposure=exposure, offset=offset)
        model.family = NegativeBinomial(theta=nb_result.theta_hat)
        model._nb_profile_result = nb_result
        logger.info(f"NB theta estimated: {nb_result.theta_hat:.4f}")

    y, exposure, offset = model_build_design_matrix(model, X, y, exposure, offset)
    model._fit_weights = np.array(exposure)
    model._fit_offset = np.array(offset) if offset is not None else None
    exposure = model._fit_weights
    offset = model._fit_offset
    model.__dict__.pop("_coef_covariance", None)
    model.__dict__.pop("_fit_active_info", None)
    model.__dict__.pop("_group_edf", None)
    if not model_has_lambda1_targets(model):
        raise ValueError(
            "fit_cv() requires at least one group targeted by the penalty. "
            "Adjust penalty.features or use fit() / fit_reml() instead."
        )

    # Lambda sequence from full data
    lambda_max = compute_lambda_max(model, y, exposure)
    if lambda_seq is None:
        lambda_seq = np.geomspace(lambda_max, lambda_max * lambda_ratio, n_lambda)
    else:
        lambda_seq = np.asarray(lambda_seq, dtype=np.float64)

    # Create fold indices
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(len(y))
    fold_indices = [arr for arr in np.array_split(indices, n_folds)]

    # Run CV
    fold_deviance = _fit_cv_folds(
        dm=model._dm,
        y=y,
        exposure=exposure,
        groups=model._groups,
        family=model._distribution,
        link=model._link,
        penalty=model.penalty,
        lambda_seq=lambda_seq,
        fold_indices=fold_indices,
        offset=offset,
        active_set=model._active_set,
    )

    mean_cv = fold_deviance.mean(axis=0)
    se_cv = fold_deviance.std(axis=0) / np.sqrt(n_folds)

    best_lambda, best_lambda_1se, best_idx, idx_1se = _select_lambda(
        lambda_seq, mean_cv, se_cv, rule
    )

    # Refit on full data with selected lambda
    path_result = None
    if refit:
        model.penalty.lambda1 = best_lambda
        model.fit(X, y, exposure=exposure, offset=offset)

    return CVResult(
        lambda_seq=lambda_seq,
        mean_cv_deviance=mean_cv,
        se_cv_deviance=se_cv,
        best_lambda=best_lambda,
        best_lambda_1se=best_lambda_1se,
        best_index=best_idx,
        best_index_1se=idx_1se,
        fold_deviance=fold_deviance,
        path_result=path_result,
    )


# ── REML adapter helpers ─────────────────────────────────────────


def model_compute_dW_deta(model, mu, eta, exposure):
    """Derivative of IRLS weights w.r.t. the linear predictor."""
    return compute_dW_deta(model._link, model._distribution, mu, eta, exposure)


def model_reml_w_correction(
    model, pirls_result, XtWX_S_inv, lambdas, reml_groups, penalty_caches, exposure, offset_arr
):
    """First-order W(ρ) correction for REML derivatives."""
    return reml_w_correction(
        model._dm,
        model._link,
        model._groups,
        pirls_result,
        XtWX_S_inv,
        lambdas,
        reml_groups,
        penalty_caches,
        exposure,
        offset_arr,
        model._distribution,
    )


def model_reml_laml_objective(
    model, y, result, lambdas, exposure, offset_arr, XtWX=None, penalty_caches=None
):
    """Laplace REML/LAML objective up to additive constants."""
    return reml_laml_objective(
        model._dm,
        model._distribution,
        model._link,
        model._groups,
        y,
        result,
        lambdas,
        exposure,
        offset_arr,
        XtWX=XtWX,
        penalty_caches=penalty_caches,
    )


def model_reml_direct_gradient(
    model, result, XtWX_S_inv, lambdas, reml_groups, penalty_ranks, phi_hat=1.0
):
    """Partial gradient of the LAML objective w.r.t. log-lambdas (fixed W)."""
    return reml_direct_gradient(
        model._dm.group_matrices,
        result,
        XtWX_S_inv,
        lambdas,
        reml_groups,
        penalty_ranks,
        phi_hat=phi_hat,
    )


def model_reml_direct_hessian(
    model,
    XtWX_S_inv,
    lambdas,
    reml_groups,
    gradient,
    penalty_ranks,
    penalty_caches=None,
    pirls_result=None,
    n_obs=0,
    phi_hat=1.0,
    dH_extra=None,
):
    """Outer Hessian of the REML criterion w.r.t. log-lambdas."""
    return reml_direct_hessian(
        model._dm.group_matrices,
        model._distribution,
        XtWX_S_inv,
        lambdas,
        reml_groups,
        gradient,
        penalty_ranks,
        penalty_caches=penalty_caches,
        pirls_result=pirls_result,
        n_obs=n_obs,
        phi_hat=phi_hat,
        dH_extra=dH_extra,
    )


def model_optimize_direct_reml(
    model,
    y,
    exposure,
    offset_arr,
    reml_groups,
    penalty_ranks,
    lambdas,
    *,
    max_reml_iter,
    reml_tol,
    verbose,
    penalty_caches=None,
    profile=None,
):
    """Optimize the direct REML objective via damped Newton (Wood 2011)."""
    return optimize_direct_reml(
        model._dm,
        model._distribution,
        model._link,
        model._groups,
        model._discrete,
        y,
        exposure,
        offset_arr,
        reml_groups,
        penalty_ranks,
        lambdas,
        max_reml_iter=max_reml_iter,
        reml_tol=reml_tol,
        verbose=verbose,
        penalty_caches=penalty_caches,
        profile=profile,
        max_analytical_per_w=getattr(model, "_max_analytical_per_w", 30),
        select_snap=getattr(model, "_select_snap", True),
    )


def model_optimize_discrete_reml_cached_w(
    model,
    y,
    exposure,
    offset_arr,
    reml_groups,
    penalty_ranks,
    lambdas,
    *,
    max_reml_iter,
    reml_tol,
    verbose,
    penalty_caches=None,
    profile=None,
):
    """Cached-W fREML optimizer for the discrete path."""
    from superglm.reml_optimizer import optimize_discrete_reml_cached_w

    return optimize_discrete_reml_cached_w(
        model._dm,
        model._distribution,
        model._link,
        model._groups,
        y,
        exposure,
        offset_arr,
        reml_groups,
        penalty_ranks,
        lambdas,
        max_reml_iter=max_reml_iter,
        reml_tol=reml_tol,
        verbose=verbose,
        penalty_caches=penalty_caches,
        profile=profile,
        max_analytical_per_w=getattr(model, "_max_analytical_per_w", 30),
    )


def model_optimize_efs_reml(
    model,
    y,
    exposure,
    offset_arr,
    reml_groups,
    penalty_ranks,
    lambdas,
    *,
    max_reml_iter,
    reml_tol,
    verbose,
    penalty_caches=None,
):
    """EFS REML optimizer for the BCD path (lambda1 > 0)."""
    from superglm.model.base import rebuild_dm_with_lambdas

    result, dm = optimize_efs_reml(
        model._dm,
        model._distribution,
        model._link,
        model._groups,
        model.penalty,
        model._active_set,
        y,
        exposure,
        offset_arr,
        reml_groups,
        penalty_ranks,
        lambdas,
        max_reml_iter=max_reml_iter,
        reml_tol=reml_tol,
        verbose=verbose,
        penalty_caches=penalty_caches,
        rebuild_dm=lambda lambdas, exposure: rebuild_dm_with_lambdas(model, lambdas, exposure),
    )
    model._dm = dm
    return result


def model_run_reml_once(
    model,
    y,
    exposure,
    offset_arr,
    reml_groups,
    penalty_ranks,
    lambdas,
    *,
    max_reml_iter,
    reml_tol,
    verbose,
    use_direct,
    penalty_caches=None,
):
    """Run a single REML fixed-point outer loop from a chosen initial lambda scale."""
    from superglm.model.base import rebuild_dm_with_lambdas

    result, dm = run_reml_once(
        model._dm,
        model._distribution,
        model._link,
        model._groups,
        model.penalty,
        model._active_set,
        y,
        exposure,
        offset_arr,
        reml_groups,
        penalty_ranks,
        lambdas,
        max_reml_iter=max_reml_iter,
        reml_tol=reml_tol,
        verbose=verbose,
        use_direct=use_direct,
        penalty_caches=penalty_caches,
        rebuild_dm=lambda lambdas, exposure: rebuild_dm_with_lambdas(model, lambdas, exposure),
    )
    model._dm = dm
    return result


def fit_reml(
    model,
    X,
    y,
    exposure=None,
    offset=None,
    *,
    sample_weight=None,
    max_reml_iter=20,
    reml_tol=1e-4,
    lambda2_init=None,
    verbose=False,
):
    """Fit with REML estimation of per-term smoothing parameters."""
    from superglm.model.base import (
        auto_detect,
        compute_lambda_max,
        model_build_design_matrix,
        model_has_lambda1_targets,
        resolve_sample_weight_alias,
    )

    exposure = resolve_sample_weight_alias(exposure, sample_weight, method_name="fit_reml()")
    if model._splines is not None and not model._specs:
        auto_detect(model, X, exposure)

    # Auto-estimate NB theta if requested
    if isinstance(model.family, NegativeBinomial) and model.family.theta == "auto":
        from superglm.nb_profile import estimate_nb_theta

        nb_result = estimate_nb_theta(model, X, y, exposure=exposure, offset=offset)
        model.family = NegativeBinomial(theta=nb_result.theta_hat)
        model._nb_profile_result = nb_result
        logger.info(f"NB theta estimated: {nb_result.theta_hat:.4f}")

    import time as _time

    _t_total_start = _time.perf_counter()
    _profile: dict = {}

    _t0 = _time.perf_counter()
    y, exposure, offset = model_build_design_matrix(model, X, y, exposure, offset)
    _profile["dm_build_s"] = _time.perf_counter() - _t0

    model._fit_weights = np.array(exposure)
    model._fit_offset = np.array(offset) if offset is not None else None
    exposure = model._fit_weights
    offset = model._fit_offset
    model.__dict__.pop("_coef_covariance", None)
    model.__dict__.pop("_fit_active_info", None)
    model.__dict__.pop("_group_edf", None)

    # Auto-calibrate lambda1 if not set
    if model.penalty.lambda1 is None:
        model.penalty.lambda1 = compute_lambda_max(model, y, exposure) * 0.1

    # Identify REML-eligible groups: penalized SSP groups with stored omega
    reml_groups: list[tuple[int, GroupSlice]] = []
    for i, g in enumerate(model._groups):
        gm = model._dm.group_matrices[i]
        if (
            isinstance(gm, SparseSSPGroupMatrix | DiscretizedSSPGroupMatrix)
            and g.penalized
            and gm.omega is not None
        ):
            reml_groups.append((i, g))

    if not reml_groups:
        logger.warning("fit_reml: no REML-eligible groups found, falling back to fit()")
        model._result = fit_pirls(
            X=model._dm,
            y=y,
            weights=exposure,
            family=model._distribution,
            link=model._link,
            groups=model._groups,
            penalty=model.penalty,
            offset=offset,
            active_set=model._active_set,
            lambda2=model.lambda2,
        )
        return model

    # Initialize per-group lambdas
    lam_init = lambda2_init if lambda2_init is not None else model.lambda2
    lambdas = {g.name: lam_init for _, g in reml_groups}

    # Build penalty caches (eigenstructure computed once, reused across iterations)
    from superglm.reml import build_penalty_caches

    penalty_caches = build_penalty_caches(model._dm.group_matrices, model._groups, reml_groups)
    penalty_ranks = {name: cache.rank for name, cache in penalty_caches.items()}

    # Direct IRLS when lambda1=0 (no L1 penalty → no BCD needed)
    offset_arr = offset if offset is not None else np.zeros(len(y))
    use_direct = model.penalty.lambda1 is not None and (
        model.penalty.lambda1 == 0 or not model_has_lambda1_targets(model)
    )

    if use_direct:
        best = model_optimize_direct_reml(
            model,
            y,
            exposure,
            offset_arr,
            reml_groups,
            penalty_ranks,
            lambdas,
            max_reml_iter=max_reml_iter,
            reml_tol=reml_tol,
            verbose=verbose,
            penalty_caches=penalty_caches,
            profile=_profile,
        )
    else:
        best = model_optimize_efs_reml(
            model,
            y,
            exposure,
            offset_arr,
            reml_groups,
            penalty_ranks,
            lambdas,
            max_reml_iter=max_reml_iter,
            reml_tol=reml_tol,
            verbose=verbose,
            penalty_caches=penalty_caches,
        )
    model._result = best.pirls_result
    model._reml_lambdas = best.lambdas
    model._reml_result = best
    lambdas = best.lambdas
    n_reml_iter = best.n_reml_iter
    converged = best.converged

    # Fix phi: known-scale families (Poisson) get phi=1.0;
    # estimated-scale families get REML profiled φ̂ instead of the raw
    # PIRLS phi = dev/(n-edf) which doesn't include the penalty.
    scale_known = getattr(model._distribution, "scale_known", True)
    if scale_known:
        phi_fixed = 1.0
    else:
        p_dim = model._dm.p
        S_final = _build_penalty_matrix(model._dm.group_matrices, model._groups, lambdas, p_dim)
        pq_final = float(best.pirls_result.beta @ S_final @ best.pirls_result.beta)
        M_p = sum(penalty_ranks[g.name] for _, g in reml_groups)
        phi_fixed = max((best.pirls_result.deviance + pq_final) / max(len(y) - M_p, 1.0), 1e-10)

    corrected = PIRLSResult(
        beta=best.pirls_result.beta,
        intercept=best.pirls_result.intercept,
        n_iter=best.pirls_result.n_iter,
        deviance=best.pirls_result.deviance,
        converged=best.pirls_result.converged,
        phi=phi_fixed,
        effective_df=best.pirls_result.effective_df,
    )
    model._result = corrected
    model._reml_result.pirls_result = corrected

    # Update spec R_inv for predict/reconstruct
    _update_reml_r_inv(model, reml_groups, lambdas)

    _profile["total_s"] = _time.perf_counter() - _t_total_start
    _profile["n_reml_iter"] = n_reml_iter
    _profile["converged"] = converged
    model._reml_profile = _profile

    eta = model._dm.matvec(model._result.beta) + model._result.intercept
    if offset is not None:
        eta = eta + offset
    eta = stabilize_eta(eta, model._link)
    mu = clip_mu(model._link.inverse(eta), model._distribution)

    model._fit_stats = _compute_fit_stats(
        y, mu, exposure, offset, model._distribution, model._link, model._result.phi
    )

    model._last_fit_meta = {"method": "fit_reml", "discrete": model._discrete}

    logger.info(f"REML converged={converged} in {n_reml_iter} iters, lambdas={lambdas}")
    return model


def _update_reml_r_inv(model, reml_groups, lambdas):
    """Update spec R_inv for predict/reconstruct after REML convergence."""
    from superglm.model.state_ops import feature_groups

    for idx, g in reml_groups:
        spec = model._specs.get(g.feature_name)
        if spec is not None and hasattr(spec, "set_reparametrisation"):
            fgroups = feature_groups(model, g.feature_name)
            r_inv_parts = []
            for fg in fgroups:
                fg_idx = next(i for i, gg in enumerate(model._groups) if gg.name == fg.name)
                fg_gm = model._dm.group_matrices[fg_idx]
                if isinstance(fg_gm, SparseSSPGroupMatrix | DiscretizedSSPGroupMatrix):
                    r_inv_parts.append(fg_gm.R_inv)
            if r_inv_parts:
                spec.set_reparametrisation(
                    np.hstack(r_inv_parts) if len(r_inv_parts) > 1 else r_inv_parts[0]
                )

    # Same for interaction specs
    for iname in model._interaction_order:
        ispec = model._interaction_specs[iname]
        if not hasattr(ispec, "set_reparametrisation"):
            continue
        fgroups = feature_groups(model, iname)
        # Check if any group was updated by REML
        updated = any(fg.name in lambdas for fg in fgroups)
        if not updated:
            continue
        if len(fgroups) > 1:
            if any(fg.subgroup_type is not None for fg in fgroups):
                r_inv_parts = []
                for fg in fgroups:
                    fg_idx = next(i for i, gg in enumerate(model._groups) if gg.name == fg.name)
                    fg_gm = model._dm.group_matrices[fg_idx]
                    if isinstance(fg_gm, SparseSSPGroupMatrix | DiscretizedSSPGroupMatrix):
                        r_inv_parts.append(fg_gm.R_inv)
                if r_inv_parts:
                    ispec.set_reparametrisation(np.hstack(r_inv_parts))
            else:
                # Per-level (SplineCategorical): gather dict
                r_inv_dict = {}
                for fg in fgroups:
                    fg_idx = next(i for i, gg in enumerate(model._groups) if gg.name == fg.name)
                    fg_gm = model._dm.group_matrices[fg_idx]
                    if isinstance(fg_gm, SparseSSPGroupMatrix | DiscretizedSSPGroupMatrix):
                        level = fg.name.split("[")[1].rstrip("]") if "[" in fg.name else fg.name
                        r_inv_dict[level] = fg_gm.R_inv
                if r_inv_dict:
                    ispec.set_reparametrisation(r_inv_dict)
        else:
            # Single group (TensorInteraction)
            fg = fgroups[0]
            fg_idx = next(i for i, gg in enumerate(model._groups) if gg.name == fg.name)
            fg_gm = model._dm.group_matrices[fg_idx]
            if isinstance(fg_gm, SparseSSPGroupMatrix | DiscretizedSSPGroupMatrix):
                ispec.set_reparametrisation(fg_gm.R_inv)
