"""Fitting logic: fit(), fit_path(), fit_reml(), and REML helpers."""

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
from superglm.reml.direct import optimize_direct_reml
from superglm.reml.efs import optimize_efs_reml
from superglm.reml.gradient import reml_direct_gradient, reml_direct_hessian
from superglm.reml.objective import reml_laml_objective
from superglm.reml.runner import run_reml_once
from superglm.reml.w_derivatives import compute_dW_deta, reml_w_correction
from superglm.solvers.irls_direct import (
    _build_penalty_matrix,
    fit_irls_direct,
)
from superglm.solvers.pirls import PIRLSResult, fit_pirls
from superglm.types import FitStats, GroupSlice, LambdaPolicy

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


def fit(
    model,
    X,
    y,
    sample_weight=None,
    offset=None,
    *,
    tol=None,
    max_iter=None,
    convergence=None,
    record_diagnostics=False,
):
    """Fit the model to data."""
    # Resolve fit controls: explicit kwargs > constructor fallback
    tol = tol if tol is not None else model._tol
    max_iter = max_iter if max_iter is not None else model._max_iter
    convergence = convergence if convergence is not None else model._convergence

    # lambda_policy is only supported in fit_reml(); reject here.
    for name, spec in model._specs.items():
        lp = getattr(spec, "_lambda_policy", None)
        if lp is not None:
            raise NotImplementedError(
                f"lambda_policy on feature '{name}' is only supported with "
                f"fit_reml(), not fit(). Use fit_reml() or remove lambda_policy."
            )

    if model._splines is not None and not model._specs:
        from superglm.model.base import auto_detect

        auto_detect(model, X, sample_weight)

    # Clear stale profile results from previous fit
    model._nb_profile_result = None
    model._tweedie_profile_result = None

    # Clear stale REML state so post-fit helpers don't use penalties
    # from a previous fit_reml() call on this model instance.
    model._reml_lambdas = None
    model._reml_penalties = None
    model._reml_result = None
    model._reml_profile = None

    # Auto-estimate NB theta if requested
    if isinstance(model.family, NegativeBinomial) and model.family.theta == "auto":
        from superglm.profiling.nb import estimate_nb_theta

        nb_result = estimate_nb_theta(model, X, y, sample_weight=sample_weight, offset=offset)
        model.family = NegativeBinomial(theta=nb_result.theta_hat)
        model._nb_profile_result = nb_result
        logger.info(f"NB theta estimated: {nb_result.theta_hat:.4f}")

    from superglm.model.base import (
        compute_lambda_max,
        model_build_design_matrix,
        model_has_lambda1_targets,
    )

    y, sample_weight, offset = model_build_design_matrix(model, X, y, sample_weight, offset)

    # Validate response for the resolved distribution
    from superglm.distributions import validate_response

    validate_response(y, model._distribution)

    model._fit_weights = np.array(sample_weight)
    model._fit_offset = np.array(offset) if offset is not None else None
    sample_weight = model._fit_weights
    offset = model._fit_offset

    # Auto-calibrate lambda1 if not set
    if model.penalty.lambda1 is None:
        model.penalty.lambda1 = compute_lambda_max(model, y, sample_weight) * 0.1
    has_lambda1_targets = model_has_lambda1_targets(model)

    # Invalidate cached properties from previous fit
    model.__dict__.pop("_coef_covariance", None)
    model.__dict__.pop("_fit_active_info", None)
    model.__dict__.pop("_fit_inference_info", None)
    model.__dict__.pop("_group_edf", None)

    # Monotone fit-time constraints are incompatible with selection_penalty (lambda1).
    # The constrained QP solver path ignores lambda1 — reject explicitly.
    if (
        any(g.monotone_engine is not None for g in model._groups)
        and model.penalty.lambda1 is not None
        and model.penalty.lambda1 > 0
        and has_lambda1_targets
    ):
        raise NotImplementedError(
            "Monotone fit-time constraints are not supported with selection_penalty > 0. "
            "Set selection_penalty=0 or use monotone_mode='postfit'."
        )

    # Guard: SCOP + QP monotone engines cannot coexist in the same model.
    _monotone_engines = {g.monotone_engine for g in model._groups if g.monotone_engine is not None}
    if len(_monotone_engines) > 1:
        raise NotImplementedError("SCOP + QP monotone terms in the same model are not supported.")

    # Guard: multiple SCOP terms in one model are not yet supported
    # (sequential Gauss-Seidel sweep does not converge reliably).
    _n_scop = sum(1 for g in model._groups if g.monotone_engine == "scop")
    if _n_scop > 1:
        raise NotImplementedError(
            "Multiple SCOP monotone terms in the same model are not yet supported."
        )

    # Direct IRLS when lambda1=0 (no L1 penalty → no BCD needed),
    # or when any group has monotone constraints (constrained QP / SCOP Newton).
    _has_constraints = any(g.constraints is not None for g in model._groups)
    _has_scop = any(g.monotone_engine == "scop" for g in model._groups)
    if (
        _has_constraints
        or _has_scop
        or (
            model.penalty.lambda1 is not None
            and (model.penalty.lambda1 == 0 or not has_lambda1_targets)
        )
    ):
        model._result, _ = fit_irls_direct(
            X=model._dm,
            y=y,
            weights=sample_weight,
            family=model._distribution,
            link=model._link,
            groups=model._groups,
            lambda2=model.lambda2,
            offset=offset,
            max_iter=max_iter,
            tol=tol,
            record_diagnostics=record_diagnostics,
            direct_solve=model._direct_solve,
            convergence=convergence,
        )
    else:
        model._result = fit_pirls(
            X=model._dm,
            y=y,
            weights=sample_weight,
            family=model._distribution,
            link=model._link,
            groups=model._groups,
            penalty=model.penalty,
            offset=offset,
            max_iter_outer=max_iter,
            tol=tol,
            active_set=model._active_set,
            lambda2=model.lambda2,
            record_diagnostics=record_diagnostics,
            convergence=convergence,
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
            iteration_log=model._result.iteration_log,
        )

    eta = model._dm.matvec(model._result.beta) + model._result.intercept
    if offset is not None:
        eta = eta + offset
    eta = stabilize_eta(eta, model._link)
    mu = clip_mu(model._link.inverse(eta), model._distribution)

    model._fit_stats = _compute_fit_stats(
        y, mu, sample_weight, offset, model._distribution, model._link, model._result.phi
    )

    model._last_fit_meta = {"method": "fit", "discrete": model._discrete}
    return model


def fit_path(
    model,
    X,
    y,
    sample_weight=None,
    offset=None,
    *,
    n_lambda=50,
    lambda_ratio=1e-3,
    lambda_seq=None,
):
    """Fit a regularization path from lambda_max down to lambda_min."""
    from superglm.model.base import (
        compute_lambda_max,
        model_build_design_matrix,
        model_has_lambda1_targets,
    )

    y, sample_weight, offset = model_build_design_matrix(model, X, y, sample_weight, offset)
    model._fit_weights = np.array(sample_weight)
    model._fit_offset = np.array(offset) if offset is not None else None
    sample_weight = model._fit_weights
    offset = model._fit_offset
    model.__dict__.pop("_coef_covariance", None)
    model.__dict__.pop("_fit_active_info", None)
    model.__dict__.pop("_fit_inference_info", None)
    model.__dict__.pop("_group_edf", None)

    # Clear stale REML state so post-fit helpers don't use penalties
    # from a previous fit_reml() call on this model instance.
    model._reml_lambdas = None
    model._reml_penalties = None
    model._reml_result = None
    model._reml_profile = None

    if not model_has_lambda1_targets(model):
        raise ValueError(
            "fit_path() requires at least one group targeted by the penalty. "
            "Adjust penalty.features or use fit() / fit_reml() instead."
        )
    lambda_max = compute_lambda_max(model, y, sample_weight)

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
            weights=sample_weight,
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

    eta = model._dm.matvec(result.beta) + result.intercept
    if offset is not None:
        eta = eta + offset
    eta = stabilize_eta(eta, model._link)
    mu = clip_mu(model._link.inverse(eta), model._distribution)
    model._fit_stats = _compute_fit_stats(
        y, mu, sample_weight, offset, model._distribution, model._link, result.phi
    )
    model._last_fit_meta = {"method": "fit_path", "discrete": model._discrete}

    return PathResult(
        lambda_seq=lambda_seq,
        coef_path=coef_path,
        intercept_path=intercept_path,
        deviance_path=deviance_path,
        n_iter_path=n_iter_path,
        converged_path=converged_path,
        edf_path=edf_path,
    )


# ── REML adapter helpers ─────────────────────────────────────────


def model_compute_dW_deta(model, mu, eta, sample_weight):
    """Derivative of IRLS weights w.r.t. the linear predictor."""
    return compute_dW_deta(model._link, model._distribution, mu, eta, sample_weight)


def model_reml_w_correction(
    model,
    pirls_result,
    XtWX_S_inv,
    lambdas,
    reml_groups,
    penalty_caches,
    sample_weight,
    offset_arr,
    w_correction_order=1,
):
    """W(ρ) correction for REML derivatives (first- or second-order)."""
    return reml_w_correction(
        model._dm,
        model._link,
        model._groups,
        pirls_result,
        XtWX_S_inv,
        lambdas,
        reml_groups,
        penalty_caches,
        sample_weight,
        offset_arr,
        model._distribution,
        w_correction_order=w_correction_order,
    )


def model_reml_laml_objective(
    model, y, result, lambdas, sample_weight, offset_arr, XtWX=None, penalty_caches=None
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
        sample_weight,
        offset_arr,
        XtWX=XtWX,
        penalty_caches=penalty_caches,
        reml_penalties=getattr(model, "_reml_penalties", None),
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
        reml_penalties=getattr(model, "_reml_penalties", None),
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
    dH2_cross=None,
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
        dH2_cross=dH2_cross,
        reml_penalties=getattr(model, "_reml_penalties", None),
    )


def model_optimize_direct_reml(
    model,
    y,
    sample_weight,
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
    w_correction_order=1,
    reml_penalties=None,
    estimated_names=None,
):
    """Optimize the direct REML objective via damped Newton (Wood 2011)."""
    return optimize_direct_reml(
        model._dm,
        model._distribution,
        model._link,
        model._groups,
        model._discrete,
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
        max_analytical_per_w=getattr(model, "_max_analytical_per_w", 30),
        select_snap=getattr(model, "_select_snap", True),
        direct_solve=getattr(model, "_direct_solve", "auto"),
        w_correction_order=w_correction_order,
        reml_penalties=reml_penalties,
        estimated_names=estimated_names,
    )


def model_optimize_discrete_reml_cached_w(
    model,
    y,
    sample_weight,
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
    from superglm.reml.discrete import optimize_discrete_reml_cached_w

    return optimize_discrete_reml_cached_w(
        model._dm,
        model._distribution,
        model._link,
        model._groups,
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
        max_analytical_per_w=getattr(model, "_max_analytical_per_w", 30),
    )


def model_optimize_efs_reml(
    model,
    y,
    sample_weight,
    offset_arr,
    reml_groups,
    penalty_ranks,
    lambdas,
    *,
    max_reml_iter,
    reml_tol,
    verbose,
    penalty_caches=None,
    reml_penalties=None,
    estimated_names=None,
    pirls_tol=1e-6,
    max_pirls_iter=100,
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
        sample_weight,
        offset_arr,
        reml_groups,
        penalty_ranks,
        lambdas,
        max_reml_iter=max_reml_iter,
        reml_tol=reml_tol,
        verbose=verbose,
        penalty_caches=penalty_caches,
        rebuild_dm=lambda lambdas, sample_weight: rebuild_dm_with_lambdas(
            model, lambdas, sample_weight
        ),
        reml_penalties=reml_penalties,
        estimated_names=estimated_names,
        pirls_tol=pirls_tol,
        max_pirls_iter=max_pirls_iter,
    )
    model._dm = dm
    return result


def model_run_reml_once(
    model,
    y,
    sample_weight,
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
    pirls_tol=1e-6,
    max_pirls_iter=100,
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
        sample_weight,
        offset_arr,
        reml_groups,
        penalty_ranks,
        lambdas,
        max_reml_iter=max_reml_iter,
        reml_tol=reml_tol,
        verbose=verbose,
        use_direct=use_direct,
        penalty_caches=penalty_caches,
        rebuild_dm=lambda lambdas, sample_weight: rebuild_dm_with_lambdas(
            model, lambdas, sample_weight
        ),
        direct_solve=getattr(model, "_direct_solve", "auto"),
        reml_penalties=getattr(model, "_reml_penalties", None),
        pirls_tol=pirls_tol,
        max_pirls_iter=max_pirls_iter,
    )
    model._dm = dm
    return result


def fit_reml(
    model,
    X,
    y,
    sample_weight=None,
    offset=None,
    *,
    max_reml_iter=20,
    reml_tol=1e-6,
    pirls_tol=1e-6,
    max_pirls_iter=100,
    lambda2_init=None,
    verbose=False,
    w_correction_order=1,
):
    """Fit with REML estimation of per-term smoothing parameters."""
    from superglm.model.base import (
        auto_detect,
        model_build_design_matrix,
        model_has_lambda1_targets,
    )

    if model._splines is not None and not model._specs:
        auto_detect(model, X, sample_weight)

    # Clear stale profile results from previous fit
    model._nb_profile_result = None
    model._tweedie_profile_result = None

    # Auto-estimate NB theta if requested
    if isinstance(model.family, NegativeBinomial) and model.family.theta == "auto":
        from superglm.profiling.nb import estimate_nb_theta

        nb_result = estimate_nb_theta(model, X, y, sample_weight=sample_weight, offset=offset)
        model.family = NegativeBinomial(theta=nb_result.theta_hat)
        model._nb_profile_result = nb_result
        logger.info(f"NB theta estimated: {nb_result.theta_hat:.4f}")

    import time as _time

    _t_total_start = _time.perf_counter()
    _profile: dict = {}

    _t0 = _time.perf_counter()
    y, sample_weight, offset = model_build_design_matrix(model, X, y, sample_weight, offset)
    _profile["dm_build_s"] = _time.perf_counter() - _t0

    model._fit_weights = np.array(sample_weight)
    model._fit_offset = np.array(offset) if offset is not None else None
    sample_weight = model._fit_weights
    offset = model._fit_offset
    model.__dict__.pop("_coef_covariance", None)
    model.__dict__.pop("_fit_active_info", None)
    model.__dict__.pop("_fit_inference_info", None)
    model.__dict__.pop("_group_edf", None)

    # lambda1=None means "no L1 selection" in the REML path — default to 0
    # so the direct IRLS optimizer (Newton REML) is used instead of BCD+EFS.
    if model.penalty.lambda1 is None:
        model.penalty.lambda1 = 0.0

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

    _has_monotone = any(g.monotone_engine is not None for g in model._groups)

    if not reml_groups and not _has_monotone:
        logger.warning("fit_reml: no REML-eligible groups found, falling back to fit()")
        model._result = fit_pirls(
            X=model._dm,
            y=y,
            weights=sample_weight,
            family=model._distribution,
            link=model._link,
            groups=model._groups,
            penalty=model.penalty,
            offset=offset,
            active_set=model._active_set,
            lambda2=model.lambda2,
            tol=pirls_tol,
            max_iter_outer=max_pirls_iter,
        )
        return model

    # Build penalty components and caches (eigenstructure computed once)
    from superglm.reml.penalty_algebra import build_penalty_caches, build_penalty_components

    reml_penalties = build_penalty_components(model._dm.group_matrices, reml_groups)
    penalty_caches = build_penalty_caches(model._dm.group_matrices, reml_groups)
    penalty_ranks = {pc.name: pc.rank for pc in reml_penalties}

    # Initialize per-component lambdas (penalty-indexed, not term-indexed)
    # Partition into fixed (policy.mode == "fixed") and estimated components.
    lam_init = lambda2_init if lambda2_init is not None else model.lambda2
    lambdas = {}
    estimated_names: set[str] = set()
    for pc in reml_penalties:
        if pc.lambda_policy is not None and pc.lambda_policy.mode == "fixed":
            lambdas[pc.name] = float(pc.lambda_policy.value)
        else:
            lambdas[pc.name] = lam_init
            estimated_names.add(pc.name)

    # Check SCOP terms separately — they bypass SSP and don't enter reml_groups.
    # SCOP terms need fixed lambda_policy to be used with fit_reml().
    _scop_groups = [g for g in model._groups if g.monotone_engine == "scop" and g.penalized]
    _any_unfixed_scop = False
    for g in _scop_groups:
        spec = model._specs.get(g.feature_name)
        lp = getattr(spec, "_lambda_policy", None) if spec is not None else None
        if lp is None or not (isinstance(lp, LambdaPolicy) and lp.mode == "fixed"):
            _any_unfixed_scop = True
            break
        # Inject fixed lambda for SCOP group into the lambda dict.
        lambdas[g.name] = float(lp.value)

    # Monotone fit-time constraints are not compatible with automatic REML.
    # When ALL penalized terms have fixed lambdas, we can skip the REML
    # optimizer and call fit_irls_direct directly with fixed lambda values.
    if _has_monotone and (estimated_names or _any_unfixed_scop):
        raise NotImplementedError(
            "Automatic smoothness selection is not yet available when monotone "
            "fit-time terms are present. Supply fixed smoothing parameters via "
            "lambda_policy=LambdaPolicy(mode='fixed', value=X), or use "
            "monotone_mode='postfit'."
        )

    # Direct IRLS when lambda1=0 or unset (no L1 penalty -> no BCD needed)
    offset_arr = offset if offset is not None else np.zeros(len(y))
    lam1 = model.penalty.lambda1
    use_direct = lam1 is None or lam1 == 0 or not model_has_lambda1_targets(model)

    if _has_monotone:
        # All lambdas are fixed (guard above ensures this).
        # Call fit_irls_direct directly — the REML optimizer can't handle
        # monotone constraints (QP / SCOP inner solvers).
        result, XtWX_S_inv = fit_irls_direct(
            X=model._dm,
            y=y,
            weights=sample_weight,
            family=model._distribution,
            link=model._link,
            groups=model._groups,
            lambda2=lambdas,
            offset=offset,
            max_iter=max_pirls_iter,
            tol=pirls_tol,
            convergence="deviance",
            reml_penalties=reml_penalties if reml_penalties else None,
        )

        model._result = result
        model._reml_lambdas = lambdas
        model._reml_penalties = reml_penalties

        # Compute fit stats (mirrors the normal REML post-fit path).
        eta = model._dm.matvec(result.beta) + result.intercept
        if offset is not None:
            eta = eta + offset
        eta = stabilize_eta(eta, model._link)
        mu = clip_mu(model._link.inverse(eta), model._distribution)

        model._fit_stats = _compute_fit_stats(
            y, mu, sample_weight, offset, model._distribution, model._link, result.phi
        )
        model._last_fit_meta = {"method": "fit_reml", "discrete": model._discrete}

        logger.info(f"fit_reml (monotone, fixed lambdas): lambdas={lambdas}")
        return model

    if not estimated_names:
        # All lambdas are fixed — skip REML optimizer; one PIRLS solve at fixed lambdas.
        # Pass estimated_names=set() so the optimizer does a single evaluation pass
        # and never moves any lambda.
        if use_direct:
            best = model_optimize_direct_reml(
                model,
                y,
                sample_weight,
                offset_arr,
                reml_groups,
                penalty_ranks,
                lambdas,
                max_reml_iter=1,
                reml_tol=1.0,  # tol=1.0 ensures convergence after 1 iter
                verbose=verbose,
                penalty_caches=penalty_caches,
                profile=_profile,
                w_correction_order=w_correction_order,
                reml_penalties=reml_penalties,
                estimated_names=estimated_names,
            )
        else:
            best = model_optimize_efs_reml(
                model,
                y,
                sample_weight,
                offset_arr,
                reml_groups,
                penalty_ranks,
                lambdas,
                max_reml_iter=1,
                reml_tol=1.0,
                verbose=verbose,
                penalty_caches=penalty_caches,
                reml_penalties=reml_penalties,
                estimated_names=estimated_names,
                pirls_tol=pirls_tol,
                max_pirls_iter=max_pirls_iter,
            )
    elif use_direct:
        best = model_optimize_direct_reml(
            model,
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
            profile=_profile,
            w_correction_order=w_correction_order,
            reml_penalties=reml_penalties,
            estimated_names=estimated_names,
        )
    else:
        best = model_optimize_efs_reml(
            model,
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
            reml_penalties=reml_penalties,
            estimated_names=estimated_names,
            pirls_tol=pirls_tol,
            max_pirls_iter=max_pirls_iter,
        )
    model._result = best.pirls_result
    model._reml_lambdas = best.lambdas
    # Rebuild penalties from the final DM so omega_ssp matches current R_inv.
    # The EFS path rebuilds the DM (and R_inv) during optimization, so the
    # pre-optimization reml_penalties have stale omega_ssp.
    if not use_direct:
        reml_penalties = build_penalty_components(model._dm.group_matrices, reml_groups)
    model._reml_penalties = reml_penalties
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
        S_final = _build_penalty_matrix(
            model._dm.group_matrices,
            model._groups,
            lambdas,
            p_dim,
            reml_penalties=reml_penalties,
        )
        pq_final = float(best.pirls_result.beta @ S_final @ best.pirls_result.beta)
        from superglm.reml.penalty_algebra import compute_total_penalty_rank

        M_p = compute_total_penalty_rank(reml_penalties)
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
        y, mu, sample_weight, offset, model._distribution, model._link, model._result.phi
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

        # Check if any group was updated by REML (component-aware:
        # multi-penalty tensor keys are "x1:x2:margin_x1" not "x1:x2")
        def _has_lambda(fg):
            if fg.name in lambdas:
                return True
            return any(k.startswith(f"{fg.name}:") for k in lambdas)

        updated = any(_has_lambda(fg) for fg in fgroups)
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
