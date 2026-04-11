"""Model-bound REML adapter helpers for fit operations."""

from __future__ import annotations

from superglm.reml.direct import optimize_direct_reml
from superglm.reml.gradient import reml_direct_gradient, reml_direct_hessian
from superglm.reml.objective import reml_laml_objective
from superglm.reml.runner import run_reml_once
from superglm.reml.w_derivatives import compute_dW_deta, reml_w_correction


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
    scop_states = None
    reml_result = getattr(model, "_reml_result", None)
    if reml_result is not None:
        scop_states = getattr(reml_result, "scop_states", None)

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
        scop_states=scop_states,
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
    from superglm.reml.efs import optimize_efs_reml

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
