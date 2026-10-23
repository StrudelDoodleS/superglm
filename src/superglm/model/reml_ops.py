"""Model-bound REML adapter helpers for fit operations."""

from __future__ import annotations

from superglm.model.fit_state import configured_penalty
from superglm.reml.direct import optimize_direct_reml
from superglm.solvers.dispersion import model_weight_semantics


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
    pirls_tol=1e-6,
    max_pirls_iter=100,
    debug_recorder=None,
    trace_run=None,
    warm_lambdas=None,
):
    """Optimize the direct REML objective via damped Newton (Wood 2011)."""
    from superglm.model.reml_execute import resolve_reml_tol

    # direct.py stops on Wood's compound bar -- a Newton engine -- so this
    # seam resolves the public None sentinel to the tight Newton default.
    reml_tol = resolve_reml_tol(reml_tol, engine="newton")
    result = optimize_direct_reml(
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
        weight_semantics=model_weight_semantics(model),
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
        pirls_tol=pirls_tol,
        max_pirls_iter=max_pirls_iter,
        debug_recorder=debug_recorder,
        trace_run=trace_run,
        warm_lambdas=warm_lambdas,
        # A Tweedie power search's candidate fits keep the plain cold start.
        presolve=False if getattr(model, "_in_power_search", False) else None,
    )
    if model._discrete:
        from superglm.model.base import rebuild_dm_with_lambdas

        model._dm = rebuild_dm_with_lambdas(model, result.lambdas, sample_weight)
    return result


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
    from superglm.model.reml_execute import resolve_reml_tol
    from superglm.reml.efs import optimize_efs_reml

    # efs.py stops on the per-step lambda-change bound -- a step engine.
    reml_tol = resolve_reml_tol(reml_tol, engine="step")
    result, dm = optimize_efs_reml(
        model._dm,
        model._distribution,
        model._link,
        model._groups,
        configured_penalty(model),
        model._active_set,
        y,
        sample_weight,
        offset_arr,
        reml_groups,
        penalty_ranks,
        lambdas,
        weight_semantics=model_weight_semantics(model),
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
