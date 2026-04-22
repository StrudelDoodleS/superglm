"""Execution helpers for the REML fitting path."""

from __future__ import annotations

import math
import time
from typing import Any

from superglm.distributions import clip_mu
from superglm.links import stabilize_eta
from superglm.model.reml_setup import promote_estimated_scop_lambdas
from superglm.solvers.irls_direct import fit_irls_direct


def _trace_rows_enabled(debug_recorder) -> bool:
    """Return whether detailed JSONL trace rows should be emitted."""
    return debug_recorder is not None and getattr(debug_recorder, "enabled_level", 0) >= 2


def _lambda_max_delta(lambda_history: list[dict[str, float]] | None) -> float:
    """Compute the max abs log-lambda delta between the last two snapshots."""
    if not lambda_history or len(lambda_history) < 2:
        return 0.0

    previous = lambda_history[-2]
    current = lambda_history[-1]
    deltas = [
        abs(math.log(current[name]) - math.log(previous[name]))
        for name in current
        if name in previous and current[name] > 0 and previous[name] > 0
    ]
    return max(deltas, default=0.0)


def _record_non_scop_reml_trace(best, debug_recorder) -> None:
    """Emit a compact REML summary row for non-SCOP helper paths."""
    if not _trace_rows_enabled(debug_recorder):
        return

    debug_recorder.append_jsonl(
        "reml",
        {
            "iteration": int(getattr(best, "n_reml_iter", 0) or 0),
            "objective_before": float(getattr(best, "objective", 0.0)),
            "objective_after": float(getattr(best, "objective", 0.0)),
            "lambda_max_delta": float(_lambda_max_delta(getattr(best, "lambda_history", None))),
            "converged": bool(getattr(best, "converged", False)),
            "path": "non_scop",
            "lambdas": {name: float(value) for name, value in getattr(best, "lambdas", {}).items()},
        },
    )


def _record_non_scop_pirls_trace(
    model,
    *,
    best,
    y,
    sample_weight,
    offset_arr,
    pirls_tol: float,
    debug_recorder,
    reml_penalties,
) -> None:
    """Replay one final direct IRLS iteration to emit PIRLS trace rows."""
    if not _trace_rows_enabled(debug_recorder):
        return

    best_result = getattr(best, "pirls_result", None)
    if best_result is None:
        return

    fit_irls_direct(
        X=model._dm,
        y=y,
        weights=sample_weight,
        family=model._distribution,
        link=model._link,
        groups=model._groups,
        lambda2=best.lambdas,
        offset=offset_arr,
        beta_init=best_result.beta,
        intercept_init=best_result.intercept,
        max_iter=1,
        tol=pirls_tol,
        convergence=model._convergence,
        reml_penalties=reml_penalties if reml_penalties else None,
        debug_recorder=debug_recorder,
        debug_context={
            "phase": "reml",
            "path": "non_scop",
            "reml_iteration": int(getattr(best, "n_reml_iter", 0) or 0),
            "trace_replay": True,
        },
    )


def run_fixed_constraint_reml(
    model,
    *,
    y,
    sample_weight,
    offset,
    pirls_tol: float,
    max_pirls_iter: int,
    lambdas: dict[str, float],
    reml_penalties: list[Any],
    compute_fit_stats,
    debug_recorder=None,
) -> None:
    """Run the fixed-lambda constrained REML path and update the model in place."""
    result, _ = fit_irls_direct(
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
        debug_recorder=debug_recorder,
        debug_context={"phase": "fixed_constraint"},
    )

    model._result = result
    model._reml_lambdas = lambdas
    model._reml_penalties = reml_penalties

    eta = model._dm.matvec(result.beta) + result.intercept
    if offset is not None:
        eta = eta + offset
    eta = stabilize_eta(eta, model._link)
    mu = clip_mu(model._link.inverse(eta), model._distribution)

    model._fit_stats = compute_fit_stats(
        y, mu, sample_weight, offset, model._distribution, model._link, result.phi
    )
    model._last_fit_meta = {"method": "fit_reml", "discrete": model._discrete}


def run_scop_efs_reml(
    model,
    *,
    y,
    sample_weight,
    offset,
    offset_arr,
    lambdas: dict[str, float],
    estimated_names: set[str],
    lam_init: float,
    reml_penalties: list[Any],
    max_reml_iter: int,
    reml_tol: float,
    pirls_tol: float,
    max_pirls_iter: int,
    verbose: bool,
    profile: dict[str, Any],
    total_start: float,
    compute_fit_stats,
    debug_recorder=None,
):
    """Run the SCOP-constrained EFS REML path and update the model in place."""
    promote_estimated_scop_lambdas(
        model._groups,
        model._specs,
        lambdas,
        estimated_names,
        lam_init,
    )

    from superglm.reml.scop_efs import optimize_scop_efs_reml

    best = optimize_scop_efs_reml(
        dm=model._dm,
        distribution=model._distribution,
        link=model._link,
        groups=model._groups,
        y=y,
        sample_weight=sample_weight,
        offset_arr=offset_arr,
        lambdas=lambdas,
        estimated_names=estimated_names,
        max_reml_iter=max_reml_iter,
        reml_tol=reml_tol,
        pirls_tol=pirls_tol,
        max_pirls_iter=max_pirls_iter,
        verbose=verbose,
        reml_penalties=reml_penalties,
        convergence=model._convergence,
        debug_recorder=debug_recorder,
    )

    model._result = best.pirls_result
    model._reml_lambdas = best.lambdas
    model._reml_penalties = best.reml_penalties if best.reml_penalties else reml_penalties
    model._reml_result = best

    eta = model._dm.matvec(best.pirls_result.beta) + best.pirls_result.intercept
    if offset is not None:
        eta = eta + offset
    eta = stabilize_eta(eta, model._link)
    mu = clip_mu(model._link.inverse(eta), model._distribution)

    model._fit_stats = compute_fit_stats(
        y,
        mu,
        sample_weight,
        offset,
        model._distribution,
        model._link,
        best.pirls_result.phi,
    )
    model._last_fit_meta = {"method": "fit_reml", "discrete": model._discrete}

    profile["total_s"] = time.perf_counter() - total_start
    profile["n_reml_iter"] = best.n_reml_iter
    profile["converged"] = best.converged
    model._reml_profile = profile
    return best


def optimize_reml_best(
    model,
    *,
    use_direct: bool,
    y,
    sample_weight,
    offset_arr,
    reml_groups,
    penalty_ranks,
    lambdas: dict[str, float],
    max_reml_iter: int,
    reml_tol: float,
    verbose: bool,
    penalty_caches,
    profile: dict[str, Any],
    w_correction_order: int,
    reml_penalties: list[Any],
    estimated_names: set[str],
    pirls_tol: float,
    max_pirls_iter: int,
    model_optimize_direct_reml,
    model_optimize_efs_reml,
    debug_recorder=None,
):
    """Run the appropriate REML optimizer and return its best result object."""
    if not estimated_names:
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
                reml_tol=1.0,
                verbose=verbose,
                penalty_caches=penalty_caches,
                profile=profile,
                w_correction_order=w_correction_order,
                reml_penalties=reml_penalties,
                estimated_names=estimated_names,
            )
            _record_non_scop_reml_trace(best, debug_recorder)
            _record_non_scop_pirls_trace(
                model,
                best=best,
                y=y,
                sample_weight=sample_weight,
                offset_arr=offset_arr,
                pirls_tol=pirls_tol,
                debug_recorder=debug_recorder,
                reml_penalties=reml_penalties,
            )
            return best
        return model_optimize_efs_reml(
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

    if use_direct:
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
            profile=profile,
            w_correction_order=w_correction_order,
            reml_penalties=reml_penalties,
            estimated_names=estimated_names,
        )
        _record_non_scop_reml_trace(best, debug_recorder)
        _record_non_scop_pirls_trace(
            model,
            best=best,
            y=y,
            sample_weight=sample_weight,
            offset_arr=offset_arr,
            pirls_tol=pirls_tol,
            debug_recorder=debug_recorder,
            reml_penalties=reml_penalties,
        )
        return best
    return model_optimize_efs_reml(
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
