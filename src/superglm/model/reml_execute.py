"""Execution helpers for the REML fitting path."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

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


def record_reml_terminal(model, debug_recorder) -> None:
    """Record success only after the fitted state has been publicly installed."""
    trace_run = getattr(debug_recorder, "trace_run", None)
    if trace_run is None or not trace_run.enabled:
        return

    solver_result = getattr(model, "_solver_result", None)
    reml_result = getattr(model, "_reml_result", None)
    state_id = getattr(solver_result, "state_id", None)
    objective = getattr(reml_result, "objective", None)
    if solver_result is None or reml_result is None or state_id is None or objective is None:
        # Fixed-constraint and SCOP compatibility paths do not yet allocate
        # canonical coefficient-state identities.  Their legacy rows remain
        # available, but they must not manufacture an authoritative terminal.
        return

    trace_run.emit_lazy(
        "terminal",
        lambda: {
            "state_id": state_id,
            "evaluation_id": getattr(solver_result, "evaluation_id", None),
            "solver": "fit_reml",
            "objective": float(objective),
            "lambdas": model._reml_lambdas,
            "dispersion": float(solver_result.phi),
            "effective_df": float(solver_result.effective_df),
            "fit_converged": bool(reml_result.converged),
            "outer_iterations": int(reml_result.n_reml_iter),
        },
        channel="reml",
        purpose="fit_reml",
    )


def run_fixed_monotone_reml(
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
    profile: dict[str, Any] | None = None,
    total_start: float | None = None,
    debug_recorder=None,
) -> Any:
    """Run the fixed-lambda monotone REML path and update the model in place."""
    has_scop = any(group.monotone_engine == "scop" for group in model._groups)
    best = None
    if has_scop:
        from superglm.reml.scop_efs import fit_fixed_scop_reml

        offset_arr = offset if offset is not None else np.zeros(len(y), dtype=np.float64)
        best = fit_fixed_scop_reml(
            model._dm,
            model._distribution,
            model._link,
            model._groups,
            y,
            sample_weight,
            offset_arr,
            lambdas,
            pirls_tol=pirls_tol,
            max_pirls_iter=max_pirls_iter,
            reml_penalties=reml_penalties if reml_penalties else None,
            convergence=model._convergence,
            debug_recorder=debug_recorder,
        )
        result = best.pirls_result
        lambdas = best.lambdas
        reml_penalties = best.reml_penalties or reml_penalties
        model._reml_result = best
    else:
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
        if result.termination_reason == "constraint_infeasible":
            raise RuntimeError(
                "fixed-lambda constrained REML fit ended at an infeasible coefficient mode"
            )
        if result.termination_reason == "constraint_kkt_incomplete":
            raise RuntimeError(
                "fixed-lambda constrained REML fit ended without a complete inner-QP "
                "KKT certificate"
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
    model._solver_result = result
    model._last_fit_meta = {
        "method": "fit_reml",
        "discrete": model._discrete,
        "lambda_strategy": "fixed",
    }
    if profile is not None:
        if total_start is not None:
            profile["total_s"] = time.perf_counter() - total_start
        profile["n_reml_iter"] = 0
        profile["converged"] = bool(result.converged)
        model._reml_profile = profile
    return best


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
    """Run the SCOP EFS REML path and update the model in place."""
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
    model._solver_result = best.pirls_result
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
    trace_run = getattr(debug_recorder, "trace_run", None)
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
                pirls_tol=pirls_tol,
                max_pirls_iter=max_pirls_iter,
                debug_recorder=debug_recorder,
                trace_run=trace_run,
            )
            _record_non_scop_reml_trace(best, debug_recorder)
            return best
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
        _record_non_scop_reml_trace(best, debug_recorder)
        return best

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
            pirls_tol=pirls_tol,
            max_pirls_iter=max_pirls_iter,
            debug_recorder=debug_recorder,
            trace_run=trace_run,
        )
        _record_non_scop_reml_trace(best, debug_recorder)
        return best
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
    _record_non_scop_reml_trace(best, debug_recorder)
    return best
