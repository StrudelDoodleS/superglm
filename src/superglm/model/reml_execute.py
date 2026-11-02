"""Execution helpers for the REML fitting path."""

from __future__ import annotations

import math
import time
from dataclasses import replace
from typing import Any

from superglm.distributions import Tweedie, clip_mu
from superglm.links import stabilize_eta
from superglm.model.reml_finalize import compute_profiled_phi
from superglm.reml.observed_geometry import ObservedModeNotConvergedError
from superglm.solvers.dispersion import model_weight_semantics
from superglm.solvers.irls_direct import fit_irls_direct
from superglm.solvers.mode_score import linear_predictor

# The Newton engines (exact and discrete cached-W) stop on Wood's compound
# bar, ``reml_tol * (1 + |objective|)``, which scales with the objective's
# magnitude: at 1e-6 a fit on a flat log-lambda direction stops with
# converged=True while published standard errors are still tens of percent
# from the determined answer; 1e-9 pins them to ~0.01%, measured at 3->7
# Newton iterations on a 97k benchmark fixture and 5->13 on the 12k stress
# design.
NEWTON_REML_TOL_DEFAULT = 1e-9

# A Newton engine converges quadratically near the optimum (8 outer
# iterations on the 678k-row freMTPL2 frequency fit).
NEWTON_MAX_REML_ITER_DEFAULT = 20


def resolve_reml_tol(reml_tol: float | None) -> float:
    """Resolve the public ``reml_tol`` sentinel to the Newton engines' default."""
    return NEWTON_REML_TOL_DEFAULT if reml_tol is None else float(reml_tol)


def resolve_max_reml_iter(max_reml_iter: int | None) -> int:
    """Resolve the public ``max_reml_iter`` sentinel to the Newton engines' default."""
    return NEWTON_MAX_REML_ITER_DEFAULT if max_reml_iter is None else int(max_reml_iter)


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


def _record_reml_trace(best, debug_recorder) -> None:
    """Emit a compact REML summary row."""
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
            "path": "newton",
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
        # The fixed-constraint path does not yet allocate canonical
        # coefficient-state identities.  Their legacy rows remain
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
) -> None:
    """Run the fixed-lambda monotone REML path and update the model in place."""
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
        weight_semantics=model_weight_semantics(model),
    )
    # Typed like the observed-geometry gates: to a power search a QP
    # candidate with no feasible mode (or no complete KKT certificate) is an
    # infeasible point to route around, not a dead search. Bare RuntimeError
    # stays reserved for invariant violations.
    if result.termination_reason == "constraint_infeasible":
        raise ObservedModeNotConvergedError(
            "fixed-lambda constrained REML fit ended at an infeasible coefficient mode"
        )
    if result.termination_reason == "constraint_kkt_incomplete":
        raise ObservedModeNotConvergedError(
            "fixed-lambda constrained REML fit ended without a complete inner-QP KKT certificate"
        )

    # Publish the scale the unconstrained fixed-lambda REML route publishes
    # (``finalize_reml_fit``): the REML profile of phi at the penalized mode
    # (Wood 2011, JRSS-B 73(1); Wood 2017, GAMs 2nd ed., section 6.2) for
    # Gaussian and Gamma, the deviance form over the penalty nullity for the
    # other estimated-scale families, and the solver's Pearson value for
    # Tweedie, as that route keeps it.
    if not isinstance(model._distribution, Tweedie):
        result = replace(
            result,
            phi=compute_profiled_phi(
                model,
                y=y,
                sample_weight=sample_weight,
                lambdas=lambdas,
                reml_penalties=reml_penalties,
                pirls_result=result,
            ),
        )

    model._result = result
    model._reml_lambdas = lambdas
    model._reml_penalties = reml_penalties

    eta = stabilize_eta(linear_predictor(model._dm, result, offset), model._link)
    mu = clip_mu(model._link.inverse(eta), model._distribution)

    model._fit_stats = compute_fit_stats(
        y,
        mu,
        sample_weight,
        offset,
        model._distribution,
        model._link,
        result.phi,
        weight_semantics=model_weight_semantics(model),
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


def optimize_reml_best(
    model,
    *,
    y,
    sample_weight,
    offset_arr,
    reml_groups,
    penalty_ranks,
    lambdas: dict[str, float],
    max_reml_iter: int,
    reml_tol: float | None,
    verbose: bool,
    penalty_caches,
    profile: dict[str, Any],
    w_correction_order: int,
    reml_penalties: list[Any],
    estimated_names: set[str],
    pirls_tol: float,
    max_pirls_iter: int,
    model_optimize_direct_reml,
    debug_recorder=None,
    warm_lambdas: dict[str, float] | None = None,
):
    """Run the REML optimizer and return its best result object."""
    reml_tol = resolve_reml_tol(reml_tol)
    if profile is not None:
        profile["reml_tol_resolved"] = float(reml_tol)
        profile["reml_warm_start_components"] = sorted(warm_lambdas or ())
    trace_run = getattr(debug_recorder, "trace_run", None)
    fixed = not estimated_names
    best = model_optimize_direct_reml(
        model,
        y,
        sample_weight,
        offset_arr,
        reml_groups,
        penalty_ranks,
        lambdas,
        # every lambda fixed: one evaluation, whose modes certify at
        # the resolved tolerance's bar (``mode_score``)
        max_reml_iter=1 if fixed else max_reml_iter,
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
        warm_lambdas=None if fixed else warm_lambdas or None,
    )
    _record_reml_trace(best, debug_recorder)
    return best
