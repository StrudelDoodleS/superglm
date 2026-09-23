"""Timed end-to-end comparison of the analytic and bounded inner phi searches.

The public outer Brent search over the Tweedie power runs twice per case, once
with production's analytic inner dispersion search and once with only that
inner search swapped for a value-only bounded minimization. Both modes are
warmed, then timed over counterbalanced repeats, and the rows report medians.
Integer and boolean fields must agree exactly across repeats.

Run on a quiet machine with the thread pools pinned:

    uv run python benchmarks/tweedie_profile_end_to_end.py --repeats 4

That the two modes agree is a correctness claim, and
``tests/test_tweedie_profile_performance.py`` asserts it on one run per case
and mode. This script only adds the timing, which a test must not assert.
"""

from __future__ import annotations

import argparse
import statistics
import time
from contextlib import ExitStack
from dataclasses import dataclass
from functools import partial
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

import superglm.profiling.tweedie as tweedie_module
from superglm import SuperGLM
from superglm.distributions import Tweedie
from superglm.features.numeric import Numeric
from superglm.features.spline import Spline
from superglm.profiling.tweedie import generate_tweedie_cpg


@dataclass(frozen=True)
class _EndToEndProfileCase:
    name: str
    X: pd.DataFrame
    y: np.ndarray
    fit_mode: str
    p_bounds: tuple[float, float]
    xatol: float
    maxiter: int


def _bounded_inner_phi_reference(
    y,
    mu,
    p,
    *,
    weights=None,
    df_resid=None,
    phi_method="mle",
    phi_start=None,
    optimizer_successes=None,
):
    """Replace only analytic inner search with bounded value-only minimization.

    The reference retains production input preparation, vector-density evaluation,
    hard log-phi bounds, and bounded-fallback tolerance. ``df_resid`` and
    ``phi_start`` are accepted for signature equivalence but are not inputs to an
    exact MLE objective.
    """
    del df_resid, phi_start
    if phi_method != "mle":
        raise AssertionError("the end-to-end bounded reference requires phi_method='mle'")

    prepared = tweedie_module._prepare_tweedie_density(y, mu, p, weights=weights)
    cache: dict[float, tuple[float, object]] = {}

    def objective(log_phi):
        key = float(log_phi)
        cached = cache.get(key)
        if cached is not None:
            return cached[0]
        evaluation = tweedie_module._evaluate_tweedie_density(
            prepared,
            float(np.exp(key)),
            compute_score=False,
        )
        nll = -float(np.mean(evaluation.logpdf))
        cache[key] = (nll, evaluation)
        return nll

    optimizer = minimize_scalar(
        objective,
        bounds=(tweedie_module._LOG_PHI_LOWER_BOUND, tweedie_module._LOG_PHI_UPPER_BOUND),
        method="bounded",
        options={"xatol": tweedie_module._PHI_BOUNDED_XATOL, "maxiter": 200},
    )
    local_optimizer_success = bool(optimizer.success)
    if optimizer_successes is not None:
        optimizer_successes.append(local_optimizer_success)
    log_phi = float(optimizer.x)
    nll = float(objective(log_phi))
    diagnostics = cache[log_phi][1].diagnostics
    objective_finite = bool(np.isfinite(nll) and np.isfinite(log_phi))
    boundary_tolerance = 4.0 * tweedie_module._PHI_BOUNDED_XATOL
    lower_boundary = bool(log_phi - tweedie_module._LOG_PHI_LOWER_BOUND <= boundary_tolerance)
    upper_boundary = bool(tweedie_module._LOG_PHI_UPPER_BOUND - log_phi <= boundary_tolerance)
    branch_signatures = {item[1].positive_saddlepoint_mask.tobytes() for item in cache.values()}
    branch_switch_detected = len(branch_signatures) > 1
    local_status = "succeeded" if local_optimizer_success else "failed"
    return tweedie_module._PhiProfileResult(
        phi=float(np.exp(log_phi)),
        nll=nll,
        # SciPy success is only local; one value-only bounded search does not
        # certify the global phi optimum, especially across branch switches.
        converged=False,
        objective_finite=objective_finite,
        n_evaluations=len(cache),
        n_score_evaluations=0,
        n_value_only_evaluations=len(cache),
        n_fallback_evaluations=0,
        optimizer="bounded-reference",
        score=None,
        used_fallback=False,
        fallback_reason=None,
        branch_switch_detected=branch_switch_detected,
        lower_boundary=lower_boundary,
        upper_boundary=upper_boundary,
        diagnostics=diagnostics,
        message=(
            f"Local derivative-free bounded minimization {local_status}; the test "
            "reference does not certify global phi convergence."
            + (" Density branch signatures changed." if branch_switch_detected else "")
        ),
    )


def _end_to_end_profile_cases() -> tuple[_EndToEndProfileCase, ...]:
    """Return deterministic ordinary-fit and REML-spline profile cases."""
    numeric_rng = np.random.default_rng(20260718)
    numeric_x = np.linspace(-1.5, 1.5, 600)
    numeric_mu = np.exp(1.1 + 0.35 * numeric_x)
    numeric_y = generate_tweedie_cpg(
        len(numeric_x),
        mu=numeric_mu,
        phi=2.5,
        p=1.6,
        rng=numeric_rng,
    )

    reml_rng = np.random.default_rng(20260719)
    reml_x = np.linspace(-1.5, 1.5, 300)
    reml_mu = np.exp(1.0 + 0.35 * reml_x + 0.2 * np.sin(np.pi * reml_x))
    reml_y = generate_tweedie_cpg(
        len(reml_x),
        mu=reml_mu,
        phi=2.5,
        p=1.6,
        rng=reml_rng,
    )
    return (
        _EndToEndProfileCase(
            name="fit-numeric",
            X=pd.DataFrame({"x": numeric_x}),
            y=numeric_y,
            fit_mode="fit",
            p_bounds=(1.3, 1.85),
            xatol=5e-3,
            maxiter=20,
        ),
        _EndToEndProfileCase(
            name="reml-spline",
            X=pd.DataFrame({"x": reml_x}),
            y=reml_y,
            fit_mode="reml",
            p_bounds=(1.35, 1.8),
            xatol=1e-2,
            maxiter=15,
        ),
    )


def _run_end_to_end_profile_once(
    mode: str,
    case: _EndToEndProfileCase,
) -> dict[str, object]:
    """Run one public outer profile and count real inner density passes."""
    if mode not in {"production-analytic-inner", "reference-bounded-inner"}:
        raise ValueError(f"unknown end-to-end benchmark mode: {mode}")

    if case.name == "fit-numeric":
        feature = Numeric()
    elif case.name == "reml-spline":
        feature = Spline(n_knots=6, penalty="ssp")
    else:
        raise ValueError(f"unknown end-to-end benchmark case: {case.name}")
    model = SuperGLM(
        family=Tweedie(p=1.5),
        selection_penalty=0,
        features={"x": feature},
    )
    real_evaluate = tweedie_module._evaluate_tweedie_density
    real_profile = tweedie_module._profile_phi_detailed
    density_calls: list[bool] = []
    inner_density_passes = 0
    local_optimizer_successes: list[bool] = []

    def counted_evaluate(prepared, phi, *, compute_score=False, **kwargs):
        density_calls.append(bool(compute_score))
        return real_evaluate(prepared, phi, compute_score=compute_score, **kwargs)

    profile_target = (
        partial(
            _bounded_inner_phi_reference,
            optimizer_successes=local_optimizer_successes,
        )
        if mode == "reference-bounded-inner"
        else real_profile
    )

    def counted_profile(*args, **kwargs):
        nonlocal inner_density_passes
        before = len(density_calls)
        result = profile_target(*args, **kwargs)
        inner_density_passes += len(density_calls) - before
        return result

    started = time.perf_counter()
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(tweedie_module, "_evaluate_tweedie_density", counted_evaluate)
        )
        stack.enter_context(patch.object(tweedie_module, "_profile_phi_detailed", counted_profile))
        result = model.estimate_p(
            case.X,
            case.y,
            p_bounds=case.p_bounds,
            xatol=case.xatol,
            maxiter=case.maxiter,
            fit_mode=case.fit_mode,
            phi_method="mle",
            method="brent",
        )
    elapsed = time.perf_counter() - started

    trace_density_passes = int(result.search_trace["phi_n_evaluations"].sum())
    # Publication re-profiles dispersion against the published refit, so the
    # instrumented count exceeds the search trace by exactly those passes:
    # result-level counters hold the winning evaluation plus the reprofile.
    winning_idx = (result.search_trace["p"] - float(result.p_hat)).abs().idxmin()
    winning_passes = int(result.search_trace.loc[winning_idx, "phi_n_evaluations"])
    publication_reprofile_passes = int(result.phi_n_evaluations) - winning_passes
    assert publication_reprofile_passes >= 0
    assert trace_density_passes + publication_reprofile_passes == inner_density_passes
    assert len(density_calls) >= inner_density_passes
    if mode == "reference-bounded-inner":
        # One inner solve per search evaluation, plus exactly one more for the
        # publication's dispersion re-profile against the published refit.
        assert len(local_optimizer_successes) == result.n_evaluations + 1
        assert not result.search_trace["phi_converged"].any()
        local_inner_optimizer_success = all(local_optimizer_successes)
    else:
        assert not local_optimizer_successes
        assert result.search_trace["phi_converged"].all()
        local_inner_optimizer_success = None
    return {
        "case": case.name,
        "fit_mode": case.fit_mode,
        "mode": mode,
        "n_observations": len(case.y),
        "outer_evaluations": int(result.n_evaluations),
        "inner_density_passes": inner_density_passes,
        "p_hat": float(result.p_hat),
        "phi_hat": float(result.phi_hat),
        "nll": float(result.nll),
        "saddle_fraction": float(result.saddlepoint_fraction),
        "phi_fallback_count": int(result.search_trace["phi_n_fallback_evaluations"].sum()),
        "elapsed_seconds": elapsed,
        "converged": bool(result.converged),
        "outer_converged": bool(result.outer_converged),
        "objective_finite": bool(result.objective_finite),
        "local_inner_optimizer_success": local_inner_optimizer_success,
    }


def _aggregate_end_to_end_runs(runs: list[dict[str, object]]) -> dict[str, object]:
    """Median floats after requiring deterministic integer and boolean fields."""
    if not runs:
        raise ValueError("end-to-end benchmark requires at least one run")
    mode = runs[0]["mode"]
    case = runs[0]["case"]
    fit_mode = runs[0]["fit_mode"]
    if any(run["mode"] != mode for run in runs):
        raise ValueError("cannot aggregate mixed end-to-end benchmark modes")
    if any(run["case"] != case or run["fit_mode"] != fit_mode for run in runs):
        raise ValueError("cannot aggregate mixed end-to-end benchmark cases")

    integer_fields = (
        "n_observations",
        "outer_evaluations",
        "inner_density_passes",
        "phi_fallback_count",
    )
    float_fields = ("p_hat", "phi_hat", "nll", "saddle_fraction")
    common_integers = {}
    for field in integer_fields:
        values = {int(run[field]) for run in runs}
        if len(values) != 1:
            raise AssertionError(f"non-deterministic {field} across repeated runs: {values}")
        common_integers[field] = values.pop()

    boolean_fields = (
        "converged",
        "outer_converged",
        "objective_finite",
        "local_inner_optimizer_success",
    )
    common_booleans = {}
    for field in boolean_fields:
        values = {run[field] for run in runs}
        if len(values) != 1:
            raise AssertionError(f"non-deterministic {field} across repeated runs: {values}")
        common_booleans[field] = values.pop()

    row: dict[str, object] = {
        "case": case,
        "fit_mode": fit_mode,
        "mode": mode,
        "repeats": len(runs),
        "elapsed_median_seconds": statistics.median(float(run["elapsed_seconds"]) for run in runs),
        **common_integers,
        **common_booleans,
    }
    row.update(
        {field: statistics.median(float(run[field]) for run in runs) for field in float_fields}
    )
    return row


def _counterbalanced_mode_orders(repeats: int) -> tuple[tuple[str, str], ...]:
    """Alternate which inner-profile mode runs first across timed repeats."""
    if repeats < 4 or repeats % 2:
        raise ValueError("end-to-end benchmark requires an even number of repeats, at least four")
    production = "production-analytic-inner"
    reference = "reference-bounded-inner"
    return tuple(
        (production, reference) if index % 2 == 0 else (reference, production)
        for index in range(repeats)
    )


def run_end_to_end_profile_benchmark(*, repeats: int = 4) -> list[dict[str, object]]:
    """Warm, counterbalance, and compare analytic and bounded inner phi searches."""
    timed_orders = _counterbalanced_mode_orders(repeats)

    original_profile = tweedie_module._profile_phi_detailed
    original_evaluate = tweedie_module._evaluate_tweedie_density
    rows = []
    for case in _end_to_end_profile_cases():
        runs_by_mode = {
            "production-analytic-inner": [],
            "reference-bounded-inner": [],
        }
        # Warm both complete public paths, but exclude warm-up timings from medians.
        for mode in runs_by_mode:
            _run_end_to_end_profile_once(mode, case)
            assert tweedie_module._profile_phi_detailed is original_profile
            assert tweedie_module._evaluate_tweedie_density is original_evaluate

        for order in timed_orders:
            for mode in order:
                runs_by_mode[mode].append(_run_end_to_end_profile_once(mode, case))
                assert tweedie_module._profile_phi_detailed is original_profile
                assert tweedie_module._evaluate_tweedie_density is original_evaluate
        rows.extend(_aggregate_end_to_end_runs(runs) for runs in runs_by_mode.values())
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=4)
    args = parser.parse_args()
    rows = run_end_to_end_profile_benchmark(repeats=args.repeats)

    print(
        "Reference change: replace only the production analytic inner phi-score search "
        "with value-only bounded minimization; retain public outer Brent and fit semantics."
    )
    for row in rows:
        print(
            "Tweedie end-to-end profile benchmark "
            f"case={row['case']} fit_mode={row['fit_mode']} mode={row['mode']} "
            f"repeats={row['repeats']} n={row['n_observations']} "
            f"outer_evaluations={row['outer_evaluations']} "
            f"inner_density_passes={row['inner_density_passes']} "
            f"p_hat={row['p_hat']:.8g} phi_hat={row['phi_hat']:.8g} "
            f"NLL={row['nll']:.8g} saddle_fraction={row['saddle_fraction']:.6f} "
            f"phi_fallback_count={row['phi_fallback_count']} "
            f"local_inner_optimizer_success={row['local_inner_optimizer_success']} "
            f"certified_converged={row['converged']} "
            f"outer_converged={row['outer_converged']} "
            f"objective_finite={row['objective_finite']} "
            f"elapsed_median_s={row['elapsed_median_seconds']:.6f}"
        )


if __name__ == "__main__":
    main()
