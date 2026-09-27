"""Timed end-to-end comparison of the Newton and bounded inner phi solves.

The public outer Brent search over the Tweedie power runs twice per case, once
with production's Newton dispersion solve (``solve_log_phi``) and once with
only that solve swapped for a value-only bounded minimization of the same
criterion. Both modes are warmed, then timed over counterbalanced repeats, and
the rows report medians. Integer and boolean fields must agree exactly across
repeats; ``inner_density_passes`` counts the compiled series passes the phi
solves make.

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

import superglm._tweedie as density_module
import superglm.model.profile_ops as profile_ops_module
import superglm.profiling.tweedie as tweedie_module
from superglm import SuperGLM, generate_tweedie_cpg
from superglm.distributions import Tweedie
from superglm.features.numeric import Numeric
from superglm.features.spline import Spline


@dataclass(frozen=True)
class _EndToEndProfileCase:
    name: str
    X: pd.DataFrame
    y: np.ndarray
    fit_mode: str
    p_bounds: tuple[float, float]
    xatol: float


def _bounded_inner_phi_reference(y, mu, weights, p, *, optimizer_successes):
    """Replace only the Newton phi solve with a bounded value-only minimization.

    It minimises the same criterion, Q(u) = D e^-u / 2 - l_sat(e^u) over
    u = log phi, through the same prepared rows, one series pass per value.
    """
    rows = density_module.TweedieRows.prepare(y, weights, p)
    deviance = float(np.sum(weights * density_module.tweedie_unit_deviance(y, mu, p)))

    def criterion(u):
        return 0.5 * deviance * np.exp(-u) - rows.saturated(float(np.exp(u)))[0]

    # A value-only search needs a finite window; these fits' log phi is order one.
    optimizer = minimize_scalar(
        criterion, bounds=(-45.0, 45.0), method="bounded", options={"xatol": 1e-9}
    )
    optimizer_successes.append(bool(optimizer.success))
    return density_module.PhiSolve(
        float(np.exp(optimizer.x)), float(optimizer.fun), float("nan"), int(optimizer.nfev)
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
        ),
        _EndToEndProfileCase(
            name="reml-spline",
            X=pd.DataFrame({"x": reml_x}),
            y=reml_y,
            fit_mode="reml",
            p_bounds=(1.35, 1.8),
            xatol=1e-2,
        ),
    )


def _run_end_to_end_profile_once(
    mode: str,
    case: _EndToEndProfileCase,
) -> dict[str, object]:
    """Run one public outer profile and count the phi solves' series passes."""
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
    real_series = density_module.series_moments
    local_optimizer_successes: list[bool] = []
    passes = {"inside_phi": 0, "phi_solves": 0, "active": False}

    def counted_series(log_t, a, **kwargs):
        passes["inside_phi"] += passes["active"]
        return real_series(log_t, a, **kwargs)

    solve = (
        partial(_bounded_inner_phi_reference, optimizer_successes=local_optimizer_successes)
        if mode == "reference-bounded-inner"
        else tweedie_module.profile_phi_at
    )

    def counted_solve(y, mu, weights, p):
        passes["active"] = True
        passes["phi_solves"] += 1
        try:
            return solve(y, mu, weights, p)
        finally:
            passes["active"] = False

    started = time.perf_counter()
    with ExitStack() as stack:
        stack.enter_context(patch.object(density_module, "series_moments", counted_series))
        # The search and the publication's re-profile each bind the solve by name.
        stack.enter_context(patch.object(tweedie_module, "profile_phi_at", counted_solve))
        stack.enter_context(patch.object(profile_ops_module, "profile_phi_at", counted_solve))
        result = model.estimate_p(
            case.X,
            case.y,
            p_bounds=case.p_bounds,
            xatol=case.xatol,
            fit_mode=case.fit_mode,
        )
    elapsed = time.perf_counter() - started

    # One solve per searched power, plus the publication's re-profile.
    assert passes["phi_solves"] == len(result.evaluations) + 1
    if mode == "reference-bounded-inner":
        assert len(local_optimizer_successes) == passes["phi_solves"]
        local_inner_optimizer_success = all(local_optimizer_successes)
    else:
        assert not local_optimizer_successes
        local_inner_optimizer_success = None
    return {
        "case": case.name,
        "fit_mode": case.fit_mode,
        "mode": mode,
        "n_observations": len(case.y),
        "outer_evaluations": len(result.evaluations),
        "inner_density_passes": passes["inside_phi"],
        "p_hat": float(result.p_hat),
        "phi_hat": float(result.phi_hat),
        "nll": float(result.nll),
        "elapsed_seconds": elapsed,
        "converged": bool(result.converged),
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

    integer_fields = ("n_observations", "outer_evaluations", "inner_density_passes")
    float_fields = ("p_hat", "phi_hat", "nll")
    common_integers = {}
    for field in integer_fields:
        values = {int(run[field]) for run in runs}
        if len(values) != 1:
            raise AssertionError(f"non-deterministic {field} across repeated runs: {values}")
        common_integers[field] = values.pop()

    boolean_fields = ("converged", "local_inner_optimizer_success")
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
    """Warm, counterbalance, and compare the Newton and bounded inner phi solves."""
    timed_orders = _counterbalanced_mode_orders(repeats)

    original_solve = tweedie_module.profile_phi_at
    original_series = density_module.series_moments
    rows = []
    for case in _end_to_end_profile_cases():
        runs_by_mode = {
            "production-analytic-inner": [],
            "reference-bounded-inner": [],
        }
        # Warm both complete public paths, but exclude warm-up timings from medians.
        for mode in runs_by_mode:
            _run_end_to_end_profile_once(mode, case)
            assert tweedie_module.profile_phi_at is original_solve
            assert density_module.series_moments is original_series

        for order in timed_orders:
            for mode in order:
                runs_by_mode[mode].append(_run_end_to_end_profile_once(mode, case))
                assert tweedie_module.profile_phi_at is original_solve
                assert density_module.series_moments is original_series
        rows.extend(_aggregate_end_to_end_runs(runs) for runs in runs_by_mode.values())
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=4)
    args = parser.parse_args()
    rows = run_end_to_end_profile_benchmark(repeats=args.repeats)

    print(
        "Reference change: replace only the production Newton phi solve with value-only "
        "bounded minimization of the same criterion; retain the outer Brent and fit semantics."
    )
    for row in rows:
        print(
            "Tweedie end-to-end profile benchmark "
            f"case={row['case']} fit_mode={row['fit_mode']} mode={row['mode']} "
            f"repeats={row['repeats']} n={row['n_observations']} "
            f"outer_evaluations={row['outer_evaluations']} "
            f"inner_density_passes={row['inner_density_passes']} "
            f"p_hat={row['p_hat']:.8g} phi_hat={row['phi_hat']:.8g} "
            f"NLL={row['nll']:.8g} "
            f"local_inner_optimizer_success={row['local_inner_optimizer_success']} "
            f"converged={row['converged']} "
            f"elapsed_median_s={row['elapsed_median_seconds']:.6f}"
        )


if __name__ == "__main__":
    main()
