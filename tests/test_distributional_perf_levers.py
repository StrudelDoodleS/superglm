from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import ThreadpoolController

import superglm.distributional.solver.assembly as assembly_module
import superglm.distributional.solver.solver as solver_module
from superglm import SuperLSS
from superglm.distributional import GaussianLS, Predictor
from superglm.distributional.families.gaussian import GaussianLS as _GaussianLS
from superglm.distributional.timing import PHASE_NAMES, FitPhaseRecorder
from superglm.features import Spline


def _blas_threads() -> list[int]:
    return [
        info["num_threads"]
        for info in ThreadpoolController().info()
        if info.get("user_api") == "blas"
    ]


class _ProbeGaussian(_GaussianLS):
    """Records the BLAS pool size seen inside the first likelihood evaluation."""

    observed: list[list[int]] = []

    def evaluate_natural(self, y, theta, plan, *, derivative_order=2):
        if not type(self).observed:
            type(self).observed.append(_blas_threads())
        return super().evaluate_natural(y, theta, plan, derivative_order=derivative_order)


def _fixture(n: int, seed: int = 3):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    z = rng.uniform(-1.0, 1.0, n)
    y = 0.3 + 0.8 * np.sin(np.pi * x) + rng.normal(scale=np.exp(-0.4 + 0.3 * z), size=n)
    return pd.DataFrame({"x": x, "z": z}), y


def _predictors(location_k: int = 12, scale_k: int = 8):
    return (
        Predictor(
            "location",
            {"x": Spline(kind="cr", k=location_k), "z": Spline(kind="cr", k=location_k)},
        ),
        Predictor("scale", {"z": Spline(kind="cr", k=scale_k)}),
    )


# Block widths [119, 30] (measured): assembly work 7.5e7 at 2,000 rows (still
# capped) and 2.2e9 at 60,000 rows (released).  The default k=12/8 widths
# [23, 8] reach only 9.3e7 at 60,000 rows and would never cross the threshold.
@pytest.mark.parametrize(("n", "released"), [(2_000, False), (60_000, True)])
def test_fit_releases_blas_only_for_large_row_space_work(monkeypatch, n: int, released: bool):
    monkeypatch.delenv("SUPERGLM_BLAS_THREADS", raising=False)
    native = _blas_threads()
    if not native or max(native) < 2:
        pytest.skip("BLAS pool has a single thread; a release is unobservable")
    _ProbeGaussian.observed = []
    frame, y = _fixture(n)
    SuperLSS(
        family=_ProbeGaussian(),
        predictors=_predictors(location_k=60, scale_k=30),
    ).fit(
        frame,
        y,
        lambdas={"location:x#wiggle": 1.0, "location:z#wiggle": 1.0, "scale:z#wiggle": 1.0},
    )
    seen = _ProbeGaussian.observed[0]
    assert (seen == native) is released
    assert _blas_threads() == native


def test_dense_matrices_are_built_once_per_layout(monkeypatch):
    calls: list[int] = []
    original = assembly_module.dense_predictor_matrices

    def counting(layout):
        calls.append(id(layout))
        return original(layout)

    monkeypatch.setattr(solver_module, "dense_predictor_matrices", counting)
    frame, y = _fixture(3_000)
    model = SuperLSS(family=GaussianLS(), predictors=_predictors())
    recorder = FitPhaseRecorder()
    model.fit_reml(frame, y, practical_reml=False, phase_recorder=recorder)
    fitted = model._require_fitted()
    smoothing = fitted.smoothing
    assert len(smoothing.coefficient_fits) >= 3
    assert len(calls) == len(set(calls))  # once per distinct layout
    assert len(calls) <= 2  # the fitted layout, plus the null model's own layout
    layout = fitted.fit_state.layout
    assert calls.count(id(layout)) == 1
    snapshot = recorder.snapshot()
    assert "dense_predictor_matrices" in PHASE_NAMES
    # The null model fits without a recorder, as it does for every other phase,
    # so the phase sees exactly the builds made inside the recorded fit.
    assert snapshot.counts["dense_predictor_matrices"] == calls.count(id(layout))


def test_memoised_matrices_do_not_change_the_fit(monkeypatch):
    frame, y = _fixture(3_000, seed=8)

    def run() -> tuple[np.ndarray, float]:
        model = SuperLSS(family=GaussianLS(), predictors=_predictors())
        model.fit_reml(frame, y, practical_reml=False)
        smoothing = model._require_fitted().smoothing
        return np.array(list(model.coef_.values())), float(smoothing.objective)

    memo_coef, memo_objective = run()
    monkeypatch.setattr(
        solver_module._DenseObservedReuseSession,
        "dense_matrices",
        lambda self, layout, **kwargs: solver_module.dense_predictor_matrices(layout),
    )
    plain_coef, plain_objective = run()
    assert memo_objective == plain_objective
    assert np.array_equal(memo_coef, plain_coef)


def test_failed_terminal_retry_on_a_reused_endpoint_skips_fisher_geometry(monkeypatch):
    """A reused endpoint carries no row derivatives.

    Two injected conditions exercise the failure path: the terminal curvature
    policy demands a retry on such an endpoint, and the mandatory tighter solve
    fails, so the pre-retry state survives without derivatives.
    The Fisher sibling of the guarded retry must then skip Fisher geometry
    instead of asking the derivative-less state for it.
    """
    import dataclasses

    from superglm.distributional.model import fit_dense_distributional
    from superglm.distributional.result import DenseSolverConfig, DistributionalEFSConfig
    from superglm.distributional.weights import WeightContract
    from superglm.types import LambdaPolicy

    seen = {"derivative_less_endpoint": False, "armed": False, "injected": 0}
    original_run = solver_module._run_iterations

    def run_or_fail(context, state, config, **kwargs):
        if seen["armed"]:
            seen["armed"] = False
            raise solver_module.DenseSolverError("injected retry failure")
        result = original_run(context, state, config, **kwargs)
        seen["derivative_less_endpoint"] = result.state.derivatives is None
        return result

    monkeypatch.setattr(solver_module, "_run_iterations", run_or_fail)
    original_resolve = solver_module.resolve_curvature

    def demand_one_retry(*args, **kwargs):
        decision = original_resolve(*args, **kwargs)
        if (
            not decision.retry_required
            and seen["injected"] == 0
            and seen["derivative_less_endpoint"]
        ):
            seen["injected"] += 1
            seen["armed"] = True
            return solver_module.CurvatureDecision(
                matrix=None,
                decomposition=None,
                telemetry=dataclasses.replace(
                    decision.telemetry,
                    reason="material_indefiniteness_retry_required",
                ),
                retry_required=True,
                state=decision.state,
            )
        return decision

    monkeypatch.setattr(solver_module, "resolve_curvature", demand_one_retry)

    n = 800
    rng = np.random.default_rng(11)
    x = rng.permutation(np.linspace(-1.0, 1.0, n))
    z = rng.permutation(np.linspace(-1.0, 1.0, n))
    y = 0.8 * np.sin(np.pi * x) + rng.normal(scale=np.exp(-1.0 + 0.4 * np.cos(np.pi * z)))
    frame = pd.DataFrame({"x": x, "z": z})

    def spline():
        return Spline(
            kind="cr",
            n_knots=8,
            knot_strategy="quantile_rows",
            lambda_policy=LambdaPolicy.estimate(),
        )

    model = fit_dense_distributional(
        frame,
        y,
        family=GaussianLS(scale_floor=1.0e-4),
        predictors=(Predictor("location", {"x": spline()}), Predictor("scale", {"z": spline()})),
        weight_contract=WeightContract(semantics="prior"),
        lambdas=None,
        config=DenseSolverConfig(coefficient_curvature="observed", tolerance=1.0e-2),
        efs_config=DistributionalEFSConfig(max_iterations=30),
        retain_rows=False,
    )

    assert seen["injected"] == 1
    assert model.smoothing is not None
