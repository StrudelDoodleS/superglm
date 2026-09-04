from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import superglm.distributional.efs as efs_module
from superglm import SuperLSS
from superglm.distributional import GaussianLS, Predictor
from superglm.distributional import fit_diagnostics as diagnostics_module
from superglm.distributional.fit_diagnostics import diagnose_distributional_fit
from superglm.distributional.result import DistributionalEFSConfig
from superglm.features import RandomEffect, Spline


@pytest.mark.parametrize(
    "old_value", [49.99, 3.7, 1.0e-3, 1.0e9, 9.999999999e9, 0.125, 12.0, 2.0e9]
)
def test_scaled_proposal_lands_exactly_on_the_maximum(old_value: float) -> None:
    config = DistributionalEFSConfig(maximum_lambda=1.0e10, minimum_lambda=1.0e-6)
    cap = config.maximum_lambda if old_value > 1.0 else 50.0
    config = DistributionalEFSConfig(maximum_lambda=cap, minimum_lambda=1.0e-6)
    step = math.log(cap) - math.log(old_value)
    lambdas, log_steps = efs_module._scaled_proposal(
        {"a": old_value},
        {"a": step},
        ("a",),
        1.0,
        config,
    )
    assert lambdas["a"] == cap
    assert log_steps["a"] == math.log(cap) - math.log(old_value)


def test_scaled_proposal_lands_exactly_on_the_minimum() -> None:
    config = DistributionalEFSConfig(maximum_lambda=1.0e10, minimum_lambda=1.0e-6)
    old_value = 2.5e-6
    step = math.log(config.minimum_lambda) - math.log(old_value)
    lambdas, _ = efs_module._scaled_proposal({"a": old_value}, {"a": step}, ("a",), 1.0, config)
    assert lambdas["a"] == config.minimum_lambda


def test_accelerated_proposal_snaps_a_near_cap_log_value() -> None:
    config = DistributionalEFSConfig(maximum_lambda=50.0, minimum_lambda=1.0e-6)
    log_value = math.log(50.0) - 1.0e-15
    result = efs_module._accelerated_proposal(
        {"a": 40.0},
        ("a",),
        np.array([log_value]),
        np.array([log_value - math.log(40.0)]),
        config,
    )
    assert result is not None
    lambdas, _ = result
    assert lambdas["a"] == 50.0


def _noise_random_effect_fit(*, practical: bool, max_lambda: float, levels: int = 20):
    rng = np.random.default_rng(3)
    labels = np.repeat(np.array([f"l{i}" for i in range(levels)]), 20)
    y = rng.normal(size=len(labels))
    frame = pd.DataFrame({"effect": labels})
    model = SuperLSS(
        family=GaussianLS(scale_floor=1.0e-4),
        predictors=(Predictor("location", {"effect": RandomEffect()}), Predictor("scale", {})),
    )
    model.fit_reml(frame, y, max_lambda=max_lambda, practical_reml=practical)
    return model, model._require_fitted().smoothing


def test_practical_window_replays_non_increasing_steps() -> None:
    _model, smoothing = _noise_random_effect_fit(practical=True, max_lambda=1.0e10)
    if smoothing.convergence_reason != "practical_plateau":
        pytest.skip("fixture did not stop on a practical plateau")
    window = smoothing.history[-smoothing.config.plateau_iterations :]
    steps = [item.max_accepted_log_step for item in window]
    assert steps == sorted(steps, reverse=True)
    forged = list(smoothing.history)
    forged[-1] = replace(forged[-1], max_accepted_log_step=steps[-2] * 2.0 + 1.0e-3)
    with pytest.raises(ValueError, match="plateau gate"):
        replace(smoothing, history=tuple(forged))


def test_lower_bound_pressure_names_components_pinned_at_the_minimum() -> None:
    config = DistributionalEFSConfig(minimum_lambda=1.0e-6, tolerance=1.0e-6)
    evidence = efs_module._FreshRawEvidence(
        components=(),
        estimated_names=("a", "b"),
        update=None,
        maximum=0.0,
        working_infinity=(),
        unresolved_upper_bound=(),
    )
    lambdas = {"a": 1.0e-6, "b": 0.5}
    raw = {"a": -0.3, "b": -0.3}
    assert efs_module._lower_bound_pressure(evidence, lambdas, raw, config) == ("a",)
    raw_inward = {"a": 0.2, "b": -0.3}
    assert efs_module._lower_bound_pressure(evidence, lambdas, raw_inward, config) == ()


def _preempt_fixture():
    rng = np.random.default_rng(7)
    labels = np.repeat(np.array(["a", "b", "c", "d"]), 10)
    y = rng.normal(size=len(labels))
    return pd.DataFrame({"effect": labels}), y


def _preempt_fit(practical: bool):
    frame, y = _preempt_fixture()
    model = SuperLSS(
        family=GaussianLS(scale_floor=1.0e-4),
        predictors=(Predictor("location", {"effect": RandomEffect()}), Predictor("scale", {})),
    )
    model.fit_reml(
        frame,
        y,
        lambdas={"location:effect#wiggle": 1000.0},
        max_lambda=1002.5,
        max_log_step=1.0e-3,
        max_reml_iter=10,
        reml_tol=1.0e-8,
        inner_tol=1.0e-10,
        reml_plateau_tol=1.0e-6,
        practical_reml=practical,
    )
    return model, model._require_fitted().smoothing


def test_practical_stop_does_not_preempt_the_exact_face() -> None:
    practical_model, practical = _preempt_fit(True)
    strict_model, strict = _preempt_fit(False)
    assert strict.convergence_reason == "lambda_change"
    assert strict_model.exact_face_components_ == ("location:effect#wiggle",)
    assert practical.convergence_reason != "practical_plateau"
    assert practical_model.exact_face_components_ == strict_model.exact_face_components_
    assert practical.converged is True
    assert practical.unresolved_upper_bound == ()
    assert abs(practical.objective - strict.objective) <= 1.0e-9 * (1.0 + abs(strict.objective))


def test_converged_results_never_carry_unresolved_upper_pressure() -> None:
    _model, smoothing = _preempt_fit(True)
    # Several invariants can fire first (the forged name is not at the cap,
    # the strict raw step is zero); all of them name the unresolved pressure.
    with pytest.raises(ValueError, match="unresolved upper"):
        replace(
            smoothing,
            converged=True,
            convergence_reason="practical_plateau",
            unresolved_upper_bound=("location:effect#wiggle",),
        )


def _start_fixture():
    rng = np.random.default_rng(11)
    n = 600
    x = rng.uniform(-1.0, 1.0, n)
    y = 0.4 + 0.9 * np.sin(np.pi * x) + rng.normal(scale=0.35, size=n)
    return pd.DataFrame({"x": x}), y


def _start_fit(**kwargs):
    frame, y = _start_fixture()
    model = SuperLSS(
        family=GaussianLS(),
        predictors=(Predictor("location", {"x": Spline(kind="cr", k=8)}), Predictor("scale", {})),
    )
    model.fit_reml(frame, y, practical_reml=False, **kwargs)
    return model._require_fitted().smoothing


def test_fit_reml_initial_lambda_sets_the_search_start() -> None:
    low = _start_fit(initial_lambda=1.0e-3)
    high = _start_fit(initial_lambda=1.0e3)
    assert low.initial_lambdas["location:x#wiggle"] == 1.0e-3
    assert high.initial_lambdas["location:x#wiggle"] == 1.0e3
    assert low.config.initial_lambda == 1.0e-3
    assert high.config.initial_lambda == 1.0e3


def test_fit_reml_initial_lambda_is_capped_by_max_lambda_and_validated() -> None:
    capped = _start_fit(initial_lambda=10.0, max_lambda=5.0)
    assert capped.config.initial_lambda == 5.0
    frame, y = _start_fixture()
    model = SuperLSS(
        family=GaussianLS(),
        predictors=(Predictor("location", {"x": Spline(kind="cr", k=8)}), Predictor("scale", {})),
    )
    with pytest.raises(ValueError, match="initial_lambda"):
        model.fit_reml(frame, y, initial_lambda=0.0)
    with pytest.raises(ValueError, match="initial_lambda"):
        model.fit_reml(frame, y, initial_lambda=float("nan"))


@pytest.mark.parametrize(
    ("residual", "expected"),
    [
        (5.0e-7, "info"),
        (9.0e-3, "info"),
        (1.5e-2, "warning"),
        (0.9, "warning"),
        (1.5, "error"),
    ],
)
def test_residual_severity_bands(residual: float, expected: str) -> None:
    assert diagnostics_module._residual_severity(residual, 1.0e-6) == expected


def test_trajectory_unsettled_severity_is_label_independent() -> None:
    model, smoothing = _noise_random_effect_fit(practical=True, max_lambda=1.0e10)
    report = diagnose_distributional_fit(model._require_fitted())
    findings = [f for f in report.findings if f.code == "smoothing.trajectory_unsettled"]
    if not findings:
        pytest.skip("fixture settled below tolerance")
    expected = diagnostics_module._residual_severity(
        smoothing.terminal_raw_max_log_step, smoothing.config.tolerance
    )
    assert findings[0].severity == expected
