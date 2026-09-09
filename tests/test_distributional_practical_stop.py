from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import superglm.distributional.efs as efs_module
from superglm import SuperLSS
from superglm.distributional import GaussianLS, Predictor
from superglm.distributional import fit_diagnostics as diagnostics_module
from superglm.distributional.fit_diagnostics import diagnose_distributional_fit
from superglm.distributional.result import DistributionalEFSConfig
from superglm.distributional.smoothing.objective import _stable_isolated_gfs_update
from superglm.features import RandomEffect, Spline
from superglm.reml.efs_update import EFSComponentState, wood_fasiolo_update
from superglm.types import LambdaPolicy


def test_saturated_update_checks_inverse_products_before_cancellation() -> None:
    factor = np.array([1.0, math.sqrt(2.0)])
    penalty = np.outer(factor, factor)
    eigenvalue = float(factor @ factor)
    projector = penalty / eigenvalue
    null_projector = np.eye(2) - projector
    lam = 1.0e10
    # H = I + λS: its two eigenspaces have eigenvalues 1 and 1 + λ||factor||².
    inverse = null_projector + projector / (1.0 + eigenvalue * lam)
    beta = np.array([factor[1], -factor[0]]) + 2.0**-40 * factor
    component = EFSComponentState("smooth", slice(0, 2), penalty, 1.0, lam, LambdaPolicy.estimate())
    fit = SimpleNamespace(
        coefficients=beta,
        coefficient_face=None,
        terminal_rank=SimpleNamespace(rank=2),
        terminal_data_curvature=np.eye(2),
        terminal_penalized_curvature=np.eye(2) + lam * penalty,
    )
    config = DistributionalEFSConfig()
    raw = wood_fasiolo_update((component,), beta, inverse, inverse_scale=1.0)
    stable, names = _stable_isolated_gfs_update((component,), fit, inverse, raw, config)

    assert names == {"smooth"}
    # The exact multiplier is 1 / ((1 + 3λ) λ 9·2^-80), safely above one.
    assert stable.raw_log_steps["smooth"] > math.log(100.0)
    assert stable.proposal_kinds["smooth"] == "gfs"
    assert stable.stationarity_log_residuals["smooth"] > 0.0
    np.testing.assert_allclose(
        math.exp(-stable.stationarity_log_residuals["smooth"]),
        lam * (stable.quadratic_forms["smooth"] + stable.trace_terms["smooth"]),
        rtol=8.0 * np.finfo(float).eps,
        atol=0.0,
    )
    projection_bound = (
        64.0 * beta.size * np.finfo(float).eps * np.linalg.norm(factor) * np.linalg.norm(beta)
    )
    assert (
        abs(math.sqrt(stable.quadratic_forms["smooth"]) - abs(float(factor @ beta)))
        <= projection_bound
    )

    # A trace rounded below the nominal saturation threshold still needs repair.
    rounded = replace(
        raw, trace_terms={"smooth": (1.0 - 2.0 * math.sqrt(np.finfo(float).eps)) / lam}
    )
    retried, names = _stable_isolated_gfs_update((component,), fit, inverse, rounded, config)
    assert names == {"smooth"}
    assert retried.raw_log_steps == stable.raw_log_steps
    assert retried.stationarity_log_residuals == stable.stationarity_log_residuals
    assert retried.trace_terms == stable.trace_terms

    _, refused = _stable_isolated_gfs_update((component,), fit, 2.0 * inverse, raw, config)
    assert refused == frozenset()


def test_saturated_update_refuses_an_unresolved_residual_product() -> None:
    penalty = np.array([[0.5, -0.5], [-0.5, 0.5]])
    common = np.eye(2) - penalty
    large = 2.0**40
    small = np.spacing(large)
    lam = 2.0**16
    curvature = large * np.ones((2, 2)) + small * np.eye(2)
    inverse = common / (2.0 * large + small) + penalty / (lam + small)
    beta = np.array([0.001, -0.001])
    component = EFSComponentState("smooth", slice(0, 2), penalty, 1.0, lam, LambdaPolicy.estimate())
    fit = SimpleNamespace(
        coefficients=beta,
        coefficient_face=None,
        terminal_rank=SimpleNamespace(rank=2),
        terminal_data_curvature=curvature,
        terminal_penalized_curvature=curvature + lam * penalty,
    )
    update = wood_fasiolo_update((component,), beta, inverse)

    # The true residual is smaller than the rounding bound of the matrix product.
    scale = float(np.sum(np.abs(penalty) * (np.abs(inverse) @ np.abs(curvature)).T))
    assert small / (lam + small) < beta.size * np.finfo(float).eps * scale
    stable, names = _stable_isolated_gfs_update(
        (component,), fit, inverse, update, DistributionalEFSConfig()
    )
    assert names == frozenset()
    assert stable is update


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


def _ridge_outward_window(*, t=0.5, moves=(0.3, 0.4, 0.5), second_moves=None):
    """Exact unit-information Gaussian ridge profile, including its covariance.

    beta=t/(1+lambda), Var(beta)=1/(1+lambda), and the rank-one LAML
    F=.5*(log(1+1/lambda)+t**2*lambda/(1+lambda)).  Its EFS raw
    log step is log((1+lambda)/(t**2*lambda)), so t<1 pushes outward
    while t>1 at large lambda returns toward a finite optimum.
    """
    names = ("mean:a",) if second_moves is None else ("mean:a", "mean:b")
    movements = [moves] if second_moves is None else [moves, second_moves]
    paths = [1.0e6 * np.exp(np.r_[0.0, np.cumsum(values)]) for values in movements]
    fits = []
    objectives = []
    for index in range(4):
        penalties = np.array([path[index] for path in paths])
        means = t / (1.0 + penalties)
        fits.append(
            SimpleNamespace(
                theta=np.column_stack((means, np.ones_like(means))),
                covariance=np.diag(1.0 / (1.0 + penalties)),
            )
        )
        objectives.append(
            float(0.5 * np.sum(np.log1p(1.0 / penalties) + t**2 * penalties / (1 + penalties)))
        )
    history = []
    for index in range(3):
        history.append(
            SimpleNamespace(
                accepted=True,
                stage="efs",
                source_fit_index=index,
                accepted_fit_index=index + 1,
                lambdas_before={name: paths[k][index] for k, name in enumerate(names)},
                lambdas_after={name: paths[k][index + 1] for k, name in enumerate(names)},
                accepted_log_steps={name: movements[k][index] for k, name in enumerate(names)},
                objective_before=objectives[index],
                objective_after=objectives[index + 1],
                objective_relative_change=abs(objectives[index + 1] - objectives[index])
                / (1.0 + abs(objectives[index])),
                activated_face_components=(),
                deactivated_face_components=(),
                revalidated_face_components=(),
                refused_face_components=(),
            )
        )
    raw_steps = {
        name: math.log((1.0 + paths[k][-1]) / (t**2 * paths[k][-1])) for k, name in enumerate(names)
    }
    config = DistributionalEFSConfig(
        practical_convergence=True,
        plateau_tolerance=1.0e-6,
        practical_parameter_tolerance=1.0e-4,
    )
    return history, fits, raw_steps, config


def _outward_window_matches(history, fits, raw_steps, config):
    from superglm.distributional.results.smoothing import _practical_outward_window

    return _practical_outward_window(
        history=history,
        coefficient_fits=fits,
        terminal_raw_log_steps=raw_steps,
        config=config,
    )


def test_practical_outward_window_distinguishes_ridge_drift_from_finite_optimum() -> None:
    outward = _ridge_outward_window(t=0.5)
    inward = _ridge_outward_window(t=2.0)
    assert _outward_window_matches(*outward)
    assert not _outward_window_matches(*inward)
    # The accepted finite ridge fit is already close to its exact nullspace:
    # both prediction and covariance error are bounded by 1/lambda.
    history, fits, _raw, _config = outward
    terminal_lambda = history[-1].lambdas_after["mean:a"]
    assert np.max(np.abs(fits[-1].theta[:, 0])) <= 0.5 / terminal_lambda
    assert np.linalg.norm(fits[-1].covariance, ord=2) <= 1.0 / terminal_lambda


@pytest.mark.parametrize(
    "moves",
    [
        (0.02, 0.03, 0.05),  # a small accepted probe is not evidence of flatness
        (0.6, -0.1, 0.7),  # net outward distance cannot conceal an oscillation
        (0.5, 0.7, 0.0),  # a clipped duplicate cannot complete the window
    ],
)
def test_practical_outward_window_requires_substantial_monotone_probes(moves) -> None:
    assert not _outward_window_matches(*_ridge_outward_window(moves=moves))


def test_practical_outward_window_checks_each_coordinate_span() -> None:
    assert not _outward_window_matches(*_ridge_outward_window(second_moves=(0.02, 0.03, 0.05)))
    assert _outward_window_matches(*_ridge_outward_window(second_moves=(0.4, 0.4, 0.4)))


@pytest.mark.parametrize("parameter", [0, 1, 2])
def test_practical_outward_window_checks_cumulative_all_parameter_change(parameter: int) -> None:
    history, fits, raw_steps, config = _ridge_outward_window()
    for index, fit in enumerate(fits):
        if parameter == 2:
            fit.theta = np.column_stack((fit.theta, np.ones(fit.theta.shape[0])))
        fit.theta[:, parameter] = 1.0 + index * 1.5e-4
    # Every individual change is below 1e-4 relative to (1+|theta|), but
    # the complete first-source-to-terminal window moves by more than it.
    assert not _outward_window_matches(history, fits, raw_steps, config)


def test_practical_outward_window_checks_cumulative_objective_change() -> None:
    history, fits, raw_steps, config = _ridge_outward_window()
    for index, item in enumerate(history):
        item.objective_before = 1.0 - index * 1.5e-6
        item.objective_after = 1.0 - (index + 1) * 1.5e-6
        item.objective_relative_change = 1.5e-6 / (1.0 + abs(item.objective_before))
    assert not _outward_window_matches(history, fits, raw_steps, config)


def test_practical_outward_window_rejects_missing_coordinate_pressure() -> None:
    history, fits, raw_steps, config = _ridge_outward_window(second_moves=(0.02, 0.03, 0.05))
    raw_steps.pop("mean:b")
    assert not _outward_window_matches(history, fits, raw_steps, config)


def test_practical_outward_window_preserves_lower_bound_authority() -> None:
    history, fits, raw_steps, config = _ridge_outward_window(second_moves=(0.0, 0.0, 0.0))
    config = replace(config, minimum_lambda=1.0e6, initial_lambda=1.0e6)
    raw_steps["mean:b"] = -0.1
    assert not _outward_window_matches(history, fits, raw_steps, config)


def test_practical_outward_window_cannot_omit_unchanged_lower_pressure() -> None:
    history, fits, raw_steps, config = _ridge_outward_window(second_moves=(0.0, 0.0, 0.0))
    config = replace(config, minimum_lambda=1.0e6, initial_lambda=1.0e6)
    raw_steps["mean:b"] = -0.1
    assert not _outward_window_matches(history, fits, raw_steps, config)
    # The coordinate is pinned by clipping, not stationary. Removing its
    # pressure used to bypass the veto because its accepted lambda never moved.
    raw_steps.pop("mean:b")
    assert not _outward_window_matches(history, fits, raw_steps, config)


@pytest.mark.parametrize("fixed_value", [0.0, 1.0e10])
def test_practical_outward_window_allows_unchanged_zero_pressure_fixed_coordinate(fixed_value):
    history, fits, raw_steps, config = _ridge_outward_window()
    for item in history:
        item.lambdas_before["mean:fixed"] = fixed_value
        item.lambdas_after["mean:fixed"] = fixed_value
        item.accepted_log_steps["mean:fixed"] = 0.0
    raw_steps["mean:fixed"] = 0.0
    assert _outward_window_matches(history, fits, raw_steps, config)


def test_practical_outward_window_requires_pressure_in_terminal_coordinate_order() -> None:
    history, fits, raw_steps, config = _ridge_outward_window(second_moves=(0.4, 0.4, 0.4))
    assert not _outward_window_matches(history, fits, dict(reversed(raw_steps.items())), config)


@pytest.mark.parametrize("window_size", [1, 2])
def test_practical_outward_window_honors_configured_probe_count(window_size) -> None:
    history, fits, raw_steps, config = _ridge_outward_window(moves=(0.1, 0.2, 1.2))
    config = replace(
        config,
        plateau_iterations=window_size,
        maximum_lambda=history[-1].lambdas_after["mean:a"],
    )
    assert _outward_window_matches(history[-window_size:], fits, raw_steps, config)


def _outward_random_effect_fit(*, practical=True, max_log_step=0.5, span=1.5, fixed_lambda=None):
    frame, y = _preempt_fixture()
    features = {"effect": RandomEffect()}
    if fixed_lambda is not None:
        frame["fixed"] = np.tile(["u", "v"], len(frame) // 2)
        features["fixed"] = RandomEffect(lambda_policy=LambdaPolicy.fixed(fixed_lambda))
    model = SuperLSS(
        family=GaussianLS(scale_floor=1.0e-4),
        predictors=(Predictor("location", features), Predictor("scale", {})),
    )
    model.fit_reml(
        frame,
        y,
        lambdas={"location:effect#wiggle": 1.0e6},
        max_lambda=1.0e6 * math.exp(span),
        max_log_step=max_log_step,
        max_reml_iter=20,
        reml_tol=1.0e-8,
        inner_tol=1.0e-10,
        reml_plateau_tol=1.0e-6,
        practical_reml=practical,
    )
    return model, model._require_fitted().smoothing


@pytest.mark.parametrize("fixed_value", [0.0, 1.0e6 * math.exp(1.5)])
def test_practical_outward_result_records_fixed_coordinate_pressure(fixed_value):
    model, smoothing = _outward_random_effect_fit(fixed_lambda=fixed_value)
    assert smoothing.convergence_reason == "practical_plateau"
    assert tuple(smoothing.terminal_raw_log_steps) == tuple(smoothing.lambdas)
    assert smoothing.terminal_raw_log_steps["location:fixed#wiggle"] == 0.0
    pressure = dict(smoothing.terminal_raw_log_steps)
    pressure.pop("location:fixed#wiggle")
    with pytest.raises(ValueError, match="raw log steps.*all terminal lambdas"):
        replace(smoothing, terminal_raw_log_steps=pressure)

    from superglm.distributional.serialization import (
        deserialize_distributional_model,
        serialize_distributional_model,
    )

    restored = deserialize_distributional_model(
        serialize_distributional_model(model._require_fitted())
    ).smoothing
    assert restored.terminal_raw_log_steps == smoothing.terminal_raw_log_steps


def test_practical_outward_cap_stop_keeps_pressure_and_replays_finite_fit() -> None:
    _model, smoothing = _outward_random_effect_fit()
    assert smoothing.convergence_reason == "practical_plateau"
    assert smoothing.converged and not smoothing.matched_certified
    assert smoothing.unresolved_upper_bound == ("location:effect#wiggle",)
    assert smoothing.terminal_fit.coefficient_face is None
    assert smoothing.terminal_raw_log_steps["location:effect#wiggle"] > 0.0
    assert len(smoothing.history) == smoothing.config.plateau_iterations
    with pytest.raises(ValueError, match="practical|outward"):
        replace(smoothing, terminal_raw_log_steps={"location:effect#wiggle": -0.1})
    with pytest.raises(ValueError, match="practical|outward"):
        replace(smoothing, terminal_raw_log_steps=None)

    _strict_model, strict = _outward_random_effect_fit(practical=False)
    assert strict.converged and strict.terminal_fit.coefficient_face is not None
    finite_theta = smoothing.terminal_fit.theta
    exact_theta = strict.terminal_fit.theta
    relative = np.abs(finite_theta - exact_theta) / (
        1.0 + np.maximum(np.abs(finite_theta), np.abs(exact_theta))
    )
    assert np.max(relative) < smoothing.config.practical_parameter_tolerance
    assert (
        abs(smoothing.objective - strict.objective) / (1.0 + abs(strict.objective))
        < smoothing.config.plateau_tolerance
    )
    finite_covariance = smoothing.terminal_fit.terminal_pseudo_inverse()
    exact_covariance = strict.terminal_fit.terminal_pseudo_inverse()
    assert (
        np.linalg.norm(finite_covariance - exact_covariance, ord=2)
        / (1.0 + np.linalg.norm(exact_covariance, ord=2))
        < smoothing.config.practical_parameter_tolerance
    )


def test_practical_outward_pressure_survives_artifact_roundtrip() -> None:
    import json

    from superglm.distributional.serialization import (
        deserialize_distributional_model,
        serialize_distributional_model,
    )

    model, smoothing = _outward_random_effect_fit()
    encoded = serialize_distributional_model(model._require_fitted())
    manifest = json.loads(encoded)["manifest"]["smoothing"]
    assert manifest["terminal_raw_log_steps"] == dict(smoothing.terminal_raw_log_steps)
    restored = deserialize_distributional_model(encoded).smoothing
    assert restored.convergence_reason == "practical_plateau"
    assert restored.unresolved_upper_bound == smoothing.unresolved_upper_bound
    assert restored.terminal_raw_log_steps == smoothing.terminal_raw_log_steps
    assert not restored.matched_certified
    with pytest.raises(TypeError):
        restored.terminal_raw_log_steps["location:effect#wiggle"] = -0.1


def test_older_artifact_without_raw_pressure_metadata_gets_optional_default() -> None:
    import json

    from superglm.distributional.serialization import (
        deserialize_distributional_model,
        serialize_distributional_model,
    )
    from tests.test_distributional_serialization import _rehash_pickled_artifact

    model, smoothing = _preempt_fit(True)
    artifact = json.loads(serialize_distributional_model(model._require_fitted()))
    artifact["manifest"]["smoothing"].pop("terminal_raw_log_steps", None)

    def omit_optional_field(restored):
        vars(restored.smoothing).pop("terminal_raw_log_steps")

    restored = deserialize_distributional_model(
        _rehash_pickled_artifact(artifact, omit_optional_field)
    ).smoothing
    assert restored.terminal_raw_log_steps is None
    assert restored.convergence_reason == smoothing.convergence_reason
    assert restored.matched_certified == smoothing.matched_certified
