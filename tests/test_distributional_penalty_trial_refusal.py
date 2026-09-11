"""Numerically unresolved penalty trials preserve the last accepted fit."""

from __future__ import annotations

import numpy as np
import pytest

import superglm.distributional.smoothing.loop as smoothing_loop
from superglm.distributional.families.gaussian import GaussianLS
from superglm.distributional.model import fit_dense_distributional
from superglm.distributional.result import DistributionalEFSConfig
from superglm.distributional.smoothing.endpoint_laml import EndpointLaplaceError
from superglm.distributional.weights import WeightContract
from superglm.reml.penalty_support import PenaltyNumericalError

from .test_distributional_efs import _patch_second_iteration_acceleration, _smooth_fixture


def _fit(**options):
    frame, response, predictors = _smooth_fixture()
    return fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(),
        weight_contract=WeightContract(semantics="prior"),
        predictors=predictors,
        lambdas={"location:x#wiggle": 0.3, "scale:z#wiggle": 0.2},
        efs_config=DistributionalEFSConfig(
            outer="efs", tolerance=1e-12, max_backtracks=2, **options
        ),
    )


def _refuse(wrapped):
    try:
        raise PenaltyNumericalError("penalty reference action is unresolved")
    except PenaltyNumericalError as exc:
        if wrapped:
            raise EndpointLaplaceError("finite face penalty is unresolved") from exc
        raise


@pytest.mark.parametrize("wrapped", [False, True])
def test_unresolved_trials_keep_initial_coefficient_and_lambda_state(monkeypatch, wrapped):
    original = smoothing_loop._laplace_objective
    calls = []

    def objective(fit, **kwargs):
        calls.append(fit)
        if len(calls) > 1:
            _refuse(wrapped)
        return original(fit, **kwargs)

    monkeypatch.setattr(smoothing_loop, "_laplace_objective", objective)
    model = _fit(max_iterations=3)
    result = model.smoothing
    assert result is not None
    assert len(calls) == 4
    assert not result.converged
    assert result.convergence_reason == "objective_rejected"
    assert result.lambdas == result.initial_lambdas
    assert result.terminal_fit_index == 0
    assert model.result is result.coefficient_fits[0] is calls[0]
    assert result.objective == result.initial_objective
    assert not result.history[0].accepted
    assert result.history[0].accepted_fit_index is None
    np.testing.assert_array_equal(model.result.coefficients, calls[0].coefficients)


def test_unresolved_first_trial_backtracks_to_a_valid_fit(monkeypatch):
    original = smoothing_loop._laplace_objective
    rejected = []
    rejected_rows = []
    calls = 0

    def objective(fit, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            rejected.append(fit)
            rejected_rows.append((fit.eta, fit.theta))
            _refuse(False)
        return original(fit, **kwargs)

    monkeypatch.setattr(smoothing_loop, "_laplace_objective", objective)
    model = _fit(max_iterations=1)
    result = model.smoothing
    assert result is not None
    iteration = result.history[0]
    assert iteration.accepted
    assert iteration.raw_backtracks >= 1
    assert iteration.accepted_fit_index != 1
    assert model.result is not rejected[0]
    recorded = result.coefficient_fits[1]
    assert recorded.eta is None and recorded.theta is None
    assert recorded.row_shape == rejected[0].eta.shape
    for name in ("coefficients", "penalty", "terminal_score", "terminal_penalized_curvature"):
        np.testing.assert_array_equal(getattr(recorded, name), getattr(rejected[0], name))
    assert (
        recorded.penalized_optimizing_log_likelihood
        == rejected[0].penalized_optimizing_log_likelihood
    )
    assert recorded.history == rejected[0].history
    assert recorded.terminal_rank is rejected[0].terminal_rank
    assert recorded.terminal_curvature == rejected[0].terminal_curvature
    # Compact only the history reference; the still-live trial stays usable.
    assert rejected[0].eta is rejected_rows[0][0]
    assert rejected[0].theta is rejected_rows[0][1]
    ceiling = iteration.objective_before + result.config.objective_tolerance * (
        1.0 + abs(iteration.objective_before)
    )
    assert iteration.objective_after <= ceiling


def test_unresolved_acceleration_falls_back_to_the_raw_proposal(monkeypatch):
    calls = 0

    def objective(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            _refuse(False)
        return -float(calls)

    monkeypatch.setattr(smoothing_loop, "_laplace_objective", objective)
    decisions = _patch_second_iteration_acceleration(monkeypatch)
    model = _fit(max_iterations=2, acceleration="multisecant")
    result = model.smoothing
    assert result is not None
    assert decisions == ["warming", "proposal"]
    iteration = result.history[1]
    assert iteration.acceleration_outcome == "rejected"
    assert iteration.accelerated_fit_index == 2
    assert iteration.accepted
    assert iteration.accepted_fit_index == 3
    assert result.terminal_fit_index == 3


@pytest.mark.parametrize("wrapped", [False, True])
def test_unresolved_initial_penalty_is_not_reported_as_a_fit(monkeypatch, wrapped):
    def objective(*args, **kwargs):
        _refuse(wrapped)

    monkeypatch.setattr(smoothing_loop, "_laplace_objective", objective)
    expected = EndpointLaplaceError if wrapped else PenaltyNumericalError
    with pytest.raises(expected, match="unresolved"):
        _fit(max_iterations=1)


@pytest.mark.parametrize("error_type", [ValueError, EndpointLaplaceError, RuntimeError])
def test_unrelated_trial_errors_propagate(monkeypatch, error_type):
    original = smoothing_loop._laplace_objective
    calls = 0

    def objective(fit, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise error_type("invalid trial provenance")
        return original(fit, **kwargs)

    monkeypatch.setattr(smoothing_loop, "_laplace_objective", objective)
    with pytest.raises(error_type, match="invalid trial provenance"):
        _fit(max_iterations=1)
