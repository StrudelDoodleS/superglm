"""Tests for immutable IRLS state snapshots and atomic trial selection."""

from __future__ import annotations

import numpy as np
import pytest

from superglm.distributions import Gaussian
from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
from superglm.links import IdentityLink
from superglm.solvers.irls_state import (
    _evaluate_irls_state,
    _immutable_array,
    _irls_trial_is_unsafe,
    _IRLSState,
    _IRLSStepDecision,
    _select_irls_trial,
)


def _synthetic_state(deviance: float, *, beta: float = 0.0, eta: float = 0.0) -> _IRLSState:
    beta_values = _immutable_array(np.array([beta]))
    eta_values = _immutable_array(np.array([eta]))
    return _IRLSState(
        beta=beta_values,
        intercept=0.0,
        eta_unclipped=eta_values,
        eta=eta_values,
        mu=eta_values,
        deviance=deviance,
    )


def test_evaluate_irls_state_freezes_complete_gaussian_snapshot() -> None:
    X = np.array([[1.0], [2.0], [4.0]])
    dm = DesignMatrix([DenseGroupMatrix(X)], n=3, p=1)
    y = np.array([2.0, 5.0, 9.0])
    weights = np.array([1.0, 2.0, 0.5])
    offset = np.array([0.25, -0.5, 1.0])
    beta = np.array([2.0])

    state = _evaluate_irls_state(
        dm,
        y,
        weights,
        Gaussian(),
        IdentityLink(),
        offset,
        beta,
        intercept=-1.0,
    )

    expected_eta = X[:, 0] * 2.0 - 1.0 + offset
    np.testing.assert_allclose(state.eta_unclipped, expected_eta)
    np.testing.assert_allclose(state.eta, expected_eta)
    np.testing.assert_allclose(state.mu, expected_eta)
    assert state.deviance == pytest.approx(float(np.sum(weights * (y - expected_eta) ** 2)))
    assert not np.shares_memory(state.beta, beta)
    for values in (state.beta, state.eta_unclipped, state.eta, state.mu):
        assert not values.flags.writeable
        with pytest.raises(ValueError):
            values[0] = 123.0


def test_select_irls_trial_accepts_safe_full_without_evaluating_callback() -> None:
    decision = _select_irls_trial(
        committed=_synthetic_state(2.0),
        proposal=_synthetic_state(3.5),
        evaluate_state=lambda alpha: pytest.fail(f"unexpected trial at {alpha}"),
    )

    assert decision == _IRLSStepDecision(1.0, 0, False)


def test_select_irls_trial_stops_at_largest_safe_fraction() -> None:
    visited: list[float] = []

    def evaluate(alpha: float) -> _IRLSState:
        visited.append(alpha)
        if alpha != 0.5:
            raise AssertionError("smaller fraction must not be evaluated")
        return _synthetic_state(3.0)

    decision = _select_irls_trial(
        committed=_synthetic_state(2.0),
        proposal=_synthetic_state(10.0),
        evaluate_state=evaluate,
    )

    assert decision == _IRLSStepDecision(0.5, 1, False)
    assert visited == [0.5]


def test_nonfinite_proposal_coefficients_can_select_safe_half_step() -> None:
    proposal = _synthetic_state(1.0, beta=np.inf)
    visited: list[float] = []

    def evaluate(alpha: float) -> _IRLSState:
        visited.append(alpha)
        return _synthetic_state(3.0, beta=1.0)

    decision = _select_irls_trial(
        committed=_synthetic_state(2.0),
        proposal=proposal,
        evaluate_state=evaluate,
    )

    assert decision == _IRLSStepDecision(0.5, 1, False)
    assert visited == [0.5]


def test_legacy_accepted_then_rejected_sequence_restores_committed_state() -> None:
    values = {0.5: 5.0, 0.25: 6.0, 0.125: 6.5, 0.0625: 7.0, 0.03125: 8.0}
    visited: list[float] = []

    def evaluate(alpha: float) -> _IRLSState:
        visited.append(alpha)
        return _synthetic_state(values[alpha])

    decision = _select_irls_trial(
        committed=_synthetic_state(2.0),
        proposal=_synthetic_state(10.0),
        evaluate_state=evaluate,
        max_halving=5,
    )

    assert decision == _IRLSStepDecision(0.0, 0, True)
    assert visited == [0.5, 0.25, 0.125, 0.0625, 0.03125]


def test_catastrophe_thresholds_are_strict_and_not_monotone() -> None:
    committed = _synthetic_state(2.0)

    assert not _irls_trial_is_unsafe(_synthetic_state(4.0), committed)
    assert _irls_trial_is_unsafe(_synthetic_state(np.nextafter(4.0, np.inf)), committed)
    assert not _irls_trial_is_unsafe(_synthetic_state(3.0), committed)
    assert not _irls_trial_is_unsafe(_synthetic_state(-2.0), committed)
    assert _irls_trial_is_unsafe(_synthetic_state(np.nextafter(-2.0, -np.inf)), committed)


def test_custom_invalid_state_callback_triggers_halving() -> None:
    visited: list[float] = []

    def evaluate(alpha: float) -> _IRLSState:
        visited.append(alpha)
        return _synthetic_state(1.0, eta=0.0)

    decision = _select_irls_trial(
        committed=_synthetic_state(2.0),
        proposal=_synthetic_state(1.0, eta=10.0),
        evaluate_state=evaluate,
        invalid_state=lambda state: bool(state.eta[0] > 5.0),
    )

    assert decision == _IRLSStepDecision(0.5, 1, False)
    assert visited == [0.5]


@pytest.mark.parametrize("max_halving", [0, -1])
def test_select_irls_trial_requires_positive_max_halving(max_halving: int) -> None:
    with pytest.raises(ValueError, match="max_halving"):
        _select_irls_trial(
            committed=_synthetic_state(2.0),
            proposal=_synthetic_state(3.0),
            evaluate_state=lambda alpha: _synthetic_state(alpha),
            max_halving=max_halving,
        )
