"""Tests for immutable IRLS state snapshots and atomic trial selection."""

from __future__ import annotations

import numpy as np
import pytest

from superglm._fit_trace import MemoryTraceSink, NullTraceSink, TraceRun
from superglm.distributions import Gaussian
from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
from superglm.links import IdentityLink
from superglm.penalties.group_lasso import GroupLasso
from superglm.solvers.irls_direct import fit_irls_direct
from superglm.solvers.irls_state import (
    _evaluate_irls_state,
    _immutable_array,
    _irls_trial_is_unsafe,
    _IRLSState,
    _IRLSStepDecision,
    _select_irls_trial,
)
from superglm.solvers.pirls import fit_pirls
from superglm.types import GroupSlice


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


def test_solver_state_excludes_relational_convergence_metadata() -> None:
    state = _synthetic_state(2.0)

    assert "convergence_value" not in state.__dataclass_fields__
    assert "termination_reason" not in state.__dataclass_fields__


def test_select_irls_trial_accepts_safe_full_without_evaluating_callback() -> None:
    decision = _select_irls_trial(
        committed=_synthetic_state(2.0),
        proposal=_synthetic_state(3.5),
        evaluate_state=lambda alpha: pytest.fail(f"unexpected trial at {alpha}"),
    )

    assert decision == _IRLSStepDecision(1.0, 0, False, trials_attempted=1)


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

    assert decision == _IRLSStepDecision(0.5, 1, False, trials_attempted=2)
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

    assert decision == _IRLSStepDecision(0.5, 1, False, trials_attempted=2)
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

    assert decision == _IRLSStepDecision(0.0, 0, True, trials_attempted=6)
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

    assert decision == _IRLSStepDecision(0.5, 1, False, trials_attempted=2)
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


class _ControlledDevianceGaussian(Gaussian):
    def __init__(self, deviance_for_mean):
        self._deviance_for_mean = deviance_for_mean

    def deviance_unit(self, y: np.ndarray, mu: np.ndarray) -> np.ndarray:
        total = self._deviance_for_mean(float(mu[0]))
        return np.full_like(y, total / len(y), dtype=float)


def _fit_controlled_pirls(
    deviance_for_mean,
    *,
    convergence: str = "deviance",
    trace_run: TraceRun | None = None,
    max_iter_outer: int = 1,
):
    n = 6
    X = np.zeros((n, 1))
    return fit_pirls(
        X,
        np.ones(n),
        np.ones(n),
        _ControlledDevianceGaussian(deviance_for_mean),
        IdentityLink(),
        [GroupSlice(name="x", start=0, end=1)],
        GroupLasso(lambda1=0.01),
        beta_init=np.zeros(1),
        intercept_init=0.0,
        max_iter_outer=max_iter_outer,
        max_iter_inner=1,
        tol=1e-12,
        record_diagnostics=True,
        convergence=convergence,
        trace_run=trace_run,
    )


def test_pirls_accepts_largest_safe_fixed_endpoint_half_step() -> None:
    def deviance_for_mean(mu: float) -> float:
        if np.isclose(mu, 0.0):
            return 2.0
        if np.isclose(mu, 1.0):
            return 10.0
        if np.isclose(mu, 0.5):
            return 3.0
        raise AssertionError(f"unexpected trial mean {mu}")

    result = _fit_controlled_pirls(deviance_for_mean)

    assert result.intercept == pytest.approx(0.5)
    assert result.deviance == pytest.approx(3.0)
    assert not result.converged
    assert result.iteration_log is not None
    assert result.iteration_log[0].step_halvings == 1
    assert not result.iteration_log[0].step_rejected


def test_pirls_nonfinite_full_trial_can_accept_safe_half_step() -> None:
    def deviance_for_mean(mu: float) -> float:
        if np.isclose(mu, 0.0):
            return 2.0
        if np.isclose(mu, 1.0):
            return np.nan
        if np.isclose(mu, 0.5):
            return 3.0
        raise AssertionError(f"unexpected trial mean {mu}")

    result = _fit_controlled_pirls(deviance_for_mean)

    assert result.intercept == pytest.approx(0.5)
    assert result.deviance == pytest.approx(3.0)
    assert result.iteration_log is not None
    assert result.iteration_log[0].step_halvings == 1
    assert not result.iteration_log[0].step_rejected


@pytest.mark.parametrize("convergence", ["deviance", "coefficients"])
def test_pirls_rejects_all_unsafe_trials_without_false_convergence(convergence: str) -> None:
    def deviance_for_mean(mu: float) -> float:
        return 2.0 if np.isclose(mu, 0.0) else 10.0

    result = _fit_controlled_pirls(deviance_for_mean, convergence=convergence)

    np.testing.assert_array_equal(result.beta, np.zeros(1))
    assert result.intercept == 0.0
    assert result.deviance == pytest.approx(2.0)
    assert not result.converged
    assert result.iteration_log is not None
    assert len(result.iteration_log) == 1
    assert result.iteration_log[0].step_halvings == 0
    assert result.iteration_log[0].step_rejected


def test_pirls_rejected_proposal_does_not_select_restored_zero_group() -> None:
    x = np.linspace(-1.0, 1.0, 6)[:, None]

    def deviance_for_mean(mu: float) -> float:
        return 2.0 if np.isclose(mu, 0.0) else 10.0

    result = fit_pirls(
        x,
        x[:, 0],
        np.ones(len(x)),
        _ControlledDevianceGaussian(deviance_for_mean),
        IdentityLink(),
        [GroupSlice(name="x", start=0, end=1)],
        GroupLasso(lambda1=0.01),
        beta_init=np.zeros(1),
        intercept_init=0.0,
        max_iter_outer=1,
        max_iter_inner=1,
        tol=1e-12,
        record_diagnostics=True,
    )

    np.testing.assert_array_equal(result.beta, np.zeros(1))
    assert result.iteration_log is not None
    assert result.iteration_log[0].step_rejected
    assert result.rank_info is not None
    assert result.rank_info.selected_group_names == ()
    assert result.rank_info.selected_columns.size == 0
    assert result.effective_df == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("outcome", "expected_alpha", "expected_attempts", "expected_committed_state"),
    [
        ("full", 1.0, 1, 2),
        ("half", 0.5, 2, 3),
        ("reject", 0.0, 6, 1),
    ],
)
def test_pirls_trace_decision_commits_an_evaluated_state(
    outcome: str,
    expected_alpha: float,
    expected_attempts: int,
    expected_committed_state: int,
) -> None:
    def deviance_for_mean(mu: float) -> float:
        if np.isclose(mu, 0.0):
            return 2.0
        if outcome == "full":
            return 3.0
        if outcome == "half" and np.isclose(mu, 0.5):
            return 3.0
        return 10.0

    sink = MemoryTraceSink()
    result = _fit_controlled_pirls(
        deviance_for_mean,
        trace_run=TraceRun(f"pirls-{outcome}", sink=sink, clock=lambda: 0.0),
    )

    evaluations = [event for event in sink.events if event.event_kind == "evaluation"]
    decisions = [event for event in sink.events if event.event_kind == "step_decision"]
    commits = [event for event in sink.events if event.event_kind == "state_commit"]
    assert len(decisions) == 1
    decision = decisions[0]
    terminal_commit = commits[-1]
    evaluated_ids = {event.payload["state_id"] for event in evaluations}

    assert decision.payload["base_state_id"] == 1
    assert decision.payload["proposal_state_id"] == 2
    assert decision.payload["accepted_alpha"] == expected_alpha
    assert decision.payload["trials_attempted"] == expected_attempts
    assert decision.payload["committed_state_id"] == expected_committed_state
    assert terminal_commit.payload["state_id"] == expected_committed_state
    assert all(commit.payload["state_id"] in evaluated_ids for commit in commits)
    assert result.state_id == expected_committed_state
    assert result.iteration_log is not None
    row = result.iteration_log[-1]
    assert row.base_state_id == 1
    assert row.proposal_state_id == 2
    assert row.committed_state_id == expected_committed_state
    assert row.trials_attempted == expected_attempts
    if outcome == "reject":
        assert not decision.payload["fit_converged"]
        assert not result.converged


def test_pirls_evaluation_trace_labels_trial_alpha_without_acceptance_claim() -> None:
    sink = MemoryTraceSink()
    _fit_controlled_pirls(
        lambda mu: 2.0 if np.isclose(mu, 0.0) else 10.0,
        trace_run=TraceRun("pirls-trial-alpha", sink=sink),
    )

    trial = next(
        event
        for event in sink.events
        if event.event_kind == "evaluation" and event.payload["phase"] == "line_search_trial"
    )
    assert trial.payload["trial_alpha"] == pytest.approx(0.5)
    assert "accepted_alpha" not in trial.payload


def test_pirls_trace_links_each_iteration_to_the_previous_commit() -> None:
    sink = MemoryTraceSink()
    result = _fit_controlled_pirls(
        lambda mu: 4.0 if np.isclose(mu, 0.0) else 2.0,
        trace_run=TraceRun("pirls-lineage", sink=sink),
        max_iter_outer=2,
    )

    decisions = [event for event in sink.events if event.event_kind == "step_decision"]
    assert len(decisions) == 2
    assert decisions[1].payload["base_state_id"] == decisions[0].payload["committed_state_id"]
    assert result.state_id == decisions[1].payload["committed_state_id"]
    assert result.converged


def test_pirls_null_trace_does_not_change_results_or_evaluation_count() -> None:
    counts: list[int] = []
    results = []
    for trace_run in (
        None,
        TraceRun("disabled", sink=NullTraceSink()),
        TraceRun("enabled", sink=MemoryTraceSink()),
    ):
        count = 0

        def deviance_for_mean(mu: float) -> float:
            nonlocal count
            count += 1
            return 2.0 if np.isclose(mu, 0.0) else 3.0

        results.append(_fit_controlled_pirls(deviance_for_mean, trace_run=trace_run))
        counts.append(count)

    assert counts == [counts[0]] * 3
    for result in results[1:]:
        np.testing.assert_array_equal(results[0].beta, result.beta)
        assert results[0].intercept == result.intercept
        assert results[0].deviance == result.deviance
        assert results[0].converged == result.converged
    assert results[0].state_id is None
    assert results[1].state_id is None
    assert results[2].state_id is not None


def test_pirls_convergence_claim_uses_the_committed_state_identity() -> None:
    sink = MemoryTraceSink()
    result = _fit_controlled_pirls(
        lambda mu: 2.0,
        trace_run=TraceRun("pirls-converged", sink=sink),
    )

    decision = next(event for event in sink.events if event.event_kind == "step_decision")
    commit = [event for event in sink.events if event.event_kind == "state_commit"][-1]
    assert result.converged
    assert decision.payload["fit_converged"]
    assert decision.payload["committed_state_id"] == result.state_id
    assert commit.payload["state_id"] == result.state_id
    assert commit.payload["fit_converged"]
    assert commit.payload["convergence_value"] == pytest.approx(0.0)


def _fit_controlled_direct(
    deviance_for_mean,
    *,
    convergence: str = "deviance",
    trace_run: TraceRun | None = None,
):
    n = 6
    X = np.zeros((n, 1))
    result, _ = fit_irls_direct(
        X,
        np.ones(n),
        np.ones(n),
        _ControlledDevianceGaussian(deviance_for_mean),
        IdentityLink(),
        [GroupSlice(name="x", start=0, end=1)],
        lambda2=0.0,
        beta_init=np.zeros(1),
        intercept_init=0.0,
        max_iter=1,
        tol=1e-12,
        record_diagnostics=True,
        convergence=convergence,
        trace_run=trace_run,
    )
    return result


def test_direct_accepts_largest_safe_fixed_endpoint_half_step_from_iteration_one() -> None:
    def deviance_for_mean(mu: float) -> float:
        if np.isclose(mu, 0.0):
            return 2.0
        if np.isclose(mu, 1.0):
            return 10.0
        if np.isclose(mu, 0.5):
            return 3.0
        raise AssertionError(f"unexpected trial mean {mu}")

    result = _fit_controlled_direct(deviance_for_mean)

    assert result.intercept == pytest.approx(0.5)
    assert result.deviance == pytest.approx(3.0)
    assert not result.converged
    assert result.iteration_log is not None
    assert result.iteration_log[0].step_halvings == 1
    assert not result.iteration_log[0].step_rejected


def test_direct_nonfinite_full_trial_can_accept_safe_half_step() -> None:
    def deviance_for_mean(mu: float) -> float:
        if np.isclose(mu, 0.0):
            return 2.0
        if np.isclose(mu, 1.0):
            return np.nan
        if np.isclose(mu, 0.5):
            return 3.0
        raise AssertionError(f"unexpected trial mean {mu}")

    result = _fit_controlled_direct(deviance_for_mean)

    assert result.intercept == pytest.approx(0.5)
    assert result.deviance == pytest.approx(3.0)
    assert result.iteration_log is not None
    assert result.iteration_log[0].step_halvings == 1
    assert not result.iteration_log[0].step_rejected


@pytest.mark.parametrize("convergence", ["deviance", "coefficients"])
def test_direct_rejects_all_unsafe_trials_without_false_convergence(convergence: str) -> None:
    def deviance_for_mean(mu: float) -> float:
        return 2.0 if np.isclose(mu, 0.0) else 10.0

    result = _fit_controlled_direct(deviance_for_mean, convergence=convergence)

    np.testing.assert_array_equal(result.beta, np.zeros(1))
    assert result.intercept == 0.0
    assert result.deviance == pytest.approx(2.0)
    assert not result.converged
    assert result.iteration_log is not None
    assert len(result.iteration_log) == 1
    assert result.iteration_log[0].step_halvings == 0
    assert result.iteration_log[0].step_rejected


@pytest.mark.parametrize(
    ("outcome", "expected_alpha", "expected_attempts", "expected_committed_state"),
    [
        ("full", 1.0, 1, 2),
        ("half", 0.5, 2, 3),
        ("reject", 0.0, 6, 1),
    ],
)
def test_direct_trace_decision_commits_an_evaluated_state(
    outcome: str,
    expected_alpha: float,
    expected_attempts: int,
    expected_committed_state: int,
) -> None:
    def deviance_for_mean(mu: float) -> float:
        if np.isclose(mu, 0.0):
            return 2.0
        if outcome == "full":
            return 3.0
        if outcome == "half" and np.isclose(mu, 0.5):
            return 3.0
        return 10.0

    sink = MemoryTraceSink()
    result = _fit_controlled_direct(
        deviance_for_mean,
        trace_run=TraceRun(f"direct-{outcome}", sink=sink, clock=lambda: 0.0),
    )

    evaluations = [event for event in sink.events if event.event_kind == "evaluation"]
    decision = next(event for event in sink.events if event.event_kind == "step_decision")
    commits = [event for event in sink.events if event.event_kind == "state_commit"]
    evaluated_ids = {event.payload["state_id"] for event in evaluations}
    assert decision.payload["base_state_id"] == 1
    assert decision.payload["proposal_state_id"] == 2
    assert decision.payload["accepted_alpha"] == expected_alpha
    assert decision.payload["trials_attempted"] == expected_attempts
    assert decision.payload["committed_state_id"] == expected_committed_state
    assert commits[-1].payload["state_id"] == expected_committed_state
    assert all(commit.payload["state_id"] in evaluated_ids for commit in commits)
    assert result.state_id == expected_committed_state
    assert result.iteration_log is not None
    row = result.iteration_log[-1]
    assert row.base_state_id == 1
    assert row.proposal_state_id == 2
    assert row.committed_state_id == expected_committed_state
    assert row.trials_attempted == expected_attempts
    if outcome == "reject":
        assert not decision.payload["fit_converged"]
        assert not result.converged


def test_direct_null_trace_does_not_change_results_or_evaluation_count() -> None:
    counts: list[int] = []
    results = []
    for trace_run in (None, TraceRun("direct-disabled", sink=NullTraceSink())):
        count = 0

        def deviance_for_mean(mu: float) -> float:
            nonlocal count
            count += 1
            return 2.0 if np.isclose(mu, 0.0) else 3.0

        results.append(_fit_controlled_direct(deviance_for_mean, trace_run=trace_run))
        counts.append(count)

    assert counts[0] == counts[1]
    np.testing.assert_array_equal(results[0].beta, results[1].beta)
    assert results[0].intercept == results[1].intercept
    assert results[0].deviance == results[1].deviance
    assert results[0].converged == results[1].converged
    assert results[0].state_id is None
    assert results[1].state_id is None


def test_direct_convergence_claim_uses_the_committed_state_identity() -> None:
    sink = MemoryTraceSink()
    result = _fit_controlled_direct(
        lambda mu: 2.0,
        trace_run=TraceRun("direct-converged", sink=sink),
    )

    decision = next(event for event in sink.events if event.event_kind == "step_decision")
    commit = [event for event in sink.events if event.event_kind == "state_commit"][-1]
    assert result.converged
    assert decision.payload["fit_converged"]
    assert decision.payload["committed_state_id"] == result.state_id
    assert commit.payload["state_id"] == result.state_id
    assert commit.payload["fit_converged"]
    assert commit.payload["convergence_value"] == pytest.approx(0.0)
