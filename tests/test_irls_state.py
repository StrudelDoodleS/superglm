"""Tests for immutable IRLS state snapshots and atomic trial selection."""

from __future__ import annotations

import numpy as np
import pytest

from superglm._fit_trace import MemoryTraceSink, NullTraceSink, TraceRun
from superglm.distributions import Gaussian, Poisson
from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
from superglm.links import IdentityLink, SqrtLink
from superglm.penalties.group_lasso import GroupLasso
from superglm.penalties.ridge import Ridge
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


def _synthetic_state(
    deviance: float,
    *,
    beta: float = 0.0,
    eta: float = 0.0,
    penalized_deviance: float | None = None,
) -> _IRLSState:
    beta_values = _immutable_array(np.array([beta]))
    eta_values = _immutable_array(np.array([eta]))
    return _IRLSState(
        beta=beta_values,
        intercept=0.0,
        eta_unclipped=eta_values,
        eta=eta_values,
        mu=eta_values,
        deviance=deviance,
        penalized_deviance=penalized_deviance,
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
        proposal=_synthetic_state(1.5),
        evaluate_state=lambda alpha: pytest.fail(f"unexpected trial at {alpha}"),
    )

    assert decision == _IRLSStepDecision(1.0, 0, False, trials_attempted=1)


def test_select_irls_trial_stops_at_largest_safe_fraction() -> None:
    visited: list[float] = []

    def evaluate(alpha: float) -> _IRLSState:
        visited.append(alpha)
        if alpha != 0.5:
            raise AssertionError("smaller fraction must not be evaluated")
        return _synthetic_state(1.5)

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
        return _synthetic_state(1.5, beta=1.0)

    decision = _select_irls_trial(
        committed=_synthetic_state(2.0),
        proposal=proposal,
        evaluate_state=evaluate,
    )

    assert decision == _IRLSStepDecision(0.5, 1, False, trials_attempted=2)
    assert visited == [0.5]


def test_all_increasing_trials_restore_committed_state_after_twenty_halvings() -> None:
    visited: list[float] = []

    def evaluate(alpha: float) -> _IRLSState:
        visited.append(alpha)
        return _synthetic_state(2.0 + alpha)

    decision = _select_irls_trial(
        committed=_synthetic_state(2.0),
        proposal=_synthetic_state(10.0),
        evaluate_state=evaluate,
        max_halving=20,
    )

    assert decision == _IRLSStepDecision(0.0, 0, True, trials_attempted=21)
    assert visited == [2.0**-depth for depth in range(1, 21)]


def test_trial_guard_rejects_material_increase_but_allows_roundoff_plateau() -> None:
    committed = _synthetic_state(2.0)
    eps = np.finfo(float).eps

    assert not _irls_trial_is_unsafe(_synthetic_state(2.0 - eps), committed)
    assert not _irls_trial_is_unsafe(_synthetic_state(2.0 + 64.0 * eps), committed)
    assert _irls_trial_is_unsafe(_synthetic_state(2.0 + 256.0 * eps), committed)
    assert _irls_trial_is_unsafe(_synthetic_state(2.01), committed)


def test_default_halving_budget_can_recover_a_deep_descent_trial() -> None:
    visited: list[float] = []

    def evaluate(alpha: float) -> _IRLSState:
        visited.append(alpha)
        return _synthetic_state(1.0 if alpha == 2.0**-12 else 3.0)

    decision = _select_irls_trial(
        committed=_synthetic_state(2.0),
        proposal=_synthetic_state(4.0),
        evaluate_state=evaluate,
    )

    assert decision == _IRLSStepDecision(2.0**-12, 12, False, trials_attempted=13)
    assert visited == [2.0**-depth for depth in range(1, 13)]


def test_evaluate_state_can_reuse_a_precomputed_linear_predictor() -> None:
    X = np.array([[1.0], [2.0], [4.0]])
    dm = DesignMatrix([DenseGroupMatrix(X)], n=3, p=1)
    y = np.array([2.0, 5.0, 9.0])
    weights = np.ones(3)
    offset = np.array([0.25, -0.5, 1.0])
    beta = np.array([2.0])
    expected_eta = X[:, 0] * beta[0] - 1.0 + offset

    dm.matvec = lambda values: pytest.fail(f"unexpected matvec for {values}")  # type: ignore[method-assign]
    state = _evaluate_irls_state(
        dm,
        y,
        weights,
        Gaussian(),
        IdentityLink(),
        offset,
        beta,
        intercept=-1.0,
        eta_unclipped=expected_eta.copy(),
    )

    np.testing.assert_array_equal(state.eta_unclipped, expected_eta)
    assert not state.eta_unclipped.flags.writeable


def test_penalty_improving_trial_uses_penalized_deviance_merit() -> None:
    """Wood step halving must allow a raw-deviance increase that reduces D + beta'S beta."""
    committed = _synthetic_state(0.0, penalized_deviance=40.0)
    proposal = _synthetic_state(9.831, penalized_deviance=17.401)

    assert not _irls_trial_is_unsafe(proposal, committed)


@pytest.mark.parametrize(
    ("committed_penalized", "candidate_penalized"),
    [(10.0, None), (None, 9.0)],
)
def test_trial_guard_rejects_incomparable_merit_definitions(
    committed_penalized: float | None,
    candidate_penalized: float | None,
) -> None:
    committed = _synthetic_state(1.0, penalized_deviance=committed_penalized)
    candidate = _synthetic_state(0.5, penalized_deviance=candidate_penalized)

    assert _irls_trial_is_unsafe(candidate, committed)


def test_direct_ridge_moves_away_from_an_interpolating_unpenalized_state() -> None:
    """A zero-deviance start must not make every positive-penalty proposal unsafe."""
    x = np.linspace(-1.0, 1.0, 21)
    y = 2.0 * x
    penalty = 10.0
    dm = DesignMatrix([DenseGroupMatrix(x)], n=len(x), p=1)

    result, _ = fit_irls_direct(
        X=dm,
        y=y,
        weights=np.ones(len(x)),
        family=Gaussian(),
        link=IdentityLink(),
        groups=[GroupSlice(name="x", start=0, end=1)],
        lambda2=penalty,
        S_override=np.array([[penalty]]),
        beta_init=np.array([2.0]),
        intercept_init=0.0,
        tol=1.0e-12,
        max_iter=10,
    )

    expected_beta = float((x @ y) / (x @ x + penalty))
    assert result.converged
    assert result.termination_reason == "converged"
    assert result.beta[0] == pytest.approx(expected_beta, rel=1e-12, abs=1e-12)
    assert result.deviance > 0.0


def test_pirls_ridge_moves_away_from_an_interpolating_unpenalized_state() -> None:
    """The composite-penalty solver uses the same Wood penalized-deviance merit."""
    x = np.linspace(-1.0, 1.0, 21)
    y = 2.0 * x
    penalty = 10.0
    result = fit_pirls(
        X=x[:, None],
        y=y,
        weights=np.ones(len(x)),
        family=Gaussian(),
        link=IdentityLink(),
        groups=[GroupSlice(name="x", start=0, end=1)],
        penalty=Ridge(lambda1=penalty),
        beta_init=np.array([2.0]),
        intercept_init=0.0,
        max_iter_outer=20,
        max_iter_inner=50,
        tol=1.0e-10,
    )

    expected_beta = float((x @ y) / (x @ x + penalty))
    assert result.converged
    assert result.termination_reason == "converged"
    # The BCD Hessian carries its documented 1e-4 conditioning ridge; the
    # regression target here is accepting the penalty-improving move.
    assert result.beta[0] == pytest.approx(expected_beta, rel=1e-4, abs=1e-4)
    assert result.deviance > 0.0


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
            return 1.5
        raise AssertionError(f"unexpected trial mean {mu}")

    result = _fit_controlled_pirls(deviance_for_mean)

    assert result.intercept == pytest.approx(0.5)
    assert result.deviance == pytest.approx(1.5)
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
            return 1.5
        raise AssertionError(f"unexpected trial mean {mu}")

    result = _fit_controlled_pirls(deviance_for_mean)

    assert result.intercept == pytest.approx(0.5)
    assert result.deviance == pytest.approx(1.5)
    assert result.iteration_log is not None
    assert result.iteration_log[0].step_halvings == 1
    assert not result.iteration_log[0].step_rejected


def test_pirls_backtracking_reuses_the_endpoint_linear_predictor(monkeypatch) -> None:
    n = 6
    dm = DesignMatrix([DenseGroupMatrix(np.zeros((n, 1)))], n=n, p=1)
    original_matvec = dm.matvec
    matvec_calls = 0

    def counted_matvec(beta: np.ndarray) -> np.ndarray:
        nonlocal matvec_calls
        matvec_calls += 1
        return original_matvec(beta)

    monkeypatch.setattr(dm, "matvec", counted_matvec)

    def deviance_for_mean(mu: float) -> float:
        if np.isclose(mu, 0.0):
            return 2.0
        if np.isclose(mu, 0.5):
            return 1.5
        return 10.0

    result = fit_pirls(
        dm,
        np.ones(n),
        np.ones(n),
        _ControlledDevianceGaussian(deviance_for_mean),
        IdentityLink(),
        [GroupSlice(name="x", start=0, end=1)],
        GroupLasso(lambda1=0.01),
        beta_init=np.zeros(1),
        intercept_init=0.0,
        max_iter_outer=1,
        max_iter_inner=1,
        tol=1e-12,
    )

    assert result.intercept == pytest.approx(0.5)
    # Initial evaluation, working residual, and proposal evaluation. The
    # fixed-endpoint line-search trial must not multiply X by beta again.
    assert matvec_calls == 3


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
        ("reject", 0.0, 21, 1),
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
            return 1.0
        if outcome == "half" and np.isclose(mu, 0.5):
            return 1.5
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
    assert trial.payload["penalized_deviance"] is not None
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
            return 2.0 if np.isclose(mu, 0.0) else 1.5

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
            return 1.5
        raise AssertionError(f"unexpected trial mean {mu}")

    result = _fit_controlled_direct(deviance_for_mean)

    assert result.intercept == pytest.approx(0.5)
    assert result.deviance == pytest.approx(1.5)
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
            return 1.5
        raise AssertionError(f"unexpected trial mean {mu}")

    result = _fit_controlled_direct(deviance_for_mean)

    assert result.intercept == pytest.approx(0.5)
    assert result.deviance == pytest.approx(1.5)
    assert result.iteration_log is not None
    assert result.iteration_log[0].step_halvings == 1
    assert not result.iteration_log[0].step_rejected


def test_direct_backtracking_reuses_the_endpoint_linear_predictor(monkeypatch) -> None:
    n = 6
    dm = DesignMatrix([DenseGroupMatrix(np.zeros((n, 1)))], n=n, p=1)
    original_matvec = dm.matvec
    matvec_calls = 0

    def counted_matvec(beta: np.ndarray) -> np.ndarray:
        nonlocal matvec_calls
        matvec_calls += 1
        return original_matvec(beta)

    monkeypatch.setattr(dm, "matvec", counted_matvec)

    def deviance_for_mean(mu: float) -> float:
        if np.isclose(mu, 0.0):
            return 2.0
        if np.isclose(mu, 0.5):
            return 1.5
        return 10.0

    result, _ = fit_irls_direct(
        dm,
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
    )

    assert result.intercept == pytest.approx(0.5)
    # Initial and proposal evaluation only. The line-search trial reuses the
    # predictor direction between those two immutable endpoint states.
    assert matvec_calls == 2


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
        ("reject", 0.0, 21, 1),
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
            return 1.0
        if outcome == "half" and np.isclose(mu, 0.5):
            return 1.5
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


def test_direct_evaluation_trace_labels_trial_alpha_without_acceptance_claim() -> None:
    sink = MemoryTraceSink()
    _fit_controlled_direct(
        lambda mu: 2.0 if np.isclose(mu, 0.0) else 10.0,
        trace_run=TraceRun("direct-trial-alpha", sink=sink),
    )

    trial = next(
        event
        for event in sink.events
        if event.event_kind == "evaluation" and event.payload["phase"] == "line_search_trial"
    )
    assert trial.payload["trial_alpha"] == pytest.approx(0.5)
    assert trial.payload["penalized_deviance"] is not None
    assert "accepted_alpha" not in trial.payload


def test_noncanonical_direct_fit_commits_monotone_penalized_merit() -> None:
    """A difficult Poisson-sqrt fit must not publish legacy uphill steps."""
    rng = np.random.default_rng(10_001)
    n, p = 250, 3
    X = rng.normal(scale=0.6, size=(n, p))
    beta_true = rng.normal(scale=0.2, size=p)
    mu = (1.5 + X @ beta_true) ** 2
    y = rng.poisson(mu)
    beta_init = rng.normal(scale=2.5, size=p)
    intercept_init = float(rng.normal(scale=2.5))
    penalty = 0.01 * np.eye(p)
    dm = DesignMatrix([DenseGroupMatrix(X)], n=n, p=p)
    sink = MemoryTraceSink()

    result, _ = fit_irls_direct(
        dm,
        y,
        np.ones(n),
        Poisson(),
        SqrtLink(),
        [GroupSlice(name="x", start=0, end=p)],
        lambda2=0.01,
        S_override=penalty,
        beta_init=beta_init,
        intercept_init=intercept_init,
        max_iter=20,
        tol=1e-9,
        trace_run=TraceRun("poisson-sqrt-monotone", sink=sink),
    )

    decisions = [event for event in sink.events if event.event_kind == "step_decision"]
    commits = [event for event in sink.events if event.event_kind == "state_commit"]
    merits = np.array([event.payload["penalized_deviance"] for event in commits])
    tolerance = 64.0 * np.finfo(float).eps * np.maximum(1.0, np.abs(merits[:-1]))

    assert result.converged
    assert any(0.0 < event.payload["accepted_alpha"] < 1.0 for event in decisions)
    assert np.all(np.diff(merits) <= tolerance)


def test_direct_null_trace_does_not_change_results_or_evaluation_count() -> None:
    counts: list[int] = []
    results = []
    for trace_run in (None, TraceRun("direct-disabled", sink=NullTraceSink())):
        count = 0

        def deviance_for_mean(mu: float) -> float:
            nonlocal count
            count += 1
            return 2.0 if np.isclose(mu, 0.0) else 1.5

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
