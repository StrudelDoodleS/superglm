"""Typed penalty-geometry refusals at the scalar REML trial boundary."""

from __future__ import annotations

import numpy as np
import pytest

import superglm.reml.direct as direct
from superglm.distributions import Gaussian
from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
from superglm.links import IdentityLink
from superglm.reml.penalty_support import PenaltyNumericalError
from superglm.types import GroupSlice, PenaltyComponent


def _fit():
    """Use real coefficient fits, objectives and derivatives for a conditioned block."""
    x = np.linspace(-2.0, 2.0, 12)
    y = 1.0 + 0.8 * x + 0.5 * np.tile([-1.0, 1.0], 6)
    dm = DesignMatrix([DenseGroupMatrix(x[:, None])], n=len(x), p=1)
    group = GroupSlice("smooth", 0, 1)
    penalty = PenaltyComponent(
        name="smooth",
        group_name="smooth",
        group_index=0,
        group_sl=group.sl,
        omega_raw=None,
        omega_ssp=np.ones((1, 1)),
        rank=1.0,
        log_det_omega_plus=0.0,
        eigvals_omega=np.ones(1),
    )
    return direct.optimize_direct_reml(
        dm=dm,
        distribution=Gaussian(),
        link=IdentityLink(),
        groups=[group],
        discrete=False,
        y=y,
        sample_weight=np.ones(len(y)),
        offset_arr=np.zeros(len(y)),
        reml_groups=[(0, group)],
        penalty_ranks={"smooth": 1.0},
        lambdas={"smooth": 1.0},
        max_reml_iter=30,
        reml_tol=1e-8,
        verbose=False,
        reml_penalties=[penalty],
        direct_solve="gram",
        weight_semantics="frequency",
    )


def test_all_refused_trials_retain_evaluated_state_without_a_success_certificate(monkeypatch):
    real_objective = direct.reml_laml_objective
    evaluated = []
    refused = []

    def refuse_moves(*args, **kwargs):
        if evaluated:
            refused.append((dict(args[6]), args[5]))
            raise PenaltyNumericalError("trial penalty geometry is unresolved")
        evaluation = real_objective(*args, **kwargs)
        evaluated.append((dict(args[6]), args[5], evaluation.value))
        return evaluation

    monkeypatch.setattr(direct, "reml_laml_objective", refuse_moves)
    result = _fit()

    assert refused
    assert not result.converged
    assert result.termination_reason == "line_search_failed"
    lambdas, candidate, objective = evaluated[0]
    assert result.lambdas == lambdas
    assert result.pirls_result is candidate
    assert result.objective == objective
    assert result.objective_history == [objective]
    for refused_lambdas, refused_state in refused:
        assert refused_state is not result.pirls_result
        assert refused_lambdas not in result.lambda_history


def test_penalty_trial_refusal_backtracks_to_a_fully_evaluated_state(monkeypatch):
    real_objective = direct.reml_laml_objective
    evaluated = []
    refused = []

    def refuse_first_move(*args, **kwargs):
        if evaluated and not refused:
            refused.append(args[5])
            raise PenaltyNumericalError("first trial penalty geometry is unresolved")
        evaluation = real_objective(*args, **kwargs)
        evaluated.append((dict(args[6]), args[5], evaluation.value))
        return evaluation

    monkeypatch.setattr(direct, "reml_laml_objective", refuse_first_move)
    result = _fit()

    assert refused
    assert result.converged
    assert result.pirls_result is not refused[0]
    assert any(
        result.lambdas == lambdas and result.pirls_result is state and result.objective == objective
        for lambdas, state, objective in evaluated
    )
    assert result.objective <= evaluated[0][2]


@pytest.mark.parametrize("error_type", [PenaltyNumericalError, ValueError, np.linalg.LinAlgError])
def test_initial_objective_refusal_still_propagates(monkeypatch, error_type):
    error = error_type("initial objective cannot be evaluated")

    def fail_initial(*args, **kwargs):
        raise error

    monkeypatch.setattr(direct, "reml_laml_objective", fail_initial)
    with pytest.raises(error_type) as excinfo:
        _fit()
    assert excinfo.value is error


def test_current_objective_refusal_after_an_accepted_trial_still_propagates(monkeypatch):
    real_objective = direct.reml_laml_objective
    evaluated_lambdas = set()
    error = PenaltyNumericalError("retained candidate geometry is unresolved")

    def fail_repeated_candidate(*args, **kwargs):
        signature = tuple(sorted(args[6].items()))
        # An accepted trial is evaluated again as the next current candidate.
        if signature in evaluated_lambdas:
            raise error
        evaluation = real_objective(*args, **kwargs)
        evaluated_lambdas.add(signature)
        return evaluation

    monkeypatch.setattr(direct, "reml_laml_objective", fail_repeated_candidate)
    with pytest.raises(PenaltyNumericalError) as excinfo:
        _fit()
    assert excinfo.value is error


@pytest.mark.parametrize("error_type", [ValueError, np.linalg.LinAlgError, TypeError])
def test_trial_contract_and_untyped_linear_algebra_errors_still_propagate(monkeypatch, error_type):
    real_objective = direct.reml_laml_objective
    calls = 0
    error = error_type("trial caller contract or unrelated implementation failed")

    def fail_trial(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise error
        return real_objective(*args, **kwargs)

    monkeypatch.setattr(direct, "reml_laml_objective", fail_trial)
    with pytest.raises(error_type) as excinfo:
        _fit()
    assert excinfo.value is error
    assert calls == 2
