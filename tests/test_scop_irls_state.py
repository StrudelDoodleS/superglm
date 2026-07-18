"""Regression tests for atomic SCOP trials in latent coordinates."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superglm import Constraint, SuperGLM
from superglm._fit_trace import MemoryTraceSink, TraceRun
from superglm.distributions import Gaussian
from superglm.features.spline import PSpline
from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
from superglm.links import IdentityLink
from superglm.model.base import model_build_design_matrix
from superglm.solvers.irls_direct import (
    _evaluate_scop_trial,
    _SCOPGroupSpec,
    _SCOPGroupState,
    _SCOPTrialState,
)
from superglm.solvers.irls_state import (
    _evaluate_irls_state,
    _immutable_array,
    _IRLSStepDecision,
)
from superglm.solvers.scop import build_scop_solver_reparam
from superglm.solvers.scop_newton import scop_joint_newton_step
from superglm.types import GroupSlice


def test_scop_half_step_interpolates_latent_not_mapped_coefficients() -> None:
    n = 3
    ordinary = np.array([-1.0, 0.0, 1.0])
    scop_column = np.array([0.5, 1.0, 2.0])
    dm = DesignMatrix(
        [DenseGroupMatrix(ordinary[:, None]), DenseGroupMatrix(scop_column[:, None])],
        n=n,
        p=2,
    )
    family = Gaussian()
    link = IdentityLink()
    y = np.zeros(n)
    weights = np.ones(n)
    offset = np.zeros(n)
    group = GroupSlice(name="shape", start=1, end=2, monotone_engine="scop")
    reparam = build_scop_solver_reparam(q_raw=2, kind="increasing")
    spec = _SCOPGroupSpec(
        group_index=1,
        group=group,
        reparam=reparam,
        B_scop=scop_column[:, None],
        S_scop=np.zeros((1, 1)),
        bin_idx=None,
    )

    committed_beta = np.array([0.0, 1.0])
    proposed_beta = np.array([4.0, 4.0])
    committed = _SCOPTrialState(
        irls=_evaluate_irls_state(
            dm, y, weights, family, link, offset, committed_beta, intercept=2.0
        ),
        groups=(
            _SCOPGroupState(
                group_index=1,
                beta_eff=_immutable_array(np.array([0.0])),
                gamma_eff=_immutable_array(np.array([1.0])),
                H_scop_penalized=None,
                last_step_norm=0.0,
                last_fisher_fallback=False,
            ),
        ),
    )
    proposed = _SCOPTrialState(
        irls=_evaluate_irls_state(
            dm, y, weights, family, link, offset, proposed_beta, intercept=6.0
        ),
        groups=(
            _SCOPGroupState(
                group_index=1,
                beta_eff=_immutable_array(np.array([np.log(4.0)])),
                gamma_eff=_immutable_array(np.array([4.0])),
                H_scop_penalized=_immutable_array(np.array([[7.0]])),
                last_step_norm=np.log(4.0),
                last_fisher_fallback=True,
            ),
        ),
    )

    retained = _evaluate_scop_trial(
        committed=committed,
        proposed=proposed,
        alpha=0.5,
        specs={1: spec},
        dm=dm,
        y=y,
        weights=weights,
        family=family,
        link=link,
        offset=offset,
    )

    assert retained.irls.beta[0] == pytest.approx(2.0)
    assert retained.irls.intercept == pytest.approx(4.0)
    assert retained.groups[0].beta_eff[0] == pytest.approx(np.log(2.0))
    assert retained.groups[0].gamma_eff[0] == pytest.approx(2.0)
    assert retained.irls.beta[1] == pytest.approx(2.0)
    assert retained.irls.beta[1] != pytest.approx(2.5)
    assert retained.groups[0].H_scop_penalized is None
    assert retained.groups[0].last_step_norm == pytest.approx(np.log(2.0))
    assert retained.groups[0].last_fisher_fallback


def _scop_fit_inputs():
    x = np.linspace(0.0, 1.0, 80)
    y = 0.5 + 2.0 * x + 0.02 * np.sin(8.0 * x)
    frame = pd.DataFrame({"x": x})
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        discrete=True,
        features={"x": PSpline(n_knots=6, constraint=Constraint.fit.increasing)},
    )
    y_out, weights, offset = model_build_design_matrix(
        model,
        frame,
        y,
        np.ones_like(y),
        None,
    )
    return model, y_out, weights, offset


def test_well_scaled_scop_fit_avoids_anchor_prediction_fallback(monkeypatch) -> None:
    """Ordinary SCOP iterations retain the native matrix matvec hot path."""
    import superglm.solvers.irls_direct as irls_direct

    model, y, weights, offset = _scop_fit_inputs()

    def unexpected_anchor_matvec(**kwargs):
        raise AssertionError("anchor prediction is reserved for unsafe translated designs")

    monkeypatch.setattr(irls_direct, "stable_centered_matvec", unexpected_anchor_matvec)
    result, _ = irls_direct.fit_irls_direct(
        model._dm,
        y,
        weights,
        model._distribution,
        model._link,
        model._groups,
        lambda2={"x": 1.0},
        offset=offset,
        max_iter=2,
    )

    assert result.converged or result.n_iter == 2


def test_first_scop_rejection_retains_initialized_latent_bundle(monkeypatch) -> None:
    import superglm.solvers.irls_direct as irls_direct

    model, y, weights, offset = _scop_fit_inputs()

    def reject_all(**kwargs):
        for depth in range(1, 6):
            kwargs["evaluate_state"](2.0**-depth)
        return _IRLSStepDecision(0.0, 0, True, trials_attempted=6)

    monkeypatch.setattr(irls_direct, "_select_irls_trial", reject_all)
    sink = MemoryTraceSink()

    result, _, states = irls_direct.fit_irls_direct(
        model._dm,
        y,
        weights,
        model._distribution,
        model._link,
        model._groups,
        lambda2={"x": 1.0},
        offset=offset,
        max_iter=1,
        record_diagnostics=True,
        return_scop_state=True,
        trace_run=TraceRun("scop-reject", sink=sink),
    )

    assert not result.converged
    assert result.iteration_log is not None
    assert result.iteration_log[0].step_rejected
    state = next(iter(states.values()))
    gamma = state["reparam"].forward(state["beta_eff"])
    np.testing.assert_allclose(state["gamma_eff"], gamma)
    np.testing.assert_allclose(result.beta[state["group_sl"]], gamma)
    assert state["H_scop_penalized"] is not None
    assert np.all(np.isfinite(state["H_scop_penalized"]))
    assert state["last_step_norm"] == 0.0
    assert not state["last_fisher_fallback"]
    offset_arr = np.zeros_like(y) if offset is None else offset
    mu = result.intercept + model._dm.matvec(result.beta) + offset_arr
    expected_deviance = float(np.sum(weights * (y - mu) ** 2))
    assert result.deviance == pytest.approx(expected_deviance)
    decision = next(event for event in sink.events if event.event_kind == "step_decision")
    commits = [event for event in sink.events if event.event_kind == "state_commit"]
    evaluated = {
        event.payload["state_id"] for event in sink.events if event.event_kind == "evaluation"
    }
    assert decision.payload["trials_attempted"] == 6
    assert decision.payload["committed_state_id"] == decision.payload["base_state_id"]
    assert commits[-1].payload["state_id"] == decision.payload["base_state_id"]
    assert all(commit.payload["state_id"] in evaluated for commit in commits)
    assert result.state_id == decision.payload["base_state_id"]


def test_scop_rejection_restores_warm_hessian_step_and_fisher_cache(monkeypatch) -> None:
    import superglm.solvers.irls_direct as irls_direct

    model, y, weights, offset = _scop_fit_inputs()
    baseline, _, baseline_states = irls_direct.fit_irls_direct(
        model._dm,
        y,
        weights,
        model._distribution,
        model._link,
        model._groups,
        lambda2={"x": 1.0},
        offset=offset,
        max_iter=5,
        return_scop_state=True,
    )
    baseline_state = next(iter(baseline_states.values()))
    monkeypatch.setattr(
        irls_direct,
        "_select_irls_trial",
        lambda **kwargs: _IRLSStepDecision(0.0, 0, True, trials_attempted=6),
    )

    rejected, _, rejected_states = irls_direct.fit_irls_direct(
        model._dm,
        y,
        weights,
        model._distribution,
        model._link,
        model._groups,
        lambda2={"x": 1.0},
        offset=offset,
        beta_init=baseline.beta,
        intercept_init=baseline.intercept,
        max_iter=1,
        record_diagnostics=True,
        return_scop_state=True,
        scop_state_init=baseline_states,
    )

    rejected_state = next(iter(rejected_states.values()))
    np.testing.assert_array_equal(rejected.beta, baseline.beta)
    assert rejected.intercept == baseline.intercept
    np.testing.assert_array_equal(rejected_state["beta_eff"], baseline_state["beta_eff"])
    np.testing.assert_array_equal(rejected_state["gamma_eff"], baseline_state["gamma_eff"])
    np.testing.assert_array_equal(
        rejected_state["H_scop_penalized"], baseline_state["H_scop_penalized"]
    )
    assert rejected_state["last_step_norm"] == baseline_state["last_step_norm"]
    assert rejected_state["last_fisher_fallback"] == baseline_state["last_fisher_fallback"]
    assert rejected.iteration_log is not None
    assert rejected.iteration_log[0].step_rejected


def test_scop_half_step_refreshes_gamma_deviance_and_retained_hessian(monkeypatch) -> None:
    import superglm.solvers.irls_direct as irls_direct

    model, y, weights, offset = _scop_fit_inputs()

    def select_half(**kwargs):
        kwargs["evaluate_state"](0.5)
        return _IRLSStepDecision(0.5, 1, False, trials_attempted=2)

    monkeypatch.setattr(irls_direct, "_select_irls_trial", select_half)
    sink = MemoryTraceSink()
    result, _, states = irls_direct.fit_irls_direct(
        model._dm,
        y,
        weights,
        model._distribution,
        model._link,
        model._groups,
        lambda2={"x": 1.0},
        offset=offset,
        max_iter=1,
        record_diagnostics=True,
        return_scop_state=True,
        trace_run=TraceRun("scop-half", sink=sink),
    )

    gi, state = next(iter(states.items()))
    gamma = state["reparam"].forward(state["beta_eff"])
    np.testing.assert_allclose(state["gamma_eff"], gamma)
    np.testing.assert_allclose(result.beta[state["group_sl"]], gamma)
    offset_arr = np.zeros_like(y) if offset is None else offset
    mu = result.intercept + model._dm.matvec(result.beta) + offset_arr
    assert result.deviance == pytest.approx(float(np.sum(weights * (y - mu) ** 2)))
    assert result.iteration_log is not None
    assert result.iteration_log[0].step_halvings == 1
    assert not result.iteration_log[0].step_rejected
    decision = next(event for event in sink.events if event.event_kind == "step_decision")
    commits = [event for event in sink.events if event.event_kind == "state_commit"]
    trial = next(
        event
        for event in sink.events
        if event.event_kind == "evaluation" and event.payload["phase"] == "scop_line_search_trial"
    )
    assert decision.payload["trials_attempted"] == 2
    assert decision.payload["committed_state_id"] == trial.payload["state_id"]
    assert trial.payload["enclosing_proposal_state_id"] == decision.payload["proposal_state_id"]
    assert commits[-1].payload["state_id"] == trial.payload["state_id"]
    assert result.state_id == trial.payload["state_id"]

    expected_input = {
        gi: {
            "beta_scop": state["beta_eff"],
            "reparam": state["reparam"],
            "B_scop": state["B_scop"],
            "S_scop": state["S_scop"],
            "bin_idx": state["bin_idx"],
        }
    }
    z_scop = y - offset_arr - result.intercept
    expected = scop_joint_newton_step(
        expected_input,
        weights,
        z_scop,
        {"x": 1.0},
        model._groups,
    )[gi].H_penalized
    np.testing.assert_allclose(state["H_scop_penalized"], expected)


def test_scop_terminal_refresh_replaces_stale_fisher_fallback_flag(monkeypatch) -> None:
    """The fallback flag must describe the refreshed terminal Hessian block."""
    from dataclasses import replace

    import superglm.solvers.irls_direct as irls_direct

    model, y, weights, offset = _scop_fit_inputs()
    original = irls_direct.scop_joint_newton_step

    def mark_terminal_refresh(*args, **kwargs):
        results = original(*args, **kwargs)
        if "debug_context" not in kwargs:
            return {
                gi: replace(result, used_fisher_fallback=True) for gi, result in results.items()
            }
        return results

    monkeypatch.setattr(irls_direct, "scop_joint_newton_step", mark_terminal_refresh)
    _, _, states = irls_direct.fit_irls_direct(
        model._dm,
        y,
        weights,
        model._distribution,
        model._link,
        model._groups,
        lambda2={"x": 1.0},
        offset=offset,
        max_iter=1,
        return_scop_state=True,
    )

    assert all(state["last_fisher_fallback"] for state in states.values())


def test_scop_trace_merit_uses_authoritative_latent_penalty(monkeypatch) -> None:
    """SCOP merit uses lambda*S_scop in beta-space, not the mapped global block."""
    import superglm.solvers.irls_direct as irls_direct

    model, y, weights, offset = _scop_fit_inputs()
    monkeypatch.setattr(
        irls_direct,
        "_select_irls_trial",
        lambda **kwargs: _IRLSStepDecision(0.0, 0, True, trials_attempted=1),
    )
    sink = MemoryTraceSink()
    lam = 2.5
    mapped_override = 37.0 * np.eye(model._dm.p)

    result, _, states = irls_direct.fit_irls_direct(
        model._dm,
        y,
        weights,
        model._distribution,
        model._link,
        model._groups,
        lambda2={"x": lam},
        S_override=mapped_override,
        offset=offset,
        max_iter=1,
        return_scop_state=True,
        trace_run=TraceRun("scop-latent-merit", sink=sink),
    )

    state = next(iter(states.values()))
    initial = next(
        event
        for event in sink.events
        if event.event_kind == "evaluation" and event.payload["phase"] == "initial"
    )
    expected = result.deviance + lam * float(
        state["beta_eff"] @ state["S_scop"] @ state["beta_eff"]
    )
    assert initial.payload["penalized_deviance"] == pytest.approx(expected)


def test_poisson_scop_terminal_inference_keeps_known_dispersion() -> None:
    """Installing terminal SCOP EDF must not profile a known family scale."""
    import superglm.solvers.irls_direct as irls_direct

    rng = np.random.default_rng(91)
    x = np.linspace(0.0, 1.0, 120)
    y = rng.poisson(np.exp(0.2 + 0.7 * x)).astype(float)
    frame = pd.DataFrame({"x": x})
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        discrete=True,
        features={"x": PSpline(n_knots=6, constraint=Constraint.fit.increasing)},
    )
    y_out, weights, offset = model_build_design_matrix(
        model,
        frame,
        y,
        np.ones_like(y),
        None,
    )

    result, _ = irls_direct.fit_irls_direct(
        model._dm,
        y_out,
        weights,
        model._distribution,
        model._link,
        model._groups,
        lambda2={"x": 1.0},
        offset=offset,
    )

    assert result.phi == 1.0
