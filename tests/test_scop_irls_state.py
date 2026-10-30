"""Regression tests for atomic SCOP trials in latent coordinates."""

from __future__ import annotations

import warnings

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
    _SCOPGroupState,
    _SCOPTrialState,
)
from superglm.solvers.irls_state import (
    _evaluate_irls_state,
    _immutable_array,
    _IRLSStepDecision,
)
from superglm.solvers.scop import build_scop_solver_reparam
from superglm.solvers.scop_newton import SCOPGroupSpec, SCOPJointNewton
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
    spec = SCOPGroupSpec(
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
        features={
            "x": PSpline(n_knots=6, constraint=Constraint.fit.increasing, shape_engine="scop")
        },
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
    import superglm.solvers.irls_direct._fit as irls_direct
    import superglm.solvers.irls_direct._scop as irls_scop

    model, y, weights, offset = _scop_fit_inputs()

    def unexpected_centred_matvec(*args, **kwargs):
        raise AssertionError("centred prediction is reserved for ill-scaled columns")

    monkeypatch.setattr(irls_scop, "_centred_system_matvec", unexpected_centred_matvec)
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
        weight_semantics="frequency",
    )

    assert result.converged or result.n_iter == 2


def test_first_scop_rejection_retains_initialized_latent_bundle(monkeypatch) -> None:
    import superglm.solvers.irls_direct._fit as irls_direct
    import superglm.solvers.irls_direct._scop as irls_scop

    model, y, weights, offset = _scop_fit_inputs()

    def reject_all(**kwargs):
        for depth in range(1, 6):
            kwargs["evaluate_state"](2.0**-depth)
        return _IRLSStepDecision(0.0, 0, True, trials_attempted=6)

    monkeypatch.setattr(irls_scop, "_select_irls_trial", reject_all)
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
        weight_semantics="frequency",
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
    import superglm.solvers.irls_direct._fit as irls_direct
    import superglm.solvers.irls_direct._scop as irls_scop

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
        weight_semantics="frequency",
    )
    baseline_state = next(iter(baseline_states.values()))
    monkeypatch.setattr(
        irls_scop,
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
        weight_semantics="frequency",
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
    import superglm.solvers.irls_direct._fit as irls_direct
    import superglm.solvers.irls_direct._scop as irls_scop

    model, y, weights, offset = _scop_fit_inputs()

    def select_half(**kwargs):
        kwargs["evaluate_state"](0.5)
        return _IRLSStepDecision(0.5, 1, False, trials_attempted=2)

    monkeypatch.setattr(irls_scop, "_select_irls_trial", select_half)
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
        weight_semantics="frequency",
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

    # The retained block is the SCOP block's own Newton Hessian at the retained
    # state: J B'WB J + lambda S + E, with E = -gamma''(theta) B'W r.
    reparam, theta = state["reparam"], state["beta_eff"]
    rows = state["B_scop"] if state["bin_idx"] is None else state["B_scop"][state["bin_idx"]]
    jacobian = reparam.jacobian_diagonal(theta)
    expected = jacobian[:, None] * (rows.T @ (weights[:, None] * rows)) * jacobian[None, :]
    expected += 1.0 * state["S_scop"]
    assert not state["last_fisher_fallback"]
    assert state["discarded_directions"].shape[0] == 0
    score = rows.T @ (weights * (y - mu))
    expected += np.diag(-reparam.second_derivative_diagonal(theta) * score)
    np.testing.assert_allclose(state["H_scop_penalized"], expected)


@pytest.mark.parametrize(
    ("alpha", "halvings", "rejected"),
    [(0.5, 1, False), (0.25, 2, False), (1.0, 0, True)],
)
def test_recorder_captures_step_quality_fields(monkeypatch, alpha, halvings, rejected) -> None:
    """The recorder must carry the step-quality fields, not just the deviance.

    Every record the SCOP corpus produces carries ``step_rejected=False`` and
    ``step_halvings=0`` -- measured: 69 records, 2 inner fits, zero of each --
    so a fixture-driven test pins the deviance and nothing else. This drives
    both fields directly instead of hoping a fixture reaches them.
    """
    import superglm.solvers.irls_direct._fit as irls_direct
    import superglm.solvers.irls_direct._scop as irls_scop

    model, y, weights, offset = _scop_fit_inputs()

    def forced_decision(**kwargs):
        kwargs["evaluate_state"](alpha)
        return _IRLSStepDecision(alpha, halvings, rejected, trials_attempted=halvings + 1)

    monkeypatch.setattr(irls_scop, "_select_irls_trial", forced_decision)
    result, _ = irls_direct.fit_irls_direct(
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
        weight_semantics="frequency",
    )

    assert result.iteration_log is not None
    assert result.iteration_log[0].step_halvings == halvings
    assert result.iteration_log[0].step_rejected is rejected


def test_scop_terminal_refresh_replaces_stale_fisher_fallback_flag(monkeypatch) -> None:
    """The fallback flag must describe the refreshed terminal Hessian block."""
    from dataclasses import replace

    import superglm.solvers.irls_direct._fit as irls_direct

    model, y, weights, offset = _scop_fit_inputs()
    original = SCOPJointNewton.block_curvature

    def mark_terminal_refresh(self, *args, **kwargs):
        results = original(self, *args, **kwargs)
        return {gi: replace(result, used_fisher_fallback=True) for gi, result in results.items()}

    monkeypatch.setattr(SCOPJointNewton, "block_curvature", mark_terminal_refresh)
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
        weight_semantics="frequency",
    )

    assert all(state["last_fisher_fallback"] for state in states.values())


def test_scop_trace_merit_uses_authoritative_latent_penalty(monkeypatch) -> None:
    """SCOP merit uses lambda*S_scop in beta-space, not the mapped global block."""
    import superglm.solvers.irls_direct._fit as irls_direct
    import superglm.solvers.irls_direct._scop as irls_scop

    model, y, weights, offset = _scop_fit_inputs()
    monkeypatch.setattr(
        irls_scop,
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
        weight_semantics="frequency",
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
    import superglm.solvers.irls_direct._fit as irls_direct

    rng = np.random.default_rng(91)
    x = np.linspace(0.0, 1.0, 120)
    y = rng.poisson(np.exp(0.2 + 0.7 * x)).astype(float)
    frame = pd.DataFrame({"x": x})
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        discrete=True,
        features={
            "x": PSpline(n_knots=6, constraint=Constraint.fit.increasing, shape_engine="scop")
        },
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
        weight_semantics="frequency",
    )

    assert result.phi == 1.0


def test_scop_fit_never_materialises_design_for_frequent_indicator(monkeypatch) -> None:
    """A bounded column whose weighted mean exceeds its spread keeps every product compact.

    Level ``b`` holds about 3/4 of the rows, so its indicator's mean exceeds its
    centred RMS (ratio ``sqrt(p / (1 - p))``, about 1.7) and
    ``_raw_centering_well_scaled`` rejects on every iteration.  The reduced
    system was centred first by ``build_centered_system``, so the rejection
    may only change how eta is formed; rebuilding that system from row chunks,
    three passes an iteration, and centring the final full system (SCOP
    columns included) in row chunks made a 678k-row fit tens of times slower
    once one indicator's weighted frequency crossed 1/2.
    """
    import superglm.solvers.irls_direct._fit as irls_direct
    import superglm.solvers.irls_direct._scop as irls_scop
    from superglm.features.categorical import Categorical

    rng = np.random.default_rng(20261004)
    n = 400
    x = np.sort(rng.uniform(0.0, 1.0, size=n))
    level = np.where(rng.uniform(size=n) < 0.75, "b", "a")
    y = 0.5 + 2.0 * x + 0.3 * (level == "b") + rng.normal(scale=0.05, size=n)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        discrete=True,
        features={
            "x": PSpline(n_knots=6, constraint=Constraint.fit.increasing, shape_engine="scop"),
            "g": Categorical(base="a"),
        },
    )
    y_out, weights, offset = model_build_design_matrix(
        model, pd.DataFrame({"x": x, "g": level}), y, np.ones_like(y), None
    )

    verdicts: list[bool] = []
    guard = irls_scop._raw_centering_well_scaled

    def recording_guard(mean_x, scale):
        verdicts.append(guard(mean_x, scale))
        return verdicts[-1]

    materialised: list[int] = []
    row_subset = DesignMatrix.row_subset

    def recording_row_subset(self, idx):
        materialised.append(len(idx))
        return row_subset(self, idx)

    monkeypatch.setattr(irls_scop, "_raw_centering_well_scaled", recording_guard)
    monkeypatch.setattr(DesignMatrix, "row_subset", recording_row_subset)
    result, _ = irls_direct.fit_irls_direct(
        model._dm,
        y_out,
        weights,
        model._distribution,
        model._link,
        model._groups,
        lambda2={"x": 1.0},
        offset=offset,
        max_iter=4,
        weight_semantics="frequency",
    )

    assert verdicts and not any(verdicts), "the frequent indicator must trip the certificate"
    assert sum(materialised) == 0, f"materialised {sum(materialised)} design rows"
    assert result.converged or result.n_iter == 4


def test_centred_scop_prediction_rounds_at_the_centred_rows() -> None:
    """``(X - 1 m') beta`` about the system's pair: error bounded by the centred rows.

    Exact-rational reference on the float inputs.  Each term passes through at
    most ``p + 4`` roundings (the centre's ``L``-term dot, a subtraction and
    the accumulation, the pair's remainder and its subtraction), so ``|t -
    t*| <= gamma_{p+4} (|x - hi| |beta| + |v| + sum |m| |beta| + |lo| |beta|)``
    (Higham 2002, sec. 3.1).  A row formed natively, ``x beta - m' beta``,
    rounds at ``u |x| |beta|``, about 1e10 times this bound.
    """
    from fractions import Fraction

    from superglm.group_matrix import CategoricalGroupMatrix
    from superglm.solvers.centered_system import build_centered_system
    from superglm.solvers.irls_direct import _centred_system_matvec

    rng = np.random.default_rng(20261004)
    n, levels = 64, 2
    translated = 1.0e10 + rng.normal(size=n)
    codes = np.where(rng.uniform(size=n) < 0.75, 1, rng.integers(-1, 1, size=n))
    dm = DesignMatrix(
        [DenseGroupMatrix(translated[:, None]), CategoricalGroupMatrix(codes, levels)],
        n=n,
        p=1 + levels,
    )
    weights = rng.uniform(0.5, 2.0, size=n)
    system = build_centered_system(
        dm=dm, W=weights, z_off=rng.normal(size=n), penalty=np.zeros((1 + levels, 1 + levels))
    )
    beta = rng.normal(size=1 + levels)

    computed = _centred_system_matvec(dm, beta, system)

    hi, lo = system.centre_pair()
    assert lo is not None and system.mean_x[2] > 0.5  # the pair, and a frequent level
    u = Fraction(2) ** -53
    k = Fraction(dm.p + 4)
    gamma = k * u / (1 - k * u)
    rows = dm.toarray()
    for i in range(n):
        exact = (Fraction(rows[i, 0]) - Fraction(hi[0]) - Fraction(lo[0])) * Fraction(beta[0])
        scale = abs(Fraction(rows[i, 0]) - Fraction(hi[0])) * abs(Fraction(beta[0]))
        scale += abs(Fraction(lo[0]) * Fraction(beta[0]))
        for j in range(1, dm.p):
            exact += (Fraction(rows[i, j]) - Fraction(hi[j])) * Fraction(beta[j])
            scale += (Fraction(rows[i, j]) + abs(Fraction(hi[j]))) * abs(Fraction(beta[j]))
        assert abs(Fraction(computed[i]) - exact) <= gamma * scale, i


def _scop_tensor_frequent_level_fit(monkeypatch, counts: dict):
    """A small monotone REML fit whose centring rungs reject on every fit: run counts, predictions."""
    import superglm._group_matrix._group_matrix_centered as centered
    import superglm.reml.scop_efs as scop_efs
    import superglm.solvers.centered_system as centered_system
    from superglm.features.categorical import Categorical

    def count(name, function):
        def counted(*args, **kwargs):
            counts[name] = counts.get(name, 0) + 1
            return function(*args, **kwargs)

        return counted

    monkeypatch.setattr(centered, "_MIN_MIXED_RAW_MOMENT_CELLS", 0)  # let the raw rung run
    for module, name in (
        (centered, "_build_pattern_plan"),
        (centered, "_try_pattern_tensor_centering"),
        (centered_system, "try_raw_moment_centering"),
        (scop_efs, "fit_irls_direct"),
    ):
        monkeypatch.setattr(module, name, count(name, getattr(module, name)))
    rng = np.random.default_rng(20261004)
    n = 1500
    frame = pd.DataFrame(
        {
            "x": rng.uniform(0.0, 1.0, n),
            "a": rng.uniform(0.0, 1.0, n),
            "b": rng.uniform(0.0, 1.0, n),
            "g": np.where(rng.uniform(size=n) < 0.75, "frequent", "rare"),
        }
    )
    eta = -1.0 + 0.8 * frame["x"] + 0.4 * np.sin(3.0 * frame["a"] * frame["b"])
    y = rng.poisson(np.exp(eta + 0.3 * (frame["g"] == "frequent"))).astype(float)
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        discrete=True,
        features={
            "x": PSpline(n_knots=6, constraint=Constraint.fit.increasing, shape_engine="scop"),
            "a": PSpline(n_knots=5),
            "b": PSpline(n_knots=5),
            "g": Categorical(base="rare"),
        },
        interactions=[("a", "b")],
    )
    model.fit_reml(frame, y, max_reml_iter=3)
    return model, np.asarray(model.predict(frame))


def test_scop_reml_run_builds_its_reduced_plan_once_and_carries_refusals(monkeypatch) -> None:
    """One pattern plan and one attempt of each rejected rung per REML run, bitwise unchanged.

    The frequent level's indicator trips the raw-centring certificate under every
    fit's weights, so the tensor pattern rung (reduced design) and the
    raw-moment rung (full design) reject each time.  Each PIRLS fit rebuilt the
    reduced design, its pattern plan, and both rejected rungs: 11 s of rejected
    raw moments and 5 s of plan rebuilds on the 678k-row fit.  A refusal
    carried through the run (``SCOPRunCentring``) leaves every fit on the
    route it reached anyway, so the run must match a run with a fresh state
    per fit bit for bit.
    """
    from superglm.solvers.irls_direct import SCOPRunCentring

    counts: dict = {}
    model, predictions = _scop_tensor_frequent_level_fit(monkeypatch, counts)
    fits = counts["fit_irls_direct"]
    assert fits >= 4
    assert counts["_build_pattern_plan"] == 1
    assert counts["_try_pattern_tensor_centering"] == 1
    assert counts["try_raw_moment_centering"] == 1

    original_bind = SCOPRunCentring.bind

    def fresh_bind(self, owners):
        self.owners = ()
        original_bind(self, owners)

    fresh_counts: dict = {}
    monkeypatch.setattr(SCOPRunCentring, "bind", fresh_bind)
    fresh_model, fresh_predictions = _scop_tensor_frequent_level_fit(monkeypatch, fresh_counts)
    assert fresh_counts["fit_irls_direct"] == fits
    assert fresh_counts["_build_pattern_plan"] == fits  # the cost the carry removes
    np.testing.assert_array_equal(predictions, fresh_predictions)
    assert model.result.deviance == fresh_model.result.deviance
    assert model.reml_diagnostics()["lambdas"] == fresh_model.reml_diagnostics()["lambdas"]


def test_the_solver_refuses_scop_beside_linear_constraints() -> None:
    """``fit_irls_direct`` refuses a SCOP group beside a linearly constrained group.

    Every model entry point refuses the mix (``test_monotone_fit.py::
    test_scop_plus_qp_raises``).  The solver's SCOP step has no constrained QP,
    so it would check the linear constraint after each step and never enforce
    it; refusing at the solver's entry leaves each fit one step method.
    Mutation: without ``_refuse_scop_with_linear_constraints`` this fit runs.
    """
    from superglm.solvers.irls_direct import fit_irls_direct
    from superglm.types import LinearConstraintSet

    rng = np.random.default_rng(3)
    n = 40
    bounded = rng.uniform(-1.0, 1.0, n)
    shaped = rng.uniform(0.0, 1.0, n)
    dm = DesignMatrix(
        [DenseGroupMatrix(bounded[:, None]), DenseGroupMatrix(shaped[:, None])], n=n, p=2
    )
    groups = [
        GroupSlice(
            name="bounded",
            start=0,
            end=1,
            constraints=LinearConstraintSet(A=np.eye(1), b=np.zeros(1)),
            monotone_engine="qp",
        ),
        GroupSlice(
            name="shape",
            start=1,
            end=2,
            monotone_engine="scop",
            scop_reparameterization=build_scop_solver_reparam(q_raw=2, kind="increasing"),
        ),
    ]
    y = 1.0 + bounded + shaped + rng.normal(0.0, 0.1, n)
    with pytest.raises(
        NotImplementedError,
        match=r"'shape' is shape-constrained and 'bounded' has linear constraints",
    ):
        fit_irls_direct(
            dm,
            y,
            np.ones(n),
            Gaussian(),
            IdentityLink(),
            groups,
            lambda2=1.0,
            weight_semantics="frequency",
        )


def _edge_of_range_data(case: str):
    """A monotone-in-``x`` response whose SCOP start or step meets the edge of the link's range."""
    rng = np.random.default_rng(3)
    x = np.sort(rng.uniform(0, 1, 80))
    if case == "binomial_log":
        y = (rng.uniform(size=80) < 0.9 + 0.09 * x).astype(float)
        return x, y, np.full(80, 2.0), "binomial", "log"
    if case == "gamma_inverse":
        mu = 1.0 / (0.05 + 1.0 * x)
        _ = rng.uniform(size=80)
        return x, rng.gamma(20, mu / 20), None, "gamma", "inverse"
    mu = np.where(x < 0.3, 0.0, 6 * (x - 0.3))
    return x, rng.poisson(mu).astype(float), None, "poisson", "identity"


def _fit_edge_of_range(case: str) -> tuple[SuperGLM, np.ndarray]:
    x, y, offset, family, link = _edge_of_range_data(case)
    model = SuperGLM(
        family=family,
        link=link,
        selection_penalty=0.0,
        features={
            "x": PSpline(n_knots=5, constraint=Constraint.fit.increasing, shape_engine="scop")
        },
    )
    model.fit(pd.DataFrame({"x": x}), y, offset=offset)
    eta = model._dm.matvec(model.result.beta) + model.result.intercept
    return model, eta + (0.0 if offset is None else offset)


def test_the_scop_search_refuses_a_trial_that_leaves_the_mean_space(monkeypatch) -> None:
    """A binomial/log SCOP fit from a start inside the space never leaves it.

    The first proposal's linear predictor is positive on some row, a binomial
    mean above one; the search refuses it and every trial like it.  Mutation:
    the search given only its observed-row predicate retains rows at
    ``eta >= 0`` (41 of 80 after 73 iterations).
    """
    import superglm.solvers.irls_direct._scop as irls_scop

    real = irls_scop._select_irls_trial
    proposal_refused: list[bool] = []

    def spy(**kwargs):
        invalid = kwargs["invalid_state"]
        proposal_refused.append(invalid is not None and invalid(kwargs["proposal"]))
        return real(**kwargs)

    monkeypatch.setattr(irls_scop, "_select_irls_trial", spy)
    model, eta = _fit_edge_of_range("binomial_log")
    assert proposal_refused[0]
    assert float(np.max(eta)) < 0.0


@pytest.mark.parametrize("case", ["poisson_identity", "gamma_inverse"])
def test_the_scop_search_refuses_a_trial_that_leaves_the_links_domain(
    monkeypatch, case: str
) -> None:
    """The refusal the SCOP step hands its line search keeps every mean the link allows.

    The event is forced: the refusal is read where the step hands it over and
    applied to the committed state with one positive-weight row moved to
    ``eta = 0``, which the identity link gives a Poisson mean of zero and the
    inverse link none, so no trajectory need propose such a trial.  Mutation:
    the SCOP search given only the mean-space refusal accepts it, and the
    characterisation's Poisson/identity SCOP REML fit returns to deviance 2028
    with an effective df of 1.4e48.
    """
    import dataclasses

    import superglm.solvers.irls_direct._scop as irls_scop

    real = irls_scop._select_irls_trial
    handed: list[tuple] = []

    def spy(**kwargs):
        handed.append((kwargs["committed"], kwargs["invalid_state"]))
        return real(**kwargs)

    monkeypatch.setattr(irls_scop, "_select_irls_trial", spy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _fit_edge_of_range(case)
    committed, invalid = handed[0]
    eta = np.array(committed.eta_unclipped)
    assert np.all(eta > 0.0)
    eta[0] = 0.0
    assert invalid is not None
    assert invalid(dataclasses.replace(committed, eta_unclipped=eta))


@pytest.mark.parametrize("case", ["poisson_identity", "gamma_inverse", "binomial_log"])
def test_the_scop_start_is_put_inside_the_mean_space_and_the_links_domain(
    monkeypatch, case: str
) -> None:
    """The latent start moves the shape term after the start intercept was chosen inside.

    The QP start is steepened (each shape coordinate's increment ``e^3``
    times its own), so the mapped start leaves the link's domain (identity,
    inverse: low ``x``) or the binomial mean space (log: high ``x``) on every
    case.  The start the first iteration commits from is inside both.
    Mutation: without the second move all three cases start outside.
    """
    import superglm.solvers.irls_direct._scop as irls_scop
    from superglm.solvers.irls_state import link_domain_violation, mean_space_violation
    from superglm.solvers.scop import SCOPSolverReparam

    real_initialize = SCOPSolverReparam.qp_initialize

    def steepened(self, *args, **kwargs):
        start = real_initialize(self, *args, **kwargs)
        start[self.free_dim :] += 3.0
        return start

    starts: list[np.ndarray] = []
    real_initial_state = irls_scop._IRLSDirectSCOPStep.initial_state

    def recorded(self, fit, committed):
        starts.append(np.asarray(committed.eta_unclipped).copy())
        return real_initial_state(self, fit, committed)

    monkeypatch.setattr(SCOPSolverReparam, "qp_initialize", steepened)
    monkeypatch.setattr(irls_scop._IRLSDirectSCOPStep, "initial_state", recorded)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, _ = _fit_edge_of_range(case)
    family, link = model._distribution, model._link
    weights = np.ones(starts[0].size)
    for outside in (mean_space_violation(family, link), link_domain_violation(link, family)):
        assert outside is None or not outside(starts[0], weights)


@pytest.mark.parametrize("case", ["poisson_identity", "gamma_inverse"])
def test_a_scop_fit_left_outside_the_links_range_says_it_is_not_converged(
    monkeypatch, case: str
) -> None:
    """A returned state outside the link's range is no fitted model, whatever stopped the loop.

    The event is forced: the start is left where the latent start put it,
    outside the range on both fixtures, and every step is rejected, so the
    fit returns that start.  It ends not converged, with a
    ``ConvergenceWarning`` that names the cause, and a reason no observed-REML
    gate reads as a mode short of its certificate.  Mutation: the
    returned-state verdict removed publishes both fits with no warning; the
    loop's own reason kept (``step_rejected``) passes those gates.
    """
    import superglm.solvers.irls_direct._scop as irls_scop
    from superglm.diagnostics.convergence import ConvergenceWarning
    from superglm.reml.observed_geometry import stopped_on_iteration_budget

    monkeypatch.setattr(
        irls_scop,
        "interior_start_intercept",
        lambda family, link, eta, w, intercept, level: intercept,
    )
    monkeypatch.setattr(
        irls_scop,
        "_select_irls_trial",
        lambda **kwargs: _IRLSStepDecision(alpha=0.0, step_halvings=0, step_rejected=True),
    )
    with pytest.warns(ConvergenceWarning, match="outside the range"):
        model, eta = _fit_edge_of_range(case)
    assert not model.result.converged
    assert model.result.termination_reason == "link_domain_infeasible"
    assert not stopped_on_iteration_budget(model.result)
    # the identity and inverse links both need a positive predictor for a positive mean
    assert int(np.sum(eta <= 0.0)) > 0


def test_a_scop_trial_carries_the_rounding_allowance_of_its_own_merit(monkeypatch) -> None:
    """The merit's allowance is a field of the trial state, read for the state the search holds.

    The search reads the allowance of the candidate and of the committed state
    it is handed; each is the one ``_with_merit`` formed for that state, a
    finite nonnegative number, and a state the search does not hold has none.
    """
    import superglm.solvers.irls_direct._fit as irls_direct
    import superglm.solvers.irls_direct._scop as irls_scop

    model, y, weights, offset = _scop_fit_inputs()
    formed: list[_SCOPTrialState] = []
    real_with_merit = irls_scop._IRLSDirectSCOPStep._with_merit

    def recording(self, trial):
        state = real_with_merit(self, trial)
        formed.append(state)
        return state

    checked: list[float] = []
    real_select = irls_scop._select_irls_trial

    def spy(**kwargs):
        candidate, current = kwargs["proposal"], kwargs["committed"]
        expected = sum(
            next(state.merit_error for state in formed if state.irls is held)
            for held in (candidate, current)
        )
        read = kwargs["merit_roundoff"](candidate, current)
        assert read == expected
        assert np.isfinite(read) and read >= 0.0
        checked.append(read)
        with pytest.raises(StopIteration):
            kwargs["merit_roundoff"](candidate, object())
        return real_select(**kwargs)

    monkeypatch.setattr(irls_scop._IRLSDirectSCOPStep, "_with_merit", recording)
    monkeypatch.setattr(irls_scop, "_select_irls_trial", spy)
    result, _ = irls_direct.fit_irls_direct(
        model._dm,
        y,
        weights,
        model._distribution,
        model._link,
        model._groups,
        lambda2={"x": 1.0},
        offset=offset,
        max_iter=3,
        weight_semantics="frequency",
    )
    assert checked
