"""One scored endpoint trial replaces the outward walk of a flat smoothing component."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Literal

import numpy as np
import pandas as pd
import pytest

from superglm.distributional import GaussianLS, Predictor
from superglm.distributional.fit_diagnostics import _ordinary_updates
from superglm.distributional.results.iteration import DistributionalEFSConfig
from superglm.distributional.smoothing.proposals import (
    _endpoint_proposal,
    endpoint_trial_names,
    flat_tail_names,
)
from superglm.features import Spline
from superglm.reml.convergence import flat_tail_certified
from superglm.reml.efs_update import EFSComponentState, wood_fasiolo_update
from superglm.types import LambdaPolicy
from tests.bound_predictor_fixtures import model_from_templates

_Outer = Literal["efs", "efs+finish", "efs+newton", "efs+stationary"]
_X, _Z, _W = "location:x#wiggle", "location:z#wiggle", "location:w#wiggle"
# A stationarity tolerance no fixed-point step meets, so the search ends at its
# practical plateau, in steps of half a log unit.
_PLATEAU = {"practical_reml": True, "reml_tol": 1.0e-13, "max_log_step": 0.5}


def _fit(outer: _Outer, *, terms=("x", "z"), policies=None, **options):
    """A strong smooth in ``x`` beside smooths of ``z`` and ``w``, which act linearly.

    ``policies`` fixes the named covariates' smoothing parameters.
    """
    rng = np.random.default_rng(21)
    frame = pd.DataFrame({name: rng.uniform(-1.0, 1.0, 600) for name in ("x", "z", "w")})
    y = 0.4 + 0.9 * np.sin(2.0 * np.pi * frame["x"]) + 0.3 * frame["z"] + 0.2 * frame["w"]
    y = y + rng.normal(scale=0.35, size=600)
    splines = {name: Spline(kind="cr", k=8) for name in terms}
    for name, policy in (policies or {}).items():
        splines[name] = Spline(kind="cr", k=8, lambda_policy=policy)
    model = model_from_templates(
        family=GaussianLS(),
        predictors=(Predictor("location", splines), Predictor("scale", {})),
    )
    model.fit_reml(frame, y, outer=outer, **options)
    return model, model._require_fitted().smoothing


def _trials(smoothing):
    return [index for index, row in enumerate(smoothing.history) if row.endpoint_components]


def _face_event(row) -> bool:
    return bool(
        row.activated_face_components
        or row.revalidated_face_components
        or row.refused_face_components
    )


@pytest.mark.parametrize("outer", ["efs", "efs+finish"])
def test_a_flat_component_is_taken_to_its_bound_once_and_judged_where_the_others_stop(
    outer: _Outer,
) -> None:
    """A component with no optimum short of infinity reaches its exact face without walking there.

    ``z`` acts linearly, so its wiggliness penalty has no finite optimum, and the
    half-unit steps force a walk of thirty rounds to the bound. One trial takes it
    there; it is held off the face while ``x`` finishes, so every fit until then is
    an ordinary one, and its face is promoted once, at the stop ``x`` earned: the
    practical plateau under ``"efs"``, the Newton finish under ``"efs+finish"``,
    which runs on the face. The fit equals the one with ``z`` fixed at its bound.
    Mutation checks: without the hold the face is promoted the round after the
    trial; without the return to the paused endgame, Fellner-Schall rounds follow
    the promotion; without the earned plateau the ``"efs"`` fit runs on to its
    round limit; without the latch the component is tried again; read as a step,
    the trial enters the diagnostics' count of ordinary updates.
    """
    model, smoothing = _fit(outer, **_PLATEAU)
    bound = smoothing.config.upper(_Z)
    history = smoothing.history

    (trial,) = _trials(smoothing)
    assert history[trial].step_source == "endpoint"
    assert history[trial].endpoint_components == (_Z,)
    assert history[trial].lambdas_after[_Z] == bound
    assert len(history[trial].coefficient_fit_indices) == 1
    promoted = [index for index, row in enumerate(history) if row.activated_face_components]
    assert len(promoted) == 1 and promoted[0] > trial + 3
    between = history[trial + 1 : promoted[0]]
    assert all(row.accepted and row.step_source == "efs" for row in between)
    assert not any(_face_event(row) for row in between)
    assert all(row.accepted_log_steps[_Z] == 0.0 for row in between)
    # The diagnostics read the search's own steps: the trial is not one of them.
    ordinary = _ordinary_updates(smoothing)
    assert all(row is not history[trial] for row in ordinary)
    assert [row for row in ordinary if row.stage == "efs"] == [*history[:trial], *between]
    after = history[promoted[0] + 1 :]
    assert after[-1].revalidated_face_components == (_Z,)
    assert [row.stage for row in after[:-1]] == ["newton"] * (len(after) - 1)
    assert bool(after[:-1]) is (outer == "efs+finish")

    assert smoothing.converged is True
    expected = "practical_plateau" if outer == "efs" else "stationary"
    assert smoothing.convergence_reason == expected
    assert model.exact_face_components_ == (_Z,)
    assert smoothing.unresolved_upper_bound == ()
    reference, fixed = _fit(outer, policies={"z": LambdaPolicy.fixed(bound)}, **_PLATEAU)
    assert smoothing.lambdas[_X] == pytest.approx(fixed.lambdas[_X], rel=1.0e-6)
    # The fitted location is of order one and crosses zero, so the bar is absolute.
    np.testing.assert_allclose(
        smoothing.terminal_fit.theta, fixed.terminal_fit.theta, rtol=0.0, atol=1.0e-6
    )
    assert model.result_.total_effective_df == pytest.approx(
        reference.result_.total_effective_df, rel=1.0e-6
    )


def test_a_search_out_of_rounds_before_judging_its_held_cap_says_so() -> None:
    """A cap the trial reached and no assessment has refused is not reported as unresolved.

    ``z`` is the only smoothing parameter estimated, and the round limit is the
    round of its trial: the search has no round left to assess the face. The fit
    reports that it ran out of rounds and names the cap. Mutation check: read as
    a refused cap, the reason is ``lambda_cap_unresolved``.
    """
    policies = {"x": LambdaPolicy.fixed(0.19)}
    _model, free = _fit("efs", policies=policies)
    (trial,) = _trials(free)
    assert free.converged is True

    model, smoothing = _fit("efs", policies=policies, max_reml_iter=trial + 1)
    assert smoothing.history[-1].step_source == "endpoint"
    assert (smoothing.converged, smoothing.convergence_reason) == (False, "max_iterations")
    assert smoothing.unresolved_upper_bound == (_Z,)
    assert model.exact_face_components_ == ()


def test_a_trial_does_not_delay_the_hand_off(monkeypatch) -> None:
    """The jump to a bound is no step of the search: the hand-off reads the other components.

    The nomination is withheld until the round in which ``x``'s step has become
    small, so under ``"efs+stationary"`` the round of the trial hands over.
    Mutation check: counting the jump as a step, the hand-off waits a round.
    """
    from superglm.distributional.smoothing import loop

    names, calls = loop.endpoint_trial_names, []

    def late(flat_tail, update, estimated_names, tried):
        calls.append(None)
        return names(flat_tail, update, estimated_names, tried) if len(calls) > 3 else ()

    monkeypatch.setattr(loop, "endpoint_trial_names", late)
    _model, smoothing = _fit("efs+stationary")
    assert _trials(smoothing) == [3]
    trial = smoothing.history[3]
    assert trial.step_source == "endpoint" and trial.max_accepted_log_step > 5.0
    assert abs(trial.accepted_log_steps[_X]) < 0.5
    assert smoothing.history[4].stage == "newton"
    assert (smoothing.converged, smoothing.convergence_reason) == (True, "stationary")


def test_a_bound_past_an_interior_optimum_is_refused_and_tried_once(monkeypatch) -> None:
    """A trial the evidence at the bound contradicts is dropped, and is the component's last.

    The nomination is forced onto ``x``, whose optimum is interior, from a start far
    below it and under a bound just above it: the bound scores better than the
    start, and the evidence there asks for a smaller value. The round spends one
    fit on the trial and then takes its ordinary step, and the fit is the unforced
    one. Mutation checks: without the test of the evidence the trial is accepted;
    without the latch every round repeats it.
    """
    from superglm.distributional.smoothing import loop

    options = {"terms": ("x",), "max_lambda": 1.0, "lambdas": {_X: 1.0e-3}, "reml_tol": 1.0e-8}
    _unforced_model, unforced = _fit("efs", **options)
    assert not _trials(unforced)
    assert 0.1 < unforced.lambdas[_X] < 0.5

    def forced(_flat_tail, _update, _estimated_names, tried):
        return () if _X in tried else (_X,)

    monkeypatch.setattr(loop, "endpoint_trial_names", forced)
    _model, smoothing = _fit("efs", **options)

    assert _trials(smoothing) == [0]
    first = smoothing.history[0]
    assert (first.step_source, first.accepted) == ("efs", True)
    assert (
        len(first.coefficient_fit_indices) == len(unforced.history[0].coefficient_fit_indices) + 1
    )
    assert first.lambdas_after[_X] == unforced.history[0].lambdas_after[_X]
    assert smoothing.lambdas[_X] == pytest.approx(unforced.lambdas[_X], rel=1.0e-9)
    assert smoothing.convergence_reason == unforced.convergence_reason


def test_a_bound_the_objective_refuses_costs_one_fit(monkeypatch) -> None:
    """A trial scored worse than the fit it starts from is dropped for the ordinary step.

    The first objective scored with ``z`` at its bound is raised far above the
    current one, so the refusal is forced. Mutation check: without the objective
    test the trial is accepted.
    """
    from superglm.distributional.smoothing import loop

    _model, plain = _fit("efs", max_log_step=0.5, max_reml_iter=8)
    (trial,) = _trials(plain)
    bound = plain.config.upper(_Z)
    objective, raised = loop._trial_laplace_objective, []

    def trial_objective(result, *, lambdas, **kwargs):
        value = objective(result, lambdas=lambdas, **kwargs)
        assert value is not None
        if lambdas[_Z] == bound and not raised:
            raised.append(value)
            return value + 1.0e9
        return value

    monkeypatch.setattr(loop, "_trial_laplace_objective", trial_objective)
    _model, smoothing = _fit("efs", max_log_step=0.5, max_reml_iter=8)

    assert _trials(smoothing) == [trial] and len(raised) == 1
    row = smoothing.history[trial]
    assert (row.step_source, row.accepted) == ("efs", True)
    assert len(row.coefficient_fit_indices) == 2
    assert row.accepted_log_steps[_Z] == pytest.approx(0.5)
    assert smoothing.lambdas[_Z] < bound


@pytest.mark.parametrize("outer", ["efs", "efs+finish"])
def test_held_caps_are_not_assessed_beside_another_components_cap(
    monkeypatch, outer: _Outer
) -> None:
    """A face is promoted with every cap in it, so beside a cap the search does not hold, none is tried.

    ``w`` starts at its bound, under a ceiling below its optimum so that it stays,
    and the evidence is scripted to show it standing there without pressure, as a
    tensor margin can. ``z``'s trial is forced in the first round and held. Where
    ``x`` stops, ``"efs"`` keeps the plateau it earned and names ``z``'s cap
    without an assessment, and ``"efs+finish"`` enters its endgame without one,
    leaving the caps to the exact gradient. Mutation check: without the test for
    the other cap, the assessors run first.
    """
    from superglm.distributional.smoothing import loop

    options = {"terms": ("x", "z", "w"), "max_lambda": 100.0, **_PLATEAU}
    bound = _fit(outer, **{**options, "max_reml_iter": 1})[1].config.upper(_W)
    evidence, endgame, events = loop._fresh_raw_evidence, loop.run_newton_endgame, []

    def unpressed(*args, held=frozenset(), **kwargs):
        built = evidence(*args, held=frozenset(held) | {_W}, **kwargs)
        return replace(built, cap_held=tuple(name for name in built.cap_held if name != _W))

    def assessor(*args, **kwargs):
        events.append("assessment")
        return None, None

    def entered(*args, **kwargs):
        events.append("endgame")
        return endgame(*args, **kwargs)

    monkeypatch.setattr(loop, "_fresh_raw_evidence", unpressed)
    monkeypatch.setattr(
        loop, "endpoint_trial_names", lambda _f, _u, _n, tried: () if _Z in tried else (_Z,)
    )
    monkeypatch.setattr(loop, "_try_joint_exact_face", assessor)
    monkeypatch.setattr(loop, "_try_exact_face", assessor)
    monkeypatch.setattr(loop, "run_newton_endgame", entered)
    model, smoothing = _fit(outer, lambdas={_X: 1.0, _Z: 1.0, _W: bound}, **options)

    assert smoothing.lambdas[_W] == bound == smoothing.lambdas[_Z]
    assert _trials(smoothing) == [0] and smoothing.history[0].step_source == "endpoint"
    assert model.exact_face_components_ == ()
    if outer == "efs+finish":
        assert events[0] == "endgame"
        return
    assert events == []
    assert (smoothing.converged, smoothing.convergence_reason) == (True, "practical_plateau")
    assert smoothing.unresolved_upper_bound == (_Z,)


def _block(curvature: np.ndarray, *, lam: float = 1.0, beta=(0.1, 0.1), name: str = "a"):
    """One identity penalty on a block whose profiled data curvature is ``curvature``."""
    width = len(curvature)
    component = EFSComponentState(
        name=name,
        coefficient_slice=slice(0, width),
        penalty=np.eye(width),
        rank=float(width),
        lambda_value=lam,
        policy=LambdaPolicy.estimate(),
    )
    inverse = np.linalg.inv(curvature + lam * np.eye(width))
    update = wood_fasiolo_update((component,), np.asarray(beta, dtype=float), inverse)
    return component, inverse, update


def test_the_flat_tail_inequality_and_its_domain() -> None:
    """``B < E (1 - E)^2`` with E in (0, 1) and B >= 0, judged only where it is defined.

    Mutation check: read as ``B < E``, the second and sixth cases pass.
    """
    edf = np.array([0.2, 0.2, 1.0, 0.0, -0.1, 0.5, np.inf, np.nan])
    fit = np.array([0.12, 0.13, 0.0, 0.0, 0.0, 0.2, np.inf, 0.0])
    with np.errstate(all="raise"):
        certified = flat_tail_certified(edf, fit)
    assert certified.tolist() == [True] + [False] * 7
    assert flat_tail_certified(0.2, 0.12).tolist() == [True]


def test_a_flat_tail_needs_nonnegative_data_curvature_on_its_block() -> None:
    """The inequality alone passes an indefinite block whose score later turns inward.

    With data curvature ``diag(100, -0.45)`` under an identity penalty the
    inequality holds at lambda = 1, yet the working score is +1.03 at lambda = 100:
    E = sum d/(d + lambda) changes sign past the negative eigenvalue. The spectrum
    test refuses the nomination; a block of the same E with nonnegative curvature
    is nominated. Mutation check: without the spectrum test both are.
    """
    config = DistributionalEFSConfig()
    beta = np.array([0.316, 0.0])
    for diagonal, expected in (([100.0, -0.45], ()), ([0.12, 0.07], ("a",))):
        curvature = np.diag(diagonal)
        component, inverse, update = _block(curvature, beta=beta)
        edf = 2.0 - update.trace_terms["a"]
        assert math.isclose(edf, sum(d / (d + 1.0) for d in np.diag(curvature)))
        assert flat_tail_certified(edf, update.quadratic_forms["a"])[0]
        names = flat_tail_names((component,), (("a", slice(0, 2)),), update, inverse, config)
        assert names == expected
    # The indefinite block at lambda = 100, from the working vector z = (D + I) beta:
    # B = lambda sum z^2 / (d + lambda)^2 now exceeds E, so the score has turned inward.
    spectrum, late = np.array([100.0, -0.45]), 100.0
    quadratic = late * np.sum(((spectrum + 1.0) * beta) ** 2 / (spectrum + late) ** 2)
    assert 0.5 * (quadratic - np.sum(spectrum / (spectrum + late))) == pytest.approx(1.03, abs=0.01)


def test_a_flat_tail_is_alone_on_its_block_and_short_of_its_bound() -> None:
    """A penalty that shares coefficients with another, or already stands at its bound, is never named.

    Mutation checks: without the isolation test the tensor margin is named;
    without the bound test the component at its bound is.
    """
    config = DistributionalEFSConfig()
    component, inverse, update = _block(np.diag([0.2, 0.1]), beta=(0.02, 0.02))
    alone = (("a", slice(0, 2)), ("other", slice(2, 4)))
    assert flat_tail_names((component,), alone, update, inverse, config) == ("a",)
    shared = (("a", slice(0, 2)), ("margin", slice(1, 3)))
    assert flat_tail_names((component,), shared, update, inverse, config) == ()

    bound = config.upper("a")
    capped, inverse, update = _block(np.diag([0.2, 0.1]) * bound, lam=bound, beta=(1e-7, 1e-7))
    assert flat_tail_certified(2.0 - bound * update.trace_terms["a"], 2.0e-14 * bound)[0]
    assert flat_tail_names((capped,), alone, update, inverse, config) == ()


def test_the_trial_waits_for_the_other_components_and_lands_on_each_ones_own_bound() -> None:
    """No trial while another component still proposes more than a log unit; none twice.

    The proposal puts each tried component on its own bound, which a relative box
    scales by the component's unit. Mutation checks: without the gate the trial is
    made beside a 1.5-unit step; with the global ceiling the scaled component
    lands a factor of its unit away.
    """
    component, inverse, update = _block(np.diag([0.2, 0.1]), beta=(0.02, 0.02))
    other = replace(component, name="b", coefficient_slice=slice(2, 4))
    update = wood_fasiolo_update(
        (component, other), np.array([0.02, 0.02, 0.5, 0.5]), np.kron(np.eye(2), inverse)
    )
    for step, expected in ((0.9, ("a",)), (-0.9, ("a",)), (1.5, ()), (-1.5, ())):
        moved = replace(update, log_steps={"a": 2.0, "b": step})
        assert endpoint_trial_names(("a",), moved, ("a", "b"), set()) == expected
    settled = replace(update, log_steps={"a": 2.0, "b": 0.1})
    assert endpoint_trial_names(("a",), settled, ("a", "b"), {"a"}) == ()
    assert endpoint_trial_names(("a", "b"), settled, ("a", "b"), {"b"}) == ("a",)

    config = DistributionalEFSConfig(
        lambda_units=(("a", 40.0),), relative_box=True, maximum_lambda=3.0e5
    )
    assert config.upper("a") == 40.0 * config.upper("b")
    current, steps = {"a": 7.0, "b": 2.0}, {"a": 1.0, "b": 0.25}
    lambdas, log_steps = _endpoint_proposal(current, steps, ("a", "b"), ("a",), config)
    assert lambdas["a"] == config.upper("a")
    assert lambdas["b"] == pytest.approx(2.0 * math.exp(0.25))
    assert log_steps["a"] == math.log(config.upper("a")) - math.log(7.0)
