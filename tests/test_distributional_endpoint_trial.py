"""One scored endpoint trial replaces the outward walk of a flat smoothing component."""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace
from typing import Literal

import numpy as np
import pandas as pd
import pytest

from superglm import ConvergenceWarning
from superglm.distributional import GaussianLS, Predictor
from superglm.distributional.fit_diagnostics import _ordinary_updates
from superglm.distributional.results.iteration import DistributionalEFSConfig
from superglm.distributional.smoothing.proposals import (
    _endpoint_proposal,
    endpoint_trial_names,
    flat_tail_fall,
    flat_tail_names,
    resolved_flat_tails,
)
from superglm.features import CubicRegressionSpline, Spline
from superglm.reml.convergence import flat_tail_certified
from superglm.reml.efs_update import EFSComponentState, wood_fasiolo_update
from superglm.types import LambdaPolicy
from tests.bound_predictor_fixtures import model_from_templates
from tests.test_distributional_newton_practical_handoff import (
    _refuse_fellner_schall_steps_after,
    _spy_on_the_endgame,
)

_Outer = Literal["efs", "efs+finish", "efs+newton", "efs+stationary"]
_X, _Z, _W = "location:x#wiggle", "location:z#wiggle", "location:w#wiggle"
# A stationarity tolerance no fixed-point step meets, so the search ends at its
# practical plateau, in steps of half a log unit.
_PLATEAU = {"practical_reml": True, "reml_tol": 1.0e-13, "max_log_step": 0.5}


class _DataCoordinateSpline(CubicRegressionSpline):
    """A subclass states no penalty null coordinates, so its block keeps the data's coordinates.

    There the stored penalty is singular only to rounding, which a smoothing
    parameter on its bound multiplies into the objective. Held in its penalty's
    null coordinates, superglm's own spline has an exactly singular penalty and
    an objective as exact on the bound as on the face.
    """


def _fit(outer: _Outer, *, terms=("x", "z"), policies=None, data_coordinates=False, **options):
    """A strong smooth in ``x`` beside smooths of ``z`` and ``w``, which act linearly.

    ``policies`` fixes the named covariates' smoothing parameters, and
    ``data_coordinates`` fits every smooth as a ``_DataCoordinateSpline``.
    """

    def spline(**spec):
        if data_coordinates:
            return _DataCoordinateSpline(n_knots=6, **spec)  # the knots of ``k=8``
        return Spline(kind="cr", k=8, **spec)

    rng = np.random.default_rng(21)
    frame = pd.DataFrame({name: rng.uniform(-1.0, 1.0, 600) for name in ("x", "z", "w")})
    y = 0.4 + 0.9 * np.sin(2.0 * np.pi * frame["x"]) + 0.3 * frame["z"] + 0.2 * frame["w"]
    y = y + rng.normal(scale=0.35, size=600)
    splines = {name: spline() for name in terms}
    for name, policy in (policies or {}).items():
        splines[name] = spline(lambda_policy=policy)
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


@pytest.mark.parametrize("outer", ["efs", "efs+finish"])
def test_no_trial_is_made_for_a_fall_the_tolerance_does_not_resolve(outer: _Outer) -> None:
    """A bound is tried for a gain the search resolves, and where no plateau can end the search.

    ``z``'s working problem can fall 0.15 on the way to its bound, with V near 288.
    At ``reml_tol=1e-2`` the score bar is 2.9, so the search makes no trial, puts
    nothing on a face, and ends within that bar of the fit that tried. At
    ``reml_tol=1e-4`` the bar is 0.029 and the trial is made. A search without
    the practical plateau stops on the step alone, and ``"efs+newton"`` hands
    over on it; a flat tail never shrinks its step, so both try at either
    tolerance. Mutation checks: without the test of the fall the first fit tries
    too; applied to every search, the last two do not.
    """
    tried_model, tried = _fit(outer, reml_tol=1.0e-4)
    assert len(_trials(tried)) == 1
    assert tried_model.exact_face_components_ == (_Z,)

    model, smoothing = _fit(outer, reml_tol=1.0e-2)
    assert _trials(smoothing) == []
    assert smoothing.converged is True
    assert model.exact_face_components_ == ()
    assert smoothing.lambdas[_Z] < smoothing.config.upper(_Z)
    assert not any(_face_event(row) for row in smoothing.history)
    forgone = smoothing.objective - tried.objective
    assert 0.0 <= forgone <= smoothing.config.score_bar(smoothing.objective)

    _, strict = _fit(outer, reml_tol=1.0e-2, practical_reml=False)
    assert len(_trials(strict)) == 1
    _, handed = _fit("efs+newton", reml_tol=1.0e-2)
    assert len(_trials(handed)) == 1


@pytest.mark.parametrize("outer", ["efs", "efs+finish"])
def test_a_flat_tail_its_own_step_puts_on_its_bound_is_tried_there(outer: _Outer) -> None:
    """A walk that reaches the bound ends as a trial: the cap is held and assessed once.

    ``z`` starts a factor of four short of its bound, where the fall left to it
    is a thousandth of the score bar, so its first step is an ordinary one. Its
    second would put it on the bound, and is made as a trial, which holds the
    cap off the face until ``x`` stops. Mutation checks: without the test of
    the fall the trial is the first round; without the trial of an arriving
    tail none is made, and the face is promoted the round after the landing.
    """
    bound = _fit(outer, max_reml_iter=1)[1].config.upper(_Z)
    model, smoothing = _fit(outer, lambdas={_X: 0.2, _Z: bound / 4.0})
    history = smoothing.history

    (trial,) = _trials(smoothing)
    walked = history[:trial]
    assert walked and all(row.accepted and row.step_source == "efs" for row in walked)
    assert all(row.lambdas_after[_Z] < bound for row in walked)
    row = history[trial]
    assert (row.step_source, row.endpoint_components, row.accepted) == ("endpoint", (_Z,), True)
    assert row.lambdas_after[_Z] == bound
    # The bound was within the walk's stride.
    assert row.accepted_log_steps[_Z] < walked[-1].accepted_log_steps[_Z]
    (promoted,) = (i for i, item in enumerate(history) if item.activated_face_components)
    between = history[trial + 1 : promoted]
    assert between and not any(_face_event(item) for item in between)
    assert all(item.accepted_log_steps[_Z] == 0.0 for item in between)

    assert smoothing.converged is True
    assert model.exact_face_components_ == (_Z,)
    assert history[-1].revalidated_face_components == (_Z,)
    assert smoothing.unresolved_upper_bound == ()


def test_a_step_rejected_at_every_length_beside_a_trial_s_cap_goes_to_the_finish(
    monkeypatch,
) -> None:
    """Beside a cap an endpoint trial holds, a rejected Fellner-Schall round is no stop.

    The trial takes at once the fall ``z``'s walk to its bound would have spread
    over the rounds to come, so ``x``'s later steps stand on their own. The
    objective is made to refuse every step after the trial's round, so the next is
    rejected at all nine lengths. The held cap is judged and promoted, and the
    finish runs on its face to the optimum the unforced search reaches. Mutation
    check: with the rejected round's verdict returned, as it is beside no trial's
    cap, the fit reports ``objective_rejected`` and does not converge.
    """
    _, reference = _fit("efs+finish")
    (trial,) = _trials(reference)
    calls = _spy_on_the_endgame(monkeypatch)
    _refuse_fellner_schall_steps_after(monkeypatch, trial + 1)
    _, smoothing = _fit("efs+finish")

    history = smoothing.history
    assert history[trial].step_source == "endpoint"
    rejected = history[trial + 1]
    assert (rejected.stage, rejected.accepted) == ("efs", False)
    assert rejected.backtracks == smoothing.config.max_backtracks
    assert rejected.lambdas_before[_Z] == smoothing.config.upper(_Z)
    assert history[trial + 2].activated_face_components == (_Z,)
    finish = history[trial + 3 : -1]
    assert finish and all(row.stage == "newton" for row in finish)
    assert history[-1].revalidated_face_components == (_Z,)
    (call,) = calls
    assert call["state"].face is not None
    assert (smoothing.converged, smoothing.convergence_reason) == (True, "stationary")
    assert smoothing.objective == pytest.approx(reference.objective, abs=1.0e-6)
    np.testing.assert_allclose(
        smoothing.terminal_fit.theta, reference.terminal_fit.theta, rtol=0.0, atol=1.0e-4
    )


@pytest.mark.parametrize(
    "kind",
    ["gradient_unresolved", "max_iterations", "objective_rejected", "derivative_unavailable"],
)
def test_a_finish_that_stalls_after_a_rejected_round_leaves_the_round_s_verdict(
    monkeypatch, kind
) -> None:
    """A rejected round earned no plateau, so a finish that stalls from it reports none.

    The round is rejected beside a trial's cap, which is promoted to its face
    before the finish runs. The fit is the one that round left, on that face: not
    converged, at the round's smoothing parameters, with the finish's steps
    dropped. Mutation check: read as a plateau's finish, the stall returns the
    search to Fellner-Schall rounds on the face, and a second rejected round
    follows the promotion.
    """
    _, reference = _fit("efs+finish")
    (trial,) = _trials(reference)
    calls = _spy_on_the_endgame(monkeypatch, lambda outcome: replace(outcome, kind=kind))
    _refuse_fellner_schall_steps_after(monkeypatch, trial + 1)
    with pytest.warns(ConvergenceWarning, match="objective_rejected"):
        _, smoothing = _fit("efs+finish")

    assert len(calls) == 1
    assert (smoothing.converged, smoothing.convergence_reason) == (False, "objective_rejected")
    rejected, promoted = smoothing.history[trial + 1 :]
    assert (rejected.stage, rejected.accepted) == ("efs", False)
    assert promoted.activated_face_components == (_Z,)
    assert smoothing.lambdas == rejected.lambdas_before


def test_a_stalled_finish_keeps_the_plateau_earned_after_a_rejection(monkeypatch) -> None:
    """A new accepted round supersedes a rejection that survived a face release.

    Feed the search an accepted trial and a stalled finish, so the verdict
    test is independent of the numerical path of either Newton finish.
    """
    from superglm.distributional.smoothing import loop

    search = loop._SmoothingSearch.__new__(loop._SmoothingSearch)
    search.outer_config = DistributionalEFSConfig(outer="efs+finish")
    search.mode = loop._OUTER_MODES["efs+finish"]
    search.estimated_names = (_X,)
    search.layout = None
    search.history = []
    search.initial_lambdas = {_X: 1.0}
    search.initial_objective = 20.0
    search.fall = math.inf
    search.cap_held = frozenset()
    search.endpoint_caps = set()
    search.beyond_cap = {}
    search.unearned = True  # rejection from before the face release
    search.endgame_stationary = False
    search.newton_budget = 1

    fit = SimpleNamespace(terminal_curvature=None)
    evidence = loop._FreshRawEvidence((), (_X,), None, 1.0e-6, (), ())
    search.coefficient_fits = [fit, fit]
    search.state = loop.EndgameState(search.initial_lambdas, fit, 20.0, None, evidence, 0)
    monkeypatch.setattr(search, "_evidence", lambda _memo: evidence)
    search.penalty_memo = None
    trial = loop._Trial(fit, 1, {_X: 1.000001}, {_X: 1.0e-6}, 19.999999)
    round_ = loop._Round(1, None, 1.0e-6, fit_indices=[1])
    round_.accept_as_proposed(trial, search.estimated_names)
    search._adopt_round(round_, trial)

    def exhausted(state, budget):
        return loop.NewtonEndgameOutcome("max_iterations", state, None, None, (), budget, 0)

    monkeypatch.setattr(search, "_endgame_at", exhausted)
    result = search._run_endgame()

    assert result is not None
    assert (result.converged, result.convergence_reason) == (True, "practical_plateau")
    assert result.history[-1].accepted
    assert result.objective == trial.objective
    assert result.lambdas == trial.lambdas


def test_a_plateau_earned_in_the_last_round_keeps_its_terminal_checks() -> None:
    """Judging the caps of a plateau already earned is no search step: the round limit does not refuse it.

    The round limit is the round in which ``x`` earns its plateau with ``z`` held.
    The face is still promoted and revalidated, in two rows past the limit, and the
    fit reports the plateau; with one round fewer the search is out of rounds
    before the plateau. Mutation check: with the limit refusing the assessment,
    the fit that earned its plateau reports ``max_iterations``.
    """
    _model, free = _fit("efs", **_PLATEAU)
    (promoted,) = (i for i, row in enumerate(free.history) if row.activated_face_components)

    model, smoothing = _fit("efs", max_reml_iter=promoted, **_PLATEAU)
    assert len(smoothing.history) == promoted + 2
    assert smoothing.history[-2].activated_face_components == (_Z,)
    assert smoothing.history[-1].revalidated_face_components == (_Z,)
    assert (smoothing.converged, smoothing.convergence_reason) == (True, "practical_plateau")
    assert model.exact_face_components_ == (_Z,)

    _short_model, short = _fit("efs", max_reml_iter=promoted - 1, **_PLATEAU)
    assert (short.converged, short.convergence_reason) == (False, "max_iterations")


def test_a_face_promoted_at_the_stop_is_revalidated_on_the_derivative_taken_there(
    monkeypatch,
) -> None:
    """A promotion and the revalidation that follows it assess one fit: its derivative is taken once.

    Under ``"efs"`` the plateau ``x`` earned promotes ``z``'s face and goes straight
    to the strict stop, which revalidates the face at the promotion's own fit. An
    endpoint derivative costs three passes over the rows (a design product for each
    predictor and one curvature assembly), and the revalidation publishes the one
    the promotion took. Mutation check: without the search's memo the same fit and
    component are evaluated twice.
    """
    from superglm.distributional.smoothing import faces

    evaluate, taken = faces.evaluate_endpoint_laplace_derivative, []

    def recorded(*args, **kwargs):
        taken.append(kwargs["component_name"])
        return evaluate(*args, **kwargs)

    monkeypatch.setattr(faces, "evaluate_endpoint_laplace_derivative", recorded)
    _model, smoothing = _fit("efs", **_PLATEAU)

    promotion, revalidation = smoothing.history[-2:]
    assert promotion.activated_face_components == (_Z,)
    assert revalidation.revalidated_face_components == (_Z,)
    assert revalidation.endpoint_direction_evidence.decision == "endpoint"
    assert (
        revalidation.endpoint_direction_evidence.analytic_derivative
        == promotion.endpoint_direction_evidence.analytic_derivative
    )
    assert taken == [_Z]


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


def _named_in_the_endgame(monkeypatch, *, forced: tuple[str, ...] = (), skip: int = 0) -> list[str]:
    """Withhold every nomination until the Newton steps run; ``forced`` names them then.

    The first ``skip`` requests of the Newton steps are withheld as well. Returns
    the kinds of the endgame runs' outcomes, in order.
    """
    from superglm.distributional.smoothing import loop

    names, endgame, kinds, requests = loop.endpoint_trial_names, loop.run_newton_endgame, [], []

    def late(flat_tail, log_steps, estimated_names, tried):
        if not kinds:
            return ()
        requests.append(None)
        if len(requests) <= skip:
            return ()
        if forced:
            return tuple(name for name in forced if name not in tried)
        return names(flat_tail, log_steps, estimated_names, tried)

    def entered(*args, **kwargs):
        kinds.append("running")
        outcome = endgame(*args, **kwargs)
        kinds[-1] = outcome.kind
        return outcome

    monkeypatch.setattr(loop, "endpoint_trial_names", late)
    monkeypatch.setattr(loop, "run_newton_endgame", entered)
    return kinds


def test_the_remaining_fall_of_a_linearly_converging_objective() -> None:
    """Falls d' then d at rate a = d / d' leave d a / (1 - a); no rate, no limit.

    Mutation check: read as the last fall itself, the first case is 0.25.
    """
    from superglm.distributional.smoothing.objective import remaining_fall

    assert remaining_fall(0.25, 0.5, 1.0e-9) == pytest.approx(0.25)
    assert remaining_fall(0.9, 1.0, 1.0e-9) == pytest.approx(8.1)
    assert remaining_fall(1.0e-3, 0.5, 1.0e-9) == pytest.approx(1.0e-3 * 2.0e-3 / 0.998)
    assert remaining_fall(1.0, 1.0, 1.0e-9) == math.inf
    assert remaining_fall(1.0, 0.5, 1.0e-9) == math.inf
    assert remaining_fall(1.0, math.inf, 1.0e-9) == math.inf
    # A fall within the objective's resolution is none, and a rise is none.
    assert remaining_fall(1.0e-10, 1.0e-12, 1.0e-9) == 0.0
    assert remaining_fall(-1.0, 0.5, 1.0e-9) == 0.0
    assert remaining_fall(math.inf, math.inf, 1.0e-9) == math.inf


def test_a_search_holding_a_trials_cap_goes_on_through_a_plateau_still_falling(
    monkeypatch,
) -> None:
    """A practical plateau reads one round's change, which a slow steady drift keeps small.

    Holding ``z``'s cap, the search asks how far the objective still has to fall
    before its plateau may stand. The estimate is scripted to say "further than
    the tolerance" at the first plateau: the search takes one more round with
    ``z`` held, and then stops at its plateau and promotes the face, as the
    unscripted fit does at once. A fit that holds no trial's cap never asks.
    Mutation checks: without the question the scripted fit stops where the
    unscripted one does; asked at every plateau, the fit with ``z`` fixed asks.
    """
    from superglm.distributional.smoothing import loop

    _model, plain = _fit("efs", **_PLATEAU)
    bound = plain.config.upper(_Z)
    estimate, asked = getattr(loop, "remaining_fall", None), []

    def scripted(fall, previous_fall, resolution):
        asked.append(estimate(fall, previous_fall, resolution))
        return math.inf if len(asked) == 1 else asked[-1]

    monkeypatch.setattr(loop, "remaining_fall", scripted, raising=False)
    model, smoothing = _fit("efs", **_PLATEAU)

    assert len(asked) == 2
    # Unscripted, each of those plateaus would have stood.
    ceiling = smoothing.config.plateau_tolerance * (1.0 + abs(smoothing.objective))
    assert max(asked) <= ceiling
    (promoted,) = (i for i, row in enumerate(smoothing.history) if row.activated_face_components)
    (expected,) = (i for i, row in enumerate(plain.history) if row.activated_face_components)
    assert promoted == expected + 1
    extra = smoothing.history[expected:promoted]
    assert all(row.accepted and row.step_source == "efs" and row.stage == "efs" for row in extra)
    assert all(row.lambdas_after[_Z] == bound and not _face_event(row) for row in extra)
    assert (smoothing.converged, smoothing.convergence_reason) == (True, "practical_plateau")
    assert model.exact_face_components_ == (_Z,)

    asked.clear()
    _fixed_model, fixed = _fit("efs", policies={"z": LambdaPolicy.fixed(bound)}, **_PLATEAU)
    assert (fixed.converged, fixed.convergence_reason) == (True, "practical_plateau")
    assert asked == []


def test_a_plateau_stands_on_an_exact_face_under_the_question_a_held_cap_asks(monkeypatch) -> None:
    """A term on an exact face and a cap held for one are the same active bound to the plateau.

    ``z`` starts at its bound, so its face is promoted before the first round and
    every fit is on the face. Where ``x`` earns its practical plateau the search
    stops and the face is revalidated; scripted to say the objective has further
    to fall at the first plateau, it takes one more round first. Which of the two
    bounds a search has can differ between BLAS kernels, since a held cap is
    promoted when its evidence wavers; the stop must not. Mutation checks: with a
    face barring the plateau the fit runs on to its fixed point; with a face not
    asking, the scripted fit stops where the unscripted one does.
    """
    from superglm.distributional.smoothing import loop

    bound = _fit("efs", max_reml_iter=1, **_PLATEAU)[1].config.upper(_Z)
    options = {"lambdas": {_Z: bound}, **_PLATEAU}
    model, plain = _fit("efs", **options)
    assert plain.history[0].activated_face_components == (_Z,)
    assert plain.history[-1].revalidated_face_components == (_Z,)
    assert all(row.stage == "efs" and row.accepted for row in plain.history[1:-1])
    assert (plain.converged, plain.convergence_reason) == (True, "practical_plateau")
    assert model.exact_face_components_ == (_Z,) and not _trials(plain)

    estimate, asked = getattr(loop, "remaining_fall", None), []

    def scripted(fall, previous_fall, resolution):
        asked.append(estimate(fall, previous_fall, resolution))
        return math.inf if len(asked) == 1 else asked[-1]

    monkeypatch.setattr(loop, "remaining_fall", scripted, raising=False)
    _model, smoothing = _fit("efs", **options)
    assert len(asked) == 2 and len(smoothing.history) == len(plain.history) + 1
    assert (smoothing.converged, smoothing.convergence_reason) == (True, "practical_plateau")


@pytest.mark.parametrize("outer", ["efs+newton", "efs+stationary"])
@pytest.mark.parametrize("where", ["line search", "converged point"])
def test_the_newton_steps_land_a_component_first_named_there(
    monkeypatch, outer: _Outer, where: str
) -> None:
    """A component named during the Newton steps is put on its bound by them, once.

    Newton in log lambda moves a component with no finite optimum about one unit an
    iteration. ``z``'s nomination is withheld until the Newton steps run. Named at
    their first iteration, it is the line search's first candidate, ``x``'s own
    step beside it; withheld one request longer, ``x`` has converged, and the
    trial is made before the converged point is returned, ``x`` standing still.
    Either way one fit puts ``z`` on its bound, its face is promoted, the gradient
    is retaken on the face with no Fellner-Schall round between, and the fit is
    the one with ``z`` fixed at its bound. Mutation checks: without the trial in
    the line search, or without the trial before a converged return, the fit
    named there walks ``z`` or leaves it short of its bound.
    """
    kinds = _named_in_the_endgame(monkeypatch, skip=int(where == "converged point"))
    model, smoothing = _fit(outer)
    bound = smoothing.config.upper(_Z)
    history = smoothing.history

    (trial,) = _trials(smoothing)
    row = history[trial]
    assert (row.stage, row.step_source, row.accepted) == ("newton", "endpoint", True)
    assert row.endpoint_components == (_Z,)
    assert row.lambdas_after[_Z] == bound
    assert len(row.coefficient_fit_indices) == 1
    if where == "line search":
        assert history[trial - 1].stage == "efs" and row.accepted_log_steps[_X] != 0.0
    else:
        assert history[trial - 1].stage == "newton" and row.accepted_log_steps[_X] == 0.0
    assert all(later.lambdas_after[_Z] == bound for later in history[trial:])

    promoted = [index for index, item in enumerate(history) if item.activated_face_components]
    assert len(promoted) == 1 and promoted[0] > trial
    if where == "converged point":
        # The landing is no step of the iteration: the verdict follows it at once.
        assert promoted[0] == trial + 1
    elif outer == "efs+stationary":
        # A jump to a bound is no secant pair: the next step forms its own Hessian.
        assert history[trial + 1].step_source == "newton"
    assert history[-1].revalidated_face_components == (_Z,)
    # After the face is promoted the endgame goes on from it: no Fellner-Schall round.
    assert all(item.stage == "newton" for item in history[promoted[0] + 1 : -1])
    assert kinds[-1] == "stationary"
    assert (smoothing.converged, smoothing.convergence_reason) == (True, "stationary")
    assert model.exact_face_components_ == (_Z,)
    assert smoothing.unresolved_upper_bound == ()

    monkeypatch.undo()
    reference, fixed = _fit(outer, policies={"z": LambdaPolicy.fixed(bound)})
    assert smoothing.lambdas[_X] == pytest.approx(fixed.lambdas[_X], rel=1.0e-5)
    np.testing.assert_allclose(
        smoothing.terminal_fit.theta, fixed.terminal_fit.theta, rtol=0.0, atol=1.0e-6
    )
    assert model.result_.total_effective_df == pytest.approx(
        reference.result_.total_effective_df, rel=1.0e-6
    )


def test_a_stationary_endgame_hands_a_trials_cap_to_the_face_assessor(monkeypatch) -> None:
    """The gradient in log lambda is inside any bar at a cap, whatever lies beyond it.

    On a tail ``V + c / lambda`` that gradient is ``-c / lambda`` while the
    derivative in ``tau = 1 / lambda`` stays ``c``, so a stationary verdict with
    ``z`` on its cap does not say whether the optimum is infinite. The verdict
    hands ``z`` to the assessor: its face is promoted, the endgame is run again on
    the face and is stationary there, and the face is revalidated. Mutation check:
    with the nomination voided, the fit ends stationary with ``z`` at its cap and
    no face.
    """
    from superglm.distributional.smoothing import loop

    endgame, runs = loop.run_newton_endgame, []

    def entered(*args, state, **kwargs):
        outcome = endgame(*args, state=state, **kwargs)
        face = () if state.face is None else state.face.component_names
        runs.append((outcome.kind, face, outcome.state.lambdas[_Z]))
        return outcome

    monkeypatch.setattr(loop, "run_newton_endgame", entered)
    model, smoothing = _fit("efs+newton")
    bound = smoothing.config.upper(_Z)

    assert runs == [("stationary", (), bound), ("stationary", (_Z,), bound)]
    events = [index for index, row in enumerate(smoothing.history) if _face_event(row)]
    assert len(events) == 2 and events[1] == events[0] + 1 == len(smoothing.history) - 1
    assert smoothing.history[events[0]].activated_face_components == (_Z,)
    assert smoothing.history[events[1]].revalidated_face_components == (_Z,)
    assert (smoothing.converged, smoothing.convergence_reason) == (True, "stationary")
    assert model.exact_face_components_ == (_Z,)


def test_handed_caps_that_were_all_there_was_to_estimate_keep_the_stationary_reason() -> None:
    """A stationary verdict stands when its handed caps leave nothing to estimate.

    ``x``'s smoothing parameter is fixed, so ``z``'s is the only one estimated. The
    endgame is stationary with ``z`` on its cap, the face is promoted, and there is
    no gradient to take again. Mutation check: without the standing verdict the
    reason is the Fellner-Schall fixed point's, ``lambda_change``.
    """
    model, smoothing = _fit("efs+newton", policies={"x": LambdaPolicy.fixed(0.19)})

    assert len(_trials(smoothing)) == 1
    assert model.exact_face_components_ == (_Z,)
    assert (smoothing.converged, smoothing.convergence_reason) == (True, "stationary")
    assert smoothing.terminal_projected_gradient_norm == 0.0


def test_a_refused_hand_over_leaves_the_stationary_verdict_standing(monkeypatch) -> None:
    """An assessment that cannot support the face records its refusal and changes nothing else.

    The endpoint derivative is made unavailable, so the assessor refuses ``z``'s
    face. The fit is the one the endgame called stationary, with ``z`` at its cap,
    converged, and the refusal is the history's last row. Mutation check: treated
    as a refused cap under pressure, the fit reports ``lambda_cap_unresolved``.
    """
    from superglm.distributional.smoothing import faces, loop
    from superglm.distributional.smoothing.endpoint_laml import EndpointLaplaceError

    endgame, runs = loop.run_newton_endgame, []

    def entered(*args, **kwargs):
        outcome = endgame(*args, **kwargs)
        runs.append(outcome)
        return outcome

    def unavailable(*args, **kwargs):
        raise EndpointLaplaceError("scripted: no endpoint derivative")

    monkeypatch.setattr(loop, "run_newton_endgame", entered)
    monkeypatch.setattr(faces, "evaluate_endpoint_laplace_derivative", unavailable)
    model, smoothing = _fit("efs+newton")
    bound = smoothing.config.upper(_Z)

    (verdict,) = runs
    assert verdict.kind == "stationary"
    refusal = smoothing.history[-1]
    assert refusal.refused_face_components == (_Z,)
    assert refusal.endpoint_assessment_failure_reason == "analytic_unavailable"
    assert [index for index, row in enumerate(smoothing.history) if _face_event(row)] == [
        len(smoothing.history) - 1
    ]
    assert (smoothing.converged, smoothing.convergence_reason) == (True, "stationary")
    assert smoothing.lambdas == dict(verdict.state.lambdas) and smoothing.lambdas[_Z] == bound
    assert smoothing.objective == verdict.state.objective
    assert model.exact_face_components_ == ()
    assert smoothing.unresolved_upper_bound == ()
    assert model.smoothing_certified_ is False


def test_a_landing_past_an_interior_optimum_is_taken_back_by_the_exact_gradient(
    monkeypatch,
) -> None:
    """The Newton steps keep a trial that scores no worse, and their gradient judges the bound.

    The nomination is forced onto ``x``, whose optimum is interior, in the first
    Newton iteration, from a start far below the optimum and under a bound just
    above it, so the bound scores better than the start. The trial is kept, the
    next gradient asks ``x`` inward, and the fit ends at the unforced optimum with
    no face. Mutation check: asked the Fellner-Schall evidence at the bound as
    well, the trial is refused and the row is an ordinary step.
    """
    options = {
        "terms": ("x",),
        "max_lambda": 1.0,
        "lambdas": {_X: 1.0e-3},
        "max_log_step": 0.5,
    }
    _unforced_model, unforced = _fit("efs+newton", **options)
    assert not _trials(unforced)

    _named_in_the_endgame(monkeypatch, forced=(_X,))
    model, smoothing = _fit("efs+newton", **options)

    (trial,) = _trials(smoothing)
    row = smoothing.history[trial]
    assert (row.stage, row.step_source, row.accepted) == ("newton", "endpoint", True)
    assert row.lambdas_after[_X] == 1.0 and len(row.coefficient_fit_indices) == 1
    assert smoothing.history[trial + 1].lambdas_after[_X] < 1.0
    assert smoothing.lambdas[_X] == pytest.approx(unforced.lambdas[_X], rel=1.0e-4)
    assert (smoothing.converged, smoothing.convergence_reason) == (True, "stationary")
    assert model.exact_face_components_ == ()


@pytest.mark.parametrize("where", ["line search", "converged point"])
def test_a_bound_the_objective_refuses_costs_the_newton_steps_one_fit(
    monkeypatch, where: str
) -> None:
    """A trial of the Newton steps scored worse than the fit it starts from is dropped.

    ``z``'s nomination is withheld until the Newton steps run (one request longer
    for the converged point), and the first objective scored with ``z`` at its
    bound is raised far above the current one. In the line search the iteration
    then takes its own step; at the converged point the refusal is a row of its
    own, and the stationary verdict stands with ``z`` short of its bound.
    Mutation check: without the objective test the trial is kept.
    """
    from superglm.distributional.smoothing import newton

    skip = int(where == "converged point")
    kinds = _named_in_the_endgame(monkeypatch, skip=skip)
    _model, plain = _fit("efs+newton")
    (expected,) = _trials(plain)
    bound = plain.config.upper(_Z)
    objective, raised = newton._laplace_objective, []

    def scored(result, *, lambdas, **kwargs):
        value = objective(result, lambdas=lambdas, **kwargs)
        if lambdas[_Z] == bound and not raised:
            raised.append(value)
            return value + 1.0e9
        return value

    monkeypatch.undo()
    kinds = _named_in_the_endgame(monkeypatch, skip=skip)
    monkeypatch.setattr(newton, "_laplace_objective", scored)
    model, smoothing = _fit("efs+newton")

    assert _trials(smoothing) == [expected] and len(raised) == 1
    row = smoothing.history[expected]
    assert row.stage == "newton" and row.lambdas_after[_Z] < bound
    if where == "line search":
        assert row.accepted and row.step_source != "endpoint"
        assert len(row.coefficient_fit_indices) == 2
        return
    assert (row.accepted, row.step_source) == (False, "endpoint")
    assert len(row.coefficient_fit_indices) == 1 and row is smoothing.history[-1]
    assert kinds == ["stationary"]
    assert (smoothing.converged, smoothing.convergence_reason) == (True, "stationary")
    assert smoothing.lambdas[_Z] == row.lambdas_before[_Z] < bound
    assert model.exact_face_components_ == ()


@pytest.mark.pins_behaviour
def test_a_stationary_endgame_hands_nothing_over_beside_another_components_cap(monkeypatch) -> None:
    """A face is promoted with every cap in it, so beside a cap no trial reached, none is assessed.

    Pins what a stationary verdict did before it handed any cap over: beside
    another component's cap it still assesses nothing.

    ``w`` starts at its bound, under a ceiling below its optimum so that it stays,
    and the Fellner-Schall evidence is scripted to show it standing there without
    pressure, as a tensor margin can; ``z``'s trial is forced in the first round.
    The endgame's verdict is scripted stationary with both at their bounds, and it
    stands without an assessment. Mutation check: handing ``z`` over whatever
    stands beside it, the assessors run.
    """
    from superglm.distributional.smoothing import loop

    options = {"terms": ("x", "z", "w"), "max_lambda": 100.0}
    bound = _fit("efs+newton", **{**options, "max_reml_iter": 1})[1].config.upper(_W)
    evidence, endgame, events = loop._fresh_raw_evidence, loop.run_newton_endgame, []

    def unpressed(*args, held=frozenset(), **kwargs):
        built = evidence(*args, held=frozenset(held) | {_W}, **kwargs)
        return replace(built, cap_held=tuple(name for name in built.cap_held if name != _W))

    def assessor(*args, **kwargs):
        events.append("assessment")
        return None, None

    def stationary(*args, **kwargs):
        outcome = endgame(*args, **kwargs)
        events.append(outcome.kind)
        return replace(outcome, kind="stationary", cap_pressure=())

    monkeypatch.setattr(loop, "_fresh_raw_evidence", unpressed)
    monkeypatch.setattr(
        loop, "endpoint_trial_names", lambda _f, _s, _n, tried: () if _Z in tried else (_Z,)
    )
    monkeypatch.setattr(loop, "_try_joint_exact_face", assessor)
    monkeypatch.setattr(loop, "_try_exact_face", assessor)
    monkeypatch.setattr(loop, "run_newton_endgame", stationary)
    model, smoothing = _fit("efs+newton", lambdas={_X: 1.0, _Z: 1.0, _W: bound}, **options)

    assert _trials(smoothing) == [0] and smoothing.history[0].step_source == "endpoint"
    assert smoothing.lambdas[_W] == bound == smoothing.lambdas[_Z]
    assert len(events) == 1 and events[0] != "assessment"
    assert (smoothing.converged, smoothing.convergence_reason) == (True, "stationary")
    assert model.exact_face_components_ == ()


def _read_finite_objectives_shifted(monkeypatch, shift: float) -> None:
    """Shift the objective of every finite state a face is compared with."""
    from superglm.distributional.smoothing import faces, loop

    objective, promote = faces._laplace_objective, loop._try_exact_face

    def finite(result, *, face, **kwargs):
        return objective(result, face=face, **kwargs) + (shift if face is None else 0.0)

    def promoted(*args, current_objective, **kwargs):
        return promote(*args, current_objective=current_objective + shift, **kwargs)

    monkeypatch.setattr(faces, "_laplace_objective", finite)
    monkeypatch.setattr(loop, "_try_exact_face", promoted)


@pytest.mark.parametrize("outer", ["efs", "efs+finish", "efs+stationary"])
def test_the_rounding_of_a_caps_objective_does_not_decide_its_face(monkeypatch, outer) -> None:
    """A cap's objective that reads a rounding above or below its face's changes nothing.

    In the data's coordinates the stored penalty is singular only to rounding,
    and at a bound of 2e10 it leaves the objective there undetermined by 3e-6,
    ten times the comparison tolerance, where the face's is exact to 1e-14. The
    cap's objective is read three tolerances higher and then three lower. Either
    way the endpoint derivative promotes ``z``'s face and the fit is the
    unshifted one. Mutation check: compared at the tolerance alone, the lower
    reading refuses the face, and the fit ends ``lambda_cap_unresolved`` or
    converged without it.
    """
    _, reference = _fit(outer, data_coordinates=True)
    shift = 3.0 * reference.config.objective_tolerance * (1.0 + abs(reference.objective))

    for reading in (shift, -shift):
        with monkeypatch.context() as patch:
            _read_finite_objectives_shifted(patch, reading)
            model, smoothing = _fit(outer, data_coordinates=True)
        assert model.exact_face_components_ == (_Z,)
        assert smoothing.convergence_reason == reference.convergence_reason
        assert smoothing.converged is True
        assert smoothing.objective == reference.objective
        assert dict(smoothing.lambdas) == dict(reference.lambdas)


@pytest.mark.pins_behaviour
@pytest.mark.parametrize(("data_coordinates", "tolerances"), [(True, 1.0e3), (False, 3.0)])
def test_a_face_above_its_cap_by_more_than_rounding_is_refused(
    monkeypatch, data_coordinates: bool, tolerances: float
) -> None:
    """Pins the refusal of a face whose objective is materially above its cap's.

    In the data's coordinates the cap's objective is read a part in a million
    lower, a hundred times the penalty's rounding there. In its penalty's null
    coordinates superglm's own spline leaves the cap's objective a rounding of
    1e-14, so three tolerances lower is material there. Either way the face is
    higher by more than rounding explains and the endpoint derivative does not
    decide alone.
    """
    _, reference = _fit("efs", data_coordinates=data_coordinates)
    shift = tolerances * reference.config.objective_tolerance * (1.0 + abs(reference.objective))
    _read_finite_objectives_shifted(monkeypatch, -shift)
    model, smoothing = _fit("efs", data_coordinates=data_coordinates)

    assert model.exact_face_components_ == ()
    assert (smoothing.converged, smoothing.convergence_reason) == (False, "lambda_cap_unresolved")


@pytest.mark.parametrize("outer", ["efs", "efs+finish"])
def test_the_rounding_of_a_held_caps_objective_does_not_shorten_the_steps_beside_it(
    monkeypatch, outer
) -> None:
    """A step beside a held cap is taken whole whichever way the cap's objective rounds.

    While ``z`` is held at its bound of 2e10 in the data's coordinates the
    objective is undetermined by 3e-6, ten times the comparison tolerance, and
    ``x``'s last step changes it by far less than that tolerance. Every trial is
    read three tolerances above the objective it is compared with, and then
    three below. Either way each round takes the step proposed and the fit is
    the unshifted one. Mutation check: compared at the tolerance alone, the
    higher reading refuses ``x``'s last step at every length, nine fits. Plain
    ``"efs"`` then ends ``objective_rejected`` and ``"efs+finish"`` at another
    smoothing parameter.
    """
    from superglm.distributional.smoothing import loop

    _, reference = _fit(outer, data_coordinates=True)
    tolerance = reference.config.objective_tolerance * (1.0 + abs(reference.objective))
    matched = loop.rank_matched_objectives

    for reading in (3.0 * tolerance, -3.0 * tolerance):
        risen = []

        def read(current, current_objective, trial, trial_objective, reading=reading, risen=risen):
            floor, candidate = matched(current, current_objective, trial, trial_objective)
            risen.append(candidate + reading > floor + tolerance)
            return floor, candidate + reading

        with monkeypatch.context() as patch:
            patch.setattr(loop, "rank_matched_objectives", read)
            _, smoothing = _fit(outer, data_coordinates=True)
        assert any(risen) is (reading > 0.0)
        assert [row.coefficient_fit_indices for row in smoothing.history] == [
            row.coefficient_fit_indices for row in reference.history
        ]
        assert smoothing.convergence_reason == reference.convergence_reason
        assert smoothing.objective == reference.objective
        assert dict(smoothing.lambdas) == dict(reference.lambdas)


@pytest.mark.parametrize(
    ("outer", "options"),
    [("efs+newton", _PLATEAU), ("efs+stationary", {"terms": ("x", "z", "w")})],
)
def test_the_rounding_of_a_held_caps_gradient_does_not_move_the_terms_that_step(
    monkeypatch, outer, options
) -> None:
    """The Newton steps beside a held cap are the same whichever way its gradient rounds.

    The gradient in log lambda vanishes at a cap, so what the endgame reads for
    ``z`` there is rounding, and ``z`` does not move. Here it is read half as
    large again and half as large on alternate passes, keeping its sign. The
    other terms take the steps they take without that, to the same fit.
    Mutation check: with a term that did not move in the quasi-Newton pair, the
    ``"efs+newton"`` fit runs to its round limit unconverged and the
    ``"efs+stationary"`` fit takes 66 rounds for 12.
    """
    from superglm.distributional.smoothing import newton

    _, reference = _fit(outer, **options)
    derivatives, passes = newton.laml_derivatives, []

    def rounded(*args, lambdas, **kwargs):
        exact = derivatives(*args, lambdas=lambdas, **kwargs)
        held = np.array([lambdas[name] == reference.config.upper(name) for name in exact.names])
        passes.append(bool(held.any()))
        factor = 1.5 if len(passes) % 2 else 0.5
        return replace(exact, gradient=np.where(held, factor * exact.gradient, exact.gradient))

    monkeypatch.setattr(newton, "laml_derivatives", rounded)
    _, smoothing = _fit(outer, **options)

    assert sum(passes) >= 3
    assert (smoothing.converged, smoothing.convergence_reason) == (True, "stationary")
    assert smoothing.iterations == reference.iterations
    assert smoothing.objective == reference.objective
    assert dict(smoothing.lambdas) == dict(reference.lambdas)


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


def test_a_flat_tail_is_named_with_the_most_its_working_problem_falls() -> None:
    """The fall of the working criterion to zero variance is at most ``(-log(1 - E) - B) / 2``.

    With curvature d and working value z in each direction of an identity
    penalty, the criterion is
    ``sum(d lambda z^2 / (d + lambda) + log(1 + d / lambda)) / 2`` and tends to
    ``sum(d z^2) / 2``. One direction with no fitted quadratic form meets the
    bound; two directions with one fall short of it and still fall. Mutation
    checks: without B the second block's bound is not the stated one; with
    ``E / 2`` for the logarithm the first block falls further than its bound.
    """
    config = DistributionalEFSConfig()
    for diagonal, beta, met in (([0.25], (0.0,), True), ([0.12, 0.07], (0.1, 0.1), False)):
        d, b = np.array(diagonal), np.array(beta)
        component, inverse, update = _block(np.diag(d), beta=b)
        blocks = (("a", slice(0, len(d))),)
        assert flat_tail_names((component,), blocks, update, inverse, config) == ("a",)
        bound = flat_tail_fall(component, update)
        z = (d + 1.0) * b / d  # the penalised solution is d z / (d + lambda) at lambda = 1
        fall = 0.5 * float(np.sum(d * z**2 / (d + 1.0) + np.log1p(d)) - np.sum(d * z**2))
        edf, quadratic = float(np.sum(d / (d + 1.0))), float(b @ b)
        assert bound == pytest.approx(0.5 * (-math.log1p(-edf) - quadratic), rel=1e-12)
        assert 0.0 < fall <= bound * (1.0 + 1e-12)
        assert (fall == pytest.approx(bound, rel=1e-12)) is met


def test_flat_tails_are_judged_by_the_fall_they_can_bring_together() -> None:
    """One trial takes every untried tail, so they are resolved as one, and go as one.

    A tail the step itself puts on its bound owes its face an assessment, so
    the set is tried whatever it can gain. Mutation checks: judged one by one,
    the pair is not named; counting a tried tail's fall, or its arrival, the one
    left is named; naming the arriving tail alone leaves the other to walk.
    """
    tails = {"a": 0.4, "b": 0.4}
    assert resolved_flat_tails(tails, set(), 0.6) == ("a", "b")
    assert resolved_flat_tails({"a": 0.4}, set(), 0.6) == ()
    assert resolved_flat_tails(tails, {"b"}, 0.6) == ()
    assert resolved_flat_tails(tails, {"b"}, 0.3) == ("a",)
    assert resolved_flat_tails(tails, set(), 0.8) == ()
    assert resolved_flat_tails(tails, set(), 0.8, arriving=("b",)) == ("a", "b")
    assert resolved_flat_tails(tails, {"b"}, 0.8, arriving=("b",)) == ()
    assert resolved_flat_tails(tails, set(), -math.inf) == ("a", "b")
    assert resolved_flat_tails({}, set(), -math.inf, arriving=("a",)) == ()


def test_the_trial_waits_for_the_other_components_and_lands_on_each_ones_own_bound() -> None:
    """No trial while another component still proposes more than a log unit; none twice.

    The proposal puts each tried component on its own bound, which a relative box
    scales by the component's unit. Mutation checks: without the gate the trial is
    made beside a 1.5-unit step; with the global ceiling the scaled component
    lands a factor of its unit away.
    """
    for step, expected in ((0.9, ("a",)), (-0.9, ("a",)), (1.5, ()), (-1.5, ())):
        moved = {"a": 2.0, "b": step}
        assert endpoint_trial_names(("a",), moved, ("a", "b"), set()) == expected
    settled = {"a": 2.0, "b": 0.1}
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
