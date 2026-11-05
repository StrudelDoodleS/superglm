"""Practical outward completion may precede the optional Newton phase."""

import dataclasses
import math
import warnings

import numpy as np
import pandas as pd
import pytest

from superglm import ConvergenceWarning, GaussianLS, SuperLSS, re, s
from superglm.distributional import GeneralizedParetoLSS, Predictor, TweedieLSS
from superglm.distributional.model import fit_dense_distributional
from superglm.distributional.posterior import posterior_covariance
from superglm.distributional.results.iteration import DistributionalEFSConfig
from superglm.distributional.results.solver import DenseSolverConfig
from superglm.distributional.smoothing import loop
from superglm.distributional.smoothing import newton as newton_module
from superglm.distributional.weights import WeightContract
from tests.test_distributional_endpoint_direction import (
    _cap_start,
    _linear_x_fixture,
    _mean_predictor,
    _tweedie_response,
)
from tests.test_distributional_practical_stop import _preempt_fixture


@pytest.mark.parametrize("outer", ["efs", "efs+newton", "efs+finish"])
@pytest.mark.parametrize("budget", [3, 4, 20])
def test_outward_practical_plateau_can_finish_before_newton(monkeypatch, outer, budget):
    """A sustained finite plateau needs no remaining budget for an exact face.

    Forcing a Newton handoff at this plateau changes the three-iteration fit
    into an unresolved cap, and the four-iteration fit exhausts its budget
    during face validation. Neither establishes that the practical fit failed.
    """
    real_endgame = loop.run_newton_endgame
    calls = []

    def record_endgame(*args, **kwargs):
        calls.append(kwargs["state"])
        return real_endgame(*args, **kwargs)

    monkeypatch.setattr(loop, "run_newton_endgame", record_endgame)
    frame, response = _preempt_fixture()
    family = GaussianLS(scale_floor=1.0e-4)
    model = SuperLSS(family, family.location(re("effect")), family.scale())
    model.fit_reml(
        frame,
        response,
        lambdas={"location:effect#wiggle": 1.0e6},
        max_lambda=1.0e6 * math.exp(1.8),
        max_log_step=0.6,
        max_reml_iter=budget,
        reml_tol=1.0e-8,
        inner_tol=1.0e-10,
        reml_plateau_tol=1.0e-6,
        practical_reml=True,
        outer=outer,
    )

    fitted = model._require_fitted()
    smoothing = fitted.smoothing
    assert fitted.result.converged
    assert np.all(np.isfinite(fitted.covariance))
    assert smoothing.converged
    assert smoothing.convergence_reason == "practical_plateau"
    assert not smoothing.matched_certified
    assert not calls
    assert smoothing.iterations <= budget
    assert smoothing.terminal_fit.coefficient_face is None
    assert smoothing.unresolved_upper_bound == ("location:effect#wiggle",)
    assert smoothing.terminal_raw_log_steps["location:effect#wiggle"] > 0.0
    assert smoothing.terminal_gradient is None


def _two_smooth_gaussian():
    """A small location-scale fit whose default search reaches an interior plateau."""
    rng = np.random.default_rng(7351)
    x = rng.uniform(-1.0, 1.0, 320)
    response = 0.6 * np.sin(2.4 * x) + np.exp(-1.0 + 0.5 * np.cos(1.8 * x)) * rng.standard_normal(
        len(x)
    )
    family = GaussianLS()
    model = SuperLSS(
        family,
        family.location(s("x", kind="cr", k=7)),
        family.scale(s("x", kind="cr", k=5)),
    )
    return model, pd.DataFrame({"x": x}), response


def _spy_on_the_endgame(monkeypatch, outcome=lambda outcome: outcome):
    real_endgame = loop.run_newton_endgame
    calls = []

    def record_endgame(*args, **kwargs):
        calls.append(kwargs)
        return outcome(real_endgame(*args, **kwargs))

    monkeypatch.setattr(loop, "run_newton_endgame", record_endgame)
    return calls


def test_a_default_plateau_finishes_tighter_than_its_own_bar_within_the_budget(monkeypatch):
    calls = _spy_on_the_endgame(monkeypatch)
    model, frame, response = _two_smooth_gaussian()
    model.fit_reml(frame, response, reml_tol=1e-6)

    fitted = model._require_fitted()
    smoothing = fitted.smoothing
    (call,) = calls
    assert call["efs_config"].outer == "efs+newton"
    assert call["efs_config"].tolerance == pytest.approx(1e-7)
    assert call["budget"] == loop._FINISH_ITERATIONS
    assert smoothing.config.outer == "efs+finish"
    assert smoothing.convergence_reason == "stationary"
    assert smoothing.matched_certified
    # A stationary stop is what the correction expands about, so nothing is disclosed.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        corrected = posterior_covariance(fitted, kind="corrected")
    assert np.all(np.diag(corrected) >= np.diag(fitted.inference.covariance))


def test_an_explicit_efs_fit_never_runs_the_endgame(monkeypatch):
    calls = _spy_on_the_endgame(monkeypatch)
    model, frame, response = _two_smooth_gaussian()
    model.fit_reml(frame, response, outer="efs")

    smoothing = model._require_fitted().smoothing
    assert not calls
    assert smoothing.newton_iterations == 0
    assert smoothing.convergence_reason != "stationary"


def test_a_plateau_on_an_exact_face_finishes_on_the_components_off_it(monkeypatch):
    calls = _spy_on_the_endgame(monkeypatch)
    frame, mu = _linear_x_fixture(seed=6)
    smoothing = fit_dense_distributional(
        frame,
        _tweedie_response(mu, seed=6),
        family=TweedieLSS(),
        predictors=(_mean_predictor(), Predictor("dispersion", {}), Predictor("power", {})),
        weight_contract=WeightContract("prior"),
        lambdas=_cap_start(),
        config=DenseSolverConfig(
            max_iterations=150, tolerance=1.0e-9, coefficient_curvature="observed"
        ),
        efs_config=DistributionalEFSConfig(
            max_iterations=120, tolerance=1.0e-12, practical_convergence=True, outer="efs+finish"
        ),
        retain_rows=True,
    ).smoothing

    assert "mean:x#wiggle" in calls[0]["state"].face.component_names
    assert "mean:x#wiggle" in smoothing.terminal_fit.coefficient_face.component_names
    assert smoothing.convergence_reason != "practical_plateau"


def _assert_the_plateau_stood(smoothing, plain):
    assert smoothing.converged
    assert smoothing.convergence_reason == "practical_plateau"
    assert not smoothing.matched_certified
    assert smoothing.newton_iterations == 0
    assert smoothing.objective == plain.objective
    assert dict(smoothing.lambdas) == dict(plain.lambdas)
    assert smoothing.iterations == plain.iterations
    assert len(smoothing.coefficient_fits) == len(plain.coefficient_fits)
    assert smoothing.terminal_fit.eta is not None


def test_the_outward_stop_and_the_interior_stop_reach_one_stationary_point(monkeypatch):
    """Rounding can decide which of the two a fit meets first, so it must not decide the answer.
    Mutation check: an outward stop with no finish publishes its plateau uncertified."""
    interior, frame, response = _two_smooth_gaussian()
    interior.fit_reml(frame, response)
    calls = _spy_on_the_endgame(monkeypatch)
    monkeypatch.setattr(loop, "_practical_outward_window", lambda **kwargs: True)
    outward, _, _ = _two_smooth_gaussian()
    outward.fit_reml(frame, response)

    reference, smoothing = interior._require_fitted().smoothing, outward._require_fitted().smoothing
    (call,) = calls
    assert call["efs_config"].outer == "efs+newton"
    assert smoothing.convergence_reason == reference.convergence_reason == "stationary"
    assert smoothing.matched_certified
    assert smoothing.objective == pytest.approx(reference.objective, abs=1e-6)


def test_an_outward_stop_keeps_its_plateau_when_the_finish_stalls(monkeypatch):
    calls = _spy_on_the_endgame(
        monkeypatch, lambda outcome: dataclasses.replace(outcome, kind="gradient_unresolved")
    )
    monkeypatch.setattr(loop, "_practical_outward_window", lambda **kwargs: True)
    model, frame, response = _two_smooth_gaussian()
    model.fit_reml(frame, response)

    smoothing = model._require_fitted().smoothing
    assert len(calls) == 1
    assert smoothing.convergence_reason == "practical_plateau"
    assert not smoothing.matched_certified


@pytest.mark.parametrize(
    "kind",
    ["gradient_unresolved", "max_iterations", "objective_rejected", "derivative_unavailable"],
)
def test_a_finish_that_stalled_leaves_the_fit_at_its_plateau(monkeypatch, kind):
    """A plateau held where the finish started says nothing about where it stalled."""

    def moved_then_stalled(outcome):
        kwargs = calls[-1]
        history, fits = kwargs["history"], kwargs["coefficient_fits"]
        history.append(dataclasses.replace(history[-1], iteration=len(history) + 1, stage="newton"))
        fits.append(fits[0])
        state = kwargs["state"]
        moved = dataclasses.replace(
            state,
            lambdas={name: 2.0 * value for name, value in state.lambdas.items()},
            fit=fits[0],
            objective=state.objective - 1.0,
            terminal_fit_index=len(fits) - 1,
        )
        return dataclasses.replace(outcome, kind=kind, state=moved)

    calls = _spy_on_the_endgame(monkeypatch, moved_then_stalled)
    model, frame, response = _two_smooth_gaussian()
    model.fit_reml(frame, response)
    plain_model, _, _ = _two_smooth_gaussian()
    plain_model.fit_reml(frame, response, outer="efs")

    fitted = model._require_fitted()
    assert len(calls) == 1
    _assert_the_plateau_stood(fitted.smoothing, plain_model._require_fitted().smoothing)
    with pytest.warns(ConvergenceWarning, match="not at a certified stationary point"):
        corrected = posterior_covariance(fitted, kind="corrected")
    assert np.all(np.isfinite(corrected))


def test_a_finish_that_moved_far_and_was_rejected_does_not_relabel_where_it_stalled():
    """Where Newton is rejected far from the plateau, the published state is the plateau's."""
    rng = np.random.default_rng(2)
    x, z = rng.uniform(-1.0, 1.0, (2, 100))
    response = rng.pareto(3.0, 100) * np.exp(0.21 * np.sin(2.6 * x)) + 0.001
    frame = pd.DataFrame({"x": x, "z": z})
    family = GeneralizedParetoLSS()
    fits = {}
    for outer in (None, "efs"):
        model = SuperLSS(
            family, family.scale(s("x", kind="cr", k=7)), family.shape(s("z", kind="cr", k=5))
        )
        fits[outer] = model.fit_reml(frame, response, outer=outer)._require_fitted().smoothing
    if fits[None].convergence_reason == "stationary":
        pytest.skip("this build's finish reaches stationarity from this plateau")
    _assert_the_plateau_stood(fits[None], fits["efs"])


def _forced_face_plateau_fit(monkeypatch, outcome, outer="efs+finish"):
    calls = _spy_on_the_endgame(monkeypatch, outcome)
    frame, mu = _linear_x_fixture(seed=6)
    smoothing = fit_dense_distributional(
        frame,
        _tweedie_response(mu, seed=6),
        family=TweedieLSS(),
        predictors=(_mean_predictor(), Predictor("dispersion", {}), Predictor("power", {})),
        weight_contract=WeightContract("prior"),
        lambdas=_cap_start(),
        config=DenseSolverConfig(
            max_iterations=150, tolerance=1.0e-9, coefficient_curvature="observed"
        ),
        efs_config=DistributionalEFSConfig(
            max_iterations=40, tolerance=1.0e-12, practical_convergence=True, outer=outer
        ),
        retain_rows=True,
    ).smoothing
    return calls, smoothing


def _refuse_every_exact_face(monkeypatch):
    monkeypatch.setattr(loop, "_try_joint_exact_face", lambda *args, **kwargs: (None, None))
    monkeypatch.setattr(loop, "_try_exact_face", lambda *args, **kwargs: (None, None))


@pytest.mark.parametrize("outer", ["efs+newton", "efs+finish"])
def test_a_refused_cap_lets_the_exact_gradient_decide_first(monkeypatch, outer):
    """A nomination is Fellner--Schall's proposal; the exact gradient answers it before a refusal.
    Mutation check: a finish that returns at the refusal never reaches the endgame."""
    _refuse_every_exact_face(monkeypatch)
    calls, _ = _forced_face_plateau_fit(monkeypatch, lambda outcome: outcome, outer)

    assert calls
    assert calls[0]["state"].evidence.unresolved_upper_bound


def test_plain_efs_has_no_gradient_to_answer_a_refused_cap(monkeypatch):
    _refuse_every_exact_face(monkeypatch)
    calls, smoothing = _forced_face_plateau_fit(monkeypatch, lambda outcome: outcome, "efs")

    assert not calls
    assert smoothing.convergence_reason == "lambda_cap_unresolved"


def test_a_stall_on_a_refused_cap_stays_unresolved(monkeypatch):
    """No plateau held and a stall decides nothing about the nominated cap.
    Mutation check: returning the plateau's stop here reports convergence that never happened."""
    _refuse_every_exact_face(monkeypatch)
    calls, smoothing = _forced_face_plateau_fit(
        monkeypatch, lambda outcome: dataclasses.replace(outcome, kind="max_iterations")
    )

    assert len(calls) == 1
    assert smoothing.convergence_reason == "lambda_cap_unresolved"
    assert not smoothing.converged


def test_a_finish_on_a_face_that_cannot_reach_stationary_leaves_the_stop_to_the_strict_test(
    monkeypatch,
):
    """Only the strict stop revalidates a face, so a plateau cannot publish one unchecked."""
    calls, smoothing = _forced_face_plateau_fit(
        monkeypatch, lambda outcome: dataclasses.replace(outcome, kind="max_iterations")
    )

    assert len(calls) == 1
    assert calls[0]["state"].face is not None
    assert smoothing.convergence_reason != "practical_plateau"
    # The search went back to Fellner--Schall from the plateau's state.
    assert smoothing.history[-1].stage == "efs"
    assert smoothing.terminal_gradient is None
    if smoothing.converged:
        assert tuple(smoothing.terminal_endpoint_directions) == tuple(
            smoothing.terminal_fit.coefficient_face.component_names
        )


def test_a_gradient_within_reml_tol_at_a_cap_is_not_cap_pressure(monkeypatch):
    """The finish stops an order tighter than ``reml_tol``; the cap decision stays at it."""
    stop_bar = 1e-4
    real_endgame = loop.run_newton_endgame
    real_derivatives = newton_module.laml_derivatives
    outward = {}
    verdicts = {}

    def forced_gradient(*args, **kwargs):
        derivatives = real_derivatives(*args, **kwargs)
        gradient = np.zeros_like(derivatives.gradient)
        gradient[derivatives.names.index(outward["name"])] = outward["value"]
        return dataclasses.replace(
            derivatives,
            gradient=gradient,
            gradient_certificate=np.zeros_like(derivatives.gradient_certificate),
        )

    def judge_then_run(*args, **kwargs):
        state = kwargs["state"]
        assert kwargs["cap_tolerance"] == state_tolerance
        outward["name"] = next(iter(state.lambdas))
        outward["value"] = -5.0 * stop_bar * (1.0 + abs(state.objective))
        with monkeypatch.context() as forced:
            forced.setattr(newton_module, "laml_derivatives", forced_gradient)
            for label, cap_tolerance in (("stop bar", None), ("reml_tol", 10.0 * stop_bar)):
                verdicts[label] = real_endgame(
                    *args,
                    **{
                        **kwargs,
                        "efs_config": dataclasses.replace(kwargs["efs_config"], tolerance=stop_bar),
                        "cap_tolerance": cap_tolerance,
                        "upper_bounds": {outward["name"]: state.lambdas[outward["name"]]},
                        "history": list(kwargs["history"]),
                        "coefficient_fits": list(kwargs["coefficient_fits"]),
                    },
                ).kind
        return real_endgame(*args, **kwargs)

    monkeypatch.setattr(loop, "run_newton_endgame", judge_then_run)
    state_tolerance = 1e-6
    model, frame, response = _two_smooth_gaussian()
    model.fit_reml(frame, response, reml_tol=state_tolerance)

    assert verdicts == {"stop bar": "cap_pressure", "reml_tol": "stationary"}


@pytest.mark.parametrize("outward", [False, True], ids=["interior-stop", "outward-stop"])
def test_a_finish_that_finds_cap_pressure_discloses_it(monkeypatch, outward):
    """Both stops report the same outcome, so rounding between them cannot choose one.
    Mutation check: an outward finish counting cap pressure as a stall keeps its plateau."""

    def at_the_cap(outcome):
        return dataclasses.replace(
            outcome, kind="cap_pressure", cap_pressure=("location:x#wiggle",)
        )

    calls = _spy_on_the_endgame(monkeypatch, at_the_cap)
    if outward:
        monkeypatch.setattr(loop, "_practical_outward_window", lambda **kwargs: True)
    model, frame, response = _two_smooth_gaussian()
    model.fit_reml(frame, response)

    fitted = model._require_fitted()
    smoothing = fitted.smoothing
    assert len(calls) == 1
    assert np.all(np.isfinite(fitted.inference.covariance))
    assert smoothing.convergence_reason == "lambda_cap_unresolved"
    assert not smoothing.converged


def test_a_default_fit_replays_with_the_finish():
    model, frame, response = _two_smooth_gaussian()
    model.fit_reml(frame, response)
    assert model._clone_with_replaced_terms({})._fit_settings["outer"] == "efs+finish"
