"""Practical outward completion may precede the optional Newton phase."""

import dataclasses
import math
import warnings

import numpy as np
import pandas as pd
import pytest

from superglm import ConvergenceWarning, GaussianLS, SuperLSS, re, s
from superglm.distributional import Predictor, TweedieLSS
from superglm.distributional.model import fit_dense_distributional
from superglm.distributional.posterior import posterior_covariance
from superglm.distributional.results.iteration import DistributionalEFSConfig
from superglm.distributional.results.solver import DenseSolverConfig
from superglm.distributional.smoothing import loop
from superglm.distributional.weights import WeightContract
from tests.test_distributional_endpoint_direction import (
    _cap_start,
    _linear_x_fixture,
    _mean_predictor,
    _tweedie_response,
)
from tests.test_distributional_practical_stop import _preempt_fixture


@pytest.mark.parametrize("outer", ["efs", "efs+newton"])
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

    (call,) = calls
    assert "mean:x#wiggle" in call["state"].face.component_names
    assert smoothing.convergence_reason == "stationary"
    assert "mean:x#wiggle" in smoothing.terminal_fit.coefficient_face.component_names


@pytest.mark.parametrize("kind", ["gradient_unresolved", "max_iterations", "objective_rejected"])
def test_a_finish_that_cannot_reach_stationary_keeps_the_plateau(monkeypatch, kind):
    calls = _spy_on_the_endgame(
        monkeypatch, lambda outcome: dataclasses.replace(outcome, kind=kind)
    )
    model, frame, response = _two_smooth_gaussian()
    model.fit_reml(frame, response)

    fitted = model._require_fitted()
    smoothing = fitted.smoothing
    assert len(calls) == 1
    assert smoothing.converged
    assert smoothing.convergence_reason == "practical_plateau"
    assert not smoothing.matched_certified
    assert smoothing.newton_iterations >= 1
    # The endgame has run, so the disclosure does not send the user back to it.
    with pytest.warns(ConvergenceWarning, match="not at a certified stationary point") as record:
        corrected = posterior_covariance(fitted, kind="corrected")
    assert not any("efs+newton" in str(item.message) for item in record)
    assert np.all(np.isfinite(corrected))


def test_a_finish_that_finds_cap_pressure_discloses_it(monkeypatch):
    def at_the_cap(outcome):
        return dataclasses.replace(
            outcome, kind="cap_pressure", cap_pressure=("location:x#wiggle",)
        )

    calls = _spy_on_the_endgame(monkeypatch, at_the_cap)
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
