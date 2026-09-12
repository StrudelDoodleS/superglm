"""Practical outward completion may precede the optional Newton phase."""

import math

import numpy as np
import pytest

from superglm import GaussianLS, Predictor, RandomEffect, SuperLSS
from superglm.distributional.smoothing import loop
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
    model = SuperLSS(
        family=GaussianLS(scale_floor=1.0e-4),
        predictors=(Predictor("location", {"effect": RandomEffect()}), Predictor("scale", {})),
    )
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
