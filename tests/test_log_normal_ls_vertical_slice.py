"""Public fits and cross-family identities for ``LogNormalLS``."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from superglm import Spline, SuperLSS
from superglm.distributional import GaussianLS, Predictor
from superglm.distributional.families.generalized_gamma import GeneralizedGammaLSS
from superglm.distributional.families.log_normal import LogNormalLS
from superglm.distributional.families.two_piece import TwoPieceLogNormalLSS
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.weights import WeightContract, resolve_likelihood_weights
from superglm.features import Numeric


def _simulate(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    z = rng.uniform(-1.0, 1.0, n)
    log_mean = 1.0 + 0.6 * np.sin(np.pi * x)
    sigma = np.exp(-0.4 + 0.3 * z)
    y = np.exp(log_mean - 0.5 * sigma**2 + sigma * rng.standard_normal(n))
    return pd.DataFrame({"x": x, "z": z}), y, log_mean, sigma


def _predictors(first="mean"):
    return (
        Predictor(first, {"x": Spline(kind="cr", k=8)}),
        Predictor("scale", {"z": Numeric()}),
    )


def _frequency_weights(n):
    return resolve_likelihood_weights(
        np.ones(n), n_observations=n, contract=WeightContract(semantics="frequency")
    )


def test_mean_form_recovers_the_simulated_surfaces_under_reml():
    frame, y, log_mean, sigma = _simulate(4000, 20260902)
    model = SuperLSS(family=LogNormalLS(), predictors=_predictors()).fit_reml(
        frame, y, method="efs"
    )
    fitted = model.predict_parameters(frame)
    assert math.sqrt(np.mean((np.log(fitted["mean"].to_numpy()) - log_mean) ** 2)) <= 0.05
    assert math.sqrt(np.mean((np.log(fitted["scale"].to_numpy()) - np.log(sigma)) ** 2)) <= 0.08
    assert np.array_equal(model.predict(frame), fitted["mean"].to_numpy())
    p90 = model.predict_quantile(frame, 0.9)
    assert np.allclose(model.predict_cdf(frame, p90), 0.9, rtol=0, atol=1e-12)
    assert np.mean(y <= p90) == pytest.approx(0.9, abs=0.02)


def test_location_form_fits_and_agrees_with_the_mean_form():
    frame, y, _, _ = _simulate(1500, 7)
    mean_form = SuperLSS(
        family=LogNormalLS(),
        predictors=(Predictor("mean", {"x": Numeric()}), Predictor("scale", {})),
    ).fit(frame, y)
    location_form = SuperLSS(
        family=LogNormalLS(parametrisation="location"),
        predictors=(Predictor("location", {"x": Numeric()}), Predictor("scale", {})),
    ).fit(frame, y)
    assert np.allclose(mean_form.predict(frame), location_form.predict(frame), rtol=1e-5, atol=0)
    assert np.allclose(
        mean_form.predict_quantile(frame, 0.5),
        location_form.predict_quantile(frame, 0.5),
        rtol=1e-5,
        atol=0,
    )


def test_fisher_curvature_request_is_honoured():
    frame, y, _, _ = _simulate(800, 3)
    model = SuperLSS(
        family=LogNormalLS(), predictors=_predictors(), coefficient_curvature="fisher"
    ).fit(frame, y, lambdas={"mean:x#wiggle": 1.0})
    assert model.coefficient_curvature == "fisher"
    assert np.all(np.isfinite(model.predict(frame)))


def test_location_form_is_gaussian_on_log_y_up_to_the_jacobian():
    """The identity that makes the family worth having: same fit, mean on the y scale."""
    frame, y, _, _ = _simulate(1200, 41)
    predictors = (Predictor("location", {"x": Numeric()}), Predictor("scale", {"z": Numeric()}))
    log_normal = SuperLSS(
        family=LogNormalLS(parametrisation="location"), predictors=predictors
    ).fit(frame, y)
    gaussian = SuperLSS(family=GaussianLS(), predictors=predictors).fit(frame, np.log(y))
    ours = log_normal.predict_parameters(frame)
    theirs = gaussian.predict_parameters(frame)
    # Two independent optimiser runs on the same likelihood stop at slightly different
    # points, so the bound is the solver's, not the identity's: measured worst
    # |a - b| / (1 + |b|) is 3.54e-08 on location and 1.77e-08 on scale.
    for name in ("location", "scale"):
        a, b = ours[name].to_numpy(), theirs[name].to_numpy()
        assert np.max(np.abs(a - b) / (1.0 + np.abs(b))) <= 1e-7


@pytest.mark.parametrize("parametrisation", ["mean", "location"])
def test_log_normal_equals_the_generalized_gamma_at_shape_zero(parametrisation):
    y = np.array([0.4, 1.3, 2.8, 7.5, 40.0])
    weights = _frequency_weights(len(y))
    first = 2.0 if parametrisation == "mean" else 0.3
    theta_two = np.tile([first, 0.8], (len(y), 1))
    theta_three = np.tile([first, 0.8, 0.0], (len(y), 1))
    log_normal = LogNormalLS(parametrisation=parametrisation)
    generalized = GeneralizedGammaLSS(parametrisation=parametrisation)
    ours = log_normal.evaluate_natural(
        y, theta_two, log_normal.bind_likelihood(y, weights, COMPLETE_OBSERVATION)
    )
    theirs = generalized.evaluate_natural(
        y, theta_three, generalized.bind_likelihood(y, weights, COMPLETE_OBSERVATION)
    )
    assert np.allclose(
        ours.reported_log_likelihood, theirs.reported_log_likelihood, rtol=0, atol=1e-14
    )
    assert np.allclose(ours.score, theirs.score[:, :2], rtol=0, atol=1e-14)
    assert np.allclose(ours.hessian_packed, theirs.hessian_packed[:, [0, 1, 3]], rtol=0, atol=1e-13)
    information_ours = log_normal.expected_information_natural(
        theta_two, log_normal.bind_likelihood(y, weights, COMPLETE_OBSERVATION)
    )
    information_theirs = generalized.expected_information_natural(
        theta_three, generalized.bind_likelihood(y, weights, COMPLETE_OBSERVATION)
    )
    assert np.allclose(information_ours, information_theirs[:, [0, 1, 3]], rtol=0, atol=1e-14)
    assert np.allclose(
        log_normal.quantile(np.full(len(y), 0.9), theta_two),
        generalized.quantile(np.full(len(y), 0.9), theta_three),
        rtol=1e-12,
        atol=0,
    )


@pytest.mark.parametrize("parametrisation", ["mean", "location"])
def test_log_normal_equals_the_two_piece_log_normal_at_zero_skew(parametrisation):
    y = np.array([0.4, 1.3, 2.8, 7.5, 40.0])
    weights = _frequency_weights(len(y))
    first = 2.0 if parametrisation == "mean" else 0.3
    theta_two = np.tile([first, 0.8], (len(y), 1))
    theta_three = np.tile([first, 0.8, 0.0], (len(y), 1))
    log_normal = LogNormalLS(parametrisation=parametrisation)
    two_piece = TwoPieceLogNormalLSS(parametrisation=parametrisation)
    ours = log_normal.evaluate_natural(
        y, theta_two, log_normal.bind_likelihood(y, weights, COMPLETE_OBSERVATION)
    )
    theirs = two_piece.evaluate_natural(
        y, theta_three, two_piece.bind_likelihood(y, weights, COMPLETE_OBSERVATION)
    )
    assert np.allclose(
        ours.reported_log_likelihood, theirs.reported_log_likelihood, rtol=0, atol=1e-14
    )
    assert np.allclose(ours.score, theirs.score[:, :2], rtol=0, atol=1e-14)
    assert np.allclose(ours.hessian_packed, theirs.hessian_packed[:, [0, 1, 3]], rtol=0, atol=1e-13)


import json  # noqa: E402
import subprocess  # noqa: E402
from functools import cache  # noqa: E402

from tests._r_harness import ROOT, r_environment, require_r_harness  # noqa: E402

_GAMLSS_PROBE = "suppressMessages(library(gamlss)); suppressMessages(library(jsonlite)); cat('ok')"


@cache
def _gamlss_available() -> bool:
    completed = subprocess.run(
        ["Rscript", "-e", _GAMLSS_PROBE],
        cwd=ROOT,
        env=r_environment(),
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0 and completed.stdout.strip().endswith("ok")


_GAMLSS_SCRIPT = """
suppressMessages(library(gamlss)); suppressMessages(library(jsonlite))
d <- read.csv(file("stdin"))
fit <- gamlss(y ~ x, sigma.formula = ~ 1, family = LOGNO, data = d,
              control = gamlss.control(c.crit = 1e-9, n.cyc = 400, trace = FALSE))
cat(toJSON(list(mu = as.numeric(coef(fit, what = "mu")),
                sigma = as.numeric(coef(fit, what = "sigma")),
                loglik = as.numeric(logLik(fit))), digits = 16))
"""


def test_location_form_matches_gamlss_logno_at_a_parametric_specification():
    require_r_harness()
    if not _gamlss_available():
        pytest.skip("requires R with gamlss and jsonlite")
    frame, y, _, _ = _simulate(600, 12)
    payload = pd.DataFrame({"x": frame["x"], "y": y}).to_csv(index=False)
    completed = subprocess.run(
        ["Rscript", "-e", _GAMLSS_SCRIPT],
        input=payload,
        cwd=ROOT,
        env=r_environment(),
        check=True,
        capture_output=True,
        text=True,
    )
    reference = json.loads(completed.stdout)
    model = SuperLSS(
        family=LogNormalLS(parametrisation="location"),
        predictors=(Predictor("location", {"x": Numeric()}), Predictor("scale", {})),
    ).fit(frame, y)
    theta = model.predict_parameters(frame)
    # gamlss LOGNO(mu, sigma) has an identity link on mu and a log link on sigma,
    # and its mu is our location: the mapping is pinned by these assertions.
    location = theta["location"].to_numpy()
    ours_slope, ours_intercept = (float(v) for v in np.polyfit(frame["x"].to_numpy(), location, 1))
    assert abs(ours_intercept - reference["mu"][0]) <= 1e-6 * (1 + abs(reference["mu"][0]))
    assert abs(ours_slope - reference["mu"][1]) <= 1e-6 * (1 + abs(reference["mu"][1]))
    sigma_ours = float(theta["scale"].iloc[0])
    assert abs(sigma_ours - math.exp(reference["sigma"][0])) <= 1e-6 * (1 + sigma_ours)
    family = LogNormalLS(parametrisation="location")
    plan = family.bind_likelihood(y, _frequency_weights(len(y)), COMPLETE_OBSERVATION)
    ours_loglik = float(
        np.sum(
            family.evaluate_natural(
                y, theta.to_numpy(), plan, derivative_order=0
            ).reported_log_likelihood
        )
    )
    reference_loglik = float(reference["loglik"][0])  # jsonlite wraps scalars in arrays
    assert abs(ours_loglik - reference_loglik) <= 1e-8 * (1 + abs(reference_loglik))
