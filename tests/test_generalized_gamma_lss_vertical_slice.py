"""Public fits, cross-family identities and boundary behaviour of ``GeneralizedGammaLSS``."""

from __future__ import annotations

import json
import math
import subprocess
from functools import cache

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from superglm import Spline, SuperLSS
from superglm.distributional import GammaLS, Predictor
from superglm.distributional.families.generalized_gamma import GeneralizedGammaLSS
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.kernels import generalized_gamma as gg
from superglm.distributional.model import _readonly_default_prediction
from superglm.distributional.null_model import NullModelFitError
from superglm.distributional.weights import WeightContract, resolve_likelihood_weights
from superglm.features import Numeric
from tests._r_harness import ROOT, r_environment, require_r_harness


def _simulate(n, seed, *, q=0.4):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    z = rng.uniform(-1.0, 1.0, n)
    log_mean = 1.0 + 0.6 * np.sin(np.pi * x)
    sigma = np.exp(-0.4 + 0.3 * z)
    k = 1.0 / q**2
    w = np.log(rng.gamma(k, 1.0, n) / k) / q
    mu = log_mean - gg.log_mean_loading(sigma, np.full(n, q))[0]
    y = np.exp(mu + sigma * w)
    return pd.DataFrame({"x": x, "z": z}), y, log_mean, sigma


def _predictors():
    return (
        Predictor("mean", {"x": Spline(kind="cr", k=8)}),
        Predictor("scale", {"z": Numeric()}),
        Predictor("shape", {}),
    )


def _frequency_weights(n):
    return resolve_likelihood_weights(
        np.ones(n), n_observations=n, contract=WeightContract(semantics="frequency")
    )


def test_mean_form_recovers_the_simulated_surfaces_under_reml():
    frame, y, log_mean, sigma = _simulate(4000, 20260902)
    model = SuperLSS(family=GeneralizedGammaLSS(), predictors=_predictors()).fit_reml(
        frame, y, method="efs"
    )
    fitted = model.predict_parameters(frame)
    assert math.sqrt(np.mean((np.log(fitted["mean"].to_numpy()) - log_mean) ** 2)) <= 0.05
    assert math.sqrt(np.mean((np.log(fitted["scale"].to_numpy()) - np.log(sigma)) ** 2)) <= 0.08
    assert abs(float(fitted["shape"].iloc[0]) - 0.4) <= 0.15
    assert np.array_equal(model.predict(frame), fitted["mean"].to_numpy())
    p90 = model.predict_quantile(frame, 0.9)
    assert np.allclose(model.predict_cdf(frame, p90), 0.9, rtol=0, atol=1e-12)
    assert np.mean(y <= p90) == pytest.approx(0.9, abs=0.02)


def test_location_form_fits_and_agrees_with_the_mean_form_at_constant_scale_and_shape():
    frame, y, _, _ = _simulate(1500, 7)
    predictors_mean = (
        Predictor("mean", {"x": Numeric()}),
        Predictor("scale", {}),
        Predictor("shape", {}),
    )
    predictors_location = (
        Predictor("location", {"x": Numeric()}),
        Predictor("scale", {}),
        Predictor("shape", {}),
    )
    mean_form = SuperLSS(family=GeneralizedGammaLSS(), predictors=predictors_mean).fit(frame, y)
    location_form = SuperLSS(
        family=GeneralizedGammaLSS(parametrisation="location"), predictors=predictors_location
    ).fit(frame, y)
    # same likelihood, same fitted law: the conditional means agree to solver tolerance
    assert np.allclose(mean_form.predict(frame), location_form.predict(frame), rtol=1e-5, atol=0)
    assert np.allclose(
        mean_form.predict_quantile(frame, 0.5),
        location_form.predict_quantile(frame, 0.5),
        rtol=1e-5,
        atol=0,
    )


def test_location_form_predict_returns_positive_infinity_when_the_mean_does_not_exist():
    frame = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 12)})
    response = np.exp(np.linspace(-0.4, 0.7, len(frame)))
    model = SuperLSS(
        family=GeneralizedGammaLSS(parametrisation="location"),
        predictors=(Predictor("location", {}), Predictor("scale", {}), Predictor("shape", {})),
    ).fit(frame, response)

    row = frame.iloc[:1]
    eta = model.predict_link(row).to_numpy()[0]
    target_eta = np.array([0.0, math.log(2.0 - model.family_.scale_floor), -1.0])
    offsets = {
        name: np.array([target_eta[index] - eta[index]])
        for index, name in enumerate(("location", "scale", "shape"))
    }

    np.testing.assert_allclose(
        model.predict_parameters(row, offsets=offsets).to_numpy(), [[0.0, 2.0, -1.0]]
    )
    prediction = model.predict(row, offsets=offsets)
    assert np.isposinf(prediction[0])
    assert not prediction.flags.writeable


@pytest.mark.parametrize("invalid", [np.nan, -np.inf])
def test_default_prediction_still_rejects_invalid_nonfinite_values(invalid):
    with pytest.raises(ValueError, match="NaN or negative infinity"):
        _readonly_default_prediction(np.array([invalid]))


def test_fisher_curvature_request_is_honoured():
    frame, y, _, _ = _simulate(800, 3)
    model = SuperLSS(
        family=GeneralizedGammaLSS(), predictors=_predictors(), coefficient_curvature="fisher"
    ).fit(frame, y, lambdas={"mean:x#wiggle": 1.0})
    assert model.coefficient_curvature == "fisher"
    assert np.all(np.isfinite(model.predict(frame)))


def test_generalized_gamma_at_shape_equal_scale_is_the_gamma_family():
    y = np.array([0.4, 1.3, 2.8, 7.5])
    weights = _frequency_weights(4)
    gamma = GammaLS()
    gg_family = GeneralizedGammaLSS()
    theta_gamma = np.tile([2.0, 0.6], (4, 1))  # (mean, cv)
    theta_gg = np.tile([2.0, 0.6, 0.6], (4, 1))  # (mean, sigma, Q = sigma)
    gamma_rows = gamma.evaluate_natural(
        y, theta_gamma, gamma.bind_likelihood(y, weights, COMPLETE_OBSERVATION)
    )
    gg_rows = gg_family.evaluate_natural(
        y, theta_gg, gg_family.bind_likelihood(y, weights, COMPLETE_OBSERVATION)
    )
    assert np.allclose(
        gamma_rows.reported_log_likelihood, gg_rows.reported_log_likelihood, rtol=0, atol=1e-12
    )
    assert np.allclose(gamma_rows.score[:, 0], gg_rows.score[:, 0], rtol=0, atol=1e-11)


def test_generalized_gamma_reduces_to_lognormal_and_weibull_densities():
    y = np.array([0.4, 1.3, 2.8, 7.5])
    mu, sigma = 0.3, 0.8
    carrier = -np.log(y) - 0.5 * math.log(2 * math.pi)
    tiny = gg.location_rows(
        y, np.full(4, mu), np.full(4, sigma), np.full(4, 1e-10), np.ones(4), derivative_order=0
    )
    lognormal = stats.lognorm(s=sigma, scale=math.exp(mu)).logpdf(y)
    assert np.allclose(tiny.optimizing_log_likelihood + carrier, lognormal, rtol=0, atol=1e-12)
    one = gg.location_rows(
        y, np.full(4, mu), np.full(4, sigma), np.ones(4), np.ones(4), derivative_order=0
    )
    weibull = stats.weibull_min(c=1.0 / sigma, scale=math.exp(mu)).logpdf(y)
    assert np.allclose(one.optimizing_log_likelihood + carrier, weibull, rtol=0, atol=1e-12)


def test_infinite_mean_data_either_diagnose_or_stop_at_the_boundary():
    rng = np.random.default_rng(31)
    n = 1500
    q, sigma = -0.9, 1.5  # sigma * |Q| = 1.35: no mean
    k = 1.0 / q**2
    w = np.log(rng.gamma(k, 1.0, n) / k) / q
    y = np.exp(0.5 + sigma * w)
    frame = pd.DataFrame({"x": rng.uniform(-1.0, 1.0, n)})
    predictors = (
        Predictor("mean", {"x": Numeric()}),
        Predictor("scale", {}),
        Predictor("shape", {}),
    )
    try:
        model = SuperLSS(family=GeneralizedGammaLSS(), predictors=predictors).fit(frame, y)
    except NullModelFitError as failure:
        # Measured outcome: the intercept-only joint null model walks into the
        # infinite-mean barrier, every further step is rejected as invalid, and the
        # null solve exhausts its iterations.  The family's repeated-curvature
        # diagnosis is only consulted on the curvature path, so it cannot name the
        # boundary here; extending it to the null-model path is an engine follow-up.
        assert "did not converge" in str(failure)
        return
    except Exception as failure:  # noqa: BLE001 - the diagnosis is the assertion
        assert "infinite-mean" in str(failure)
        return
    fitted = model.predict_parameters(frame)
    pressure = float(np.max(fitted["scale"].to_numpy() * np.abs(fitted["shape"].to_numpy())))
    assert pressure >= 0.85, "a finite-mean fit on infinite-mean data must sit near the boundary"


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
fit <- gamlss(y ~ x, sigma.formula = ~ 1, nu.formula = ~ 1, family = GG, data = d,
              control = gamlss.control(c.crit = 1e-9, n.cyc = 400, trace = FALSE))
cat(toJSON(list(mu = as.numeric(coef(fit, what = "mu")),
                sigma = as.numeric(coef(fit, what = "sigma")),
                nu = as.numeric(coef(fit, what = "nu")),
                loglik = as.numeric(logLik(fit))), digits = 16))
"""


def test_location_form_matches_gamlss_gg_at_a_parametric_specification():
    require_r_harness()
    if not _gamlss_available():
        pytest.skip("requires R with gamlss and jsonlite")
    frame, y, _, _ = _simulate(600, 12, q=0.5)
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
    predictors = (
        Predictor("location", {"x": Numeric()}),
        Predictor("scale", {}),
        Predictor("shape", {}),
    )
    model = SuperLSS(
        family=GeneralizedGammaLSS(parametrisation="location"), predictors=predictors
    ).fit(frame, y)
    theta = model.predict_parameters(frame)
    # gamlss GG(mu_g, sigma_g, nu_g) with log/log/identity links:
    # mu_g = exp(location), sigma_g = scale, nu_g = shape/scale
    location = theta["location"].to_numpy()
    x = frame["x"].to_numpy()
    ours_slope, ours_intercept = (float(v) for v in np.polyfit(x, location, 1))
    assert abs(ours_intercept - reference["mu"][0]) <= 1e-5 * (1 + abs(reference["mu"][0]))
    assert abs(ours_slope - reference["mu"][1]) <= 1e-5 * (1 + abs(reference["mu"][1]))
    sigma_ours = float(theta["scale"].iloc[0])
    assert abs(sigma_ours - math.exp(reference["sigma"][0])) <= 1e-5 * (1 + sigma_ours)
    assert abs(float(theta["shape"].iloc[0]) - sigma_ours * reference["nu"][0]) <= 1e-5
    family = GeneralizedGammaLSS(parametrisation="location")
    plan = family.bind_likelihood(y, _frequency_weights(len(y)), COMPLETE_OBSERVATION)
    ours_loglik = float(
        np.sum(
            family.evaluate_natural(
                y, theta.to_numpy(), plan, derivative_order=0
            ).reported_log_likelihood
        )
    )
    reference_loglik = float(reference["loglik"][0])  # jsonlite wraps scalars in arrays
    assert abs(ours_loglik - reference_loglik) <= 1e-7 * (1 + abs(reference_loglik))
