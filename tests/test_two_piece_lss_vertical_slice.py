"""Public fits, cross-family identities and kink behaviour of the two-piece families."""

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
from superglm.distributional import GaussianLS, Predictor
from superglm.distributional.families.generalized_gamma import GeneralizedGammaLSS
from superglm.distributional.families.two_piece import TwoPieceLogNormalLSS, TwoPieceNormalLSS
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.kernels import two_piece as tp
from superglm.distributional.weights import WeightContract, resolve_likelihood_weights
from superglm.features import Numeric
from tests._r_harness import ROOT, r_environment, require_r_harness


def _frequency_weights(n):
    return resolve_likelihood_weights(
        np.ones(n), n_observations=n, contract=WeightContract(semantics="frequency")
    )


def _simulate(n, seed, *, eps=0.45):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    z = rng.uniform(-1.0, 1.0, n)
    log_mean = 1.0 + 0.6 * np.sin(np.pi * x)
    sigma = np.exp(-0.4 + 0.3 * z)
    skew = np.full(n, eps)
    mu = log_mean - tp.log_mean_loading(sigma, skew)[0]
    variate = tp.two_piece_quantile(rng.uniform(size=n), mu, sigma, skew)
    return pd.DataFrame({"x": x, "z": z}), np.exp(variate), log_mean, sigma


def _predictors(first="mean"):
    return (
        Predictor(first, {"x": Spline(kind="cr", k=8)}),
        Predictor("scale", {"z": Numeric()}),
        Predictor("skew", {}),
    )


def test_mean_form_recovers_the_simulated_surfaces_under_reml():
    frame, y, log_mean, sigma = _simulate(4000, 20260902)
    model = SuperLSS(family=TwoPieceLogNormalLSS(), predictors=_predictors()).fit_reml(
        frame, y, method="efs"
    )
    fitted = model.predict_parameters(frame)
    assert math.sqrt(np.mean((np.log(fitted["mean"].to_numpy()) - log_mean) ** 2)) <= 0.05
    assert math.sqrt(np.mean((np.log(fitted["scale"].to_numpy()) - np.log(sigma)) ** 2)) <= 0.10
    assert abs(float(fitted["skew"].iloc[0]) - 0.45) <= 0.12
    assert np.array_equal(model.predict(frame), fitted["mean"].to_numpy())
    p90 = model.predict_quantile(frame, 0.9)
    assert np.allclose(model.predict_cdf(frame, p90), 0.9, rtol=0, atol=1e-11)
    assert np.mean(y <= p90) == pytest.approx(0.9, abs=0.02)


def test_location_form_reaches_the_same_law_as_the_mean_form():
    frame, y, _, _ = _simulate(1500, 7)
    mean_form = SuperLSS(
        family=TwoPieceLogNormalLSS(),
        predictors=(
            Predictor("mean", {"x": Numeric()}),
            Predictor("scale", {}),
            Predictor("skew", {}),
        ),
    ).fit(frame, y)
    location_form = SuperLSS(
        family=TwoPieceLogNormalLSS(parametrisation="location"),
        predictors=(
            Predictor("location", {"x": Numeric()}),
            Predictor("scale", {}),
            Predictor("skew", {}),
        ),
    ).fit(frame, y)
    assert np.allclose(mean_form.predict(frame), location_form.predict(frame), rtol=1e-4, atol=0)
    assert np.allclose(
        mean_form.predict_quantile(frame, 0.5),
        location_form.predict_quantile(frame, 0.5),
        rtol=1e-4,
        atol=0,
    )


def test_the_real_line_family_recovers_a_skewed_location_scale_surface():
    rng = np.random.default_rng(101)
    n = 3000
    x = rng.uniform(-1.0, 1.0, n)
    location = 0.5 + 1.2 * x
    sigma = np.full(n, 0.8)
    skew = np.full(n, -0.5)
    y = tp.two_piece_quantile(rng.uniform(size=n), location, sigma, skew)
    frame = pd.DataFrame({"x": x})
    model = SuperLSS(
        family=TwoPieceNormalLSS(),
        predictors=(
            Predictor("location", {"x": Numeric()}),
            Predictor("scale", {}),
            Predictor("skew", {}),
        ),
    ).fit(frame, y)
    fitted = model.predict_parameters(frame)
    assert math.sqrt(np.mean((fitted["location"].to_numpy() - location) ** 2)) <= 0.08
    assert abs(float(fitted["scale"].iloc[0]) - 0.8) <= 0.05
    assert abs(float(fitted["skew"].iloc[0]) + 0.5) <= 0.08
    assert np.allclose(
        model.predict(frame),
        fitted["location"].to_numpy()
        + 2.0 * fitted["skew"].to_numpy() * fitted["scale"].to_numpy() * math.sqrt(2.0 / math.pi),
        rtol=1e-12,
        atol=0,
    )


def test_two_piece_log_normal_at_zero_skew_is_the_generalized_gamma_at_shape_zero():
    y = np.array([0.4, 1.3, 2.8, 7.5])
    weights = _frequency_weights(4)
    two_piece = TwoPieceLogNormalLSS()
    generalized = GeneralizedGammaLSS()
    theta_two_piece = np.tile([2.0, 0.8, 0.0], (4, 1))
    theta_generalized = np.tile([2.0, 0.8, 0.0], (4, 1))
    ours = two_piece.evaluate_natural(
        y, theta_two_piece, two_piece.bind_likelihood(y, weights, COMPLETE_OBSERVATION)
    )
    theirs = generalized.evaluate_natural(
        y, theta_generalized, generalized.bind_likelihood(y, weights, COMPLETE_OBSERVATION)
    )
    assert np.allclose(
        ours.reported_log_likelihood, theirs.reported_log_likelihood, rtol=0, atol=1e-14
    )
    for channel in (0, 1):  # the mean and scale channels; the third is a different shape
        assert np.allclose(ours.score[:, channel], theirs.score[:, channel], rtol=0, atol=1e-14)
    for channel in (0, 1, 3):
        assert np.allclose(
            ours.hessian_packed[:, channel], theirs.hessian_packed[:, channel], rtol=0, atol=1e-13
        )


def test_two_piece_normal_at_zero_skew_is_the_gaussian_family():
    y = np.array([-1.2, 0.3, 2.4])
    weights = _frequency_weights(3)
    two_piece = TwoPieceNormalLSS()
    gaussian = GaussianLS()
    theta = np.array([[0.4, 0.9, 0.0]] * 3)
    ours = two_piece.evaluate_natural(
        y, theta, two_piece.bind_likelihood(y, weights, COMPLETE_OBSERVATION)
    )
    theirs = gaussian.evaluate_natural(
        y, theta[:, :2], gaussian.bind_likelihood(y, weights, COMPLETE_OBSERVATION)
    )
    assert np.allclose(
        ours.reported_log_likelihood, theirs.reported_log_likelihood, rtol=0, atol=1e-14
    )
    assert np.allclose(ours.score[:, :2], theirs.score, rtol=0, atol=1e-14)
    # their packed (0,0), (0,1), (1,1) are our channels 0, 1 and 3
    assert np.allclose(ours.hessian_packed[:, [0, 1, 3]], theirs.hessian_packed, rtol=0, atol=1e-14)
    assert np.allclose(two_piece.cdf(y, theta), stats.norm(0.4, 0.9).cdf(y), rtol=0, atol=1e-15)


def test_the_hessian_kink_does_not_hide_behind_a_silent_nonconvergence():
    """An FD refusal at the kink must be a named reason, not a silent flag."""
    frame, y, _, _ = _simulate(1200, 5, eps=0.7)
    model = SuperLSS(family=TwoPieceLogNormalLSS(), predictors=_predictors()).fit_reml(
        frame, y, method="efs"
    )
    assert model.smoothing_convergence_reason_ is not None
    assert isinstance(model.smoothing_certified_, bool)
    if not model.smoothing_certified_:
        assert str(model.smoothing_convergence_reason_).strip() != ""
    assert np.all(np.isfinite(model.predict(frame)))
    # the kink sits under every observation; a Newton path broken by it would not recover the skew
    fitted_skew = float(model.predict_parameters(frame)["skew"].iloc[0])
    assert abs(fitted_skew - 0.7) <= 0.15, fitted_skew


def test_fisher_curvature_request_is_honoured():
    frame, y, _, _ = _simulate(800, 3)
    model = SuperLSS(
        family=TwoPieceLogNormalLSS(), predictors=_predictors(), coefficient_curvature="fisher"
    ).fit(frame, y, lambdas={"mean:x#wiggle": 1.0})
    assert model.coefficient_curvature == "fisher"
    assert np.all(np.isfinite(model.predict(frame)))


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


_SN2_SCRIPT = """
suppressMessages(library(gamlss)); suppressMessages(library(jsonlite))
d <- read.csv(file("stdin"))
fit <- gamlss(y ~ x, sigma.formula = ~ 1, nu.formula = ~ 1, family = SN2, data = d,
              control = gamlss.control(c.crit = 1e-9, n.cyc = 400, trace = FALSE))
cat(toJSON(list(mu = as.numeric(coef(fit, what = "mu")),
                sigma = as.numeric(coef(fit, what = "sigma")),
                nu = as.numeric(coef(fit, what = "nu")),
                loglik = as.numeric(logLik(fit))), digits = 16))
"""


def test_real_line_family_matches_gamlss_sn2_at_a_parametric_specification():
    require_r_harness()
    if not _gamlss_available():
        pytest.skip("requires R with gamlss and jsonlite")
    rng = np.random.default_rng(88)
    n = 800
    x = rng.uniform(-1.0, 1.0, n)
    y = tp.two_piece_quantile(rng.uniform(size=n), 0.4 + 0.9 * x, np.full(n, 0.7), np.full(n, 0.35))
    frame = pd.DataFrame({"x": x})
    payload = pd.DataFrame({"x": x, "y": y}).to_csv(index=False)
    completed = subprocess.run(
        ["Rscript", "-e", _SN2_SCRIPT],
        input=payload,
        cwd=ROOT,
        env=r_environment(),
        check=True,
        capture_output=True,
        text=True,
    )
    reference = json.loads(completed.stdout)
    model = SuperLSS(
        family=TwoPieceNormalLSS(),
        predictors=(
            Predictor("location", {"x": Numeric()}),
            Predictor("scale", {}),
            Predictor("skew", {}),
        ),
    ).fit(frame, y)
    theta = model.predict_parameters(frame)
    ours_slope, ours_intercept = (float(v) for v in np.polyfit(x, theta["location"].to_numpy(), 1))
    # A 1e-6 relative bound on every fitted parameter is not attainable against
    # gamlss here. Measured on this 800-row specification:
    # intercept 2.161e-06, nu 1.632e-06, slope 1.807e-07, sigma_SN2 4.963e-07.
    # The two-piece log-likelihood is C1 but not C2, so gamlss's RS/CG outer
    # cycle and our Newton fit stop at different points of the same flat
    # optimum; 1e-5 is the bound the larger two force.  The log-likelihood,
    # which is what the two optimisers actually agree on, stays within 1e-8
    # with four orders of headroom (measured 1.792e-12).
    assert abs(ours_intercept - reference["mu"][0]) <= 1e-5 * (1 + abs(reference["mu"][0]))
    assert abs(ours_slope - reference["mu"][1]) <= 1e-5 * (1 + abs(reference["mu"][1]))
    # SN2 links: sigma log, nu log; our mapping is nu^2 = (1+eps)/(1-eps),
    # sigma_SN2 = scale sqrt(1 - eps^2).
    eps = float(theta["skew"].iloc[0])
    scale = float(theta["scale"].iloc[0])
    nu_ours = math.sqrt((1.0 + eps) / (1.0 - eps))
    sigma_ours = scale * math.sqrt(1.0 - eps * eps)
    assert abs(nu_ours - math.exp(reference["nu"][0])) <= 1e-5 * (1 + nu_ours)
    assert abs(sigma_ours - math.exp(reference["sigma"][0])) <= 1e-5 * (1 + sigma_ours)
    family = TwoPieceNormalLSS()
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
