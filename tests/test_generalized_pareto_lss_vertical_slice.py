"""Public fits, recovery and cross-checks for ``GeneralizedParetoLSS``."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from superglm import Spline, SuperLSS
from superglm.distributional import Predictor
from superglm.distributional.families.generalized_pareto import GeneralizedParetoLSS
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.kernels import generalized_pareto as gp
from superglm.distributional.weights import WeightContract, resolve_likelihood_weights
from superglm.features import Numeric
from tests._generalized_pareto_lss_oracles import fit_scale_regression, profile_shape

_TRUE_SHAPE = 0.3


def _simulate(n, seed, *, shape=_TRUE_SHAPE):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    log_scale = 0.4 + 0.7 * np.sin(np.pi * x)
    scale = np.exp(log_scale)
    excess = scale * np.expm1(-shape * np.log(rng.random(n))) / shape
    return pd.DataFrame({"x": x}), excess, log_scale


def _predictors(k=6):
    return (Predictor("scale", {"x": Spline(kind="cr", k=k)}), Predictor("shape", {}))


def test_the_scale_surface_and_the_shape_are_recovered_under_reml():
    frame, y, log_scale = _simulate(12_000, 20260902)
    model = SuperLSS(family=GeneralizedParetoLSS(), predictors=_predictors()).fit_reml(
        frame, y, method="efs"
    )
    fitted = model.predict_parameters(frame)
    residual = np.log(fitted["scale"].to_numpy()) - log_scale
    assert math.sqrt(float(np.mean((residual - residual.mean()) ** 2))) <= 0.12
    assert (
        abs(float(residual.mean())) <= 0.10
    )  # the level, which the shape is partly confounded with
    assert abs(float(fitted["shape"].iloc[0]) - _TRUE_SHAPE) <= 0.06
    assert np.allclose(
        model.predict(frame),
        fitted["scale"].to_numpy() / (1.0 - fitted["shape"].to_numpy()),
        rtol=0,
        atol=1e-12,
    )


def test_the_predicted_quantile_round_trips_and_covers_the_sample():
    frame, y, _ = _simulate(12_000, 5)
    # _predictors() carries a penalised spline, so the fixed-lambda path needs the
    # lambda named; how the smoothing is chosen is not what this test is about.
    model = SuperLSS(family=GeneralizedParetoLSS(), predictors=_predictors()).fit(
        frame, y, lambdas={"scale:x#wiggle": 1.0}
    )
    p90 = model.predict_quantile(frame, 0.9)
    assert p90.shape == (len(frame),) and not p90.flags.writeable
    assert np.allclose(model.predict_cdf(frame, p90), 0.9, rtol=0, atol=1e-12)
    assert float(np.mean(y <= p90)) == pytest.approx(0.9, abs=0.02)
    per_row = model.predict_quantile(frame, np.linspace(0.05, 0.95, len(frame)))
    assert per_row.shape == (len(frame),)


def test_the_unpenalised_fit_matches_an_independent_scipy_maximum_likelihood_fit():
    frame, y, _ = _simulate(4_000, 11)
    x = frame["x"].to_numpy()
    model = SuperLSS(
        family=GeneralizedParetoLSS(),
        predictors=(Predictor("scale", {"x": Numeric()}), Predictor("shape", {})),
    ).fit(frame, y)
    fitted = model.predict_parameters(frame)
    design = np.column_stack((np.ones(len(x)), x))
    coefficients, shape, log_likelihood = fit_scale_regression(y, design)
    ours = np.polyfit(x, np.log(fitted["scale"].to_numpy()), 1)
    assert abs(float(ours[1]) - float(coefficients[0])) <= 1e-4 * (1 + abs(float(coefficients[0])))
    assert abs(float(ours[0]) - float(coefficients[1])) <= 1e-4 * (1 + abs(float(coefficients[1])))
    assert abs(float(fitted["shape"].iloc[0]) - shape) <= 1e-4
    family = GeneralizedParetoLSS()
    weights = resolve_likelihood_weights(
        np.ones(len(y)), n_observations=len(y), contract=WeightContract(semantics="frequency")
    )
    plan = family.bind_likelihood(y, weights, COMPLETE_OBSERVATION)
    ours_log_likelihood = float(
        np.sum(
            family.evaluate_natural(
                y, fitted.to_numpy(), plan, derivative_order=0
            ).reported_log_likelihood
        )
    )
    assert abs(ours_log_likelihood - log_likelihood) <= 1e-8 * (1 + abs(log_likelihood))
    # the profile at the truth is below the maximum, and the maximum is at our shape
    grid = np.array([0.15, shape, 0.55])
    profile = profile_shape(y, design, grid)
    assert profile[1] >= profile[0] and profile[1] >= profile[2]


def test_fisher_curvature_request_is_honoured():
    frame, y, _ = _simulate(2_000, 3)
    model = SuperLSS(
        family=GeneralizedParetoLSS(), predictors=_predictors(), coefficient_curvature="fisher"
    ).fit(frame, y, lambdas={"scale:x#wiggle": 1.0})
    assert model.coefficient_curvature == "fisher"
    assert np.all(np.isfinite(model.predict(frame)))


def test_narrow_walls_keep_an_interior_shape_inside_them():
    frame, y, _ = _simulate(6_000, 8, shape=0.3)
    model = SuperLSS(
        family=GeneralizedParetoLSS(shape_lower=0.05, shape_upper=0.5),
        predictors=(Predictor("scale", {"x": Numeric()}), Predictor("shape", {})),
    ).fit(frame, y)
    shape = float(model.predict_parameters(frame)["shape"].iloc[0])
    assert 0.05 < shape < 0.5
    assert abs(shape - 0.3) <= 0.08


def test_data_beyond_the_upper_wall_presses_against_it_or_stops():
    """The wall is a hard constraint, so the honest outcome is a pressed fit or a named failure."""
    frame, y, _ = _simulate(3_000, 8, shape=0.7)
    try:
        model = SuperLSS(
            family=GeneralizedParetoLSS(shape_lower=0.05, shape_upper=0.35),
            predictors=(Predictor("scale", {"x": Numeric()}), Predictor("shape", {})),
        ).fit(frame, y)
    except Exception as failure:  # noqa: BLE001 - the named failure is the assertion
        message = str(failure).lower()
        assert any(
            word in message for word in ("wall", "shape", "converge", "curvature", "null model")
        ), message
        return
    shape = float(model.predict_parameters(frame)["shape"].iloc[0])
    assert 0.05 < shape < 0.35
    assert shape > 0.3, "data generated above the upper wall must press against it"


def test_the_density_agrees_with_scipy_genpareto_at_the_fitted_parameters():
    frame, y, _ = _simulate(1_000, 13)
    model = SuperLSS(
        family=GeneralizedParetoLSS(),
        predictors=(Predictor("scale", {"x": Numeric()}), Predictor("shape", {})),
    ).fit(frame, y)
    theta = model.predict_parameters(frame).to_numpy()
    family = GeneralizedParetoLSS()
    weights = resolve_likelihood_weights(
        np.ones(len(y)), n_observations=len(y), contract=WeightContract(semantics="frequency")
    )
    plan = family.bind_likelihood(y, weights, COMPLETE_OBSERVATION)
    ours = family.evaluate_natural(y, theta, plan, derivative_order=0).reported_log_likelihood
    reference = stats.genpareto.logpdf(y, c=theta[:, 1], scale=theta[:, 0])
    assert np.allclose(ours, reference, rtol=0, atol=1e-12)


def test_the_threshold_splice_recipe_composes_a_tail_probability():
    """The documented recipe: body below u, exceedance rate, GPD on the excesses above."""
    rng = np.random.default_rng(19)
    n = 8_000
    frame = pd.DataFrame({"x": rng.uniform(-1.0, 1.0, n)})
    # a genuinely Pareto-tailed book: the generalized Pareto is threshold-stable, so the
    # excesses over any threshold keep the same shape and the tail fit is well posed
    scale = np.exp(0.2 + 0.3 * frame["x"].to_numpy())
    losses = scale * np.expm1(-0.3 * np.log(rng.random(n))) / 0.3
    threshold = float(np.quantile(losses, 0.9))
    above = losses > threshold
    excess = losses[above] - threshold
    tail = SuperLSS(
        family=GeneralizedParetoLSS(),
        predictors=(Predictor("scale", {"x": Numeric()}), Predictor("shape", {})),
    ).fit(frame.loc[above], excess)
    exceedance_rate = float(np.mean(above))
    survival = exceedance_rate * (1.0 - tail.predict_cdf(frame.loc[above], excess))
    assert np.all(survival >= 0.0) and np.all(survival <= exceedance_rate)
    assert float(np.mean(survival)) == pytest.approx(exceedance_rate / 2.0, abs=0.02)


def test_the_kernel_at_zero_shape_is_the_exponential_and_the_family_cannot_reach_it():
    y = np.array([0.4, 1.3, 2.8, 7.5])
    rows = gp.scale_rows(y, np.full(4, 2.0), np.zeros(4), np.ones(4), derivative_order=0)
    # measured: the difference is exactly zero, so 1e-15 is a version-robustness margin only
    assert np.allclose(
        rows.optimizing_log_likelihood, stats.expon(scale=2.0).logpdf(y), rtol=0, atol=1e-15
    )
    with pytest.raises(ValueError):
        GeneralizedParetoLSS().cdf(y, np.tile([2.0, 0.0], (4, 1)))


import json  # noqa: E402
import subprocess  # noqa: E402
from functools import cache  # noqa: E402

from tests._r_harness import ROOT, r_environment, require_r_harness  # noqa: E402

_GP_PROBE = (
    "suppressMessages(library(gamlss)); suppressMessages(library(jsonlite)); "
    "cat(if (exists('GP')) 'ok' else 'missing')"
)


@cache
def _gamlss_gp_available() -> bool:
    completed = subprocess.run(
        ["Rscript", "-e", _GP_PROBE],
        cwd=ROOT,
        env=r_environment(),
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0 and completed.stdout.strip().endswith("ok")


_GP_SCRIPT = """
suppressMessages(library(gamlss)); suppressMessages(library(jsonlite))
d <- read.csv(file("stdin"))
fit <- gamlss(y ~ x, sigma.formula = ~ 1, family = GP, data = d,
              control = gamlss.control(c.crit = 1e-9, n.cyc = 400, trace = FALSE))
cat(toJSON(list(mu = as.numeric(coef(fit, what = "mu")),
                sigma = as.numeric(coef(fit, what = "sigma")),
                loglik = as.numeric(logLik(fit))), digits = 16))
"""


def test_the_fit_matches_gamlss_gp_at_a_parametric_specification():
    """gamlss GP is the Lomax form of the GPD: mu = psi/xi, sigma = 1/xi, both on log links.

    The mapping is pinned here, not assumed.  The log-likelihood check is
    mapping-free: if it agrees and the parameter checks do not, the mapping
    constant is what changed and must be re-derived from the gamlss.dist manual,
    never absorbed into a looser tolerance.
    """
    require_r_harness()
    if not _gamlss_gp_available():
        pytest.skip("requires R with gamlss (providing GP) and jsonlite")
    frame, y, _ = _simulate(1_500, 12, shape=0.4)
    payload = pd.DataFrame({"x": frame["x"], "y": y}).to_csv(index=False)
    completed = subprocess.run(
        ["Rscript", "-e", _GP_SCRIPT],
        input=payload,
        cwd=ROOT,
        env=r_environment(),
        check=True,
        capture_output=True,
        text=True,
    )
    reference = json.loads(completed.stdout)
    # jsonlite boxes every scalar in a length-one array unless auto_unbox is set
    reference_log_likelihood = float(np.ravel(reference["loglik"])[0])
    model = SuperLSS(
        family=GeneralizedParetoLSS(),
        predictors=(Predictor("scale", {"x": Numeric()}), Predictor("shape", {})),
    ).fit(frame, y)
    theta = model.predict_parameters(frame)
    family = GeneralizedParetoLSS()
    weights = resolve_likelihood_weights(
        np.ones(len(y)), n_observations=len(y), contract=WeightContract(semantics="frequency")
    )
    plan = family.bind_likelihood(y, weights, COMPLETE_OBSERVATION)
    ours_log_likelihood = float(
        np.sum(
            family.evaluate_natural(
                y, theta.to_numpy(), plan, derivative_order=0
            ).reported_log_likelihood
        )
    )
    assert abs(ours_log_likelihood - reference_log_likelihood) <= 1e-8 * (
        1 + abs(reference_log_likelihood)
    )

    shape = float(theta["shape"].iloc[0])
    log_scale = np.log(theta["scale"].to_numpy())
    slope, intercept = np.polyfit(frame["x"].to_numpy(), log_scale, 1)
    # log mu = log psi - log xi with a constant xi, so the slope is shared and the intercept shifts
    assert abs(float(slope) - reference["mu"][1]) <= 1e-5 * (1 + abs(reference["mu"][1]))
    assert abs(float(intercept) - (reference["mu"][0] + math.log(shape))) <= 1e-5 * (
        1 + abs(intercept)
    )
    assert abs(shape - 1.0 / math.exp(reference["sigma"][0])) <= 1e-5 * (1 + shape)
