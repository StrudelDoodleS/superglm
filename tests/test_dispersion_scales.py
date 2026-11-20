"""Which scale each reader of a fitted model takes, on every route that publishes one.

``phi`` is Pearson's estimate at the fitting design's mean, and standard errors read it.
``phi_likelihood`` is the exact likelihood's profile: ML at the fitted mean on ``fit()``,
``fit_path()`` and a REML fit with nothing to smooth, and the REML profile on the REML
routes. Densities (the log-likelihood, AIC, quantile residuals, the Vuong test's per-row
log-likelihoods) and variance components read it (McCullagh & Nelder 1989, Sec. 8.3.6;
mgcv's ``sig2`` and ``reml.scale``). The responses are heavier-tailed than the Gamma, so
the two scales differ by a factor of two or more and a reader on the wrong one fails.

Every reference is computed here: Pearson's statistic from the response, weights and
fitted mean; Gamma and Gaussian densities and the Gamma ML from scipy.
"""

from __future__ import annotations

import pickle
import warnings
from functools import cache

import numpy as np
import pandas as pd
import pytest
from scipy import optimize, special, stats

from superglm import (
    Categorical,
    Constraint,
    FactorSmooth,
    LambdaPolicy,
    Numeric,
    PSpline,
    RandomEffect,
    Spline,
    SuperGLM,
    vuong_test,
)
from superglm.distributions import Gamma, Gaussian, Tweedie, weighted_log_likelihood
from superglm.solvers.pirls import PIRLSResult

N = 600
# Identities between float64 sums of N rows: far above their rounding, far below the
# factor of two separating the scales.
RTOL = 1e-10
# The library's profile root against scipy's: both solve to near machine precision.
ML_RTOL = 1e-8


@cache
def _frame(seed: int = 7) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "x": rng.uniform(0.0, 1.0, N),
            "c": rng.choice(list("abcdef"), N),
            "z": rng.normal(size=N),
        }
    )
    level = {k: v for k, v in zip("abcdef", (0.0, 0.2, -0.1, 0.3, -0.25, 0.1), strict=True)}
    eta = 0.4 + np.sin(3.0 * X["x"].to_numpy()) + 0.25 * X["z"].to_numpy()
    eta = eta + X["c"].map(level).to_numpy()
    exposure = rng.uniform(0.2, 1.5, N)
    return X, eta, exposure


def _response(family: str) -> tuple[np.ndarray, np.ndarray | None]:
    X, eta, exposure = _frame()
    rng = np.random.default_rng(11)
    if family == "gamma":
        # Mean-one lognormal noise, CV^2 = e^1.5 - 1 = 3.5: Pearson about 3.5, the
        # Gamma likelihood's scale far lower.
        return np.exp(eta) * rng.lognormal(-0.75, np.sqrt(1.5), N), None
    if family == "gaussian":
        return eta + 0.6 * rng.standard_t(3, N), None
    counts = rng.poisson(0.4 * exposure * np.exp(eta))
    totals = np.array([rng.lognormal(0.0, 1.2, k).sum() for k in counts])
    return totals / exposure, exposure


def _distribution(family: str):
    return {"gamma": Gamma(), "gaussian": Gaussian(), "tweedie": Tweedie(p=1.5)}[family]


def _variance(family: str, mu: np.ndarray) -> np.ndarray:
    return mu**2 if family == "gamma" else np.ones_like(mu)


def _features(route: str) -> dict:
    if route == "fit_reml_no_smooth":
        return {"c": Categorical(), "z": Numeric()}
    if route == "fit_reml_fixed_monotone":
        shape = PSpline(
            n_knots=5, constraint=Constraint.fit.increasing, lambda_policy=LambdaPolicy.fixed(1.0)
        )
        return {"x": shape, "c": Categorical(), "z": Numeric()}
    if route == "fit_reml_qp_monotone":
        return {
            "x": PSpline(n_knots=5, constraint=Constraint.fit.increasing),
            "c": Categorical(),
            "z": Numeric(),
        }
    if route == "fit_reml_shape_repair":
        shape = PSpline(n_knots=6, constraint=Constraint.postfit.increasing)
        return {"x": shape, "c": Categorical(), "z": Numeric()}
    if route == "fit_reml_factor_smooth":
        return {"x": Spline(n_knots=6), "z": Numeric()}  # the smooth reads its group itself
    if route.endswith("_random_effect"):
        return {"x": Spline(n_knots=6), "c": RandomEffect(), "z": Numeric()}
    return {"x": Spline(n_knots=6), "c": Categorical(), "z": Numeric()}


# Each route that publishes the two scales, and the families it is checked on.
ROUTES = {
    "gamma": (
        "fit",
        "fit_path",
        "fit_reml",
        "fit_reml_binned",
        "fit_reml_no_smooth",
        "fit_reml_fixed_monotone",
        "fit_reml_qp_monotone",
        "fit_reml_shape_repair",
    ),
    "gaussian": ("fit", "fit_reml"),
    "tweedie": ("fit", "fit_reml", "estimate_p"),
}
CASES = [(family, route) for family, routes in ROUTES.items() for route in routes]
ML_ROUTES = ("fit", "fit_path", "fit_reml_no_smooth")


@cache
def _fitted(family: str, route: str) -> SuperGLM:
    X, _, _ = _frame()
    y, w = _response(family)
    kwargs = {"family": _distribution(family), "features": _features(route)}
    if route != "fit_path":
        kwargs["selection_penalty"] = 0.0
    if route == "fit_reml_binned":
        kwargs["discrete"] = True
    if route == "fit_reml_factor_smooth":
        kwargs["interactions"] = [FactorSmooth("x", group="c", basis="sz", k=5)]
    model = SuperGLM(**kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if route == "fit":
            model.fit(X, y, sample_weight=w)
        elif route == "fit_path":
            model.fit_path(X, y, sample_weight=w, n_lambda=4)
        elif route == "estimate_p":
            model.estimate_p(X, y, sample_weight=w, p_bounds=(1.3, 1.7), ci_alpha=None)
        else:
            model.fit_reml(X, y, sample_weight=w)
            if route == "fit_reml_shape_repair":
                model.apply_shape_postfit(X, sample_weight=w, n_grid=80)
    return model


def _weights(family: str) -> np.ndarray:
    _, w = _response(family)
    return np.ones(N) if w is None else w


def _pearson(family: str, model: SuperGLM) -> float:
    """Pearson's statistic over its residual degrees of freedom, at the fitting mean."""
    y, _ = _response(family)
    w = _weights(family)
    mu = np.asarray(model._fit_mu, dtype=np.float64)
    if family == "tweedie":
        variance = mu ** float(model._distribution.p)
    else:
        variance = _variance(family, mu)
    chi2 = float(np.sum(w * (y - mu) ** 2 / variance))
    return chi2 / (np.count_nonzero(w > 0.0) - float(model.result.effective_df))


def _gamma_ml(y: np.ndarray, mu: np.ndarray) -> float:
    """The Gamma ML scale at a fixed mean: n (log k - digamma k) = D / 2, k = 1/phi."""
    half_deviance = float(np.sum((y - mu) / mu - np.log(y / mu)))

    def score(log_k: float) -> float:
        k = np.exp(log_k)
        return len(y) * (np.log(k) - special.digamma(k)) - half_deviance

    return float(np.exp(-optimize.brentq(score, -20.0, 20.0, xtol=1e-15, rtol=1e-15)))


def _ml_scale(family: str, model: SuperGLM) -> float:
    y, _ = _response(family)
    mu = np.asarray(model._fit_mu, dtype=np.float64)
    if family == "gamma":
        return _gamma_ml(y, mu)
    return float(np.mean((y - mu) ** 2))


def _log_density(family: str, y: np.ndarray, mu: np.ndarray, phi: float) -> np.ndarray:
    if family == "gamma":
        return stats.gamma.logpdf(y, a=1.0 / phi, scale=mu * phi)
    return stats.norm.logpdf(y, loc=mu, scale=np.sqrt(phi))


def _likelihood_reference(family: str, route: str, model: SuperGLM) -> float:
    """scipy's ML on the ML routes; the REML profile (checked against mgcv) elsewhere."""
    if route in ML_ROUTES:
        return _ml_scale(family, model)
    return float(model.result.phi_likelihood)


@pytest.mark.parametrize(("family", "route"), CASES)
def test_phi_is_pearson_at_the_fitting_mean(family, route):
    model = _fitted(family, route)
    expected = _pearson(family, model)
    assert model.result.phi == pytest.approx(expected, rel=RTOL)
    scale_row = next(line for line in str(model.summary()).splitlines() if "Scale (phi)" in line)
    assert "(Pearson)" in scale_row


@pytest.mark.parametrize(("family", "route"), CASES)
def test_standard_errors_read_pearson(family, route):
    model = _fitted(family, route)
    X, _, _ = _frame()
    y, w = _response(family)
    metrics = model.metrics(X, y, sample_weight=w)
    root = np.sqrt(_pearson(family, model))
    expected = metrics.coefficient_se_raw["z"] * root
    np.testing.assert_allclose(metrics.coefficient_se["z"], expected, rtol=RTOL)
    row = next(r for r in model.summary()._coef_rows if r.name == "z")
    # A binned or path summary reads the fitting design's covariance, 3e-5 from the
    # evaluation rows' here; the wrong scale would move it by a factor of 1.7.
    summary_rtol = 1e-3 if route in ("fit_reml_binned", "fit_path") else RTOL
    assert row.se == pytest.approx(float(expected[0]), rel=summary_rtol)


@pytest.mark.parametrize(
    ("family", "route"), [c for c in CASES if c[1] in ML_ROUTES and c[0] != "tweedie"]
)
def test_likelihood_scale_is_ml_at_the_fitted_mean(family, route):
    model = _fitted(family, route)
    X, _, _ = _frame()
    y, _ = _response(family)
    ml = _ml_scale(family, model)
    fit_mu = np.asarray(model._fit_mu, dtype=np.float64)
    expected = float(np.sum(_log_density(family, y, fit_mu, ml)))
    assert model.summary()._info["log_likelihood"] == pytest.approx(expected, rel=RTOL)
    assert model.result.phi_likelihood == pytest.approx(ml, rel=ML_RTOL)
    assert model.result.phi_likelihood_method == "ml"


@pytest.mark.parametrize(
    ("family", "route"), [case for case in CASES if case[0] in ("gamma", "gaussian")]
)
def test_densities_read_the_likelihood_scale(family, route):
    """The log-likelihood, AIC, the summary's log-likelihood and the quantile residuals."""
    model = _fitted(family, route)
    X, _, _ = _frame()
    y, _ = _response(family)
    assert model.result.phi == pytest.approx(_pearson(family, model), rel=RTOL)
    phi_likelihood = _likelihood_reference(family, route, model)
    if family == "gamma":
        # The two scales are far apart, so a density on the SE scale cannot pass.
        assert model.result.phi > 1.5 * phi_likelihood
    mu = np.asarray(model.predict(X), dtype=np.float64)
    metrics = model.metrics(X, y)
    expected = float(np.sum(_log_density(family, y, mu, phi_likelihood)))
    assert metrics.log_likelihood == pytest.approx(expected, rel=RTOL)
    assert metrics.aic == pytest.approx(-2.0 * expected + 2.0 * metrics.effective_df, rel=RTOL)
    fit_mu = np.asarray(model._fit_mu, dtype=np.float64)
    summary_ll = float(np.sum(_log_density(family, y, fit_mu, phi_likelihood)))
    assert model.summary()._info["log_likelihood"] == pytest.approx(summary_ll, rel=RTOL)
    if family == "gamma":
        u = stats.gamma.cdf(y, a=1.0 / phi_likelihood, scale=mu * phi_likelihood)
    else:
        u = stats.norm.cdf(y, loc=mu, scale=np.sqrt(phi_likelihood))
    # The library clips the transform to [1e-10, 1 - 1e-10] before inverting it.
    expected_residuals = stats.norm.ppf(np.clip(u, 1e-10, 1.0 - 1e-10))
    np.testing.assert_allclose(
        metrics.residuals("quantile"), expected_residuals, rtol=1e-9, atol=1e-9
    )


@pytest.mark.parametrize("route", ROUTES["tweedie"])
def test_tweedie_densities_read_the_likelihood_scale(route):
    model = _fitted("tweedie", route)
    X, _, _ = _frame()
    y, w = _response("tweedie")
    mu = np.asarray(model.predict(X), dtype=np.float64)
    dist = model._distribution
    observed = model.metrics(X, y, sample_weight=w).log_likelihood
    se_scale = _pearson("tweedie", model)
    assert model.result.phi == pytest.approx(se_scale, rel=RTOL)
    wrong = weighted_log_likelihood(dist, y, mu, w, se_scale, weight_semantics="prior")
    assert abs(observed - wrong) > 1.0
    likelihood = model.result.phi_likelihood
    assert se_scale > 1.2 * likelihood
    expected = weighted_log_likelihood(dist, y, mu, w, likelihood, weight_semantics="prior")
    assert observed == pytest.approx(expected, rel=RTOL)
    assert abs(observed - wrong) > 1e3 * abs(observed - expected) + 1.0


def test_vuong_per_row_log_likelihoods_read_the_likelihood_scale():
    X, _, _ = _frame()
    y, _ = _response("gamma")
    a, b = _fitted("gamma", "fit"), _fitted("gamma", "fit_reml")
    rows = []
    for route, model in (("fit", a), ("fit_reml", b)):
        mu = np.asarray(model.predict(X), dtype=np.float64)
        rows.append(_log_density("gamma", y, mu, _likelihood_reference("gamma", route, model)))
    result = vuong_test(a, b, X, y, correction="none")
    assert result.mean_diff == pytest.approx(float(np.mean(rows[0] - rows[1])), rel=1e-9)


@pytest.mark.parametrize("family", ["gamma", "tweedie"])
def test_variance_components_pair_lambda_with_the_likelihood_scale(family):
    """tau^2 = phi_likelihood / lambda, as mgcv's ``gam.vcomp`` pairs sp with ``reml.scale``."""
    model = _fitted(family, "fit_reml_random_effect")
    component = model.random_effects("c")
    se_scale = _pearson(family, model)
    assert model.result.phi == pytest.approx(se_scale, rel=RTOL)
    assert component.phi < se_scale / 1.2
    phi_likelihood = model.result.phi_likelihood
    assert component.phi == pytest.approx(phi_likelihood, rel=ML_RTOL)
    assert component.tau_squared == pytest.approx(
        phi_likelihood / component.lambda_value, rel=ML_RTOL
    )

    smooth = _fitted(family, "fit_reml_factor_smooth").factor_smooth("x:c:sz", grid=5)
    likelihood = _fitted(family, "fit_reml_factor_smooth").result.phi_likelihood
    assert smooth.phi == pytest.approx(likelihood, rel=ML_RTOL)
    for name, variance in smooth.variance_components.items():
        assert variance == pytest.approx(likelihood / smooth.lambdas[name], rel=ML_RTOL)


def _overview(model: SuperGLM) -> dict:
    from superglm.export.summary import build_summary_export_payload

    return {row.metric: row.value for row in build_summary_export_payload(model).overview}


def test_a_result_saved_before_the_split_keeps_its_one_scale():
    """An old pickle has one ``phi``; it keeps it for both readers and says so."""
    fresh = _fitted("gamma", "fit_reml")
    overview = _overview(fresh)
    assert overview.get("Scale estimator") == "pearson"
    assert overview.get("Likelihood phi estimator") == "reml_profile"
    assert overview.get("Likelihood phi") == pytest.approx(fresh.result.phi_likelihood)

    saved = pickle.loads(pickle.dumps(fresh))
    for name in ("_result", "_solver_result"):
        state = dict(getattr(saved, name).__getstate__())
        for field in ("phi_likelihood", "solver_phi_likelihood", "phi_method"):
            state.pop(field, None)
        state.pop("phi_likelihood_method", None)
        old = PIRLSResult.__new__(PIRLSResult)
        old.__setstate__(state)
        saved.__dict__[name] = old
    saved._summary_cache = None
    saved._fit_metrics_cache = None
    overview = _overview(saved)
    assert overview.get("Scale estimator") == "legacy"
    assert "Likelihood phi" not in overview
    X, _, _ = _frame()
    y, _ = _response("gamma")
    mu = np.asarray(saved.predict(X), dtype=np.float64)
    expected = float(np.sum(_log_density("gamma", y, mu, saved.result.phi)))
    assert saved.metrics(X, y).log_likelihood == pytest.approx(expected, rel=RTOL)
