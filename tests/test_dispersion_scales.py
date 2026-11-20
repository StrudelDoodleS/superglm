"""Which scale each reader of a fitted model takes, on every route that publishes one.

``phi`` is Pearson's estimate at the fitting design's mean, and standard errors read it.
``phi_likelihood`` is the exact likelihood's profile: ML at the fitted mean on ``fit()``,
``fit_path()`` and a REML fit with nothing to smooth, and the REML profile on the REML
routes. Densities (the log-likelihood, AIC, quantile residuals, the Vuong test's per-row
log-likelihoods) and variance components read it (McCullagh & Nelder 1989, Sec. 8.3.6;
mgcv's ``sig2`` and ``reml.scale``). The responses are heavier-tailed than the Gamma, so
the two scales differ by a factor of two or more and a reader on the wrong one fails.

Every reference is computed here: Pearson's statistic from the response, weights and
fitted mean; the Gamma, Gaussian and Tweedie (p = 1.5, a Bessel series) densities; and the
ML or REML scale as scipy's root or maximiser of the exact criterion, never the library's.
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
from superglm.distributions import Gamma, Gaussian, Tweedie
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


def _tweedie_log_density(y: np.ndarray, mu: np.ndarray, row_phi: np.ndarray, p: float):
    """Tweedie 1 < p < 2 as its compound Poisson sum: a Poisson count of Gamma jumps of
    shape (2 - p)/(p - 1), summed in logs over enough counts to cover every row."""
    rate = mu ** (2.0 - p) / (row_phi * (2.0 - p))
    jump = row_phi * (p - 1.0) * mu ** (p - 1.0)
    shape = (2.0 - p) / (p - 1.0)
    out = -rate
    pos = y > 0.0
    yp, rp, jp = y[pos], rate[pos], jump[pos]
    top = int(np.max(5.0 * (rp + yp / (jp * shape)))) + 50
    n = np.arange(1, top + 1, dtype=np.float64)[None, :]
    terms = (
        n * np.log(rp)[:, None]
        - special.gammaln(n + 1.0)
        + stats.gamma.logpdf(yp[:, None], a=n * shape, scale=jp[:, None])
    )
    out[pos] = -rp + special.logsumexp(terms, axis=1)
    return out


def _log_density(
    family: str, y: np.ndarray, mu: np.ndarray, phi: float, w=None, p: float = 1.5
) -> np.ndarray:
    """The prior-weight density of each row, ``f(y; mu, phi / w)``."""
    row_phi = phi / (np.ones_like(y) if w is None else w)
    if family == "gamma":
        return stats.gamma.logpdf(y, a=1.0 / row_phi, scale=mu * row_phi)
    if family == "tweedie":
        return _tweedie_log_density(y, mu, row_phi, p)
    return stats.norm.logpdf(y, loc=mu, scale=np.sqrt(row_phi))


def _penalty(model: SuperGLM) -> tuple[float, int]:
    """``beta' S beta`` at the published mode, and Wood's nullity ``1 + p - rank(S)``.

    The fixtures' designs are identified, so the nullity is the unpenalized dimension.
    """
    p = model._dm.p
    beta = model._solver_pirls_result().beta
    S = np.zeros((p, p))
    structure = np.zeros((p, p))
    for component in model._reml_penalties:
        lam = float(model._reml_lambdas[component.name])
        width = component.group_sl.stop - component.group_sl.start
        block = np.eye(width) if component.penalty_kind == "identity" else component.omega_ssp
        S[component.group_sl, component.group_sl] += lam * block
        if lam > 0.0:
            structure[component.group_sl, component.group_sl] += block
    return float(beta @ S @ beta), 1 + p - int(np.linalg.matrix_rank(structure))


def _profile_reference(
    family: str, y: np.ndarray, mu: np.ndarray, w=None, *, quadratic=0.0, nullity=0.0, p=1.5
) -> float:
    """argmax over phi of ``sum log f(y; mu, phi/w) - b'Sb / (2 phi) + (M_p / 2) log phi``.

    Zero nullity and penalty give the ML at ``mu`` (Wood 2011, Eq. 4). Gaussian is closed
    form and the unit-weight Gamma a score root (k = 1/phi); Tweedie is maximised.
    """
    weights = np.ones_like(y) if w is None else w
    if family == "gaussian":
        deviance = float(np.sum(weights * (y - mu) ** 2))
        return (deviance + quadratic) / (np.count_nonzero(weights > 0.0) - nullity)
    if family == "gamma":
        assert w is None
        half = 0.5 * (2.0 * float(np.sum((y - mu) / mu - np.log(y / mu))) + quadratic)

        def score(log_k: float) -> float:
            k = np.exp(log_k)
            return k * (len(y) * (log_k - special.digamma(k)) - half) - 0.5 * nullity

        return float(np.exp(-optimize.brentq(score, -20.0, 20.0, xtol=1e-15, rtol=1e-15)))

    def criterion(log_phi: float) -> float:
        phi = np.exp(log_phi)
        ll = np.sum(_log_density(family, y, mu, phi, weights, p))
        return -(ll - quadratic / (2.0 * phi) + 0.5 * nullity * log_phi)

    solved = optimize.minimize_scalar(
        criterion, bounds=(-12.0, 12.0), method="bounded", options={"xatol": 1e-12}
    )
    return float(np.exp(solved.x))


# A maximiser locates its optimum to about the root of its criterion's rounding over its
# curvature, near 1e-8 relative here; a wrong estimator (ML for REML, a lost nullity, the
# Pearson scale) is 1e-2 to 1e0 away.
PROFILE_RTOL = {"gaussian": 1e-10, "gamma": 1e-9, "tweedie": 1e-6}


def _likelihood_reference(family: str, route: str, model: SuperGLM) -> float:
    """The exact likelihood's profile at the fitting mean: ML on ``fit()``, ``fit_path()``
    and a REML fit with nothing to smooth, the REML profile elsewhere."""
    y, w = _response(family)
    mu = np.asarray(model._fit_mu, dtype=np.float64)
    p = float(getattr(model._distribution, "p", 1.5))
    if route in ML_ROUTES or route == "estimate_p":
        return _profile_reference(family, y, mu, w, p=p)
    quadratic, nullity = _penalty(model)
    return _profile_reference(family, y, mu, w, quadratic=quadratic, nullity=nullity, p=p)


@pytest.mark.parametrize(("family", "route"), CASES)
def test_phi_is_pearson_at_the_fitting_mean(family, route):
    model = _fitted(family, route)
    expected = _pearson(family, model)
    assert model.result.phi == pytest.approx(expected, rel=RTOL)
    scale_row = next(line for line in str(model.summary()).splitlines() if "Scale (phi)" in line)
    assert "(Pearson)" in scale_row


# The routes whose standard errors this change moved; the rest already read Pearson.
SE_UNCHANGED = {
    ("gamma", "fit"),
    ("gamma", "fit_path"),
    ("gamma", "fit_reml_no_smooth"),
    ("gaussian", "fit"),
    ("gaussian", "fit_reml"),
    ("tweedie", "fit"),
    ("tweedie", "fit_reml"),
}


def _assert_standard_errors_read_pearson(family, route):
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


@pytest.mark.parametrize(("family", "route"), [c for c in CASES if c not in SE_UNCHANGED])
def test_standard_errors_read_pearson(family, route):
    _assert_standard_errors_read_pearson(family, route)


@pytest.mark.pins_behaviour
@pytest.mark.parametrize(("family", "route"), sorted(SE_UNCHANGED))
def test_standard_errors_read_pearson_where_they_already_did(family, route):
    """Pins the routes whose SEs read Pearson before the split (``fit``, ``fit_path``, a
    Gaussian REML fit, a REML fit with nothing to smooth, Tweedie ``fit_reml``): splitting
    the scales must leave them on Pearson."""
    _assert_standard_errors_read_pearson(family, route)


@pytest.mark.parametrize(("family", "route"), CASES)
def test_likelihood_scale_is_the_exact_likelihoods_profile(family, route):
    """ML at the fitted mean, or the REML profile, against scipy's root or maximiser."""
    model = _fitted(family, route)
    reference = _likelihood_reference(family, route, model)
    assert model.result.phi_likelihood == pytest.approx(reference, rel=PROFILE_RTOL[family])
    ml = route in ML_ROUTES or route == "estimate_p"
    assert model.result.phi_likelihood_method == ("ml" if ml else "reml_profile")
    y, w = _response(family)
    fit_mu = np.asarray(model._fit_mu, dtype=np.float64)
    p = float(getattr(model._distribution, "p", 1.5))
    expected = float(np.sum(_log_density(family, y, fit_mu, model.result.phi_likelihood, w, p)))
    assert model.summary()._info["log_likelihood"] == pytest.approx(expected, rel=RTOL)


@pytest.mark.parametrize(("family", "route"), CASES)
def test_densities_read_the_likelihood_scale(family, route):
    """The log-likelihood, AIC, the summary's log-likelihood and the quantile residuals."""
    model = _fitted(family, route)
    X, _, _ = _frame()
    y, w = _response(family)
    assert model.result.phi == pytest.approx(_pearson(family, model), rel=RTOL)
    reference = _likelihood_reference(family, route, model)
    phi_likelihood = model.result.phi_likelihood
    assert phi_likelihood == pytest.approx(reference, rel=PROFILE_RTOL[family])
    if family != "gaussian":
        # The two scales are far apart, so a density on the SE scale cannot pass.
        assert model.result.phi > 1.2 * reference
    mu = np.asarray(model.predict(X), dtype=np.float64)
    metrics = model.metrics(X, y, sample_weight=w)
    p = float(getattr(model._distribution, "p", 1.5))
    expected = float(np.sum(_log_density(family, y, mu, phi_likelihood, w, p)))
    assert metrics.log_likelihood == pytest.approx(expected, rel=RTOL)
    assert metrics.aic == pytest.approx(-2.0 * expected + 2.0 * metrics.effective_df, rel=RTOL)
    if family == "tweedie":
        return  # a Tweedie PIT is randomised on its point mass
    if family == "gamma":
        u = stats.gamma.cdf(y, a=1.0 / phi_likelihood, scale=mu * phi_likelihood)
    else:
        u = stats.norm.cdf(y, loc=mu, scale=np.sqrt(phi_likelihood))
    # The library clips the transform to [1e-10, 1 - 1e-10] before inverting it.
    expected_residuals = stats.norm.ppf(np.clip(u, 1e-10, 1.0 - 1e-10))
    np.testing.assert_allclose(
        metrics.residuals("quantile"), expected_residuals, rtol=1e-9, atol=1e-9
    )


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


def _simulated_median(model, X, y, *, n_sim, seed, w=None) -> np.ndarray:
    figure = model.plot_diagnostics(X, y, sample_weight=w, n_sim=n_sim, seed=seed)
    line = next(ln for ln in figure.axes[0].lines if ln.get_label() == "Simulated median")
    return np.asarray(line.get_ydata(), dtype=np.float64)


def test_qq_envelope_simulates_the_likelihood_scale():
    """The envelope's draws and their PIT both sit at ``phi_likelihood``, so each simulated
    residual is a standard normal draw; drawn at Pearson's scale they would spread wider."""
    model = _fitted("gamma", "fit_reml")
    X, _, _ = _frame()
    y, _ = _response("gamma")
    phi = _likelihood_reference("gamma", "fit_reml", model)
    assert model.result.phi > 1.5 * phi
    mu = np.asarray(model.predict(X), dtype=np.float64)
    n_sim, seed = 5, 3
    sorted_draws = []
    for replicate in np.random.SeedSequence(seed).spawn(3)[2].spawn(n_sim):
        response_stream, _ = replicate.spawn(2)
        rng = np.random.default_rng(response_stream)
        draw = rng.gamma(shape=np.ones(N) / phi, scale=mu * phi)
        u = stats.gamma.cdf(draw, a=1.0 / phi, scale=mu * phi)
        sorted_draws.append(np.sort(stats.norm.ppf(np.clip(u, 1e-10, 1.0 - 1e-10))))
    expected = np.median(sorted_draws, axis=0)
    observed = _simulated_median(model, X, y, n_sim=n_sim, seed=seed)
    np.testing.assert_allclose(observed, expected, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("family", ["gamma", "tweedie"])
def test_variance_components_pair_lambda_with_the_reml_profile(family):
    """tau^2 = phi_REML / lambda, as mgcv's ``gam.vcomp`` pairs sp with ``reml.scale``."""
    model = _fitted(family, "fit_reml_random_effect")
    component = model.random_effects("c")
    se_scale = _pearson(family, model)
    assert model.result.phi == pytest.approx(se_scale, rel=RTOL)
    reference = _likelihood_reference(family, "fit_reml_random_effect", model)
    assert se_scale > 1.2 * reference
    assert component.phi == pytest.approx(reference, rel=PROFILE_RTOL[family])
    assert component.tau_squared == pytest.approx(
        reference / component.lambda_value, rel=PROFILE_RTOL[family]
    )
    assert component.standard_deviation == pytest.approx(np.sqrt(component.tau_squared))

    fitted = _fitted(family, "fit_reml_factor_smooth")
    smooth = fitted.factor_smooth("x:c:sz", grid=5)
    likelihood = fitted.result.phi_likelihood
    assert fitted.result.phi > 1.2 * likelihood
    assert smooth.phi == pytest.approx(likelihood, rel=RTOL)
    for name, variance in smooth.variance_components.items():
        assert variance == pytest.approx(likelihood / smooth.lambdas[name], rel=RTOL)


def test_a_lifted_random_effect_standard_deviation_reads_its_variance_scale():
    """o2h8's reproducer: in a lifted weight unit the standard deviation is still the root
    of tau^2, not of Pearson's phi over lambda."""
    import math
    from dataclasses import replace

    model = pickle.loads(pickle.dumps(_fitted("gamma", "fit_reml_random_effect")))
    solver = model._solver_result
    exponent = 200
    lifted = replace(
        solver,
        weight_exponent=exponent,
        solver_phi=math.ldexp(solver.phi, exponent),
        solver_phi_likelihood=math.ldexp(solver.phi_likelihood, exponent),
    )
    model._solver_result = model._result = lifted
    component = model.random_effects("c")
    assert component.tau_squared == pytest.approx(
        solver.phi_likelihood / component.lambda_value, rel=RTOL
    )
    assert component.standard_deviation == pytest.approx(np.sqrt(component.tau_squared), rel=RTOL)


def test_estimate_p_reml_variance_component_reads_the_reml_profile():
    """``estimate_p(fit_mode="reml")`` keeps the ML ``phi_hat`` for its densities, but its
    lambda was selected by REML, so tau^2 pairs it with the REML profile: the value
    ``fit_reml`` publishes at the same p and lambda (a5m3's review)."""
    X, _, _ = _frame()
    y, w = _response("tweedie")
    features = {"x": Spline(n_knots=6), "c": RandomEffect(), "z": Numeric()}
    model = SuperGLM(family=Tweedie(p=1.5), selection_penalty=0.0, features=features)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        profile = model.estimate_p(
            X, y, sample_weight=w, p_bounds=(1.3, 1.7), ci_alpha=None, fit_mode="reml"
        )
    assert model.result.phi_likelihood == pytest.approx(profile.phi_hat, rel=RTOL)
    component = model.random_effects("c")
    quadratic, nullity = _penalty(model)
    mu = np.asarray(model._fit_mu, dtype=np.float64)
    p_hat = float(model._distribution.p)
    reference = _profile_reference(
        "tweedie", y, mu, w, quadratic=quadratic, nullity=nullity, p=p_hat
    )
    assert reference > 1.005 * profile.phi_hat  # REML's nullity sets the two apart
    assert component.phi == pytest.approx(reference, rel=PROFILE_RTOL["tweedie"])
    assert component.tau_squared == pytest.approx(
        reference / component.lambda_value, rel=PROFILE_RTOL["tweedie"]
    )


def _overview(model: SuperGLM) -> dict:
    from superglm.export.summary import build_summary_export_payload

    return {row.metric: row.value for row in build_summary_export_payload(model).overview}


def _strip_split(model: SuperGLM) -> SuperGLM:
    """``model`` as a pickle saved before the split restores it."""
    saved = pickle.loads(pickle.dumps(model))
    for name in ("_result", "_solver_result"):
        state = dict(getattr(saved, name).__getstate__())
        for field in ("phi_likelihood", "solver_phi_likelihood", "phi_method"):
            state.pop(field, None)
        state.pop("phi_likelihood_method", None)
        old = PIRLSResult.__new__(PIRLSResult)
        old.__setstate__(state)
        saved.__dict__[name] = old
    if saved._reml_result is not None:
        saved._reml_result.pirls_result = saved._solver_result
    saved._summary_cache = None
    saved._fit_metrics_cache = None
    return saved


def test_a_result_saved_before_the_split_keeps_its_one_scale():
    """An old pickle has one ``phi``; it keeps it for both readers and says so."""
    fresh = _fitted("gamma", "fit_reml")
    overview = _overview(fresh)
    assert overview.get("Scale estimator") == "pearson"
    assert overview.get("Likelihood phi estimator") == "reml_profile"
    assert overview.get("Likelihood phi") == pytest.approx(fresh.result.phi_likelihood)

    saved = _strip_split(fresh)
    assert saved.result.phi_likelihood == saved.result.phi
    overview = _overview(saved)
    assert overview.get("Scale estimator") == "legacy"
    assert "Likelihood phi" not in overview
    text = str(saved.summary())
    assert "Likelihood phi" not in text
    assert "Likelihood phi" not in saved.summary()._repr_html_()
    X, _, _ = _frame()
    y, _ = _response("gamma")
    mu = np.asarray(saved.predict(X), dtype=np.float64)
    expected = float(np.sum(_log_density("gamma", y, mu, saved.result.phi)))
    assert saved.metrics(X, y).log_likelihood == pytest.approx(expected, rel=RTOL)


def test_a_summary_cached_before_the_split_exports_a_legacy_estimator():
    saved = _strip_split(_fitted("gamma", "fit"))
    saved.summary()
    for cached in saved._summary_cache.values():
        for key in ("phi_method", "phi_likelihood", "phi_likelihood_method"):
            cached._info.pop(key, None)
    assert _overview(saved).get("Scale estimator") == "legacy"


def test_a_known_scale_path_summary_shows_the_scale_its_errors_read():
    """A Poisson ``fit_path`` result holds Pearson's statistic in ``phi``; its standard
    errors read 1, so the summary shows 1 and names no estimator (a5m3's review)."""
    X, eta, _ = _frame()
    y = np.random.default_rng(3).poisson(np.exp(eta)).astype(np.float64)
    model = SuperGLM(family="poisson", features=_features("fit"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_path(X, y, n_lambda=4)
    assert model.result.phi != pytest.approx(1.0, rel=1e-3)
    assert model.result.phi_method == "known"
    summary = model.summary()
    row = next(r for r in summary._coef_rows if r.name == "z")
    metrics = model.metrics(X, y)
    assert row.se == pytest.approx(float(metrics.coefficient_se_raw["z"][0]), rel=1e-3)
    scale_row = next(line for line in str(summary).splitlines() if "Scale (phi)" in line)
    scale_cell = scale_row.split("Pearson chi2")[0]
    assert "1.000" in scale_cell and "(Pearson)" not in scale_cell
    overview = _overview(model)
    assert overview.get("Scale (phi)") == 1.0
    assert overview.get("Scale estimator") == "known"


# --- weights below 2**-512: the fit runs in a lifted weight unit -------------------------
# Every reproducer below fails where the likelihood scale fell back to Pearson in that unit.


def _gaussian_numeric(y, w=None, semantics="prior", x=None):
    x = np.array([-1.0, -1.0, 1.0, 1.0]) if x is None else x
    frame = pd.DataFrame({"x": x})
    model = SuperGLM(
        family="gaussian",
        features={"x": Numeric()},
        selection_penalty=0.0,
        weight_semantics=semantics,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(frame, y, sample_weight=w)
    return model, frame


@pytest.mark.parametrize("semantics", ["prior", "frequency"])
def test_a_lifted_gaussian_fit_reads_its_ml_scale(semantics):
    """x4z6's reproducer: weights 2**-520 publish the unit-weight fit's log-likelihood."""
    y = np.array([1.0, 3.0, 4.0, 6.0])
    w = np.full(4, 2.0**-520)
    model, frame = _gaussian_numeric(y, w, semantics)
    assert model.result.weight_exponent > 0
    assert model.result.phi_likelihood_method == "ml"
    mu = model.predict(frame)
    if semantics == "prior":
        ml = float(np.sum(w * (y - mu) ** 2)) / 4.0
        expected = float(np.sum(stats.norm.logpdf(y, mu, np.sqrt(ml / w))))
        unit, _ = _gaussian_numeric(y, None, semantics)
        assert expected == pytest.approx(unit.metrics(frame, y).log_likelihood, rel=RTOL)
    else:
        ml = float(np.sum(w * (y - mu) ** 2) / np.sum(w))
        expected = float(np.sum(w * stats.norm.logpdf(y, mu, np.sqrt(ml))))
    assert model.result.phi_likelihood == pytest.approx(ml, rel=RTOL)
    assert model.metrics(frame, y, sample_weight=w).log_likelihood == pytest.approx(
        expected, rel=RTOL
    )


def test_a_lifted_three_row_gaussian_fit_reads_its_ml_scale():
    """e9c1's reproducer: weights 2**-600, row variance 0.889 at ML against Pearson's 2.67."""
    y = np.array([0.0, 2.0, 0.0])
    w = np.full(3, 2.0**-600)
    model, frame = _gaussian_numeric(y, w, x=np.array([-1.0, 0.0, 1.0]))
    mu = model.predict(frame)
    variance = float(np.mean((y - mu) ** 2))
    assert model.result.phi_likelihood / w[0] == pytest.approx(variance, rel=RTOL)
    expected = float(np.sum(stats.norm.logpdf(y, mu, np.sqrt(variance))))
    assert model.metrics(frame, y, sample_weight=w).log_likelihood == pytest.approx(
        expected, rel=RTOL
    )


def test_a_tiny_normal_gaussian_scale_is_not_floored():
    """x4z6's reproducer: an ML scale of 6.4e-301, a normal float, is the density's scale."""
    y = np.array([1.0, 3.0, 4.0, 6.0]) * 8e-151
    model, frame = _gaussian_numeric(y)
    mu = model.predict(frame)
    ml = float(np.mean((y - mu) ** 2))
    assert ml < 1e-300
    assert model.result.phi_likelihood == pytest.approx(ml, rel=1e-9)
    expected = float(np.sum(stats.norm.logpdf(y, mu, np.sqrt(ml))))
    assert model.metrics(frame, y).log_likelihood == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize(
    ("x", "y", "n_sim", "seed"),
    [
        # x4z6's reproducer, then e9c1's.
        (np.linspace(-1, 1, 80), lambda x: 10 + 2 * x + 0.001 * np.sin(50 * x), 32, 42),
        (np.linspace(-1, 1, 30), lambda x: x + 1e-3 * np.cos(10 * x), 3, 21),
    ],
    ids=["x4z6", "e9c1"],
)
def test_a_lifted_qq_envelope_matches_the_unit_weight_one(x, y, n_sim, seed):
    """Subnormal prior weights (1e-320) describe the same model as unit weights."""
    response = y(x)
    curves = []
    for unit in (1.0, 1e-320):
        w = np.full(len(x), unit)
        model, frame = _gaussian_numeric(response, w, x=x)
        curves.append(_simulated_median(model, frame, response, n_sim=n_sim, seed=seed, w=w))
    assert np.ptp(curves[1]) > 1.0
    np.testing.assert_allclose(curves[1], curves[0], rtol=1e-6, atol=1e-6)


def _lifted_estimate_p_agrees(model, X, y, w, profile):
    """Every density reader and the summary read ``phi_hat`` at ``p_hat``."""
    p_hat = float(model._distribution.p)
    assert model.result.weight_exponent > 0
    assert model.result.phi_likelihood == pytest.approx(profile.phi_hat, rel=RTOL)
    mu = np.asarray(model.predict(X), dtype=np.float64)
    expected = float(np.sum(_log_density("tweedie", y, mu, profile.phi_hat, w, p_hat)))
    assert model.metrics(X, y, sample_weight=w).log_likelihood == pytest.approx(expected, rel=1e-9)
    fit_mu = np.asarray(model._fit_mu, dtype=np.float64)
    summary = float(np.sum(_log_density("tweedie", y, fit_mu, profile.phi_hat, w, p_hat)))
    assert model.summary()._info["log_likelihood"] == pytest.approx(summary, rel=1e-9)
    return expected


def test_a_lifted_estimate_p_reads_phi_hat_everywhere():
    """a5m3's reproducer: exposure weights times 2**-600."""
    X, _, _ = _frame()
    y, w = _response("tweedie")
    w = np.ldexp(w, -600)
    model = SuperGLM(family=Tweedie(p=1.5), selection_penalty=0.0, features=_features("fit"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        profile = model.estimate_p(X, y, sample_weight=w, p_bounds=(1.3, 1.7), ci_alpha=None)
    _lifted_estimate_p_agrees(model, X, y, w, profile)


def test_a_lifted_estimate_p_fit_mode_fit_agrees_with_its_summary():
    """o2h8's reproducer: ``summary()`` and ``metrics()`` agree, at -918.08."""
    rng = np.random.default_rng(42)
    n = 400
    X = pd.DataFrame({"x": rng.uniform(0.0, 1.0, n), "z": rng.normal(size=n)})
    eta = 0.4 + np.sin(3.0 * X["x"].to_numpy()) + 0.25 * X["z"].to_numpy()
    exposure = rng.uniform(0.3, 3.0, n)
    counts = rng.poisson(0.4 * exposure * np.exp(eta))
    y = np.array([rng.lognormal(0.0, 1.2, k).sum() for k in counts]) / exposure
    w = np.full(n, 2.0**-600)
    model = SuperGLM(
        family=Tweedie(p=1.5),
        selection_penalty=0.0,
        features={"x": Spline(n_knots=5), "z": Numeric()},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        profile = model.estimate_p(
            X, y, sample_weight=w, fit_mode="fit", p_bounds=(1.3, 1.7), ci_alpha=None
        )
    assert _lifted_estimate_p_agrees(model, X, y, w, profile) == pytest.approx(-918.08, abs=0.01)


# --- shape repair with an aliased predictor --------------------------------------------


def _repaired(family, X, y, features):
    model = SuperGLM(family=family, features=features, selection_penalty=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(X, y)
        model.apply_shape_postfit(X)
    return model


@pytest.mark.parametrize("family", ["gaussian", "tweedie"])
def test_shape_repair_profile_counts_identified_coefficients(family):
    """A duplicated numeric column leaves the repaired REML scale where the identified
    model's is (e9c1's and x4z6's reproducers): the profile's nullity is the certified
    rank's, not the coefficient width's."""
    if family == "gaussian":
        x = np.linspace(0.0, 1.0, 50)
        z = np.sin(3 * x)
        y = 2 + np.cos(6 * x) + 0.1 * np.sin(40 * x)
        distribution = Gaussian()
    else:
        x = np.linspace(0.0, 1.0, 150)
        z = np.sin(91 * x)
        y = np.exp(1 - 2 * x + 0.7 * z + 0.25 * np.cos(12 * x))
        y = y * np.random.default_rng(175).lognormal(-0.125, 0.5, len(x))
        distribution = Tweedie(p=1.5)
    X = pd.DataFrame({"x": x, "z": z, "same_z": z})
    features = {"x": PSpline(n_knots=5, constraint=Constraint.postfit.increasing), "z": Numeric()}
    identified = _repaired(distribution, X, y, features)
    aliased = _repaired(distribution, X, y, {**features, "same_z": Numeric()})
    mu = np.asarray(identified._fit_mu, dtype=np.float64)
    quadratic, nullity = _penalty(identified)
    reference = _profile_reference(family, y, mu, quadratic=quadratic, nullity=nullity)
    for model in (identified, aliased):
        assert model.result.phi_likelihood == pytest.approx(reference, rel=PROFILE_RTOL[family])
        expected = float(
            np.sum(_log_density(family, y, np.asarray(model._fit_mu), model.result.phi_likelihood))
        )
        assert model.summary()._info["log_likelihood"] == pytest.approx(expected, rel=1e-9)
