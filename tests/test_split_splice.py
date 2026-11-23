"""Fixed-threshold severity splicing, with independent distribution and likelihood oracles."""

from __future__ import annotations

import pickle

import mpmath
import numpy as np
import pandas as pd
import pytest
from scipy import integrate, optimize, special, stats

from superglm import GammaLS, GeneralizedParetoLSS, LogNormalLS, SplitSplice, SuperGLM, SuperLSS
from superglm.distributional.weights import LikelihoodWeightError


def _components(family=None, *, frequency=False):
    family = GammaLS() if family is None else family
    tail = GeneralizedParetoLSS()
    semantics = "frequency" if frequency else "prior"
    center = (
        family.location()
        if isinstance(family, LogNormalLS) and family.parametrisation == "location"
        else family.mean()
    )
    return (
        SuperLSS(family, center, family.scale(), weight_semantics=semantics),
        SuperGLM(family="binomial", features={}, weight_semantics=semantics),
        SuperLSS(tail, tail.scale(), tail.shape(), weight_semantics=semantics),
    )


def _frame(n):
    return pd.DataFrame({"x": np.linspace(-1, 1, n)}, index=10 + np.arange(n) // 2)


def _fit(X, y, u, models=None, **kwargs):
    body, exceedance, tail = _components() if models is None else models
    return SplitSplice.fit(X, y, threshold=u, body=body, exceedance=exceedance, tail=tail, **kwargs)


def _sample(n=4000):
    rng = np.random.default_rng(570)
    large = rng.random(n) < 0.25
    y = stats.gamma.ppf(rng.random(n) * stats.gamma.cdf(100, 3, scale=20), 3, scale=20)
    y[large] = 100 + stats.genpareto.rvs(0.25, scale=35, size=large.sum(), random_state=rng)
    return _frame(n), y, 100.0


@pytest.fixture(scope="module")
def fitted():
    return _fit(*_sample())


def _population(monkeypatch, splice, body, tail, q):
    # Keep verified fit assembly, and supply known, row-varying population parameters.
    monkeypatch.setattr(
        splice.body,
        "predict_parameters",
        lambda X, **kw: pd.DataFrame(np.broadcast_to(body, (len(X), 2))),
    )
    monkeypatch.setattr(
        splice.tail,
        "predict_parameters",
        lambda X, **kw: pd.DataFrame(np.broadcast_to(tail, (len(X), 2))),
    )
    monkeypatch.setattr(splice.exceedance, "predict", lambda X, **kw: np.broadcast_to(q, (len(X),)))


def test_exact_splice_and_boundary(monkeypatch, fitted):
    u, q = fitted.threshold, 0.25
    body = stats.gamma(3, scale=20)
    tail = stats.genpareto(0.25, scale=35)
    _population(monkeypatch, fitted, (60, 1 / np.sqrt(3)), (35, 0.25), q)
    y = np.array([-np.inf, 0, 2, 50, np.nextafter(u, 0), u, 101, 400, np.inf])
    X = _frame(len(y))
    expected = np.where(y <= u, (1 - q) * body.cdf(y) / body.cdf(u), 1 - q * tail.sf(y - u))
    values = fitted.predict_cdf(X, y)
    np.testing.assert_allclose(values, expected, rtol=2e-12, atol=1e-15)
    assert np.all(np.diff(values) >= 0)
    assert fitted.predict_survival(X[:1], u)[0] == q
    assert fitted.predict_cdf(X[:1], u)[0] == 1 - q
    assert fitted.predict_quantile(X[:1], 1 - q)[0] == u
    interior = np.array([1e-8, 0.2, 0.55, 0.74999, 0.75, 0.75001, 0.9, 0.99, 1 - 1e-12])
    quantiles = fitted.predict_quantile(X, interior)
    np.testing.assert_allclose(fitted.predict_cdf(X, quantiles), interior, rtol=2e-12)
    np.testing.assert_allclose(
        fitted.predict(X),
        (1 - q) * integrate.quad(lambda v: v * body.pdf(v), 0, u)[0] / body.cdf(u)
        + q * (u + tail.mean()),
        rtol=2e-12,
    )


def test_population_layer_matches_the_570_probe(monkeypatch):
    X, y, _ = _sample()
    shape = 1 / 1.2**2
    body = stats.gamma(shape, scale=5000 / shape)
    u = body.ppf(0.9)
    # The independently projected GPD matches the likelihood under the Gamma
    # population, using dimensionless excesses to condition the optimization.
    money = 5000 / shape
    gamma = stats.gamma(shape)
    nodes, weights = special.roots_laguerre(64)
    density = weights * np.exp(gamma.logpdf(u / money + nodes) + nodes) / 0.1

    def loss(params):
        xi, log_scale = params
        return (
            -density @ stats.genpareto.logpdf(nodes, xi, scale=np.exp(log_scale))
            if xi >= 0
            else np.inf
        )

    projection = optimize.minimize(loss, [0.03, 0.0], method="Nelder-Mead", options={"xatol": 1e-9})
    assert projection.success
    xi, scale = projection.x[0], money * np.exp(projection.x[1])
    splice = _fit(X, y * (u / 100), u)
    _population(monkeypatch, splice, (5000, 1.2), (scale, xi), 0.1)
    # The population projection parameters are calculated independently in the scope probe.
    below = splice.predict_layer(X[:1], 10_000, u - 10_000)[0]
    above = splice.predict_layer(X[:1], u, 20_000 - u)[0]
    total = splice.predict_layer(X[:1], 10_000, 10_000)[0]
    np.testing.assert_allclose([below, above, total], [317.398, 442.523, 759.921], atol=0.002)
    assert total == pytest.approx(below + above, rel=2e-13)


def _partial_moment_layer(moments, a, w, u, q):
    a, w, u, q = map(mpmath.mpf, (a, w, u, q))
    b = a + w
    mass, first_moment = moments(a, b)
    capped_mass, _ = moments(b, u)
    normalizer, _ = moments(0, u)
    return float(q * w + (1 - q) * (first_moment - a * mass + w * capped_mass) / normalizer)


def test_gamma_layers_against_partial_moments(monkeypatch, fitted):
    cases = [
        (1.0001452528582397, 0.009445593735401357, 0.999, 50.0, 1e-8),
        (1.0001452528582397, 0.009445593735401357, 10.0, 50.0, 1e-8),
        (1.0, 0.009, 1.0, 1e-5, 0.0),
        (20.0, 1.2, 0.9, 1e-10, 0.0),
        (20.0, 1.2, 10.0, 75.0, 0.02),
        (0.2, 1 / np.sqrt(2), 40.0, 50.0, 0.0),
    ]
    X = _frame(1)
    u = fitted.threshold
    with mpmath.workdps(50):
        for mean, cv, a, w, q in cases:
            shape = 1 / mpmath.mpf(cv) ** 2
            scale = mpmath.mpf(mean) / shape

            def moments(lo, hi):
                return (
                    mpmath.gammainc(shape, lo / scale, hi / scale, regularized=True),
                    mean * mpmath.gammainc(shape + 1, lo / scale, hi / scale, regularized=True),
                )

            _population(monkeypatch, fitted, (mean, cv), (5, 0.2), q)
            expected = _partial_moment_layer(moments, a, w, u, q)
            assert fitted.predict_layer(X, a, w)[0] == pytest.approx(expected, rel=2e-10, abs=0)


def test_lognormal_layers_against_partial_moments(monkeypatch):
    X, y, u = _sample(800)
    fitted = _fit(X, 10 * y, 10 * u, _components(LogNormalLS(parametrisation="location")))
    cases = [
        (0.0, 0.011, 1.0, 999.0, 0.0),
        (0.0009701, 0.0113818, 0.999, 999.0, 0.1),
        (0.0, 0.011, 10.0, 500.0, 1e-8),
        (0.0, 0.011, 1.0, 1e-9, 0.0),
        (2.0, 1.5, 10.0, 150.0, 0.02),
        (0.0, 0.2, 80.0, 800.0, 0.0),
    ]
    with mpmath.workdps(50):
        for mu, sigma, a, w, q in cases:
            mu_mp, sigma_mp = map(mpmath.mpf, (mu, sigma))
            mean = mpmath.exp(mu_mp + sigma_mp**2 / 2)

            def moments(lo, hi):
                z_lo = (mpmath.log(lo) - mu_mp) / sigma_mp
                z_hi = (mpmath.log(hi) - mu_mp) / sigma_mp
                return (
                    mpmath.ncdf(-z_lo) - mpmath.ncdf(-z_hi),
                    mean * (mpmath.ncdf(sigma_mp - z_lo) - mpmath.ncdf(sigma_mp - z_hi)),
                )

            _population(monkeypatch, fitted, (mu, sigma), (5, 0.2), q)
            expected = _partial_moment_layer(moments, a, w, fitted.threshold, q)
            assert fitted.predict_layer(X[:1], a, w)[0] == pytest.approx(expected, rel=2e-10, abs=0)


@pytest.mark.parametrize("lognormal", [False, True])
def test_narrow_deep_body_layers_and_deep_left_normalization(monkeypatch, fitted, lognormal):
    X = _frame(4)
    if lognormal:
        X_train, y, u = _sample()
        fitted = _fit(X_train, y, u, _components(LogNormalLS(parametrisation="location")))
        body = stats.lognorm(0.1, scale=1.0)
        theta = (0.0, 0.1)
    else:
        body = stats.gamma(2, scale=0.1)
        theta = (0.2, 1 / np.sqrt(2))
    q = np.array([0.0, 0.0, 0.0, 1e-200])
    _population(monkeypatch, fitted, theta, (5, 0.2), q)
    a = np.array([0.9, 25.0, 68.0, 1e72])
    limit = np.array([1e-10, 3e-7, 5e-5, 1e70])
    u = fitted.threshold
    log_u = body.logcdf(u)
    expected = []
    for left, width in zip(a[:3], limit[:3], strict=True):
        log_sf_u = body.logsf(u)

        def sf(t):
            log_sf = body.logsf(left + width * t)
            return np.exp(log_sf - log_u) * -np.expm1(log_sf_u - log_sf)

        expected.append(width * integrate.quad(sf, 0, 1, epsabs=0)[0])
    layers = fitted.predict_layer(X, a, limit)
    np.testing.assert_allclose(layers[:3], expected, rtol=2e-10, atol=0)
    # The tail survival underflows on its own, as does the weighted probability.
    # A much larger width makes the layer itself representable in a second probe.
    _population(monkeypatch, fitted, theta, (5, 0.2), np.array([0.0, 0.0, 0.0, 1e-20]))
    log_tail_sf = stats.genpareto.logsf(a[-1] - u, 0.2, scale=5)
    assert layers[-1] == 0.0
    average = integrate.quad(
        lambda t: np.exp(
            stats.genpareto.logsf(a[-1] + t * limit[-1] - u, 0.2, scale=5) - log_tail_sf
        ),
        0,
        1,
    )[0]
    expected_tail = np.exp(np.log(1e-20) + np.log(limit[-1]) + log_tail_sf) * average
    assert fitted.predict_layer(X, a, limit)[-1] == pytest.approx(expected_tail, rel=2e-12, abs=0)

    if lognormal:
        u = fitted.threshold
        mu = np.log(u) + 42
        _population(monkeypatch, fitted, (mu, 1), (5, 0.2), 0.1)
        reference = stats.norm.logcdf(-42)
        y_at = u * np.exp(-0.02)
        expected_cdf = 0.9 * np.exp(stats.norm.logcdf(np.log(y_at) - mu) - reference)
        p = fitted.predict_cdf(X[:1], y_at)[0]
        assert p == pytest.approx(expected_cdf, rel=2e-12)
        assert fitted.predict_quantile(X[:1], p)[0] == pytest.approx(y_at, rel=2e-12)
        a0, w = u - 0.02, 0.03
        expected_layer = (
            integrate.quad(
                lambda t: -0.9 * np.expm1(stats.norm.logcdf(np.log(t) - mu) - reference) + 0.1,
                a0,
                u,
            )[0]
            + integrate.quad(lambda t: 0.1 * stats.genpareto.sf(t - u, 0.2, scale=5), u, a0 + w)[0]
        )
        assert fitted.predict_layer(X[:1], a0, w)[0] == pytest.approx(expected_layer, rel=2e-10)
    else:
        # The unconditioned Gamma CDF underflows here, although the truncated
        # CDF and its inverse are both ordinary float64 numbers.
        _population(monkeypatch, fitted, (1000, 1 / np.sqrt(1000)), (5, 0.2), 0.1)
        with mpmath.workdps(40):
            expected_cdf = 0.9 * float(mpmath.gammainc(1000, 0, 99) / mpmath.gammainc(1000, 0, u))
        p = fitted.predict_cdf(X[:1], 99)[0]
        assert p == pytest.approx(expected_cdf, rel=2e-11)
        assert fitted.predict_quantile(X[:1], p)[0] == pytest.approx(99, rel=2e-12)


def test_extreme_exceedance_probabilities(monkeypatch, fitted):
    X = _frame(4)
    q = np.array([0.0, 1e-200, 1 - 2**-52, 1.0])
    u = fitted.threshold
    _population(monkeypatch, fitted, (60, 1 / np.sqrt(3)), (35, 0.25), q)
    np.testing.assert_array_equal(fitted.predict_cdf(X, u), 1 - q)
    np.testing.assert_array_equal(fitted.predict_survival(X, u), q)
    np.testing.assert_allclose(
        fitted.predict_survival(X, 500),
        q * stats.genpareto.sf(500 - u, 0.25, scale=35),
        rtol=2e-13,
        atol=0,
    )
    p = np.array([0.99, 0.99, (1 - q[2]) / 2, 1 - 2**-52])
    y = fitted.predict_quantile(X, p)
    assert np.all(y[:3] < u) and y[3] > u
    np.testing.assert_allclose(fitted.predict_cdf(X, y), p, rtol=2e-12, atol=0)


def test_censoring_ties_weights_and_manual_fit_parity():
    X, y, u = _sample()
    y[:2] = u
    right = y > u + 60
    right[1] = True  # strictly above its known censoring limit, exactly u
    y = np.minimum(y, u + 60)
    count = np.resize(np.array([1, 2, 1, 3, 0]), len(X))
    mask = (y > u) | right
    options = {"retain_rows": False}
    body, binary, tail = _components(frequency=True)
    body.fit(X.iloc[~mask], y[~mask], count[~mask], truncation=(None, u), **options)
    active = count > 0
    binary.fit(X.iloc[active], mask[active].astype(float), count[active])
    tail.fit(X.iloc[mask], y[mask] - u, count[mask], censored=right[mask], **options)
    splice = _fit(
        X,
        pd.Series(y, index=X.index),
        u,
        _components(frequency=True),
        censored=pd.Series(right, index=X.index),
        sample_weight=pd.Series(count, index=X.index),
        body_fit_options=options,
        tail_fit_options=options,
    )
    X_test = X.iloc[::61]
    for model, manual in ((splice.body, body), (splice.tail, tail)):
        np.testing.assert_array_equal(
            model.predict_parameters(X_test), manual.predict_parameters(X_test)
        )
        assert (
            model.training_telemetry().inner_iterations
            == manual.training_telemetry().inner_iterations
        )
        assert (
            model.training_telemetry().execution_backend_identifier
            == manual.training_telemetry().execution_backend_identifier
        )
        work, reference = model.diagnose().profile, manual.diagnose().profile
        assert work is not None and reference is not None
        assert work.coefficient_fits == reference.coefficient_fits
    np.testing.assert_array_equal(splice.exceedance.predict(X_test), binary.predict(X_test))
    assert splice.exceedance.result.n_iter == binary.result.n_iter
    assert splice.exceedance.result.direct_backend == binary.result.direct_backend
    copies = [
        pickle.loads(pickle.dumps(part)) for part in (splice.body, splice.exceedance, splice.tail)
    ]
    restored = [SplitSplice(*copies), pickle.loads(pickle.dumps(splice))]
    for new in restored:
        assert new.threshold == u
        for fn, args in (
            ("predict_cdf", (105,)),
            ("predict_survival", (1000,)),
            ("predict_quantile", (0.95,)),
            ("predict", ()),
            ("predict_layer", (90, 16)),
        ):
            np.testing.assert_array_equal(
                getattr(new, fn)(X_test, *args), getattr(splice, fn)(X_test, *args)
            )


def test_wrong_cutoff_tail_origin_and_split_are_refused():
    X, y, u = _sample()
    body, binary, tail = _components()
    large = y > u
    body.fit(X.iloc[~large], y[~large], truncation=(None, u + 20), retain_rows=False)
    tail.fit(X.iloc[large], y[large], retain_rows=False)
    verified = _fit(X, y, u)
    with pytest.raises(ValueError, match="body.*SplitSplice"):
        SplitSplice(body, verified.exceedance, verified.tail)
    with pytest.raises(ValueError, match="tail.*SplitSplice"):
        SplitSplice(verified.body, verified.exceedance, tail)
    other = _fit(X.iloc[::-1], y[::-1], u)
    with pytest.raises(ValueError, match="same split"):
        SplitSplice(verified.body, other.exceedance, verified.tail)
    verified.tail.fit(X.iloc[large], y[large])
    with pytest.raises(ValueError, match="tail.*SplitSplice"):
        verified.predict(X.iloc[:1])


@pytest.mark.parametrize("name", ["y", "sample_weight", "censored"])
def test_series_row_alignment_is_checked_before_fit(name):
    X, y, u = _sample(100)
    args = dict(y=y, sample_weight=np.ones(len(y)), censored=np.zeros(len(y), dtype=bool))
    args[name] = pd.Series(args[name], index=X.index[::-1])
    with pytest.raises(ValueError, match=f"{name} index"):
        _fit(X, u=u, **args)


def test_ambiguous_censoring_fit_args_and_prior_weights_are_refused():
    X, y, u = _sample(100)
    with pytest.raises(ValueError, match="membership"):
        _fit(X, y, u, censored=y < u)
    with pytest.raises(ValueError, match="truncation"):
        _fit(X, y, u, body_fit_options={"truncation": (None, u + 20)})
    with pytest.raises(ValueError, match="frequency"):
        _fit(X, y, u, sample_weight=np.full(len(y), 2))
    with pytest.raises(LikelihoodWeightError, match="integers"):
        _fit(X, y, u, sample_weight=np.full(len(y), 1.5))


@pytest.mark.parametrize("component", ["body", "tail"])
def test_split_slices_declared_offsets(component):
    X, y, u = _sample(800)
    X["fixed"] = 0.15 * np.sin(y)
    X["array"] = 0.10 * np.cos(y)
    X["indexed"] = 0.07 * np.sin(y / 3)

    def models(sources):
        body, binary, tail = _components()
        if component == "body":
            family = body.family
            body = SuperLSS(family, family.mean(offset=sources), family.scale())
        else:
            family = tail.family
            tail = SuperLSS(family, family.scale(offset=sources), family.shape())
        return body, binary, tail

    array = X["array"].to_numpy()
    indexed = X["indexed"]
    splice = _fit(X, y, u, models(("fixed", array, indexed)))
    column_splice = _fit(X, y, u, models(("fixed", "array", "indexed")))
    key = "mean" if component == "body" else "scale"
    offset_options = {f"{component}_offsets": {key: array + indexed}}
    np.testing.assert_allclose(
        splice.predict_layer(X, 95, 15, **offset_options),
        column_splice.predict_layer(X, 95, 15),
        rtol=2e-13,
        atol=0,
    )
    _fit(X, y, u, (splice.body, splice.exceedance, splice.tail))

    wrong_index = pd.Series(array, index=X.index[::-1])
    with pytest.raises(ValueError, match="index must match X on the original rows"):
        _fit(X, y, u, models(wrong_index))
    component_rows = y <= u if component == "body" else y > u
    with pytest.raises(ValueError, match=f"length {len(X)}"):
        _fit(X, y, u, models(array[component_rows]))


@pytest.mark.parametrize("sentinel", ["zero loss", "nonfinite loss", "ambiguous", "missing flag"])
def test_zero_frequency_rows_are_deleted_before_value_checks(sentinel):
    X, y, u = _sample(800)
    counts = np.resize([1, 2, 1, 3], len(X))
    counts[5] = 0
    keep = counts > 0
    offset = np.linspace(-0.2, 0.2, len(X))
    right = np.zeros(len(X), dtype=bool)
    expected = _fit(
        X.iloc[keep],
        y[keep],
        u,
        _components(frequency=True),
        sample_weight=counts[keep],
        exceedance_fit_options={"offset": offset[keep]},
    )
    if sentinel == "zero loss":
        y[5] = 0.0
    elif sentinel == "nonfinite loss":
        y[5] = np.nan
    elif sentinel == "ambiguous":
        y[5], right[5] = u / 2, True
    else:
        right = right.astype(object)
        right[5] = pd.NA
    offset[5] = np.nan
    actual = _fit(
        X,
        pd.Series(y, index=X.index),
        u,
        _components(frequency=True),
        sample_weight=pd.Series(counts, index=X.index),
        censored=pd.Series(right, index=X.index),
        exceedance_fit_options={"offset": pd.Series(offset, index=X.index)},
    )
    score = X.iloc[keep]
    np.testing.assert_array_equal(
        actual.predict_layer(score, 95, 15, exceedance_offset=offset[keep]),
        expected.predict_layer(score, 95, 15, exceedance_offset=offset[keep]),
    )


def test_seed_544_against_independent_fits():
    a, scale = 1 / 1.2**2, 5000 * 1.2**2
    truth = stats.gamma(a, scale=scale)
    u = truth.ppf(0.9)
    y = np.random.default_rng(544).gamma(a, scale, 200_000)
    X = _frame(len(y))
    large = y > u
    q = large.mean()
    scaled_body = y[~large] / scale

    def nll(log_parameters):
        shape, unit_scale = np.exp(log_parameters)
        law = stats.gamma(shape, scale=unit_scale)
        return law.logcdf(u / scale) - law.logpdf(scaled_body).mean()

    result = optimize.minimize(nll, [np.log(a), 0], method="Nelder-Mead", options={"xatol": 1e-9})
    assert result.success
    a_hat, scale_hat = np.exp(result.x)
    body = stats.gamma(a_hat, scale=scale_hat * scale)
    xi, _, beta = stats.genpareto.fit((y[large] - u) / scale, floc=0)
    tail = stats.genpareto(xi, scale=beta * scale)
    mean = (1 - q) * integrate.quad(lambda v: v * body.pdf(v), 0, u)[0] / body.cdf(u)
    mean += q * (u + tail.mean())
    p = np.array([0.5, 0.9, 0.99])
    quantiles = np.where(p <= 1 - q, body.ppf(p * body.cdf(u) / (1 - q)), u + tail.isf((1 - p) / q))
    splice = _fit(
        X, y, u, body_fit_options={"retain_rows": False}, tail_fit_options={"retain_rows": False}
    )
    np.testing.assert_allclose(splice.predict_quantile(X.iloc[:3], p), quantiles, rtol=5e-5)
    assert splice.predict(X.iloc[:1])[0] == pytest.approx(mean, rel=5e-5)
    assert splice.predict_cdf(X.iloc[:1], u)[0] == pytest.approx(1 - q, abs=1e-8)
