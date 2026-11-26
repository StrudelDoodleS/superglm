"""CDF and quantile of ``GammaLS`` and ``GeneralizedGammaLSS`` at large gamma shape.

Above gamma shape 1e4 both families, and the Gamma and Poisson GLMs' quantile
residuals, read the log tail's uniform expansion in place of SciPy's
incomplete gamma, whose power series stops at 2000 terms.

Oracle.  For the gamma variate ``G`` of shape ``a``, ``z = sqrt(a) log(G/a)``
has density ``exp(c + a (t - expm1(t)))``, ``t = z/sqrt(a)``, within
``O(a^-1/2)`` of the standard normal's.  The smaller tail is its mpmath
integral over panels of at most two e-folds; the other is the complement.  It
reads the float64 amount exactly and shares no formula with SciPy or with the
log tail.

Tolerance.  A tail probability has condition number about ``w^2`` in the
standardised log amount ``w``, so a value is held to ``2^11 u max(1, w^2)`` of
the smaller tail, plus ``8 u`` of the CDF itself where it is near one.
A gamma quantile is a float64 beside its mean, which resolves ``w`` to
``u/scale`` at best (``scale = cv/sqrt(weight)``).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import special

from superglm import Numeric, SuperGLM
from superglm._weight_units import likelihood_scale
from superglm.distributional.families.gamma import GammaLS
from superglm.distributional.families.generalized_gamma import GeneralizedGammaLSS

mp = pytest.importorskip("mpmath")

_EPS = np.finfo(float).eps
_DIGITS = 30
# Far left, ordinary, far right.
_POINTS = (-6.0, -5.5, 0.5, 5.5)
_LEVELS = (1e-9, 0.5, 1.0 - 1e-9)
_GAMMA_LAWS = [(1.0, 1e-4, 1.0), (250.0, 1e-6, 1.0), (3.0, 0.3, 1e6)]


def _gamma_tails(shape, log_ratio):
    """``(P, Q)`` of gamma shape ``shape`` at ``x = shape exp(log_ratio)``, as mpmath numbers."""
    root = mp.sqrt(shape)
    start = log_ratio * root
    constant = shape * mp.log(shape) - shape - mp.loggamma(shape) - mp.log(shape) / 2

    def exponent(z):
        t = z / root
        return shape * (t - mp.expm1(t))

    peak = exponent(start)
    step = -1 if start <= 0 else 1
    ends = [start]
    while exponent(ends[-1]) > peak - 70:
        rate = abs(root * mp.expm1(ends[-1] / root))
        ends.append(ends[-1] + step * min(mp.mpf(1), 2 / max(rate, mp.mpf("1e-9"))))
    # mp.quad stops on an absolute error, so the integrand is one at the point.
    tail = mp.exp(constant + peak) * mp.quad(lambda z: mp.exp(exponent(z) - peak), ends[::step])
    return (tail, 1 - tail) if step < 0 else (1 - tail, tail)


def _generalized_gamma_tails(y, mu, sigma, q):
    """Reference ``(F, S)`` of the location-form law at the float64 amount ``y``."""
    w = (mp.log(mp.mpf(float(y))) - mp.mpf(mu)) / mp.mpf(sigma)
    lower, upper = _gamma_tails(1 / mp.mpf(q) ** 2, mp.mpf(q) * w)
    return (lower, upper) if q > 0 else (upper, lower)


def _assert_cdf(actual, lower, upper, w):
    rtol = 2**10 * _EPS * max(1.0, w * w)
    assert abs(mp.mpf(float(actual)) - lower) <= rtol * min(lower, upper) + 4 * _EPS * lower


def _assert_level(level, lower, upper, resolution=0.0):
    """The reference tail at a returned quantile is the requested one, on its smaller side."""
    w = float(special.ndtri(level))
    rtol = 2**10 * _EPS * (max(1.0, w * w) + (1.0 + abs(w)) * resolution)
    exact = mp.mpf(float(level))
    if level <= 0.5:
        assert abs(lower - exact) <= rtol * exact
    else:
        assert abs(upper - (1 - exact)) <= rtol * (1 - exact)


def _generalized_gamma_meets_the_reference(q):
    family = GeneralizedGammaLSS(parametrisation="location")
    mu, sigma = 0.3, 0.5
    y = np.exp(mu + sigma * np.array(_POINTS))
    cdf = family.cdf(y, np.tile([mu, sigma, q], (len(y), 1)))
    levels = np.array(_LEVELS)
    quantile = family.quantile(levels, np.tile([mu, sigma, q], (len(levels), 1)))
    with mp.workdps(_DIGITS):
        for value, amount, w in zip(cdf, y, _POINTS, strict=True):
            _assert_cdf(value, *_generalized_gamma_tails(amount, mu, sigma, q), w)
        for level, amount in zip(levels, quantile, strict=True):
            _assert_level(level, *_generalized_gamma_tails(amount, mu, sigma, q))


@pytest.mark.parametrize("q", [1e-3, 1e-4, -1e-4, 1e-6, -1e-8])
def test_generalized_gamma_cdf_and_quantile_match_the_density_integral_near_log_normal(q):
    _generalized_gamma_meets_the_reference(q)


@pytest.mark.pins_behaviour
@pytest.mark.parametrize("q", [0.0101, 0.0099])
def test_generalized_gamma_meets_the_reference_either_side_of_the_shape_switch(q):
    """Pins that SciPy's incomplete gamma just below gamma shape 1e4 and the log tail just
    above it meet the same reference to the same tolerance, so the switch moves no number."""
    _generalized_gamma_meets_the_reference(q)


@pytest.mark.parametrize(("mean", "cv", "weight"), _GAMMA_LAWS)
def test_gamma_cdf_matches_the_density_integral_at_large_shape(mean, cv, weight):
    family = GammaLS()
    scale = cv / np.sqrt(weight)
    y = mean * np.exp(scale * np.array(_POINTS))
    theta = np.tile([mean, cv], (len(y), 1))
    cdf = family.cdf_prior_weighted(y, theta, np.full(len(y), weight))
    if weight == 1.0:
        assert np.array_equal(family.cdf(y, theta), cdf)
    with mp.workdps(_DIGITS):
        shape = mp.mpf(weight) / mp.mpf(cv) ** 2
        for value, amount, w in zip(cdf, y, _POINTS, strict=True):
            lower, upper = _gamma_tails(shape, mp.log(mp.mpf(float(amount)) / mp.mpf(mean)))
            _assert_cdf(value, lower, upper, w)


@pytest.mark.parametrize(("mean", "cv", "weight"), _GAMMA_LAWS)
def test_gamma_quantile_inverts_the_density_integral_at_large_shape(mean, cv, weight):
    family = GammaLS()
    levels = np.array(_LEVELS)
    theta = np.tile([mean, cv], (len(levels), 1))
    quantile = family.quantile_prior_weighted(levels, theta, np.full(len(levels), weight))
    if weight == 1.0:
        assert np.array_equal(family.quantile(levels, theta), quantile)
    with mp.workdps(_DIGITS):
        shape = mp.mpf(weight) / mp.mpf(cv) ** 2
        for level, amount in zip(levels, quantile, strict=True):
            lower, upper = _gamma_tails(shape, mp.log(mp.mpf(float(amount)) / mp.mpf(mean)))
            _assert_level(level, lower, upper, resolution=np.sqrt(weight) / cv)


@pytest.mark.parametrize("cv", [1e-9, 1e-30, 1e-170])
def test_gamma_distribution_functions_stay_probabilities_at_any_concentration(cv):
    """A law no fit produces needs no digits: a finite probability that never decreases."""
    family = GammaLS()
    mean = 2.0
    y = mean * np.array([0.5, 1.0 - 6e-9, 1.0, 1.0 + 6e-9, 2.0])
    levels = np.array(_LEVELS)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cdf = family.cdf(y, np.tile([mean, cv], (len(y), 1)))
        quantile = family.quantile(levels, np.tile([mean, cv], (len(levels), 1)))
    assert cdf[0] == 0.0 and cdf[-1] == 1.0 and np.all(np.diff(cdf) >= 0.0)
    assert cdf[2] == pytest.approx(0.5, abs=cv + 2.0 * _EPS)
    assert np.all(np.diff(quantile) >= 0.0)
    assert quantile == pytest.approx(mean, rel=8.0 * cv + _EPS)
    if cv == 1e-9:
        # Six standard deviations out the law is the normal in log amount to O(cv).
        w = np.log1p((y[[1, 3]] - mean) / mean) / cv
        assert cdf[[1, 3]] == pytest.approx(special.ndtr(w), rel=1e-6)


_TAIL = (-6.0, -5.5, -5.0, -4.6)


def _glm_residuals(family, y, weights, x):
    """Quantile residuals of a prior-weighted GLM on ``x``, its fitted means and dispersion."""
    frame = pd.DataFrame({"x": x})
    model = SuperGLM(family=family, selection_penalty=0.0, features={"x": Numeric()})
    model.fit(frame, y, sample_weight=weights)
    residuals = model.metrics(frame, y, sample_weight=weights).residuals("quantile")
    return residuals, model.predict(frame), likelihood_scale(model.result).phi


def _score(probability):
    return special.ndtri(np.clip(float(probability), 1e-10, 1.0 - 1e-10))


def test_gamma_glm_quantile_residuals_match_the_density_integral_at_large_prior_weight():
    """A row's residual is ``Phi^-1`` of its gamma CDF at shape ``w / phi``, clipped at 1e-10.

    The first rows sit 4.6 to 6 standard deviations below the mean, where SciPy's series
    stops short at this shape (2e8). The CDF tolerance above bounds the residual's.
    """
    rng = np.random.default_rng(7)
    n, weight = 200, 1e8
    x = rng.standard_normal(n)
    mean = np.exp(1.0 + 0.3 * x)
    shape = weight / 0.5
    y = rng.gamma(shape, mean / shape)
    y[: len(_TAIL)] = mean[: len(_TAIL)] * np.exp(np.array(_TAIL) / np.sqrt(shape))
    residuals, mu, phi = _glm_residuals("gamma", y, np.full(n, weight), x)
    rows = slice(0, 2 * len(_TAIL))
    with mp.workdps(_DIGITS):
        fitted_shape = mp.mpf(weight) / mp.mpf(phi)
        for residual, amount, fitted in zip(residuals[rows], y[rows], mu[rows], strict=True):
            log_ratio = mp.log(mp.mpf(float(amount)) / mp.mpf(float(fitted)))
            expected = _score(_gamma_tails(fitted_shape, log_ratio)[0])
            assert residual == pytest.approx(expected, abs=2**10 * _EPS * max(1.0, expected**2))


def test_poisson_glm_quantile_residuals_match_the_density_integral_at_large_counts():
    """A row's residual lies between ``Phi^-1`` of ``F(k - 1)`` and ``F(k)``, ``F(k) = Q(k + 1, w mu)``.

    The first counts sit 4.6 to 6 standard deviations above the mean (about 1.4e8), where
    SciPy's series stops short. A probability near one is a float64 within ``eps`` of it,
    ``eps / phi(z)`` in the residual.
    """
    rng = np.random.default_rng(11)
    n, exposure = 200, 1e9
    x = rng.standard_normal(n)
    lam = exposure * np.exp(-2.0 + 0.2 * x)
    counts = rng.poisson(lam).astype(float)
    head = slice(0, len(_TAIL))
    counts[head] = np.round(lam[head] - np.array(_TAIL) * np.sqrt(lam[head]))
    rate = counts / exposure
    residuals, mu, _ = _glm_residuals("poisson", rate, np.full(n, exposure), x)
    rows = slice(0, 2 * len(_TAIL))
    with mp.workdps(_DIGITS):
        for residual, observed, fitted in zip(residuals[rows], rate[rows], mu[rows], strict=True):
            count, mean = np.round(exposure * observed), mp.mpf(float(exposure * fitted))
            low, high = (
                _score(_gamma_tails(mp.mpf(k + 1.0), mp.log(mean / (k + 1.0)))[1])
                for k in (count - 1.0, count)
            )
            density = np.exp(-0.5 * high * high) / np.sqrt(2.0 * np.pi)
            tol = 4.0 * _EPS / density + 2**10 * _EPS * max(1.0, high * high)
            assert low - tol <= residual <= high + tol


@pytest.mark.pins_behaviour
@pytest.mark.parametrize("family", ["gamma", "poisson"])
def test_glm_residual_at_an_overflowing_prior_weight_stays_saturated(family):
    """Pins master's behaviour where ``w / phi`` or the weighted count ``w y`` overflows.

    At prior weight 1e308 a response twice its mean is far above the law, so the residual
    is the cap ``Phi^-1(1 - 1e-10)``. Mutation check: sending a row whose shape is
    infinite to the log tail makes the Poisson endpoints NaN and the draw raise.
    """
    rng = np.random.default_rng(5)
    frame = pd.DataFrame({"x": rng.standard_normal(60)})
    y = rng.gamma(4.0, 0.25, 60) if family == "gamma" else rng.poisson(1.0, 60).astype(float)
    model = SuperGLM(family=family, selection_penalty=0.0, features={"x": Numeric()})
    model.fit(frame, y)
    row = frame.iloc[:1]
    response = 2.0 * model.predict(row)
    residual = model.metrics(row, response, sample_weight=np.array([1e308])).residuals("quantile")
    assert residual == pytest.approx([special.ndtri(1.0 - 1e-10)], rel=1e-12)


@pytest.mark.pins_behaviour
@pytest.mark.parametrize(
    ("family", "observed", "response", "weight", "level"),
    [
        # Endpoints either side of the shape switch: F(9999) from SciPy, F(10000) the log tail.
        ("poisson", [0.0, 1.0, 1.0, 2.0], 10000.0 / 9192.0, 9192.0, 1.0 - 1e-10),
        # y / mu rounds to zero at an ordinary shape.
        ("gamma", [0.5, 1.5, 0.5, 1.5], 1e-20, 1.0, 1e-10),
    ],
)
def test_glm_residual_at_the_edges_of_the_tail_switch_is_the_cap(
    family, observed, response, weight, level
):
    """Pins master's behaviour at two boundary rows: the capped residual, with no warning.

    Mutation checks: drawing between unordered Poisson endpoints raises ``high - low < 0``;
    forming ``log1p(-1)`` for the Gamma row without silencing it warns of a division by zero.
    """
    frame = pd.DataFrame(index=range(len(observed)))
    model = SuperGLM(family=family, selection_penalty=0.0, features={})
    model.fit(frame, np.array(observed))
    metrics = model.metrics(frame.iloc[:1], np.array([response]), sample_weight=np.array([weight]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        residual = metrics.residuals("quantile")
    assert residual == pytest.approx([special.ndtri(level)], rel=1e-12)


@pytest.mark.pins_behaviour
@pytest.mark.parametrize("q", [1e-155, -1e-155, 0.0])
def test_generalized_gamma_near_zero_shape_raises_no_overflow(q):
    """Pins master's behaviour at a tiny Q (the log-normal limit): no overflow, no warning.

    Choosing the large-shape branch must not form ``1/Q^2``. Mutation check: testing
    ``1/Q^2 > 1e4`` overflows at ``Q = 1e-155``.
    """
    family = GeneralizedGammaLSS(parametrisation="location")
    theta = np.array([[0.0, 1.0, q]] * 3)
    with warnings.catch_warnings(), np.errstate(over="raise", divide="raise", invalid="raise"):
        warnings.simplefilter("error")
        cdf = family.cdf(np.array([0.5, 1.0, 2.0]), theta)
        quantile = family.quantile(np.array([1e-6, 0.5, 1.0 - 1e-6]), theta)
    assert cdf == pytest.approx(special.ndtr(np.log([0.5, 1.0, 2.0])), rel=1e-12)
    assert quantile[1] == pytest.approx(1.0, rel=1e-14)
    assert np.all(np.diff(quantile) > 0.0)


@pytest.mark.pins_behaviour
def test_an_ordinary_law_never_calls_the_log_tail_kernel(monkeypatch):
    """Pins master's behaviour below the switch: no call into the compiled log tail.

    Its first call compiles it, so an ordinary law must not reach it. Mutation check:
    calling the kernel with zero rows counts one call per function.
    """
    from superglm.distributional.kernels import generalized_gamma as kernels

    calls = []
    original = kernels.gg_log_tail_jets

    def counted(*args, **kwargs):
        calls.append(len(args[0]))
        return original(*args, **kwargs)

    monkeypatch.setattr(kernels, "gg_log_tail_jets", counted)
    y, levels = np.array([0.5, 1.0, 2.0]), np.array([0.1, 0.5, 0.9])
    gamma, theta = GammaLS(), np.tile([1.0, 0.5], (3, 1))
    gamma.cdf(y, theta)
    gamma.quantile(levels, theta)
    gg = GeneralizedGammaLSS(parametrisation="location")
    theta = np.tile([0.0, 0.5, 0.5], (3, 1))
    gg.cdf(y, theta)
    gg.quantile(levels, theta)
    rng = np.random.default_rng(3)
    x = rng.standard_normal(60)
    _glm_residuals("gamma", rng.gamma(4.0, 0.25, 60), np.full(60, 2.0), x)
    _glm_residuals("poisson", rng.poisson(3.0, 60) / 2.0, np.full(60, 2.0), x)
    assert calls == []
