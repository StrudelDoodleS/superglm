"""Public variances retain representable results across intermediate range limits."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

from superglm import GammaLS, GaussianLS, LogNormalLS, TweedieLSS


def _decimal(value):
    return Decimal.from_float(float(value))


def _assert_variance(actual, expected, condition=1.0):
    # gamma_32 covers the bounded products, power and final rounding. The
    # expression condition accounts for absolute error in an exponential's
    # argument; this is needed only on transcendental recovery paths.
    u = _decimal(np.finfo(float).eps / 2)
    allowance = 32 * u / (1 - 32 * u) * _decimal(condition) * abs(expected)
    allowance += 4 * _decimal(np.nextafter(0.0, 1.0))
    assert np.isfinite(actual)
    assert abs(_decimal(actual) - expected) <= allowance


@pytest.mark.parametrize("kind", ["gaussian", "gamma"])
@pytest.mark.parametrize(
    "spread,weight", [(1e155, 1e10), (1e-170, 1e-100), (1e-160, 1e-20), (2.5, 4.0)]
)
def test_prior_variance_combines_square_and_weight_before_range_rounding(kind, spread, weight):
    family = GaussianLS(scale_floor=0.0) if kind == "gaussian" else GammaLS()
    row = [0.0, spread] if kind == "gaussian" else [spread, 1.0]
    with localcontext() as context, np.errstate(all="ignore"):
        context.prec = 200
        expected = _decimal(spread) ** 2 / _decimal(weight)
        actual = family.variance_prior_weighted(np.array([row]), weight)
        _assert_variance(actual[0], expected)
        assert not actual.flags.writeable


@pytest.mark.parametrize("weight", [1.0, 1e-300, 1e300])
def test_gamma_prior_variance_retains_both_mean_and_cv_factors(weight):
    with localcontext() as context, np.errstate(all="ignore"):
        context.prec = 200
        mean, cv = 1e250, 1e-250
        expected = (_decimal(mean) * _decimal(cv)) ** 2 / _decimal(weight)
        actual = GammaLS().variance_prior_weighted(np.array([[mean, cv]]), weight)[0]
        _assert_variance(actual, expected)


@pytest.mark.parametrize(
    "mean,scale",
    [(1e-170, 10.0), (1e-200, 30.0), (1e160, 1e-10), (1e200, 1e-200), (2.0, 0.7)],
)
def test_log_normal_mean_variance_recovers_squared_mean_and_exponential(mean, scale):
    with localcontext() as context, np.errstate(all="ignore"):
        # Tiny sigma needs enough precision to resolve exp(sigma^2) - 1.
        context.prec = 500
        square = _decimal(scale) ** 2
        expected = _decimal(mean) ** 2 * (square.exp() - 1)
        actual = LogNormalLS(scale_floor=0.0).variance(np.array([[mean, scale]]))[0]
        condition = 1.0
        if square > 700:
            condition += 2 * abs(float(_decimal(mean).ln())) + float(square)
        _assert_variance(actual, expected, condition)


@pytest.mark.parametrize(
    "location,scale", [(-1000.0, 30.0), (-400.0, 10.0), (355.0, 1e-10), (0.3, 0.7)]
)
def test_log_normal_location_variance_combines_location_and_scale(location, scale):
    with localcontext() as context, np.errstate(all="ignore"):
        context.prec = 200
        square = _decimal(scale) ** 2
        expected = (2 * _decimal(location) + square).exp() * (square.exp() - 1)
        actual = LogNormalLS(parametrisation="location", scale_floor=0.0).variance(
            np.array([[location, scale]])
        )[0]
        condition = 1 + 2 * abs(location) + 2 * float(square)
        _assert_variance(actual, expected, condition)


@pytest.mark.parametrize(
    "mean,dispersion,power,weight",
    [
        (1e-250, 1e100, 1.5, 1.0),
        (1e250, 1e-100, 1.5, 1.0),
        (1e-250, 1e100, 1.7, 1e-50),
        (1e250, 1e100, 1.5, 1e200),
        (1e-200, 1e-100, 1.5, 1e-200),
        (2.0, 0.7, 1.3, 4.0),
    ],
)
def test_tweedie_variance_combines_fractional_power_dispersion_and_weight(
    mean, dispersion, power, weight
):
    with localcontext() as context, np.errstate(all="ignore"):
        context.prec = 200
        expected = _decimal(dispersion) * _decimal(mean) ** _decimal(power) / _decimal(weight)
        family = TweedieLSS()
        row = np.array([[mean, dispersion, power]])
        actual = family.variance_prior_weighted(row, weight)[0]
        # Splitting mu into mantissa and exponent bounds the power argument
        # error by u * abs(exponent * power) * log(2).
        condition = 1 + abs(np.frexp(mean)[1] * power) * np.log(2.0)
        _assert_variance(actual, expected, condition)
        if weight == 1.0:
            _assert_variance(family.variance(row)[0], expected, condition)


@pytest.mark.parametrize(
    "family,row",
    [
        (GaussianLS(scale_floor=0.0), [0.0, 1e200]),
        (GammaLS(), [1e200, 1.0]),
        (TweedieLSS(), [1e200, 1e200, 1.5]),
        (LogNormalLS(), [1.0, 30.0]),
        (LogNormalLS(parametrisation="location"), [0.0, 30.0]),
    ],
)
def test_true_variance_overflow_remains_positive_infinity(family, row):
    with np.errstate(all="ignore"):
        assert np.isposinf(family.variance(np.array([row]))[0])
        if not isinstance(family, LogNormalLS):
            assert np.isposinf(family.variance_prior_weighted(np.array([row]), 1.0)[0])


@pytest.mark.parametrize(
    "family,row",
    [(GaussianLS(), [0.0, 1.0]), (GammaLS(), [1.0, 1.0]), (TweedieLSS(), [1.0, 1.0, 1.5])],
)
@pytest.mark.parametrize("weight", [0.0, -1.0, np.inf, np.nan])
def test_variance_recovery_preserves_prior_weight_validation(family, row, weight):
    with pytest.raises(ValueError, match="prior weights"):
        family.variance_prior_weighted(np.array([row]), weight)
