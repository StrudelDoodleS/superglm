"""Gamma row identities with representable results and extreme intermediates."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

from superglm.distributions import Gamma, Gaussian
from superglm.links import LogLink
from superglm.solvers import working_rows
from superglm.solvers.dispersion import pearson_residual_degrees_of_freedom


def _decimal(value: float) -> Decimal:
    return Decimal.from_float(float(value))


def _assert_rounding_bound(actual: float, expected: Decimal, *, operations: int = 16) -> None:
    """Allow rounded scalar operations and one final subnormal ulp."""
    wanted = float(expected)
    assert np.isfinite(actual)
    assert np.isfinite(wanted)
    unit_roundoff = np.finfo(np.float64).eps / 2.0
    gamma = operations * unit_roundoff / (1.0 - operations * unit_roundoff)
    bound = gamma * abs(wanted) + np.nextafter(0.0, 1.0)
    with localcontext() as context:
        context.prec = 200
        assert abs(_decimal(actual) - expected) <= _decimal(bound)


@pytest.mark.parametrize("prefer_observed", [False, True])
@pytest.mark.parametrize(
    ("weight", "mean", "ratio"),
    [
        (1.0, 1.0, 1.0),
        (1.0e250, 1.0e30, 1.0),
        (1.0e300, 1.0e30, 0.25),
        (1.0e300, 1.0e30, 4.0),
        (1.0e-300, 1.0e-30, 1.0),
        (1.0e-280, 1.0e-30, 1.7),
        (1.0e-270, 1.0e-30, 4.0),
        (1.0e-320, 1.0e-30, 0.25),
    ],
)
def test_gamma_rows_keep_representable_curvature_and_weighted_response(
    weight: float, mean: float, ratio: float, prefer_observed: bool
) -> None:
    """Squaring or multiplying before cancellation must not erase finite rows."""
    eta = np.log(np.array([mean]))
    mu = np.exp(eta)
    y = ratio * mu
    weights = np.array([weight])

    rows = working_rows.coefficient_working_rows(
        distribution=Gamma(),
        link=LogLink(),
        y=y,
        mu=mu,
        eta=eta,
        sample_weight=weights,
        prefer_observed=prefer_observed,
    )

    assert rows.curvature_source == ("observed" if prefer_observed else "fisher")
    assert rows.fallback_reason is None
    with localcontext() as context:
        context.prec = 200
        response = _decimal(y[0])
        location = _decimal(mu[0])
        expected_weight = _decimal(weight)
        if prefer_observed:
            expected_weight *= response / location
            expected_response = _decimal(eta[0]) + (response - location) / response
        else:
            expected_response = _decimal(eta[0]) + (response - location) / location
        _assert_rounding_bound(float(rows.weights[0]), expected_weight)
        _assert_rounding_bound(float(rows.response[0]), expected_response)
        _assert_rounding_bound(
            float(rows.weights[0] * rows.response[0]), expected_weight * expected_response
        )
    if not prefer_observed:
        np.testing.assert_array_equal(rows.weights, weights)
        assert not np.shares_memory(rows.weights, weights)


def test_gamma_observed_rows_do_not_need_an_unscaled_response_mean_ratio() -> None:
    """The weighted curvature can fit even when y/mu itself overflows."""
    eta = np.log(np.array([1.0e-30]))
    mu = np.exp(eta)
    y = np.array([1.0e300])
    weight = np.array([1.0e-100])

    rows = working_rows.coefficient_working_rows(
        distribution=Gamma(),
        link=LogLink(),
        y=y,
        mu=mu,
        eta=eta,
        sample_weight=weight,
        prefer_observed=True,
    )

    assert rows.curvature_source == "observed"
    assert rows.fallback_reason is None
    with localcontext() as context:
        context.prec = 200
        expected = _decimal(weight[0]) * _decimal(y[0]) / _decimal(mu[0])
        _assert_rounding_bound(float(rows.weights[0]), expected)
    assert np.all(np.isfinite(rows.weights * rows.response))


@pytest.mark.parametrize("prefer_observed", [False, True])
def test_gamma_zero_weight_row_does_not_require_a_representable_working_response(
    prefer_observed: bool,
) -> None:
    """An inactive huge ratio must contribute exact zero to W and Wz."""
    eta = np.log(np.array([1.0, 1.0e-30]))
    mu = np.exp(eta)
    y = np.array([2.0, 1.0e300])
    rows = working_rows.coefficient_working_rows(
        distribution=Gamma(),
        link=LogLink(),
        y=y,
        mu=mu,
        eta=eta,
        sample_weight=np.array([1.0, 0.0]),
        prefer_observed=prefer_observed,
    )

    assert rows.curvature_source == ("observed" if prefer_observed else "fisher")
    assert rows.weights[1] == 0.0
    assert np.isfinite(rows.response[1])
    assert rows.weights[1] * rows.response[1] == 0.0


@pytest.mark.parametrize(
    ("response", "mean", "weight"),
    [(2.0, 1.0, 1.0e308), (1.0e-300, 1.0e30, 1.0e300)],
)
def test_unrepresentable_gamma_observed_rows_still_fall_back_as_a_whole(
    response: float,
    mean: float,
    weight: float,
) -> None:
    """Neither true curvature overflow nor response overflow becomes a fake Newton row."""
    eta = np.log(np.array([mean, 1.0]))
    mu = np.exp(eta)
    rows = working_rows.coefficient_working_rows(
        distribution=Gamma(),
        link=LogLink(),
        y=np.array([response, 2.0]),
        mu=mu,
        eta=eta,
        sample_weight=np.array([weight, 0.5]),
        prefer_observed=True,
    )

    assert rows.curvature_source == "fisher"
    assert rows.fallback_reason == "invalid_observed_rows"
    np.testing.assert_array_equal(rows.weights, np.array([weight, 0.5]))
    assert np.all(np.isfinite(rows.response))


def test_gamma_observed_total_weight_overflow_uses_finite_fisher_system() -> None:
    """Finite individual Newton rows do not certify an unrepresentable intercept sum."""
    weights = np.full(2, 4.0e307)
    rows = working_rows.coefficient_working_rows(
        distribution=Gamma(),
        link=LogLink(),
        y=np.full(2, 3.0),
        mu=np.ones(2),
        eta=np.zeros(2),
        sample_weight=weights,
        prefer_observed=True,
    )

    assert rows.curvature_source == "fisher"
    assert rows.fallback_reason == "invalid_observed_rows"
    np.testing.assert_array_equal(rows.weights, weights)
    assert np.isfinite(np.sum(rows.weights))
    assert np.isfinite(np.sum(rows.weights * rows.response))


@pytest.mark.parametrize(
    ("y", "mu", "weights"),
    [
        ([2.0e30, 3.0e30], [1.0e30, 2.0e30], [1.0e250, 1.0e250]),
        ([2.0e-30, 3.0e-30], [1.0e-30, 2.0e-30], [1.0e-300, 1.0e-300]),
        ([1.7e-30], [1.0e-30], [1.0e-250]),
        ([1.0e200], [1.0e30], [1.0e-150]),
        ([2.0e-30], [1.0e-30], [1.0e-320]),
        ([1.0e300, 2.0], [1.0e-30, 1.0], [0.0, 1.0]),
    ],
)
def test_gamma_pearson_sum_preserves_finite_scaled_squared_residuals(
    y: list[float],
    mu: list[float],
    weights: list[float],
) -> None:
    """A square or weighted numerator may overflow although the Pearson sum fits."""
    observed = np.array(y)
    fitted = np.array(mu)
    prior_weights = np.array(weights)
    actual = working_rows.pearson_chi2(
        distribution=Gamma(),
        y=observed,
        mu=fitted,
        sample_weight=prior_weights,
    )

    with localcontext() as context:
        context.prec = 200
        expected = sum(
            _decimal(w) * ((_decimal(value) - _decimal(mean)) / _decimal(mean)) ** 2
            for value, mean, w in zip(y, mu, weights, strict=True)
        )
        _assert_rounding_bound(actual, expected, operations=16 + len(y))


def test_gamma_pearson_frequency_phi_stays_finite_after_extreme_response_scaling() -> None:
    y = np.array([2.0e30, 3.0e30])
    mu = np.array([1.0e30, 2.0e30])
    weights = np.full(2, 1.0e250)
    chi2 = working_rows.pearson_chi2(
        distribution=Gamma(),
        y=y,
        mu=mu,
        sample_weight=weights,
    )
    df = pearson_residual_degrees_of_freedom(weights, 1.0, weight_semantics="frequency")

    _assert_rounding_bound(chi2 / df, Decimal("0.625"), operations=24)


def test_gamma_pearson_true_overflow_remains_nonfinite() -> None:
    """The range repair must not cap an unrepresentable positive statistic."""
    chi2 = working_rows.pearson_chi2(
        distribution=Gamma(),
        y=np.array([1.0e300]),
        mu=np.ones(1),
        sample_weight=np.ones(1),
    )
    assert np.isposinf(chi2)


def test_gamma_fisher_specialization_does_not_override_custom_variance() -> None:
    class DoubleVarianceGamma(Gamma):
        def variance(self, mu):
            return 2.0 * mu**2

    family = DoubleVarianceGamma()
    y = np.array([2.0, 4.0])
    mu = np.array([1.0, 2.0])
    weights = np.array([2.0, 4.0])
    rows = working_rows.coefficient_working_rows(
        distribution=family,
        link=LogLink(),
        y=y,
        mu=mu,
        eta=np.log(mu),
        sample_weight=weights,
        prefer_observed=True,
    )

    np.testing.assert_array_equal(rows.weights, weights / 2.0)
    assert rows.curvature_source == "fisher"
    assert (
        working_rows.pearson_chi2(
            distribution=family,
            y=y,
            mu=mu,
            sample_weight=weights,
        )
        == 3.0
    )


def test_ordinary_non_gamma_pearson_arithmetic_is_preserved() -> None:
    y = np.array([0.25, 2.5, -3.0])
    mu = np.array([0.75, -1.0, 4.0])
    weights = np.array([0.125, 2.0, 3.0])
    expected = float(np.sum(weights * (y - mu) ** 2))
    assert (
        working_rows.pearson_chi2(
            distribution=Gaussian(),
            y=y,
            mu=mu,
            sample_weight=weights,
        )
        == expected
    )
