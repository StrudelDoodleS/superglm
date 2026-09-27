from __future__ import annotations

import numpy as np
import pytest
from scipy.special import gammaln

import superglm._tweedie as density_module
from superglm import tweedie_logpdf
from superglm._tweedie_series import series_moments
from superglm.distributions import Tweedie
from superglm.links import LogLink
from superglm.model.fit_ops import _compute_fit_stats


def _log_t_with_series_mode(a: float, mode: int) -> float:
    return float(np.log(mode + 1.0) + gammaln(a * (mode + 1.0)) - gammaln(a * mode))


def test_exact_series_starts_near_distant_mode() -> None:
    exact, log_sum, expected_j, variance_j = series_moments(
        np.array([_log_t_with_series_mode(1.5, 90_000)]),
        1.5,
    )

    assert exact.tolist() == [True]
    assert np.isfinite(log_sum[0])
    assert expected_j[0] == pytest.approx(90_000.7, rel=2.0e-9)
    assert variance_j[0] > 0.0


@pytest.mark.parametrize("p", [1.000001, 1.01, 1.5, 1.99, 1.999999])
def test_unit_deviance_is_exactly_zero_when_response_equals_extreme_mean(p: float) -> None:
    values = np.array([1.0e-20, 1.0, 1.0e12])

    actual = Tweedie(p).deviance_unit(values, values)

    np.testing.assert_array_equal(actual, np.zeros_like(values))


def test_fit_stats_pearson_is_zero_for_equal_subnormal_tweedie_values() -> None:
    value = np.array([1.0e-300])

    stats = _compute_fit_stats(
        value,
        value,
        np.ones(1),
        None,
        Tweedie(1.5),
        LogLink(),
        1.0,
        null_mu=value,
        weight_semantics="prior",
    )

    assert stats.pearson_chi2 == 0.0


@pytest.mark.parametrize(
    ("y", "mu", "phi", "p", "weight", "expected"),
    [
        (0.017, 0.02, 0.004, 1.0001, 0.4, -242.08168865838033),
        (9000.0, 10000.0, 200.0, 1.01, 0.5, -8.636743836168788),
        (0.22, 0.15, 0.125, 1.25, 4.0, 1.0417074672233964),
        (0.03, 4.0, 0.7, 1.5, 0.2, -2.2657661799346407),
        (80.0, 50.0, 5.3, 1.75, 4.0, -5.206967741958142),
        (0.0002, 0.001, 1.3, 1.99, 0.1, 5.575914834713504),
        (
            0.04564326798684731,
            2.859891821890267,
            0.10602153698295053,
            1.05,
            1.0,
            -25.217701008861372,
        ),
    ],
)
def test_public_density_matches_neutral_high_precision_reference(
    y: float,
    mu: float,
    phi: float,
    p: float,
    weight: float,
    expected: float,
) -> None:
    actual = tweedie_logpdf(
        np.array([y]),
        np.array([mu]),
        phi,
        p,
        weights=np.array([weight]),
    )

    assert actual[0] == pytest.approx(expected, rel=0.0, abs=2.5e-9)


def test_tweedie_fit_stats_reuses_one_density_normalizer(monkeypatch) -> None:
    y = np.array([0.0, 0.3, 1.2, 4.5])
    mu = np.array([0.2, 0.5, 1.5, 3.7])
    null_mu = np.full_like(y, 1.1)
    weights = np.array([0.4, 0.8, 1.2, 1.8])
    family = Tweedie(1.55)
    expected_ll = float(np.sum(tweedie_logpdf(y, mu, 0.8, 1.55, weights=weights)))
    expected_null_ll = float(np.sum(tweedie_logpdf(y, null_mu, 0.8, 1.55, weights=weights)))
    real_series = density_module.series_moments
    calls = 0

    def counted(log_t, a):
        nonlocal calls
        calls += 1
        return real_series(log_t, a)

    monkeypatch.setattr(density_module, "series_moments", counted)

    stats = _compute_fit_stats(
        y,
        mu,
        weights,
        None,
        family,
        LogLink(),
        0.8,
        null_mu=null_mu,
        weight_semantics="prior",
    )

    assert calls == 1
    assert stats.log_likelihood == pytest.approx(expected_ll, abs=1.0e-11)
    assert stats.null_log_likelihood == pytest.approx(expected_null_ll, abs=1.0e-11)
