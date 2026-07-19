from __future__ import annotations

import numpy as np
import pytest
from scipy.special import ive

import superglm.profiling.tweedie as tweedie_module
from superglm.distributions import Tweedie
from superglm.links import LogLink
from superglm.model.fit_ops import _compute_fit_stats
from superglm.profiling.tweedie import (
    _evaluate_tweedie_density,
    _prepare_tweedie_density,
    _tweedie_logpdf_impl,
    estimate_phi,
    tweedie_logpdf,
)


@pytest.mark.parametrize("p", [1.000001, 1.01, 1.5, 1.99, 1.999999])
def test_unit_deviance_is_exactly_zero_when_response_equals_extreme_mean(p: float) -> None:
    values = np.array([1.0e-20, 1.0, 1.0e12])

    actual = Tweedie(p).deviance_unit(values, values)

    np.testing.assert_array_equal(actual, np.zeros_like(values))


def test_pearson_phi_preserves_valid_tiny_means() -> None:
    y = np.array([1.0e-12, 2.0e-12])
    mu = np.array([1.0e-20, 2.0e-20])
    p = 1.5
    expected = float(np.mean((y - mu) ** 2 / mu**p))

    actual = estimate_phi(y, mu, p)

    assert actual == pytest.approx(expected, rel=2.0e-15)


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


def test_default_density_uses_exact_series_instead_of_saddlepoint() -> None:
    actual, diagnostics = _tweedie_logpdf_impl(
        np.array([0.04564326798684731]),
        np.array([2.859891821890267]),
        0.10602153698295053,
        1.05,
    )

    assert actual[0] == pytest.approx(-25.217701008861372, abs=2.5e-9)
    assert diagnostics.n_series == 1
    assert diagnostics.n_saddlepoint == 0


def test_half_power_density_uses_stable_exact_bessel_form_at_tiny_phi() -> None:
    y = np.array([1.35])
    mu = np.array([1.35001])
    weights = np.array([4.0])
    phi = 4.0e-8
    root_y = np.sqrt(y)
    root_mu = np.sqrt(mu)
    bessel_argument = 4.0 * weights * root_y / phi
    expected = (
        np.log(2.0 * weights / (phi * root_y))
        + np.log(ive(1, bessel_argument))
        - 2.0 * weights * np.square(root_y - root_mu) / (phi * root_mu)
    )

    actual, diagnostics = _tweedie_logpdf_impl(
        y,
        mu,
        phi,
        1.5,
        weights=weights,
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-13)
    assert diagnostics.n_series == 1
    assert diagnostics.n_saddlepoint == 0


def test_exact_series_log_phi_score_matches_finite_difference() -> None:
    y = np.array([0.04564326798684731, 9000.0])
    mu = np.array([2.859891821890267, 10000.0])
    weights = np.array([1.0, 0.5])
    p = 1.05
    phi = 0.10602153698295053
    prepared = _prepare_tweedie_density(y, mu, p, weights=weights)

    evaluated = _evaluate_tweedie_density(prepared, phi, compute_score=True)
    step = 1.0e-5
    upper = _evaluate_tweedie_density(prepared, phi * np.exp(step)).logpdf
    lower = _evaluate_tweedie_density(prepared, phi * np.exp(-step)).logpdf
    finite_difference = -float(np.mean(upper - lower)) / (2.0 * step)

    assert evaluated.diagnostics.n_series == 2
    assert evaluated.score_valid
    assert evaluated.log_phi_score is not None
    assert float(np.mean(evaluated.log_phi_score)) == pytest.approx(
        finite_difference,
        rel=2.0e-6,
        abs=2.0e-7,
    )


def test_tweedie_fit_stats_reuses_one_density_normalizer(monkeypatch) -> None:
    y = np.array([0.0, 0.3, 1.2, 4.5])
    mu = np.array([0.2, 0.5, 1.5, 3.7])
    null_mu = np.full_like(y, 1.1)
    weights = np.array([0.4, 0.8, 1.2, 1.8])
    family = Tweedie(1.55)
    expected_ll = float(np.sum(tweedie_logpdf(y, mu, 0.8, 1.55, weights=weights)))
    expected_null_ll = float(np.sum(tweedie_logpdf(y, null_mu, 0.8, 1.55, weights=weights)))
    real_evaluate = tweedie_module._evaluate_tweedie_density
    calls = 0

    def counted(prepared, phi, *, compute_score=False):
        nonlocal calls
        calls += 1
        return real_evaluate(prepared, phi, compute_score=compute_score)

    monkeypatch.setattr(tweedie_module, "_evaluate_tweedie_density", counted)

    stats = _compute_fit_stats(
        y,
        mu,
        weights,
        None,
        family,
        LogLink(),
        0.8,
        null_mu=null_mu,
    )

    assert calls == 1
    assert stats.log_likelihood == pytest.approx(expected_ll, abs=1.0e-11)
    assert stats.null_log_likelihood == pytest.approx(expected_null_ll, abs=1.0e-11)
