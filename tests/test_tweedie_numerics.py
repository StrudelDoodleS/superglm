from __future__ import annotations

import numpy as np
import pytest

from superglm.distributions import Tweedie
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
