"""Tweedie density, fitted/null pair, dispersion solver and simulation on the one series."""

import math

import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from scipy.special import i1e

from superglm._tweedie import (
    TweedieRows,
    generate_tweedie_cpg,
    solve_log_phi,
    tweedie_logpdf,
    tweedie_logpdf_pair,
    tweedie_unit_deviance,
)

EPS = np.finfo(np.float64).eps


def _book(p=1.5, phi=2.0, n=4000, seed=11):
    rng = np.random.default_rng(seed)
    mu = np.exp(rng.normal(0.0, 0.5, n))
    return generate_tweedie_cpg(n, mu, phi, p, rng=rng), mu


def test_logpdf_equals_bessel_closed_form_at_p15():
    y, mu = _book()
    y, mu = y[y > 0][:200], mu[y > 0][:200]
    phi = 2.0
    # p = 1.5, w = 1: log f = log(2 / phi) - log(y) / 2 + log I_1(z) - z - d / (2 phi), z = 4 sqrt(y) / phi.
    # The saturated canonical term c(y, y) / phi = -z cancels the e^z of the Bessel
    # function, so the scaled i1e appears alone.
    z = 4.0 * np.sqrt(y) / phi
    deviance = tweedie_unit_deviance(y, mu, 1.5)
    reference = np.log(2.0 / phi) - 0.5 * np.log(y) + np.log(i1e(z)) - deviance / (2 * phi)
    error = np.abs(tweedie_logpdf(y, mu, phi, 1.5) - reference)
    np.testing.assert_array_less(error, 64 * EPS * np.maximum(1, np.abs(reference)))


def test_zero_rows_are_the_exact_atom():
    mu = np.array([0.3, 2.0])
    value = tweedie_logpdf(np.zeros(2), mu, 0.7, 1.3, weights=np.array([1.0, 2.5]))
    np.testing.assert_allclose(value, -np.array([1.0, 2.5]) * mu**0.7 / (0.7 * 0.7), rtol=4 * EPS)


def test_logpdf_pair_null_shares_the_saturated_term():
    y, mu = _book(p=1.3)
    null_mu = np.full_like(mu, y.mean())
    fitted, null = tweedie_logpdf_pair(y, mu, null_mu, 1.7, 1.3)
    np.testing.assert_array_equal(fitted, tweedie_logpdf(y, mu, 1.7, 1.3))
    np.testing.assert_allclose(null, tweedie_logpdf(y, null_mu, 1.7, 1.3), rtol=1e-13, atol=1e-13)


def test_series_refusal_names_power_dispersion_and_rows():
    # At p = 1.5 the peak index is 2 sqrt(y) / phi: 2e10 for the first row, past
    # the work bound inside the exact-integer range (spec section 10), and 2e6
    # for the second, which the series sums (17,000 terms).
    rows = TweedieRows.prepare(np.array([1e8, 1.0]), np.ones(2), 1.5)
    with pytest.raises(FloatingPointError, match=r"1 of 2 positive rows at p=1\.5, phi=1e-06"):
        rows.row_saturated(1e-6)


@pytest.mark.parametrize("p", [1.05, 1.3, 1.5, 1.8, 1.95])
def test_solve_log_phi_is_the_profile_minimiser(p):
    y, mu = _book(p=p)
    rows = TweedieRows.prepare(y, np.ones_like(y), p)
    deviance = float(np.sum(tweedie_unit_deviance(y, mu, p)))
    solved = solve_log_phi(rows, deviance)

    def criterion(u):
        return 0.5 * deviance * math.exp(-u) - rows.saturated(math.exp(u))[0]

    brute = minimize_scalar(
        criterion,
        bounds=(math.log(solved.phi) - 1, math.log(solved.phi) + 1),
        method="bounded",
        options={"xatol": 1e-10},
    )
    assert math.log(solved.phi) == pytest.approx(brute.x, abs=1e-7)
    assert solved.criterion == pytest.approx(criterion(math.log(solved.phi)), rel=1e-12)
    # Score at the root: Q'(u) = -D e^{-u}/2 + T(u); zero up to its evaluation round-off.
    score = -0.5 * deviance / solved.phi + rows.saturated(solved.phi)[1]
    assert abs(score) <= 1e-9 * rows.size
    assert solved.curvature > 0 and solved.n_passes <= 12


def test_solve_log_phi_refuses_no_interior_optimum():
    y = np.zeros(50)
    with pytest.raises(ValueError, match="no finite interior optimum"):
        solve_log_phi(TweedieRows.prepare(y, np.ones(50), 1.5), 3.0)
    y, mu = _book()
    with pytest.raises(ValueError, match="positive finite deviance"):
        solve_log_phi(TweedieRows.prepare(y, np.ones_like(y), 1.5), 0.0)


def test_frequency_counts_equal_replicated_rows():
    y, _ = _book(p=1.4, n=300)
    counts = np.random.default_rng(2).integers(1, 4, y.size).astype(float)
    counted = TweedieRows.prepare(y, counts, 1.4, frequency=True).saturated(1.3)
    replicated = TweedieRows.prepare(
        np.repeat(y, counts.astype(int)), np.ones(int(counts.sum())), 1.4
    ).saturated(1.3)
    np.testing.assert_allclose(counted, replicated, rtol=1e-13)
