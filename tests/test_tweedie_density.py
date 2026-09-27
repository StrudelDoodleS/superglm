"""Tweedie density, fitted/null pair, dispersion solver and simulation on the one series."""

import math

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize_scalar
from scipy.special import i1e

from superglm import SuperGLM
from superglm._tweedie import (
    TweedieRows,
    generate_tweedie_cpg,
    solve_log_phi,
    tweedie_logpdf,
    tweedie_logpdf_pair,
    tweedie_unit_deviance,
)
from superglm.distributions import Tweedie
from superglm.features.numeric import Numeric

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


def _reml_scale_criterion(rows, deviance, nullity, u):
    """Q(u) = D e^-u / 2 - l_sat(e^u) - (M / 2)(log 2 pi + u), evaluated directly."""
    saturated = rows.saturated(math.exp(u))[0]
    return 0.5 * deviance * math.exp(-u) - saturated - 0.5 * nullity * (math.log(2 * math.pi) + u)


def test_solve_log_phi_with_a_nullity_is_the_reml_scale_minimiser():
    # Wood (2011) Eq. 4's scale term: the constant (M / 2) log 2 pi moves the
    # criterion, not phi, so only a direct evaluation of Q pins it.
    y, mu = _book(p=1.4, n=2000, seed=5)
    rows = TweedieRows.prepare(y, np.ones_like(y), 1.4)
    deviance, nullity = float(np.sum(tweedie_unit_deviance(y, mu, 1.4))) + 7.0, 12.0
    solved = solve_log_phi(rows, deviance, nullity)
    u = math.log(solved.phi)
    assert solved.criterion == pytest.approx(
        _reml_scale_criterion(rows, deviance, nullity, u), rel=1e-12
    )
    # Q'(u) = -D e^-u / 2 + T(u) - M / 2 vanishes to its round-off.
    score = -0.5 * deviance / solved.phi + rows.saturated(solved.phi)[1] - 0.5 * nullity
    assert abs(score) <= 1e-9 * rows.size


def test_interior_optimum_exists_until_the_nullity_reaches_2n_over_p_minus_1():
    # Q's upper tail slopes as N / (p - 1) - M / 2, so an interior minimum
    # exists exactly while M < 2 N / (p - 1); 3/4 of that limit still has one.
    y, mu = _book(p=1.5, n=400, seed=9)
    rows = TweedieRows.prepare(y, np.ones_like(y), 1.5)
    deviance = float(np.sum(tweedie_unit_deviance(y, mu, 1.5)))
    limit = 2.0 * rows.size / 0.5
    solved = solve_log_phi(rows, deviance, 0.75 * limit)
    score = -0.5 * deviance / solved.phi + rows.saturated(solved.phi)[1] - 0.375 * limit
    assert solved.curvature > 0.0 and abs(score) <= 1e-9 * rows.size
    with pytest.raises(ValueError, match="no finite interior optimum"):
        solve_log_phi(rows, deviance, limit)


def test_frequency_counts_solve_as_the_replicated_rows_at_a_nullity():
    # The likelihood size of counted rows is their total count. At p = 1.5 an
    # interior optimum needs M < 4 N; with every row counted three times,
    # M = 5 N lies inside the replicated book's limit of 12 N but beyond the
    # limit of its N distinct rows.
    y, mu = _book(p=1.5, n=300, seed=4)
    counts = np.full(y.size, 3.0)
    nullity = 5.0 * float(np.count_nonzero(y))
    deviance = float(np.sum(counts * tweedie_unit_deviance(y, mu, 1.5)))
    counted = solve_log_phi(TweedieRows.prepare(y, counts, 1.5, frequency=True), deviance, nullity)
    replicated_rows = TweedieRows.prepare(np.repeat(y, 3), np.ones(3 * y.size), 1.5)
    replicated = solve_log_phi(replicated_rows, deviance, nullity)
    assert counted.phi == pytest.approx(replicated.phi, rel=1e-12)
    assert counted.criterion == pytest.approx(replicated.criterion, rel=1e-12)


@pytest.mark.parametrize("p", [1.2, 1.4, 1.5, 1.8])
def test_near_perfect_tweedie_fit_does_not_fail_in_fit_statistics(p):
    # An exact curve leaves phi at round-off (~1e-26), where every row's series
    # peak index is past the work bound. The fit is published; its likelihood,
    # which the series cannot evaluate there, is NaN with a warning naming why.
    x = np.linspace(-1.0, 1.0, 40)
    y = np.exp(0.3 + 0.5 * x)
    model = SuperGLM(family=Tweedie(p=p), selection_penalty=0, features={"x": Numeric()})
    with pytest.warns(RuntimeWarning, match="series cannot evaluate 40 of 40"):
        model.fit(pd.DataFrame({"x": x}), y)
    assert np.isfinite(model.result.phi) and np.isfinite(model.result.deviance)
    assert np.isnan(model._fit_stats.log_likelihood)
    assert np.isnan(model._fit_stats.null_log_likelihood)
    assert "Log-Likelihood" in str(model.summary())


class _UnitScoreRows(TweedieRows):
    """l_sat(u) = -u, so T = 1 and Q'(u) = 1 - e^-u at D = 2, M = 0: the root is u = 0."""

    def saturated(self, phi):
        return -math.log(phi), 1.0, 0.0


def test_solve_log_phi_stops_on_an_exact_root():
    # Two rows start Newton at log(D / N) = 0, the root itself, where the score is
    # exactly 0.0 and u becomes the bracket's lower end; a bracket test that
    # excludes its ends bisects away from the root instead of stopping there.
    rows = _UnitScoreRows(1.5, 1.0, np.zeros(2), np.zeros(2), np.zeros(2), None)
    solved = solve_log_phi(rows, 2.0)
    assert solved.phi == 1.0 and solved.n_passes == 1


SCORE_NOISE = 1e-4


class _NoisyScoreRows(_UnitScoreRows):
    """T = 1 plus a fixed pseudo-random error in [-SCORE_NOISE, SCORE_NOISE).

    The error is keyed on the bits of u (Knuth's multiplicative hash) and is 1e4
    times the score the 1e-8 step tolerance resolves at curvature e^-u ~ 1. That is
    the state of real rows at peak indices near 1e7: the weighted SCOP fit in
    test_pearson_scale_weights reaches phi ~ 1.5e-7, where T cancels two sums of 3e9
    and carries round-off up to 5e-4 over a curvature of 17.
    """

    def saturated(self, phi):
        bits = int(np.float64(math.log(phi)).view(np.uint64))
        draw = (((bits * 0x9E3779B97F4A7C15) % 2**64) >> 11) / 2.0**52 - 1.0
        value, score, slope = super().saturated(phi)
        return value, score + SCORE_NOISE * draw, slope


def test_solve_log_phi_settles_when_score_round_off_outlasts_the_newton_step():
    rows = _NoisyScoreRows(1.5, 1.0, np.zeros(1), np.zeros(1), np.zeros(1), None)
    solved = solve_log_phi(rows, 2.0)
    # The evaluated score has the true sign wherever |1 - e^-u| > SCORE_NOISE, so
    # a bracket end that ever held a sign lies within SCORE_NOISE (1 + SCORE_NOISE)
    # of the root. The solver stops on a Newton move of at most 1e-8, whose
    # score bounds its start the same way, or on a bracket of width at most 2e-8.
    assert abs(math.log(solved.phi)) <= SCORE_NOISE * (1.0 + SCORE_NOISE) + 2e-8
    assert solved.curvature > 0.0


def test_a_repeated_solve_makes_no_series_pass(monkeypatch):
    # REML re-profiles its accepted line-search point with the same (Dp, Mp).
    y, mu = _book()
    rows = TweedieRows.prepare(y, np.ones_like(y), 1.5)
    deviance = float(np.sum(tweedie_unit_deviance(y, mu, 1.5)))
    first = solve_log_phi(rows, deviance, 2.0)
    monkeypatch.setattr(TweedieRows, "saturated", lambda self, phi: pytest.fail("series pass"))
    assert solve_log_phi(rows, deviance, 2.0) is first


def test_frequency_counts_equal_replicated_rows():
    y, _ = _book(p=1.4, n=300)
    counts = np.random.default_rng(2).integers(1, 4, y.size).astype(float)
    counted = TweedieRows.prepare(y, counts, 1.4, frequency=True).saturated(1.3)
    replicated = TweedieRows.prepare(
        np.repeat(y, counts.astype(int)), np.ones(int(counts.sum())), 1.4
    ).saturated(1.3)
    np.testing.assert_allclose(counted, replicated, rtol=1e-13)
