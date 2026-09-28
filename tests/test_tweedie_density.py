"""Tweedie density, fitted/null pair, dispersion solver and simulation on the one series."""

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize_scalar
from scipy.special import i1e

import superglm._tweedie as density_module
from superglm import SuperGLM
from superglm._tweedie import (
    TweedieRows,
    _corrected_saddlepoint,
    generate_tweedie_cpg,
    saddlepoint_switch,
    solve_log_phi,
    tweedie_logpdf,
    tweedie_logpdf_pair,
    tweedie_unit_deviance,
)
from superglm._tweedie_series import series_moments
from superglm.distributions import Tweedie
from superglm.features.numeric import Numeric
from superglm.reml.scale import prepare_tweedie_reml_scale_data, profile_tweedie_reml_scale

EPS = np.finfo(np.float64).eps
ORACLE = json.loads((Path(__file__).parent / "fixtures" / "tweedie_series_oracle.json").read_text())


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


@pytest.mark.parametrize(
    "y, mu, phi, weight, expected",
    [
        # Past the switch with zero deviance: the saddlepoint
        # -(1/2) log(2 pi (phi / w) y^p), its corrections below 1e-75.
        (1.0, 1.0, 1e-308, 1.0, -0.5 * (math.log(2 * math.pi) + math.log(1e-308))),
        (1e150, 1e150, 1e250, 1e250, -0.5 * (math.log(2 * math.pi) + 1.5 * math.log(1e150))),
        # The atom -w mu^(2-p) / (phi (2 - p)).
        (0.0, 1.0, 1e308, 1e308, -2.0),
    ],
)
def test_log_density_is_finite_where_its_intermediate_products_overflow(
    y, mu, phi, weight, expected
):
    # c w, c w / phi, w d and 2 phi overflow here; the density depends on phi / w.
    value = tweedie_logpdf(np.array([y]), np.array([mu]), phi, 1.5, weights=np.array([weight]))
    # The result comes from logs of magnitude up to |log phi| + |log w| + p |log y|.
    scale = abs(math.log(phi)) + abs(math.log(weight)) + 1.5 * abs(math.log(y or 1.0))
    assert value[0] == pytest.approx(expected, abs=8 * EPS * scale)


def test_a_summed_row_whose_canonical_product_overflows_keeps_its_density():
    # At p = 1.5, y = 1 and phi / w = 1 the peak index is 2, a summed row. With
    # w = phi = 1e308, c w = -4e308 overflows, so c w / phi = -4 comes from log t.
    base = tweedie_logpdf(np.ones(1), np.ones(1), 1.0, 1.5)
    scaled = tweedie_logpdf(np.ones(1), np.ones(1), 1e308, 1.5, weights=np.array([1e308]))
    # log t gains (a + 1)(log w - log phi) = 2 (709 - 709), each log rounded to
    # eps. l_sat moves by at most E J + |c w / phi| = 2 + 4 per unit of log t.
    assert scaled[0] == pytest.approx(base[0], abs=2 * 2 * EPS * math.log(1e308) * (2 + 4))


def test_saddlepoint_rows_take_the_quotient_where_it_is_a_normal_number():
    # w = 1e290 and phi = 1e280 put the row past the switch with |c w / phi| = 4e10,
    # a normal number. From log t, log |c w / phi| carries the rounding of
    # (a + 1)(log w - log phi), 268 eps in l_sat here.
    got = TweedieRows.prepare(np.array([3.0]), np.array([1e290]), 1.5).row_saturated(1e280)[0][0]
    # The corrected saddlepoint at these float inputs, to 50 digits (mpmath).
    reference = 9.7700277152590607655
    # The quotient route rounds a product, a quotient and a few logs of order-one terms.
    assert got == pytest.approx(reference, abs=8 * EPS * (1.0 + reference))


@pytest.mark.parametrize("p", [1.2, 1.5, 1.8])
def test_log_density_is_finite_where_the_unit_deviance_overflows(p):
    # At y / mu = 1e616, d = 2 y mu^(1-p) / (p - 1) (1 - B/A + C/A) overflows, with
    # B/A = (mu / y)^(p-1) / (2 - p) and C/A below 1e-100. The weight brings w d / 2
    # back to between e^143 and e^567, where |l_sat| is far below its rounding.
    y, mu, w = 1e308, 1e-308, 1e-308
    value = tweedie_logpdf(np.array([y]), np.array([mu]), 1.0, p, weights=np.array([w]))
    expected = -math.exp(math.log(w) + math.log(y) + (1 - p) * math.log(mu) - math.log(p - 1))
    # It is the exponential of a sum of logs of magnitude up to |log w| + |log y| + |log mu|.
    scale = abs(math.log(w)) + abs(math.log(y)) + abs(math.log(mu))
    assert value[0] == pytest.approx(expected, rel=4 * EPS * scale)


def test_the_ml_dispersion_admits_a_row_whose_unit_deviance_overflows():
    # The same row's w d = 4e154 is finite although d is not.
    from superglm.profiling.tweedie import profile_phi_at

    y, mu = _book(p=1.5, n=200, seed=3)
    y, mu = np.append(y, 1e308), np.append(mu, 1e-308)
    weights = np.append(np.ones(200), 1e-308)
    assert math.isfinite(profile_phi_at(y, mu, weights, 1.5).phi)


@pytest.mark.parametrize("p", [1.2, 1.5, 1.8, 1.95])
def test_unit_deviance_is_accurate_below_half_the_mean(p):
    """0 < y < mu/2: y - mu is inexact there, so the log ratio must not come from 1 + delta.

    The old log1p((y - mu) / mu) route carried a relative error of order
    eps * mu / y -- over 1e-6 of a unit deviance at y = 1e-9 mu, p = 1.8 --
    enough noise to stall a PIRLS line search short of a certifiable mode.
    """
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 50
    # Means from 0.1 through 1e6 to 1e100, where log y - log mu would err by
    # ulps of |log mu| ~ 230 and break the bound below, and a quotient y / mu
    # that underflows to a subnormal.
    mu = np.append(np.geomspace(0.1, 1e100, 80), 1e10)
    y = np.append(np.geomspace(1e-12, 0.45, 80) * mu[:-1], 1e-300)
    got = tweedie_unit_deviance(y, mu, p)
    for value, response, mean in zip(got, y, mu, strict=True):
        Y, M, P = mp.mpf(response), mp.mpf(mean), mp.mpf(p)
        exact = 2 * (
            Y ** (2 - P) / ((1 - P) * (2 - P)) - Y * M ** (1 - P) / (1 - P) + M ** (2 - P) / (2 - P)
        )
        # In units of u = eps / 2: a normal y / mu rounds once and its log adds
        # 2u |L|, L = log(y / mu); log(y) - log(mu) is within 3u (|log y| + |log mu|).
        # Each factor of g has log-sensitivity at most 2 - p + 1/log 2 < 2.45 to L,
        # the products (2 - p) L and (p - 1) L round by u |L|, and g = first - second
        # cancels by at most 6.2, around 13 roundings besides.
        log_ratio = abs(math.log(response) - math.log(mean))
        if response / mean >= np.finfo(np.float64).tiny:
            log_error = 1.0 + 2.0 * log_ratio
        else:
            log_error = 3.0 * (abs(math.log(response)) + abs(math.log(mean)))
        bound = (EPS / 2) * (6.2 * (2.45 * log_error + log_ratio) + 41.0)
        assert abs(float(mp.mpf(value) / exact - 1)) <= bound


def test_logpdf_pair_null_shares_the_saturated_term():
    y, mu = _book(p=1.3)
    null_mu = np.full_like(mu, y.mean())
    fitted, null = tweedie_logpdf_pair(y, mu, null_mu, 1.7, 1.3)
    # Both routes evaluate the same saturated rows and the same deviance on the
    # same arrays, so they agree bitwise.
    np.testing.assert_array_equal(fitted, tweedie_logpdf(y, mu, 1.7, 1.3))
    np.testing.assert_array_equal(null, tweedie_logpdf(y, null_mu, 1.7, 1.3))


def _saddlepoint_saturated(y, w, phi, p):
    """Small-dispersion (saddlepoint) saturated log density of a positive row (Jorgensen 1997)."""
    return -0.5 * (math.log(2 * math.pi) + math.log(phi) - np.log(w) + p * np.log(y))


def _series_bounds(p, y, w, phi, peak, variance):
    """Float64 bounds on the series' l_sat, T and T' at one row, to first order.

    Each term's log q(j) = j log t - lgamma(j + 1) - lgamma(a j) rounds within
    delta = 16 eps of its parts' magnitudes at the peak (test_tweedie_series), and
    log W inherits at most delta. The weights exp(q - peak) err by at most 2 delta
    relative, which moves E[J] by at most 2 delta sqrt(Var J) and Var J by at most
    4 delta Var J (Cauchy-Schwarz; E|(J - E J)^2 - Var J| <= 2 Var J); T and T'
    scale those by a + 1 and (a + 1)^2. Each of the three adds two parts of size
    (a + 1) j, the canonical term c w / phi = -(a + 1) j among them, at 16 eps.
    """
    a = (2 - p) / (p - 1)
    log_t = a * (math.log(y) - math.log(p - 1)) - math.log(2 - p)
    log_t += (a + 1) * (math.log(w) - math.log(phi))
    delta = 16 * EPS * (abs(peak * log_t) + math.lgamma(peak + 1) + abs(math.lgamma(a * peak)))
    parts = 16 * EPS * 2 * (a + 1) * peak
    return (
        delta + parts + 16 * EPS * abs(math.log(y)),
        2 * (a + 1) * delta * math.sqrt(variance) + parts,
        4 * (a + 1) ** 2 * delta * variance + parts,
    )


def _saddle_bounds(p, y, peak, value):
    """Bounds on the corrected saddlepoint's l_sat, T and T' at one row.

    Its remainder is B3 e^3 + B4 e^4 + ..., e = 1/((2 - p) j), with |B3| <= 1/360
    (Stirling's value at p -> 1 and 2, smaller between) and |B4| under a third of
    that on the oracle. e is proportional to phi, so the remainder's first and
    second log-phi derivatives weight e^k by k and k^2. The sum over integer j
    adds the lattice term 2 exp(-2 pi^2 (p - 1) j) (Poisson summation with
    Var J ~ (p - 1) j), each log-phi derivative of it at most 2 pi j larger. The
    float64 formula rounds within 8 eps of its logs.
    """
    e = 1 / ((2 - p) * peak)
    lattice = 2 * math.exp(-2 * math.pi**2 * (p - 1) * peak)
    return (
        (e**3 + e**4) / 360 + lattice + 8 * EPS * (1 + abs(value) + abs(math.log(y))),
        (3 * e**3 + 4 * e**4) / 360 + 2 * math.pi * peak * lattice + 8 * EPS,
        (9 * e**3 + 16 * e**4) / 360 + (2 * math.pi * peak) ** 2 * lattice + 8 * EPS * e,
    )


@pytest.mark.parametrize(
    "row", ORACLE["saturated"], ids=lambda r: f"p{r['p']}-j{r['peak_index']:.0e}"
)
def test_saturated_rows_match_the_50_digit_oracle_on_both_sides_of_the_switch(row):
    """l_sat, T and T' against 50 digits at peak indices 1e2 to 1e7, astride every switch.

    Past the switch the bound is the corrected saddlepoint's remainder, about
    e^3 / 360. The uncorrected saddlepoint misses by |A| e = p (3 - p) / (24 (2 - p) j),
    and a wrong sign or constant in A misses by twice that, far above the bound at
    every row past the switch. Below it the bound is the series' own float64 error.
    """
    _assert_within_the_arm_bound(row, row["peak_index"] > saddlepoint_switch(row["p"]))


@pytest.mark.parametrize(
    "row", ORACLE["switch"], ids=lambda r: f"p{r['p']}-{r['arm']}-j{r['peak_index']:.0f}"
)
def test_each_side_of_the_switch_meets_the_bound_of_its_more_accurate_arm(row):
    """A quarter of and four times the switch as generated: the row meets that side's arm's bound.

    The test above takes its bound from the code's own switch, so it would
    follow a switch moved anywhere; these rows are fixed numbers. A factor 4
    either side of the balance point moves the ratio of the arms' errors by
    4^4. At a quarter of the switch the saddlepoint's remainder is outside the
    series bound (6.4e-10 against 2.8e-11 in l_sat at p = 1.5); below p = 1.007,
    where the switch is the lattice floor, its lattice term 2 exp(-2 pi^2 Var J)
    is (1.5e-4 at p = 1.001). At four times it the series' float64 error is
    outside the saddlepoint's bound (1.5e-6 against 2.2e-14 in T at p = 1.001).
    A switch moved past either row, or a dropped floor, fails here.
    """
    _assert_within_the_arm_bound(row, row["arm"] == "saddlepoint")


def _assert_within_the_arm_bound(row, saddle):
    """l_sat, T and T' of one oracle row within the float64 bound of the named arm."""
    p, y, w, phi, peak = (row[k] for k in ("p", "y", "w", "phi", "peak_index"))
    reference = [float(row[k]) for k in ("l_sat", "score", "slope")]
    got = TweedieRows.prepare(np.array([y]), np.array([w]), p).row_saturated(phi)
    if saddle:
        bounds = _saddle_bounds(p, y, peak, reference[0])
    else:
        # T' = -(a + 1)^2 Var J + (a + 1) j, and 1 / (a + 1) = p - 1.
        variance = (peak / (p - 1) - reference[2]) * (p - 1) ** 2
        bounds = _series_bounds(p, y, w, phi, peak, variance)
    for value, exact, bound in zip(got, reference, bounds, strict=True):
        assert abs(value[0] - exact) <= bound


@pytest.mark.parametrize("p", [1.001, 1.002, 1.005, 1.01, 1.05, 1.2, 1.5, 1.8, 1.95, 1.99])
def test_the_switch_is_continuous_within_both_bounds(p):
    """At the switch the series and the corrected saddlepoint differ by at most their two bounds."""
    y, w = 1.7, 2.5
    peak = saddlepoint_switch(p)
    phi = w * y ** (2 - p) / ((2 - p) * peak)
    rows = TweedieRows.prepare(np.array([y]), np.array([w]), p)
    log_t = rows.log_t_unit_phi - (rows.a + 1) * math.log(phi)
    _, log_w, mean_j, var_j = series_moments(log_t, rows.a)
    canonical = rows.saturated_canonical / phi
    series = (
        log_w - rows.log_y + canonical,
        (rows.a + 1) * mean_j + canonical,
        -((rows.a + 1) ** 2) * var_j - canonical,
    )
    saddle = _corrected_saddlepoint(p, np.log(-canonical), rows.log_y)
    bounds = np.add(
        _series_bounds(p, y, w, phi, peak, var_j[0]), _saddle_bounds(p, y, peak, saddle[0][0])
    )
    for summed, closed_form, bound in zip(series, saddle, bounds, strict=True):
        assert abs(summed[0] - closed_form[0]) <= bound
    # A relative 1e-9 either side of it routes the row to each arm.
    around = log_t[0] + (rows.a + 1) * np.array([-1e-9, 1e-9])
    assert series_moments(around, rows.a, max_mode=peak)[0].tolist() == [True, False]


def test_rows_past_the_switch_take_the_corrected_saddlepoint():
    # At p = 1.5 the peak index is 2 w sqrt(y) / phi: 4e10, 2e24 (past 2**52) and
    # 2e6, all past the switch near 1.3e3, so no series term is summed.
    y, w, phi, p = np.array([1e8, 1e36, 1.0]), np.array([2.0, 1.0, 1.0]), 1e-6, 1.5
    value, score, slope = TweedieRows.prepare(y, w, p).row_saturated(phi)
    e = phi * y ** (p - 2) / w  # 1 / ((2 - p) j)
    first = p * (p - 3) / 24 * e
    second = -p * (p - 1) * (p - 2) * (p - 3) / 48 * e**2
    expected = _saddlepoint_saturated(y, w, phi, p) + first + second
    # Rearranged from the same four logs: a few ulps of their magnitudes. e is
    # formed through about eight roundings on each side.
    rounding = 8 * EPS * (math.log(2 * math.pi) - math.log(phi) + np.log(w) + p * np.log(y))
    np.testing.assert_array_less(np.abs(value - expected), rounding)
    np.testing.assert_allclose(score, 0.5 - first - 2 * second, rtol=16 * EPS, atol=0)
    np.testing.assert_allclose(slope, -first - 4 * second, rtol=16 * EPS, atol=0)


@pytest.mark.parametrize("via", ["solve", "reml"])
@pytest.mark.parametrize("nullity", [0.0, 3.0, 30.0])
def test_a_round_off_exact_fit_profiles_to_the_saddlepoint_root(nullity, via):
    """D / N = 1e-26 puts every peak index past 2**52, where the root is log(D / (N - M)).

    T = N / 2 + O(e) there with e = 1 / ((2 - p) j) ~ 1e-26, so the root of
    Q' = T - M / 2 - D e^-u / 2 is log(D / (N - M)) to 1e-25, and the solve stops
    within its step tolerance of it. A fixed +-45 window instead stopped on its own
    unevaluated limit, log phi = -45.
    """
    y = np.exp(0.3 + 0.5 * np.linspace(-1.0, 1.0, 40))
    rows = TweedieRows.prepare(y, np.ones_like(y), 1.5)
    deviance = 1e-26 * rows.size
    if via == "solve":
        phi = solve_log_phi(rows, deviance, nullity).phi
    else:
        data = prepare_tweedie_reml_scale_data(y, np.ones_like(y), 1.5, weight_semantics="prior")
        phi = profile_tweedie_reml_scale(data, deviance, nullity).phi
    expected = math.log(deviance / (rows.size - nullity))
    bound = density_module._NEWTON_STEP_TOL + 4 * EPS * abs(expected)
    assert abs(math.log(phi) - expected) <= bound


def _assert_root_within_the_step_tolerance(rows, deviance, nullity, solved):
    """The stop rule leaves the root of Q' within _NEWTON_STEP_TOL of log phi.

    A Newton stop is a move of at most the tolerance, whose residual is second
    order in it; a bisection stop is the midpoint of a sign-change bracket of
    width at most twice it. The score at u -+ 2 tol then has sign -/+ with a
    margin of at least Q'' tol, far above its evaluation round-off: a few eps
    (a + 1) j log j per row (_SERIES_ERROR's measurement) at order-one j.
    """
    u, tol = math.log(solved.phi), density_module._NEWTON_STEP_TOL

    def score(v):
        return -0.5 * deviance * math.exp(-v) + rows.saturated(math.exp(v))[1] - 0.5 * nullity

    assert score(u - 2 * tol) < 0.0 < score(u + 2 * tol)


def _criterion_round_off(rows, deviance, nullity, phi):
    """Q rounds to 32 eps of its terms' magnitudes at phi.

    The terms are log W, log y and c w / phi of summed rows, the logs of
    saddlepoint rows, D / (2 phi) and the nullity term.
    """
    u = math.log(phi)
    log_t = rows.log_t_unit_phi - (rows.a + 1.0) * u
    summed = series_moments(log_t, rows.a, max_mode=saddlepoint_switch(rows.p))[0]
    canonical = np.where(summed, rows.saturated_canonical / phi, 0.0)
    log_w = rows.row_saturated(phi)[0] + rows.log_y - canonical
    count = np.ones_like(rows.log_y) if rows.count is None else rows.count
    magnitudes = float(count @ (np.abs(log_w) + np.abs(rows.log_y) + np.abs(canonical)))
    magnitudes += 0.5 * deviance / phi + 0.5 * nullity * abs(math.log(2 * math.pi) + u)
    return 32.0 * EPS * magnitudes


def _assert_carried_criterion(rows, deviance, nullity, solved, direct):
    """The solver's Q against a direct evaluation at its phi.

    The solver carries l_sat to first order over its last move of at most the
    step tolerance, leaving |T'| tol^2 / 2 with |T'| <= Q'' + D / (2 phi); both
    values carry the round-off of Q at points that close together.
    """
    tol = density_module._NEWTON_STEP_TOL
    remainder = (solved.curvature + 0.5 * deviance / solved.phi) * tol**2
    round_off = 2.0 * _criterion_round_off(rows, deviance, nullity, solved.phi)
    assert abs(solved.criterion - direct) <= remainder + round_off


def _assert_value_minimiser(rows, deviance, nullity, solved, brute, xatol):
    """Placement against a value-only bounded Brent search of Q, and its cost.

    Q rounds to _criterion_round_off, so a value-only search cannot place a
    minimum of curvature Q'' more finely than sqrt(2 round-off / Q''); fminbound
    (Brent 1973) stops once its bracket is within 2 (sqrt(eps) |x| + xatol / 3),
    and Newton on a step of the tolerance. Each Newton pass and each reference
    value costs one series pass.
    """
    u = math.log(solved.phi)
    round_off = _criterion_round_off(rows, deviance, nullity, solved.phi)
    du = (
        math.sqrt(2.0 * round_off / solved.curvature)
        + 2.0 * (math.sqrt(EPS) * abs(brute.x) + xatol / 3.0)
        + density_module._NEWTON_STEP_TOL
    )
    assert abs(u - brute.x) <= du
    assert solved.n_passes < brute.nfev


def test_solve_log_phi_converges_with_rows_past_the_work_bound():
    # Two exactly fitted rows so large that at phi ~ 2 their peak indices, sqrt(y),
    # are 1e15 (past the term bound) and 1e20 (past 2**52).
    y, mu = _book(p=1.5)
    y, mu = np.append(y, [1e30, 1e40]), np.append(mu, [1e30, 1e40])
    rows = TweedieRows.prepare(y, np.ones_like(y), 1.5)
    deviance = float(np.sum(tweedie_unit_deviance(y, mu, 1.5)))
    solved = solve_log_phi(rows, deviance)

    def criterion(u):
        return 0.5 * deviance * math.exp(-u) - rows.saturated(math.exp(u))[0]

    u = math.log(solved.phi)
    brute = minimize_scalar(
        criterion, bounds=(u - 1, u + 1), method="bounded", options={"xatol": 1e-10}
    )
    _assert_value_minimiser(rows, deviance, 0.0, solved, brute, 1e-10)
    _assert_carried_criterion(rows, deviance, 0.0, solved, criterion(u))
    _assert_root_within_the_step_tolerance(rows, deviance, 0.0, solved)


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
    _assert_value_minimiser(rows, deviance, 0.0, solved, brute, 1e-10)
    _assert_carried_criterion(rows, deviance, 0.0, solved, criterion(math.log(solved.phi)))
    _assert_root_within_the_step_tolerance(rows, deviance, 0.0, solved)
    assert solved.curvature > 0


def _near_one_book(p, n, seed, *, heavy):
    """Draws near the Poisson lattice; heavy books spread weights and means over decades."""
    rng = np.random.default_rng(seed)
    spread = 2.0 if heavy else 0.5
    mu = np.exp(rng.normal(0.0 if heavy else 1.0, spread, n))
    phi = float(np.exp(rng.normal(0.0, 1.5))) if heavy else 2.0
    y = generate_tweedie_cpg(n, mu=mu, phi=phi, p=p, rng=rng)
    w = np.exp(rng.normal(0.0, 1.5, n)) if heavy else rng.uniform(0.5, 2.0, n)
    return y, mu, w


def test_newton_bisects_when_a_proposal_lands_on_a_bracket_end():
    # At p = 1.018 the start's Newton step clips to -2 onto a point of negative
    # curvature, whose 2-unit walk lands exactly on the start: a bracket test
    # that admitted its ends took both moves and cycled to the 60-step limit.
    y, mu, w = _near_one_book(1.018, 2000, 1, heavy=False)
    rows = TweedieRows.prepare(y, w, 1.018)
    deviance = float(np.sum(w * tweedie_unit_deviance(y, mu, 1.018)))
    solved = solve_log_phi(rows, deviance)
    _assert_root_within_the_step_tolerance(rows, deviance, 0.0, solved)


@pytest.mark.parametrize("nullity", [0.0, 2.0])
@pytest.mark.parametrize("p", [1.004, 1.02, 1.08, 1.2])
def test_solve_log_phi_is_the_global_minimum_near_the_poisson_lattice(p, nullity):
    """No point of a dense grid lies below the solution on small heavy-weight books.

    The grid minimum bounds the global one from above, so the check is one-sided:
    the solution may sit below it, never above by more than round-off.
    """
    for seed in range(4):
        y, mu, w = _near_one_book(p, 12, seed, heavy=True)
        fitted = mu * np.exp(np.random.default_rng(seed).normal(0.0, 0.3, y.size))
        deviance = float(np.sum(w * tweedie_unit_deviance(y, fitted, p)))
        rows = TweedieRows.prepare(y, w, p)
        _assert_no_grid_point_below(rows, deviance, nullity, solve_log_phi(rows, deviance, nullity))


def _assert_no_grid_point_below(rows, deviance, nullity, solved, points=2001):
    """No point of a dense grid over log phi +- 8 lies below the solution.

    The grid minimum bounds the global one from above, so the check is one-sided:
    the solution may sit below it, never above by more than round-off.
    """
    grid = math.log(solved.phi) + np.linspace(-8.0, 8.0, points)
    values = [_reml_scale_criterion(rows, deviance, nullity, u) for u in grid]
    lowest = math.exp(grid[int(np.argmin(values))])
    tol = density_module._NEWTON_STEP_TOL
    slack = (solved.curvature + 0.5 * deviance / solved.phi) * tol**2
    slack += 2.0 * _criterion_round_off(rows, deviance, nullity, solved.phi)
    slack += _criterion_round_off(rows, deviance, nullity, lowest)
    assert solved.criterion <= min(values) + slack


def test_a_minimum_on_the_one_jump_edge_is_found():
    # One positive row at p = 1.0056 with a REML nullity of 2. Once its peak index
    # falls below 1, E J = 1 holds exactly, so Q' = 0 at the window's upper edge
    # itself, and Q is lowest there: 1.6 below the Newton root.
    p = 1.0055923284422708
    y = np.array([0.0, 2.611596628644268, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    mu = np.array(
        [0.6242569258595899, 1.9665878664058265, 0.1879213977573198, 0.4585709804405651]
        + [0.2529772244620755, 0.048495461291874105, 0.6629478433791297, 0.8198142938793933]
    )
    weights = np.array(
        [1.4721287059185717, 1.3481089957640897, 3.205294003910986, 1.8228471642283448]
        + [0.525550697600261, 0.992372885967053, 0.8713436230186044, 0.08242028595594196]
    )
    deviance = float(np.sum(weights * tweedie_unit_deviance(y, mu, p)))
    rows = TweedieRows.prepare(y, weights, p)
    _assert_no_grid_point_below(rows, deviance, 2.0, solve_log_phi(rows, deviance, 2.0), 8001)


def test_the_scan_samples_each_lattice_period_four_times():
    # Four rows tiled 50 times at p = 1.022: one sample per lattice period 1 / j
    # steps over the global minimum (by 0.15 in Q); four per period do not.
    p = 1.022323821143762
    y = np.tile([1.3282268467711957, 1.647160781221533, 0.35026059430826734, 319.9989853303057], 50)
    mu = np.tile(
        [0.8699897850055411, 1.3471297746888966, 0.7736905022225017, 306.4179282087071], 50
    )
    weights = np.tile(
        [0.0768496996044912, 0.7755868918100032, 5.967696664624229, 0.4575881655301412], 50
    )
    deviance = float(np.sum(weights * tweedie_unit_deviance(y, mu, p)))
    rows = TweedieRows.prepare(y, weights, p)
    _assert_no_grid_point_below(rows, deviance, 0.0, solve_log_phi(rows, deviance), 8001)


def test_the_lattice_comparison_makes_no_pass_where_no_row_bends_the_profile():
    # About 3000 positive rows at p = 1.05: a row's lattice term bends Q by at most
    # 8 pi^2 j^2 exp(-2 pi^2 j (p - 1)) <= 44 there, below an eighth of N.
    y, mu = _book(p=1.05)
    deviance = float(np.sum(tweedie_unit_deviance(y, mu, 1.05)))
    rows = TweedieRows.prepare(y, np.ones_like(y), 1.05)
    start = math.log(deviance / rows.size)
    newton = density_module._newton_log_phi(rows, deviance, 0.0, start)
    fresh = TweedieRows.prepare(y, np.ones_like(y), 1.05)
    assert solve_log_phi(fresh, deviance).n_passes == newton.n_passes


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
    direct = _reml_scale_criterion(rows, deviance, nullity, math.log(solved.phi))
    _assert_carried_criterion(rows, deviance, nullity, solved, direct)
    _assert_root_within_the_step_tolerance(rows, deviance, nullity, solved)


def test_interior_optimum_exists_until_the_nullity_reaches_2n_over_p_minus_1():
    # Q's upper tail slopes as N / (p - 1) - M / 2, so an interior minimum
    # exists exactly while M < 2 N / (p - 1); 3/4 of that limit still has one.
    y, mu = _book(p=1.5, n=400, seed=9)
    rows = TweedieRows.prepare(y, np.ones_like(y), 1.5)
    deviance = float(np.sum(tweedie_unit_deviance(y, mu, 1.5)))
    limit = 2.0 * rows.size / 0.5
    solved = solve_log_phi(rows, deviance, 0.75 * limit)
    assert solved.curvature > 0.0
    _assert_root_within_the_step_tolerance(rows, deviance, 0.75 * limit, solved)
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
    # Each stop leaves log phi within the step tolerance of the root and Q within
    # its carried remainder and round-off of the minimum.
    tol = density_module._NEWTON_STEP_TOL
    assert abs(math.log(counted.phi / replicated.phi)) <= 2.0 * tol
    remainder = (replicated.curvature + 0.5 * deviance / replicated.phi) * tol**2
    round_off = _criterion_round_off(replicated_rows, deviance, nullity, replicated.phi)
    assert abs(counted.criterion - replicated.criterion) <= 2.0 * (remainder + round_off)


@pytest.mark.parametrize("p", [1.2, 1.4, 1.5, 1.8])
def test_near_perfect_tweedie_fit_does_not_fail_in_fit_statistics(p):
    # An exact curve leaves phi at round-off (~1e-26), where every row's series
    # peak index is past the work bound; the saddlepoint gives the likelihood.
    x = np.linspace(-1.0, 1.0, 40)
    y = np.exp(0.3 + 0.5 * x)
    X = pd.DataFrame({"x": x})
    model = SuperGLM(family=Tweedie(p=p), selection_penalty=0, features={"x": Numeric()})
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        model.fit(X, y)
    assert np.isfinite(model.result.phi) and np.isfinite(model.result.deviance)
    assert np.isfinite(model._fit_stats.log_likelihood)
    assert np.isfinite(model._fit_stats.null_log_likelihood)
    assert np.isfinite(model.metrics(X, y).aic)
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


class _DoubleWellRows(TweedieRows):
    """Q(u) = u^4 / 4 - u^2 / 2 at D = 2, M = 0: a local maximum at the start u = 0."""

    def saturated(self, phi):
        u = math.log(phi)
        # l_sat = e^-u - Q, so T = e^-u + Q' and T' = Q'' - e^-u.
        return (
            math.exp(-u) - (u**4 / 4 - u**2 / 2),
            math.exp(-u) + u**3 - u,
            3 * u**2 - 1 - math.exp(-u),
        )


def test_solve_log_phi_walks_up_from_an_exact_stationary_point_of_negative_curvature():
    # The score is exactly 0.0 at the start, making u the lower bracket end; a
    # walk signed by -score went to u - 2, outside the bracket, and bisected to +inf.
    rows = _DoubleWellRows(1.5, 1.0, np.zeros(2), np.zeros(2), np.zeros(2), None)
    solved = solve_log_phi(rows, 2.0)
    assert abs(math.log(solved.phi) - 1.0) <= 2.0 * density_module._NEWTON_STEP_TOL


SCORE_NOISE = 1e-4


class _NoisyScoreRows(_UnitScoreRows):
    """T = 1 plus a fixed pseudo-random error in [-SCORE_NOISE, SCORE_NOISE).

    The error is keyed on the bits of u (Knuth's multiplicative hash) and is 1e4
    times the score the 1e-8 step tolerance resolves at curvature e^-u ~ 1. That
    was the state of real rows at peak indices near 1e7 before the saddlepoint
    switch: T cancelled two sums of 3e9 and carried round-off up to 5e-4.
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
    counted_rows = TweedieRows.prepare(y, counts, 1.4, frequency=True)
    replicated = TweedieRows.prepare(
        np.repeat(y, counts.astype(int)), np.ones(int(counts.sum())), 1.4
    ).saturated(1.3)
    # The same row values summed in two orders: each sum of n terms is within
    # (n - 1) eps of the sum of their magnitudes (Higham 2002, Eq. 4.4).
    magnitudes = counted_rows.count @ np.abs(np.asarray(counted_rows.row_saturated(1.3)).T)
    bound = 2.0 * counted_rows.size * EPS * magnitudes
    assert np.all(np.abs(np.subtract(counted_rows.saturated(1.3), replicated)) <= bound)
