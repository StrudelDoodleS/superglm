"""Row-local mathematical tests for the private generalized gamma kernel."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from scipy import stats

from superglm.distributional.kernels import generalized_gamma as gg
from tests._generalized_gamma_lss_oracles import (
    PACKED,
    complex_step_score,
    mp_derivatives,
    mp_expected_information,
    mp_log_density,
    mp_log_mean_loading,
    mp_mean_expected_information,
)


def _rel(a, b):
    return abs(a - b) / (1.0 + abs(b))


@pytest.mark.parametrize("x", [0.3, 1.0, 2.5, 7.9, 8.0, 8.1, 25.0, 400.0, 1.0e6])
def test_stirling_remainders_match_mpmath_across_the_series_switch(x):
    mp = pytest.importorskip("mpmath")
    with mp.workdps(50):
        xm = mp.mpf(x)
        s = mp.loggamma(xm) - (xm - mp.mpf(1) / 2) * mp.log(xm) + xm - mp.log(2 * mp.pi) / 2
        s1 = mp.digamma(xm) - mp.log(xm) + 1 / (2 * xm)
        s2 = mp.polygamma(1, xm) - 1 / xm - 1 / (2 * xm**2)
    got = (
        gg.stirling_remainder(np.array([x]))[0],
        gg.stirling_remainder_d1(np.array([x]))[0],
        gg.stirling_remainder_d2(np.array([x]))[0],
    )
    for value, reference in zip(got, (float(s), float(s1), float(s2)), strict=True):
        assert _rel(value, reference) <= 5.0e-14


@pytest.mark.parametrize(
    ("name", "reference"),
    [
        ("series_r2", lambda u: (math.expm1(u) - u) / u**2),
        ("series_e1", lambda u: math.expm1(u) / u),
        ("series_u2", lambda u: (u * math.exp(u) - math.expm1(u)) / u**2),
        ("series_t3", lambda u: (2 * (math.expm1(u) - u) - u * math.expm1(u)) / u**3),
        (
            "series_t3_d1",
            lambda u: (
                (4 * u * math.expm1(u) - u * u * math.exp(u) - 6 * math.expm1(u) + 6 * u) / u**4
            ),
        ),
    ],
)
@pytest.mark.parametrize("u", [-3.0, -1.0001, -0.9999, -0.3, 0.3, 0.9999, 1.0001, 2.5])
def test_u_series_functions_agree_with_direct_formulas_away_from_zero(name, reference, u):
    mp = pytest.importorskip("mpmath")
    # the direct formula is itself cancellation-prone near 0, so use mpmath as the judge
    with mp.workdps(50):
        um = mp.mpf(u)
        exact = {
            "series_r2": (mp.expm1(um) - um) / um**2,
            "series_e1": mp.expm1(um) / um,
            "series_u2": (um * mp.exp(um) - mp.expm1(um)) / um**2,
            "series_t3": (2 * (mp.expm1(um) - um) - um * mp.expm1(um)) / um**3,
            "series_t3_d1": (
                4 * um * mp.expm1(um) - um * um * mp.exp(um) - 6 * mp.expm1(um) + 6 * um
            )
            / um**4,
        }[name]
    value = getattr(gg, name)(np.array([u]))[0]
    assert _rel(value, float(exact)) <= 2.0e-15
    assert math.isfinite(reference(u))


def test_u_series_limits_at_zero():
    zero = np.array([0.0])
    assert gg.series_r2(zero)[0] == 0.5
    assert gg.series_e1(zero)[0] == 1.0
    assert gg.series_u2(zero)[0] == 0.5
    assert gg.series_t3(zero)[0] == pytest.approx(-1.0 / 6.0, abs=1e-17)
    assert gg.series_t3_d1(zero)[0] == pytest.approx(-1.0 / 12.0, abs=1e-17)


@pytest.mark.parametrize(
    "v",
    [-0.8001, -0.7999, -0.7, -0.2501, -0.2499, -0.1, 0.1, 0.2499, 0.2501, 0.7, 0.7999, 0.8001, 3.0],
)
def test_v_series_functions_match_mpmath(v):
    mp = pytest.importorskip("mpmath")
    with mp.workdps(50):
        vm = mp.mpf(v)
        l1 = mp.log1p(vm) / vm
        l1d = (vm / (1 + vm) - mp.log1p(vm)) / vm**2
        l2 = (mp.log1p(vm) - vm) / vm**2
        num = vm - (1 + vm / 2) * mp.log1p(vm)
        m3 = num / vm**3
        dnum = 1 - mp.log1p(vm) / 2 - (1 + vm / 2) / (1 + vm)
        m3d = (dnum * vm - 3 * num) / vm**4
    pairs = [
        (gg.series_l1, l1),
        (gg.series_l1_d1, l1d),
        (gg.series_l2, l2),
        (gg.series_m3, m3),
        (gg.series_m3_d1, m3d),
    ]
    for function, exact in pairs:
        assert _rel(function(np.array([v]))[0], float(exact)) <= 5.0e-15


def test_v_series_limits_at_zero():
    zero = np.array([0.0])
    assert gg.series_l1(zero)[0] == 1.0
    assert gg.series_l1_d1(zero)[0] == -0.5
    assert gg.series_l2(zero)[0] == -0.5
    assert gg.series_m3(zero)[0] == pytest.approx(-1.0 / 12.0, abs=1e-17)
    assert gg.series_m3_d1(zero)[0] == pytest.approx(1.0 / 12.0, abs=1e-17)


_ROW_POINTS = [
    # (y, mu, sigma, Q)
    (3.7, 0.7, 0.9, 0.7),
    (0.4, 0.7, 0.9, -0.5),
    (12.0, 1.1, 1.3, 1.6),
    (2.0, 0.2, 0.5, 1.0e-3),
    (5.0, 0.9, 2.0, -1.0e-4),
    (1.5, 0.3, 1.2, 0.05),
    (0.8, 0.3, 0.7, -0.05),
    (9.0, 0.5, 0.8, -1.2),
    (2.5, 0.5, 0.9, 0.2),
    (2.5, 0.5, 0.9, -0.2),
    (60.0, 0.5, 0.6, 0.3),
    (2.0, 0.2, 0.5, 0.0),
]


def _location(y, mu, sigma, q, multiplier=1.0, order=2):
    return gg.location_rows(
        np.array([y]),
        np.array([mu]),
        np.array([sigma]),
        np.array([q]),
        np.array([multiplier]),
        derivative_order=order,
    )


@pytest.mark.parametrize(("y", "mu", "sigma", "q"), _ROW_POINTS)
def test_location_log_density_score_hessian_match_mpmath(y, mu, sigma, q):
    pytest.importorskip("mpmath")
    q_ref = q if q != 0.0 else 1.0e-9  # the reference has no Q = 0 branch
    evaluated = _location(y, mu, sigma, q)
    carrier = -math.log(y) - 0.5 * math.log(2.0 * math.pi)
    reference = float(mp_log_density(y, mu, sigma, q_ref))
    # the Q = 0 row is judged at Q = 1e-9, where the density itself moves by s_Q * 1e-9
    density_tolerance = 1.0e-13 if q != 0.0 else 1.0e-8
    assert (
        _rel(float(evaluated.optimizing_log_likelihood[0]) + carrier, reference)
        <= density_tolerance
    )

    def f(m, s, qq):
        return mp_log_density(y, m, s, qq)

    orders = [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (2, 0, 0),
        (1, 1, 0),
        (1, 0, 1),
        (0, 2, 0),
        (0, 1, 1),
        (0, 0, 2),
    ]
    refs = mp_derivatives(f, (mu, sigma, q_ref), orders)
    tolerance = 1.0e-12 if q != 0.0 else 1.0e-8  # Q=0 row compared at Q=1e-9
    for index in range(3):
        assert _rel(float(evaluated.score[0, index]), refs[index]) <= tolerance
    for channel in range(6):
        assert _rel(float(evaluated.hessian_packed[0, channel]), refs[3 + channel]) <= (
            10.0 * tolerance
        )


@pytest.mark.parametrize(
    ("y", "mu", "sigma", "q"),
    [(3.7, 0.7, 0.9, 0.7), (12.0, 1.1, 1.3, 1.6), (2.5, 0.5, 0.9, 0.2)],
)
def test_location_score_matches_complex_step_of_the_textbook_density(y, mu, sigma, q):
    evaluated = _location(y, mu, sigma, q, order=1)
    for index in range(3):
        exact = complex_step_score(y, (mu, sigma, q), index)
        assert _rel(float(evaluated.score[0, index]), exact) <= 1.0e-12


def test_location_rows_return_exactly_the_requested_derivative_order():
    for order, has_score, has_hessian in ((0, False, False), (1, True, False), (2, True, True)):
        evaluated = _location(3.7, 0.7, 0.9, 0.7, order=order)
        assert (evaluated.score is not None) is has_score
        assert (evaluated.hessian_packed is not None) is has_hessian
        assert evaluated.valid.dtype == np.bool_ and bool(evaluated.valid[0])


def test_location_rows_scale_linearly_with_the_frequency_multiplier():
    one = _location(3.7, 0.7, 0.9, -0.4, multiplier=1.0)
    three = _location(3.7, 0.7, 0.9, -0.4, multiplier=3.0)
    assert np.allclose(
        three.optimizing_log_likelihood, 3.0 * one.optimizing_log_likelihood, rtol=0, atol=1e-14
    )
    assert np.allclose(three.score, 3.0 * one.score, rtol=0, atol=1e-13)
    assert np.allclose(three.hessian_packed, 3.0 * one.hessian_packed, rtol=0, atol=1e-13)


@pytest.mark.parametrize("order", [0, 1, 2])
def test_location_rows_flag_an_overflowing_exponent_instead_of_raising(order):
    """``exp(Q w)`` beyond float64 is an infeasible step, not an error.

    ``y = 1e300`` at ``mu = 0``, ``sigma = 1e-3`` and ``Q = 1`` puts ``Q w`` at
    6.9e5, three orders of magnitude past the ``exp`` overflow threshold.  The
    dense solver answers ``valid = False`` by shortening the step, whereas an
    exception aborts the fit, so every channel has to come back finite.  The
    feasible neighbour in the same call must be untouched.
    """
    rows = gg.location_rows(
        np.array([1.0e300, 2.0]),
        np.array([0.0, 0.2]),
        np.array([1.0e-3, 0.5]),
        np.array([1.0, 1.0]),
        np.ones(2),
        derivative_order=order,
    )
    assert rows.valid.tolist() == [False, True]
    assert np.all(np.isfinite(rows.optimizing_log_likelihood))
    assert rows.optimizing_log_likelihood[0] == 0.0
    alone = _location(2.0, 0.2, 0.5, 1.0, order=order)
    assert rows.optimizing_log_likelihood[1] == alone.optimizing_log_likelihood[0]
    if rows.score is not None:
        assert np.all(np.isfinite(rows.score)) and np.all(rows.score[0] == 0.0)
        assert np.array_equal(rows.score[1], alone.score[0])
    if rows.hessian_packed is not None:
        assert np.all(np.isfinite(rows.hessian_packed))
        assert np.all(rows.hessian_packed[0] == 0.0)
        assert np.array_equal(rows.hessian_packed[1], alone.hessian_packed[0])


def test_location_rows_refuse_nonpositive_response_and_scale():
    with pytest.raises(gg.GeneralizedGammaDomainError):
        _location(0.0, 0.7, 0.9, 0.7)
    with pytest.raises(gg.GeneralizedGammaDomainError):
        _location(1.0, 0.7, 0.0, 0.7)


@pytest.mark.parametrize(("sigma", "q"), [(0.9, 0.6), (1.3, -0.5), (0.8, 0.05), (0.7, 0.0)])
def test_location_expected_information_equals_minus_mean_hessian_and_score_outer_product(sigma, q):
    rng = np.random.default_rng(20260902)
    n = 2_000_000
    q_draw = q if q != 0.0 else 1.0e-9
    k = 1.0 / q_draw**2
    gamma = rng.gamma(k, 1.0, n)
    w = np.log(gamma / k) / q_draw
    y = np.exp(0.3 + sigma * w)
    rows = gg.location_rows(
        y, np.full(n, 0.3), np.full(n, sigma), np.full(n, q), np.ones(n), derivative_order=2
    )
    information = gg.location_expected_information(
        np.array([sigma]), np.array([q]), np.array([1.0])
    )[0]
    minus_hessian = -rows.hessian_packed
    outer = np.stack([rows.score[:, i] * rows.score[:, j] for i, j in PACKED], axis=1)
    for sample in (minus_hessian, outer):
        # reduce along a contiguous axis so numpy sums pairwise: the row-wise reduction
        # of an (n, 6) array accumulates naively and carries n * eps of rounding
        channels = np.ascontiguousarray(sample.T)
        mean = channels.mean(axis=1)
        spread = channels.std(axis=1)
        # H_mumu = -1/sigma^2 is constant at Q = 0: nothing to score a z against
        constant = spread <= 1.0e-12 * (1.0 + np.abs(information))
        assert np.allclose(mean[constant], information[constant], rtol=1e-12)
        z = (mean[~constant] - information[~constant]) / (spread[~constant] / math.sqrt(n))
        assert np.max(np.abs(z)) < 4.5


def test_location_expected_information_at_the_lognormal_point():
    information = gg.location_expected_information(
        np.array([0.9]), np.array([0.0]), np.array([1.0])
    )[0]
    sigma = 0.9
    expected = [1 / sigma**2, 0.0, -1 / (2 * sigma), 2 / sigma**2, 0.0, 5.0 / 12.0]
    assert np.allclose(information, expected, rtol=0, atol=1e-15)


_INFORMATION_POINTS = [
    # (sigma, Q); the small-|Q| ladder walks down to the |Q| = 1e-8 series switch
    (0.9, 0.6),
    (1.3, -0.5),
    (0.8, 0.05),
    (0.9, 1.6),
    (0.7, -1.2),
    (0.5, 2.5),
    (0.9, 1.0e-2),
    (0.9, -1.0e-2),
    (0.9, 1.0e-3),
    (0.9, 1.0e-4),
    (0.9, 1.0e-5),
    (0.9, 1.0e-6),
    (2.0, -1.0e-6),
    (0.9, 1.0e-7),
    (1.5, -3.0e-8),
    (0.9, 1.4821026336916461e-08),
    (0.9, 1.0000001e-08),
]


@pytest.mark.parametrize(("sigma", "q"), _INFORMATION_POINTS)
def test_location_expected_information_matches_the_cross_entropy_reference(sigma, q):
    """Every channel against ``-d^2/dtheta'^2 E_theta[log f_theta']`` at 60+ digits.

    The reference shares only the density with the kernel: it never evaluates
    the digamma and trigamma identities the packed information is built from,
    so it fails both on a wrong derivation and on a cancelling evaluation.
    """
    pytest.importorskip("mpmath")
    got = gg.location_expected_information(np.array([sigma]), np.array([q]), np.array([1.0]))[0]
    reference = mp_expected_information(sigma, q)
    for channel, exact in enumerate(reference):
        assert abs(float(got[channel]) - exact) <= 1.0e-12 * abs(exact), channel


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_location_expected_information_is_continuous_across_the_zero_shape_switch(sign):
    """The ``|Q| = 1e-8`` branch switch moves no channel by more than the step in ``Q``.

    ``I_ms`` and ``I_sQ`` are ``O(Q)``, so a pair straddling the switch by one
    part in ``1e6`` may differ by ``2e-6`` and no more; ``I_QQ`` and the rest
    are flat there.
    """
    sigma = 0.9
    below = sign * 1.0e-8 * (1.0 - 1.0e-6)
    above = sign * 1.0e-8 * (1.0 + 1.0e-6)
    pair = gg.location_expected_information(np.full(2, sigma), np.array([below, above]), np.ones(2))
    for channel in range(6):
        series, direct = float(pair[0, channel]), float(pair[1, channel])
        assert abs(direct - series) <= 1.0e-5 * abs(series), channel


def _full_information(packed):
    matrix = np.zeros((3, 3))
    for channel, (i, j) in enumerate(PACKED):
        matrix[i, j] = matrix[j, i] = float(packed[channel])
    return matrix


# recorded 2026-09-02 on this tree, unit multiplier, mean form evaluated at m = 1
# (its I_mm channel is 1/(sigma^2 m^2), so the mean-form conditioning moves with m)
_CONDITIONING_LADDER = [0.0, 1.0e-8, -1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2, -1.0e-2, 0.1, -0.1]
_RECORDED_CONDITION_AT_ZERO = {0.5: (51.12, 58.24), 0.9: (18.19, 24.05), 1.5: (9.17, 22.30)}


@pytest.mark.parametrize("sigma", [0.5, 0.9, 1.5])
def test_information_conditioning_at_the_lognormal_point_and_down_a_small_shape_ladder(sigma):
    """Record conditioning at ``Q = 0`` before any fit.

    Both forms stay positive definite down to the series switch, the location
    determinant is exactly ``1/(3 sigma^4)`` at ``Q = 0``, and the mean form is
    worse conditioned by at most 2.5x across the checked ladder.
    """
    location_at_zero, mean_at_zero = _RECORDED_CONDITION_AT_ZERO[sigma]
    for q in _CONDITIONING_LADDER:
        scale, shape, unit = np.array([sigma]), np.array([q]), np.array([1.0])
        location = _full_information(gg.location_expected_information(scale, shape, unit)[0])
        mean = _full_information(gg.mean_expected_information(unit, scale, shape, unit)[0])
        for matrix in (location, mean):
            assert float(np.linalg.eigvalsh(matrix)[0]) > 0.0, q
            assert float(np.linalg.cond(matrix)) <= 70.0, q
        if q == 0.0:
            assert np.linalg.det(location) == pytest.approx(1.0 / (3.0 * sigma**4), rel=1.0e-12)
            assert np.linalg.cond(location) == pytest.approx(location_at_zero, rel=1.0e-3)
            assert np.linalg.cond(mean) == pytest.approx(mean_at_zero, rel=1.0e-3)
            assert np.linalg.cond(mean) / np.linalg.cond(location) <= 2.5


def test_mean_form_conditioning_degrades_only_against_the_infinite_mean_boundary():
    """Where the mean form does hurt, and that the location form is the way out.

    At ``sigma = 1.5``, ``Q = -0.6`` the row is at ``sigma |Q| = 0.9``, nine
    tenths of the way to the boundary where ``E[Y]`` stops existing.  The mean
    form's condition number there is 1.1e5 against the location form's 9.5.
    """
    scale, shape, unit = np.array([1.5]), np.array([-0.6]), np.array([1.0])
    location = _full_information(gg.location_expected_information(scale, shape, unit)[0])
    mean = _full_information(gg.mean_expected_information(unit, scale, shape, unit)[0])
    assert float(np.linalg.eigvalsh(mean)[0]) > 0.0
    assert 1.0e4 < float(np.linalg.cond(mean)) < 1.0e7
    assert float(np.linalg.cond(location)) < 10.0


_LOADING_POINTS = [
    (0.9, 0.7),
    (1.3, -0.5),
    (0.5, 1.0e-3),
    (2.0, -1.0e-4),
    (1.2, 0.05),
    (0.7, -0.05),
    (1.5, -0.6),
    (0.8, 2.0),
    (1.0, 1.0e-6),
    (0.9, -0.5),
    (1.0, 0.26),
    (1.0, -0.26),
    (0.5, 0.49),
]


@pytest.mark.parametrize(("sigma", "q"), _LOADING_POINTS)
def test_log_mean_loading_and_derivatives_match_mpmath(sigma, q):
    pytest.importorskip("mpmath")
    values = gg.log_mean_loading(np.array([sigma]), np.array([q]))
    orders = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]

    def g(s, qq):
        return mp_log_mean_loading(s, qq)

    refs = mp_derivatives(g, (sigma, q), orders)
    tolerances = (1.0e-13, 1.0e-13, 1.0e-12, 1.0e-13, 1.0e-12, 1.0e-11)
    for value, reference, tolerance in zip(values, refs, tolerances, strict=True):
        assert _rel(float(value[0]), reference) <= tolerance


def test_log_mean_loading_exact_lognormal_limit():
    sigma = 0.9
    values = gg.log_mean_loading(np.array([sigma]), np.array([0.0]))
    expected = (
        sigma**2 / 2,
        sigma,
        -(sigma**3) / 6 - sigma / 2,
        1.0,
        -(sigma**2) / 2 - 0.5,
        sigma**4 / 6 + sigma**2 / 2,
    )
    for value, reference in zip(values, expected, strict=True):
        assert _rel(float(value[0]), reference) <= 1.0e-15


def test_mean_exists_mask():
    sigma = np.array([0.9, 0.9, 2.5, 2.5, 0.5, 0.7])
    shape = np.array([0.7, -0.5, -0.5, 0.5, 0.0, -1.0])
    assert gg.mean_exists(sigma, shape).tolist() == [True, True, False, True, True, True]
    assert gg.mean_exists(np.array([1.0]), np.array([-1.0])).tolist() == [False]


_MEAN_POINTS = [
    (3.7, 2.5, 0.9, 0.7),
    (0.4, 2.5, 0.9, -0.5),
    (2.0, 1.4, 0.5, 1.0e-3),
    (5.0, 3.0, 1.2, -0.3),
    (1.5, 1.7, 1.2, 0.05),
    (9.0, 2.0, 0.8, -1.2),
    (2.5, 1.9, 0.9, 0.2),
    (2.5, 1.9, 0.9, 0.0),
]


@pytest.mark.parametrize(("y", "m", "sigma", "q"), _MEAN_POINTS)
def test_mean_form_rows_match_mpmath_through_the_chain_rule(y, m, sigma, q):
    pytest.importorskip("mpmath")
    q_ref = q if q != 0.0 else 1.0e-9
    rows = gg.mean_rows(
        np.array([y]),
        np.array([m]),
        np.array([sigma]),
        np.array([q]),
        np.array([1.0]),
        derivative_order=2,
    )

    def f(mm, s, qq):
        import mpmath as mp

        mu = mp.log(mm) - mp_log_mean_loading(s, qq)
        return mp_log_density(y, mu, s, qq)

    orders = [
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (2, 0, 0),
        (1, 1, 0),
        (1, 0, 1),
        (0, 2, 0),
        (0, 1, 1),
        (0, 0, 2),
    ]
    refs = mp_derivatives(f, (m, sigma, q_ref), orders)
    tolerance = 1.0e-12 if q != 0.0 else 1.0e-8
    carrier = -math.log(y) - 0.5 * math.log(2.0 * math.pi)
    assert _rel(float(rows.optimizing_log_likelihood[0]) + carrier, refs[0]) <= 10.0 * tolerance
    for index in range(3):
        assert _rel(float(rows.score[0, index]), refs[1 + index]) <= tolerance
    for channel in range(6):
        assert _rel(float(rows.hessian_packed[0, channel]), refs[4 + channel]) <= 10.0 * tolerance
    assert bool(rows.valid[0])


def test_mean_form_rows_flag_infinite_mean_rows_with_finite_placeholders():
    rows = gg.mean_rows(
        np.array([1.0, 2.0]),
        np.array([1.0, 1.0]),
        np.array([2.5, 0.9]),
        np.array([-0.5, -0.5]),
        np.ones(2),
        derivative_order=2,
    )
    assert rows.valid.tolist() == [False, True]
    assert np.all(np.isfinite(rows.optimizing_log_likelihood))
    assert np.all(np.isfinite(rows.score)) and np.all(np.isfinite(rows.hessian_packed))
    assert rows.optimizing_log_likelihood[0] == 0.0 and np.all(rows.score[0] == 0.0)


@pytest.mark.parametrize(("sigma", "q"), [(0.9, 0.6), (1.3, -0.5), (0.8, 0.05)])
def test_mean_form_expected_information_is_the_jacobian_congruence(sigma, q):
    m = 2.2
    logc, cs, cq, _, _, _ = (
        float(v[0]) for v in gg.log_mean_loading(np.array([sigma]), np.array([q]))
    )
    packed = gg.location_expected_information(np.array([sigma]), np.array([q]), np.array([1.0]))[0]
    full = np.zeros((3, 3))
    for channel, (i, j) in enumerate(PACKED):
        full[i, j] = full[j, i] = packed[channel]
    jac = np.array([[1.0 / m, -cs, -cq], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    expected = jac.T @ full @ jac
    got = gg.mean_expected_information(
        np.array([m]), np.array([sigma]), np.array([q]), np.array([1.0])
    )[0]
    for channel, (i, j) in enumerate(PACKED):
        assert _rel(float(got[channel]), float(expected[i, j])) <= 1.0e-13


_MEAN_INFORMATION_POINTS = [
    # (mean, sigma, Q); the congruence test above cannot see a wrong closed form,
    # so these run the whole mean-form chain against the external reference
    (2.2, 0.9, 0.6),
    (2.2, 1.3, -0.5),
    (2.2, 0.8, 0.05),
    (3.0, 1.2, -0.3),
    (2.0, 0.9, 1.6),
    (0.7, 0.7, -1.2),
    (1.9, 0.9, -0.2),
    (1.4, 0.5, 1.0e-3),
    (2.5, 0.6, 1.0e-5),
    (1.1, 1.5, -1.0e-6),
    (2.2, 0.9, 1.0000001e-08),
]


@pytest.mark.parametrize(("mean", "sigma", "q"), _MEAN_INFORMATION_POINTS)
def test_mean_form_expected_information_matches_the_cross_entropy_reference(mean, sigma, q):
    """The mean form end to end against the reference, not against its own inputs.

    ``test_mean_form_expected_information_is_the_jacobian_congruence`` builds its
    expectation from ``location_expected_information`` and the loading
    derivatives, so it certifies the chaining and nothing else.  Here the
    reparametrisation happens inside the function the reference differentiates,
    so a wrong location channel, a wrong ``log C`` derivative and a wrong
    congruence all fail.  Worst measured relative error 3.5e-13 (``I_mQ`` at
    sigma 0.9, Q 0.6, where the chained difference itself cancels).
    """
    pytest.importorskip("mpmath")
    got = gg.mean_expected_information(
        np.array([mean]), np.array([sigma]), np.array([q]), np.array([1.0])
    )[0]
    reference = mp_mean_expected_information(mean, sigma, q)
    for channel, exact in enumerate(reference):
        assert abs(float(got[channel]) - exact) <= 1.0e-11 * abs(exact), channel


@pytest.mark.parametrize(
    ("mu", "sigma", "q"), [(0.7, 0.9, 0.7), (0.7, 0.9, -0.5), (1.1, 1.3, 1.6), (0.2, 0.5, 0.03)]
)
def test_cdf_quantile_and_mean_match_scipy_gengamma(mu, sigma, q):
    k = 1.0 / q**2
    reference = stats.gengamma(a=k, c=q / sigma, scale=math.exp(mu - sigma * math.log(k) / q))
    y = np.array([0.3, 1.0, 2.7, 9.0])
    ours = gg.generalized_gamma_cdf(y, np.full(4, mu), np.full(4, sigma), np.full(4, q))
    assert np.allclose(ours, reference.cdf(y), rtol=0, atol=1e-12)
    p = np.array([0.01, 0.5, 0.9, 0.999])
    quantiles = gg.generalized_gamma_quantile(p, np.full(4, mu), np.full(4, sigma), np.full(4, q))
    assert np.allclose(quantiles, reference.ppf(p), rtol=1e-10, atol=0)
    assert np.allclose(
        gg.generalized_gamma_cdf(quantiles, np.full(4, mu), np.full(4, sigma), np.full(4, q)),
        p,
        rtol=0,
        atol=1e-12,
    )
    mean = gg.mean_of_location(np.array([mu]), np.array([sigma]), np.array([q]))[0]
    assert _rel(float(mean), float(reference.mean())) <= 1.0e-10


def test_cdf_and_quantile_at_the_lognormal_point_and_infinite_mean():
    y = np.array([0.5, 2.0])
    ours = gg.generalized_gamma_cdf(y, np.zeros(2), np.full(2, 0.8), np.zeros(2))
    assert np.allclose(ours, stats.lognorm(s=0.8).cdf(y), rtol=0, atol=1e-15)
    assert gg.mean_of_location(np.array([0.0]), np.array([2.5]), np.array([-0.5]))[0] == np.inf
    back = gg.location_of_mean(np.array([3.0]), np.array([0.8]), np.array([0.3]))
    assert _rel(float(gg.mean_of_location(back, np.array([0.8]), np.array([0.3]))[0]), 3.0) <= 1e-14


def _draw(rng, n, mu, sigma, q):
    k = 1.0 / q**2
    w = np.log(rng.gamma(k, 1.0, n) / k) / q
    return np.exp(mu + sigma * w)


@pytest.mark.parametrize(
    ("mu", "sigma", "q"), [(1.5, 0.8, 0.9), (1.5, 0.8, -0.7), (0.3, 1.4, 0.05)]
)
def test_location_initialiser_recovers_the_truth_to_a_few_percent(mu, sigma, q):
    rng = np.random.default_rng(20260902)
    y = _draw(rng, 400_000, mu, sigma, q)
    theta = gg.initialize_generalized_gamma(
        y, np.ones(len(y)), parametrisation="location", scale_floor=0.01
    )
    assert theta.shape == (len(y), 3) and np.all(theta == theta[0])
    assert abs(theta[0, 0] - mu) <= 0.02
    assert abs(theta[0, 1] - sigma) <= 0.02
    assert abs(theta[0, 2] - q) <= 0.03


def test_mean_initialiser_returns_the_sample_mean_scale_and_a_finite_mean_start():
    rng = np.random.default_rng(3)
    y = _draw(rng, 200_000, 1.0, 0.6, 0.5)
    theta = gg.initialize_generalized_gamma(
        y, np.ones(len(y)), parametrisation="mean", scale_floor=0.01
    )
    assert gg.mean_exists(theta[:1, 1], theta[:1, 2]).all()
    assert abs(theta[0, 0] / float(np.mean(y)) - 1.0) <= 0.05


def test_mean_initialiser_floors_the_scale_without_leaving_the_finite_mean_region():
    """A non-default ``scale_floor`` must not push the start back over the boundary.

    ``sigma`` is raised to the floor after the mean-existence loop has chosen
    ``Q``, so a floor far above the sample scale re-enters ``sigma |Q| >= 1`` on
    the ``Q < 0`` side and the returned mean is ``inf``.  Every other initialiser
    test passes ``scale_floor=0.01``, where the brentq bracket caps ``|Q|`` at 50
    and ``sigma |Q| <= 0.5``, so none of them can see it.
    """
    rng = np.random.default_rng(20260902)
    y = _draw(rng, 200_000, 0.0, 0.02, -0.9)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        theta = gg.initialize_generalized_gamma(
            y, np.ones(len(y)), parametrisation="mean", scale_floor=1.5
        )
    assert np.all(np.isfinite(theta))
    assert theta[0, 1] >= 1.5
    assert gg.mean_exists(theta[:1, 1], theta[:1, 2]).all()
    assert theta[0, 1] * abs(theta[0, 2]) < 1.0
    assert any(
        issubclass(item.category, gg.GeneralizedGammaInitializationWarning) for item in caught
    )


def test_initialiser_frequency_mass_equals_literal_replication():
    y = np.array([0.7, 1.9, 4.2, 0.3, 2.2])
    mass = np.array([1.0, 3.0, 2.0, 1.0, 2.0])
    replicated = np.repeat(y, mass.astype(int))
    a = gg.initialize_generalized_gamma(y, mass, parametrisation="location", scale_floor=0.01)[0]
    b = gg.initialize_generalized_gamma(
        replicated, np.ones(len(replicated)), parametrisation="location", scale_floor=0.01
    )[0]
    assert np.allclose(a, b, rtol=0, atol=1e-12)


def test_initialiser_warns_when_log_skewness_is_outside_the_family_and_shrinks_for_the_mean():
    rng = np.random.default_rng(11)
    log_y = rng.standard_normal(50_000)
    log_y[:500] += (
        60.0  # one percent of the rows 60 sd out: log-skew about 9, beyond the family's 2
    )
    heavy = np.exp(log_y)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        theta = gg.initialize_generalized_gamma(
            heavy, np.ones(len(heavy)), parametrisation="mean", scale_floor=0.01
        )
    assert any(
        issubclass(item.category, gg.GeneralizedGammaInitializationWarning) for item in caught
    )
    assert gg.mean_exists(theta[:1, 1], theta[:1, 2]).all()
    assert theta[0, 1] >= 0.01
