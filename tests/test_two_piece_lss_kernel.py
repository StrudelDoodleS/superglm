"""Row-local mathematical tests for the private two-piece kernel."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import special, stats

from superglm.distributional.kernels import two_piece as tp
from tests._two_piece_lss_oracles import (
    PACKED,
    expectation_by_quadrature,
    mp_loading_derivatives,
    mp_log_density,
    mp_log_mean_loading,
    mp_log_mean_loading_by_quadrature,
    mp_mean_log_density,
    mp_packed,
    normalising_mass,
    piecewise_density,
    sn2_log_density,
    textbook_log_density,
)

HALF_LOG_TWO_PI = 0.5 * math.log(2.0 * math.pi)


def _rel(a, b):
    return abs(a - b) / (1.0 + abs(b))


def _row(t, mu, sigma, eps, order=2):
    return tp.location_rows(
        np.array([t]),
        np.array([mu]),
        np.array([sigma]),
        np.array([eps]),
        np.ones(1),
        derivative_order=order,
    )


@pytest.mark.parametrize("eps", [-0.85, -0.4, 0.0, 0.35, 0.9])
def test_the_density_normalises_to_one_and_matches_the_kernel(eps):
    assert normalising_mass(eps) == pytest.approx(1.0, abs=1e-12)
    for t in (-2.5, -0.3, 0.0, 0.9, 3.4):
        mu, sigma = 0.4, 0.8
        evaluated = _row(t, mu, sigma, eps, order=0)
        ours = float(evaluated.optimizing_log_likelihood[0]) - HALF_LOG_TWO_PI
        assert _rel(ours, textbook_log_density(t, mu, sigma, eps)) <= 1e-14


def test_positive_skew_widens_the_right_piece_and_signs_the_sample_skewness():
    """Pin the convention that positive skew widens the right piece."""
    eps = 0.6
    mu, sigma = 0.0, 1.0
    right = float(_row(2.0, mu, sigma, eps, order=0).optimizing_log_likelihood[0])
    left = float(_row(-2.0, mu, sigma, eps, order=0).optimizing_log_likelihood[0])
    assert right > left, "eps > 0 must put more mass to the right of the mode"
    rng = np.random.default_rng(20260902)
    for sign in (1.0, -1.0):
        draws = tp.two_piece_quantile(
            rng.uniform(size=200_000),
            np.zeros(200_000),
            np.ones(200_000),
            np.full(200_000, sign * 0.6),
        )
        assert math.copysign(1.0, float(stats.skew(draws))) == sign
        assert math.copysign(1.0, float(np.mean(draws))) == sign


@pytest.mark.parametrize(
    ("t", "mu", "sigma", "eps"),
    [
        (0.4, 0.1, 0.7, 0.0),
        (2.5, 0.3, 0.9, 0.45),
        (-0.9, 0.3, 0.9, 0.45),
        (5.0, 1.2, 0.5, -0.6),
        (-3.0, 1.2, 0.5, -0.6),
        (1.7, -0.2, 1.8, 0.85),
        (3.0, 0.0, 0.25, -0.85),
    ],
)
def test_location_rows_match_mpmath_score_and_hessian(t, mu, sigma, eps):
    pytest.importorskip("mpmath")
    evaluated = _row(t, mu, sigma, eps)
    assert (
        _rel(
            float(evaluated.optimizing_log_likelihood[0]), float(mp_log_density(t, mu, sigma, eps))
        )
        <= 1e-14
    )
    score, hessian = mp_packed(lambda a, b, c: mp_log_density(t, a, b, c), (mu, sigma, eps))
    for index, reference in enumerate(score):
        assert _rel(float(evaluated.score[0, index]), reference) <= 1e-13
    for index, reference in enumerate(hessian):
        assert _rel(float(evaluated.hessian_packed[0, index]), reference) <= 1e-12


def test_the_score_is_continuous_and_the_hessian_jumps_across_the_kink():
    """The score is C1 at ``z = 0`` while the Hessian jumps there.

    mpmath's ``mp.diff`` cannot straddle the kink, so this is the one derivative
    check written against the closed forms instead of the oracle.
    """
    mu, sigma, eps, h = 0.3, 0.8, 0.6, 1e-9
    left, exact, right = (_row(t, mu, sigma, eps) for t in (mu - h, mu, mu + h))
    # C1: the score is continuous through the kink (the mu channel carries the
    # step in |z|/s, which vanishes with z; measured |d score| <= 9.8e-09 at h = 1e-9)
    for neighbour in (left, right):
        assert np.allclose(neighbour.score[0], exact.score[0], rtol=0, atol=1e-8)
    assert np.allclose(exact.score[0], [0.0, -1.0 / sigma, 0.0], rtol=0, atol=0)
    # C2 fails: the mu-mu channel is -1/(s^2 sigma^2), so it jumps by exactly
    # ((1 + eps)/(1 - eps))^2 = 16 here
    h_left, h_right = float(left.hessian_packed[0, 0]), float(right.hessian_packed[0, 0])
    assert h_left == pytest.approx(-1.0 / ((1.0 - eps) ** 2 * sigma**2), rel=1e-14)
    assert h_right == pytest.approx(-1.0 / ((1.0 + eps) ** 2 * sigma**2), rel=1e-14)
    assert h_left / h_right == pytest.approx(((1.0 + eps) / (1.0 - eps)) ** 2, rel=1e-14)
    # the kernel's piece test is `left = z < 0.0`, so the row at exactly t == mu
    # is a right-piece row, not a left-piece one
    assert float(exact.hessian_packed[0, 0]) == h_right
    assert float(exact.hessian_packed[0, 0]) != h_left


def test_location_rows_return_exactly_the_requested_derivative_order():
    for order, has_score, has_hessian in ((0, False, False), (1, True, False), (2, True, True)):
        evaluated = _row(0.7, 0.2, 0.9, 0.3, order=order)
        assert (evaluated.score is not None) is has_score
        assert (evaluated.hessian_packed is not None) is has_hessian
    with pytest.raises(ValueError):
        _row(0.7, 0.2, 0.9, 0.3, order=3)


def test_location_rows_scale_linearly_with_the_frequency_multiplier():
    t = np.array([-0.4, 0.9, 2.2])
    args = (np.full(3, 0.2), np.full(3, 0.8), np.full(3, 0.4))
    unit = tp.location_rows(t, *args, np.ones(3), derivative_order=2)
    mass = np.array([1.0, 3.0, 7.0])
    weighted = tp.location_rows(t, *args, mass, derivative_order=2)
    assert np.allclose(
        weighted.optimizing_log_likelihood, mass * unit.optimizing_log_likelihood, rtol=0, atol=0
    )
    assert np.allclose(weighted.score, mass[:, None] * unit.score, rtol=0, atol=0)
    assert np.allclose(weighted.hessian_packed, mass[:, None] * unit.hessian_packed, rtol=0, atol=0)


def test_location_rows_refuse_an_unsupported_scale_or_skew():
    ones = np.ones(1)
    for sigma, eps in ((0.0, 0.3), (-1.0, 0.3), (0.8, 1.0), (0.8, -1.0), (0.8, 1.5)):
        with pytest.raises(tp.TwoPieceDomainError):
            tp.location_rows(
                np.array([0.5]),
                np.array([0.1]),
                np.array([sigma]),
                np.array([eps]),
                ones,
                derivative_order=2,
            )
    with pytest.raises(tp.TwoPieceDomainError):
        tp.location_rows(
            np.array([0.5, 0.6]),
            np.array([0.1]),
            np.array([0.8]),
            np.array([0.3]),
            ones,
            derivative_order=2,
        )


def test_location_rows_flag_a_nonrepresentable_row_instead_of_raising():
    evaluated = tp.location_rows(
        np.array([1.0e200, 0.5]),
        np.zeros(2),
        np.array([1.0e-160, 0.8]),
        np.zeros(2),
        np.ones(2),
        derivative_order=2,
    )
    assert evaluated.valid.tolist() == [False, True]
    assert np.all(np.isfinite(evaluated.optimizing_log_likelihood))
    assert np.all(np.isfinite(evaluated.score)) and np.all(np.isfinite(evaluated.hessian_packed))


@pytest.mark.parametrize(("sigma", "eps"), [(0.7, 0.0), (1.3, 0.45), (0.4, -0.8), (2.0, 0.9)])
def test_location_expected_information_equals_both_quadrature_expectations(sigma, eps):
    mu = 0.35
    analytic = tp.location_expected_information(np.array([sigma]), np.array([eps]), np.ones(1))[0]

    def channels(w):
        evaluated = tp.location_rows(
            np.array([mu + sigma * w]),
            np.array([mu]),
            np.array([sigma]),
            np.array([eps]),
            np.ones(1),
            derivative_order=2,
        )
        return evaluated.score[0], evaluated.hessian_packed[0]

    for index, (i, j) in enumerate(PACKED):
        minus_hessian = expectation_by_quadrature(lambda w: -channels(w)[1][index], eps)
        outer = expectation_by_quadrature(lambda w: channels(w)[0][i] * channels(w)[0][j], eps)
        assert _rel(minus_hessian, float(analytic[index])) <= 1e-12
        assert _rel(outer, float(analytic[index])) <= 1e-12


@pytest.mark.parametrize(("sigma", "eps"), [(0.9, 0.0), (1.4, 0.6), (0.6, -0.55)])
def test_location_information_equality_holds_in_monte_carlo(sigma, eps):
    rng = np.random.default_rng(4242)
    n = 400_000
    mu = 0.2
    draws = tp.two_piece_quantile(
        rng.uniform(size=n), np.full(n, mu), np.full(n, sigma), np.full(n, eps)
    )
    evaluated = tp.location_rows(
        draws, np.full(n, mu), np.full(n, sigma), np.full(n, eps), np.ones(n), derivative_order=2
    )
    analytic = tp.location_expected_information(np.array([sigma]), np.array([eps]), np.ones(1))[0]
    for index, (i, j) in enumerate(PACKED):
        product = evaluated.score[:, i] * evaluated.score[:, j]
        standard_error = float(np.std(product)) / math.sqrt(n)
        assert abs(float(np.mean(product)) - analytic[index]) <= 6.0 * standard_error + 1e-9
        minus_hessian = -evaluated.hessian_packed[:, index]
        spread = float(np.std(minus_hessian)) / math.sqrt(n)
        assert abs(float(np.mean(minus_hessian)) - analytic[index]) <= 6.0 * spread + 1e-9


def test_information_is_block_orthogonal_and_nonsingular_at_the_symmetric_point():
    for sigma in (0.5, 0.9, 1.5):
        packed = tp.location_expected_information(np.array([sigma]), np.array([0.0]), np.ones(1))[0]
        assert packed[1] == 0.0 and packed[4] == 0.0  # mu _|_ sigma, sigma _|_ eps
        matrix = np.zeros((3, 3))
        for index, (i, j) in enumerate(PACKED):
            matrix[i, j] = matrix[j, i] = packed[index]
        eigenvalues = np.linalg.eigvalsh(matrix)
        assert eigenvalues.min() > 0.0
        # det = (2/sigma^2) (3 - 8/pi) / sigma^2 at eps = 0
        expected = (2.0 / sigma**2) * (3.0 - 8.0 / math.pi) / sigma**2
        assert _rel(float(np.linalg.det(matrix)), expected) <= 1e-12


@pytest.mark.parametrize(
    ("sigma", "eps", "condition"),
    [(0.5, 0.0, 29.68), (0.9, 0.0, 29.99), (1.5, 0.0, 56.84), (0.9, 0.9, 29.99)],
)
def test_information_conditioning_is_pinned_at_and_around_the_symmetric_point(
    sigma, eps, condition
):
    """Pin information conditioning at and around the symmetric point."""
    packed = tp.location_expected_information(np.array([sigma]), np.array([eps]), np.ones(1))[0]
    matrix = np.zeros((3, 3))
    for index, (i, j) in enumerate(PACKED):
        matrix[i, j] = matrix[j, i] = packed[index]
    eigenvalues = np.linalg.eigvalsh(matrix)
    assert eigenvalues.min() > 0.0
    assert eigenvalues.max() / eigenvalues.min() == pytest.approx(condition, rel=1e-3)


@pytest.mark.parametrize(
    ("sigma", "eps", "condition"),
    [(0.5, 0.0, 21.42), (0.9, 0.0, 9.33), (1.5, 0.0, 17.52), (1.5, 0.9, 3183.67)],
)
def test_mean_form_information_conditioning_is_pinned_at_unit_mean(sigma, eps, condition):
    """The mean form's conditioning is stated at ``m = 1``; it is scale-dependent.

    These values justify the public ``skew_bound = 0.9``: the corner
    ``(1.5, 0.9)`` sets the wall, while the location form's own numbers there
    (29.68 / 56.84) do not show it.
    """
    packed = tp.mean_expected_information(
        np.ones(1), np.array([sigma]), np.array([eps]), np.ones(1)
    )[0]
    matrix = np.zeros((3, 3))
    for index, (i, j) in enumerate(PACKED):
        matrix[i, j] = matrix[j, i] = packed[index]
    eigenvalues = np.linalg.eigvalsh(matrix)
    assert eigenvalues.min() > 0.0
    assert eigenvalues.max() / eigenvalues.min() == pytest.approx(condition, rel=1e-3)


def test_location_expected_information_scales_with_the_multiplier_and_checks_shapes():
    mass = np.array([1.0, 4.0])
    packed = tp.location_expected_information(np.full(2, 0.8), np.full(2, 0.3), mass)
    unit = tp.location_expected_information(np.full(2, 0.8), np.full(2, 0.3), np.ones(2))
    assert np.allclose(packed, mass[:, None] * unit, rtol=0, atol=0)
    assert not packed.flags.writeable
    with pytest.raises(tp.TwoPieceDomainError):
        tp.location_expected_information(np.ones(2), np.zeros(3), np.ones(2))


@pytest.mark.parametrize("sigma", [0.1, 0.5, 1.0, 2.0, 3.0])
@pytest.mark.parametrize("eps", [-0.9, -0.5, 0.0, 0.35, 0.9])
def test_log_mean_loading_matches_the_closed_form_and_the_quadrature(sigma, eps):
    pytest.importorskip("mpmath")
    got = tp.log_mean_loading(np.array([sigma]), np.array([eps]))
    assert _rel(float(got[0][0]), float(mp_log_mean_loading(sigma, eps))) <= 1e-14
    assert _rel(float(got[0][0]), float(mp_log_mean_loading_by_quadrature(sigma, eps))) <= 1e-13
    for value, reference in zip(
        (float(v[0]) for v in got[1:]), mp_loading_derivatives(sigma, eps), strict=True
    ):
        assert _rel(value, reference) <= 1e-12


@pytest.mark.parametrize(
    ("sigma", "eps"),
    [(12.0, -0.9), (12.0, 0.0), (12.0, 0.9), (20.0, -0.9), (20.0, 0.9), (38.0, 0.0)],
)
def test_log_mean_loading_survives_where_the_naive_log_domain_dies(sigma, eps):
    """Check the stable log-domain form at and beyond sigma = 12.

    ``log_mean_loading``'s docstring requires the log domain.  The naive form
    is ``log(Phi(-a1))``, and ``ndtr(-a1)`` underflows to exactly zero once
    ``a1 = sigma (1 - eps)`` reaches 38 (it is still 5.73e-300 at 37).
    ``(20.0, -0.9)`` is the point that bites: ``a1 = 38`` while the left piece
    still carries about 8e-05 of the mass, so dropping it is a visible error
    rather than a rounding one; at ``(38.0, 0.0)`` the same underflow is
    harmless because the left piece is ``e^-727`` of the total.

    Against the exact-arithmetic closed form over this set, the worst errors
    are 2.71e-15 on ``log K`` and 2.99e-11 on the five derivatives, both at
    sigma = 12, eps = -0.9.
    The quadrature oracle is deliberately not used here: above
    ``sigma (1 + eps) ~ 20`` it is itself the limiting factor, losing 1.11e-06
    at (20.0, 0.9), so it cannot referee the kernel this far out.
    """
    pytest.importorskip("mpmath")
    a1 = sigma * (1.0 - eps)
    assert (float(special.ndtr(-a1)) == 0.0) == (a1 >= 38.0)
    got = tp.log_mean_loading(np.array([sigma]), np.array([eps]))
    assert np.all(np.isfinite([float(v[0]) for v in got]))
    assert _rel(float(got[0][0]), float(mp_log_mean_loading(sigma, eps))) <= 1e-14
    for value, reference in zip(
        (float(v[0]) for v in got[1:]), mp_loading_derivatives(sigma, eps), strict=True
    ):
        assert _rel(value, reference) <= 5e-11


@pytest.mark.parametrize("sigma", [0.3, 1.1, 2.0])
def test_log_mean_loading_limits_at_zero_skew(sigma):
    logk, k_s, k_e, k_ss, _, _ = (
        float(v[0]) for v in tp.log_mean_loading(np.array([sigma]), np.array([0.0]))
    )
    assert _rel(logk, sigma * sigma / 2.0) <= 1e-15
    assert _rel(k_s, sigma) <= 1e-15
    assert _rel(k_ss, 1.0) <= 1e-15
    # K_e does NOT collapse to sigma sqrt(2/pi); it is this closed form, whose
    # own sigma -> 0 limit is 2 sigma sqrt(2/pi) (twice the obvious guess,
    # because E[W] = 2 eps sqrt(2/pi) carries the factor of two).
    limit = (1.0 + sigma * sigma) * (2.0 * stats.norm.cdf(sigma) - 1.0) + 2.0 * sigma * (
        stats.norm.pdf(sigma)
    )
    assert _rel(k_e, limit) <= 1e-13
    small = tp.log_mean_loading(np.array([1e-6]), np.array([0.0]))[2][0]
    assert _rel(float(small), 2.0 * 1e-6 * math.sqrt(2.0 / math.pi)) <= 1e-11


def test_mean_and_location_coordinates_invert_each_other():
    sigma, eps = np.array([0.8, 1.4]), np.array([0.3, -0.6])
    mean = np.array([2.5, 9.0])
    mu = tp.location_of_mean(mean, sigma, eps)
    assert np.allclose(tp.mean_of_location(mu, sigma, eps), mean, rtol=1e-14, atol=0)
    assert np.all(np.isfinite(mu))
    assert np.allclose(
        tp.real_line_mean(np.array([0.5, -1.0]), sigma, eps),
        np.array([0.5, -1.0]) + 2.0 * eps * sigma * math.sqrt(2.0 / math.pi),
        rtol=1e-15,
        atol=0,
    )


@pytest.mark.parametrize(
    ("y", "mean", "sigma", "eps"),
    [
        (1.0, 2.0, 0.5, 0.0),
        (4.0, 2.0, 0.5, 0.4),
        (0.6, 2.0, 0.5, 0.4),
        (9.0, 1.0, 1.5, -0.7),
        (0.2, 1.0, 1.5, -0.7),
        (3.0, 5.0, 0.9, 0.85),
    ],
)
def test_mean_form_rows_match_mpmath_through_the_chain_rule(y, mean, sigma, eps):
    pytest.importorskip("mpmath")
    evaluated = tp.mean_rows(
        np.array([y]),
        np.array([mean]),
        np.array([sigma]),
        np.array([eps]),
        np.ones(1),
        derivative_order=2,
    )
    assert (
        _rel(
            float(evaluated.optimizing_log_likelihood[0]),
            float(mp_mean_log_density(y, mean, sigma, eps)),
        )
        <= 1e-14
    )
    score, hessian = mp_packed(lambda a, b, c: mp_mean_log_density(y, a, b, c), (mean, sigma, eps))
    for index, reference in enumerate(score):
        assert _rel(float(evaluated.score[0, index]), reference) <= 1e-12
    for index, reference in enumerate(hessian):
        assert _rel(float(evaluated.hessian_packed[0, index]), reference) <= 1e-11


@pytest.mark.parametrize(
    ("mean", "sigma", "eps"), [(2.0, 0.8, 0.0), (5.0, 1.2, 0.5), (0.7, 0.4, -0.75)]
)
def test_mean_form_expected_information_is_the_jacobian_congruence(mean, sigma, eps):
    got = tp.mean_expected_information(
        np.array([mean]), np.array([sigma]), np.array([eps]), np.ones(1)
    )[0]
    packed = tp.location_expected_information(np.array([sigma]), np.array([eps]), np.ones(1))[0]
    base = np.zeros((3, 3))
    for index, (i, j) in enumerate(PACKED):
        base[i, j] = base[j, i] = packed[index]
    _, k_s, k_e = (float(v[0]) for v in tp.log_mean_loading(np.array([sigma]), np.array([eps]))[:3])
    jacobian = np.array([[1.0 / mean, -k_s, -k_e], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    congruence = jacobian.T @ base @ jacobian
    for index, (i, j) in enumerate(PACKED):
        assert _rel(float(got[index]), float(congruence[i, j])) <= 1e-12
    assert np.linalg.eigvalsh(congruence).min() > 0.0


def test_mean_expected_information_scales_with_the_multiplier_and_checks_shapes():
    """The mean form obeys the same multiplier law as the location form."""
    mass = np.array([1.0, 4.0])
    mean, sigma, eps = np.full(2, 3.0), np.full(2, 0.8), np.full(2, 0.3)
    packed = tp.mean_expected_information(mean, sigma, eps, mass)
    unit = tp.mean_expected_information(mean, sigma, eps, np.ones(2))
    assert np.allclose(packed, mass[:, None] * unit, rtol=0, atol=0)
    assert not packed.flags.writeable
    with pytest.raises(tp.TwoPieceDomainError):
        tp.mean_expected_information(np.ones(2), np.ones(2), np.zeros(3), np.ones(2))


def test_mean_form_information_equality_holds_in_monte_carlo():
    rng = np.random.default_rng(11)
    n = 400_000
    mean, sigma, eps = 5.0, 1.2, 0.5
    mu = math.log(mean) - float(tp.log_mean_loading(np.array([sigma]), np.array([eps]))[0][0])
    variate = tp.two_piece_quantile(
        rng.uniform(size=n), np.full(n, mu), np.full(n, sigma), np.full(n, eps)
    )
    evaluated = tp.mean_rows(
        np.exp(variate),
        np.full(n, mean),
        np.full(n, sigma),
        np.full(n, eps),
        np.ones(n),
        derivative_order=2,
    )
    analytic = tp.mean_expected_information(
        np.array([mean]), np.array([sigma]), np.array([eps]), np.ones(1)
    )[0]
    for index, (i, j) in enumerate(PACKED):
        product = evaluated.score[:, i] * evaluated.score[:, j]
        standard_error = float(np.std(product)) / math.sqrt(n)
        assert abs(float(np.mean(product)) - analytic[index]) <= 6.0 * standard_error + 1e-9


def test_mean_rows_scale_with_the_multiplier_and_refuse_a_nonpositive_response():
    y = np.array([0.5, 2.0])
    args = (np.full(2, 3.0), np.full(2, 0.7), np.full(2, 0.2))
    unit = tp.mean_rows(y, *args, np.ones(2), derivative_order=2)
    mass = np.array([2.0, 5.0])
    weighted = tp.mean_rows(y, *args, mass, derivative_order=2)
    assert np.allclose(weighted.score, mass[:, None] * unit.score, rtol=0, atol=0)
    with pytest.raises(tp.TwoPieceDomainError):
        tp.mean_rows(np.array([0.0, 2.0]), *args, np.ones(2), derivative_order=2)


@pytest.mark.parametrize("eps", [-0.85, -0.3, 0.0, 0.55, 0.9])
def test_cdf_matches_quadrature_and_quantile_inverts_it(eps):
    from scipy import integrate

    mu, sigma = 0.4, 0.9
    for w in (-3.0, -0.7, 0.0, 0.4, 2.6):
        lower = integrate.quad(lambda x: piecewise_density(x, eps), -60.0, min(w, 0.0))[0]
        upper = integrate.quad(lambda x: piecewise_density(x, eps), 0.0, max(w, 0.0))[0]
        got = float(
            tp.two_piece_cdf(
                np.array([mu + sigma * w]), np.array([mu]), np.array([sigma]), np.array([eps])
            )[0]
        )
        assert abs(got - (lower + upper)) <= 1e-11
    p = np.array([1e-6, 0.02, 0.3, 0.5, 0.75, 0.999])
    n = len(p)
    quantiles = tp.two_piece_quantile(p, np.full(n, mu), np.full(n, sigma), np.full(n, eps))
    back = tp.two_piece_cdf(quantiles, np.full(n, mu), np.full(n, sigma), np.full(n, eps))
    assert np.allclose(back, p, rtol=0, atol=1e-13)
    assert np.all(np.diff(quantiles) > 0.0)


def test_cdf_and_quantile_reduce_to_the_normal_at_zero_skew():
    mu, sigma = -0.2, 1.3
    t = np.array([-3.0, -0.2, 0.5, 4.0])
    n = len(t)
    zeros = np.zeros(n)
    got = tp.two_piece_cdf(t, np.full(n, mu), np.full(n, sigma), zeros)
    assert np.allclose(got, stats.norm(mu, sigma).cdf(t), rtol=0, atol=1e-15)
    p = np.array([0.05, 0.4, 0.5, 0.97])
    quantiles = tp.two_piece_quantile(p, np.full(4, mu), np.full(4, sigma), np.zeros(4))
    assert np.allclose(quantiles, stats.norm(mu, sigma).ppf(p), rtol=1e-14, atol=0)


def test_quantile_refuses_probabilities_outside_the_open_unit_interval():
    for bad in (0.0, 1.0, -0.1, 1.5, np.nan):
        with pytest.raises(tp.TwoPieceDomainError):
            tp.two_piece_quantile(np.array([bad]), np.zeros(1), np.ones(1), np.zeros(1))


def test_standard_skewness_is_monotone_and_its_inversion_is_exact():
    grid = np.linspace(-0.899, 0.899, 400)
    values = tp.standard_skewness(grid)
    assert np.all(np.diff(values) > 0.0)
    assert float(tp.standard_skewness(np.array([0.899]))[0]) == pytest.approx(0.9655, abs=5e-4)
    assert float(tp.standard_skewness(np.array([0.0]))[0]) == 0.0
    for eps in (-0.8, -0.25, 0.0, 0.45, 0.87):
        target = float(tp.standard_skewness(np.array([eps]))[0])
        recovered, clamped = tp.skew_from_sample_skewness(target, bound=0.9)
        assert not clamped
        assert abs(recovered - eps) <= 1e-10


@pytest.mark.parametrize("bound", [0.5, 0.9])
def test_skew_inversion_clamps_and_warns_beyond_the_family_range(bound):
    """The clamp must land inside the caller's own bound, not inside the default."""
    with pytest.warns(tp.TwoPieceInitializationWarning, match="outside the two-piece range"):
        value, clamped = tp.skew_from_sample_skewness(3.5, bound=bound)
    assert clamped and 0.0 < value < bound


@pytest.mark.parametrize(
    ("mu", "sigma", "eps"), [(0.5, 0.7, 0.0), (-0.3, 1.1, 0.6), (2.0, 0.45, -0.7)]
)
def test_location_initialiser_recovers_the_truth_on_a_large_sample(mu, sigma, eps):
    rng = np.random.default_rng(7)
    n = 200_000
    variate = tp.two_piece_quantile(
        rng.uniform(size=n), np.full(n, mu), np.full(n, sigma), np.full(n, eps)
    )
    theta = tp.initialize_two_piece(
        variate, np.ones(n), parametrisation="location", scale_floor=0.01, skew_bound=0.9
    )
    assert theta.shape == (n, 3)
    assert np.all(theta == theta[0])
    # mu = mean - 2 eps sigma sqrt(2/pi) inherits the skew error with a lever
    # of 2 sigma sqrt(2/pi), so its bound is proportional to sigma.  Measured
    # over this grid x five seeds at n = 2e5: max |d mu| / sigma = 0.0298,
    # max |d sigma| / sigma = 0.0042, max |d eps| = 0.0179.
    assert abs(float(theta[0, 0]) - mu) <= 0.05 * sigma
    assert abs(float(theta[0, 1]) - sigma) <= 0.02 * sigma
    assert abs(float(theta[0, 2]) - eps) <= 0.03


def test_mean_initialiser_agrees_with_the_location_start_through_the_loading():
    rng = np.random.default_rng(19)
    n = 20_000
    mu, sigma, eps = 0.3, 0.9, 0.4
    variate = tp.two_piece_quantile(
        rng.uniform(size=n), np.full(n, mu), np.full(n, sigma), np.full(n, eps)
    )
    mean_start = tp.initialize_two_piece(
        variate, np.ones(n), parametrisation="mean", scale_floor=0.01, skew_bound=0.9
    )
    location_start = tp.initialize_two_piece(
        variate, np.ones(n), parametrisation="location", scale_floor=0.01, skew_bound=0.9
    )
    assert np.allclose(mean_start[:, 1:], location_start[:, 1:], rtol=0, atol=0)
    implied = float(
        tp.mean_of_location(location_start[:1, 0], location_start[:1, 1], location_start[:1, 2])[0]
    )
    assert float(mean_start[0, 0]) == pytest.approx(implied, rel=1e-14)
    assert implied > math.exp(mu)  # the loading lifts the mean above the median


def test_initialiser_floors_the_scale_on_a_degenerate_sample():
    rng = np.random.default_rng(23)
    flat = 1.0 + 1e-9 * rng.standard_normal(50)
    floored = tp.initialize_two_piece(
        flat, np.ones(50), parametrisation="mean", scale_floor=0.25, skew_bound=0.9
    )
    assert float(floored[0, 1]) > 0.25


def test_initialiser_frequency_mass_equals_literal_replication():
    variate = np.array([-0.4, 0.2, 1.7])
    counts = np.array([1.0, 3.0, 2.0])
    weighted = tp.initialize_two_piece(
        variate, counts, parametrisation="location", scale_floor=0.0, skew_bound=0.9
    )
    replicated = tp.initialize_two_piece(
        np.repeat(variate, counts.astype(int)),
        np.ones(6),
        parametrisation="location",
        scale_floor=0.0,
        skew_bound=0.9,
    )
    assert np.allclose(weighted[0], replicated[0], rtol=1e-12, atol=1e-12)


def test_initialiser_refuses_an_unsupported_parametrisation():
    with pytest.raises(tp.TwoPieceDomainError):
        tp.initialize_two_piece(
            np.array([0.1, 0.5]),
            np.ones(2),
            parametrisation="median",
            scale_floor=0.01,
            skew_bound=0.9,
        )


@pytest.mark.parametrize(("sigma", "eps"), [(0.8, 0.4), (1.5, -0.6), (0.3, 0.85)])
def test_our_density_equals_the_gamlss_sn2_density_under_the_published_mapping(sigma, eps):
    """``nu^2 = (1+eps)/(1-eps)``, ``sigma_SN2 = sigma sqrt(1 - eps^2)``.

    The identity includes SN2's normalising constant, so no fitted constant is
    absorbed anywhere; the R test in the vertical slice only confirms the
    package agrees.
    """
    nu = math.sqrt((1.0 + eps) / (1.0 - eps))
    sigma_sn2 = sigma * math.sqrt(1.0 - eps * eps)
    mu = 0.25
    for t in (-2.3, -0.4, 0.5, 3.1):
        ours = (
            float(_row(t, mu, sigma, eps, order=0).optimizing_log_likelihood[0]) - HALF_LOG_TWO_PI
        )
        assert _rel(ours, sn2_log_density(t, mu, sigma_sn2, nu)) <= 1e-14
