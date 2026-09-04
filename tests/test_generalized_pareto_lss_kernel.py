"""Kernel numerics for the generalized Pareto family."""

from __future__ import annotations

import math

import numpy as np
import pytest

from superglm.distributional.kernels import generalized_pareto as gp


def _rel(a, b):
    return abs(a - b) / (1.0 + abs(b))


_SWITCH_LADDER = [
    1e-12,
    1e-10,
    1e-6,
    1e-3,
    0.01,
    0.1,
    0.25,
    0.4,
    0.49,
    0.5,
    0.51,
    0.7,
    1.0,
    3.0,
    40.0,
    -0.05,
    -0.3,
    -0.49,
    -0.5,
    -0.51,
    -0.6,
    -0.9,
]


@pytest.mark.parametrize("z", _SWITCH_LADDER)
def test_z_series_functions_match_mpmath_across_the_switch(z):
    mp = pytest.importorskip("mpmath")
    with mp.workdps(50):
        zm = mp.mpf(z)
        l1 = mp.log1p(zm) / zm
        d = (mp.log1p(zm) - zm / (1 + zm)) / zm**2
        e = (1 / (1 + zm) ** 2 - 2 * d) / zm
    pairs = ((gp.series_l1, l1), (gp.series_d, d), (gp.series_e, e))
    for function, exact in pairs:
        assert _rel(float(function(np.array([z]))[0]), float(exact)) <= 5.0e-15


def test_z_series_limits_at_zero_are_exact():
    zero = np.array([0.0])
    assert gp.series_l1(zero)[0] == 1.0
    assert gp.series_d(zero)[0] == 0.5
    assert gp.series_e(zero)[0] == pytest.approx(-2.0 / 3.0, abs=1e-17)
    assert gp.series_e1(zero)[0] == 1.0


@pytest.mark.parametrize("a", [-4.0, -0.9, -0.3, 0.3, 0.9, 2.0, 12.0])
def test_e1_series_matches_mpmath(a):
    mp = pytest.importorskip("mpmath")
    with mp.workdps(50):
        exact = mp.expm1(mp.mpf(a)) / mp.mpf(a)
    assert _rel(float(gp.series_e1(np.array([a]))[0]), float(exact)) <= 5.0e-15


def _direct_l1(a):
    return np.log1p(a) / a


def _direct_d(a):
    return (np.log1p(a) - a / (1.0 + a)) / (a * a)


def _direct_e(a):
    return (1.0 / ((1.0 + a) * (1.0 + a)) - 2.0 * _direct_d(a)) / a


def test_the_series_switch_is_continuous_in_every_channel():
    """The two branches agree at the switch, evaluated at the *same* ``z``.

    Stepping across the radius instead (``0.4999999999`` against
    ``0.5000000001``) moves ``z`` by 2e-10, which moves ``E`` by 2.7e-09 through
    its own slope alone: that measures the derivative, not the branch.
    """
    inside = np.array([-0.4999999999, -0.49, -0.4, 0.4, 0.49, 0.4999999999])
    pairs = ((gp.series_l1, _direct_l1), (gp.series_d, _direct_d), (gp.series_e, _direct_e))
    for series, direct in pairs:
        assert np.allclose(series(inside), direct(inside), rtol=0, atol=5e-15)
    outside = np.array([-0.5, 0.5])
    for series, direct in pairs:
        assert np.allclose(series(outside), direct(outside), rtol=0, atol=0.0)


def test_the_series_radius_and_term_count_are_the_measured_ones():
    assert gp._Z_SERIES_RADIUS == 0.5
    assert gp._Z_SERIES_TERMS == 70


from tests._generalized_pareto_lss_oracles import (  # noqa: E402
    PACKED,
    mp_derivatives,
    mp_log_density,
    naive_score_and_hessian,
    textbook_log_density,
)

_ROW_POINTS = [
    # (y, psi, xi)
    (0.0, 1.0, 0.3),
    (0.7, 1.0, 0.3),
    (3.7, 2.0, 0.5),
    (250.0, 2.0, 0.9),
    (1.2, 0.5, 0.05),
    (1.2, 0.5, 1.0e-3),
    (1.2, 0.5, 1.0e-6),
    (1.2, 0.5, 1.0e-9),
    (900.0, 0.5, 1.0e-6),
    (5.0, 3.0, 0.999),
    (5.0, 3.0, 1.0e-12),
    (0.05, 1.0, 0.25),
    (40.0, 1.0, 0.25),
    (1.0, 1.0, 0.5),
    (2.0, 1.0, 0.02),
    (1.0e-8, 1.0, 0.4),
    (1.0e6, 1000.0, 0.7),
    (4.0, 1.0, -0.2),
]


def _rows(y, psi, xi, multiplier=1.0, order=2):
    return gp.scale_rows(
        np.array([y]),
        np.array([psi]),
        np.array([xi]),
        np.array([multiplier]),
        derivative_order=order,
    )


@pytest.mark.parametrize(("y", "psi", "xi"), _ROW_POINTS)
def test_rows_match_mpmath_in_log_density_score_and_every_hessian_channel(y, psi, xi):
    pytest.importorskip("mpmath")
    evaluated = _rows(y, psi, xi)

    def density(scale, shape):
        return mp_log_density(y, scale, shape)

    orders = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
    references = mp_derivatives(density, (psi, xi), orders)
    assert _rel(float(evaluated.optimizing_log_likelihood[0]), references[0]) <= 1.0e-14
    for index in range(2):
        assert _rel(float(evaluated.score[0, index]), references[1 + index]) <= 1.0e-14
    for channel in range(3):
        assert _rel(float(evaluated.hessian_packed[0, channel]), references[3 + channel]) <= 1.0e-14
    assert bool(evaluated.valid[0])


@pytest.mark.parametrize(("y", "psi", "xi"), [(1.2, 0.5, 1.0e-9), (5.0, 3.0, 1.0e-12)])
def test_the_naive_form_is_the_reason_the_series_branch_exists(y, psi, xi):
    """The kernel is exact at tiny shape; the textbook 1/xi form is not."""
    pytest.importorskip("mpmath")

    def density(scale, shape):
        return mp_log_density(y, scale, shape)

    reference_hessian = mp_derivatives(density, (psi, xi), [(0, 2)])[0]
    ours = float(_rows(y, psi, xi).hessian_packed[0, 2])
    _, naive = naive_score_and_hessian(y, psi, xi)
    assert _rel(ours, reference_hessian) <= 1.0e-14
    assert _rel(float(naive[2]), reference_hessian) > 1.0
    # the naive log density is fine: log1p absorbs it, so the branch is bought by derivatives only
    assert (
        _rel(float(textbook_log_density(y, psi, xi)), float(mp_log_density(y, psi, xi))) <= 1.0e-14
    )


def test_rows_return_exactly_the_requested_derivative_order():
    for order, has_score, has_hessian in ((0, False, False), (1, True, False), (2, True, True)):
        evaluated = _rows(3.7, 2.0, 0.5, order=order)
        assert (evaluated.score is not None) is has_score
        assert (evaluated.hessian_packed is not None) is has_hessian
        assert evaluated.valid.dtype == np.bool_ and bool(evaluated.valid[0])
    with pytest.raises(gp.GeneralizedParetoDomainError):
        _rows(3.7, 2.0, 0.5, order=3)


def test_rows_scale_linearly_with_the_frequency_multiplier():
    one = _rows(3.7, 2.0, 0.4)
    three = _rows(3.7, 2.0, 0.4, multiplier=3.0)
    assert np.allclose(
        three.optimizing_log_likelihood, 3.0 * one.optimizing_log_likelihood, rtol=0, atol=1e-14
    )
    assert np.allclose(three.score, 3.0 * one.score, rtol=0, atol=1e-13)
    assert np.allclose(three.hessian_packed, 3.0 * one.hessian_packed, rtol=0, atol=1e-13)


def test_rows_refuse_a_negative_excess_and_a_nonpositive_scale():
    with pytest.raises(gp.GeneralizedParetoDomainError):
        _rows(-1.0e-12, 1.0, 0.3)
    with pytest.raises(gp.GeneralizedParetoDomainError):
        _rows(1.0, 0.0, 0.3)
    with pytest.raises(gp.GeneralizedParetoDomainError):
        _rows(1.0, 1.0, math.nan)
    with pytest.raises(gp.GeneralizedParetoDomainError):
        gp.scale_rows(
            np.array([1.0, 2.0]), np.array([1.0]), np.array([0.3]), np.ones(2), derivative_order=2
        )


def test_rows_flag_the_negative_shape_endpoint_with_finite_placeholders():
    """The walls forbid xi < 0 today; the mask exists so widening them needs no kernel change."""
    evaluated = gp.scale_rows(
        np.array([9.0, 1.0]),
        np.array([1.0, 1.0]),
        np.array([-0.5, -0.5]),
        np.ones(2),
        derivative_order=2,
    )
    assert evaluated.valid.tolist() == [False, True]
    assert np.all(np.isfinite(evaluated.optimizing_log_likelihood))
    assert np.all(np.isfinite(evaluated.score)) and np.all(np.isfinite(evaluated.hessian_packed))
    assert evaluated.optimizing_log_likelihood[0] == 0.0
    assert np.all(evaluated.score[0] == 0.0) and np.all(evaluated.hessian_packed[0] == 0.0)


def test_the_zero_excess_row_is_finite_and_matches_the_closed_form():
    evaluated = _rows(0.0, 2.0, 0.4)
    assert float(evaluated.optimizing_log_likelihood[0]) == pytest.approx(-math.log(2.0), abs=1e-16)
    assert float(evaluated.score[0, 0]) == pytest.approx(-0.5, abs=1e-16)
    assert float(evaluated.score[0, 1]) == 0.0
    assert float(evaluated.hessian_packed[0, 0]) == pytest.approx(0.25, abs=1e-16)
    assert float(evaluated.hessian_packed[0, 1]) == 0.0
    assert float(evaluated.hessian_packed[0, 2]) == 0.0


@pytest.mark.parametrize("xi", [0.0, 1.0e-14, 1.0e-12])
def test_the_zero_shape_kernel_is_the_exponential(xi):
    """The kernel reaches the exponential limit to first order in ``xi``.

    The bounds are the leading term of the ``xi`` expansion, doubled, plus a
    float64 floor: the generalized Pareto genuinely differs from the exponential
    by ``O(xi t^2)`` in the density, ``O(xi t^2 / psi)`` in the scale score and
    ``O(xi t^3 / psi^2)`` in the scale curvature, so a flat bound would be
    testing the mathematics away rather than the code.  Measured deviations at
    ``xi = 1e-12`` are 7.50e-12, 1.00e-11 and 1.63e-11, and they fall by exactly
    100x at ``xi = 1e-14``, which is what makes them the expansion and not noise.
    """
    y = np.array([0.0, 0.4, 1.3, 2.8, 10.0])
    psi = 2.0
    t = float(np.max(y / psi))
    floor = 1.0e-13
    evaluated = gp.scale_rows(y, np.full(5, psi), np.full(5, xi), np.ones(5), derivative_order=2)
    assert np.allclose(
        evaluated.optimizing_log_likelihood,
        -math.log(psi) - y / psi,
        rtol=0,
        atol=floor + 2.0 * xi * t**2,
    )
    assert np.allclose(
        evaluated.score[:, 0],
        (y / psi - 1.0) / psi,
        rtol=0,
        atol=floor + 2.0 * xi * t**2 / psi,
    )
    assert np.allclose(
        evaluated.hessian_packed[:, 0],
        (1.0 - 2.0 * y / psi) / psi**2,
        rtol=0,
        atol=floor + 2.0 * xi * t**3 / psi**2,
    )


@pytest.mark.parametrize(("psi", "xi"), [(1.0, 0.3), (2.0, 0.05), (0.7, 0.8), (1.0, 1.0e-9)])
def test_expected_information_equals_minus_mean_hessian_and_score_outer_product(psi, xi):
    rng = np.random.default_rng(20260902)
    n = 2_000_000
    y = psi * np.expm1(-xi * np.log(rng.random(n))) / xi
    rows = gp.scale_rows(y, np.full(n, psi), np.full(n, xi), np.ones(n), derivative_order=2)
    information = gp.expected_information(np.array([psi]), np.array([xi]), np.array([1.0]))[0]
    minus_hessian = -rows.hessian_packed
    z_hessian = (minus_hessian.mean(axis=0) - information) / (
        minus_hessian.std(axis=0) / math.sqrt(n)
    )
    outer = np.stack([rows.score[:, i] * rows.score[:, j] for i, j in PACKED], axis=1)
    z_outer = (outer.mean(axis=0) - information) / (outer.std(axis=0) / math.sqrt(n))
    assert np.max(np.abs(z_hessian)) < 4.5
    assert np.max(np.abs(z_outer)) < 4.5


def test_expected_information_closed_form_and_the_exponential_limit():
    psi, xi = 1.7, 0.3
    got = gp.expected_information(np.array([psi]), np.array([xi]), np.array([2.0]))[0]
    expected = (
        np.array(
            [
                1.0 / (psi**2 * (2 * xi + 1)),
                1.0 / (psi * (2 * xi + 1) * (xi + 1)),
                2.0 / ((2 * xi + 1) * (xi + 1)),
            ]
        )
        * 2.0
    )
    assert np.allclose(got, expected, rtol=0, atol=1e-15)
    limit = gp.expected_information(np.array([psi]), np.array([0.0]), np.array([1.0]))[0]
    assert np.allclose(limit, [1.0 / psi**2, 1.0 / psi, 2.0], rtol=0, atol=1e-15)
    with pytest.raises(gp.GeneralizedParetoDomainError, match="-1/2"):
        gp.expected_information(np.array([1.0]), np.array([-0.6]), np.array([1.0]))


def test_information_conditioning_across_the_shape_walls_is_recorded_before_any_fit():
    """§5 item 3: pinned so a change to the information cannot pass silently."""
    ladder = {
        1.0e-9: (6.8541, 1.000000, 0.381966),
        0.01: (6.7339, 0.961075, 0.377784),
        0.1: (5.8370, 0.688705, 0.343496),
        0.3: (4.5883, 0.369823, 0.283903),
        0.5: (3.8664, 0.222222, 0.239741),
        0.9: (3.1107, 0.098932, 0.178337),
        0.99: (3.0101, 0.084738, 0.167783),
    }
    for xi, (condition, determinant, smallest) in ladder.items():
        packed = gp.expected_information(np.array([1.0]), np.array([xi]), np.array([1.0]))[0]
        full = np.array([[packed[0], packed[1]], [packed[1], packed[2]]])
        eigenvalues = np.linalg.eigvalsh(full)
        assert eigenvalues[0] > 0.0
        assert _rel(float(eigenvalues[-1] / eigenvalues[0]), condition) <= 1e-4
        assert _rel(float(np.linalg.det(full)), determinant) <= 1e-5
        assert _rel(float(eigenvalues[0]), smallest) <= 1e-5


import warnings  # noqa: E402

from scipy import stats  # noqa: E402


@pytest.mark.parametrize(
    ("psi", "xi"), [(1.0, 0.3), (2.5, 0.05), (0.4, 0.85), (1.0, 1.0e-8), (1.0, 0.999)]
)
def test_cdf_quantile_and_mean_match_scipy_genpareto(psi, xi):
    reference = stats.genpareto(c=xi, scale=psi)
    y = np.array([0.0, 0.01, 0.5, 2.0, 17.0, 400.0])
    ours = gp.generalized_pareto_cdf(y, np.full(6, psi), np.full(6, xi))
    assert np.allclose(ours, reference.cdf(y), rtol=0, atol=1e-14)
    p = np.array([1.0e-6, 0.1, 0.5, 0.9, 0.999])
    quantiles = gp.generalized_pareto_quantile(p, np.full(5, psi), np.full(5, xi))
    assert np.allclose(quantiles, reference.ppf(p), rtol=1e-12, atol=0)
    back = gp.generalized_pareto_cdf(quantiles, np.full(5, psi), np.full(5, xi))
    assert np.allclose(back, p, rtol=0, atol=1e-14)
    assert (
        _rel(
            float(gp.generalized_pareto_mean(np.array([psi]), np.array([xi]))[0]),
            float(reference.mean()),
        )
        <= 1e-13
    )


def test_mean_is_infinite_at_and_beyond_the_unit_shape_and_cdf_saturates_past_an_endpoint():
    means = gp.generalized_pareto_mean(np.array([1.0, 1.0, 1.0]), np.array([0.5, 1.0, 1.5]))
    assert means.tolist() == [2.0, np.inf, np.inf]
    beyond = gp.generalized_pareto_cdf(
        np.array([9.0, 0.5]), np.array([1.0, 1.0]), np.array([-0.5, -0.5])
    )
    assert beyond.tolist() == [
        1.0,
        pytest.approx(float(stats.genpareto(c=-0.5, scale=1.0).cdf(0.5)), abs=1e-15),
    ]


def test_quantile_refuses_probabilities_outside_the_open_unit_interval():
    for bad in (0.0, 1.0, -0.1, 1.1, math.nan):
        with pytest.raises(gp.GeneralizedParetoDomainError, match=r"\(0, 1\)"):
            gp.generalized_pareto_quantile(np.array([bad]), np.array([1.0]), np.array([0.3]))


def _draw_excess(rng, n, psi, xi):
    return psi * np.expm1(-xi * np.log(rng.random(n))) / xi


@pytest.mark.parametrize(
    ("psi", "xi", "scale_bound", "shape_bound"),
    [
        # The bounds are the estimator's own sampling spread, not a numerical
        # tolerance.  Above xi = 1/4 the fourth moment diverges, so the sample
        # variance the moment start divides by has infinite variance and the
        # error stops being O(n^{-1/2}): measured worst over twelve seeds at
        # n = 200,000 is 2.3% / 0.016 at xi = 0.30, 0.9% / 0.006 at xi = 0.10
        # and 8.0% / 0.047 at xi = 0.45.
        (1.0, 0.30, 0.05, 0.03),
        (2.0, 0.10, 0.05, 0.03),
        (0.5, 0.45, 0.12, 0.07),
    ],
)
def test_the_moment_initialiser_recovers_the_truth_on_a_large_sample(
    psi, xi, scale_bound, shape_bound
):
    rng = np.random.default_rng(20260902)
    y = _draw_excess(rng, 200_000, psi, xi)
    theta = gp.initialize_generalized_pareto(y, np.ones(len(y)), shape_lower=0.0, shape_upper=1.0)
    assert theta.shape == (len(y), 2) and np.all(theta == theta[0])
    assert abs(float(theta[0, 0]) / psi - 1.0) <= scale_bound
    assert abs(float(theta[0, 1]) - xi) <= shape_bound


def test_the_initialiser_stays_strictly_inside_the_configured_walls():
    rng = np.random.default_rng(4)
    y = _draw_excess(rng, 20_000, 1.0, 0.3)
    for lower, upper in ((0.0, 1.0), (0.05, 0.4), (0.0, 0.2), (0.5, 1.0)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", gp.GeneralizedParetoInitializationWarning)
            theta = gp.initialize_generalized_pareto(
                y, np.ones(len(y)), shape_lower=lower, shape_upper=upper
            )
        assert lower < float(theta[0, 1]) < upper
        assert float(theta[0, 0]) > 0.0


def test_the_initialiser_warns_and_falls_back_for_near_exponential_excesses():
    rng = np.random.default_rng(6)
    y = rng.exponential(1.5, 50_000)  # xi = 0 exactly: the moment start sits on the lower wall
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        theta = gp.initialize_generalized_pareto(
            y, np.ones(len(y)), shape_lower=0.0, shape_upper=1.0
        )
    assert any(
        issubclass(item.category, gp.GeneralizedParetoInitializationWarning) for item in caught
    )
    assert float(theta[0, 1]) == pytest.approx(0.25, abs=1e-12)
    assert float(theta[0, 0]) > 0.0


def test_the_initialiser_does_not_warn_on_an_infinite_variance_sample_but_is_pulled_to_a_half():
    """Measured: at xi = 0.9 the moment estimate lands at ~0.5, inside the walls, so no fallback."""
    rng = np.random.default_rng(20260902)
    y = _draw_excess(rng, 50_000, 1.0, 0.9)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        theta = gp.initialize_generalized_pareto(
            y, np.ones(len(y)), shape_lower=0.0, shape_upper=1.0
        )
    assert not any(
        issubclass(item.category, gp.GeneralizedParetoInitializationWarning) for item in caught
    )
    assert 0.4 <= float(theta[0, 1]) < 0.5


def test_initialiser_frequency_mass_equals_literal_replication():
    y = np.array([0.7, 1.9, 4.2, 0.3, 2.2])
    mass = np.array([1.0, 3.0, 2.0, 1.0, 2.0])
    replicated = np.repeat(y, mass.astype(int))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", gp.GeneralizedParetoInitializationWarning)
        a = gp.initialize_generalized_pareto(y, mass, shape_lower=0.0, shape_upper=1.0)[0]
        b = gp.initialize_generalized_pareto(
            replicated, np.ones(len(replicated)), shape_lower=0.0, shape_upper=1.0
        )[0]
    assert np.allclose(a, b, rtol=0, atol=1e-12)
