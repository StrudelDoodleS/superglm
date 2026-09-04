"""Primitive log-normal kernel: references, reuse, information and functionals."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from superglm.distributional.kernels import gaussian as gaussian_kernel
from superglm.distributional.kernels import log_normal as ln
from tests._log_normal_ls_oracles import (
    PACKED,
    mp_location_derivatives,
    mp_log_optimizing,
    scipy_log_density,
)

HALF_LOG_TWO_PI = 0.5 * math.log(2.0 * math.pi)
_POINTS = (
    (3.7, 0.7, 0.9),
    (0.4, 0.7, 0.9),
    (12.0, 1.1, 1.3),
    (2.0, 0.2, 0.5),
    (5.0, 0.9, 2.0),
    (0.8, 0.3, 0.7),
    (2.5, 0.5, 0.011),
    (60.0, 0.5, 0.6),
    (1.0, 0.0, 0.05),
)


def _rows(y, mu, sigma, multiplier=None, *, order=2):
    y = np.asarray(y, dtype=float)
    n = len(y)
    multiplier = np.ones(n) if multiplier is None else np.asarray(multiplier, dtype=float)
    return ln.location_rows(
        y, np.full(n, mu), np.full(n, sigma), multiplier, derivative_order=order
    )


def test_optimizing_plus_carrier_is_the_scipy_log_normal_density():
    grid = np.array([0.05, 0.4, 1.0, 2.5, 9.0, 60.0, 1000.0])
    for mu, sigma in ((0.7, 0.9), (-1.0, 0.3), (2.0, 1.7), (0.0, 0.011), (3.0, 2.5)):
        evaluated = _rows(grid, mu, sigma, order=0)
        carrier = -np.log(grid) - HALF_LOG_TWO_PI
        reference = scipy_log_density(grid, mu, sigma)
        assert np.allclose(
            evaluated.optimizing_log_likelihood + carrier, reference, rtol=1e-14, atol=0
        )


def test_location_score_and_hessian_match_mpmath():
    mp = pytest.importorskip("mpmath")
    worst = 0.0
    for y, mu, sigma in _POINTS:
        evaluated = _rows([y], mu, sigma)
        reference_score, reference_hessian = mp_location_derivatives(mp, y, mu, sigma)
        assert math.isclose(
            float(evaluated.optimizing_log_likelihood[0]),
            mp_log_optimizing(mp, y, mu, sigma),
            rel_tol=1e-14,
        )
        for computed, reference in (
            (evaluated.score[0], reference_score),
            (evaluated.hessian_packed[0], reference_hessian),
        ):
            error = np.abs(computed - reference) / (1.0 + np.abs(reference))
            worst = max(worst, float(error.max()))
            assert error.max() <= 1e-13
    assert worst <= 1e-13  # measured 3.7e-15 by research/log-normal_formulas.py


def test_location_channels_are_the_gaussian_kernels_at_log_y():
    """The reuse claim, pinned: only the half-log-2-pi constant differs."""
    y = np.array([0.4, 1.3, 2.8, 7.5, 40.0])
    mu, sigma, multiplier = 0.3, 0.8, np.array([1.0, 2.0, 1.0, 3.0, 1.0])
    evaluated = _rows(y, mu, sigma, multiplier)
    reference = gaussian_kernel.evaluate_gaussian_rows(
        np.log(y),
        np.full(len(y), mu),
        np.full(len(y), sigma),
        multiplier,
        "frequency",
        derivative_order=2,
    )
    assert np.allclose(
        evaluated.optimizing_log_likelihood,
        reference.optimizing_log_likelihood + multiplier * HALF_LOG_TWO_PI,
        rtol=0,
        atol=1e-14,
    )
    assert np.array_equal(evaluated.score, reference.score)
    assert np.array_equal(evaluated.hessian_packed, reference.hessian_packed)


def test_multiplier_replicates_every_channel_and_outputs_are_read_only():
    y = np.array([0.7, 1.9, 4.2])
    counts = np.array([1.0, 3.0, 2.0])
    weighted = _rows(y, 0.4, 0.9, counts)
    unit = _rows(np.repeat(y, counts.astype(int)), 0.4, 0.9)
    for index, count in enumerate(counts.astype(int)):
        start = int(counts[:index].sum())
        assert math.isclose(
            float(weighted.optimizing_log_likelihood[index]),
            float(unit.optimizing_log_likelihood[start : start + count].sum()),
            rel_tol=1e-14,
        )
        assert np.allclose(
            weighted.score[index], unit.score[start : start + count].sum(0), rtol=1e-14, atol=0
        )
    assert not weighted.score.flags.writeable
    assert not weighted.hessian_packed.flags.writeable
    assert weighted.valid.dtype == np.bool_ and weighted.valid.all()


def test_exactly_the_requested_derivative_order_is_returned():
    for order, has_score, has_hessian in ((0, False, False), (1, True, False), (2, True, True)):
        evaluated = _rows([1.0, 2.0], 0.3, 0.8, order=order)
        assert (evaluated.score is not None) is has_score
        assert (evaluated.hessian_packed is not None) is has_hessian
    with pytest.raises(ValueError):
        _rows([1.0, 2.0], 0.3, 0.8, order=True)


def test_row_arguments_are_validated():
    with pytest.raises(ln.LogNormalDomainError):
        _rows([1.0, -2.0], 0.3, 0.8)
    with pytest.raises(ln.LogNormalDomainError):
        _rows([1.0, 2.0], 0.3, 0.0)
    with pytest.raises(ln.LogNormalDomainError):
        ln.location_rows(
            np.array([1.0, 2.0]),
            np.array([0.3]),
            np.array([0.8, 0.8]),
            np.ones(2),
            derivative_order=0,
        )


def test_location_expected_information_is_the_diagonal_and_matches_monte_carlo():
    sigma = np.array([0.4, 0.9, 1.6])
    information = ln.location_expected_information(sigma, np.ones(3))
    assert np.allclose(information[:, 0], 1.0 / sigma**2, rtol=0, atol=1e-15)
    assert np.array_equal(information[:, 1], np.zeros(3))
    assert np.allclose(information[:, 2], 2.0 / sigma**2, rtol=0, atol=1e-15)
    assert np.array_equal(
        information, gaussian_kernel.gaussian_expected_information(sigma, np.ones(3), "frequency")
    )
    rng = np.random.default_rng(20260902)
    n = 200_000
    for mu, scale in ((0.3, 0.9), (-0.5, 0.4)):
        y = np.exp(mu + scale * rng.standard_normal(n))
        evaluated = ln.location_rows(
            y, np.full(n, mu), np.full(n, scale), np.ones(n), derivative_order=2
        )
        analytic = ln.location_expected_information(np.array([scale]), np.array([1.0]))[0]
        negated = -np.asarray(evaluated.hessian_packed)
        outer = np.stack([evaluated.score[:, i] * evaluated.score[:, j] for i, j in PACKED], axis=1)
        for sample in (negated, outer):
            deviation = sample.std(0)
            # H_mumu = -1/sigma^2 does not depend on the draw: its standard error is
            # zero up to np.std's summation round-off, so it is checked exactly.
            deterministic = deviation <= 1e-9 * np.maximum(np.abs(sample.mean(0)), 1.0)
            assert np.allclose(
                sample.mean(0)[deterministic], analytic[deterministic], rtol=1e-9, atol=0
            )
            stochastic = ~deterministic
            z = (sample.mean(0)[stochastic] - analytic[stochastic]) / (
                deviation[stochastic] / math.sqrt(n)
            )
            assert np.abs(z).max() <= 5.0


def test_cdf_and_quantile_match_scipy_and_round_trip():
    p = np.array([1e-8, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0 - 1e-8])
    for mu, sigma in ((0.7, 0.9), (-1.0, 0.3), (2.0, 1.7)):
        location = np.full(len(p), mu)
        scale = np.full(len(p), sigma)
        quantile = ln.log_normal_quantile(p, location, scale)
        reference = stats.lognorm(s=sigma, scale=math.exp(mu)).ppf(p)
        assert np.allclose(quantile, reference, rtol=1e-13, atol=0)
        assert np.allclose(ln.log_normal_cdf(quantile, location, scale), p, rtol=1e-12, atol=0)
        assert np.allclose(
            ln.log_normal_cdf(quantile, location, scale),
            stats.lognorm(s=sigma, scale=math.exp(mu)).cdf(quantile),
            rtol=1e-13,
            atol=0,
        )
    with pytest.raises(ln.LogNormalDomainError):
        ln.log_normal_quantile(np.array([0.0, 0.5]), np.zeros(2), np.ones(2))
    with pytest.raises(ln.LogNormalDomainError):
        ln.log_normal_quantile(np.array([0.5, 1.0]), np.zeros(2), np.ones(2))


from tests._log_normal_ls_oracles import mp_log_mean_loading, mp_mean_derivatives  # noqa: E402

_MEAN_POINTS = (
    (3.7, 2.5, 0.9),
    (0.4, 2.5, 0.9),
    (2.0, 1.4, 0.5),
    (5.0, 3.0, 1.2),
    (9.0, 2.0, 0.8),
    (1.0, 1.0, 2.5),
    (2.5, 1.9, 0.011),
    (1e4, 500.0, 1.5),
)


def _mean_rows(y, mean, sigma, multiplier=None, *, order=2):
    y = np.asarray(y, dtype=float)
    n = len(y)
    multiplier = np.ones(n) if multiplier is None else np.asarray(multiplier, dtype=float)
    return ln.mean_rows(y, np.full(n, mean), np.full(n, sigma), multiplier, derivative_order=order)


def test_mean_loading_matches_the_quadrature_reference():
    mp = pytest.importorskip("mpmath")
    for sigma in (0.011, 0.3, 0.9, 1.5, 2.5):
        loading, first, second = ln.log_mean_loading(np.array([sigma]))
        reference = mp_log_mean_loading(mp, sigma)
        for computed, expected in zip(
            (float(loading[0]), float(first[0]), float(second[0])), reference, strict=True
        ):
            assert abs(computed - expected) <= 1e-12 * (1.0 + abs(expected))


def test_mean_and_location_coordinates_invert_each_other():
    mean = np.array([0.5, 2.0, 137.0])
    sigma = np.array([0.2, 0.9, 1.6])
    location = ln.location_of_mean(mean, sigma)
    assert np.allclose(location, np.log(mean) - 0.5 * sigma**2, rtol=0, atol=1e-15)
    assert np.allclose(ln.mean_of_location(location, sigma), mean, rtol=1e-14, atol=0)
    # the mean always exists, but it may exceed float64 and that is a legitimate answer
    # (exp(752) overflows; exp(702) does not, so the probe has to be past 709.78)
    assert math.isinf(float(ln.mean_of_location(np.array([750.0]), np.array([2.0]))[0]))


def test_mean_form_score_and_hessian_match_mpmath():
    mp = pytest.importorskip("mpmath")
    worst = 0.0
    for y, mean, sigma in _MEAN_POINTS:
        evaluated = _mean_rows([y], mean, sigma)
        reference_score, reference_hessian = mp_mean_derivatives(mp, y, mean, sigma)
        for computed, reference in (
            (evaluated.score[0], reference_score),
            (evaluated.hessian_packed[0], reference_hessian),
        ):
            error = np.abs(computed - reference) / (1.0 + np.abs(reference))
            worst = max(worst, float(error.max()))
            assert error.max() <= 1e-13
    assert worst <= 1e-13  # measured 1.1e-14 by research/log-normal_formulas.py


def test_mean_form_is_the_location_form_at_the_same_law():
    """Same distribution, different coordinates: the optimising channel is unchanged."""
    y = np.array([0.4, 1.3, 2.8, 7.5])
    mean, sigma = 2.0, 0.8
    location = float(ln.location_of_mean(np.array([mean]), np.array([sigma]))[0])
    assert np.allclose(
        _mean_rows(y, mean, sigma, order=0).optimizing_log_likelihood,
        _rows(y, location, sigma, order=0).optimizing_log_likelihood,
        rtol=0,
        atol=1e-14,
    )


def test_mean_expected_information_is_the_jacobian_congruence():
    mean = np.array([1.0, 2.5, 7.0])
    sigma = np.array([0.4, 0.9, 1.6])
    information = ln.mean_expected_information(mean, sigma, np.ones(3))
    assert np.allclose(information[:, 0], 1.0 / (sigma**2 * mean**2), rtol=1e-14, atol=0)
    assert np.allclose(information[:, 1], -1.0 / (sigma * mean), rtol=1e-14, atol=0)
    assert np.allclose(information[:, 2], 1.0 + 2.0 / sigma**2, rtol=1e-14, atol=0)
    for row, (m, s) in enumerate(zip(mean, sigma, strict=True)):
        jacobian = np.array([[1.0 / m, -s], [0.0, 1.0]])
        location = ln.location_expected_information(np.array([s]), np.array([1.0]))[0]
        dense = np.array([[location[0], location[1]], [location[1], location[2]]])
        congruent = jacobian.T @ dense @ jacobian
        assert np.allclose(
            information[row],
            [congruent[0, 0], congruent[0, 1], congruent[1, 1]],
            rtol=1e-13,
            atol=0,
        )
        assert np.linalg.det(congruent) == pytest.approx(2.0 / (s**4 * m**2), rel=1e-12)


def test_information_conditioning_is_recorded_before_any_fit():
    """The log-normal information remains positive definite across the checked scale."""
    expected_mean_condition = {0.5: 2.9413, 0.9: 5.0604, 1.5: 11.6957}
    for sigma in (0.5, 0.9, 1.5):
        location = ln.location_expected_information(np.array([sigma]), np.array([1.0]))[0]
        dense = np.array([[location[0], location[1]], [location[1], location[2]]])
        eigenvalues = np.linalg.eigvalsh(dense)
        assert eigenvalues[0] > 0.0
        assert eigenvalues[-1] / eigenvalues[0] == pytest.approx(2.0, rel=1e-12)
        for mean in (1.0, 5.0):
            packed = ln.mean_expected_information(
                np.array([mean]), np.array([sigma]), np.array([1.0])
            )[0]
            dense_mean = np.array([[packed[0], packed[1]], [packed[1], packed[2]]])
            eigenvalues = np.linalg.eigvalsh(dense_mean)
            assert eigenvalues[0] > 0.0
            assert np.linalg.det(dense_mean) == pytest.approx(2.0 / (sigma**4 * mean**2), rel=1e-12)
            if mean == 1.0:
                assert eigenvalues[-1] / eigenvalues[0] == pytest.approx(
                    expected_mean_condition[sigma], rel=1e-4
                )


def test_initialiser_recovers_the_log_moments_in_both_forms():
    rng = np.random.default_rng(11)
    y = np.exp(0.7 + 1.3 * rng.standard_normal(4000))
    mass = np.ones(len(y))
    location_start = ln.initialize_log_normal(y, mass, parametrisation="location", scale_floor=0.01)
    assert location_start.shape == (len(y), 2)
    assert not location_start.flags.writeable
    assert np.allclose(location_start[:, 0], np.mean(np.log(y)), rtol=0, atol=1e-14)
    assert np.allclose(location_start[:, 1], np.std(np.log(y)), rtol=1e-14, atol=0)
    mean_start = ln.initialize_log_normal(y, mass, parametrisation="mean", scale_floor=0.01)
    assert np.allclose(
        ln.location_of_mean(mean_start[:, 0], mean_start[:, 1]),
        location_start[:, 0],
        rtol=0,
        atol=1e-12,
    )


def test_initialiser_weights_are_replication_and_the_floor_binds():
    y = np.array([0.7, 1.9, 4.2])
    counts = np.array([1.0, 3.0, 2.0])
    weighted = ln.initialize_log_normal(y, counts, parametrisation="location", scale_floor=0.01)
    replicated = ln.initialize_log_normal(
        np.repeat(y, counts.astype(int)),
        np.ones(int(counts.sum())),
        parametrisation="location",
        scale_floor=0.01,
    )
    assert np.allclose(weighted[0], replicated[0], rtol=1e-14, atol=0)
    flat = np.full(50, 3.0)
    floored = ln.initialize_log_normal(flat, np.ones(50), parametrisation="mean", scale_floor=0.25)
    floored_location = ln.initialize_log_normal(
        flat, np.ones(50), parametrisation="location", scale_floor=0.25
    )
    assert float(floored[0, 1]) > 0.25
    # the floor is applied before the mean is formed, so on a floor-binding sample
    # the mean-form start still maps back to exactly the location-form start
    assert float(floored[0, 1]) == float(floored_location[0, 1])
    assert float(floored_location[0, 0]) == pytest.approx(math.log(3.0), abs=1e-12)
    assert float(ln.location_of_mean(floored[:1, 0], floored[:1, 1])[0]) == pytest.approx(
        math.log(3.0), abs=1e-12
    )
    with pytest.raises(ln.LogNormalDomainError):
        ln.initialize_log_normal(y, counts, parametrisation="median", scale_floor=0.01)
