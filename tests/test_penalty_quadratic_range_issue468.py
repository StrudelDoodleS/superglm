"""Penalty range quadratics stay finite when only their projection overflows."""

from fractions import Fraction

import numpy as np
import pytest

from superglm.reml.penalty_algebra import _penalty_range_basis, penalty_component_quadratic
from superglm.types import PenaltyComponent


def _component(omega, rank):
    return PenaltyComponent(
        name="x",
        group_name="x",
        group_index=0,
        group_sl=slice(0, len(omega)),
        omega_raw=None,
        omega_ssp=omega,
        rank=float(rank),
    )


@pytest.mark.parametrize("width", [2, 4])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_penalty_quadratic_does_not_overflow_a_finite_range_value(width, sign):
    # The rank-one range projection is sqrt(width) * beta[0], outside
    # binary64, but beta' Omega beta is finite. Before issue468's fix the
    # projection overflowed before the tiny eigenvalue could reduce it.
    omega = np.full((width, width), 1e-310)
    beta = np.full(width, sign * 1.5e308)
    exact = sum(
        Fraction(float(beta[i])) * Fraction(float(omega[i, j])) * Fraction(float(beta[j]))
        for i in range(width)
        for j in range(width)
    )
    expected = float(exact)  # The independent oracle rounds only its final value.
    assert np.isfinite(expected) and expected > 0.0

    actual = penalty_component_quadratic(_component(omega, 1), beta)

    assert np.isfinite(actual)
    # This well-conditioned rank-one value carries O(width*u) backward
    # error plus the subnormal eigenvalue's absolute rounding at the scale
    # of omega. The tolerance permits either eigenvector sign on any BLAS.
    u = 2.0**-53
    smallest = float(np.nextafter(0.0, 1.0))
    tolerance = 32 * width * u + width * smallest / float(omega[0, 0])
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=0.0)


def test_penalty_quadratic_scaling_preserves_the_normal_range_result_bit_for_bit():
    rng = np.random.default_rng(468)
    root = rng.normal(size=(4, 7))
    omega = root.T @ root * 3.7
    beta = rng.normal(size=7) * 250.0
    component = _component(omega, 4)
    values, vectors = _penalty_range_basis(component, omega, 4)
    projection = vectors.T @ beta
    unscaled = float(np.sum(values * projection * projection))

    assert penalty_component_quadratic(component, beta) == unscaled


def test_a_large_null_space_coefficient_does_not_erase_a_small_range_coefficient():
    # Global coefficient normalization would underflow the second entry,
    # although the original projection and every quadratic intermediate
    # are normal. Scaling is needed only for an overflowing projection.
    omega = np.diag([0.0, 1e200])
    beta = np.array([1e308, 1e-200])
    expected = float(Fraction(float(omega[1, 1])) * Fraction(float(beta[1])) ** 2)

    actual = penalty_component_quadratic(_component(omega, 1), beta)

    np.testing.assert_allclose(actual, expected, rtol=8 * 2.0**-53, atol=0.0)


@pytest.mark.parametrize("last_coefficient", [1e-100, 1e103])
def test_overflow_fallback_preserves_penalty_modes_at_different_scales(last_coefficient):
    omega = np.zeros((3, 3))
    omega[:2, :2] = 1e-310
    omega[2, 2] = 1e100
    beta = np.array([1.5e308, 1.5e308, last_coefficient])
    exact = sum(
        Fraction(float(beta[i])) * Fraction(float(omega[i, j])) * Fraction(float(beta[j]))
        for i in range(3)
        for j in range(3)
    )
    expected = float(exact)
    actual = penalty_component_quadratic(_component(omega, 2), beta)
    # Each block's eigenvalue has its own scale. Normalizing both with the
    # largest eigenvalue would erase the small block's finite large energy.
    tolerance = 64 * 2.0**-53 + 4 * np.nextafter(0.0, 1.0) / omega[0, 0]
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=0.0)
