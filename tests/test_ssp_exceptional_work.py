"""Exceptional SSP moments keep exact source arithmetic without row-level rationals."""

import math
from fractions import Fraction

import numpy as np
import pytest
import scipy.sparse as sp

import superglm._group_matrix._group_matrix_core as core
import superglm._group_matrix._group_matrix_discretized as discrete
import superglm._group_matrix._group_matrix_kernels as kernels
from superglm.group_matrix import DesignMatrix, DiscretizedSSPGroupMatrix, SparseSSPGroupMatrix
from tests._exact_reference import exact_matmul, exact_weighted_gram

_U = 2.0**-53


def _gamma(k: int) -> float:
    return k * _U / (1.0 - k * _U)


def _book():
    n, width = 21_000, 10
    coefficients = np.array([0.125, 0.25, 0.25, 0.375])
    support = np.zeros((width, width))
    for row in range(width):
        support[row, (row + np.arange(4)) % width] = coefficients
    bins = np.arange(n) % width
    transform = np.eye(width) + np.random.default_rng(918).normal(scale=0.1, size=(width, width))
    weights = np.ones(n)
    # 300 binades below the rest: no power-of-two shift places the weights
    # in the gate's range, so the exact route serves them.
    weights[0] = math.ldexp(1.0, -300)
    rhs = np.ones(n)
    rhs[0] = -math.ldexp(1.0, -170)
    return support, bins, transform, weights, rhs


def _literal_book_moments(support, transform, weights, rhs):
    """Aggregate the ten repeated source rows independently as exact rationals."""
    width = len(support)
    effective = [
        [
            sum(
                Fraction(float(value)) * Fraction(float(transform[index, column]))
                for index, value in enumerate(row)
            )
            for column in range(width)
        ]
        for row in support
    ]
    masses = [Fraction(len(weights) // width) for _ in range(width)]
    rhs_masses = masses.copy()
    masses[0] += Fraction(float(weights[0])) - 1
    rhs_masses[0] += Fraction(float(rhs[0])) - 1
    gram = np.array(
        [
            [
                float(sum(mass * row[left] * row[right] for mass, row in zip(masses, effective)))
                for right in range(width)
            ]
            for left in range(width)
        ]
    )
    transpose = tuple(
        np.array(
            [
                float(sum(mass * row[column] for mass, row in zip(channel, effective)))
                for column in range(width)
            ]
        )
        for channel in (masses, rhs_masses)
    )
    return gram, *transpose


@pytest.mark.parametrize("route", ["sparse", "discrete", "fused"])
def test_large_exceptional_ssp_uses_rationals_only_for_final_outputs(route, monkeypatch):
    support, bins, transform, weights, rhs = _book()
    expected = _literal_book_moments(support, transform, weights, rhs)
    width = transform.shape[1]
    output_count = width**2 + (2 * width if route == "fused" else 0)
    conversions = []

    def counted(*args):
        conversions.append(1)
        assert len(conversions) <= output_count, "row-level Fraction work returned"
        return Fraction(*args)

    class CountedFraction:
        def __new__(cls, *args):
            return counted(*args)

        from_float = staticmethod(counted)

    calls = []
    exact = kernels._exact_ssp_moments

    def recorded(*args, **kwargs):
        calls.append(len(args[2]))
        return exact(*args, **kwargs)

    monkeypatch.setattr(kernels, "Fraction", CountedFraction)
    monkeypatch.setattr(core, "_exact_ssp_moments", recorded)
    monkeypatch.setattr(discrete, "_exact_ssp_moments", recorded)
    if route == "sparse":
        group = SparseSSPGroupMatrix(sp.csr_matrix(support[bins]), transform)
        actual = (group.gram(weights),)
    else:
        group = DiscretizedSSPGroupMatrix(support, transform, bins)
        actual = group.gram_rmatvec(weights, rhs) if route == "fused" else (group.gram(weights),)
    assert calls == [len(weights)]
    assert conversions
    for observed, reference in zip(actual, expected):
        np.testing.assert_array_equal(observed, reference)


def test_large_ordinary_ssp_still_uses_native_arithmetic(monkeypatch):
    support, bins, transform, weights, _rhs = _book()
    weights[:] = 1.0
    group = SparseSSPGroupMatrix(sp.csr_matrix(support[bins]), transform)

    def forbidden(*args, **kwargs):
        raise AssertionError("ordinary SSP operands entered exceptional arithmetic")

    monkeypatch.setattr(core, "_exact_ssp_moments", forbidden)
    actual = group.gram(weights)
    reference = transform.T @ ((len(weights) // len(support)) * support.T @ support) @ transform
    # Four dyadic basis values and integral repeated-row counts are exact;
    # only the two ten-term transform products incur rounding here.
    u = np.finfo(float).eps / 2
    gamma = 4 * transform.shape[0] * u / (1 - 4 * transform.shape[0] * u)
    magnitude = (
        np.abs(transform).T
        @ ((len(weights) // len(support)) * np.abs(support).T @ np.abs(support))
        @ np.abs(transform)
    )
    assert np.all(np.abs(actual - reference) <= gamma * magnitude)


@pytest.mark.parametrize("route", ["sparse", "discrete", "fused"])
def test_wide_signed_weights_retain_the_least_subnormal_moment(route):
    weights = np.array([math.ldexp(1.0, 1000), np.nextafter(0.0, 1.0), -math.ldexp(1.0, 1000)])
    rhs = np.array([math.ldexp(1.0, 900), math.ldexp(1.0, -1000), -math.ldexp(1.0, 900)])
    if route == "sparse":
        group = SparseSSPGroupMatrix(sp.csr_matrix(np.ones((3, 1))), np.ones((1, 1)))
    else:
        group = DiscretizedSSPGroupMatrix(np.ones((1, 1)), np.ones((1, 1)), np.zeros(3, int))
    if route == "fused":
        gram, xtw, xtwz = group.gram_rmatvec(weights, rhs)
        np.testing.assert_array_equal(xtw, [weights[1]])
        np.testing.assert_array_equal(xtwz, [rhs[1]])
    else:
        gram = group.gram(weights)
    np.testing.assert_array_equal(gram, [[weights[1]]])


@pytest.mark.parametrize("route", ["sparse", "discrete"])
def test_tiny_positive_weight_cannot_be_lost_before_transform_cancellation(route):
    basis = np.array([[1.0, 1.0], [1.0, -1.0]])
    transform = np.array([[1.0], [-1.0]])
    weights = np.array([1.0, 1e-40])
    group = (
        SparseSSPGroupMatrix(sp.csr_matrix(basis), transform)
        if route == "sparse"
        else DiscretizedSSPGroupMatrix(basis, transform, np.arange(2))
    )
    expected = float(4 * Fraction(float(weights[1])))
    np.testing.assert_array_equal(group.gram(weights), [[expected]])
    # Exponent-range admission alone would send this to the lossy sandwich.
    native = transform.T @ (basis.T @ (weights[:, None] * basis)) @ transform
    assert native[0, 0] != expected


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_dyadic_scale_scan_preserves_inactive_source_validation(bad):
    basis = np.array([[bad], [1.0]])
    transform = np.array([[math.ldexp(1.0, 400)]])
    weights = np.array([0.0, math.ldexp(1.0, -800)])
    actual = kernels._exact_ssp_moments(basis, transform, weights)
    np.testing.assert_array_equal(actual[0], [[1.0]])
    with pytest.raises(np.linalg.LinAlgError, match="finite source factors"):
        kernels._exact_ssp_moments(basis, transform, weights, np.ones(2))
    with pytest.raises(np.linalg.LinAlgError, match="finite source factors"):
        kernels._exact_ssp_moments(np.zeros((2, 1)), np.array([[bad]]), np.zeros(2))
    with pytest.raises(np.linalg.LinAlgError, match="finite source factors"):
        kernels._exact_ssp_moments(np.zeros((2, 1)), np.ones((1, 1)), np.array([bad, 0.0]))


def _separated_rows(n: int = 600, bins: int = 12):
    """A support whose last column only the last two bins reach, those bins' rows at 2**-140.

    Rows whose fitted mean ran to zero carry working weights like these
    beside ordinary ones, and the last column's moments then sit near
    1e-42, so a bound read there is relative to them.
    """
    rng = np.random.default_rng(140)
    basis = np.zeros((bins, 4))
    basis[: bins - 2, :3] = rng.uniform(0.0, 1.0, (bins - 2, 3))
    basis[bins - 2 :, 3] = rng.uniform(0.5, 1.0, 2)
    transform = np.zeros((4, 4))
    transform[:3, :3] = np.eye(3) + 0.3 * rng.standard_normal((3, 3))
    transform[3, 3] = 1.7
    index = np.arange(n) % bins
    separated = index >= bins - 2
    weights = rng.uniform(0.5, 2.0, n)
    weights[separated] *= math.ldexp(1.0, -140)
    rhs = rng.standard_normal(n)
    rhs[separated] *= math.ldexp(1.0, -150)
    return DiscretizedSSPGroupMatrix(basis, transform, index), weights, rhs


@pytest.mark.parametrize("route", ["gram", "fused", "assembly"])
def test_tiny_working_weights_take_the_ordinary_route_at_a_power_of_two_shift(route, monkeypatch):
    """Weights 2**-140 below the rest take the ordinary route at ``w' = 2**k w``.

    The gate (every operand in ``[2**-128, 2**128]``) sent such a block to
    the exact rational route.  On ``w'`` the ordinary route obeys the
    standard model, so ``2**-k`` times its moments meet ``|M^ - M| <=
    gamma_{n+m} |S|'(|S| o |w|_b)`` against the exact moments of its support
    ``S = fl(B R)``, ``|w|_b`` the bin sums of ``|w|`` (``_shifted_moments``),
    for the gram, ``X'w`` and ``X'(wz)`` alike.  Mutation checks: returning
    the shifted moments unshifted, or shifting ``X'(wz)`` back by the
    weights' shift, leaves the bound by many orders; without the shifted
    route the exact one is called.
    """
    group, weights, rhs = _separated_rows()
    exact = []
    original = discrete._exact_ssp_moments

    def recorded(*args, **kwargs):
        exact.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(discrete, "_exact_ssp_moments", recorded)
    if route == "gram":
        moments = (group.gram(weights),)
    elif route == "fused":
        moments = group.gram_rmatvec(weights, rhs)
    else:
        plan = DesignMatrix([group], len(weights), 4).execution_plan
        result = plan.moments(weights, rhs=(rhs,), include_xtw=True)
        moments = (result.gram, result.xtw, result.xt_rhs[0])
    assert not exact
    support = group.B_unique @ group.R_inv
    rows = support[group.bin_idx]
    n, m = len(weights), group.n_bins
    gamma = _gamma(n + m)
    mass = np.bincount(group.bin_idx, weights=np.abs(weights), minlength=m)
    rhs_mass = np.bincount(group.bin_idx, weights=np.abs(rhs), minlength=m)
    references = (
        (
            exact_weighted_gram(rows, rows, weights),
            np.abs(support).T @ (np.abs(support) * mass[:, None]),
        ),
        (exact_matmul((rows.T, weights)), np.abs(support).T @ mass),
        (exact_matmul((rows.T, rhs)), np.abs(support).T @ rhs_mass),
    )
    for moment, (reference, magnitude) in zip(moments, references):
        assert np.all(np.abs(moment - reference) <= gamma * magnitude)
    # The separated column's moments are themselves of order 2**-140.
    assert 0.0 < moments[0][3, 3] < math.ldexp(1.0, -130)


def test_a_shift_back_that_would_round_keeps_the_exact_route(monkeypatch):
    """Weights the shift places, whose moments would go subnormal shifted back, stay exact.

    ``2**-k`` times the shifted route's moments is exact only while it stays
    normal; at weights ``2**-1000`` and a basis near ``2**-40`` the gram is
    near ``2**-1080``, so the shift back would round and the exact route
    keeps the once-rounded moments.  Mutation check: without the round-trip
    test the shifted route is taken.
    """
    group = DiscretizedSSPGroupMatrix(
        np.full((2, 1), math.ldexp(1.0, -40)) * np.array([[1.0], [3.0]]),
        np.ones((1, 1)),
        np.array([0, 1, 1]),
    )
    weights = np.full(3, math.ldexp(1.0, -1000)) * np.array([1.0, 1.5, 1.25])
    exact = []
    original = discrete._exact_ssp_moments

    def recorded(*args, **kwargs):
        exact.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(discrete, "_exact_ssp_moments", recorded)
    gram = group.gram(weights)
    assert exact == [1]
    np.testing.assert_array_equal(
        gram, original(group.B_unique, group.R_inv, weights, bin_indices=group.bin_idx)[0]
    )
