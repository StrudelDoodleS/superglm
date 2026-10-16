"""The factored exact SSP moments round the same exact rationals as the row-at-a-time reference.

``_exact_ssp_moments`` forms ``R'(B'MB)R`` and ``R'(B'm)`` from vectorised
exact bin sums; ``tests/_exact_ssp_reference.py`` keeps the row-at-a-time
version it replaced.  Both are exact integer arithmetic rounded to float64
once per output, so they must agree bit for bit on any finite input.
"""

import math
import sys

import numpy as np
import pytest
import scipy.sparse as sp

import superglm._group_matrix._group_matrix_kernels as kernels
from tests._exact_ssp_reference import reference_exact_ssp_moments


def _bits(values):
    return np.asarray(values, dtype=np.float64).view(np.uint64)


def _assert_bitwise(actual, expected):
    for observed, reference in zip(actual, expected, strict=True):
        if reference is None:
            assert observed is None
        else:
            np.testing.assert_array_equal(_bits(observed), _bits(reference))


def _stress_vector(rng, n, *, signed):
    """Ordinary values beside subnormal, 1e-300, 1e-86 and 1e200 rows, and zeros."""
    values = rng.uniform(0.05, 2.0, n)
    draw = rng.random(n)
    tiny = draw < 0.05
    values[tiny] = 1e-86 * rng.uniform(0.5, 2.0, int(np.sum(tiny)))
    values[(draw >= 0.05) & (draw < 0.07)] = 1e-300
    values[(draw >= 0.07) & (draw < 0.08)] = math.ldexp(1.0, -1074)
    values[(draw >= 0.08) & (draw < 0.09)] = math.ldexp(3.0, -1060)
    values[(draw >= 0.09) & (draw < 0.095)] = 1e200
    values[(draw >= 0.095) & (draw < 0.11)] = 0.0
    if signed:
        values *= np.where(rng.random(n) < 0.3, -1.0, 1.0)
    return values


def _banded_support(rng, rows, width, nonzeros):
    support = np.zeros((rows, width))
    starts = rng.integers(0, width - nonzeros + 1, rows)
    for offset in range(nonzeros):
        support[np.arange(rows), starts + offset] = rng.uniform(0.0, 1.0, rows)
    return support


def _transform(rng, k, width):
    transform = np.eye(k, width) + rng.normal(scale=0.3, size=(k, width))
    transform[rng.random((k, width)) < 0.2] = 0.0
    # A large integral entry beside zeros: the scale is then positive.
    transform[0, 0] = math.ldexp(1.0, 60)
    return transform


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("with_rhs", [False, True])
def test_discrete_bins_match_the_reference_bit_for_bit(seed, with_rhs):
    rng = np.random.default_rng(seed)
    n_bins, k, width, n = 257, 9, 8, 6000
    support = _banded_support(rng, n_bins, k, 4)
    bins = rng.integers(0, n_bins - 3, n)
    weights = _stress_vector(rng, n, signed=True)
    rhs = _stress_vector(rng, n, signed=True) if with_rhs else None
    # A bin whose weights cancel exactly, one whose rows are all zero, and a
    # last bin with no rows whose NaN support row nothing may read.
    bins[:2] = n_bins - 3
    weights[:2] = [0.75, -0.75]
    bins[2:4] = n_bins - 2
    weights[2:4] = 0.0
    if rhs is not None:
        rhs[2:4] = 0.0
    support[n_bins - 1, 0] = np.nan
    transform = _transform(rng, k, width)
    expected = reference_exact_ssp_moments(support, transform, weights, rhs, bin_indices=bins)
    actual = kernels._exact_ssp_moments(support, transform, weights, rhs, bin_indices=bins)
    _assert_bitwise(actual, expected)


@pytest.mark.parametrize("with_rhs", [False, True])
def test_sparse_rows_with_duplicate_columns_match_the_reference(with_rhs):
    rng = np.random.default_rng(11)
    n, k, width = 3000, 7, 7
    canonical = sp.csr_matrix(_banded_support(rng, n, k, 3))
    # Two duplicates of row 0's first stored column: their exact sum enters,
    # not a canonicalized, rounded one.
    counts = np.diff(canonical.indptr)
    counts[0] += 2
    data = np.r_[canonical.data[:3], math.ldexp(1.0, -60), 1.0, canonical.data[3:]]
    indices = np.r_[canonical.indices[:3], canonical.indices[[0, 0]], canonical.indices[3:]]
    basis = sp.csr_matrix((data, indices, np.r_[0, np.cumsum(counts)]), shape=(n, k))
    assert not basis.has_canonical_format
    weights = _stress_vector(rng, n, signed=False)
    rhs = _stress_vector(rng, n, signed=True) if with_rhs else None
    transform = _transform(rng, k, width)
    _assert_bitwise(
        kernels._exact_ssp_moments(basis, transform, weights, rhs),
        reference_exact_ssp_moments(basis, transform, weights, rhs),
    )


def test_dense_rows_and_many_bins_match_the_reference():
    rng = np.random.default_rng(23)
    k, width = 6, 5
    transform = _transform(rng, k, width)
    basis = _banded_support(rng, 4000, k, 3)
    weights = _stress_vector(rng, 4000, signed=True)
    _assert_bitwise(
        kernels._exact_ssp_moments(basis, transform, weights, weights),
        reference_exact_ssp_moments(basis, transform, weights, weights),
    )
    support = _banded_support(rng, 20_000, k, 3)
    bins = rng.integers(0, 20_000, 50_000)
    weights = _stress_vector(rng, bins.size, signed=False)
    _assert_bitwise(
        kernels._exact_ssp_moments(support, transform, weights, bin_indices=bins),
        reference_exact_ssp_moments(support, transform, weights, bin_indices=bins),
    )


def test_bin_sums_keep_what_a_rounded_bin_sum_would_absorb():
    """Two bins share a support row; their masses cancel to the one tiny weight.

    The exact Gram is ``2**-60 b b'``; a bin sum rounded to float64 first
    loses ``2**-60`` beside 1 and leaves zero (mutation check: summing the
    bins with ``np.bincount`` before the exact route fails here).
    """
    support = np.array([[0.5, 0.25], [0.5, 0.25], [1.0, 0.0]])
    transform = np.array([[1.0, 0.5], [-0.25, 1.0]])
    bins = np.array([0, 0, 1, 2])
    weights = np.array([1.0, math.ldexp(1.0, -60), -1.0, 0.0])
    expected = reference_exact_ssp_moments(support, transform, weights, weights, bin_indices=bins)
    actual = kernels._exact_ssp_moments(support, transform, weights, weights, bin_indices=bins)
    _assert_bitwise(actual, expected)
    assert actual[0][0, 0] != 0.0


def _python_calls(function, *args, **kwargs):
    calls = [0]

    def profiler(frame, event, arg):
        if event == "call":
            calls[0] += 1

    sys.setprofile(profiler)
    try:
        function(*args, **kwargs)
    finally:
        sys.setprofile(None)
    return calls[0]


@pytest.mark.parametrize("moments", [kernels._exact_ssp_moments, reference_exact_ssp_moments])
def test_bin_aggregation_makes_no_python_call_per_observation(moments):
    """The exact bin sums are array operations, not a Python call per row.

    The row-at-a-time version converts every weight with a Python call, so
    its call count grows with the rows: the reference is the mutation check
    (it must fail the bound, and does).
    """
    rng = np.random.default_rng(5)
    support = _banded_support(rng, 40, 6, 3)
    transform = _transform(rng, 6, 5)

    def calls(n):
        bins = rng.integers(0, 40, n)
        weights = _stress_vector(rng, n, signed=False)
        return _python_calls(moments, support, transform, weights, weights, bin_indices=bins)

    # The reference converts each weight with a Python call, at least one
    # call per added row; the bound allows one per hundred.
    small, large = calls(2_000), calls(32_000)
    vectorised = large - small < (32_000 - 2_000) // 100
    assert vectorised is (moments is kernels._exact_ssp_moments), (small, large)


@pytest.mark.parametrize("width", [1, 3, 29])
def test_a_rectangular_projection_matches_the_reference_and_narrow_ones_stay_narrow(
    width, monkeypatch
):
    """A projected group's ``R`` is ``k x r`` (a decomposed tensor's bilinear subgroup: ``r = 1``).

    Dense 30-column support rows, as a tensor's are. A narrow ``R`` is applied
    to each row first, so the exact Gram is ``r x r`` and not ``k x k`` per
    mass class; a wide one keeps the raw Gram. Both match the reference bit
    for bit. Mutation check: always accumulating the raw ``k x k`` Gram (as
    at 236c0483) fails the width assertion for ``r = 1`` and ``r = 3``.
    """
    rng = np.random.default_rng(40 + width)
    k, n_bins = 30, 200
    support = rng.uniform(0.05, 1.0, (n_bins, k))
    transform = rng.normal(size=(k, width))
    bins = rng.integers(0, n_bins, 5_000)
    weights = _stress_vector(rng, bins.size, signed=False)
    rhs = _stress_vector(rng, bins.size, signed=True)
    widths = []
    accumulate = kernels._exact_gram_moments

    def recorded(*args):
        widths.append(args[-1])
        return accumulate(*args)

    monkeypatch.setattr(kernels, "_exact_gram_moments", recorded)
    _assert_bitwise(
        kernels._exact_ssp_moments(support, transform, weights, rhs, bin_indices=bins),
        reference_exact_ssp_moments(support, transform, weights, rhs, bin_indices=bins),
    )
    assert widths == [width if width < 10 else k]
