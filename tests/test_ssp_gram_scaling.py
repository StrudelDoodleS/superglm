"""Gram invariants of equivalent factored SSP designs."""

import math
from itertools import permutations

import numpy as np
import pytest
import scipy.sparse as sp

from superglm.group_matrix import (
    DenseGroupMatrix,
    DesignMatrix,
    DiscretizedSSPGroupMatrix,
    SparseSSPGroupMatrix,
)


def _group(kind, basis, transform):
    if kind == "discrete":
        return DiscretizedSSPGroupMatrix(basis, transform, np.arange(len(basis)))
    return SparseSSPGroupMatrix(sp.csr_matrix(basis), transform)


@pytest.mark.parametrize("kind", ["discrete", "sparse"])
@pytest.mark.parametrize("exponent", [0, 600, -600])
def test_ssp_gram_preserves_equivalent_reciprocal_basis_scaling(kind, exponent):
    group = _group(
        kind, np.array([[math.ldexp(1.0, exponent)]]), np.array([[math.ldexp(1.0, -exponent)]])
    )
    expected_design = np.ones((1, 1))
    np.testing.assert_array_equal(group.toarray(), expected_design)
    expected = DenseGroupMatrix(expected_design).gram(np.ones(1))
    actual = group.gram(np.ones(1))
    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual, expected, rtol=8 * np.finfo(float).eps, atol=0)


@pytest.mark.parametrize("exponent", [0, 600, -600])
def test_fused_ssp_gram_and_execution_share_the_represented_design(exponent):
    group = _group(
        "discrete",
        np.array([[math.ldexp(1.0, exponent)]]),
        np.array([[math.ldexp(1.0, -exponent)]]),
    )
    weights = np.ones(1)
    rhs = np.array([2.0])
    gram, xtw, xtrhs = group.gram_rmatvec(weights, rhs)
    np.testing.assert_allclose(gram, [[1.0]], rtol=8 * np.finfo(float).eps, atol=0)
    np.testing.assert_allclose(xtw, [1.0], rtol=8 * np.finfo(float).eps, atol=0)
    np.testing.assert_allclose(xtrhs, [2.0], rtol=8 * np.finfo(float).eps, atol=0)
    moments = DesignMatrix([group], 1, 1).execution_plan.moments(
        weights, rhs=(rhs,), include_xtw=True
    )
    np.testing.assert_allclose(moments.gram, [[1.0]], rtol=8 * np.finfo(float).eps, atol=0)


@pytest.mark.parametrize("kind", ["discrete", "sparse"])
@pytest.mark.parametrize("weight_exponent", [-600, 600])
def test_ssp_weighted_root_retains_finite_small_and_large_weight_answers(kind, weight_exponent):
    # sqrt(W)*X=1 exactly, although a raw basis square can overflow or vanish.
    basis_exponent = -weight_exponent
    transform_exponent = weight_exponent // 2
    group = _group(
        kind,
        np.array([[math.ldexp(1.0, basis_exponent)]]),
        np.array([[math.ldexp(1.0, transform_exponent)]]),
    )
    actual = group.gram(np.array([math.ldexp(1.0, weight_exponent)]))
    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual, [[1.0]], rtol=8 * np.finfo(float).eps, atol=0)


def test_discrete_ssp_does_not_overflow_bin_mass_before_a_finite_gram():
    group = DiscretizedSSPGroupMatrix(
        np.array([[math.ldexp(1.0, -512)]]), np.ones((1, 1)), np.array([0, 0])
    )
    weights = np.full(2, math.ldexp(1.0, 1023))
    gram, xtw, xtrhs = group.gram_rmatvec(weights, weights)
    # The exact bin mass is 2**1024, but the three requested outputs are finite.
    np.testing.assert_allclose(gram, [[1.0]], rtol=8 * np.finfo(float).eps, atol=0)
    np.testing.assert_allclose(xtw, [math.ldexp(1.0, 512)], rtol=8 * np.finfo(float).eps, atol=0)
    np.testing.assert_array_equal(xtrhs, xtw)


@pytest.mark.parametrize("kind", ["discrete", "sparse"])
def test_exceptional_ssp_gram_preserves_signed_weight_cancellation(kind):
    scale = math.ldexp(1.0, 600)
    group = _group(kind, scale * np.array([[1.0], [2.0]]), np.array([[1.0 / scale]]))
    # X=(1,2), so X.T diag(1,-1/8) X = 1/2, an exactly positive scalar.
    np.testing.assert_allclose(
        group.gram(np.array([1.0, -0.125])), [[0.5]], rtol=8 * np.finfo(float).eps, atol=0
    )


@pytest.mark.parametrize("kind", ["discrete", "sparse"])
def test_ssp_source_product_cancellation_can_have_a_finite_exact_gram(kind):
    a = math.ldexp(1.0, 600)
    # Exact X=a*a-a*a+1=1. Neither individual large product is representable.
    group = _group(kind, np.array([[a, a, 1.0]]), np.array([[a], [-a], [1.0]]))
    np.testing.assert_array_equal(group.gram(np.ones(1)), [[1.0]])


def test_ssp_gram_does_not_require_an_unrequested_transpose_moment():
    group = DiscretizedSSPGroupMatrix(np.array([[0.75]]), np.ones((1, 1)), np.zeros(3, int))
    weights = np.full(3, math.ldexp(1.0, 1023))
    expected = math.ldexp(27.0 / 16.0, 1023)
    np.testing.assert_allclose(
        group.gram(weights), [[expected]], rtol=8 * np.finfo(float).eps, atol=0
    )
    with pytest.raises(np.linalg.LinAlgError, match="moment.*representable"):
        group.gram_rmatvec(weights, weights)


@pytest.mark.parametrize("kind", ["discrete", "sparse"])
def test_ordinary_ssp_gram_keeps_the_native_arithmetic_route(kind, monkeypatch):
    import superglm._group_matrix._group_matrix_core as core
    import superglm._group_matrix._group_matrix_discretized as discrete

    def forbidden(*args, **kwargs):
        raise AssertionError("ordinary SSP operands must not use exact arithmetic")

    monkeypatch.setattr(core, "_exact_ssp_moments", forbidden)
    monkeypatch.setattr(discrete, "_exact_ssp_moments", forbidden)
    group = _group(kind, np.array([[1.0, 0.0], [1.0, 1.0]]), np.eye(2))
    np.testing.assert_array_equal(group.gram(np.ones(2)), [[2.0, 1.0], [1.0, 1.0]])
    if kind == "discrete":
        np.testing.assert_array_equal(
            group.gram_rmatvec(np.ones(2), np.ones(2))[0], [[2.0, 1.0], [1.0, 1.0]]
        )


@pytest.mark.parametrize(
    "values", list(permutations([math.ldexp(1.0, 600), 1.0, -math.ldexp(1.0, 600)]))
)
def test_exceptional_ssp_gram_sums_stored_csr_duplicates_exactly(values):
    # CSR duplicates denote their sum: every storage order represents X=1.
    basis = sp.csr_matrix((np.array(values), np.zeros(3, int), np.array([0, 3])), shape=(1, 1))
    assert not basis.has_canonical_format
    group = SparseSSPGroupMatrix(basis, np.ones((1, 1)))
    np.testing.assert_array_equal(group.gram(np.ones(1)), [[1.0]])
    np.testing.assert_array_equal(group.B.data, values)
    np.testing.assert_array_equal(group.B.indices, np.zeros(3, int))


@pytest.mark.parametrize("operand", ["basis", "transform", "weights"])
def test_ssp_range_scan_preserves_supported_float16_native_gram(operand, monkeypatch):
    import superglm._group_matrix._group_matrix_discretized as discrete

    def forbidden(*args, **kwargs):
        raise AssertionError("ordinary float16 inputs must keep native moment arithmetic")

    monkeypatch.setattr(discrete, "_exact_ssp_moments", forbidden)
    operands = {
        "basis": np.array([[1.0, 0.0], [1.0, 1.0]]),
        "transform": np.eye(2),
        "weights": np.ones(2),
    }
    operands[operand] = operands[operand].astype(np.float16)
    group = _group("discrete", operands["basis"], operands["transform"])
    np.testing.assert_array_equal(group.gram(operands["weights"]), [[2.0, 1.0], [1.0, 1.0]])
    if operand != "weights":
        gram, xtw, xtrhs = group.gram_rmatvec(operands["weights"], np.ones(2))
        np.testing.assert_array_equal(gram, [[2.0, 1.0], [1.0, 1.0]])
        np.testing.assert_array_equal(xtw, [2.0, 1.0])
        np.testing.assert_array_equal(xtrhs, xtw)


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize(
    ("bins", "weights"),
    [([-1], [1.0]), ([1], [1.0]), ([0], [1.0, 2.0]), ([0], [])],
)
def test_exceptional_discrete_ssp_preserves_bin_and_weight_domain(bins, weights, fused):
    group = DiscretizedSSPGroupMatrix(
        np.array([[math.ldexp(1.0, 600)]]),
        np.array([[math.ldexp(1.0, -600)]]),
        np.array(bins),
    )
    with pytest.raises(ValueError):
        if fused:
            group.gram_rmatvec(np.array(weights), np.ones(len(bins)))
        else:
            group.gram(np.array(weights))


@pytest.mark.parametrize("rhs", [[], [1.0, 2.0]])
def test_exceptional_discrete_ssp_requires_one_rhs_per_observation(rhs):
    group = DiscretizedSSPGroupMatrix(
        np.array([[math.ldexp(1.0, 600)]]),
        np.array([[math.ldexp(1.0, -600)]]),
        np.zeros(1, int),
    )
    with pytest.raises(ValueError):
        group.gram_rmatvec(np.ones(1), np.array(rhs))
