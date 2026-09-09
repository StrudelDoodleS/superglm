"""Signed factor-smooth cross-products and their bounded native dispatch."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from superglm._group_matrix import _group_matrix_algebra as algebra
from superglm.group_matrix import DenseGroupMatrix, FactorSmoothGroupMatrix


def _fixture(discrete, factor_basis, *, q=3):
    support = np.array([[1.0, 0.0, 0.25, 0.0], [0.5, 0.5, 0.0, 0.0], [0.0, 0.25, 0.5, 0.25]])
    bins = np.tile(np.arange(3), 8)
    # Level 1 is unused; the last (SZ reference) level has observations.
    codes = np.tile([0, 2, 3, 0, 3, 2], 4)
    natural_map = np.array([[1.0, -0.5], [0.25, 1.0], [-0.5, 0.25], [0.5, -1.0]])
    gm = FactorSmoothGroupMatrix(
        support if discrete else sp.csr_matrix(support[bins]),
        codes,
        4,
        natural_map=natural_map,
        levels=("a", "unused", "c", "reference"),
        repeated_penalty_components=(("wiggle", np.eye(2)),),
        factor_basis=factor_basis,
        bin_idx=bins if discrete else None,
    )
    rng = np.random.default_rng(834)
    # Keep a non-contiguous dense view: the native kernels accept strides.
    dense = DenseGroupMatrix(rng.normal(size=(len(codes), 2 * q))[:, ::2])
    weights = np.tile([0.0, -2.0, 1.0, 0.5, -0.25, 3.0], 4)
    # Construct the stored-design oracle independently of matrix methods and
    # the production contrast helper, including a rectangular natural map.
    level_design = np.eye(4)[codes]
    if factor_basis == "sz":
        level_design = level_design[:, :-1] - level_design[:, -1:]
    reference = (level_design[:, :, None] * (support[bins] @ natural_map)[:, None, :]).reshape(
        len(codes), -1
    )
    absolute_design = (
        np.abs(level_design[:, :, None]) * (np.abs(support[bins]) @ np.abs(natural_map))[:, None, :]
    ).reshape(len(codes), -1)
    return gm, dense, weights, reference, absolute_design


@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("factor_basis", ["fs", "sz"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("zero_weights", [False, True])
def test_signed_cross_matches_stored_design(discrete, factor_basis, reverse, zero_weights):
    gm, dense, weights, reference, absolute_design = _fixture(discrete, factor_basis)
    if zero_weights:
        weights[:] = 0.0
    expected = reference.T @ (weights[:, None] * dense.M)
    scale = np.linalg.norm(absolute_design.T @ (np.abs(weights[:, None] * dense.M)), ord=np.inf)
    # Forward error is bounded by absolute products, not by a potentially
    # cancelled answer. Account for both observation and raw-map reductions.
    tolerance = 8 * (len(weights) + gm.raw_width + 2) * np.finfo(float).eps * scale
    left, right = (dense, gm) if reverse else (gm, dense)
    actual = algebra._cross_gram(left, right, weights)
    np.testing.assert_allclose(actual, expected.T if reverse else expected, rtol=0, atol=tolerance)


@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("factor_basis", ["fs", "sz"])
@pytest.mark.parametrize("reverse", [False, True])
def test_narrow_cross_uses_native_kernel_without_expansion(
    monkeypatch, discrete, factor_basis, reverse
):
    gm, dense, weights, _, _ = _fixture(discrete, factor_basis)

    def forbidden(*args, **kwargs):
        raise AssertionError("eligible factor-smooth cross must use its native batched kernel")

    monkeypatch.setattr(algebra, "_cross_gram_by_columns", forbidden)
    monkeypatch.setattr(FactorSmoothGroupMatrix, "toarray", forbidden)
    monkeypatch.setattr(DenseGroupMatrix, "toarray", forbidden)
    monkeypatch.setattr(sp.csr_matrix, "toarray", forbidden)
    profile = {}
    left, right = (dense, gm) if reverse else (gm, dense)
    result = algebra._cross_gram(left, right, weights, profile=profile)
    assert result.shape == (left.shape[1], right.shape[1])
    assert profile["block_cross_factor_smooth_dense_calls"] == 1
    assert "block_cross_fallback_s" not in profile


@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("factor_basis", ["fs", "sz"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "case",
    [
        "cap",
        "cap_raw",
        "singleton",
        "wide",
        "float32",
        "weight_float32",
        "weight_subclass",
        "dense_subclass",
        "fs_subclass",
    ],
)
def test_ineligible_cross_preserves_column_fallback(
    monkeypatch, discrete, factor_basis, case, reverse
):
    q = 1 if case == "singleton" else 9 if case == "wide" else 3
    gm, dense, weights, reference, _ = _fixture(discrete, factor_basis, q=q)
    if case == "cap":
        monkeypatch.setattr(algebra, "_MAX_CROSS_EXPANSION_BYTES", 1)
    elif case == "cap_raw":
        # Enough for intermediates if mistakenly budgeted at public block
        # width, but not for the actual wider raw natural-basis blocks.
        monkeypatch.setattr(
            algebra, "_MAX_CROSS_EXPANSION_BYTES", 8 * 6 * gm.n_levels * gm.block_size * q
        )
    elif case == "float32":
        dense = DenseGroupMatrix(dense.M.astype(np.float32))
    elif case == "weight_float32":
        weights = weights.astype(np.float32)
    elif case == "weight_subclass":

        class ScaledWeights(np.ndarray):
            def __mul__(self, other):
                return 2.0 * (np.asarray(self) * other)

        weights = weights.view(ScaledWeights)
    elif case == "dense_subclass":

        class ScaledDense(DenseGroupMatrix):
            def matvec(self, v):
                return 2.0 * super().matvec(v)

        dense = ScaledDense(dense.M)
    elif case == "fs_subclass":

        class ScaledFactor(FactorSmoothGroupMatrix):
            def rmatvec(self, w):
                return 2.0 * super().rmatvec(w)

        gm = ScaledFactor(
            gm.B_unique if discrete else gm.B,
            gm.codes,
            gm.n_levels,
            natural_map=gm.natural_map,
            levels=gm.levels,
            repeated_penalty_components=gm.repeated_penalty_components,
            factor_basis=gm.factor_basis,
            bin_idx=gm.bin_idx,
        )

    def forbidden(*args, **kwargs):
        raise AssertionError("ineligible cross must preserve the established fallback")

    monkeypatch.setattr(FactorSmoothGroupMatrix, "factor_smooth_dense_cross_gram", forbidden)
    profile = {}
    left, right = (dense, gm) if reverse else (gm, dense)
    actual = algebra._cross_gram(left, right, weights, profile=profile)
    expected = reference.T @ (weights[:, None] * dense.M)
    if case in ("dense_subclass", "fs_subclass"):
        expected *= 2.0
    scale = np.linalg.norm(np.abs(reference).T @ np.abs(weights[:, None] * dense.M), ord=np.inf)
    tolerance = 16 * (len(weights) + gm.raw_width) * np.finfo(float).eps * scale
    np.testing.assert_allclose(actual, expected.T if reverse else expected, rtol=0, atol=tolerance)
    assert "block_cross_fallback_s" in profile
    assert "block_cross_factor_smooth_dense_calls" not in profile


@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("operand", ["basis", "natural_map", "dense", "weights"])
def test_extreme_operands_retain_finite_legacy_arithmetic(monkeypatch, discrete, operand):
    gm, dense, weights, _, _ = _fixture(discrete, "fs")
    basis = gm.B_unique if discrete else gm._data
    # For the basis case, reassociating W*basis overflows, whereas the
    # established basis*(W*dense) produces finite cross-products.
    if operand == "basis":
        basis *= 2.0**700
        weights *= 2.0**500
        dense.M *= 2.0**-600
    elif operand == "natural_map":
        gm.natural_map *= 2.0**700
        basis *= 2.0**-700
    elif operand == "dense":
        dense.M *= 2.0**700
        weights *= 2.0**-700
    else:
        weights *= 2.0**700
        dense.M *= 2.0**-700
    expected = algebra._cross_gram_by_columns(gm, dense, weights)
    assert np.isfinite(expected).all()

    def forbidden(*args, **kwargs):
        raise AssertionError("range-sensitive products must preserve legacy arithmetic")

    monkeypatch.setattr(FactorSmoothGroupMatrix, "factor_smooth_dense_cross_gram", forbidden)
    actual = algebra._cross_gram(gm, dense, weights)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("factor_basis", ["fs", "sz"])
@pytest.mark.parametrize("reverse", [False, True])
def test_csr_array_subclass_is_declined_before_creating_view(factor_basis, reverse):
    gm, dense, weights, _, _ = _fixture(False, factor_basis)
    left, right = (dense, gm) if reverse else (gm, dense)
    expected = algebra._cross_gram_by_columns(left, right, weights)

    class CustomData(np.ndarray):
        def __getitem__(self, key):
            raise AssertionError("eligibility must not invoke custom CSR array indexing")

    gm._data = gm._data.view(CustomData)
    profile = {}
    actual = algebra._cross_gram(left, right, weights, profile=profile)
    np.testing.assert_array_equal(actual, expected)
    assert "block_cross_fallback_s" in profile
    assert "block_cross_factor_smooth_dense_calls" not in profile
