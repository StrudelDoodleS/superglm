"""Lossless row-support compression for factored SSP group matrices.

Compression here is deduplication of repeated design rows, never binning, so it
must leave every fitted quantity unchanged.  It is unrelated to ``discrete=True``.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from superglm._group_matrix._group_matrix_support import detect_row_support


def test_detect_row_support_compresses_repeated_rows():
    base = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]])
    rows = base[np.array([0, 1, 2, 0, 1, 0])]

    result = detect_row_support(sp.csr_matrix(rows))

    assert result is not None
    b_unique, row_index = result
    assert b_unique.shape == (3, 2)
    assert row_index.shape == (6,)
    np.testing.assert_allclose(b_unique[row_index], rows)


def test_detect_row_support_declines_when_rows_are_distinct():
    rows = np.arange(20.0).reshape(10, 2)

    assert detect_row_support(sp.csr_matrix(rows)) is None


def test_detect_row_support_declines_on_empty_basis():
    assert detect_row_support(sp.csr_matrix((0, 3))) is None


def test_detect_row_support_respects_max_ratio():
    base = np.array([[1.0, 0.0], [0.0, 2.0]])
    rows = base[np.array([0, 1, 0, 1])]

    assert detect_row_support(sp.csr_matrix(rows), max_ratio=0.9) is not None
    assert detect_row_support(sp.csr_matrix(rows), max_ratio=0.1) is None


def test_support_compressed_gram_matches_sparse_ssp():
    from superglm._group_matrix._group_matrix_core import SparseSSPGroupMatrix
    from superglm._group_matrix._group_matrix_discretized import (
        SupportCompressedSSPGroupMatrix,
    )

    rng = np.random.default_rng(0)
    base = rng.normal(size=(40, 6))
    idx = rng.integers(0, 40, 5000)
    basis = sp.csr_matrix(base[idx])
    r_inv = rng.normal(size=(6, 4))
    weights = np.abs(rng.normal(1.0, 0.2, 5000))

    reference = SparseSSPGroupMatrix(basis, r_inv)
    b_unique, row_index = detect_row_support(basis)
    compressed = SupportCompressedSSPGroupMatrix(b_unique, r_inv, row_index)

    assert compressed.is_lossless_support is True
    assert compressed.shape == reference.shape
    np.testing.assert_allclose(
        compressed.gram(weights), reference.gram(weights), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(compressed.toarray(), reference.toarray(), rtol=1e-12, atol=1e-12)
    vector = rng.normal(size=4)
    np.testing.assert_allclose(
        compressed.matvec(vector), reference.matvec(vector), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        compressed.rmatvec(weights), reference.rmatvec(weights), rtol=1e-12, atol=1e-12
    )


def test_support_compressed_handles_signed_weights():
    """REML's W-correction passes arbitrary-sign weights; sqrt paths would fail."""
    from superglm._group_matrix._group_matrix_core import SparseSSPGroupMatrix
    from superglm._group_matrix._group_matrix_discretized import (
        SupportCompressedSSPGroupMatrix,
    )

    rng = np.random.default_rng(5)
    base = rng.normal(size=(25, 5))
    idx = rng.integers(0, 25, 3000)
    basis = sp.csr_matrix(base[idx])
    r_inv = rng.normal(size=(5, 3))
    signed = rng.normal(0.0, 1.0, 3000)

    reference = SparseSSPGroupMatrix(basis, r_inv)
    b_unique, row_index = detect_row_support(basis)
    compressed = SupportCompressedSSPGroupMatrix(b_unique, r_inv, row_index)

    np.testing.assert_allclose(
        compressed.gram(signed), reference.gram(signed), rtol=1e-11, atol=1e-11
    )


def _make_pair(n, n_support, p_b, p_g, seed):
    """Build the same block as both a SparseSSP and a support-compressed group."""
    from superglm._group_matrix._group_matrix_core import SparseSSPGroupMatrix
    from superglm._group_matrix._group_matrix_discretized import (
        SupportCompressedSSPGroupMatrix,
    )

    gen = np.random.default_rng(seed)
    base = gen.normal(size=(n_support, p_b))
    basis = sp.csr_matrix(base[gen.integers(0, n_support, n)])
    r_inv = gen.normal(size=(p_b, p_g))
    b_unique, row_index = detect_row_support(basis)
    return (
        SparseSSPGroupMatrix(basis, r_inv),
        SupportCompressedSSPGroupMatrix(b_unique, r_inv, row_index),
    )


def test_cross_gram_uses_fast_path_for_compressed_groups():
    from superglm._group_matrix._group_matrix_algebra import _cross_gram

    n = 4000
    reference_i, compressed_i = _make_pair(n, 30, 5, 3, seed=10)
    reference_j, compressed_j = _make_pair(n, 25, 4, 2, seed=11)
    weights = np.abs(np.random.default_rng(1).normal(1.0, 0.2, n))

    baseline_profile: dict = {}
    expected = _cross_gram(reference_i, reference_j, weights, profile=baseline_profile)
    profile: dict = {}
    actual = _cross_gram(compressed_i, compressed_j, weights, profile=profile)

    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)
    # Assert the positive branch, not merely the absence of the negative one, so
    # the test cannot pass vacuously if profiling keys are ever renamed.
    assert "block_cross_fallback_s" in baseline_profile, (
        f"expected uncompressed groups to take the fallback; profile={baseline_profile}"
    )
    assert "block_cross_disc_disc_s" in profile, (
        f"compressed groups did not take the 2-D histogram path; profile={profile}"
    )
    assert "block_cross_fallback_s" not in profile, (
        f"compressed groups took the column-at-a-time fallback; profile={profile}"
    )
