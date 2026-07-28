"""Lossless row-support compression for factored SSP group matrices.

Compression here is deduplication of repeated design rows, never binning, so it
must leave every fitted quantity unchanged.  It is unrelated to ``discrete=True``.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from superglm._group_matrix._group_matrix_support import (
    detect_row_support,
    plan_row_support,
)


def _repeated_basis(n=20_000, n_support=60, p_b=9, nnz_row=4, seed=0):
    """A B-spline-shaped basis: locally supported rows drawn from a small support."""
    gen = np.random.default_rng(seed)
    base = np.zeros((n_support, p_b))
    for row in range(n_support):
        cols = gen.choice(p_b, size=nnz_row, replace=False)
        base[row, cols] = gen.normal(size=nnz_row)
    row_index = gen.integers(0, n_support, n)
    return sp.csr_matrix(base[row_index]), base, row_index


def test_plan_row_support_compresses_repeated_rows():
    basis, base, row_index = _repeated_basis()

    result = plan_row_support(basis, row_index)

    assert result is not None
    b_unique, returned_index = result
    assert b_unique.shape == base.shape
    np.testing.assert_allclose(b_unique[returned_index], basis.toarray())


def test_plan_row_support_declines_when_compression_would_be_slower():
    """40% distinct rows on a sparse spline basis costs more than it saves."""
    n = 20_000
    basis, _, row_index = _repeated_basis(n=n, n_support=n * 4 // 10, seed=1)

    assert plan_row_support(basis, row_index) is None


def test_plan_row_support_declines_when_support_buffer_too_large():
    basis, _, row_index = _repeated_basis(n=20_000, n_support=60)

    assert plan_row_support(basis, row_index, max_support_bytes=1) is None


def test_detect_row_support_matches_plan_row_support():
    basis, _, row_index = _repeated_basis()

    derived = detect_row_support(basis)
    planned = plan_row_support(basis, row_index)

    assert derived is not None and planned is not None
    np.testing.assert_allclose(derived[0][derived[1]], planned[0][planned[1]])


def test_detect_row_support_declines_when_rows_are_distinct():
    rows = np.arange(20.0).reshape(10, 2)

    assert detect_row_support(sp.csr_matrix(rows)) is None


def test_detect_row_support_declines_on_empty_basis():
    assert detect_row_support(sp.csr_matrix((0, 3))) is None


def test_nan_rows_decline_rather_than_compress_incorrectly():
    """The module claims exactness, so NaN must never be merged into a group."""
    base = np.array([[1.0, 0.0], [np.nan, 2.0]])
    rows = base[np.array([0, 1] * 500)]

    result = detect_row_support(sp.csr_matrix(rows))

    if result is not None:
        b_unique, row_index = result
        reconstructed = b_unique[row_index]
        finite = np.isfinite(rows)
        np.testing.assert_allclose(reconstructed[finite], rows[finite])
        assert np.isnan(reconstructed[~finite]).all()


def test_negative_zero_does_not_corrupt_reconstruction():
    base = np.array([[0.0, 1.0], [-0.0, 1.0], [2.0, 3.0]])
    rows = base[np.array([0, 1, 2] * 400)]

    result = detect_row_support(sp.csr_matrix(rows))

    if result is not None:
        b_unique, row_index = result
        # -0.0 and 0.0 may merge; that is harmless under + and *, but the
        # reconstruction must still be numerically equal to the input.
        np.testing.assert_allclose(b_unique[row_index], rows)


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


def test_row_subset_preserves_lossless_type():
    """A CV split must not silently convert a lossless group into a binned one."""
    from superglm._group_matrix._group_matrix_discretized import (
        SupportCompressedSSPGroupMatrix,
    )

    rng = np.random.default_rng(2)
    b_unique = rng.normal(size=(12, 4))
    row_index = rng.integers(0, 12, 400)
    r_inv = rng.normal(size=(4, 3))
    compressed = SupportCompressedSSPGroupMatrix(b_unique, r_inv, row_index)

    subset = compressed.row_subset(np.arange(0, 400, 3))

    assert type(subset) is SupportCompressedSSPGroupMatrix
    assert subset.is_lossless_support is True


def test_rebuild_with_lambdas_preserves_lossless_type():
    """Every REML lambda update rebuilds the design; the marker must survive."""
    import pandas as pd

    from superglm import SuperGLM
    from superglm._group_matrix._group_matrix_discretized import (
        SupportCompressedSSPGroupMatrix,
    )
    from superglm.dm_builder import rebuild_design_matrix_with_lambdas
    from superglm.features.spline import Spline

    rng = np.random.default_rng(4)
    n = 3000
    frame = pd.DataFrame({"age": rng.integers(18, 90, n).astype(float)})
    weights = np.full(n, 1.0)
    response = rng.poisson(0.2, n).astype(float)

    model = SuperGLM(
        family="poisson",
        selection_penalty=None,
        discrete=False,
        features={"age": Spline(kind="ps", k=8)},
    )
    model._build_design_matrix(frame, response, weights, None)

    original = model._dm.group_matrices[0]
    if not isinstance(original, SupportCompressedSSPGroupMatrix):
        import pytest

        pytest.skip("design build does not yet produce compressed groups (Task 4)")

    rebuilt = rebuild_design_matrix_with_lambdas(model._dm, model._groups, {"age": 2.0}, weights)
    assert type(rebuilt.group_matrices[0]) is SupportCompressedSSPGroupMatrix


def test_design_summary_does_not_label_lossless_group_as_binned():
    """design_summary() is public and is the only surface describing storage."""
    from superglm._group_matrix._group_matrix_discretized import (
        SupportCompressedSSPGroupMatrix,
    )
    from superglm.model.design_summary import _representation_metadata

    rng = np.random.default_rng(6)
    compressed = SupportCompressedSSPGroupMatrix(
        rng.normal(size=(10, 4)), rng.normal(size=(4, 3)), rng.integers(0, 10, 100)
    )

    metadata = _representation_metadata(compressed)

    assert metadata.representation != "discretized-ssp"
    assert metadata.specialised_discrete_route != "binned-ssp", (
        "a discrete=False fit must not be reported as taking the binned fREML route"
    )
    assert metadata.compressed is True


def test_support_compressed_class_is_publicly_reachable():
    """Pickled models must not carry a private module path."""
    import pickle

    from superglm import group_matrix as public
    from superglm._group_matrix._group_matrix_discretized import (
        SupportCompressedSSPGroupMatrix,
    )

    assert hasattr(public, "SupportCompressedSSPGroupMatrix")
    assert SupportCompressedSSPGroupMatrix.__module__ == "superglm.group_matrix"

    rng = np.random.default_rng(8)
    original = SupportCompressedSSPGroupMatrix(
        rng.normal(size=(6, 3)), rng.normal(size=(3, 2)), rng.integers(0, 6, 50)
    )
    restored = pickle.loads(pickle.dumps(original))
    assert type(restored) is SupportCompressedSSPGroupMatrix
    np.testing.assert_allclose(restored.toarray(), original.toarray())


# (p_b, nnz_row, support_ratio, measured_speedup) at n=200_000, median of 5 after
# warm-up.  The gate must agree with the sign of these measurements: compress
# where compression was actually faster, decline where it was actually slower.
_CALIBRATION = [
    (9, 4, 0.0005, 20.2),
    (9, 4, 0.05, 8.3),
    (9, 4, 0.40, 1.68),
    (9, 4, 0.95, 0.74),
    (20, 4, 0.005, 17.4),
    (20, 4, 0.40, 0.73),
    (81, 81, 0.05, 407.3),
    (81, 81, 0.40, 9.96),
    (81, 81, 0.95, 4.46),
]


def test_gate_agrees_with_measured_crossover():
    """The cost model is calibrated, not derived; pin it to the measurements."""
    from superglm._group_matrix._group_matrix_support import _estimated_speedup

    n = 200_000
    for p_b, nnz_row, ratio, measured in _CALIBRATION:
        n_support = int(n * ratio)
        predicted = _estimated_speedup(n, n_support, p_b, nnz=n * nnz_row)
        should_compress = predicted >= 1.5
        was_faster = measured >= 1.0
        assert should_compress == was_faster, (
            f"p_b={p_b} nnz_row={nnz_row} ratio={ratio}: gate says "
            f"{'compress' if should_compress else 'decline'} (predicted {predicted:.2f}) "
            f"but measured speedup was {measured}x"
        )


def test_gate_accepts_the_real_world_blocks():
    """The two shapes measured on freMTPL2 must both be accepted."""
    from superglm._group_matrix._group_matrix_support import _estimated_speedup

    # DrivAge k=10 spline: 82 distinct rows in 100k, 4 nonzeros per row.
    assert _estimated_speedup(100_000, 82, 9, nnz=400_000) >= 1.5
    # DrivAge:BonusMalus tensor: 2664 distinct rows, fully dense rows.
    assert _estimated_speedup(100_000, 2_664, 81, nnz=8_100_000) >= 1.5
