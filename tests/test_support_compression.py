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


def _assert_same_partition(labels_a, labels_b):
    """Two groupings are the same partition iff their joint has no new classes."""
    joint = labels_a.astype(np.int64) * (int(labels_b.max()) + 1) + labels_b
    n_joint = len(np.unique(joint))
    assert n_joint == len(np.unique(labels_a)) == len(np.unique(labels_b))


def test_hashed_grouping_matches_byte_keyed_grouping():
    from superglm._group_matrix._group_matrix_support import (
        _row_index_chunked,
        _row_index_hashed,
    )

    basis, _, _ = _repeated_basis(n=5000, n_support=40, seed=9)
    _assert_same_partition(
        _row_index_hashed(basis, chunk_rows=16),
        _row_index_chunked(basis, chunk_rows=16),
    )

    special = np.array([[np.nan, 1.0], [np.nan, 1.0], [0.0, 2.0], [-0.0, 2.0]])
    rows = sp.csr_matrix(special[np.tile(np.arange(4), 300)])
    _assert_same_partition(
        _row_index_hashed(rows, chunk_rows=7),
        _row_index_chunked(rows, chunk_rows=7),
    )


def test_hash_collision_falls_back_to_byte_keyed_grouping(monkeypatch):
    """With a degenerate hash every row collides; the result must stay exact."""
    from superglm._group_matrix import _group_matrix_support as mod

    monkeypatch.setattr(
        mod,
        "_row_hash_multipliers",
        lambda p_b: np.zeros(max(p_b, 1), dtype=np.uint64),
    )
    basis, base, row_index = _repeated_basis(n=3000, n_support=25, seed=12)

    grouped = mod._row_index_hashed(basis, chunk_rows=64)

    _assert_same_partition(grouped, row_index.astype(np.intp))
    dense = basis.toarray()
    first = np.full(int(grouped.max()) + 1, -1, dtype=np.intp)
    first[grouped[::-1]] = np.arange(len(grouped) - 1, -1, -1)
    np.testing.assert_array_equal(dense[first][grouped], dense)


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
    assert isinstance(original, SupportCompressedSSPGroupMatrix), (
        f"expected the exact path to compress this block, got {type(original).__name__}"
    )

    rebuilt = rebuild_design_matrix_with_lambdas(
        model._dm, model._groups, {"age": 2.0}, weights, 1.0
    )
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


def test_compression_does_not_change_fitted_results(monkeypatch):
    """The release gate: compression is storage-only and must move no result."""
    import pandas as pd

    from superglm import SuperGLM
    from superglm._group_matrix import _group_matrix_support
    from superglm.features.spline import Spline

    rng = np.random.default_rng(7)
    n = 4000
    frame = pd.DataFrame(
        {
            "age": rng.integers(18, 90, n).astype(float),
            "bm": rng.integers(50, 130, n).astype(float),
        }
    )
    weights = rng.uniform(0.2, 1.0, n)
    response = rng.poisson(0.2, n) / weights

    def fit():
        model = SuperGLM(
            family="poisson",
            selection_penalty=None,
            discrete=False,
            features={"age": Spline(kind="ps", k=10), "bm": Spline(kind="ps", k=10)},
        )
        model._add_interaction("age", "bm")
        return model.fit_reml(frame, response, sample_weight=weights)

    compressed = fit()
    monkeypatch.setattr(_group_matrix_support, "detect_row_support", lambda *a, **k: None)
    uncompressed = fit()

    np.testing.assert_allclose(
        compressed.result.beta, uncompressed.result.beta, rtol=1e-8, atol=1e-8
    )
    np.testing.assert_allclose(compressed.result.deviance, uncompressed.result.deviance, rtol=1e-9)
    np.testing.assert_allclose(
        compressed.metrics(frame, response, sample_weight=weights).effective_df,
        uncompressed.metrics(frame, response, sample_weight=weights).effective_df,
        rtol=1e-6,
    )


def _count_row_densifies(monkeypatch, calls):
    """Record the row count of every csr densify so tests can pin allocations."""
    orig_toarray = sp.csr_matrix.toarray
    orig_todense = sp.csr_matrix.todense

    def toarray(self, *args, **kwargs):
        out = orig_toarray(self, *args, **kwargs)
        calls.append(out.shape[0])
        return out

    def todense(self, *args, **kwargs):
        out = orig_todense(self, *args, **kwargs)
        calls.append(out.shape[0])
        return out

    monkeypatch.setattr(sp.csr_matrix, "toarray", toarray)
    monkeypatch.setattr(sp.csr_matrix, "todense", todense)


def test_declined_scan_never_densifies_the_support_block(monkeypatch):
    """Review finding: the no-repeats case densified the ENTIRE basis as
    'representatives' before the byte-budget gate could refuse it. Decline
    must now cost only the bounded hash chunks."""
    rng = np.random.default_rng(20)
    n = 70_000
    basis = sp.csr_matrix(rng.normal(size=(n, 3)))  # distinct rows a.s.
    calls = []
    _count_row_densifies(monkeypatch, calls)

    assert detect_row_support(basis) is None
    assert calls, "expected the chunked hash scan to densify bounded blocks"
    assert max(calls) <= 65_536  # pre-fix: one 70_000-row representative block


def test_accepted_scan_densifies_the_support_exactly_once(monkeypatch):
    """Review finding: the accept path densified the representative block
    twice (verification, then plan_row_support). It must be built once and
    threaded through."""
    basis, base, row_index = _repeated_basis()
    calls = []
    _count_row_densifies(monkeypatch, calls)

    result = detect_row_support(basis)

    assert result is not None
    b_unique, derived = result
    np.testing.assert_array_equal(b_unique[derived], basis.toarray())
    assert calls.count(60) == 1  # pre-fix: 2 identical 60-row densifies
