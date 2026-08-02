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


def _hash_grouping(basis, chunk_rows):
    from superglm._group_matrix import _group_matrix_support as mod

    hashes = mod._row_hashes(basis, chunk_rows)
    _, _, row_index = np.unique(hashes, return_index=True, return_inverse=True)
    return np.asarray(row_index, dtype=np.intp).ravel()


def test_hashed_grouping_matches_byte_keyed_grouping():
    from superglm._group_matrix._group_matrix_support import _row_index_chunked

    basis, _, _ = _repeated_basis(n=5000, n_support=40, seed=9)
    _assert_same_partition(
        _hash_grouping(basis, chunk_rows=16),
        _row_index_chunked(basis, chunk_rows=16),
    )

    special = np.array([[np.nan, 1.0], [np.nan, 1.0], [0.0, 2.0], [-0.0, 2.0]])
    rows = sp.csr_matrix(special[np.tile(np.arange(4), 300)])
    _assert_same_partition(
        _hash_grouping(rows, chunk_rows=7),
        _row_index_chunked(rows, chunk_rows=7),
    )


def test_hash_collision_falls_back_to_byte_keyed_grouping(monkeypatch):
    """With a degenerate hash every row collides; the PRODUCTION path must
    detect it, regroup byte-keyed, re-run the gates on the true support, and
    still return an exact compression."""
    from superglm._group_matrix import _group_matrix_support as mod

    monkeypatch.setattr(
        mod,
        "_row_hash_multipliers",
        lambda p_b: np.zeros(max(p_b, 1), dtype=np.uint64),
    )
    basis, base, row_index = _repeated_basis()  # known accept-shape (60 groups)

    result = detect_row_support(basis)

    assert result is not None  # gates re-ran on the true 60-group support
    b_unique, derived = result
    _assert_same_partition(derived, row_index.astype(np.intp))
    np.testing.assert_array_equal(b_unique[derived], basis.toarray())


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


def test_hash_scan_chunks_honor_the_byte_ceiling(monkeypatch):
    """Review finding: fixed 65,536-row chunks densify gigabytes on wide
    tensor bases before any gate runs. Chunks must be sized in bytes."""
    rng = np.random.default_rng(21)
    n, p_b = 5000, 256
    basis = sp.csr_matrix(rng.normal(size=(n, p_b)))  # distinct rows a.s.
    calls = []
    _count_row_densifies(monkeypatch, calls)

    budget = 1 << 20  # 1 MiB => 512 rows per chunk at p_b=256
    assert detect_row_support(basis, max_support_bytes=budget) is None
    assert calls
    assert max(calls) <= budget // (p_b * 8)


# ── spline_cat: one shared support, one gram per level ─────────────
#
# A varying-coefficient term stores a CSR basis shared by its levels plus a row
# subset per level, and each level runs its own weighted gram.  Compression
# replaces both with one dense block of distinct rows.  The levels partition the
# rows between them but each reads the whole shared support, so the cost model
# is told how many grams run over it.


def _spline_cat_frame(n=6000, distinct=60, levels=3, seed=11):
    """A repeated rating factor crossed with a small categorical."""
    import pandas as pd

    gen = np.random.default_rng(seed)
    pool = np.linspace(18.0, 90.0, distinct)
    x = pool[gen.integers(0, distinct, n)]
    level = gen.integers(0, levels, n).astype(str)
    frame = pd.DataFrame({"x": x, "f": pd.Categorical(level)})
    response = gen.poisson(0.4, n).astype(np.float64)
    return frame, response


def _fit_spline_cat(frame, response, kind="cr", k=10, sample_weight=None):
    from superglm import SuperGLM
    from superglm.features.categorical import Categorical
    from superglm.features.spline import Spline

    model = SuperGLM(
        family="poisson",
        selection_penalty=None,
        discrete=False,
        features={"x": Spline(kind=kind, k=k), "f": Categorical()},
        interactions=[("x", "f")],
    )
    model.fit_reml(frame, response, sample_weight=sample_weight)
    return model


def _spline_cat_blocks(model):
    return [gm for gm in model._dm.group_matrices if "SplineCategorical" in type(gm).__name__]


def test_verified_plan_declines_without_touching_the_basis(monkeypatch):
    """The decline is the case the spline_cat caller pays on every continuous
    covariate, so it must not densify the basis to reach it."""
    from superglm._group_matrix._group_matrix_support import plan_verified_row_support

    rng = np.random.default_rng(30)
    n = 70_000
    basis = sp.csr_matrix(rng.normal(size=(n, 3)))  # distinct rows a.s.
    calls = []
    _count_row_densifies(monkeypatch, calls)

    assert plan_verified_row_support(basis, np.arange(n)) is None
    assert not calls, f"declined block densified {calls} rows"


def test_verified_plan_matches_detection_when_it_accepts():
    from superglm._group_matrix._group_matrix_support import plan_verified_row_support

    basis, _, row_index = _repeated_basis()

    verified = plan_verified_row_support(basis, row_index)
    detected = detect_row_support(basis)

    assert verified is not None and detected is not None
    np.testing.assert_array_equal(verified[0][verified[1]], basis.toarray())
    np.testing.assert_allclose(verified[0][verified[1]], detected[0][detected[1]])


def test_verified_plan_rejects_a_grouping_the_basis_contradicts():
    """The caller's grouping is a claim about the basis, not a licence."""
    from superglm._group_matrix._group_matrix_support import (
        plan_row_support,
        plan_verified_row_support,
    )

    basis, _, row_index = _repeated_basis()
    # Merge two genuinely different groups: 0 and 1 become one.
    wrong = np.where(row_index == 1, 0, row_index).astype(np.intp)
    wrong = np.unique(wrong, return_inverse=True)[1].ravel().astype(np.intp)

    trusted = plan_row_support(basis, wrong)
    assert trusted is not None
    assert not np.array_equal(trusted[0][trusted[1]], basis.toarray()), (
        "the wrong grouping was supposed to be detectably wrong"
    )

    verified = plan_verified_row_support(basis, wrong)
    assert verified is not None
    np.testing.assert_array_equal(verified[0][verified[1]], basis.toarray())


def test_gram_repeats_scales_only_the_dense_support_term():
    """Levels share the support but partition the rows: the bincount term is
    counted once, the dense gram once per level."""
    from superglm._group_matrix._group_matrix_support import (
        _BLAS_ADVANTAGE,
        _estimated_speedup,
    )

    n, n_support, p_b, nnz = 100_000, 500, 20, 2_000_000
    current = n * (nnz / n) * ((nnz / n) + 1.0) / 2.0

    for repeats in (1, 3, 7):
        expected = _BLAS_ADVANTAGE * current / (n + repeats * n_support * p_b**2)
        assert _estimated_speedup(n, n_support, p_b, nnz, repeats) == expected


def test_level_count_can_decline_a_block_a_single_gram_would_accept():
    """The multiplier has to be able to change the decision, or it is decoration."""
    from superglm._group_matrix._group_matrix_support import (
        DEFAULT_MIN_SPEEDUP,
        _estimated_speedup,
    )

    n, n_support, p_b, nnz = 100_000, 50_000, 20, 2_000_000

    assert _estimated_speedup(n, n_support, p_b, nnz, 1) >= DEFAULT_MIN_SPEEDUP
    assert _estimated_speedup(n, n_support, p_b, nnz, 6) < DEFAULT_MIN_SPEEDUP


def test_spline_cat_exact_path_compresses_a_repeated_covariate():
    """The whole point: a rating factor recorded in whole years must not store
    one basis row per observation on the exact path."""
    from superglm._group_matrix._group_matrix_discretized import (
        SupportCompressedSplineCategoricalGroupMatrix,
    )

    frame, response = _spline_cat_frame()
    blocks = _spline_cat_blocks(_fit_spline_cat(frame, response))

    assert blocks, "expected the model to build spline_cat level blocks"
    for block in blocks:
        assert type(block) is SupportCompressedSplineCategoricalGroupMatrix
        assert block.is_lossless_support is True
        assert block.n_bins < frame.shape[0]


def test_spline_cat_compression_leaves_every_fitted_quantity_unchanged(monkeypatch):
    """The release gate: compression must move no result beyond rounding.

    NOT bit-identity, and the distinction is the point of the weighting here.
    The level weights are the one thing the two paths do not compute the same
    way -- the compressed side aggregates them onto the shared support rows
    before ``compute_projected_R_inv`` sees them, the CSR side passes them per
    observation -- so a sum is reordered and the coefficients move at rounding
    scale.  Measured at 3.5e-9 absolute on this fixture; pinned an order of
    magnitude looser so the test does not fail on a different BLAS.
    """
    from superglm._group_matrix._group_matrix_discretized import (
        SupportCompressedSplineCategoricalGroupMatrix,
    )
    from superglm.features import interaction as interaction_module

    frame, response = _spline_cat_frame()
    weights = np.random.default_rng(31).uniform(0.2, 1.0, frame.shape[0])

    compressed = _fit_spline_cat(frame, response, sample_weight=weights)
    assert all(
        type(block) is SupportCompressedSplineCategoricalGroupMatrix
        for block in _spline_cat_blocks(compressed)
    ), "the comparison is vacuous unless the first fit actually compressed"

    monkeypatch.setattr(interaction_module, "_plan_spline_cat_support", lambda *a, **k: None)
    plain = _fit_spline_cat(frame, response, sample_weight=weights)
    assert not any(hasattr(block, "B_unique") for block in _spline_cat_blocks(plain)), (
        "the reference fit was supposed to stay on the CSR representation"
    )

    np.testing.assert_allclose(compressed._result.beta, plain._result.beta, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(compressed._result.deviance, plain._result.deviance, rtol=1e-9)
    np.testing.assert_allclose(
        compressed.predict(frame), plain.predict(frame), rtol=1e-8, atol=1e-8
    )
    np.testing.assert_allclose(
        compressed.metrics(frame, response, sample_weight=weights).effective_df,
        plain.metrics(frame, response, sample_weight=weights).effective_df,
        rtol=1e-6,
    )


def test_spline_cat_declines_when_the_covariate_never_repeats():
    """Guard, not a fix: a continuous covariate has nothing to deduplicate, and
    routing it through the dense support would slow the column-at-a-time
    cross-gram it still has to take."""
    import pandas as pd

    from superglm._group_matrix._group_matrix_core import SplineCategoricalGroupMatrix

    gen = np.random.default_rng(12)
    n = 6000
    frame = pd.DataFrame(
        {"x": gen.gamma(2.0, 1.5, n), "f": pd.Categorical(gen.integers(0, 3, n).astype(str))}
    )
    response = gen.poisson(0.4, n).astype(np.float64)

    blocks = _spline_cat_blocks(_fit_spline_cat(frame, response))

    assert blocks
    for block in blocks:
        assert type(block) is SplineCategoricalGroupMatrix


def test_spline_cat_compressed_block_algebra_matches_the_csr_block():
    from superglm._group_matrix._group_matrix_core import SplineCategoricalGroupMatrix
    from superglm._group_matrix._group_matrix_discretized import (
        SupportCompressedSplineCategoricalGroupMatrix,
    )

    gen = np.random.default_rng(13)
    n, n_support, p_b, p_g = 4000, 30, 6, 4
    base = gen.normal(size=(n_support, p_b))
    row_index = gen.integers(0, n_support, n).astype(np.intp)
    basis = sp.csr_matrix(base[row_index])
    r_inv = gen.normal(size=(p_b, p_g))
    rows = np.flatnonzero(gen.integers(0, 3, n) == 1)
    weights = gen.normal(1.0, 0.3, n)  # signed: REML's W-correction passes these

    reference = SplineCategoricalGroupMatrix(basis, r_inv, rows)
    compressed = SupportCompressedSplineCategoricalGroupMatrix(base, r_inv, row_index, rows)

    assert compressed.shape == reference.shape
    np.testing.assert_allclose(compressed.toarray(), reference.toarray(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        compressed.gram(weights), reference.gram(weights), rtol=1e-11, atol=1e-11
    )
    vector = gen.normal(size=p_g)
    np.testing.assert_allclose(
        compressed.matvec(vector), reference.matvec(vector), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        compressed.rmatvec(weights), reference.rmatvec(weights), rtol=1e-11, atol=1e-11
    )
    for actual, expected in zip(
        compressed.gram_rmatvec(weights, weights * 0.5),
        reference.gram_rmatvec(weights, weights * 0.5),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)


def test_spline_cat_row_subset_preserves_lossless_type():
    """A CV split must not silently convert a lossless level block into a binned one."""
    from superglm._group_matrix._group_matrix_discretized import (
        SupportCompressedSplineCategoricalGroupMatrix,
    )

    gen = np.random.default_rng(14)
    compressed = SupportCompressedSplineCategoricalGroupMatrix(
        gen.normal(size=(12, 4)),
        gen.normal(size=(4, 3)),
        gen.integers(0, 12, 400).astype(np.intp),
        np.arange(0, 400, 2, dtype=np.intp),
    )

    subset = compressed.row_subset(np.arange(0, 400, 3))

    assert type(subset) is SupportCompressedSplineCategoricalGroupMatrix
    assert subset.is_lossless_support is True
    np.testing.assert_allclose(subset.toarray(), compressed.toarray()[np.arange(0, 400, 3)])


def test_spline_cat_rebuild_with_lambdas_preserves_lossless_type():
    """Every REML lambda update rebuilds the design; the marker must survive."""
    from superglm._group_matrix._group_matrix_discretized import (
        SupportCompressedSplineCategoricalGroupMatrix,
    )
    from superglm.dm_builder import rebuild_design_matrix_with_lambdas

    frame, response = _spline_cat_frame()
    model = _fit_spline_cat(frame, response)
    weights = np.ones(frame.shape[0])
    lambdas = {group.name: 2.0 for group in model._groups}

    rebuilt = rebuild_design_matrix_with_lambdas(model._dm, model._groups, lambdas, weights, 1.0)

    original = _spline_cat_blocks(model)
    assert original, "expected spline_cat blocks to rebuild"
    rebuilt_blocks = [
        gm for gm in rebuilt.group_matrices if "SplineCategorical" in type(gm).__name__
    ]
    assert len(rebuilt_blocks) == len(original)
    for block in rebuilt_blocks:
        assert type(block) is SupportCompressedSplineCategoricalGroupMatrix


def test_design_summary_does_not_label_lossless_spline_cat_as_binned():
    from superglm._group_matrix._group_matrix_discretized import (
        SupportCompressedSplineCategoricalGroupMatrix,
    )
    from superglm.model.design_summary import _representation_metadata

    gen = np.random.default_rng(15)
    compressed = SupportCompressedSplineCategoricalGroupMatrix(
        gen.normal(size=(10, 4)),
        gen.normal(size=(4, 3)),
        gen.integers(0, 10, 100).astype(np.intp),
        np.arange(50, dtype=np.intp),
    )

    metadata = _representation_metadata(compressed)

    assert metadata.representation != "discretized-spline-categorical"
    assert metadata.specialised_discrete_route != "binned-spline-categorical", (
        "a discrete=False fit must not be reported as taking the binned fREML route"
    )
    assert metadata.compressed is True


def test_spline_cat_compressed_class_sits_where_its_siblings_do():
    """Reachable from ``superglm.group_matrix``, and deliberately NOT from the root.

    Reviewer question on #193: the package root neither imports this class nor
    lists it, so ``from superglm import ...`` fails.  That is the convention,
    not an omission -- no group-matrix class is a root export, and this one is
    not special.  It is rewritten onto ``superglm.group_matrix`` for the same
    reason ``SupportCompressedSSPGroupMatrix`` is: a pickled model must not
    carry a private module path.  Reachability for pickle and a root export are
    different things, and only the first is claimed.

    Pinned in both directions so that adding this one to the root without
    adding its twelve siblings fails here.
    """
    import pickle

    import superglm
    from superglm import group_matrix as public
    from superglm._group_matrix._group_matrix_discretized import (
        SupportCompressedSplineCategoricalGroupMatrix,
        SupportCompressedSSPGroupMatrix,
    )

    assert hasattr(public, "SupportCompressedSplineCategoricalGroupMatrix")
    assert SupportCompressedSplineCategoricalGroupMatrix.__module__ == "superglm.group_matrix"

    # Same treatment as the SSP twin, in both directions.
    assert SupportCompressedSSPGroupMatrix.__module__ == "superglm.group_matrix"
    for name in (
        "SupportCompressedSplineCategoricalGroupMatrix",
        "SupportCompressedSSPGroupMatrix",
    ):
        assert not hasattr(superglm, name), (
            f"{name} became a package-root export; no group-matrix class is one, so "
            "either this class is special (say why) or the whole family moved"
        )
        assert name not in getattr(superglm, "__all__", ())

    gen = np.random.default_rng(16)
    original = SupportCompressedSplineCategoricalGroupMatrix(
        gen.normal(size=(6, 3)),
        gen.normal(size=(3, 2)),
        gen.integers(0, 6, 50).astype(np.intp),
        np.arange(0, 50, 2, dtype=np.intp),
    )
    restored = pickle.loads(pickle.dumps(original))
    assert type(restored) is SupportCompressedSplineCategoricalGroupMatrix
    np.testing.assert_allclose(restored.toarray(), original.toarray())


def _build_spline_cat_groups(x_spline, x_cat, kind="cr", k=10):
    """Run ``SplineCategorical.build`` the way the design-matrix builder does."""
    from superglm.features.categorical import Categorical
    from superglm.features.interaction import SplineCategorical
    from superglm.features.spline import Spline

    spline_spec = Spline(kind=kind, k=k)
    spline_spec.build_knots_and_penalty(np.asarray(x_spline, dtype=np.float64))
    cat_spec = Categorical()
    cat_spec.build(np.asarray(x_cat))

    term = SplineCategorical("x", "f")
    return term.build(
        np.asarray(x_spline, dtype=np.float64),
        np.asarray(x_cat),
        {"x": spline_spec, "f": cat_spec},
    )


def test_dominant_reference_level_rows_do_not_buy_compression():
    """Reviewer finding on #193: the base level is absorbed into the main
    effect, so its rows appear in NO emitted block.  Charging the CSR side for
    work that will never run lets the gate accept on the strength of a majority
    level that cannot benefit.

    The shipped benchmark samples levels uniformly and cannot see this, which
    is the whole point of pinning it here.
    """
    gen = np.random.default_rng(40)
    n, n_active = 200_000, 2_000
    x = np.empty(n, dtype=np.float64)
    x_cat = np.empty(n, dtype="<U1")

    # The base level repeats heavily, so the basis looks compressible overall.
    base_pool = np.linspace(18.0, 90.0, 72)
    x[n_active:] = base_pool[gen.integers(0, base_pool.size, n - n_active)]
    x_cat[n_active:] = "0"
    # The one non-base level is continuous: nothing repeats, so there is
    # nothing for deduplication to buy on the rows that actually reach a block.
    x[:n_active] = gen.uniform(18.0, 90.0, n_active)
    x_cat[:n_active] = "1"

    groups = _build_spline_cat_groups(x, x_cat)

    assert groups, "expected one block for the single non-base level"
    for info in groups:
        assert info.spline_cat_basis_unique is None, (
            f"compression was accepted on {n:,} rows of CSR work when only "
            f"{n_active:,} rows are in any emitted block, and those repeat nothing"
        )
        assert info.spline_cat_basis is not None
        assert info.spline_cat_support_lossless is False


def test_support_holds_only_rows_a_non_base_level_can_reach():
    """The support is shared across levels, so a row only the base level ever
    visits still widens every level's dense gram.  It must not be retained."""
    gen = np.random.default_rng(41)
    n = 60_000
    active_values = np.arange(20, dtype=np.float64)
    base_only_values = np.arange(100, 140, dtype=np.float64)

    x = np.empty(n, dtype=np.float64)
    x_cat = np.empty(n, dtype="<U1")
    half = n // 2
    # Base level sits on 40 x-values nothing else reaches.
    x[:half] = base_only_values[gen.integers(0, base_only_values.size, half)]
    x_cat[:half] = "0"
    # Two non-base levels share 20 x-values between them.
    x[half:] = active_values[gen.integers(0, active_values.size, n - half)]
    x_cat[half:] = np.where(gen.integers(0, 2, n - half) == 0, "1", "2")

    groups = _build_spline_cat_groups(x, x_cat)

    assert groups
    for info in groups:
        assert info.spline_cat_basis_unique is not None, "expected this shape to compress"
        assert info.spline_cat_support_lossless is True
        assert info.spline_cat_basis_unique.shape[0] == active_values.size, (
            "support retained rows only a base-level observation reaches: "
            f"{info.spline_cat_basis_unique.shape[0]} rows for "
            f"{active_values.size} reachable x values"
        )


def test_balanced_levels_still_compress():
    """Control beside the two above: the fix must decline the skewed case
    without declining the case compression exists for."""
    gen = np.random.default_rng(42)
    n = 60_000
    pool = np.linspace(18.0, 90.0, 72)
    x = pool[gen.integers(0, pool.size, n)]
    x_cat = gen.integers(0, 3, n).astype(str)

    groups = _build_spline_cat_groups(x, x_cat)

    assert groups
    for info in groups:
        assert info.spline_cat_basis_unique is not None
        assert info.spline_cat_support_lossless is True


def test_expanded_cross_gram_is_chunked_by_bytes(monkeypatch):
    """Reviewer finding on #193: the cell cap routes here, and this path then
    expanded every shared row at once -- so the cap moved the memory instead of
    bounding it.  A lossless support does not bound the row count."""
    from superglm._group_matrix import _group_matrix_algebra as algebra

    gen = np.random.default_rng(43)
    n_rows, p_i, p_j = 50_000, 20, 20
    b_i = gen.normal(size=(400, p_i))
    b_j = gen.normal(size=(350, p_j))
    idx_i = gen.integers(0, 400, n_rows).astype(np.intp)
    idx_j = gen.integers(0, 350, n_rows).astype(np.intp)
    weights = np.abs(gen.normal(1.0, 0.2, n_rows))

    expanded_rows = []
    original = algebra._expand_support_rows

    def spy(B_unique, bin_idx):
        expanded_rows.append(int(np.size(bin_idx)))
        return original(B_unique, bin_idx)

    monkeypatch.setattr(algebra, "_expand_support_rows", spy)

    budget = 1 << 20  # 1 MiB
    actual = algebra._support_support_raw_cross(b_i, idx_i, b_j, idx_j, weights, max_bytes=budget)

    expected = (b_i[idx_i]).T @ (b_j[idx_j] * weights[:, None])
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)

    assert expanded_rows, "expected the fallback to expand support rows"
    allowed = algebra._cross_expansion_chunk_rows(p_i, p_j, budget)
    assert max(expanded_rows) <= allowed, (
        f"expanded {max(expanded_rows)} rows at once against a {allowed}-row budget"
    )
    assert max(expanded_rows) * (p_i + p_j) * 8 <= budget
    assert sum(expanded_rows) == 2 * n_rows  # both sides, every row, exactly once


def test_expanded_cross_gram_chunking_matches_the_unchunked_contraction():
    """Chunking is a partition of a sum over rows; only the order changes."""
    from superglm._group_matrix import _group_matrix_algebra as algebra

    gen = np.random.default_rng(44)
    n_rows = 9_000
    b_i = gen.normal(size=(60, 5))
    b_j = gen.normal(size=(40, 7))
    idx_i = gen.integers(0, 60, n_rows).astype(np.intp)
    idx_j = gen.integers(0, 40, n_rows).astype(np.intp)
    weights = gen.normal(0.0, 1.0, n_rows)  # signed, as REML passes

    one_shot = algebra._support_support_raw_cross(
        b_i, idx_i, b_j, idx_j, weights, max_bytes=1 << 30
    )
    chunked = algebra._support_support_raw_cross(b_i, idx_i, b_j, idx_j, weights, max_bytes=512)

    np.testing.assert_allclose(chunked, one_shot, rtol=1e-11, atol=1e-11)


def _forbid_2d_histogram(monkeypatch, pair_name):
    """Make the joint histogram fatal so a cap test cannot pass by taking it."""
    from superglm._group_matrix import _group_matrix_algebra as algebra

    def refuse(*args, **kwargs):
        raise AssertionError(f"{pair_name} built an unbounded joint histogram")

    monkeypatch.setattr(algebra, "_disc_disc_2d_hist", refuse)


def _spline_cat_support_block(n, n_support, p_b, p_g, rows, seed):
    from superglm._group_matrix._group_matrix_discretized import (
        SupportCompressedSplineCategoricalGroupMatrix,
    )

    gen = np.random.default_rng(seed)
    base = gen.normal(size=(n_support, p_b))
    row_index = gen.integers(0, n_support, n).astype(np.intp)
    block = SupportCompressedSplineCategoricalGroupMatrix(
        base, gen.normal(size=(p_b, p_g)), row_index, rows
    )
    block.spline_cat_feature = "f"
    block.spline_cat_level = "1"
    return block


def test_wide_supports_cross_gram_without_a_joint_histogram(monkeypatch):
    """Two spline_cat terms over the same factor multiply their support sizes
    into one histogram.  A lossless support is bounded by the row count, not by
    a bin count, so that product has to be capped."""
    from superglm._group_matrix import _group_matrix_algebra as algebra

    n = 3000
    rows = np.arange(0, n, 2, dtype=np.intp)
    weights = np.abs(np.random.default_rng(17).normal(1.0, 0.2, n))
    left = _spline_cat_support_block(n, 400, 5, 3, rows, seed=18)
    right = _spline_cat_support_block(n, 350, 4, 2, rows, seed=19)

    expected = algebra._cross_gram_spline_categorical_spline_categorical(left, right, weights)

    monkeypatch.setattr(algebra, "_MAX_DISC_DISC_HIST_CELLS", 1_000)
    _forbid_2d_histogram(monkeypatch, "spline_cat x spline_cat")
    actual = algebra._cross_gram_spline_categorical_spline_categorical(left, right, weights)

    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)


def test_wide_supports_cross_gram_on_partial_row_overlap_without_a_histogram(monkeypatch):
    """Same cap, on the branch that intersects two different row sets."""
    from superglm._group_matrix import _group_matrix_algebra as algebra

    n = 3000
    weights = np.abs(np.random.default_rng(20).normal(1.0, 0.2, n))
    left = _spline_cat_support_block(n, 400, 5, 3, np.arange(0, n, 2, dtype=np.intp), seed=21)
    right = _spline_cat_support_block(n, 350, 4, 2, np.arange(0, n, 3, dtype=np.intp), seed=22)
    right.spline_cat_feature = "g"

    expected = algebra._cross_gram_spline_categorical_spline_categorical(left, right, weights)

    monkeypatch.setattr(algebra, "_MAX_DISC_DISC_HIST_CELLS", 1_000)
    _forbid_2d_histogram(monkeypatch, "spline_cat x spline_cat")
    actual = algebra._cross_gram_spline_categorical_spline_categorical(left, right, weights)

    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)


def test_tensor_by_wide_spline_cat_cross_gram_without_a_joint_histogram(monkeypatch):
    """A per-feature ``discrete=True`` spline can put a binned tensor beside a
    lossless spline_cat, and that pair histograms too."""
    from superglm._group_matrix import _group_matrix_algebra as algebra
    from superglm._group_matrix._group_matrix_discretized import (
        DiscretizedTensorGroupMatrix,
    )

    gen = np.random.default_rng(23)
    n, n_bins1, n_bins2 = 3000, 12, 9
    b1 = gen.normal(size=(n_bins1, 3))
    b2 = gen.normal(size=(n_bins2, 2))
    idx1 = gen.integers(0, n_bins1, n).astype(np.intp)
    idx2 = gen.integers(0, n_bins2, n).astype(np.intp)
    joint = np.einsum("ij,ik->ijk", b1[idx1], b2[idx2]).reshape(n, 6)
    tensor = DiscretizedTensorGroupMatrix(
        b1,
        b2,
        idx1,
        idx2,
        joint,
        gen.normal(size=(6, 4)),
        np.arange(n, dtype=np.intp),
        tensor_id=0,
    )
    rows = np.arange(0, n, 2, dtype=np.intp)
    spline_cat = _spline_cat_support_block(n, 400, 5, 3, rows, seed=24)
    weights = np.abs(gen.normal(1.0, 0.2, n))

    expected = algebra._cross_gram_tensor_spline_categorical(tensor, spline_cat, weights)

    monkeypatch.setattr(algebra, "_MAX_DISC_DISC_HIST_CELLS", 1_000)
    _forbid_2d_histogram(monkeypatch, "tensor x spline_cat")
    actual = algebra._cross_gram_tensor_spline_categorical(tensor, spline_cat, weights)

    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)
