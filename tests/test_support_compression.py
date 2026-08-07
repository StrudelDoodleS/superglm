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
        # Explicit loose tolerance: compression only reorders floating-point
        # accumulation, but under the tight publication bar the optimizer
        # walks the flat tensor direction long enough for that reordering to
        # separate the two lambda paths at ~1e-7 in coefficients. The gate's
        # subject is that compression is storage-only along a matched
        # optimizer path, which the loose bar keeps matched.
        return model.fit_reml(frame, response, sample_weight=weights, reml_tol=1e-6)

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

    expanded_rows = _spy_on_row_expansion(monkeypatch)

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


def _spy_on_row_expansion(monkeypatch):
    """Record the row count of every support expansion, so a chunk bound can be
    asserted rather than described.

    One definition, used by every chunking test, so that a site added later
    inherits every assertion the others make rather than a subset.
    """
    from superglm._group_matrix import _group_matrix_algebra as algebra

    seen: list[int] = []
    original = algebra._expand_support_rows

    def spy(B_unique, bin_idx):
        seen.append(int(np.size(bin_idx)))
        return original(B_unique, bin_idx)

    monkeypatch.setattr(algebra, "_expand_support_rows", spy)
    return seen


# Every cross-shaped aggregate goes through one of these, and each is
# responsible for keeping ``n_bins * n_cols`` inside _MAX_AGGREGATE_CELLS.
# A new call site anywhere else is the ninth instance of the bug that took
# five review rounds to stop finding, so it fails this file instead.
_BOUNDED_AGGREGATE_HELPERS = frozenset(
    {
        "_chunked_support_bincount_2d",
        "_support_csr_raw_cross",
    }
)

# Call sites whose output cannot run away, each with the reason. Every entry
# here is a claim that has to survive the pairing test below -- an allow-list
# with a wrong justification is worse than no invariant test, because it turns
# an unbounded site into a documented "we checked this". One of the original
# four said the two dimensions came from the same block; they did not, on the
# `DiscretizedSSP x SparseSSP` pairing, and that is why every entry now names
# the caller that supplies `n_bins` rather than reasoning about the callee.
_AGGREGATE_CALLS_BOUNDED_BY_CONSTRUCTION = {
    ("_aggregate_group_matrix_columns", "_weighted_bincount_2d"): (
        "no cross shape: this aggregates ONE generated column at a time, so the "
        "output is (n_bins, 1) per pass regardless of either block's width"
    ),
    ("_cross_gram_categorical_spline_categorical", "_csr_weighted_bincount"): (
        "n_bins is n_levels+1 of the categorical parent, which is a factor "
        "cardinality and not a support size, so it cannot scale with n"
    ),
    ("_cross_gram_tensor_spline_categorical", "_csr_weighted_bincount"): (
        "the CSR spline_cat branch: n_bins is the tensor's marginal bin count "
        "and n_cols its own basis width, both bin/width quantities that a "
        "lossless support cannot inflate"
    ),
    ("_cross_gram_tensor_spline_categorical", "_weighted_bincount_2d"): (
        "(n_bins1, K_cat), same reasoning; and the K2-deep accumulator built "
        "from it is now explicitly budgeted, inverting the loop nesting rather "
        "than growing past _MAX_AGGREGATE_CELLS"
    ),
    ("_agg_by_bin", "_csr_weighted_bincount"): (
        "IS cross-shaped and is NOT exempt on its own account: every caller "
        "checks _agg_by_bin_fits first and routes to the column-at-a-time "
        "_cross_gram_by_columns when the output would not fit"
    ),
    ("_agg_by_bin", "_weighted_bincount_2d"): (
        "same: guarded by _agg_by_bin_fits at every caller. The generic tail "
        "has also already materialised gm.toarray() at (n, p), so the "
        "aggregate is never the dominant allocation on that branch"
    ),
}

# Entries above that are guarded rather than structurally impossible: these
# name a guard that must exist, so the pairing test can check the guard is real.
_AGGREGATE_CALLS_GUARDED_AT_CALLER = {
    "_agg_by_bin": "_agg_by_bin_fits",
}


def _aggregate_call_sites():
    """Every call to a cross-shaped aggregation kernel, with its function."""
    import ast
    import pathlib

    import superglm._group_matrix._group_matrix_algebra as algebra

    source = pathlib.Path(algebra.__file__).read_text()
    tree = ast.parse(source)
    kernels = {"_weighted_bincount_2d", "_csr_weighted_bincount"}
    sites = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Name)
                and inner.func.id in kernels
            ):
                sites.append((node.name, inner.func.id, inner.lineno))
    return sites


def test_every_cross_shaped_aggregate_is_bounded():
    """The invariant, not another site-specific test.

    Eight instances of one bug were found across five review rounds, each fix
    revealing the next, and the eighth was inside the helper written to fix the
    seventh.  They are all the same sentence: an aggregate whose output shape
    takes its row count from one block and its column count from the other,
    while every gate in the subsystem bounds a block against its OWN width.

    The 2-D histogram kernel never produced one of these, because every one of
    its call sites is guarded by a cell cap.  The two bincount kernels produced
    all eight, because none of theirs was.  So the rule is enforced here: a
    cross-shaped aggregate lives in a bounded helper, or it is listed with the
    reason it cannot grow.
    """
    unaccounted = []
    for function, kernel, lineno in _aggregate_call_sites():
        if function in _BOUNDED_AGGREGATE_HELPERS:
            continue
        if (function, kernel) in _AGGREGATE_CALLS_BOUNDED_BY_CONSTRUCTION:
            continue
        unaccounted.append(f"{function}:{lineno} calls {kernel}")

    assert not unaccounted, (
        "cross-shaped aggregate with no bound:\n  "
        + "\n  ".join(unaccounted)
        + "\n\nThis kernel allocates (n_bins, n_cols). If those two dimensions "
        "come from DIFFERENT blocks, no gate in this subsystem bounds their "
        "product -- the support gate bounds n_support against its own block's "
        "width, and the histogram caps bound bin counts against each other. "
        "Route the call through a helper in _BOUNDED_AGGREGATE_HELPERS, or add "
        "it to _AGGREGATE_CALLS_BOUNDED_BY_CONSTRUCTION with the reason its "
        "output cannot grow."
    )


def test_every_agg_by_bin_caller_is_guarded():
    """Every caller of ``_agg_by_bin``, whatever its enclosing function is named.

    The first version of this audit walked only functions named ``_cross_gram``.
    ``_random_effect_cross_gram`` calls ``_agg_by_bin`` directly, so a
    high-cardinality random effect beside a wide raw-basis SSP term allocated
    ``n_levels x p_b`` with the audit green -- the scope hole was pre-declared as
    "syntactic and single-module" and found inside the hour. Scope is now every
    function in the module.
    """
    import ast
    import pathlib

    import superglm._group_matrix._group_matrix_algebra as algebra

    guard = _AGGREGATE_CALLS_GUARDED_AT_CALLER["_agg_by_bin"]
    assert hasattr(algebra, guard), f"the named guard {guard} does not exist"

    tree = ast.parse(pathlib.Path(algebra.__file__).read_text())
    unguarded = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        calls = [
            inner
            for inner in ast.walk(node)
            if isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Name)
            and inner.func.id == "_agg_by_bin"
        ]
        if not calls:
            continue
        guards = [
            inner
            for inner in ast.walk(node)
            if isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Name)
            and inner.func.id == guard
        ]
        if len(guards) < len(calls):
            unguarded.append(
                f"{node.name}: {len(calls)} call(s) to _agg_by_bin, "
                f"{len(guards)} call(s) to {guard} (line {calls[0].lineno})"
            )

    assert not unguarded, (
        "unguarded cross-shaped aggregate:\n  "
        + "\n  ".join(unguarded)
        + f"\n\n_agg_by_bin returns (n_bins, width-of-gm) with the two "
        "dimensions from different blocks. Every caller must check "
        f"{guard} first and fall back to _cross_gram_by_columns."
    )


def test_agg_by_bin_width_never_under_reports_the_allocated_width():
    """The load-bearing line of the guard, tested rather than left to review.

    ``_agg_by_bin_width`` infers the width the aggregate is ALLOCATED at by
    sniffing attributes, because it differs from ``shape[1]``: the SSP branches
    aggregate in basis space and apply ``R_inv`` afterwards, which is how the
    guard's first draft came to budget 4 where 600 was allocated. Under-report
    for any type and the guard permits exactly what it exists to stop, so this
    runs the real aggregation and compares against what the kernels were asked
    for.
    """
    from superglm._group_matrix import _group_matrix_algebra as algebra
    from superglm._group_matrix._group_matrix_core import (
        CategoricalGroupMatrix,
        DenseGroupMatrix,
        SparseGroupMatrix,
        SparseSSPGroupMatrix,
        SplineCategoricalGroupMatrix,
    )
    from superglm._group_matrix._group_matrix_discretized import (
        SupportCompressedSSPGroupMatrix,
    )

    gen = np.random.default_rng(150)
    n, n_bins = 2_000, 8
    bin_idx = gen.integers(0, n_bins, n).astype(np.intp)
    weights = np.abs(gen.normal(1.0, 0.2, n))
    basis = sp.csr_matrix(gen.normal(size=(n, 9)) * (gen.random((n, 9)) < 0.4))

    blocks = {
        # shape[1] is R_inv's width (4), the allocation is at _p_b (9).
        "SparseSSPGroupMatrix": SparseSSPGroupMatrix(basis, gen.normal(size=(9, 4))),
        "SparseGroupMatrix": SparseGroupMatrix(basis),
        "DenseGroupMatrix": DenseGroupMatrix(gen.normal(size=(n, 5))),
        "CategoricalGroupMatrix": CategoricalGroupMatrix(gen.integers(0, 4, n).astype(np.intp), 4),
        "SupportCompressedSSPGroupMatrix": SupportCompressedSSPGroupMatrix(
            gen.normal(size=(30, 6)),
            gen.normal(size=(6, 3)),
            gen.integers(0, 30, n).astype(np.intp),
        ),
        "SplineCategoricalGroupMatrix": SplineCategoricalGroupMatrix(
            basis, gen.normal(size=(9, 4)), np.arange(0, n, 2, dtype=np.intp)
        ),
    }

    for name, gm in blocks.items():
        widths: list[int] = []
        original_csr = algebra._csr_weighted_bincount
        original_2d = algebra._weighted_bincount_2d
        try:
            algebra._csr_weighted_bincount = lambda d, i, p, n_cols, b, w, nb, _o=original_csr: (
                widths.append(int(n_cols)) or _o(d, i, p, n_cols, b, w, nb)
            )
            algebra._weighted_bincount_2d = lambda b, w, M, nb, _o=original_2d: (
                widths.append(int(np.shape(M)[1])) or _o(b, w, M, nb)
            )
            result = algebra._agg_by_bin(gm, bin_idx, weights, n_bins)
        finally:
            algebra._csr_weighted_bincount = original_csr
            algebra._weighted_bincount_2d = original_2d

        claimed = algebra._agg_by_bin_width(gm)
        observed = max([*widths, int(result.shape[1])])
        assert claimed >= observed, (
            f"{name}: guard budgets against width {claimed} but the aggregation "
            f"allocated at width {observed}, so the guard under-counts by "
            f"{observed / max(claimed, 1):.1f}x"
        )


def test_bin_and_width_bounded_aggregates_do_not_grow_with_n(monkeypatch):
    """Measure the two allow-list entries that still rest on an argument.

    They claim `(n_bins1, K_cat)` is a bin count by a basis width, so a lossless
    support cannot inflate it. That is checkable: hold the bins and widths fixed,
    grow n tenfold, and the largest aggregate must not move.
    """
    from superglm._group_matrix import _group_matrix_algebra as algebra
    from superglm._group_matrix._group_matrix_discretized import (
        DiscretizedTensorGroupMatrix,
    )

    largest: dict[int, int] = {}
    for n in (5_000, 50_000):
        gen = np.random.default_rng(151)
        n_bins1, n_bins2, k1, k2 = 11, 7, 3, 2
        b1 = gen.normal(size=(n_bins1, k1))
        b2 = gen.normal(size=(n_bins2, k2))
        idx1 = gen.integers(0, n_bins1, n).astype(np.intp)
        idx2 = gen.integers(0, n_bins2, n).astype(np.intp)
        joint = np.einsum("ij,ik->ijk", b1[idx1], b2[idx2]).reshape(n, k1 * k2)
        tensor = DiscretizedTensorGroupMatrix(
            b1,
            b2,
            idx1,
            idx2,
            joint,
            gen.normal(size=(k1 * k2, 4)),
            np.arange(n, dtype=np.intp),
            0,
        )
        rows = np.arange(0, n, 2, dtype=np.intp)
        # n grows tenfold; the support and the widths deliberately do not.
        spline_cat = _spline_cat_support_block(n, 300, 5, 3, rows, seed=152)
        weights = np.abs(gen.normal(1.0, 0.2, n))

        # Force the over-cap fallback; the histogram path does not use this
        # kernel at all, so without this the spy records nothing.
        monkeypatch.setattr(algebra, "_MAX_DISC_DISC_HIST_CELLS", 1)
        cells: list[int] = []
        original = algebra._weighted_bincount_2d
        try:
            algebra._weighted_bincount_2d = lambda b, w, M, nb, _o=original: (
                cells.append(int(nb) * int(np.shape(M)[1])) or _o(b, w, M, nb)
            )
            algebra._cross_gram_tensor_spline_categorical(tensor, spline_cat, weights)
        finally:
            algebra._weighted_bincount_2d = original
        largest[n] = max(cells)

    assert largest[5_000] == largest[50_000], (
        f"the aggregate grew with n, so it is not bin-count x basis-width after all: {largest}"
    )


def test_the_invariant_test_can_actually_fail():
    """A guard that passes because it looks at nothing is worse than no guard."""
    sites = _aggregate_call_sites()

    assert sites, "the AST scan found no aggregation calls at all"
    assert any(fn in _BOUNDED_AGGREGATE_HELPERS for fn, _, _ in sites), (
        "no call site is inside a bounded helper, so the allow-list is doing "
        "all the work and a new site would land in neither category"
    )
    assert {fn for fn, _, _ in sites} - _BOUNDED_AGGREGATE_HELPERS, (
        "every site is in a helper, so the by-construction list is untested"
    )


def _mixed_spline_cat_pair(n, rows, seed):
    """One compressed block and one CSR block over the SAME rows.

    This is the pairing the compression gate creates: two ``spline_cat`` terms
    on one factor where one covariate repeats and the other does not.
    """
    from superglm._group_matrix._group_matrix_core import SplineCategoricalGroupMatrix

    gen = np.random.default_rng(seed)
    compressed = _spline_cat_support_block(n, 400, 5, 3, rows, seed=seed + 1)

    base = np.zeros((n, 6))
    for row in range(n):
        cols = gen.choice(6, size=2, replace=False)
        base[row, cols] = gen.normal(size=2)
    csr = SplineCategoricalGroupMatrix(sp.csr_matrix(base), gen.normal(size=(6, 2)), rows)
    csr.spline_cat_feature = "f"
    csr.spline_cat_level = "1"
    return compressed, csr


def test_mixed_compressed_and_csr_pair_never_densifies_an_observation_block(monkeypatch):
    """Reviewer P1 on 29f8e34, and a regression this PR introduced rather than
    exposed: before the class choice, two exact blocks contracted sparse
    against sparse.  Compressing one of them sent the pair down a branch that
    expanded the compressed side AND densified the weighted CSR side, making
    both worse than they had been.
    """
    from superglm._group_matrix import _group_matrix_algebra as algebra

    n = 30_000
    rows = np.arange(0, n, 2, dtype=np.intp)
    compressed, csr = _mixed_spline_cat_pair(n, rows, seed=110)
    weights = np.abs(np.random.default_rng(111).normal(1.0, 0.2, n))

    # Reference: the all-CSR pairing this used to be, which is what the mixed
    # branch has to reproduce.
    dense_i = compressed.toarray()
    dense_j = csr.toarray()
    expected = dense_i.T @ (dense_j * weights[:, None])

    seen = _spy_on_row_expansion(monkeypatch)
    forbid_dense = _forbid_csr_densify(monkeypatch)

    actual = algebra._cross_gram_spline_categorical_spline_categorical(compressed, csr, weights)

    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)
    _assert_expansion_bounded(seen)
    assert not forbid_dense, f"densified a CSR block of {forbid_dense} rows"

    # And the transposed orientation, which takes the other branch.
    actual_t = algebra._cross_gram_spline_categorical_spline_categorical(csr, compressed, weights)
    np.testing.assert_allclose(actual_t, expected.T, rtol=1e-9, atol=1e-9)
    _assert_expansion_bounded(seen)
    assert not forbid_dense


def test_mixed_pair_on_partial_row_overlap_never_densifies(monkeypatch):
    """Same, on the branch that intersects two different row sets."""
    from superglm._group_matrix import _group_matrix_algebra as algebra

    n = 30_000
    compressed, _ = _mixed_spline_cat_pair(n, np.arange(0, n, 2, dtype=np.intp), seed=112)
    _, csr = _mixed_spline_cat_pair(n, np.arange(0, n, 3, dtype=np.intp), seed=114)
    csr.spline_cat_feature = "g"
    weights = np.abs(np.random.default_rng(115).normal(1.0, 0.2, n))

    dense_i = compressed.toarray()
    dense_j = csr.toarray()
    expected = dense_i.T @ (dense_j * weights[:, None])

    seen = _spy_on_row_expansion(monkeypatch)
    forbid_dense = _forbid_csr_densify(monkeypatch)

    actual = algebra._cross_gram_spline_categorical_spline_categorical(compressed, csr, weights)

    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)
    _assert_expansion_bounded(seen)
    assert not forbid_dense, f"densified a CSR block of {forbid_dense} rows"


def _assert_expansion_bounded(seen):
    """The property is that expansion is BOUNDED, not that it never happens.

    Asserting "no expansion" pinned one implementation: when the mixed helper
    moved from aggregating the CSR side to contracting over bounded row chunks,
    three tests broke without anything getting worse. Bounded is the requirement.
    """
    from superglm._group_matrix import _group_matrix_algebra as algebra

    if not seen:
        return
    allowed = algebra._cross_expansion_chunk_rows(0, 0, algebra._MAX_CROSS_EXPANSION_BYTES)
    assert max(seen) <= allowed, (
        f"expanded {max(seen)} rows at once against a {allowed}-row ceiling"
    )


def _forbid_csr_densify(monkeypatch):
    """Record the row count of any CSR block densified to observation rows."""
    densified: list[int] = []
    for name in ("toarray", "todense"):
        original = getattr(sp.csr_matrix, name)

        def spy(self, *args, _orig=original, **kwargs):
            if self.shape[0] > 1_000:  # a (p_i, p_j) result is not the concern
                densified.append(self.shape[0])
            return _orig(self, *args, **kwargs)

        monkeypatch.setattr(sp.csr_matrix, name, spy)
    return densified


def test_tensor_fallback_bounds_its_channel_accumulator(monkeypatch):
    """Reviewer P1 on 8084eb7: row chunking bounded ``block`` but not ``agg``,
    which is ``(K2, n_bins1, K_cat)`` and follows a per-feature ``n_bins`` that
    nothing caps.

    Observable without measuring memory: holding every channel lets each row be
    expanded once, one channel at a time expands them K2 times. So the
    expansion count says which nesting ran.
    """
    from superglm._group_matrix import _group_matrix_algebra as algebra
    from superglm._group_matrix._group_matrix_discretized import (
        DiscretizedTensorGroupMatrix,
    )

    gen = np.random.default_rng(131)
    n, n_bins1, n_bins2 = 20_000, 40, 9
    k1, k2 = 3, 2
    b1 = gen.normal(size=(n_bins1, k1))
    b2 = gen.normal(size=(n_bins2, k2))
    idx1 = gen.integers(0, n_bins1, n).astype(np.intp)
    idx2 = gen.integers(0, n_bins2, n).astype(np.intp)
    joint = np.einsum("ij,ik->ijk", b1[idx1], b2[idx2]).reshape(n, k1 * k2)
    tensor = DiscretizedTensorGroupMatrix(
        b1,
        b2,
        idx1,
        idx2,
        joint,
        gen.normal(size=(k1 * k2, 4)),
        np.arange(n, dtype=np.intp),
        0,
    )
    rows = np.arange(0, n, 2, dtype=np.intp)
    spline_cat = _spline_cat_support_block(n, 300, 5, 3, rows, seed=132)
    weights = np.abs(gen.normal(1.0, 0.2, n))

    monkeypatch.setattr(algebra, "_MAX_DISC_DISC_HIST_CELLS", 1)  # force the fallback
    expected = algebra._cross_gram_tensor_spline_categorical(tensor, spline_cat, weights)

    # Below K2 * n_bins1 * K_cat (400) so the loops invert, but above n_bins1
    # (40) so tiling is meaningful rather than pinned at the irreducible floor.
    monkeypatch.setattr(algebra, "_MAX_AGGREGATE_CELLS", 100)
    seen = _spy_on_row_expansion(monkeypatch)
    actual = algebra._cross_gram_tensor_spline_categorical(tensor, spline_cat, weights)

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    _assert_expansion_bounded(seen)
    # One pass per (channel, basis tile). The tile count is derived, not
    # assumed, so this still pins the inversion when the tiling changes.
    k_cat = spline_cat.B_unique.shape[1]
    tile = algebra._aggregate_column_chunk(n_bins1, k_cat)
    n_tiles = -(-k_cat // tile)
    assert sum(seen) == k2 * n_tiles * rows.size, (
        f"expected {k2} channels x {n_tiles} tiles x {rows.size} rows, saw {sum(seen)}"
    )
    assert sum(seen) > rows.size, "the all-channel accumulator was still built"
    assert n_bins1 * tile <= algebra._MAX_AGGREGATE_CELLS, (
        "a single channel tile still exceeds the aggregate budget"
    )


def test_compressed_ssp_beside_a_wide_sparse_term_bounds_the_aggregate(monkeypatch):
    """Reviewer P1 on 8084eb7, against the ALLOW-LIST rather than the source.

    A support-compressed `DiscretizedSSPGroupMatrix` crossed with a sparse term
    sends `_agg_by_bin` the discrete block's `n_bins` and the sparse block's
    width. Different blocks, so the exemption's stated reason -- that both came
    from the same block -- was false, and a narrow large support beside a wide
    sparse term still allocated without limit.
    """
    from superglm._group_matrix import _group_matrix_algebra as algebra
    from superglm._group_matrix._group_matrix_core import SparseSSPGroupMatrix
    from superglm._group_matrix._group_matrix_discretized import (
        SupportCompressedSSPGroupMatrix,
    )

    gen = np.random.default_rng(130)
    n, n_bins, p_narrow, p_wide = 4_000, 900, 3, 600
    compressed = SupportCompressedSSPGroupMatrix(
        gen.normal(size=(n_bins, p_narrow)),
        gen.normal(size=(p_narrow, 2)),
        gen.integers(0, n_bins, n).astype(np.intp),
    )
    wide_basis = sp.csr_matrix(gen.normal(size=(n, p_wide)) * (gen.random((n, p_wide)) < 0.02))
    wide = SparseSSPGroupMatrix(wide_basis, gen.normal(size=(p_wide, 4)))
    weights = np.abs(gen.normal(1.0, 0.2, n))

    expected = compressed.toarray().T @ (wide.toarray() * weights[:, None])

    budget = 10_000  # n_bins * p_wide = 540,000, far over
    monkeypatch.setattr(algebra, "_MAX_AGGREGATE_CELLS", budget)
    allocated = []
    original = algebra._csr_weighted_bincount

    def spy(data, indices, indptr, n_cols, bin_idx, W, n_bins_arg):
        allocated.append(int(n_bins_arg) * int(n_cols))
        return original(data, indices, indptr, n_cols, bin_idx, W, n_bins_arg)

    monkeypatch.setattr(algebra, "_csr_weighted_bincount", spy)

    actual = algebra._cross_gram(compressed, wide, weights)

    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)
    assert all(cells <= budget for cells in allocated), (
        f"allocated {max(allocated)} cells against a {budget}-cell budget"
    )


def test_mixed_chunk_bounds_the_sparse_payload_not_just_the_expansion(monkeypatch):
    """Reviewer P1 on 2e3af3e: the chunk budgeted the dense expansion only.

    ``nnz`` per row belongs to the OTHER block -- a cardinal-CR basis is
    structurally dense -- so a narrow compressed side permits a huge row count
    and the weighted CSR slice follows it. Five compressed columns admit ~1.68M
    rows, which against a 20-column dense CSR is tens of millions of entries.
    """
    from superglm._group_matrix import _group_matrix_algebra as algebra

    gen = np.random.default_rng(160)
    n, n_support, p_b, p_csr = 6_000, 40, 5, 20
    b_unique = gen.normal(size=(n_support, p_b))
    support_idx = gen.integers(0, n_support, n).astype(np.intp)
    # Structurally dense, as a cardinal-CR interaction basis is.
    dense_csr = sp.csr_matrix(gen.normal(size=(n, p_csr)))
    weights = np.abs(gen.normal(1.0, 0.2, n))

    expected = b_unique[support_idx].T @ (dense_csr.toarray() * weights[:, None])

    budget = 4_000  # bytes
    monkeypatch.setattr(algebra, "_MAX_CROSS_EXPANSION_BYTES", budget)
    seen = _spy_on_row_expansion(monkeypatch)

    actual = algebra._support_csr_raw_cross(b_unique, support_idx, dense_csr, weights)

    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)
    assert seen, "expected the helper to run"

    assert sum(seen) == n, "the chunks must cover every row exactly once"

    # Do NOT re-derive ``nnz <= budget // 12`` from indptr here.  That is the
    # arithmetic ``_mixed_chunk_stop`` already used to CHOOSE the chunk, so
    # asserting it back observes no allocation and cannot fail -- which is
    # exactly how the 2.7x overshoot below survived the last time this class of
    # bug was fixed.  Measure the bytes instead; see the test that follows.


def test_mixed_chunk_peak_bytes_stay_inside_the_budget_the_sizing_assumes(monkeypatch):
    """The chunk budget assumes 12 B per stored entry.  Hold it to that.

    ``csr[a:b].multiply(w[:, None])`` costs three live buffers per entry, not
    one: the slice is a full copy and scipy routes the product through COO.
    Measured 32.2 B/entry, so a 64 MiB bound really peaked near 170 MiB.

    This asserts TRACED BYTES at the moment of the weighting, which is the
    quantity the budget is denominated in.  A structural assertion (no
    ``.multiply`` in the loop) would pin today's implementation rather than the
    property; a byte count survives a rewrite.
    """
    import tracemalloc

    from superglm._group_matrix import _group_matrix_algebra as algebra

    gen = np.random.default_rng(4)
    n_rows, p_csr = 12_000, 20
    dense_csr = sp.csr_matrix(gen.normal(size=(n_rows, p_csr)))
    weights = np.abs(gen.normal(1.0, 0.2, n_rows))
    nnz = int(dense_csr.nnz)

    tracemalloc.start()
    try:
        baseline = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        chunk = algebra._weighted_row_chunk(dense_csr, weights, 0, n_rows)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    bytes_per_entry = (peak - baseline) / nnz
    assert bytes_per_entry <= 12.0, (
        f"weighting the chunk peaked at {bytes_per_entry:.1f} bytes per stored "
        f"entry against the {12} the chunk budget assumes"
    )

    # Bit-exact against the route it replaces: same two floats, same order.
    reference = dense_csr.multiply(weights[:, None]).tocsr()
    assert np.array_equal(chunk.toarray(), reference.toarray())


def test_mixed_pairing_peak_stays_inside_the_two_budgets_it_maintains(monkeypatch):
    """End to end, through the public helper, on traced bytes.

    ``_support_csr_raw_cross`` maintains two budgets: the dense expansion of
    the compressed side, and the sparse payload beside it.  Their sum is the
    honest ceiling, so ~2x ``_MAX_CROSS_EXPANSION_BYTES`` is the bound -- and
    it holds regardless of how the weighting is implemented, which is what
    makes this survive a rewrite.

    Measured 3.11x before the chunk weighting was fixed and 1.73x after, so
    this discriminates on BEHAVIOUR rather than on the presence of a helper.
    """
    import tracemalloc

    from superglm._group_matrix import _group_matrix_algebra as algebra

    gen = np.random.default_rng(4)
    n_rows, p_b, p_csr, n_support = 12_000, 5, 20, 40
    b_unique = gen.normal(size=(n_support, p_b))
    support_idx = gen.integers(0, n_support, n_rows).astype(np.intp)
    dense_csr = sp.csr_matrix(gen.normal(size=(n_rows, p_csr)))
    weights = np.abs(gen.normal(1.0, 0.2, n_rows))

    budget = 1 << 16
    monkeypatch.setattr(algebra, "_MAX_CROSS_EXPANSION_BYTES", budget)

    expected = b_unique[support_idx].T @ (dense_csr.toarray() * weights[:, None])

    tracemalloc.start()
    try:
        baseline = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        actual = algebra._support_csr_raw_cross(b_unique, support_idx, dense_csr, weights)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)

    ratio = (peak - baseline) / budget
    assert ratio <= 2.0, (
        f"mixed pairing peaked at {ratio:.2f}x the {budget}-byte budget; the two "
        f"budgets it maintains (dense expansion, sparse payload) sum to 2x"
    )


def test_narrow_support_beside_a_wide_csr_term_bounds_the_aggregate(monkeypatch):
    """Reviewer P1 on 259fce2, inside the helper written to fix the previous
    P1: the compression gate bounds ``n_support`` against the COMPRESSED
    block's width, never against the paired CSR block's.  A narrow compressed
    term beside a wide CSR term passes both gates and still allocates without
    limit.  The shipped tests used widths 5 and 6 and could not see it."""
    from superglm._group_matrix import _group_matrix_algebra as algebra

    gen = np.random.default_rng(120)
    n, n_support, p_narrow, p_wide = 4_000, 40, 3, 700
    b_unique = gen.normal(size=(n_support, p_narrow))
    support_idx = gen.integers(0, n_support, n).astype(np.intp)
    wide = sp.csr_matrix(gen.normal(size=(n, p_wide)) * (gen.random((n, p_wide)) < 0.02))
    weights = np.abs(gen.normal(1.0, 0.2, n))

    expected = b_unique[support_idx].T @ (wide.toarray() * weights[:, None])

    # Lowered so a small fixture exercises the multi-pass path; the real budget
    # would need an n_support in the tens of thousands to trip on p_wide=700.
    budget = 4_000
    monkeypatch.setattr(algebra, "_MAX_AGGREGATE_CELLS", budget)

    # Observe the PROPERTY -- nothing observation-shaped and nothing
    # cross-shaped gets built -- rather than one implementation's mechanism.
    # An earlier version spied on the bincount kernel and broke when the helper
    # stopped using one, which is a test encoding the fix instead of the
    # requirement.
    seen = _spy_on_row_expansion(monkeypatch)
    forbid_dense = _forbid_csr_densify(monkeypatch)

    actual = algebra._support_csr_raw_cross(b_unique, support_idx, wide, weights)

    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)
    _assert_expansion_bounded(seen)
    assert not forbid_dense, f"densified the wide CSR term ({forbid_dense} rows)"
    assert sum(seen) == n, "each row should be expanded exactly once, in one pass"


def test_two_large_supports_never_build_a_cross_shaped_aggregate(monkeypatch):
    """Reviewer P1 on 259fce2: a compressed main effect beside a compressed
    ``spline_cat`` whose joint support exceeds the histogram cap used to fall
    through to ``_agg_by_bin``, whose output is ``(n_bins_main, p_spline_cat)``
    -- a row count from one block against a width from the other, bounded by
    neither support gate."""
    from superglm._group_matrix import _group_matrix_algebra as algebra
    from superglm._group_matrix._group_matrix_discretized import (
        SupportCompressedSSPGroupMatrix,
    )

    gen = np.random.default_rng(121)
    n, n_bins_main, p_main = 20_000, 300, 4
    main = SupportCompressedSSPGroupMatrix(
        gen.normal(size=(n_bins_main, p_main)),
        gen.normal(size=(p_main, 3)),
        gen.integers(0, n_bins_main, n).astype(np.intp),
    )
    rows = np.arange(0, n, 2, dtype=np.intp)
    spline_cat = _spline_cat_support_block(n, 250, 5, 3, rows, seed=122)
    weights = np.abs(gen.normal(1.0, 0.2, n))

    expected = algebra._cross_gram(main, spline_cat, weights)

    # Force the histogram cap to decline, which is what a large joint support does.
    monkeypatch.setattr(algebra, "_MAX_DISC_DISC_HIST_CELLS", 1)
    called = []
    original = algebra._agg_by_bin
    monkeypatch.setattr(
        algebra,
        "_agg_by_bin",
        lambda *a, **k: called.append(1) or original(*a, **k),
    )

    actual = algebra._cross_gram(main, spline_cat, weights)

    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)
    assert not called, (
        "declining at the cell cap fell through to _agg_by_bin, whose output "
        "is cross-shaped and bounded by neither support gate"
    )


def test_agg_by_bin_expands_the_spline_cat_level_in_chunks(monkeypatch):
    """Reviewer finding on d2c4863, and the site my own sweep missed.

    Reached when ``_cross_gram_discrete_spline_categorical`` declines at its
    cell cap and dispatch falls through to the disc-x-non-disc branch.  That
    fall-through is newly reachable because a lossless support makes ``n_bins``
    large on both sides -- binned at the 256 default the cell product is 65,536,
    three orders under the cap.
    """
    from superglm._group_matrix import _group_matrix_algebra as algebra

    gen = np.random.default_rng(90)
    n, n_bins, p_g = 40_000, 16, 4
    rows = np.arange(0, n, 2, dtype=np.intp)
    spline_cat = _spline_cat_support_block(n, 600, 5, p_g, rows, seed=91)
    bin_idx = gen.integers(0, n_bins, n).astype(np.intp)
    weights = np.abs(gen.normal(1.0, 0.2, n))

    expected = algebra._agg_by_bin(spline_cat, bin_idx, weights, n_bins)

    monkeypatch.setattr(algebra, "_MAX_CROSS_EXPANSION_BYTES", 1 << 12)
    seen = _spy_on_row_expansion(monkeypatch)

    actual = algebra._agg_by_bin(spline_cat, bin_idx, weights, n_bins)

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    assert seen, "expected the aggregation to expand support rows"
    allowed = algebra._cross_expansion_chunk_rows(p_g, 0, 1 << 12)
    assert max(seen) <= allowed, f"expanded {max(seen)} rows at once against a {allowed}-row budget"
    assert sum(seen) == rows.size


def test_support_gate_thresholds_are_not_frozen_at_import(monkeypatch):
    """A threshold bound as a default argument cannot be lowered by a test, so
    a test written to force a decision silently observes the default instead.

    This is the defect that already produced one false pass in this branch; it
    survived on all three support entry points.
    """
    from superglm._group_matrix import _group_matrix_support as support

    basis, _, row_index = _repeated_basis()  # a known accept-shape

    assert support.detect_row_support(basis) is not None
    assert support.plan_row_support(basis, row_index) is not None
    assert support.plan_verified_row_support(basis, row_index) is not None

    # An unreachable speedup must turn every one of them down.
    monkeypatch.setattr(support, "DEFAULT_MIN_SPEEDUP", 1e12)

    assert support.detect_row_support(basis) is None
    assert support.plan_row_support(basis, row_index) is None
    assert support.plan_verified_row_support(basis, row_index) is None


def test_spline_cat_pre_gate_and_inner_gate_cannot_disagree(monkeypatch):
    """``_plan_spline_cat_support`` runs a cheap gate and then calls one that
    re-runs it.  They must resolve the SAME threshold: the outer read its value
    at call time while the inner used a frozen default, so a patch made the two
    disagree and the block fell back to CSR while the test believed it had
    forced acceptance."""
    from superglm._group_matrix import _group_matrix_support as support
    from superglm.features.interaction import _plan_spline_cat_support

    n = 20_000
    # A shape the default threshold turns down: 40% distinct rows on a sparse
    # basis costs more than it saves.  x is the grouping itself, so the offered
    # grouping matches the basis and verification cannot be what declines.
    basis, _, row_index = _repeated_basis(n=n, n_support=n * 4 // 10, seed=93)
    x = row_index.astype(np.float64)
    active = np.arange(n, dtype=np.intp)

    assert _plan_spline_cat_support(basis, x, active, n_levels=2) is None

    # Patch the threshold DOWN, which is the direction a test forcing
    # acceptance would use.  With the inner gate frozen at the default, the
    # outer accepts on the patched value and the inner declines on the old one,
    # so this returns None and reads as "the gate declined" -- believable and
    # wrong.  Both gates must now resolve the same number.
    monkeypatch.setattr(support, "DEFAULT_MIN_SPEEDUP", 0.5)
    assert _plan_spline_cat_support(basis, x, active, n_levels=2) is not None


def test_tensor_by_spline_cat_fallback_expands_rows_in_chunks(monkeypatch):
    """Reviewer finding on 814a0e0, the sibling of the bounded support-support
    fallback: over the cell cap this branch materialised the whole level."""
    from superglm._group_matrix import _group_matrix_algebra as algebra
    from superglm._group_matrix._group_matrix_discretized import (
        DiscretizedTensorGroupMatrix,
    )

    gen = np.random.default_rng(70)
    n, n_bins1, n_bins2 = 40_000, 12, 9
    b1 = gen.normal(size=(n_bins1, 3))
    b2 = gen.normal(size=(n_bins2, 2))
    idx1 = gen.integers(0, n_bins1, n).astype(np.intp)
    idx2 = gen.integers(0, n_bins2, n).astype(np.intp)
    joint = np.einsum("ij,ik->ijk", b1[idx1], b2[idx2]).reshape(n, 6)
    tensor = DiscretizedTensorGroupMatrix(
        b1, b2, idx1, idx2, joint, gen.normal(size=(6, 4)), np.arange(n, dtype=np.intp), 0
    )
    rows = np.arange(0, n, 2, dtype=np.intp)
    spline_cat = _spline_cat_support_block(n, 400, 5, 3, rows, seed=71)
    weights = np.abs(gen.normal(1.0, 0.2, n))

    expected = algebra._cross_gram_tensor_spline_categorical(tensor, spline_cat, weights)

    # Force the over-cap branch, then hold it to the expansion budget.
    monkeypatch.setattr(algebra, "_MAX_DISC_DISC_HIST_CELLS", 1)
    monkeypatch.setattr(algebra, "_MAX_CROSS_EXPANSION_BYTES", 1 << 13)
    seen = _spy_on_row_expansion(monkeypatch)

    actual = algebra._cross_gram_tensor_spline_categorical(tensor, spline_cat, weights)

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    assert seen, "expected the over-cap branch to expand support rows"
    allowed = algebra._cross_expansion_chunk_rows(5, 0, 1 << 13)
    assert max(seen) <= allowed, f"expanded {max(seen)} rows at once against a {allowed}-row budget"
    # Each row expanded exactly once across all K2 passes, not once per pass.
    assert sum(seen) == rows.size


def test_categorical_by_spline_cat_expands_rows_in_chunks(monkeypatch):
    """Found by sweeping for the same class rather than reported.

    Pre-dates support compression, but compression makes it hot: a Categorical
    main effect beside a spline_cat term is every model this targets.
    """
    from superglm._group_matrix import _group_matrix_algebra as algebra
    from superglm._group_matrix._group_matrix_core import CategoricalGroupMatrix

    gen = np.random.default_rng(72)
    n = 40_000
    codes = gen.integers(0, 4, n).astype(np.int32)
    gm_cat = CategoricalGroupMatrix(codes, 4)
    rows = np.arange(0, n, 2, dtype=np.intp)
    spline_cat = _spline_cat_support_block(n, 300, 6, 3, rows, seed=73)
    weights = np.abs(gen.normal(1.0, 0.2, n))

    expected = algebra._cross_gram_categorical_spline_categorical(gm_cat, spline_cat, weights)

    monkeypatch.setattr(algebra, "_MAX_CROSS_EXPANSION_BYTES", 1 << 13)
    seen = _spy_on_row_expansion(monkeypatch)

    actual = algebra._cross_gram_categorical_spline_categorical(gm_cat, spline_cat, weights)

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    assert seen, "expected the aggregation to expand support rows"
    allowed = algebra._cross_expansion_chunk_rows(6, 0, 1 << 13)
    assert max(seen) <= allowed, f"expanded {max(seen)} rows at once against a {allowed}-row budget"
    assert sum(seen) == rows.size


def test_row_chunking_below_the_threshold_is_bit_identical():
    """Chunking reorders a sum, so it must not engage on ordinary fits."""
    from superglm._group_matrix import _group_matrix_algebra as algebra
    from superglm._group_matrix._group_matrix_kernels import _weighted_bincount_2d

    gen = np.random.default_rng(74)
    n, n_bins, p_b = 5_000, 7, 6
    b_unique = gen.normal(size=(50, p_b))
    support_idx = gen.integers(0, 50, n).astype(np.intp)
    bin_idx = gen.integers(0, n_bins, n).astype(np.intp)
    weights = gen.normal(0.0, 1.0, n)

    one_shot = _weighted_bincount_2d(bin_idx, weights, b_unique[support_idx], n_bins)
    chunked = algebra._chunked_support_bincount_2d(bin_idx, weights, b_unique, support_idx, n_bins)

    # Default budget dwarfs this block, so the loop runs once: bit-identical.
    np.testing.assert_array_equal(chunked, one_shot)


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
