"""Chunked loops in the design-matrix, feature, screening and metrics code, across their chunk edges.

A loop over ``range(0, N, STEP)`` meets one chunk on any fixture smaller than its
production step, so a loop that drops or repeats the rows past the first chunk
passes every such test.  Each test here shrinks the step through its own knob (a
module constant read at call time, a byte budget the step is derived from, or the
function's chunk argument) until the data spans several chunks with a ragged last
one, and compares the result with the same quantity on the whole data or with a
NumPy reference formed in the test, within a bound derived from the arithmetic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as sp

from superglm import Numeric, SuperGLM
from superglm._group_matrix import _group_matrix_algebra as algebra
from superglm._group_matrix import _group_matrix_core as core
from superglm._group_matrix._group_matrix_discretized import (
    DiscretizedTensorGroupMatrix,
    SupportCompressedSplineCategoricalGroupMatrix,
)
from superglm.features.spline import Spline
from superglm.group_matrix import (
    CategoricalGroupMatrix,
    DenseGroupMatrix,
    DesignMatrix,
    DiscretizedSSPGroupMatrix,
    SparseSSPGroupMatrix,
)

U = 2.0**-53


def _gamma(k: int) -> float:
    """``gamma_k = k u / (1 - k u)`` (Higham 2002, sec. 3.1)."""
    return k * U / (1 - k * U)


def _assert_within(actual, expected, bound) -> None:
    """Every entry of ``actual`` lies within ``bound`` of ``expected``."""
    actual, expected, bound = map(np.asarray, (actual, expected, bound))
    assert actual.shape == expected.shape
    excess = np.abs(actual - expected) - bound
    assert np.all(excess <= 0.0), f"worst excess {np.max(excess):.3e} over the bound"


def _csr_reversing_row(dense: np.ndarray, row: int) -> sp.csr_matrix:
    """``dense`` as CSR with one row's entries stored in reverse column order: not canonical."""
    canonical = sp.csr_matrix(dense)
    start, stop = canonical.indptr[row], canonical.indptr[row + 1]
    data, indices = canonical.data.copy(), canonical.indices.copy()
    data[start:stop] = data[start:stop][::-1]
    indices[start:stop] = indices[start:stop][::-1]
    matrix = sp.csr_matrix((data, indices, canonical.indptr.copy()), shape=dense.shape)
    assert not matrix.has_canonical_format
    return matrix


def _assert_weighted_moments(moments, dense: np.ndarray, W: np.ndarray, chunks: int) -> None:
    """``(gram, X'W1, centred gram)`` against the dense design's two-pass moments.

    The raw sums are one sum of ``n`` products per entry in any chunking, so each
    route is within ``gamma_{n+2}`` of the exact sum of absolute terms.  The
    centred Gram merges ``chunks`` co-moments, accurate to ``O(gamma_{n+K})`` times
    ``sqrt(C_ii C_jj)`` (``_anchored_weighted_moments``); the dense two-pass
    reference is within ``gamma_{n+3}`` of the same scale, its centre's rounding
    entering only at second order.
    """
    gram, xtw1, centred = moments
    n = dense.shape[0]
    absolute = np.abs(dense)
    _assert_within(
        gram,
        dense.T @ (W[:, None] * dense),
        2 * _gamma(n + 2) * absolute.T @ (W[:, None] * absolute),
    )
    _assert_within(xtw1, dense.T @ W, 2 * _gamma(n + 1) * absolute.T @ W)
    deviation = dense - (dense.T @ W) / W.sum()
    expected = deviation.T @ (W[:, None] * deviation)
    scale = np.sqrt(np.outer(np.diag(expected), np.diag(expected)))
    _assert_within(centred, expected, 4 * _gamma(n + chunks + 4) * scale)


def test_anchor_centring_sums_and_writes_every_support_chunk(monkeypatch):
    """Anchor-centred support rows in four-row chunks, against NumPy on the whole support.

    The shift ``mass' (v - anchor) / sum_w`` is one sum over the 37 support rows
    in any chunking, and each centred row ``((v - anchor) - shift) T`` is formed
    from its own row.  Both routes round ``v - anchor`` identically, so the shift
    differs by at most ``2 gamma_{b+1} mass' |v - anchor| / sum_w`` (``b`` support
    rows) and a centred entry by its products' ``2 gamma_{q+2}`` on ``(|v - anchor|
    + |shift|) |T|``, plus that shift difference carried through ``|T|``.
    """
    from superglm._group_matrix import _column_local_centering as local

    rng = np.random.default_rng(20261119)
    n, bins, q = 900, 37, 4
    support = 10.0 + rng.normal(size=(bins, q))
    transform = rng.normal(size=(q, 3))
    codes = np.concatenate([np.arange(bins), rng.integers(0, bins, n - bins)])
    W = rng.uniform(0.5, 2.0, n)
    weighted_z = W * rng.normal(size=n)
    sum_w = float(W.sum())
    columns = np.array([0, 2])
    group = DiscretizedSSPGroupMatrix(support, transform, codes)
    monkeypatch.setattr(local, "_SUPPORT_CHUNK_BYTES", 8 * q * 4)  # four support rows a chunk
    centred, _, (_, shift), _, _ = local._anchor_centred_columns(
        group, columns, W, weighted_z, sum_w
    )

    mass = np.bincount(codes, weights=W, minlength=bins)
    offset = support - support[int(np.argmax(mass))]
    raw_shift = (mass @ offset) / sum_w
    T = transform[:, columns]
    shift_error = 2 * _gamma(bins + 1) * (mass @ np.abs(offset)) / sum_w
    _assert_within(
        shift, raw_shift @ T, (shift_error + 2 * _gamma(q + 1) * np.abs(raw_shift)) @ np.abs(T)
    )
    _assert_within(
        centred,
        (offset - raw_shift) @ T,
        2 * _gamma(q + 2) * ((np.abs(offset) + np.abs(raw_shift)) @ np.abs(T))
        + shift_error @ np.abs(T),
    )


def test_tensor_by_spline_cat_channel_tiles_sum_every_row_chunk(monkeypatch):
    """The per-channel tile path of a tensor x spline_cat cross, its level rows in 389-row chunks.

    Against the projected rows' product ``T' W S`` formed in NumPy.  Both are one
    sum over the level's rows of products of four factors, mapped through ``R_t``
    and ``R_c``: each is within ``gamma_{n+16}`` of the exact value on the scale
    ``|R_t|' (|B_t|' W |B_c|) |R_c|`` of the raw rows.
    """
    rng = np.random.default_rng(1909)
    n, n_bins1, n_bins2, k1, k2 = 6_000, 40, 9, 3, 2
    b1 = rng.normal(size=(n_bins1, k1))
    b2 = rng.normal(size=(n_bins2, k2))
    idx1 = rng.integers(0, n_bins1, n).astype(np.intp)
    idx2 = rng.integers(0, n_bins2, n).astype(np.intp)
    joint = np.einsum("ij,ik->ijk", b1[idx1], b2[idx2]).reshape(n, k1 * k2)
    R_t = rng.normal(size=(k1 * k2, 4))
    tensor = DiscretizedTensorGroupMatrix(
        b1, b2, idx1, idx2, joint, R_t, np.arange(n, dtype=np.intp), 0
    )
    rows = np.arange(0, n, 2, dtype=np.intp)
    base, R_c = rng.normal(size=(300, 5)), rng.normal(size=(5, 3))
    row_index = rng.integers(0, 300, n).astype(np.intp)
    spline_cat = SupportCompressedSplineCategoricalGroupMatrix(base, R_c, row_index, rows)
    spline_cat.spline_cat_feature, spline_cat.spline_cat_level = "f", "1"
    W = rng.uniform(0.5, 1.5, n)

    # No joint histogram and an aggregate cap below K2 * n_bins1 * K_cat (400): the
    # channel-by-channel tiles, each over row chunks of the expansion budget.
    monkeypatch.setattr(algebra, "_MAX_DISC_DISC_HIST_CELLS", 1)
    monkeypatch.setattr(algebra, "_MAX_AGGREGATE_CELLS", 100)
    monkeypatch.setattr(algebra, "_MAX_CROSS_EXPANSION_BYTES", 8 * 5 * 389)
    expanded: list[int] = []
    expand = algebra._expand_support_rows
    monkeypatch.setattr(
        algebra,
        "_expand_support_rows",
        lambda values, codes: expanded.append(len(codes)) or expand(values, codes),
    )
    actual = algebra._cross_gram_tensor_spline_categorical(tensor, spline_cat, W)
    assert max(expanded) == 389 < rows.size

    dense_cat = spline_cat.toarray()
    raw_cat = np.zeros((n, base.shape[1]))
    raw_cat[rows] = np.abs(base[row_index[rows]])
    magnitude = np.abs(R_t).T @ (np.abs(joint).T @ (W[:, None] * raw_cat)) @ np.abs(R_c)
    expected = tensor.toarray().T @ (W[:, None] * dense_cat)
    _assert_within(actual, expected, 2 * _gamma(n + 16) * magnitude)


def test_expanded_support_cross_sums_every_row_chunk():
    """``B_i' W B_j`` over 173-row chunks of expanded support rows, against the expanded rows' product.

    The contraction is one sum of ``n`` products of three factors per entry, so
    either evaluation is within ``gamma_{n+2}`` of the exact sum of absolute terms.
    """
    rng = np.random.default_rng(2163)
    n = 2_000
    b_i, b_j = rng.normal(size=(60, 5)), rng.normal(size=(40, 7))
    idx_i = rng.integers(0, 60, n).astype(np.intp)
    idx_j = rng.integers(0, 40, n).astype(np.intp)
    W = rng.normal(size=n)  # signed, as REML passes
    actual = algebra._support_support_raw_cross_expanded(
        b_i, idx_i, b_j, idx_j, W, max_bytes=8 * (5 + 7) * 173
    )
    left, right = b_i[idx_i], b_j[idx_j]
    magnitude = np.abs(left).T @ (np.abs(W)[:, None] * np.abs(right))
    _assert_within(actual, left.T @ (W[:, None] * right), 2 * _gamma(n + 2) * magnitude)


def test_cancellation_sensitive_sparse_grams_sum_every_projected_row_block(monkeypatch):
    """A cancelling SSP block's own Gram and its cross Gram, projected in 389-row blocks.

    The basis' second column is ``1 + 1e-8 x``, and the transform differences it
    against the first, so the projected rows ``B R`` are exact (Sterbenz) on both
    routes and only the weighted sum over rows rounds: within ``gamma_{n+2}`` of
    the exact sum of absolute terms, whatever the blocking.
    """
    n, p = 3_000, 8
    basis = np.zeros((n, p))
    basis[:, 0], basis[:, 1] = 1.0, 1.0 + 1e-8 * np.linspace(-1.0, 1.0, n)
    transform = np.zeros((p, p))
    transform[:2, 0], transform[0, 1:] = [1.0, -1.0], 1.0
    left, right = (SparseSSPGroupMatrix(sp.csr_matrix(basis), transform.copy()) for _ in range(2))
    W = np.linspace(0.5, 1.5, n)
    pointer = left.B.indptr.dtype.itemsize
    # Both routes leave NumPy's iterator buffer and two row pointers out of the budget,
    # then spend 2 * 8 * p bytes and two pointers a row.
    budget = 8 * np.getbufsize() + 2 * pointer + (16 * p + 2 * pointer) * 389
    monkeypatch.setattr(core, "_MAX_SSP_GRAM_WORKSPACE_BYTES", budget)
    monkeypatch.setattr(algebra, "_MAX_CROSS_EXPANSION_BYTES", budget)
    blocks: list[int] = []
    for module in (core, algebra):
        chunk = module._csr_row_chunk
        monkeypatch.setattr(
            module,
            "_csr_row_chunk",
            lambda csr, start, stop, chunk=chunk: (
                blocks.append(stop - start) or chunk(csr, start, stop)
            ),
        )

    own = left.gram(W)
    assert max(blocks) == 389 < n
    blocks.clear()
    cross = algebra._cross_gram_sparse_ssp(left, right, W)
    assert max(blocks) == 389 < n

    projected = basis @ transform
    expected = projected.T @ (W[:, None] * projected)
    bound = 2 * _gamma(n + 2) * np.abs(projected).T @ (W[:, None] * np.abs(projected))
    _assert_within(own, expected, bound)
    _assert_within(cross, expected, bound)


def test_a_saturated_gram_reads_rows_as_a_dense_view_only_after_checking_each_one(monkeypatch):
    """A full CSR basis with one row stored out of order, past the first ten-row check block.

    Only a check of every row sees that the stored entries are not the dense
    rows in order; the blocked dense products then render each block, and the
    Gram is within ``gamma_{n+2}`` of the exact sum of absolute terms.
    """
    rows, width = 40, 4
    dense = (1 + np.arange(rows * width).reshape(rows, width) % 11) / 16.0
    basis = _csr_reversing_row(dense, 25)
    W = np.linspace(0.5, 1.5, rows)
    pointer, index = basis.indptr.dtype.itemsize, basis.indices.dtype.itemsize
    monkeypatch.setattr(core, "_MAX_SSP_GRAM_WORKSPACE_BYTES", 10 * (4 * pointer + 2 * index + 2))
    actual = core._saturated_ssp_gram(basis, W)
    magnitude = np.abs(dense).T @ (W[:, None] * np.abs(dense))
    _assert_within(actual, dense.T @ (W[:, None] * dense), 2 * _gamma(rows + 2) * magnitude)


def test_centred_right_hand_sides_sum_every_row_chunk():
    """``(X - c)' r`` in seven-row chunks, on the compact route and on the row-chunked fallback.

    Each row is centred by one rounding on every route, identically, so the
    results differ from the dense product only in the order of one sum of ``n``
    products: within ``gamma_{n+4}`` of the exact sum of absolute terms each.
    The fallback's SSP rows are dyadic with small integer transform entries, so
    every route forms them exactly.
    """
    from superglm._group_matrix import _group_matrix_centered as centered

    rng = np.random.default_rng(1310)
    n = 101
    dense = DenseGroupMatrix(1e3 + rng.normal(size=(n, 2)))
    level = CategoricalGroupMatrix(rng.integers(-1, 3, n), 3)
    raw = (1 + np.arange(n * 4).reshape(n, 4) % 7) / 8.0
    raw[::3, 1] = 0.0
    ssp = SparseSSPGroupMatrix(sp.csr_matrix(raw), rng.integers(-2, 3, size=(4, 3)).astype(float))
    W = rng.uniform(0.5, 2.0, n)
    z = rng.normal(size=n)

    for groups, route in (([dense, level], "compact"), ([dense, ssp], "fallback")):
        dm = DesignMatrix(groups, n, sum(group.shape[1] for group in groups))
        X = dm.toarray()
        centre = X.T @ W / W.sum()
        if route == "compact":
            actual = centered._compact_centered_rmatvec(
                dm=dm, rows=W * z, mean_x=centre, mean_lo=None, chunk_size=7
            )
        else:
            assert (
                centered._compact_centered_rmatvec(dm=dm, rows=W * z, mean_x=centre, mean_lo=None)
                is None
            )
            actual = centered.centered_rhs(dm=dm, W=W, mean_x=centre, z_centered=z, chunk_size=7)
        deviation = X - centre
        bound = 2 * _gamma(n + 4) * np.abs(deviation).T @ np.abs(W * z)
        _assert_within(actual, deviation.T @ (W * z), bound)


def test_a_wrong_grouping_in_the_last_row_chunk_fails_verification():
    """A non-canonical basis verifies its grouping in four-row chunks, the last one ragged.

    The grouping is exact bit comparison, so the representatives are the dense
    rows themselves and a single wrong label in row 21 (the sixth chunk) refuses.
    """
    from superglm._group_matrix import _group_matrix_support as support

    patterns = np.array([[0.5, 0.0, 1.25], [0.0, 2.0, 0.0], [1.0, 1.0, 0.0], [3.0, 0.0, 0.75]])
    labels = np.arange(23, dtype=np.intp) % 4
    basis = _csr_reversing_row(patterns[labels], 4)
    first = np.arange(4, dtype=np.intp)
    representatives = support._verified_representatives(basis, first, labels, 4)
    np.testing.assert_array_equal(representatives, patterns)
    wrong = labels.copy()
    wrong[21] = (wrong[21] + 1) % 4
    assert support._verified_representatives(basis, first, wrong, 4) is None


def test_a_text_columns_length_segment_covers_every_slice(monkeypatch):
    """The byte stream of a text column fed in four-row slices, against the one built here.

    One length segment (uint32 per row, zero for a null), the total text length,
    then the non-null values' UTF-8 bytes: bytes, compared exactly.
    """
    from superglm import _frame as frame_module

    words = ["α", "bc", "déjà", "日本語", "x" * 9, "y"]
    values = [None if i % 6 == 4 else f"{words[(3 * i) % len(words)]}{i}" for i in range(29)]

    class Recorder:
        def __init__(self):
            self.parts: list[bytes] = []

        def update(self, data) -> None:
            self.parts.append(bytes(data))

    recorder = Recorder()
    monkeypatch.setattr(frame_module, "_TEXT_CHUNK_ROWS", 4)
    frame_module._update_polars_text(recorder, pl.Series(values, dtype=pl.String))

    lengths = np.array([0 if v is None else len(v.encode()) for v in values], dtype="<u4")
    text = "".join(v for v in values if v is not None).encode("utf-8")
    expected = (
        lengths.nbytes.to_bytes(8, "little")
        + lengths.tobytes()
        + len(text).to_bytes(8, "little")
        + text
    )
    assert b"".join(recorder.parts) == expected


def test_pooled_row_blocks_partition_the_rows():
    """The blocks are consecutive, cover every row once, and end on a ragged block."""
    from superglm._parallel import pooled_row_blocks

    blocks = pooled_row_blocks(lambda start, stop: (start, stop), 23, 5)
    assert blocks == [(0, 5), (5, 10), (10, 15), (15, 20), (20, 23)]


def test_the_numeric_numeric_factor_reduces_every_row_chunk(monkeypatch):
    """``R'R`` of the pair factor reduced in 317-row chunks, against ``X'X`` of its weighted rows.

    Each of the ``K`` chunk merges is a Householder QR of at most ``chunk + 5``
    rows and 5 columns, backward stable column by column (Higham 2002, Thm.
    19.4), and the reference ``X'X`` is within ``gamma_n ||x_j|| ||x_k||``, so the
    entries agree within ``(4 K gamma_{5 (chunk + 5)} + gamma_n) ||x_j|| ||x_k||``.
    """
    from superglm.screening import _pair_factor as pair_factor

    rng = np.random.default_rng(611)
    n, chunk = 2_000, 317
    z1, z2 = rng.normal(size=n), rng.normal(scale=3.0, size=n)
    score, w = rng.normal(size=n), rng.uniform(0.2, 2.0, n)
    monkeypatch.setattr(pair_factor, "_FACTOR_CHUNK_DOUBLES", 5 * chunk)
    factor = pair_factor.numeric_numeric_factor(z1, z2, score, w)

    root = np.sqrt(w)
    X = np.column_stack((root, root * z2, root * z1, root * z1 * z2, score / root))
    norms = np.linalg.norm(X, axis=0)
    merges = -(-n // chunk)
    bound = (4 * merges * _gamma(5 * (chunk + 5)) + _gamma(n)) * np.outer(norms, norms)
    _assert_within(factor.joint.T @ factor.joint, X.T @ X, bound)


def test_structured_trace_and_base_rows_merge_every_level_chunk(monkeypatch):
    """A spline_cat pair's profiled trace and unemitted overlap rows, over chunks of levels.

    With the chunk budget at 54 doubles the trace runs 13 levels in chunks of
    three and the five unemitted levels in chunks of two.  The trace is
    ``||(I - P) C||_F^2`` for the emitted interaction columns ``C`` off the overlap
    span ``[B | levels]``, here formed by a QR of the weighted observation rows;
    its forward error is bounded through the overlap's condition number as in
    ``test_structured_variance_matches_the_observation_space_quadratic``.  The
    unemitted rows' factor reproduces their Gram within the backward error of
    its QR merges, as for the numeric pair factor.
    """
    from superglm.screening import _structured as structured

    rng = np.random.default_rng(448)
    x = np.linspace(-1.0, 1.0, 9)
    B = np.column_stack((x, x**2))
    n_levels, emitted = 13, np.arange(8)
    W_cell = rng.uniform(0.5, 2.0, (x.size, n_levels))
    monkeypatch.setattr(structured, "_TRACE_CHUNK_DOUBLES", 54)
    assert structured._trace_chunk_width(x.size, 2, n_levels) == 3
    assert structured._trace_chunk_width(x.size, 3, n_levels - emitted.size) == 2
    pair = structured.spline_cat_moments(B, np.eye(2), np.zeros_like(W_cell), W_cell, emitted)

    level = np.repeat(np.arange(n_levels), x.size)
    point = np.tile(np.arange(x.size), n_levels)
    root = np.sqrt(W_cell[point, level])[:, None]
    overlap = root * np.column_stack((B[point], np.eye(n_levels)[level]))
    candidate = root * np.column_stack([B[point] * (level == q)[:, None] for q in emitted])
    Q = np.linalg.qr(overlap)[0]
    residual = candidate - Q @ (Q.T @ candidate)
    condition = np.linalg.cond(overlap)
    bound = 128 * overlap.shape[0] * U * condition**2 * np.sum(candidate**2)
    _assert_within(pair.profiled_trace, np.sum(residual**2), bound)

    base = level >= emitted.size
    rows = root[base] * np.column_stack((np.ones(x.size * 5), B[point[base]]))
    norms = np.linalg.norm(rows, axis=0)
    merges = 2 * 3  # a batched QR and a merge for each of the three chunks
    bound = (4 * merges * _gamma(3 * (2 * x.size + 3)) + _gamma(rows.shape[0])) * np.outer(
        norms, norms
    )
    _assert_within(pair.overlap_rows.T @ pair.overlap_rows, rows.T @ rows, bound)


def test_tensor_design_rows_fill_every_chunk(monkeypatch):
    """Tensor design rows formed in eight-row chunks, against the same 61 rows formed whole.

    Each row is ``kron(B1 P1, B2 P2) R`` of its own covariates; a chunk can change
    only the blocking of the two products, so the two agree within ``2
    gamma_{k1 + k2 + p + 1}`` of ``kron(|B1| |P1|, |B2| |P2|) |R|``.
    """
    from superglm.features import _basis_kernels
    from superglm.features import interaction as interaction_module

    rng = np.random.default_rng(2706)
    x1, x2 = rng.uniform(0.0, 10.0, 61), rng.uniform(0.0, 5.0, 61)
    margins = {"a": Spline(kind="ps", n_knots=5), "b": Spline(kind="ps", n_knots=4)}
    margins["a"].build(x1)
    margins["b"].build(x2)
    tensor = interaction_module.TensorInteraction("a", "b", n_knots=(5, 4))
    tensor.build(x1, x2, margins)
    R = rng.normal(size=(tensor._p1 * tensor._p2, 7))
    tensor.set_reparametrisation(R)
    monkeypatch.setattr(interaction_module, "_TENSOR_SCORE_CHUNK_SIZE", 8)
    chunked = tensor.transform(x1, x2)
    monkeypatch.setattr(interaction_module, "_TENSOR_SCORE_CHUNK_SIZE", x1.size)
    whole = tensor.transform(x1, x2)

    magnitudes, widths = [], []
    for x, info in ((x1, tensor._marginal1), (x2, tensor._marginal2)):
        raw = _basis_kernels.dense_raw_basis(info.raw_basis_eval, np.clip(x, info.lo, info.hi))
        magnitudes.append(np.abs(raw) @ np.abs(info.projection))
        widths.append(raw.shape[1])
    rows = (magnitudes[0][:, :, None] * magnitudes[1][:, None, :]).reshape(x1.size, -1)
    bound = 2 * _gamma(sum(widths) + R.shape[0] + 1) * rows @ np.abs(R)
    _assert_within(chunked, whole, bound)


def test_evaluation_design_moments_sum_every_frame_chunk(monkeypatch):
    """A fitted model's evaluation design, its weighted moments over 17-row frame chunks.

    Against the same design taken in one chunk; tolerances as in
    ``_assert_weighted_moments``.
    """
    from superglm.inference import _metrics_design as metrics_design

    rng = np.random.default_rng(263)
    n = 150
    X = pd.DataFrame({"x": rng.uniform(-2.0, 2.0, n), "z": rng.normal(size=n)})
    y = np.sin(X["x"].to_numpy()) + 0.3 * X["z"].to_numpy() + rng.normal(scale=0.2, size=n)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": Spline(n_knots=6), "z": Numeric()},
    ).fit(X, y)
    design = metrics_design.EvaluationDesign(model, X, np.arange(len(model.result.beta)))
    dense = design.toarray()
    W = rng.uniform(0.5, 2.0, n)
    monkeypatch.setattr(metrics_design, "_MAX_DESIGN_CHUNK_ROWS", 17)
    assert design.chunk_rows == 17
    _assert_weighted_moments(design.weighted_moments(W), dense, W, chunks=-(-n // 17))


def test_projected_design_moments_sum_every_design_chunk(monkeypatch):
    """A grouped design's rows read in 13-row chunks, through the projected design's moments.

    The identity face keeps every row as the design gives it; against the dense
    design, tolerances as in ``_assert_weighted_moments``.
    """
    from superglm.inference import _metrics_design as metrics_design

    rng = np.random.default_rng(349)
    n = 120
    groups = [
        DenseGroupMatrix(rng.normal(size=(n, 2))),
        CategoricalGroupMatrix(rng.integers(-1, 3, n), 3),
        DiscretizedSSPGroupMatrix(
            rng.normal(size=(7, 4)), rng.normal(size=(4, 3)), rng.integers(0, 7, n)
        ),
    ]
    dm = DesignMatrix(groups, n, 8)
    dense = dm.toarray()
    W = rng.uniform(0.5, 2.0, n)
    monkeypatch.setattr(metrics_design, "_MAX_DESIGN_CHUNK_ROWS", 13)
    design = metrics_design.ProjectedDesign(dm, np.eye(dm.p))
    _assert_weighted_moments(design.weighted_moments(W), dense, W, chunks=-(-n // 13))


def test_piecewise_weighted_gram_sums_every_row_chunk(monkeypatch):
    """The hat Gram ``H'WH`` in 97-row chunks, against the dense hat matrix's product.

    Every term is nonnegative inside the knot span, so each entry of either
    route is within ``gamma_{n+3}`` relative of the exact sum.
    """
    from superglm.features import piecewise as piecewise_module

    rng = np.random.default_rng(661)
    x = rng.uniform(0.0, 10.0, 1_001)
    W = rng.uniform(0.5, 2.0, x.size)
    spec = piecewise_module.Piecewise(breaks=[2.0, 4.5, 7.0])
    spec.build(x)
    knots = spec._knots
    seg = spec._segment_index(x)
    frac = (x - knots[seg]) / (knots[seg + 1] - knots[seg])
    monkeypatch.setattr(piecewise_module, "_GRAM_CHUNK_ROWS", 97)
    actual = spec._weighted_gram(seg, frac, W)
    H = spec._hat_values(x)
    expected = H.T @ (W[:, None] * H)
    _assert_within(actual, expected, 2 * _gamma(x.size + 3) * expected)


def test_square_diagonal_fills_every_column_block():
    """``diag((H^-1 O)^2)`` from five-column blocks of unit columns, against the dense solve.

    Entry ``i`` is ``sum_v (H^-1 O)_vi (O H^-1)_vi``; each solve's forward error is
    ``O(n u kappa(H))`` of its column, so the two agree within
    ``(4 gamma_{3n} kappa + gamma_n) ||A_{:,i}|| ||A_{i,:}||``.
    """
    from superglm.solvers._structured.nested import _square_diagonal_by_columns

    rng = np.random.default_rng(1069)
    width = 23
    root = rng.normal(size=(width, width))
    H = root @ root.T + width * np.eye(width)
    other = rng.normal(size=(width, width))
    operator = other + other.T
    actual = _square_diagonal_by_columns(
        lambda b: np.linalg.solve(H, b), lambda b: operator @ b, width, block=5
    )
    A = np.linalg.solve(H, operator)
    bound = (4 * _gamma(3 * width) * np.linalg.cond(H) + _gamma(width)) * (
        np.linalg.norm(A, axis=0) * np.linalg.norm(A, axis=1)
    )
    _assert_within(actual, np.einsum("iv,vi->i", A, A), bound)
