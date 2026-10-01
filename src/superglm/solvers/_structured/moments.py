"""Compact sufficient-statistic assembly for structured systems."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
import scipy.sparse
from numpy.typing import NDArray

from superglm._group_matrix._group_matrix_algebra import _cross_gram
from superglm._group_matrix._group_matrix_centered import _compensated_add
from superglm._group_matrix._group_matrix_kernels import _dense_small_weighted_moments
from superglm.factor_smooth_geometry import adjoint_sum_to_zero_blocks
from superglm.group_matrix import (
    CategoricalGroupMatrix,
    DenseGroupMatrix,
    DiscretizedSSPGroupMatrix,
    FactorSmoothGroupMatrix,
    GroupMatrix,
    RandomEffectGroupMatrix,
)
from superglm.solvers._structured.border import BorderGenerators, reduce_generators
from superglm.solvers._structured.layout import (
    FactorSmoothLeafLayout,
    _validate_structured_inputs,
    build_factor_smooth_leaf_layout,
)
from superglm.solvers._structured.leaf_kernels import (
    _center_rows,
    _coded_rows,
    _segment_means,
    _shifted_sums,
    _sparse_rows,
)
from superglm.solvers._structured.nested import (
    IndicatorCells,
    NestedDataOperator,
    NestedLeafStatistics,
    NestedStructuredLayout,
    _divide_rows,
)
from superglm.solvers._structured.operators import (
    BlockSymmetricOperator,
    SumToZeroBlockOperator,
)
from superglm.solvers._structured.retired import module_getattr
from superglm.solvers._structured.selection import _parent_codes
from superglm.types import GroupSlice


@dataclass(frozen=True)
class FactorSmoothMomentSystem:
    """Raw FactorSmooth moments and working sums, never factored.

    The moment kernel an ``fs`` fit keeps for operators that only enter traces
    (the REML weight-derivative operators, one-engine design §3.4 and perf
    finding F5); its factor comes from the leaf system instead
    (``block_leaves.build_factor_smooth_leaf_system``).
    """

    operator: BlockSymmetricOperator
    xtw_small: NDArray
    xtw_structured: NDArray
    xtwz_small: NDArray
    xtwz_structured: NDArray
    sum_w: float
    sum_wz: float
    dominant_group_index: int
    dominant_group_name: str


@dataclass(frozen=True)
class NestedStructuredSystem:
    """Leaf-form data operator and working sufficient statistics of a nested chain.

    ``operator`` is the unaugmented data operator (``a = w``, deviation
    ``None``) that every signed operator of the fit is built about.  The
    structured vectors are in node order: parent entries are subtree sums of
    the leaf ones (§4), never a row pass.  ``dominant_group_name`` is the leaf,
    for reporting only.
    """

    operator: NestedDataOperator
    xtw_small: NDArray
    xtw_structured: NDArray
    xtwz_small: NDArray
    xtwz_structured: NDArray
    sum_w: float
    sum_wz: float
    chain_group_indices: tuple[int, ...]
    chain_group_names: tuple[str, ...]
    dominant_group_name: str
    # X_b' Wz in the factor's centred coordinates, sum_r (Wz)_r (x_r - c) with
    # c = operator.leaf.center, formed from the row pass's centred rows (None
    # without a right-hand side): the Newton solve's border right-hand side.
    xtwz_small_centred: NDArray | None = None


@dataclass(frozen=True)
class SumToZeroMomentSystem:
    """Raw all-level ``sz`` moments with public ``K - 1`` transpose products, never factored.

    The moment kernel an ``sz`` fit keeps for operators that only enter
    traces (the REML weight-derivative operators, as ``FactorSmoothMomentSystem``
    for ``fs``); its factor comes from the leaf system and the balance tree
    (``balance_tree.SumToZeroTreeFactor``).
    """

    operator: SumToZeroBlockOperator
    xtw_small: NDArray
    xtw_structured: NDArray
    xtwz_small: NDArray
    xtwz_structured: NDArray
    raw_xtw_structured: NDArray
    sum_w: float
    sum_wz: float
    dominant_group_index: int
    dominant_group_name: str


def _group_rows(matrix: GroupMatrix, rows: NDArray) -> NDArray:
    """``matrix[rows]`` dense; a discretized spline projects only the gathered basis rows.

    Its ``toarray`` projects the whole support table, which for a lossless
    support (one bin per distinct value) costs a table per chunk.
    """
    if isinstance(matrix, DiscretizedSSPGroupMatrix):
        return matrix.B_unique[matrix.bin_idx[rows]] @ matrix.R_inv
    return np.asarray(matrix.row_subset(rows).toarray(), dtype=np.float64)


def _leaf_rows(
    layout: NestedStructuredLayout | FactorSmoothLeafLayout,
    out: NDArray,
    lo: int,
    hi: int,
    center: NDArray,
) -> None:
    """Write the dense border rows ``X_b[leaf_order[lo:hi]] - c`` into ``out`` (``layout.leaf_rows``)."""
    rows = layout.leaf_rows
    _coded_rows(
        out,
        rows.codes[lo:hi],
        rows.tables,
        rows.table_start,
        rows.table_width,
        rows.table_column,
        rows.one_hot_width,
        rows.one_hot_column,
        center,
    )
    _uncoded_rows(layout, out, lo, hi, center)


def _uncoded_rows(
    layout: NestedStructuredLayout | FactorSmoothLeafLayout,
    out: NDArray,
    lo: int,
    hi: int,
    center: NDArray,
) -> None:
    """``_leaf_rows`` for the blocks without codes (sparse, gathered, generic); others untouched."""
    rows = layout.leaf_rows
    for column, matrix, basis in rows.sparse:
        _sparse_rows(out, column, matrix.indptr, matrix.indices, matrix.data, basis, lo, hi, center)
    order = layout.leaf_order[lo:hi]
    for column, values in rows.gathered:
        stop = column + values.shape[1]
        out[:, column:stop] = values[order] - center[column:stop]
    for column, matrix in rows.generic:
        block = _group_rows(matrix, order)
        stop = column + block.shape[1]
        out[:, column:stop] = block - center[column:stop]


def _indicator_rmatvec(layout: NestedStructuredLayout, values: NDArray) -> NDArray:
    """``X_b' values`` ``(q,)`` on the random-effect border blocks through their own
    transposes, densifying no rows; zero on every other column."""
    out = np.zeros(len(layout.small_indices))
    for cells in layout.sparse_indicators:
        out[cells.columns] = layout.small_matrices[cells.block].rmatvec(values)
    return out


def nested_prior_statistics(
    layout: NestedStructuredLayout,
    prior_weights: NDArray | None,
    *,
    chunk_size: int = 8192,
) -> tuple[NDArray, BorderGenerators | None]:
    """The border centre ``c0`` and the structural null generators for these prior weights.

    One-engine design §3.2: every border column that is not one-hot is
    centred on its shifted prior-weighted mean ``c0_j = x_ref,j + sum_r
    omega_r (x_rj - x_ref,j) / sum_r omega_r`` (``omega`` the prior weights,
    ``x_ref`` the first row in leaf order with ``omega > 0``), from the rows
    the row pass itself forms (``_leaf_rows``) in its fixed chunks and serial
    order (``_prior_center``).  The rule is chosen by type, never by the
    column's values: it is equivariant to a shift of the column to one
    rounding, and a column constant on its weighted rows centres to an exact
    zero on them.  One-hot columns (``CategoricalGroupMatrix`` blocks, random
    effects included) keep 0 by type, so their entries stay exact zeros and
    ones.  §3.6 step 1: the exact null generators of the border's data part
    (``_border_generators``).  ``None`` prior weights are unit weights.

    Cache contract (``NestedStructuredLayout.prior_cache``): owned by the
    layout and living as long as it; keyed by the prior weights themselves,
    held as a read-only copy beside the caller's array and matched by
    identity with that array or by exact equality with the copy, at most two
    entries.  The border matrices and codes never change, so nothing else
    invalidates an entry; no working weight, lambda or penalty enters one.
    """
    n = len(layout.leaf_order)
    weights = np.ones(n) if prior_weights is None else np.asarray(prior_weights, dtype=np.float64)
    if weights.shape != (n,):
        raise ValueError("prior_weights must match the design rows.")
    for source, held, center, generators in layout.prior_cache:
        if source is weights or np.array_equal(held, weights):
            return center, generators
    held = np.array(weights, dtype=np.float64, copy=True)
    held.setflags(write=False)
    center = _prior_center(layout, held, chunk_size)
    generators = _border_generators(layout, held)
    layout.prior_cache.insert(0, (weights, held, center, generators))
    del layout.prior_cache[2:]
    return center, generators


def _prior_center(layout: NestedStructuredLayout, weights: NDArray, chunk_size: int) -> NDArray:
    """The §3.2 centre ``(q,)`` of the border columns (``nested_prior_statistics``).

    A coded block (a discretized spline's support table) takes the shifted sum
    on its compact form, by type (design §3.2: "computed once per design from
    compact forms"; Li & Wood 2020, Algorithm 0): every row of bin ``b`` is
    the table row ``T_b``, so ``sum_r omega_r (x_r - x_ref) = sum_b Omega_b
    (T_b - T_ref)`` with ``Omega_b`` the bin's prior weight, ``x_ref = T_ref``
    the first weighted row's table row.  The differences are the row pass's
    own, so a column constant on the weighted rows still shifts by exact
    zeros and centres to ``x_ref``; the sum has ``m`` terms instead of ``n``,
    with ``Omega_b`` rounding at ``gamma_(n_b)`` (Higham 2002, Lemma 3.1), the
    same ``gamma_n sum omega |x - x_ref|`` order of bound.  It costs ``O(n)``
    for the bin weights and ``O(m k)`` per lambda rebuild of the table,
    where the row pass cost ``O(n q)``.  Every other dense block keeps the
    serial row pass in leaf order, column by column as before.
    """
    center = np.zeros(len(layout.small_indices))
    dense_columns = np.flatnonzero(~layout.indicator_columns)
    width = len(dense_columns)
    if width:
        omega = weights[layout.leaf_order]
        positive = np.flatnonzero(omega > 0.0)
        if not positive.size:
            raise ValueError("prior weights must contain a positive entry.")
        zero = np.zeros(width)
        reference = np.empty((1, width))
        first = int(positive[0])
        _leaf_rows(layout, reference, first, first + 1, zero)
        total = np.zeros(width)
        leaf_rows = layout.leaf_rows
        if leaf_rows.sparse or leaf_rows.gathered or leaf_rows.generic:
            # coded columns stay exact zeros here; their sums are replaced below
            buffer = np.zeros((min(chunk_size, len(omega)), width))
            for lo in range(0, len(omega), chunk_size):
                hi = min(lo + chunk_size, len(omega))
                rows = buffer[: hi - lo]
                _uncoded_rows(layout, rows, lo, hi, zero)
                _shifted_sums(rows, omega[lo:hi], reference[0], total)
        stops = np.append(leaf_rows.table_start, leaf_rows.tables.size)[1:]
        masses = _table_masses(layout, weights, omega, stops)
        for block, (start, stop) in enumerate(zip(leaf_rows.table_start, stops, strict=True)):
            column, block_width = leaf_rows.table_column[block], leaf_rows.table_width[block]
            table = leaf_rows.tables[start:stop].reshape(-1, block_width)
            reference_bin = leaf_rows.codes[first, block]
            shift = (table - table[reference_bin]) * masses[block][:, None]
            total[column : column + block_width] = np.sum(shift, axis=0)
        center[dense_columns] = reference[0] + total / float(np.sum(omega))
    offset = 0
    for matrix in layout.small_matrices:
        width = matrix.shape[1]
        if isinstance(matrix, CategoricalGroupMatrix):
            center[offset : offset + width] = 0.0
        offset += width
    center.setflags(write=False)
    return center


def _table_masses(
    layout: NestedStructuredLayout, weights: NDArray, omega: NDArray, stops: NDArray
) -> tuple[NDArray, ...]:
    """Each coded block's prior weight per bin, ``Omega_b = sum_{r: bin b} omega_r`` (``_prior_center``).

    Summed in leaf order (``np.bincount``, serial).  Cache contract: a slot of
    the lineage cache (``layout.lineage_cache``, shared by every lambda
    rebuild of the design, which passes the bin codes through unchanged),
    keyed by the leaf-ordered codes array itself (``LeafRows.codes``, a
    lineage entry) and the prior weights (held as a read-only copy, matched
    by identity or exact equality), at most two entries.  Bin codes and prior
    weights are fixed for a design, so nothing else invalidates an entry; no
    working weight, lambda, basis or penalty enters one.
    """
    rows = layout.leaf_rows
    slot = layout.lineage_cache.setdefault(("nested_slot", "prior_mass"), [])
    for codes, source, held, masses in slot:
        if codes is rows.codes and (source is weights or np.array_equal(held, weights)):
            return masses
    sizes = (stops - rows.table_start) // np.maximum(rows.table_width, 1)
    masses = tuple(
        np.bincount(rows.codes[:, block], weights=omega, minlength=int(size))
        for block, size in enumerate(sizes)
    )
    held = np.array(weights, dtype=np.float64, copy=True)
    held.setflags(write=False)
    slot.insert(0, (rows.codes, weights, held, masses))
    del slot[2:]
    return masses


def _border_generators(
    layout: NestedStructuredLayout | FactorSmoothLeafLayout, weights: NDArray
) -> BorderGenerators | None:
    """The exact null generators of the border's data part (design §3.6 step 1), by term type.

    Every ``RandomEffectGroupMatrix`` border block is complete (a training row
    with no level is refused at bind), so its block sum is the intercept
    column; the super-root has eliminated the intercept, so that sum is an
    exact null of the border's data part for any working weights.  Every
    pair of border random-effect blocks whose level codes nest (each child
    code meets a single parent code: the pattern, once per design, never a
    value) adds, per parent level, the parent indicator minus its children's.
    Both are exact nulls on every row of positive prior weight, restricted to
    the levels with positive prior exposure (a choice of basis for the same
    null space, like the reference level; see below).  ``reduce_generators``
    keeps an independent set, reduced to the identity on reference columns
    chosen by prior exposure.  ``None`` without a random-effect border block.
    """
    blocks, offset = [], 0
    for block, matrix in enumerate(layout.small_matrices):
        width = matrix.shape[1]
        if isinstance(matrix, RandomEffectGroupMatrix):
            blocks.append(
                (layout.local_groups[block].name, np.arange(offset, offset + width), matrix)
            )
        offset += width
    if not blocks:
        return None
    exposure = np.zeros(offset)
    for _, columns, matrix in blocks:
        exposure[columns] = np.bincount(matrix.codes, weights=weights, minlength=len(columns))
    # Every generator is supported on the levels with positive prior exposure:
    # a level without any is an exactly zero column on the weighted rows, a
    # data null by itself whose row of Q is exact zeros; left out of the
    # generators it stays decoupled, and its coefficient an exact zero.
    exposed = exposure > 0.0
    candidates, labels = [], []
    for name, columns, _ in blocks:
        vector = np.zeros(offset)
        vector[columns[exposed[columns]]] = 1.0
        if np.any(vector):
            candidates.append(vector)
            labels.append(f"{name}: the sum of its exposed levels")
    for child_name, child_columns, child in blocks:
        for parent_name, parent_columns, parent in blocks:
            if parent is child:
                continue
            codes = _parent_codes(child.codes, parent.codes, len(child_columns))
            if codes is None:
                continue
            for level in np.flatnonzero(exposed[parent_columns]):
                vector = np.zeros(offset)
                vector[parent_columns[level]] = 1.0
                children = child_columns[codes == level]
                vector[children[exposed[children]]] = -1.0
                candidates.append(vector)
                labels.append(f"{parent_name}[{level}] minus its {child_name} levels")
    if not candidates:
        return None
    return reduce_generators(np.column_stack(candidates), exposure, tuple(labels))


def _centered_leaf_pass(
    layout: NestedStructuredLayout,
    weights: NDArray,
    leaf_magnitude: NDArray,
    mean: NDArray | None,
    chunk_size: int,
    border_center: NDArray,
    rhs: NDArray | None = None,
    error: NDArray | None = None,
) -> tuple[NDArray, NDArray, NDArray, NDArray | None, NDArray | None, NDArray]:
    """Return leaf means, the centred within-leaf scatter, the absolute mass, the deviations,
    with ``rhs`` ``sum_r rhs_r (x_r - c)``, and ``sum_r a_r (x_r - c)``, both on the dense columns.

    The exact centred row pass of §3.4 and §3.6 (decision 1) in ONE pass over
    the border rows: rows are taken in leaf order (``layout.leaf_order``)
    in chunks of ``chunk_size`` rows, each chunk written once into a reused
    buffer from the blocks' own arrays in leaf order (``layout.leaf_rows``)
    and centred on the border centre ``c`` (``border_center``), so the pass holds a few
    ``chunk_size x q`` buffers whatever the largest leaf.  Every row sum runs
    in the compiled serial kernels of ``leaf_kernels``, in leaf order.

    Without ``mean`` (the data pass) it forms, for any sign pattern of the
    rows (signed-rows note §4.1, one-engine design §3.3), the leaf centres on
    the ABSOLUTE weights, ``mu_l = x_ref - c + sum_r |a_r| ((x_r - c) - (x_ref
    - c)) / sum_r |a_r|`` about each leaf's first weighted row (0 where the
    leaf has no weighted row; ``leaf_magnitude`` holds ``sum_{r in l} |a_r|``),
    so a column constant on a leaf's weighted rows gives that constant
    exactly, and the carried deviation about that centre,
    ``delta_l = sum_r a_r (x_r - mu_l) = -2 sum_{r: a_r < 0} |a_r| (x_r -
    mu_l)``, whose second form is an exact zero (an empty sum) for a leaf
    without a negative row: rows ``a >= 0`` reproduce the non-negative
    construction bit for bit, and the deviation is returned as ``None`` when
    every entry is an exact zero.  A leaf cut by a chunk edge is centred piece
    by piece on each piece's own centre and its pieces are combined after the
    pass (``_combine_leaf_pieces``).  With ``mean`` (a signed operator about
    its factor's data centres) it forms ``dev_l = sum_r a_r (x_r - m_l)``,
    summed over a cut leaf's pieces.  The scatter ``sum_r a_r (x_r - m_l)(x_r
    - m_l)'`` is ``P'P - N'N`` per chunk (``_root_scatter``), from the centred
    rows of positive and negative weight, and never subtracts raw moments, so
    such a column has an exactly zero row and column; chunks accumulate with
    compensated addition and the result is symmetrized.  The absolute mass is
    ``sum_r e_r (x_rj - m_lj)^2`` with ``e_r`` the scale of row ``r``'s weight
    error (``error``, in row order; ``|a|`` when not given): the caller that
    builds the rows supplies it, ``w`` for Fisher rows and ``w0 (|u^2/V| +
    |(y - mu) factor|)`` for observed rows (design §3.3), chosen by the type of
    rows, never by their values.

    The layout's random-effect border blocks (``sparse_indicators``, by type)
    are never materialized.  A row's centred entry of a one-hot column is exactly zero
    on every level its leaf's rows never take, so each chunk forms those
    blocks as sparse rows on the levels of each row's leaf
    (``_indicator_entries``), their segment means in the same shifted form on
    cells (``_indicator_means``); every product, sum and piece update then
    reads the same nonzero terms as the dense pass.  The pass works on the
    columns ``[dense | indicator]`` and returns the layout's order; a given
    ``mean`` must vanish off each leaf's levels, as a data pass's means do.
    """
    n = len(weights)
    order, starts = layout.leaf_order, layout.leaf_starts
    present = np.flatnonzero(np.diff(starts))
    edges = np.append(starts[present], n)
    indicators, sparse = layout.sparse_indicators, layout.indicator_columns
    # the pass's own column order: the dense columns, then the sparse indicator blocks
    columns = np.concatenate((np.flatnonzero(~sparse), np.flatnonzero(sparse)))
    q, width, center = len(columns), int(np.count_nonzero(~sparse)), border_center[~sparse]
    data_pass = mean is None
    given_mean = None
    if mean is None:
        mean = np.zeros((len(leaf_magnitude), q))
        deviation = np.zeros_like(mean)
    else:
        given_mean = np.asarray(mean, dtype=np.float64)
        mean = given_mean[:, columns]
        deviation = np.zeros_like(mean)
    within = np.zeros((q, q))
    compensation = np.zeros_like(within)
    absolute = np.zeros(q)
    ordered = weights[order]
    # the caller's error scale in leaf order; without one each row is charged |a|
    error_ordered = None if error is None else np.asarray(error, np.float64)[order]
    rows_buffer = np.empty((min(chunk_size, n), width))
    split_buffer = np.empty_like(rows_buffer)
    # a x for the indicator blocks' cross products, which only they read
    weighted_buffer = np.empty_like(rows_buffer) if q > width else np.zeros((0, width))
    no_rows = np.zeros((0, width))
    pieces: list[tuple] = []
    rhs_sums = None if rhs is None else np.zeros(width)
    rhs_ordered = None if rhs is None else np.asarray(rhs, dtype=np.float64)[order]
    # X_b'a on the dense columns in the factor's centred coordinates, as rhs_sums
    weight_sums = np.zeros(width)
    no_shift = np.zeros(width)
    for lo in range(0, n, chunk_size):
        hi = min(lo + chunk_size, n)
        # the present leaves with rows in [lo, hi) and their segments of the chunk
        span = slice(np.searchsorted(edges, lo, side="right") - 1, np.searchsorted(edges, hi))
        leaves, begin, end = present[span], edges[span], edges[span.start + 1 : span.stop + 1]
        segment = np.maximum(begin, lo) - lo
        counts = np.minimum(end, hi) - lo - segment
        cut = (begin < lo) | (end > hi)
        bounds = np.append(segment, hi - lo)
        centered, split = rows_buffer[: hi - lo], split_buffer[: hi - lo]
        weighted = weighted_buffer[: hi - lo]
        _leaf_rows(layout, centered, lo, hi, center)
        if rhs_sums is not None and rhs_ordered is not None:
            # the border's normal-equations right-hand side in the factor's
            # centred coordinates, from these rows before any other centring
            _shifted_sums(centered, rhs_ordered[lo:hi], no_shift, rhs_sums)
        a = ordered[lo:hi]
        _shifted_sums(centered, a, no_shift, weight_sums)
        magnitude = np.abs(a)
        error_chunk = magnitude if error_ordered is None else error_ordered[lo:hi]
        # exact-zero skip: a chunk without a negative row adds nothing to delta
        signed_chunk = data_pass and bool(np.any(a < 0.0))
        if data_pass:
            # each segment's first weighted row (any row of a segment without one)
            nonzero = np.append(np.flatnonzero(a), len(a))
            first_weighted = nonzero[np.searchsorted(nonzero, segment)]
            reference = np.minimum(first_weighted, segment + counts - 1)
            # a cut leaf's piece is centred on the |a|-mean of its own rows
            total = np.where(cut, np.add.reduceat(magnitude, segment), leaf_magnitude[leaves])
            means = np.empty((len(leaves), width))
            _segment_means(centered, magnitude, bounds, reference, total, means)
            mean[leaves, :width] = means
            _indicator_means(
                indicators, mean, width, lo, magnitude, leaves, segment, counts, reference, total
            )
        else:
            means = mean[leaves, :width]
        sums = no_rows if data_pass else np.zeros((len(leaves), width))
        negative = np.zeros((len(leaves), width)) if signed_chunk else no_rows
        positive, negatives = _center_rows(
            centered, weighted, split, a, error_chunk, bounds, means, absolute, sums, negative
        )
        dense = _root_scatter(split[:positive], split[len(split) - negatives :])
        entries = _indicator_entries(indicators, mean, width, lo, leaves, counts)
        row, column, value = entries
        _compensated_add(
            within, compensation, _chunk_scatter(dense, weighted, entries, a, q - width)
        )
        absolute[width:] += np.bincount(
            column, weights=error_chunk[row] * value**2, minlength=q - width
        )
        owner = np.repeat(np.arange(len(leaves)), counts)
        if not data_pass:
            deviation[leaves, :width] += sums
            deviation[leaves, width:] += _segment_sums(entries, a, owner, len(leaves), q - width)
            continue
        # the carried deviation of every segment about its own centre, -2 sum
        # over its negative rows of |a| (x - mu)
        piece_deviation = np.zeros((len(leaves), q))
        negative_mass = np.zeros(len(leaves))
        if signed_chunk:
            negative_weight = np.maximum(-a, 0.0)
            piece_deviation[:, :width] = -2.0 * negative
            piece_deviation[:, width:] = -2.0 * _segment_sums(
                entries, negative_weight, owner, len(leaves), q - width
            )
            negative_mass = np.add.reduceat(negative_weight, segment)
            deviation[leaves[~cut]] = piece_deviation[~cut]
        if not np.any(cut):
            continue
        # a cut segment (the chunk's first or last) keeps its signed and absolute
        # weights W_p, A_p, its centre mu_p, E_p = sum e, t_p = sum e (x - mu_p),
        # its deviation and negative mass for the combination
        cut_index = np.flatnonzero(cut)
        residual_cut = np.zeros((len(cut_index), q))
        for position, index in enumerate(cut_index):
            rows = slice(segment[index], segment[index] + counts[index])
            residual_cut[position, :width] = error_chunk[rows] @ centered[rows]
        residual_cut[:, width:] = _segment_sums(
            entries, error_chunk, owner, len(leaves), q - width
        )[cut]
        pieces.append(
            (
                leaves[cut],
                np.add.reduceat(a, segment)[cut],
                total[cut],
                mean[leaves[cut]],
                np.add.reduceat(error_chunk, segment)[cut],
                residual_cut,
                piece_deviation[cut],
                negative_mass[cut],
            )
        )
    if pieces:
        _combine_leaf_pieces(
            pieces, leaf_magnitude, mean, within, compensation, absolute, deviation
        )
    layout_order = np.argsort(columns)
    return (
        # a signed pass only reads the means it was given, so they come back as
        # given: the operator then shares its factor's frozen means (bitwise
        # equal by construction) instead of keeping a (K, q) copy of them
        mean[:, layout_order] if given_mean is None else given_mean,
        0.5 * (within + within.T)[np.ix_(layout_order, layout_order)],
        absolute[layout_order],
        None if data_pass and not np.any(deviation) else deviation[:, layout_order],
        rhs_sums,
        weight_sums,
    )


def _root_scatter(positive: NDArray, negative: NDArray) -> NDArray:
    """``P'P - N'N`` ``(q, q)`` by BLAS ``syrk``, its upper triangle mirrored.

    ``P`` and ``N`` hold ``sqrt(|a_r|) (x_r - m)`` for a chunk's rows of positive
    and negative weight (``leaf_kernels._center_rows``).  Each product
    ``fl(sqrt|a|) x_i fl(sqrt|a|) x_j`` is ``|a| x_i x_j (1 + theta_4)``, so the
    entry is within ``gamma_(m+5) sum_r |a_r| |x_ri| |x_rj|`` of the exact
    scatter (Higham 2002, Lemma 3.1 and eq. 3.4, one more rounding for the
    difference): the form of the ``X' diag(a) X`` product it replaces, three
    roundings more per term, at half its flops.  A column that is exactly zero
    on every weighted row keeps an exactly zero row and column.  With no dense
    column (a border of random-effect blocks alone) the block is empty and no
    BLAS call is made: ``syrk`` rejects a zero leading dimension.
    """
    if not positive.shape[1]:
        return np.zeros((0, 0))
    syrk = scipy.linalg.get_blas_funcs("syrk", (positive,))
    upper = syrk(1.0, positive.T) - syrk(1.0, negative.T)
    return upper + np.triu(upper, 1).T


def _chunk_scatter(
    dense: NDArray,
    weighted: NDArray,
    entries: tuple[NDArray, NDArray, NDArray],
    a: NDArray,
    width: int,
) -> NDArray:
    """A chunk's ``sum_r a_r x_r x_r'`` on ``[dense | indicator]``, the indicator part from its entries.

    ``dense`` is the dense columns' block (``_root_scatter``) and ``weighted``
    their rows times ``a``, which the cross products read.  ``width`` is the
    number of indicator columns; without any the dense block is the whole
    product, never paying scipy's per-call overhead.
    """
    if not width:
        return dense
    row, column, value = entries
    indicator = scipy.sparse.csr_array((value, (row, column)), shape=(len(a), width))
    weighted_indicator = scipy.sparse.csr_array((a[row] * value, (row, column)), indicator.shape)
    cross = indicator.T @ weighted
    return np.block([[dense, cross.T], [cross, (indicator.T @ weighted_indicator).toarray()]])


def _indicator_means(
    indicators: tuple[IndicatorCells, ...],
    mean: NDArray,
    width: int,
    lo: int,
    a: NDArray,
    leaves: NDArray,
    segment: NDArray,
    counts: NDArray,
    reference: NDArray,
    total: NDArray,
) -> None:
    """Write a chunk's segment means of the sparse indicator columns into ``mean[:, width:]``.

    The shifted form of the dense columns, on the segment's cells: about its
    reference row, whose level carries ``1 - sum_{r off that level} a_r / W``
    and every other level ``sum_{r on it} a_r / W`` (0 where ``W == 0``), so a
    level constant on the segment's weighted rows gives exactly 1 or 0.  The
    chunk's cells are those of its consecutive leaves, one contiguous run.
    """
    offset = width
    for cells in indicators:
        own = cells.row_cell[lo : lo + len(a)]
        first, stop = cells.start[leaves[0]], cells.start[leaves[-1] + 1]
        cell = np.arange(first, stop)
        owner = np.repeat(np.arange(len(leaves)), cells.start[leaves + 1] - cells.start[leaves])
        taken = own >= 0
        level_sum = np.bincount(own[taken] - first, weights=a[taken], minlength=stop - first)
        reference_cell = own[reference]
        off_reference = np.add.reduceat(a * (own != np.repeat(reference_cell, counts)), segment)
        on_reference = cell == reference_cell[owner]
        shift = np.where(on_reference, -off_reference[owner], level_sum)
        weight = total[owner]
        ratio = np.divide(shift, weight, out=np.zeros(len(cell)), where=weight != 0.0)
        mean[leaves[owner], offset + cells.level[cell]] = np.where(
            weight != 0.0, on_reference + ratio, 0.0
        )
        offset += len(cells.columns)


def _indicator_entries(
    indicators: tuple[IndicatorCells, ...],
    mean: NDArray,
    width: int,
    lo: int,
    leaves: NDArray,
    counts: NDArray,
) -> tuple[NDArray, NDArray, NDArray]:
    """A chunk's centred indicator rows ``x - m`` on the levels of each row's leaf.

    Returns ``(row, column, value)``: chunk row, column among the indicator
    columns and ``x_rj - mean[l, width + column]``.  Every entry left out is an
    exact zero of the dense pass: the row does not take the level and its
    leaf's mean there is 0.
    """
    parts = [(np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp), np.empty(0))]
    leaf, offset = np.repeat(leaves, counts), 0
    for cells in indicators:
        per_row = cells.start[leaf + 1] - cells.start[leaf]
        row = np.repeat(np.arange(len(leaf)), per_row)
        first = cells.start[leaf] - np.cumsum(per_row) + per_row
        cell = np.repeat(first, per_row) + np.arange(len(row))
        column = offset + cells.level[cell]
        value = (cells.row_cell[lo + row] == cell) - mean[leaf[row], width + column]
        parts.append((row, column, value))
        offset += len(cells.columns)
    row, column, value = (np.concatenate(values) for values in zip(*parts, strict=True))
    return row, column, value


def _segment_sums(
    entries: tuple[NDArray, NDArray, NDArray],
    weights: NDArray,
    owner: NDArray,
    segments: int,
    width: int,
) -> NDArray:
    """``sum_{r in s} weights_r v_rj`` ``(segments, width)`` over sparse row entries ``v``."""
    row, column, value = entries
    key = owner[row] * width + column
    sums = np.bincount(key, weights=weights[row] * value, minlength=segments * width)
    return sums.reshape(segments, width)


def _combine_leaf_pieces(
    pieces: list[tuple],
    leaf_magnitude: NDArray,
    mean: NDArray,
    within: NDArray,
    compensation: NDArray,
    absolute: NDArray,
    deviation: NDArray,
) -> None:
    """Combine the pieces of every leaf a chunk edge cut, in place (data pass).

    Each piece ``p`` of leaf ``l`` was centred on its own ``|a|``-centre
    ``mu_p`` (0 when its absolute weight ``A_p`` is 0), added ``sum a (x -
    mu_p)(x - mu_p)'`` to ``within`` and ``sum e (x - mu_p)^2`` to ``absolute``,
    and records its signed weight ``W_p``, ``A_p``, ``mu_p``, ``E_p = sum e``,
    ``t_p = sum e (x - mu_p)``, its deviation ``delta_p = sum a (x - mu_p)`` and
    its negative mass ``N_p = sum_{a < 0} |a|``.  The leaf centre is the
    ``A``-weighted mean of the piece centres shifted about the heaviest piece,
    ``mu_l = mu_h + sum_p A_p (mu_p - mu_h) / A_l``, so a column constant on the
    leaf's weighted rows keeps that constant and a zero scatter row.  With
    ``d_p = mu_p - mu_l`` the exact identities (Chan, Golub & LeVeque 1979, for
    any centres; signed-rows note §4.1)

        sum_{r in l} a (x - mu_l)(x - mu_l)' = sum_p [W_p-scatter + delta_p d_p' + d_p delta_p' + W_p d_p d_p'],
        sum_{r in l} a (x - mu_l) = sum_p [delta_p + W_p d_p] = sum_p [delta_p - 2 N_p d_p],
        sum_{r in l} e (x - mu_l)^2 = sum_p [U_p + 2 d_p t_p + E_p d_p^2]

    update the scatter, the deviation (its second form uses ``sum_p A_p d_p =
    0``, an exact zero for pieces without negative rows) and the absolute mass.
    """
    leaf, weight, total, centre, error, error_residual, piece_deviation, negative = (
        np.concatenate(values) for values in zip(*pieces, strict=True)
    )
    new_leaf = np.diff(leaf, prepend=-1) != 0
    first, owner = np.flatnonzero(new_leaf), np.cumsum(new_leaf) - 1
    reference = centre[np.lexsort((-total, leaf))[first]]
    shift = np.add.reduceat(total[:, None] * (centre - reference[owner]), first, axis=0)
    leaf_total = leaf_magnitude[leaf[first]]
    combined = np.where(
        leaf_total[:, None] != 0.0, reference + _divide_rows(shift, leaf_total), 0.0
    )
    mean[leaf[first]] = combined
    delta = centre - combined[owner]
    update = delta.T @ (weight[:, None] * delta)
    if np.any(piece_deviation) or np.any(negative):
        cross = piece_deviation.T @ delta
        update = update + cross + cross.T
        deviation[leaf[first]] = np.add.reduceat(
            piece_deviation - 2.0 * negative[:, None] * delta, first, axis=0
        )
    _compensated_add(within, compensation, update)
    absolute += np.sum(2.0 * delta * error_residual + error[:, None] * delta**2, axis=0)


def _nested_pass(
    layout: NestedStructuredLayout,
    group_matrices: list[GroupMatrix],
    weights: NDArray,
    *,
    mean: NDArray | None = None,
    chunk_size: int = 8192,
    prior_weights: NDArray | None = None,
    center: NDArray | None = None,
    rhs: NDArray | None = None,
    error: NDArray | None = None,
) -> tuple[NestedLeafStatistics, NDArray | None, NDArray]:
    """One row pass of a nested chain: per-leaf statistics of the row weights.

    ``weight`` comes from the leaf kernel; ``mean``, ``within``, ``absolute``
    and ``deviation`` from ``_centered_leaf_pass`` about the border centre.
    ``mean=None`` is the data pass (``|w|``-centred data means and their
    carried deviation, design §3.3) about the centre of ``prior_weights``
    (``nested_prior_statistics``; the design's prior weights, ``None`` for
    unit weights), and it carries the border's structural null generators for
    the factor.  A given ``mean`` is a signed pass about those means and needs
    the ``center`` of the factor it belongs to (``data_operator.leaf.center``);
    it carries no generators.  ``error`` ``(n,)`` is the per-row scale of the
    weights' rounding (``|weights|`` when not given; the caller passes it by
    the type of its rows, design §3.3).  With ``rhs`` (the working right-hand
    side ``Wz``) it also returns ``sum_r rhs_r (x_r - c)`` on the border's
    dense columns (those outside the sparse random-effect blocks, in layout
    order), from the pass's own centred rows: the sum rounds at ``|x - c|``
    rather than at a column's offset.  It always returns ``sum_r weights_r
    (x_r - c)`` on the same columns, from the same rows.

    The running error bound's inputs (signed-rows note §4.4): ``error_mass``
    ``sum_{r in l} e_r`` per leaf and the two rounding constants of the pass,
    ``gamma_weight = (n_leaf + 16) eps`` for the sequential leaf sums of
    ``omega_l`` and ``delta_l`` (``n_leaf`` the largest leaf's rows; 16 the
    roundings that form one observed row's weight) and ``gamma_scatter =
    (n_chunk + 21) eps`` for the within scatter, a ``syrk`` over at most
    ``n_chunk`` rows (Higham 2002, Lemma 3.1 and eq. 3.4, plus the row's own
    roundings).
    """
    leaf = group_matrices[layout.leaf_group_index]
    if not isinstance(leaf, RandomEffectGroupMatrix):
        raise ValueError("The nested leaf group must be a RandomEffectGroupMatrix.")
    if (mean is None) != (center is None):
        raise ValueError(
            "A signed nested pass takes its factor's leaf means and centre together; "
            "a data pass takes neither."
        )
    generators = None
    if center is None:
        center, generators = nested_prior_statistics(layout, prior_weights, chunk_size=chunk_size)
    values = np.asarray(weights, dtype=np.float64)
    leaf_weight = leaf.rmatvec(values)
    leaf_magnitude = leaf.rmatvec(np.abs(values))
    error_mass = leaf_magnitude
    if error is not None:
        error = np.asarray(error, dtype=np.float64)
        if error.shape != values.shape or not np.all(error >= np.abs(values)):
            raise ValueError("The row error scale must match the weights and bound |weights|.")
        error_mass = leaf.rmatvec(error)
    leaf_mean, within, absolute, deviation, rhs_dense, weight_dense = _centered_leaf_pass(
        layout,
        values,
        leaf_magnitude,
        mean,
        chunk_size,
        np.asarray(center, dtype=np.float64),
        rhs,
        error,
    )
    eps = float(np.finfo(np.float64).eps)
    n = len(values)
    largest_leaf = int(np.max(np.diff(layout.leaf_starts), initial=0))
    statistics = NestedLeafStatistics(
        weight=leaf_weight,
        mean=leaf_mean,
        within=within,
        absolute=absolute,
        center=center,
        deviation=deviation,
        indicator=layout.indicator_columns,
        generators=generators,
        error_mass=error_mass,
        rounding=((largest_leaf + 16) * eps, (min(n, chunk_size) + 21) * eps),
        indicator_pattern=layout.indicator_pattern,
    )
    return statistics, rhs_dense, weight_dense


def build_nested_structured_system(
    group_matrices: list[GroupMatrix],
    groups: list[GroupSlice],
    W: NDArray,
    Wz: NDArray | None,
    *,
    layout: NestedStructuredLayout,
    mean: NDArray | None = None,
    prior_weights: NDArray | None = None,
    center: NDArray | None = None,
    error: NDArray | None = None,
) -> NestedStructuredSystem:
    """Build the nested data operator and sufficient statistics from one set of rows.

    ``mean=None`` is the data system of a PIRLS iterate, centred by the fit's
    ``prior_weights`` (``nested_prior_statistics``); a signed W-derivative
    operator passes its factor's ``data_operator.leaf.mean`` and ``.center`` (§6) and
    ``Wz=None``: it has no right-hand side, so ``X'Wz`` is the exact zero,
    formed without a row pass.  ``error`` is the rows' weight-error scale
    (``_nested_pass``): the observed geometry passes its observed rows' scale,
    every other caller's rows are charged at ``|W|``.  Only the leaf group touches rows: one centred
    leaf pass, whose centred sums also give ``X_b'w`` and ``X_b'Wz`` on the
    dense border columns (``sum a (x - c) + c sum a``; a random-effect block
    through its own transpose).  Every parent level is a subtree sum; the
    border Gram ``X_b'WX_b`` is never formed (decision 1).
    """
    weights, weighted_rhs, leaf = _validate_structured_inputs(
        group_matrices,
        groups,
        W,
        np.zeros_like(W) if Wz is None else Wz,
        layout.leaf_group_index,
    )
    if any(
        matrix is not group_matrices[index]
        for matrix, index in zip(layout.small_matrices, layout.small_group_indices, strict=True)
    ) or layout.chain_group_names != tuple(groups[i].name for i in layout.chain_group_indices):
        raise ValueError("Nested layout does not match the supplied grouped design.")
    leaf_statistics, xtwz_dense, xtw_dense = _nested_pass(
        layout,
        group_matrices,
        weights,
        mean=mean,
        prior_weights=prior_weights,
        center=center,
        rhs=None if Wz is None else weighted_rhs,
        error=error,
    )
    operator = NestedDataOperator(
        tree=layout.tree,
        leaf=leaf_statistics,
        small_indices=layout.small_indices,
        structured_indices=layout.structured_indices,
    )
    # X_b'a from the pass's centred sums, X_b'a = sum a (x - c) + c sum a on the
    # dense columns (the fs leaf system's convention, block_leaves._assemble_system);
    # a random-effect block (c = 0) through its own transpose
    dense = ~layout.indicator_columns
    border_center = leaf_statistics.center[dense]
    sum_w, sum_wz = float(np.sum(weights)), float(np.sum(weighted_rhs))
    xtw_small = _indicator_rmatvec(layout, weights)
    xtw_small[dense] = xtw_dense + sum_w * border_center
    xtwz_small = np.zeros(len(layout.small_indices))
    xtwz_small_centred = None
    if xtwz_dense is not None:
        xtwz_small = _indicator_rmatvec(layout, weighted_rhs)
        # the factor's centred coordinates: the random-effect blocks keep c = 0
        xtwz_small_centred = xtwz_small.copy()
        xtwz_small_centred[dense] = xtwz_dense
        xtwz_small[dense] = xtwz_dense + sum_wz * border_center
    return NestedStructuredSystem(
        operator=operator,
        xtw_small=xtw_small,
        xtw_structured=np.concatenate(layout.tree.subtree_sum(leaf_statistics.weight)),
        xtwz_small=xtwz_small,
        xtwz_small_centred=xtwz_small_centred,
        xtwz_structured=(
            np.zeros(layout.structured_indices.size)
            if Wz is None
            else np.concatenate(layout.tree.subtree_sum(leaf.rmatvec(weighted_rhs)))
        ),
        sum_w=sum_w,
        sum_wz=sum_wz,
        chain_group_indices=layout.chain_group_indices,
        chain_group_names=layout.chain_group_names,
        dominant_group_name=layout.leaf_group_name,
    )


def _optimized_discrete_factor_smooth_cross(
    dominant: FactorSmoothGroupMatrix,
    matrix: GroupMatrix,
    weights: NDArray,
    cell_weights: NDArray | None,
) -> NDArray | None:
    """Use compact cell crosses when the small matrix has eligible geometry."""
    if not dominant.is_discrete:
        return None
    if type(matrix) is DenseGroupMatrix:
        return dominant.factor_smooth_discrete_dense_cell_cross_gram(weights, matrix.M)
    if cell_weights is None:  # pragma: no cover - structured assembly invariant
        raise RuntimeError("discrete FactorSmooth cell weights are unavailable")
    return dominant.factor_smooth_discrete_shared_bin_cross_gram(cell_weights, matrix)


def build_block_structured_system(
    group_matrices: list[GroupMatrix],
    groups: list[GroupSlice],
    W: NDArray,
    Wz: NDArray,
    *,
    dominant_group_index: int,
    layout: FactorSmoothLeafLayout | None = None,
) -> FactorSmoothMomentSystem | SumToZeroMomentSystem:
    """Build exact FactorSmooth moments without a full coefficient Gram matrix.

    The moment kernel of the REML weight-derivative operators, which only
    enter traces (``FactorSmoothMomentSystem``); ``layout`` is the term's leaf
    layout (built here for ``None``).
    """
    if len(group_matrices) != len(groups):
        raise ValueError("group_matrices and groups must have the same length.")
    if not 0 <= dominant_group_index < len(group_matrices):
        raise IndexError("dominant_group_index is outside group_matrices.")
    dominant = group_matrices[dominant_group_index]
    if not isinstance(dominant, FactorSmoothGroupMatrix):
        raise ValueError("The dominant block group must be a FactorSmoothGroupMatrix.")
    weights = np.asarray(W, dtype=np.float64)
    weighted_rhs = np.asarray(Wz, dtype=np.float64)
    if weights.ndim != 1 or weighted_rhs.shape != weights.shape:
        raise ValueError("W and Wz must be one-dimensional arrays with identical shape.")
    if len(weights) != dominant.shape[0] or any(
        matrix.shape[0] != len(weights) for matrix in group_matrices
    ):
        raise ValueError("All group matrices, W, and Wz must have the same row count.")

    if layout is None:
        layout = build_factor_smooth_leaf_layout(
            group_matrices,
            groups,
            dominant_group_index=dominant_group_index,
        )
    if (
        layout.dominant_group_index != dominant_group_index
        or layout.dominant_group_name != groups[dominant_group_index].name
        or len(layout.small_matrices) != len(group_matrices) - 1
        or any(
            matrix is not group_matrices[index]
            for matrix, index in zip(
                layout.small_matrices,
                layout.small_group_indices,
                strict=True,
            )
        )
    ):
        raise ValueError("Structured block layout does not match the grouped design.")

    cell_weights = None
    if dominant.is_discrete:
        (
            cell_weights,
            D,
            raw_xtw_structured,
            raw_xtwz_structured,
        ) = dominant.factor_smooth_discrete_cell_moments(
            weights,
            weighted_rhs,
        )
    else:
        (
            D,
            raw_xtw_structured,
            raw_xtwz_structured,
        ) = dominant.factor_smooth_sufficient_stats(
            weights,
            weighted_rhs,
        )

    if len(layout.small_indices):
        if layout.dense_small_matrix is not None:
            A, xtw_small, xtwz_small = _dense_small_weighted_moments(
                layout.dense_small_matrix,
                weights,
                weighted_rhs,
            )
            if dominant.is_discrete:
                C = dominant.factor_smooth_discrete_dense_cell_cross_gram(
                    weights,
                    layout.dense_small_matrix,
                )
            else:
                C = dominant.factor_smooth_dense_cross_gram(
                    weights,
                    layout.dense_small_matrix,
                )
        else:
            if layout.small_execution_plan is None:  # pragma: no cover - layout invariant
                raise RuntimeError("Structured small block has no execution plan.")
            small_moments = layout.small_execution_plan._moments_prevalidated(
                weights,
                rhs=(weighted_rhs,),
                include_xtw=True,
                # the weight-derivative operators this kernel serves are signed
                # by type (one-engine design §3.3): no row's sign picks the path
                signed=True,
            )
            if small_moments.xtw is None:  # pragma: no cover - requested above
                raise RuntimeError("Structured small moment plan omitted X'W.")
            A = small_moments.gram
            xtw_small = small_moments.xtw
            xtwz_small = small_moments.xt_rhs[0]
            cross_blocks = []
            for matrix in layout.small_matrices:
                optimized_cross = _optimized_discrete_factor_smooth_cross(
                    dominant,
                    matrix,
                    weights,
                    cell_weights,
                )
                if optimized_cross is not None:
                    cross_blocks.append(optimized_cross)
                    continue
                if dominant.factor_basis == "sz":
                    raw_cross = np.empty(
                        (
                            dominant.n_levels,
                            dominant.block_size,
                            matrix.shape[1],
                        ),
                        dtype=np.float64,
                    )
                    unit = np.zeros(matrix.shape[1], dtype=np.float64)
                    for column in range(matrix.shape[1]):
                        unit[column] = 1.0
                        rows = matrix.matvec(unit)
                        raw_cross[:, :, column] = dominant.factor_smooth_dense_cross_gram(
                            weights,
                            rows[:, None],
                        )[:, :, 0]
                        unit[column] = 0.0
                    cross_blocks.append(raw_cross)
                else:
                    cross_blocks.append(
                        _cross_gram(dominant, matrix, weights).reshape(
                            dominant.n_levels,
                            dominant.block_size,
                            matrix.shape[1],
                        )
                    )
            C = np.concatenate(cross_blocks, axis=2)
    else:
        A = np.empty((0, 0), dtype=np.float64)
        C = np.empty(
            (dominant.n_levels, dominant.block_size, 0),
            dtype=np.float64,
        )
        xtw_small = np.empty(0, dtype=np.float64)
        xtwz_small = np.empty(0, dtype=np.float64)

    if dominant.factor_basis == "sz":
        # Tabmat/BLAS assembly is mathematically symmetric but may leave
        # opposite triangles a few ulps apart.  Canonicalize at the moment
        # boundary before the constrained factor's strict symmetry check.
        A = 0.5 * (A + A.T)
        xtw_structured = adjoint_sum_to_zero_blocks(raw_xtw_structured)
        xtwz_structured = adjoint_sum_to_zero_blocks(raw_xtwz_structured)
        operator = SumToZeroBlockOperator(
            A=A,
            C=C,
            D=D,
            small_indices=layout.small_indices,
            structured_indices=layout.structured_indices,
        )
        return SumToZeroMomentSystem(
            operator=operator,
            xtw_small=xtw_small,
            xtw_structured=xtw_structured,
            xtwz_small=xtwz_small,
            xtwz_structured=xtwz_structured,
            raw_xtw_structured=raw_xtw_structured,
            sum_w=float(np.sum(weights)),
            sum_wz=float(np.sum(weighted_rhs)),
            dominant_group_index=dominant_group_index,
            dominant_group_name=layout.dominant_group_name,
        )
    xtw_structured = raw_xtw_structured
    xtwz_structured = raw_xtwz_structured
    operator = BlockSymmetricOperator(
        A=A,
        C=C,
        D=D,
        small_indices=layout.small_indices,
        structured_indices=layout.structured_indices,
    )
    return FactorSmoothMomentSystem(
        operator=operator,
        xtw_small=xtw_small,
        xtw_structured=xtw_structured,
        xtwz_small=xtwz_small,
        xtwz_structured=xtwz_structured,
        sum_w=float(np.sum(weights)),
        sum_wz=float(np.sum(weighted_rhs)),
        dominant_group_index=dominant_group_index,
        dominant_group_name=layout.dominant_group_name,
    )


def build_structured_system(
    group_matrices: list[GroupMatrix],
    groups: list[GroupSlice],
    W: NDArray,
    Wz: NDArray,
    *,
    dominant_group_index: int,
    layout: FactorSmoothLeafLayout | NestedStructuredLayout | None = None,
    prior_weights: NDArray | None = None,
    error: NDArray | None = None,
    signed: bool = False,
):
    """Dispatch sufficient-statistic construction by layout.

    A nested chain (a lone random effect is a chain of one) builds its leaf
    statistics; a FactorSmooth term, ``fs`` or ``sz``, builds its leaf system
    (its layout is built here for ``None``).  ``prior_weights`` (the fit's
    prior weights; ``None`` for unit weights) set the layout's border centre
    (``nested_prior_statistics``, ``block_leaves.factor_smooth_prior_statistics``),
    and ``error`` is the rows' weight-error scale.  ``signed`` declares
    observed-curvature rows by type (one-engine design §3.3), which the leaf
    system factors by the J-orthogonal route.
    """
    if isinstance(layout, NestedStructuredLayout):
        return build_nested_structured_system(
            group_matrices, groups, W, Wz, layout=layout, prior_weights=prior_weights, error=error
        )
    if not isinstance(group_matrices[dominant_group_index], FactorSmoothGroupMatrix):
        raise TypeError(
            "A RandomEffect structured system needs its nested chain layout "
            "(get_structured_layout); a lone random effect is a chain of one."
        )
    from superglm.solvers._structured.block_leaves import build_factor_smooth_leaf_system

    if layout is None:
        layout = build_factor_smooth_leaf_layout(
            group_matrices, groups, dominant_group_index=dominant_group_index
        )
    return build_factor_smooth_leaf_system(
        layout, W, Wz, prior_weights=prior_weights, error=error, signed=signed
    )


# Retired by the one engine (design §3.12): the systems of the retired
# block-Schur, range-space sz and scalar factors, importable at their pickled
# path as inert stand-ins so that models saved by v0.35.0 load; release 0.37.0
# may drop them.
__getattr__ = module_getattr(
    __name__,
    frozenset(
        {"BlockStructuredSystem", "ScalarStructuredSystem", "SumToZeroBlockStructuredSystem"}
    ),
)
