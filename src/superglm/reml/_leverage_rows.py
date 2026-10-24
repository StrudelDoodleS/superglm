"""Solved centred rows ``H^-1 x_r`` and their leverages, for the REML weight-derivative pass.

The W(rho) gradient and the outer Hessian's first-order W terms read, for
each row ``r`` of the centred design ``x_r = (x - c) - c_lo``, the solved row
``y_r = H^-1 x_r`` and the leverage ``h_r = x_r' y_r``
(``w_derivatives._leverage_derivative_terms``).  :func:`solved_row_blocks`
returns them as row-block jobs by one of two routes; the support route's
blocks are independent, so the caller forms them on a pool and sums their
terms in row order (bitwise the same at every worker count).

**Support route.**  A design whose groups are dense columns or indexed
supports (a group's rows are rows of a small table, ``values[codes] @ R``)
never forms ``X_c H^-1`` as a product.  Each such group ``g`` keeps a table
``T_g = V_g H^-1[g, :]`` with one row per support row, and ``y_r`` is the sum
of one table row per group.  For a categorical and for a spline-by-category
level, most rows are zero: ``V_g`` is the raw support and the constant
``-c_g' H^-1[g, :]`` enters every row once (``base``), so a row adds only the
table rows of the groups it is in.  Every other indexed group is centred
before its product, ``V_g = values_g - c_g``, as the chunked route centres
it.  Dense columns are centred by their exact pair and multiplied per block.
The work per row is one table row per group a row is in, against one ``p x
p`` product per row on the chunked route.

**Error.**  Write ``|.|`` entrywise.  On the chunked route each entry of
``y_r`` lies within ``gamma_(p+2) |x_r|' |H^-1|`` of its exact value
(centring, then a dot product of length ``p``; Higham 2002, secs. 3.1, 3.5).
On the support route a centred group's table entry carries ``gamma_(w+1)``
of ``|x_r[g]|' |H^-1[g, :]|`` (``w`` its width) and a raw group's
``gamma_w |v_r[g]|' |H^-1[g, :]|`` plus ``gamma_w |c_g|' |H^-1[g, :]|`` in
``base``, and the sum over at most ``G`` groups adds ``gamma_G`` of the sum of
the terms' magnitudes; with ``K = p + G + 2`` every entry lies within
``gamma_K (|x_r| + |c_z|)' |H^-1|``, where ``c_z`` is the centre on the raw
groups' columns and zero elsewhere.  The raw groups' rows are in
``[0, 1]`` (a one-hot) or a spline basis, and their centres are level means
of such rows, so the extra term is bounded by the level's share of the
centre.  ``h_r`` adds ``gamma_p |x_r|' |y_r|``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

import numpy as np
from numba import njit  # type: ignore[import-untyped]
from numpy.typing import NDArray

from superglm._blas_threads import release_for_serial_row_passes
from superglm.group_matrix import DesignMatrix
from superglm.solvers.centered_system import iter_grouped_design_chunks

_BLOCK_BYTES = 32 * 1024 * 1024


@njit(cache=True, nogil=True)
def _solved_rows_and_leverages(
    solved,
    leverage,
    rows,
    codes,
    base,
    table,
    offsets,
    zero_codes,
    supports,
    support_offsets,
    widths,
    columns,
    dense_values,
    dense_columns,
    dense_inverse,
):
    """Each row's ``y = base + sum_g table[offsets[g] + code_g] + sum_d x_d H^-1[d, :]`` and ``h = x' y``.

    A group whose code is its zero code adds no table row (its centre is in
    ``base``); ``h`` reads every group's centred support row, the zero row
    included, and the dense columns' centred values, while ``y`` is in cache.
    """
    n_groups = codes.shape[0]
    width = solved.shape[1]
    n_dense = dense_columns.shape[0]
    for position in range(rows.shape[0]):
        k = rows[position]
        target = solved[position]
        for j in range(width):
            target[j] = base[j]
        for g in range(n_groups):
            code = codes[g, k]
            if code == zero_codes[g]:
                continue
            source = table[offsets[g] + code]
            for j in range(width):
                target[j] += source[j]
        for d in range(n_dense):
            value = dense_values[position, d]
            if value != 0.0:
                source = dense_inverse[d]
                for j in range(width):
                    target[j] += value * source[j]
        total = 0.0
        for g in range(n_groups):
            group_width = widths[g]
            start = support_offsets[g] + codes[g, k] * group_width
            column = columns[g]
            for j in range(group_width):
                total += supports[start + j] * target[column + j]
        for d in range(n_dense):
            total += dense_values[position, d] * target[dense_columns[d]]
        leverage[position] = total


@dataclass(frozen=True)
class _SupportPlan:
    """One call's tables (owner: :func:`solved_row_blocks`; lifetime: that call)."""

    base: NDArray
    table: NDArray
    offsets: NDArray
    zero_codes: NDArray
    supports: NDArray
    support_offsets: NDArray
    widths: NDArray
    columns: NDArray
    full_codes: list  # (group position, codes over all rows)
    level_codes: list  # (group position, sorted rows, codes on those rows, zero code)
    dense: list  # (column, width, matrix)
    dense_columns: NDArray
    dense_inverse: NDArray


_POOLED_ROW_KERNELS = (_solved_rows_and_leverages,)
"""The kernel a pooled solved-row block runs on its worker: ``nogil``, never ``prange``,
writing only the block's own arrays."""


def _support_plan(
    dm: DesignMatrix,
    inverse: NDArray,
    centre_hi: NDArray,
    centre_lo: NDArray | None,
) -> _SupportPlan | None:
    """The support route's tables, else ``None`` (another group type, or too large)."""
    from superglm._group_matrix._group_matrix_centered import _reads_as_float64
    from superglm._group_matrix._group_matrix_core import ONE_HOT_CODE_TYPES
    from superglm._parallel import resolve_max_memory
    from superglm.group_matrix import (
        DenseGroupMatrix,
        DiscretizedSCOPGroupMatrix,
        DiscretizedSplineCategoricalGroupMatrix,
        DiscretizedSSPGroupMatrix,
    )

    if type(dm) is not DesignMatrix:
        return None
    # Sized from metadata before any table is formed.
    rows = 0
    for gm in dm.group_matrices:
        if type(gm) is DenseGroupMatrix:
            continue
        if isinstance(gm, DiscretizedSSPGroupMatrix):
            rows += int(gm.B_unique.shape[0])
        elif isinstance(gm, DiscretizedSCOPGroupMatrix):
            rows += int(gm.B_scop_unique.shape[0])
        elif isinstance(gm, DiscretizedSplineCategoricalGroupMatrix):
            rows += int(gm.n_bins) + 1
        elif type(gm) in ONE_HOT_CODE_TYPES:
            rows += int(gm.n_levels) + 1
        else:
            return None
    if 2 * rows > dm.n or 8 * rows * dm.p > resolve_max_memory() // 4:
        return None

    p = dm.p
    base = np.zeros(p, dtype=np.float64)
    tables: list[NDArray] = []
    supports: list[NDArray] = []
    meta: list[tuple[int, int, int]] = []  # (table rows before, zero code, column)
    full_codes: list[tuple[int, NDArray]] = []
    level_codes: list[tuple[int, NDArray, NDArray, int]] = []
    dense: list[tuple[int, int, NDArray]] = []
    table_rows = 0
    column = 0
    for gm in dm.group_matrices:
        width = int(gm.shape[1])
        cols = slice(column, column + width)
        centre = centre_hi[cols]
        centre_low = None if centre_lo is None else centre_lo[cols]
        if type(gm) is DenseGroupMatrix:
            dense.append((column, width, gm.M))
            column += width
            continue
        level: tuple[NDArray, NDArray] | None = None
        codes: NDArray | None = None
        if isinstance(gm, DiscretizedSplineCategoricalGroupMatrix):
            values, transform, level = gm.B_unique, gm.R_inv, (gm.row_idx, gm.bin_idx_level)
        elif isinstance(gm, DiscretizedSSPGroupMatrix):
            values, transform, codes = gm.B_unique, gm.R_inv, gm.bin_idx
        elif isinstance(gm, DiscretizedSCOPGroupMatrix):
            values, transform, codes = gm.B_scop_unique, None, gm.bin_idx
        elif type(gm) in ONE_HOT_CODE_TYPES:
            values, transform, codes = None, None, gm.codes
        else:  # pragma: no cover - sized above
            return None
        if any(
            operand is not None and not _reads_as_float64(np.asarray(operand).dtype)
            for operand in (values, transform)
        ):
            return None
        if values is None:  # categorical: the identity, its base level the zero row
            raw = np.eye(width, dtype=np.float64)
        else:
            raw = np.asarray(values, dtype=np.float64)
            if transform is not None:
                raw = raw @ np.asarray(transform, dtype=np.float64)
        position = len(meta)
        if level is not None or values is None:
            # Zero rows outside the level (or at the base level): raw table,
            # the centre once in ``base``, the centred zero row last.
            zero = raw.shape[0]
            tables.append(raw @ inverse[cols, :])
            base -= centre @ inverse[cols, :]
            centred = np.vstack((raw, np.zeros((1, width)))) - centre
            if centre_low is not None:
                base -= centre_low @ inverse[cols, :]
                centred -= centre_low
            supports.append(centred)
            if level is not None:
                level_rows, level_bins = level
                order = np.argsort(level_rows, kind="stable")
                level_codes.append(
                    (
                        position,
                        np.ascontiguousarray(level_rows[order], dtype=np.intp),
                        np.ascontiguousarray(level_bins[order], dtype=np.intp),
                        zero,
                    )
                )
            else:
                full_codes.append((position, np.asarray(codes)))
        else:
            zero = -1
            centred = raw - centre
            if centre_low is not None:
                centred -= centre_low
            tables.append(centred @ inverse[cols, :])
            supports.append(centred)
            full_codes.append((position, np.asarray(codes)))
        meta.append((table_rows, zero, column))
        table_rows += tables[-1].shape[0]
        column += width
    if not meta:
        return None
    widths = np.array([support.shape[1] for support in supports], dtype=np.intp)
    dense_columns = np.array(
        [column + j for column, width, _ in dense for j in range(width)], dtype=np.intp
    )
    support_offsets = np.cumsum([0, *(support.size for support in supports)])[:-1]
    return _SupportPlan(
        base=base,
        table=np.ascontiguousarray(np.vstack(tables)),
        offsets=np.array([entry[0] for entry in meta], dtype=np.intp),
        zero_codes=np.array([entry[1] for entry in meta], dtype=np.intp),
        supports=np.concatenate([support.ravel() for support in supports]),
        support_offsets=np.asarray(support_offsets, dtype=np.intp),
        widths=widths,
        columns=np.array([entry[2] for entry in meta], dtype=np.intp),
        full_codes=full_codes,
        level_codes=level_codes,
        dense=dense,
        dense_columns=dense_columns,
        dense_inverse=np.ascontiguousarray(inverse[dense_columns, :]),
    )


def _support_block_rows(p: int) -> int:
    """Rows per support-route block: ``_BLOCK_BYTES`` of solved rows, at least 256, at most 8192."""
    return max(256, min(8192, _BLOCK_BYTES // (8 * max(p, 1))))


def _support_block(
    plan: _SupportPlan,
    start: int,
    stop: int,
    p: int,
    centre_hi: NDArray,
    centre_lo: NDArray | None,
    active: NDArray,
) -> tuple[NDArray, NDArray, NDArray] | None:
    """Rows ``start:stop``'s ``(rows, y, h)`` on the support route, ``None`` when none is active."""
    local = np.flatnonzero(active[start:stop])
    if local.size == 0:
        return None
    codes = np.empty((len(plan.offsets), stop - start), dtype=np.intp)
    for position, full in plan.full_codes:
        codes[position] = full[start:stop]
    for position, level_rows, level_bins, zero in plan.level_codes:
        first = int(np.searchsorted(level_rows, start, side="left"))
        last = int(np.searchsorted(level_rows, stop, side="left"))
        codes[position] = zero
        codes[position, level_rows[first:last] - start] = level_bins[first:last]
    local = np.ascontiguousarray(local, dtype=np.intp)
    dense_values = np.empty((local.size, plan.dense_columns.size), dtype=np.float64)
    filled = 0
    for column, width, matrix in plan.dense:
        cols = slice(column, column + width)
        values = np.asarray(matrix[start:stop], dtype=np.float64)[local] - centre_hi[cols]
        if centre_lo is not None:
            values -= centre_lo[cols]
        dense_values[:, filled : filled + width] = values
        filled += width
    solved = np.empty((local.size, p), dtype=np.float64)
    leverage = np.empty(local.size, dtype=np.float64)
    _solved_rows_and_leverages(
        solved,
        leverage,
        local,
        codes,
        plan.base,
        plan.table,
        plan.offsets,
        plan.zero_codes,
        plan.supports,
        plan.support_offsets,
        plan.widths,
        plan.columns,
        dense_values,
        plan.dense_columns,
        plan.dense_inverse,
    )
    return start + local, solved, leverage


def _chunked_blocks(
    dm: DesignMatrix,
    inverse: NDArray,
    centre_hi: NDArray,
    centre_lo: NDArray | None,
    active: NDArray,
) -> Iterator[tuple[NDArray, NDArray, NDArray]]:
    for start, stop, design in iter_grouped_design_chunks(dm):
        local = np.flatnonzero(active[start:stop])
        if local.size == 0:
            continue
        centred = design[local]
        centred -= centre_hi
        if centre_lo is not None:
            centred -= centre_lo
        solved = centred @ inverse
        leverage = np.einsum("ij,ij->i", solved, centred, optimize=True)
        yield start + local, solved, leverage


@dataclass(frozen=True)
class RowBlocks:
    """The row blocks of one solved-row pass, in row order.

    ``jobs`` are callables, each returning one block's ``(rows, y, h)`` or
    ``None`` when no row in it is active.  On the support route they are
    independent (``block_bytes`` is one job's working set) and may run on a
    pool; on the chunked route ``block_bytes`` is ``None`` and the jobs read
    one stream of design chunks, so they run in order on the calling thread.
    """

    jobs: Iterable[Callable[[], tuple[NDArray, NDArray, NDArray] | None]]
    block_bytes: int | None


def solved_row_blocks(
    dm: DesignMatrix,
    inverse: NDArray,
    centre_hi: NDArray,
    centre_lo: NDArray | None,
    active: NDArray,
) -> RowBlocks:
    """The active rows' solved centred rows and leverages ``(rows, y, h)``, block by block.

    ``rows`` indexes the design; ``y[k] = H^-1 x_rows[k]`` and ``h[k] =
    x_rows[k]' y[k]`` for the centred row ``x = (x - centre_hi) - centre_lo``.
    Rows outside ``active`` are skipped and never formed, so a row that
    carries no weight cannot overflow into the sums.  A block's values do not
    depend on which thread forms it or on any other block.
    """
    inverse = np.ascontiguousarray(inverse, dtype=np.float64)
    plan = _support_plan(dm, inverse, centre_hi, centre_lo)
    if plan is None:
        release_for_serial_row_passes(dm.p)
        chunks = _chunked_blocks(dm, inverse, centre_hi, centre_lo, active)
        return RowBlocks(jobs=((lambda block=block: block) for block in chunks), block_bytes=None)
    n, p = dm.n, dm.p
    rows = _support_block_rows(p)

    def job(start: int):
        return lambda: _support_block(
            plan, start, min(start + rows, n), p, centre_hi, centre_lo, active
        )

    # A job holds its solved rows and the caller's products of them, a few
    # copies of ``rows x p`` at once.
    return RowBlocks(jobs=[job(start) for start in range(0, n, rows)], block_bytes=4 * 8 * rows * p)


def _warmup_leverage_rows() -> None:
    """Compile the solved-row kernel for the operands its exact-path caller passes (empty)."""
    i64, f64 = np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
    f64_2d, i64_2d = np.zeros((0, 0), dtype=np.float64), np.zeros((0, 0), dtype=np.int64)
    _solved_rows_and_leverages(
        f64_2d, f64, i64, i64_2d, f64, f64_2d, i64, i64, f64, i64, i64, i64, f64_2d, i64, f64_2d
    )
