"""A joint Gram's row passes: one per distinct pair of index columns.

Each block of ``X_k' diag(c) X_l`` between two binned groups is ``S_a' H S_b``,
``H`` the table of ``c`` over the groups' two index columns (Li and Wood 2020,
Algorithm 1). A location-scale model has several curvature channels and its
predictors usually share covariates, so many blocks ask for the same pair of
columns with different weights. One row pass over the two columns forms all of
them: a table cell holds one entry per channel, so the row's entries for every
channel share a cache line. This is the multi-output scan of an aggregate
engine (Schleich et al., *SIGMOD* 2019, sec. 3.5) over Li and Wood's tables.

A categorical's sink level is a table level like any other: rows at the base
level add to a cell the contraction never reads, and the row pass has no
branch to mispredict.

Rows reach the tables in shards. A worker takes one shard and runs its rows
through every table, so a row stays in that worker's cache from its channels to
its last table; one task per table would stream every channel from memory once
per table, and memory bandwidth stops such a pass from scaling. Keeping each
vector in cache across a query's operators is vectorized execution (Boncz et
al., *CIDR* 2005, sec. 4), and constant-sized work units that workers claim at
run time and take through a whole pipeline are morsels (Leis et al., *SIGMOD*
2014, secs. 1 and 3.3). Each shard has its own tables, summed in row order, and
the shards merge in order; the partition depends on the rows and tables alone,
so the bits do not depend on the workers.

A shard's tables are one worker's own, so memory grows with the shards (thread-local
aggregation, Xue and Marcus, arXiv:2505.04153, table 2), and a design whose tables leave
too few shards cannot fill a pool that way. Its pass keeps one set of tables and
splits the work the other way, as Schleich et al. (sec. 1.2) pair domain with task
parallelism: morsels of rows form their channels on the pool in one bounded buffer,
a pipeline breaker in Leis et al.'s terms, and then each table is one task over that
buffer. Each table still sums its rows in row order, so again the bits do not depend
on the workers.

Owner of a :class:`JointTablePlan`: the fit session's ``PassFactors``. It holds
each support projection and the index columns' identities, reads no weight,
penalty or coefficient, and lives as long as the session's designs, so nothing
invalidates it. Owner of the tables: one assembly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit
from numpy.typing import NDArray

from superglm._group_matrix._block_queue import BlockTask
from superglm._group_matrix._group_matrix_algebra import _MAX_DISC_DISC_HIST_CELLS
from superglm._group_matrix._group_matrix_core import (
    CategoricalGroupMatrix,
    RandomEffectGroupMatrix,
)
from superglm._group_matrix._group_matrix_discretized import (
    DiscretizedSSPGroupMatrix,
    SupportCompressedSSPGroupMatrix,
)
from superglm._group_matrix._group_matrix_kernels import _native, _ssp_gram_needs_exact
from superglm._parallel import pooled_for, pooled_row_blocks
from superglm.distributional.layout import StackedLayout
from superglm.distributional.solver.packing import packed_pairs
from superglm.group_matrix import DesignMatrix

_SUPPORTS = (DiscretizedSSPGroupMatrix, SupportCompressedSSPGroupMatrix)
_LEVELS = (CategoricalGroupMatrix, RandomEffectGroupMatrix)
# Rows a table shard aims to hold: enough shards for the workers to balance positive rows.
_SHARD_ROWS = 2**15
# Rows one shard task holds in its row stack at a time.
STACK_BLOCK_ROWS = 2**14
# Fewer shards than this cannot fill a pool's workers, so the pass keeps one set of tables.
# Fixed, as the bits must not depend on the workers.
_MIN_SHARDS = 8
# One-set pass. A morsel is one pool item, small enough that a buffer gives a pool's workers
# many, and its rows outweigh claiming it. A buffer holds the channels until the table tasks read
# them: short enough to stay in a shared cache, long enough that a table's lines are worth
# moving to the worker that next claims it.
MORSEL_ROWS = 2**12
TABLE_BLOCK_ROWS = 2**17


# One kernel per table width, each channel's row vector its own operand, so a row reads its
# channels and writes one cell's entries with no lookup in between.
@njit(cache=True)
def _rows_1(table, left, right, n_right, a):
    for row in range(left.shape[0]):
        table[left[row] * n_right + right[row]] += a[row]


@njit(cache=True)
def _rows_2(table, left, right, n_right, a, b):
    for row in range(left.shape[0]):
        cell = (left[row] * n_right + right[row]) * 2
        table[cell] += a[row]
        table[cell + 1] += b[row]


@njit(cache=True)
def _rows_3(table, left, right, n_right, a, b, c):
    for row in range(left.shape[0]):
        cell = (left[row] * n_right + right[row]) * 3
        table[cell] += a[row]
        table[cell + 1] += b[row]
        table[cell + 2] += c[row]


@njit(cache=True)
def _rows_4(table, left, right, n_right, a, b, c, d):
    for row in range(left.shape[0]):
        cell = (left[row] * n_right + right[row]) * 4
        table[cell] += a[row]
        table[cell + 1] += b[row]
        table[cell + 2] += c[row]
        table[cell + 3] += d[row]


_MAX_TABLE_CHANNELS = 4


@njit(cache=True, nogil=True)
def _predictor_rows(stack, first, last, row_start, index, columns, owners, tables, constants):
    """Columns ``first..last`` of ``stack`` become each predictor's value on its rows.

    Each group's table at the row's level, the group's coefficients read through
    its levels once per pass (Lang et al. 2014), summed in the design's order and
    then the intercept, as the row route and prediction sum them.
    """
    count = last - first
    for k in range(constants.shape[0]):
        stack[k, first:last] = 0.0
    for g in range(columns.shape[0]):
        column = index[columns[g], row_start + first : row_start + last]
        table = tables[g]
        target = stack[owners[g], first:last]
        for i in range(count):
            target[i] += table[column[i]]
    for k in range(constants.shape[0]):
        stack[k, first:last] += constants[k]


@njit(cache=True, error_model="numpy")
def _majorant_weights(stack, first, last, diagonal, cross_channel, cross_left, cross_right, base):
    """Columns ``first..last`` of rows ``base + k`` take ``channel_weights``'s weight of predictor k.

    ``w_k = |c_kk| + sum_l |c_kl| rho_kl`` with ``rho_kl = sqrt(|c_kk|) / sqrt(|c_ll|)`` where
    both diagonals are positive and 1 elsewhere, each row's terms added in that function's
    order. One channel at a time over the rows, with no branch, so the loops vectorize.
    """
    count = last - first
    roots = np.empty((diagonal.shape[0], count))
    for k in range(diagonal.shape[0]):
        size = stack[diagonal[k], first:last]
        total = stack[base + k, first:last]
        root = roots[k]
        for i in range(count):
            total[i] = abs(size[i])
            root[i] = math.sqrt(abs(size[i]))
    for at in range(cross_channel.shape[0]):
        k = cross_left[at]
        j = cross_right[at]
        size = stack[cross_channel[at], first:last]
        total_k = stack[base + k, first:last]
        total_j = stack[base + j, first:last]
        root_k = roots[k]
        root_j = roots[j]
        for i in range(count):
            term = abs(size[i])
            balance = root_k[i] / root_j[i]
            both = (root_k[i] > 0.0) & (root_j[i] > 0.0)
            total_k[i] += term * balance if both else term
            total_j[i] += term / balance if both else term


@njit(cache=True, nogil=True)
def _channels_in_range(stack, channels, first, last) -> bool:
    """Whether every value of columns ``first..last`` of the leading ``channels`` rows is plain.

    Zero or in ``[2**-128, 2**128]``, the range ``_ssp_gram_needs_exact`` admits;
    NaN and infinities are not. Inside it no sum of up to 2**63 rows or product
    with in-range factors can leave float64's normal range.
    """
    inside = True
    for channel in range(channels):
        values = stack[channel, first:last]
        for row in range(values.shape[0]):
            size = abs(values[row])
            inside &= (2.0**-128 <= size <= 2.0**128) or size == 0.0
    return inside


@njit(cache=True, nogil=True)
def _rows_finite(stack, rows, first, last) -> bool:
    """Whether the leading ``rows`` rows of columns ``first..last`` are finite: ``x - x`` is 0 then."""
    total = 0.0
    for row in range(rows):
        values = stack[row, first:last]
        for i in range(values.shape[0]):
            total += values[i] - values[i]
    return total == 0.0


@njit(cache=True, nogil=True)
def _prepare_rows(
    stack, first, last, checked, diagonal, cross_channel, cross_left, cross_right, base
) -> bool:
    """Test the leading ``checked`` rows of ``stack``, False if not plain; else form the weights.

    Rows ``base + k`` of columns ``first..last`` take the majorant weights.
    """
    if not _channels_in_range(stack, checked, first, last):
        return False
    _majorant_weights(stack, first, last, diagonal, cross_channel, cross_left, cross_right, base)
    return True


@njit(cache=True, nogil=True)
def _add_tables(
    flat, starts, widths, lefts, rights, seconds, picks,
    index, stack, first, last, row_start, from_table, to_table,
):  # fmt: skip
    """Add columns ``first..last`` of ``stack``, column ``i`` being row ``row_start + i``, to tables."""
    for t in range(from_table, to_table):
        left = index[lefts[t], row_start + first : row_start + last]
        right = index[rights[t], row_start + first : row_start + last]
        table = flat[starts[t] : starts[t + 1]]
        second = seconds[t]
        pick = picks[t]
        a = stack[pick[0], first:last]
        if widths[t] == 1:
            _rows_1(table, left, right, second, a)
            continue
        b = stack[pick[1], first:last]
        if widths[t] == 2:
            _rows_2(table, left, right, second, a, b)
            continue
        c = stack[pick[2], first:last]
        if widths[t] == 3:
            _rows_3(table, left, right, second, a, b, c)
            continue
        _rows_4(table, left, right, second, a, b, c, stack[pick[3], first:last])


@njit(cache=True, nogil=True)
def _add_totals(totals, stack, first, last):
    for row in range(stack.shape[0]):
        totals[row] += np.sum(stack[row, first:last])


@njit(cache=True, nogil=True)
def _joint_rows(
    flat, totals, starts, widths, lefts, rights, seconds, picks,
    index, stack, first, last, row_start, checked,
    diagonal, cross_channel, cross_left, cross_right, base,
):  # fmt: skip
    """Columns ``first..last`` of ``stack`` into one shard's tables; False, adding nothing, if not plain."""
    if not _prepare_rows(
        stack, first, last, checked, diagonal, cross_channel, cross_left, cross_right, base
    ):
        return False
    _add_tables(
        flat, starts, widths, lefts, rights, seconds, picks,
        index, stack, first, last, row_start, 0, widths.shape[0],
    )  # fmt: skip
    _add_totals(totals, stack, first, last)
    return True


_POOLED_ROW_KERNELS = (
    _predictor_rows,
    _channels_in_range,
    _rows_finite,
    _prepare_rows,
    _add_tables,
    _add_totals,
    _joint_rows,
)


def _warmup_joint_tables() -> None:
    """Compile the row kernels for the operands a fit passes: a read-only index, one stack."""
    index = _native(np.zeros((1, 1), dtype=np.uint16))
    stack = np.zeros((4, 1))
    position = np.zeros(1, dtype=np.intp)
    _predictor_rows(stack, 0, 1, 0, index, position, position, np.zeros((1, 1)), np.zeros(1))
    _rows_finite(stack, 1, 0, 1)
    table_layout = (
        np.array([0, 1], dtype=np.intp), np.ones(1, dtype=np.intp),
        position, position, position, np.zeros((1, 4), dtype=np.intp),
    )  # fmt: skip
    _prepare_rows(stack, 0, 1, 1, position, position, position, position, 3)
    _add_tables(np.zeros(1), *table_layout, index, stack, 0, 1, 0, 0, 1)
    _add_totals(np.zeros(4), stack, 0, 1)
    _joint_rows(
        np.zeros(1), np.zeros(4), *table_layout,
        index, stack, 0, 1, 0, 1, position, position, position, position, 3,
    )  # fmt: skip


@dataclass(frozen=True)
class _Side:
    """One group: its index column, its matrix over the levels, its coefficients.

    ``support`` is ``B_unique @ R_inv``, which the tables contract; ``None`` is a
    categorical's one-hot levels, whose matrix is the identity without its sink level.
    ``factors`` are the group's own two arrays. A predictor multiplies them as every other
    route does, ``B_unique @ (R_inv @ beta)``, since the other association can cancel apart.
    """

    column: int
    support: NDArray[np.float64] | None
    width: int
    columns: slice
    factors: tuple[NDArray[np.float64], NDArray[np.float64]] | None


@dataclass(frozen=True)
class _Table:
    """One row pass: ``picks``, at most four rows of the stack, over one or two index columns.

    ``right`` None is a level table. A pair table runs from its smaller column to its larger.
    """

    left: int
    right: int | None
    picks: tuple[int, ...]
    cells: int


@dataclass(frozen=True)
class _Block:
    """``X_a' diag(channel) X_b``, read from entry ``slot`` of each cell of table ``table``.

    ``flipped`` says the left group's column is the table's second.
    """

    left: _Side
    right: _Side
    channel_pair: tuple[int, int]
    table: int
    slot: int
    flipped: bool


def _plain_float(array) -> bool:
    return type(array) is np.ndarray and array.dtype == np.float64


def _rows_are_cheaper(left: _Side, right: _Side, levels: list[int], n: int) -> bool:
    """Whether a row-by-row product beats two supports' table, by ``_cross_gram``'s estimate."""
    n_i, n_j = levels[left.column], levels[right.column]
    if n_i * n_j > _MAX_DISC_DISC_HIST_CELLS:
        return True
    if left.support is None or right.support is None:
        return False
    p_i, p_j = left.width, right.width
    hist_work = n_i * n_j * (1 + p_i) + n_j * p_i * p_j
    row_work = 128_000 + n * (512 + p_i + p_j + p_i * p_j)
    return 4 * row_work < 3 * hist_work


def _split_tables(left, right, channels, cells) -> list[_Table]:
    """A column pair's tables, each at most ``_MAX_TABLE_CHANNELS`` wide, in channel order."""
    ordered = sorted(channels)
    return [
        _Table(left, right, tuple(ordered[at : at + _MAX_TABLE_CHANNELS]), cells)
        for at in range(0, len(ordered), _MAX_TABLE_CHANNELS)
    ]


class JointTablePlan:
    """Which tables a layout's assemblies form, which channels each holds, and what reads them."""

    __slots__ = (
        "block_rows",
        "blocks",
        "cross",
        "diagonal",
        "group_columns",
        "group_owners",
        "index",
        "intercepts",
        "levels",
        "n_channels",
        "n_inputs",
        "n_shards",
        "offsets",
        "one_table_set",
        "shard_rows",
        "sides",
        "slot",
        "stack_rows",
        "table_bytes",
        "table_layout",
        "tables",
    )

    def __init__(self, layout: StackedLayout, sides, indices, levels, budget: int) -> None:
        self.sides = sides
        # One narrow array of every distinct column, the rows' levels contiguous per column:
        # the row passes read two columns sequentially and need no wider index.
        self.index = np.empty((len(indices), layout.predictors[0].design.n), dtype=np.uint16)
        for position, index in enumerate(indices):
            self.index[position] = index
        self.index.setflags(write=False)
        self.levels = levels
        self.intercepts = tuple(state.intercept_index for state in layout.predictors)
        self.group_columns = np.array([side.column for row in sides for side in row], dtype=np.intp)
        self.group_owners = np.array(
            [k for k, row in enumerate(sides) for _side in row], dtype=np.intp
        )
        self.offsets = tuple(
            (k, state.offset) for k, state in enumerate(layout.predictors) if np.any(state.offset)
        )
        channels = packed_pairs(len(sides))
        self.n_channels = len(channels)
        # The channels and scores a row pass supplies, then each predictor's row weight
        # for the terminal certificate's majorant, formed from them.
        self.n_inputs = self.n_channels + len(sides)
        self.stack_rows = self.n_inputs + len(sides)
        self.diagonal = np.array([channels.index((k, k)) for k in range(len(sides))], dtype=np.intp)
        off_diagonal = [
            (index, k, other) for index, (k, other) in enumerate(channels) if k != other
        ]
        self.cross = tuple(
            np.array([entry[part] for entry in off_diagonal], dtype=np.intp) for part in range(3)
        )
        pairs: dict[tuple[int, int | None], set[int]] = {}
        needs = []

        def wants(left: int, right: int | None, channel: int) -> None:
            pairs.setdefault((left, right), set()).add(channel)

        for channel, (k, other) in enumerate(channels):
            for i, left in enumerate(sides[k]):
                for j, right in enumerate(sides[other]):
                    if k == other and j < i:
                        continue
                    key = (
                        (left.column, None)
                        if left.column == right.column
                        else tuple(sorted((left.column, right.column)))
                    )
                    wants(*key, channel)
                    needs.append((channel, (k, other), left, right, key))
            if self.intercepts[k] is not None:
                for right in sides[other]:
                    wants(right.column, None, channel)
            if k != other and self.intercepts[other] is not None:
                for left in sides[k]:
                    wants(left.column, None, channel)
        for k, row in enumerate(sides):
            for side in row:
                wants(side.column, None, self.n_channels + k)
                wants(side.column, None, self.n_inputs + k)
        self.tables = tuple(
            table
            for key in sorted(pairs, key=lambda key: (key[0], -1 if key[1] is None else key[1]))
            for table in _split_tables(
                key[0],
                key[1],
                pairs[key],
                levels[key[0]] * (1 if key[1] is None else levels[key[1]]),
            )
        )
        # (left column, right column or None, channel) -> (table, slot)
        self.slot = {
            (table.left, table.right, channel): (index, slot)
            for index, table in enumerate(self.tables)
            for slot, channel in enumerate(table.picks)
        }
        self.blocks = tuple(
            _Block(
                left,
                right,
                pair,
                *self.slot[(*key, channel)],
                key[1] is not None and left.column > right.column,
            )
            for channel, pair, left, right, key in needs
        )
        widths = np.array([len(table.picks) for table in self.tables], dtype=np.intp)
        cells = np.array([table.cells for table in self.tables], dtype=np.intp)
        starts = np.concatenate(([0], np.cumsum(widths * cells))).astype(np.intp)
        picks = np.zeros((len(self.tables), _MAX_TABLE_CHANNELS), dtype=np.intp)
        for position, table in enumerate(self.tables):
            picks[position, : len(table.picks)] = table.picks
        # A level table is a pair table with no second column: ``index, index, 0`` puts each
        # row at its own level.
        self.table_layout = (
            starts,
            widths,
            np.array([table.left for table in self.tables], dtype=np.intp),
            np.array(
                [table.left if table.right is None else table.right for table in self.tables],
                dtype=np.intp,
            ),
            np.array(
                [0 if table.right is None else levels[table.right] for table in self.tables],
                dtype=np.intp,
            ),
            picks,
        )
        self.table_bytes = 8 * int(starts[-1])
        # Several shards' tables take at most half the allowance, and the rows' stacks share
        # the rest. One set of tables may take more, and its buffer what they leave.
        rows = self.index.shape[1]
        wanted = max(1, math.ceil(rows / _SHARD_ROWS))
        self.n_shards = min(wanted, max(1, budget // 2 // max(self.table_bytes, 1)))
        self.one_table_set = self.n_shards < _MIN_SHARDS
        if self.one_table_set:
            self.n_shards = 1
        self.shard_rows = max(1, math.ceil(rows / self.n_shards))
        left = (budget - self.table_bytes) // (8 * self.stack_rows)
        self.block_rows = max(MORSEL_ROWS, min(TABLE_BLOCK_ROWS, left))

    @property
    def shard_bytes(self) -> int:
        """Bytes every shard's tables hold together."""
        return self.n_shards * self.table_bytes

    def predictor_tables(self, coefficients: NDArray) -> tuple[NDArray, NDArray]:
        """``(tables, constants)``: each group's value per level and each predictor's intercept."""
        sides = [side for row in self.sides for side in row]
        tables = np.zeros((len(sides), max((self.levels[s.column] for s in sides), default=0)))
        for position, side in enumerate(sides):
            beta = coefficients[side.columns]
            if side.factors is None:
                tables[position, : side.width] = beta
            else:
                basis, transform = side.factors
                tables[position, : basis.shape[0]] = basis @ (transform @ beta)
        constants = np.array(
            [0.0 if at is None else coefficients[at] for at in self.intercepts], dtype=np.float64
        )
        return tables, constants

    def fill_predictors(
        self, stack, first, last, row_start, tables, constants, *, offsets: bool = True
    ) -> None:
        """Columns ``first..last`` of ``stack`` take the predictors of rows from ``row_start``."""
        _predictor_rows(
            stack, first, last, row_start, self.index, self.group_columns, self.group_owners,
            tables, constants,
        )  # fmt: skip
        if offsets:
            for k, offset in self.offsets:
                stack[k, first:last] += offset[row_start + first : row_start + last]

    def predictors_finite(self, stack, count: int) -> bool:
        """Whether the predictors in the first ``count`` columns of ``stack`` are finite."""
        return _rows_finite(stack, len(self.sides), 0, count)

    @property
    def shares_last_predictor(self) -> bool:
        """Whether the last predictor takes one value on every row: an intercept, no offset."""
        last = len(self.sides) - 1
        return not self.sides[last] and all(k != last for k, _offset in self.offsets)


def joint_table_plan(layout: StackedLayout, budget: int) -> JointTablePlan | None:
    """The layout's plan, or None where a design has no one index column per group.

    None also for a support whose factors need the exact route, for tables over
    ``budget`` bytes, and for a pair of supports whose row-by-row product is
    the cheaper (few rows against many bins).
    """
    n = layout.predictors[0].design.n
    sides: list[tuple[_Side, ...]] = []
    indices: list[NDArray] = []
    levels: list[int] = []
    one_hot: list[bool] = []

    def column_of(index: NDArray, count: int, levels_are_one_hot: bool) -> int:
        for position, other in enumerate(indices):
            if (
                levels[position] == count
                and one_hot[position] == levels_are_one_hot
                and (other is index or np.array_equal(other, index))
            ):
                return position
        indices.append(index)
        levels.append(count)
        one_hot.append(levels_are_one_hot)
        return len(indices) - 1

    for state in layout.predictors:
        design = state.design
        if type(design) is not DesignMatrix or design.n != n:
            return None
        cursor = state.coefficient_slice.start + int(state.intercept_index is not None)
        row = []
        for group in design.group_matrices:
            width = int(group.shape[1])
            kind = type(group)
            if kind in _SUPPORTS:
                factors = (group.B_unique, group.R_inv)
                if not all(_plain_float(factor) for factor in factors) or _ssp_gram_needs_exact(
                    *factors
                ):
                    return None
                index, count, support = group.bin_idx, int(group.n_bins), factors[0] @ factors[1]
            elif kind in _LEVELS:
                factors = None
                index, count, support = group.codes, int(group.n_levels) + 1, None
            else:
                return None
            # The row passes index their tables without bounds checks.
            if index.size and not (0 <= index.min() and index.max() < count):
                return None
            column = column_of(index, count, support is None)
            row.append(_Side(column, support, width, slice(cursor, cursor + width), factors))
            cursor += width
        sides.append(tuple(row))
    if max(levels, default=0) > 1 << 16:
        return None
    plan = JointTablePlan(layout, tuple(sides), tuple(indices), tuple(levels), budget)
    if plan.table_bytes > budget or any(
        _rows_are_cheaper(left, right, levels, n)
        for k, row in enumerate(sides)
        for left in row
        for other in sides[k:]
        for right in other
        if left.column != right.column
    ):
        return None
    return plan


def stack_in_range(stack: NDArray[np.float64], rows: int) -> bool:
    """Whether every channel of a row block suits the tables' plain float64 route."""
    return bool(_channels_in_range(stack, stack.shape[0], 0, rows))


def _contract(left: _Side, right: _Side, table: NDArray) -> NDArray:
    """``S_a' table S_b``, in the association with the fewer multiply-adds; a one-hot side slices."""
    if left.support is None:
        kept = table[: left.width]
        return kept[:, : right.width] if right.support is None else kept @ right.support
    if right.support is None:
        return left.support.T @ table[:, : right.width]
    n_i, n_j = table.shape
    if left.width * n_i * n_j + left.width * n_j * right.width <= (
        n_i * n_j * right.width + left.width * n_i * right.width
    ):
        return np.dot(np.dot(left.support.T, table), right.support)
    return np.dot(left.support.T, np.dot(table, right.support))


def _level_contract(left: _Side, right: _Side, sums: NDArray) -> NDArray:
    """``S_a' diag(sums) S_b`` for two groups on one index column, both supports or both one-hot."""
    if left.support is None:
        return np.diag(sums[: left.width])
    return np.dot(left.support.T, right.support * sums[:, None])


def _side_sums(side: _Side, sums: NDArray) -> NDArray:
    """``X_side' values`` from the values' sums over the side's levels."""
    return sums[: side.width] if side.support is None else side.support.T @ sums


def _noop(_value) -> None:
    return None


class JointTables:
    """One assembly's tables: rows add into their shard's, the shards merge, and blocks read once."""

    def __init__(self, plan: JointTablePlan) -> None:
        self.plan = plan
        self._shards = np.zeros((plan.n_shards, plan.table_bytes // 8))
        self._shard_totals = np.zeros((plan.n_shards, plan.stack_rows))
        self._merged = np.empty(0)
        self._totals = np.empty(0)

    def add_block(self, shard, stack, first, last, row_start, checked) -> bool:
        """Columns ``first..last`` of ``stack`` into ``shard``; see :func:`_joint_rows`."""
        plan = self.plan
        return _joint_rows(
            self._shards[shard], self._shard_totals[shard], *plan.table_layout,
            plan.index, stack, first, last, row_start, checked,
            plan.diagonal, *plan.cross, plan.n_inputs,
        )  # fmt: skip

    def prepare_rows(self, stack, first, last, checked) -> bool:
        """Test and form one morsel for :meth:`add_by_table`; see :func:`_prepare_rows`."""
        plan = self.plan
        return _prepare_rows(stack, first, last, checked, plan.diagonal, *plan.cross, plan.n_inputs)

    def add_by_table(self, stack, first, last, row_start) -> None:
        """Columns ``first..last``, prepared, into the one set of tables, a task per table and totals."""
        plan = self.plan
        flat, totals = self._shards[0], self._shard_totals[0]
        n_tables = len(plan.tables)

        def task(at: int) -> None:
            if at == n_tables:
                _add_totals(totals, stack, first, last)
                return
            _add_tables(
                flat, *plan.table_layout, plan.index, stack, first, last, row_start, at, at + 1
            )

        pooled_for(task, list(range(n_tables + 1)), 1 << 12)

    def add_rows(self, start: int, stop: int, stack: NDArray[np.float64]) -> None:
        """Rows ``[start, stop)``, already found plain, whose channels are ``stack``'s columns."""
        size = self.plan.shard_rows
        if self.plan.one_table_set:
            filled = stop - start
            for first in range(0, filled, TABLE_BLOCK_ROWS):
                last = min(first + TABLE_BLOCK_ROWS, filled)

                def morsel(begin: int, end: int, first: int = first) -> None:
                    self.prepare_rows(stack, first + begin, first + end, 0)

                pooled_row_blocks(morsel, last - first, MORSEL_ROWS)
                self.add_by_table(stack, first, last, start)
            return

        def task(shard: int) -> None:
            first = max(start, shard * size) - start
            last = min(stop, (shard + 1) * size) - start
            self.add_block(shard, stack, first, last, start, 0)

        pooled_for(task, list(range(start // size, (stop - 1) // size + 1)), 1 << 12)

    def merge(self) -> None:
        """Add the shards in order; what the blocks read from here on. One shard is the sum."""
        if len(self._shards) == 1:
            self._merged, self._totals = self._shards[0], self._shard_totals[0]
        else:
            self._merged = np.sum(self._shards, axis=0)
            self._totals = np.sum(self._shard_totals, axis=0)
        self.release()

    def release(self) -> None:
        """Let the shards' tables go; a pass that declines holds none through its fallback."""
        self._shards = self._shard_totals = np.empty((0, 0))

    def _entries(self, table: int, slot: int) -> NDArray:
        """One channel's entries of a table, in cell order."""
        starts = self.plan.table_layout[0]
        width = len(self.plan.tables[table].picks)
        return self._merged[starts[table] + slot : starts[table + 1] : width]

    def _pair_table(self, block: _Block) -> NDArray:
        table = self.plan.tables[block.table]
        levels = self.plan.levels
        rows = self._entries(block.table, block.slot).reshape(
            levels[table.left], levels[table.right]
        )
        return rows.T if block.flipped else rows

    def contract_tasks(self, score: NDArray, curvature: NDArray) -> list[BlockTask]:
        """One task per block, adding it and its transpose into ``curvature``, and then the score."""
        plan = self.plan
        tasks: list[BlockTask] = []

        def add_block(block: _Block, value: NDArray) -> None:
            k, other = block.channel_pair
            if k == other and block.left is block.right:
                # One triangle, so symmetry is bit-for-bit exact.
                value = np.triu(value) + np.triu(value, k=1).T
                curvature[block.left.columns, block.left.columns] += value
                return
            curvature[block.left.columns, block.right.columns] += value
            curvature[block.right.columns, block.left.columns] += value.T

        for block in plan.blocks:
            if plan.tables[block.table].right is None:

                def run(_cache, _profile, block=block):
                    sums = self._entries(block.table, block.slot)
                    return _level_contract(block.left, block.right, sums)
            else:

                def run(_cache, _profile, block=block):
                    return _contract(block.left, block.right, self._pair_table(block))

            cost = block.left.width * block.right.width * plan.tables[block.table].cells

            def place(value, block=block):
                add_block(block, value)

            tasks.append(BlockTask(len(tasks), float(cost), 8 * cost, run, place))

        def linear(_cache, _profile):
            self._add_linear(score, curvature)

        tasks.append(BlockTask(len(tasks), float(curvature.shape[0]) ** 2, 8, linear, _noop))
        return tasks

    def _sums(self, column: int, channel: int) -> NDArray:
        return self._entries(*self.plan.slot[(column, None, channel)])

    def majorant_sums(self):
        """``(predictor k, position, level sums of k's row weight)`` per group."""
        plan = self.plan
        for k, row in enumerate(plan.sides):
            for position, side in enumerate(row):
                sums = self._sums(side.column, plan.n_inputs + k)
                yield k, position, sums if side.support is not None else sums[: side.width]

    def majorant_total(self, k: int) -> float:
        return float(self._totals[self.plan.n_inputs + k])

    def _add_linear(self, score: NDArray, curvature: NDArray) -> None:
        """The score, and every block with an intercept: both read level sums and totals only."""
        plan = self.plan
        n_channels = plan.n_channels
        for k, row in enumerate(plan.sides):
            intercept = plan.intercepts[k]
            for side in row:
                score[side.columns] += _side_sums(side, self._sums(side.column, n_channels + k))
            if intercept is not None:
                score[intercept] += self._totals[n_channels + k]
        for channel, (k, other) in enumerate(packed_pairs(len(plan.sides))):
            left_intercept, right_intercept = plan.intercepts[k], plan.intercepts[other]
            if left_intercept is not None:
                for side in plan.sides[other]:
                    values = _side_sums(side, self._sums(side.column, channel))
                    curvature[left_intercept, side.columns] += values
                    curvature[side.columns, left_intercept] += values
            if k != other and right_intercept is not None:
                for side in plan.sides[k]:
                    values = _side_sums(side, self._sums(side.column, channel))
                    curvature[side.columns, right_intercept] += values
                    curvature[right_intercept, side.columns] += values
            if left_intercept is not None and right_intercept is not None:
                curvature[left_intercept, right_intercept] += self._totals[channel]
                if left_intercept != right_intercept:
                    curvature[right_intercept, left_intercept] += self._totals[channel]
