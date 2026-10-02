"""Nested random-effect chain elimination (fix D).

The mathematics, accuracy bounds and refusal rules are in
``notes/research/2026-09-26-nested-random-effect-elimination.md`` (cited below
as §N) and the exact contracts in
``notes/research/lean-gaussian-certificate/NestedElimination.lean``.

Geometry.  A chain is ``L >= 1`` random-effect terms, each strictly nested in
the next coarser one; a lone random effect is a chain of one, whose leaves are
its roots.  Level 0 is the coarsest (its nodes are the roots) and
level ``L - 1`` the finest (the leaves, the only level whose codes touch
rows).  Every tree node carries one coefficient.  Node order is level-major,
coarsest first: node ``tree.offsets[j] + u`` is local node ``u`` of level
``j``, and ``structured_indices[tree.offsets[j] + u]`` is its global
coefficient index.  The border ("small") block holds every other coefficient,
crossed random effects included; in augmented coordinates the intercept is
border column 0 and global index 0.

Every operator here is leaf-form (§3.6).  With ``M`` the leaf-to-node
incidence (``M[l, u] = 1`` iff ``u`` is ``l`` or an ancestor of ``l``), row
weights ``a_r`` and border rows ``x_r``,

    O = [[M' diag(a_leaf) M, M' C_leaf], [C_leaf' M, A]],
    a_leaf[l] = sum_{r in l} a_r,  C_leaf[l] = sum_{r in l} a_r x_r,  A = X_b' diag(a) X_b,

and the penalized Hessian is ``H = O_w + blockdiag(diag(lambda_node), S_b)``.

Refusals follow the structured-factor convention.  ``np.linalg.LinAlgError``
reports what an iterate's numbers caused (non-finite statistics, a tree or
super-root pivot within its certified uncertainty or below minus it, material
negative Schur curvature, exact intercept aliasing); callers such as the
observed-geometry build rely on that type.  ``ValueError`` reports a
malformed call (shapes, partitions, coordinates, an operator built about
other leaf means).  ``TypeError`` reports an operator kind the nested factor
does not represent (for example a single-level ``SymmetricBlockOperator``).

The factors are verified against exact rational references in
``tests/test_nested_schur_factor.py``; ``tests/test_nested_structured_fit.py``
runs complete fits against the dense backend.
"""

from __future__ import annotations

import dataclasses
import itertools
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import scipy.linalg
import scipy.sparse
from numpy.typing import NDArray

from superglm._blas_threads import narrow_kernel_blas_threads
from superglm._group_matrix._group_matrix_core import (
    CategoricalGroupMatrix,
    DenseGroupMatrix,
    RandomEffectGroupMatrix,
    SparseSSPGroupMatrix,
)
from superglm._group_matrix._group_matrix_discretized import (
    DiscretizedSSPGroupMatrix,
    SupportCompressedSSPGroupMatrix,
)
from superglm.solvers._structured.border import (
    BorderCertificate,
    BorderGenerators,
    factor_border,
)
from superglm.solvers._structured.factors import _penalty_product
from superglm.solvers._structured.operators import (
    CenteredBlockOperator,
    LowRankSymmetricOperator,
    SumBlockOperator,
)
from superglm.solvers.hessian_factor import _component_indices, _component_omega
from superglm.types import PenaltyComponent

if TYPE_CHECKING:
    from superglm._group_matrix._group_matrix_execution import MatrixExecutionPlan
    from superglm.group_matrix import GroupMatrix
    from superglm.solvers._structured.factors import DerivativeDirection
    from superglm.solvers._structured.operators import CompactSymmetricOperator
    from superglm.types import GroupSlice

# Border blocks whose rows the row pass forms from their own arrays (``LeafRows``),
# by exact type.  A RandomEffect border block is never among them: its columns
# are sparse indicators by type (``NestedStructuredLayout.sparse_indicators``).
_TABLE_TYPES = (DiscretizedSSPGroupMatrix, SupportCompressedSSPGroupMatrix)
_ONE_HOT_TYPES = (CategoricalGroupMatrix,)
# The saved-model format of the nested factors (one-engine design §3.12).  A
# factor pickled by another build (master 94359786, the uncommitted speed
# build) holds another set of derived attributes but always the inputs it
# factored: it is rebuilt from those by the current engine on first use, with
# a notice, and a state written by this format is restored as is.  Release
# 0.37.0 may drop the foreign-state rebuild.
_NESTED_STATE_FORMAT = 1
_REBUILT_NOTICE = (
    "This model was saved by an earlier superglm build: its nested random-effect "
    "factor was rebuilt with the current solver at the saved coefficients and "
    "smoothing parameters, so standard errors, leverage and summaries use the "
    "current engine."
)


def _restore_nested_state(instance, state: dict) -> None:
    """``__setstate__`` of the nested factors: keep a foreign state for a lazy rebuild."""
    if state.get("_state_format") == _NESTED_STATE_FORMAT:
        instance.__dict__.update(state)
    else:
        instance.__dict__["_retired_state"] = dict(state)


def _pending_nested_state(instance) -> dict:
    """``__getstate__`` of the nested factors: a foreign state not yet rebuilt is saved as it came.

    A factor loaded from another build holds only ``_retired_state`` until
    first use.  Saving or copying it then must write that foreign state
    itself, which ``_restore_nested_state`` keeps again for the same lazy
    rebuild; the wrapper would be wrapped once more and lose its inputs.
    """
    pending = instance.__dict__.get("_retired_state")
    return instance.__dict__ if pending is None else pending


def _rebuild_on_first_use(instance, name: str, rebuild: Callable[[dict], None]):
    """``__getattr__`` of the nested factors: rebuild a foreign state once, then look up again.

    Reached only for an attribute the instance lacks.  Dunder lookups (pickle,
    copy) never trigger the rebuild, and the retained state is taken out of
    the instance before the rebuild runs, so a lookup the rebuild itself makes
    fails normally instead of recursing.
    """
    if name.startswith("__"):
        raise AttributeError(name)
    state = instance.__dict__.pop("_retired_state", None)
    if state is None:
        raise AttributeError(f"{type(instance).__name__!r} object has no attribute {name!r}")
    rebuild(state)
    return getattr(instance, name)


def _frozen(values, dtype) -> NDArray:
    """A read-only array of ``dtype`` that no other reference can write.

    An array that is already read-only, of this dtype and owns its data is
    returned as it is (a fresh array this module froze, or an earlier
    ``_frozen`` result); anything else is copied.  The per-build copies of
    the leaf means were 142 MB each on pg17 E (perf scout F5).
    """
    if (
        isinstance(values, np.ndarray)
        and values.dtype == dtype
        and values.base is None
        and not values.flags.writeable
    ):
        return values
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _sealed(array: NDArray) -> NDArray:
    """Mark an array this module has just created, and never hands out writable, read-only."""
    array.setflags(write=False)
    return array


@dataclass(frozen=True, eq=False)
class NestedTree:
    """Forest of a nested chain: level sizes and child-to-parent pointers.

    ``sizes[j]`` is ``K_j``, the coefficient count of level ``j`` (observed and
    unobserved levels alike).  ``parent[j]`` is an ``intp`` array of length
    ``K_j``: for ``j >= 1`` the local index at level ``j - 1`` of each node's
    parent, in ``[0, K_{j-1})``; ``parent[0]`` is all ``-1`` (roots).  A node
    with no weighted row below it may point at any parent (the layout uses 0):
    its accumulated weight is exactly zero, so ``sigma = omega / D = 0`` severs
    it (§3.7, §10).  ``offsets[j]`` is the first node-order position of level
    ``j`` and ``offsets[L] == n_nodes``.

    ``subtree_sum`` is ``M'`` extended to every level and ``path_sum`` is ``M``
    (§3.1); both accept one value or a row of values per node.  Equality is
    identity; two trees describe the same forest when their sizes and parent
    arrays are equal.
    """

    sizes: tuple[int, ...]
    parent: tuple[NDArray, ...]
    offsets: NDArray = field(init=False)
    n_nodes: int = field(init=False)

    def __post_init__(self) -> None:
        sizes = tuple(int(size) for size in self.sizes)
        if not sizes or min(sizes) < 1:
            raise ValueError("A nested chain needs at least one non-empty level.")
        if len(self.parent) != len(sizes):
            raise ValueError("parent must hold one array per level.")
        parent = tuple(_frozen(values, np.intp) for values in self.parent)
        if parent[0].shape != (sizes[0],) or np.any(parent[0] != -1):
            raise ValueError("parent[0] must mark every root with -1.")
        for j in range(1, len(sizes)):
            if parent[j].shape != (sizes[j],):
                raise ValueError(f"parent[{j}] must have shape ({sizes[j]},).")
            if np.any((parent[j] < 0) | (parent[j] >= sizes[j - 1])):
                raise ValueError(f"parent[{j}] points outside level {j - 1}.")
        offsets = _frozen(np.concatenate(([0], np.cumsum(sizes))), np.intp)
        object.__setattr__(self, "sizes", sizes)
        object.__setattr__(self, "parent", parent)
        object.__setattr__(self, "offsets", offsets)
        object.__setattr__(self, "n_nodes", int(offsets[-1]))

    @property
    def depth(self) -> int:
        """Number of chain levels ``L``."""
        return len(self.sizes)

    @cached_property
    def _child_sums(self) -> tuple[scipy.sparse.csr_array, ...]:
        """Per level ``j >= 1`` the ``K_{j-1} x K_j`` child-to-parent incidence.

        Cache contract: owned by this immutable tree and living exactly as long
        as it; the parent arrays never change, so nothing invalidates it.
        """
        return tuple(
            scipy.sparse.csr_array(
                (np.ones(size), (parent, np.arange(size))),
                shape=(previous, size),
            )
            for previous, size, parent in zip(
                self.sizes[:-1], self.sizes[1:], self.parent[1:], strict=True
            )
        )

    def subtree_sum(self, leaf_values: NDArray) -> tuple[NDArray, ...]:
        """Return per level the sums of ``leaf_values`` over each node's leaves.

        ``leaf_values`` has shape ``(K_{L-1},)`` or ``(K_{L-1}, m)``; level ``j``
        of the result has shape ``(K_j,)`` or ``(K_j, m)``.  The leaf level is
        returned as given (converted to float64).
        """
        values = np.asarray(leaf_values, dtype=np.float64)
        if values.shape[:1] != (self.sizes[-1],):
            raise ValueError(f"leaf_values must have {self.sizes[-1]} rows.")
        sums = [values]
        for child_sum in reversed(self._child_sums):
            sums.append(child_sum @ sums[-1])
        return tuple(reversed(sums))

    def path_sum(self, per_level: Sequence[NDArray]) -> NDArray:
        """Return per leaf the sum of node values over its ancestors-or-self.

        ``per_level[j]`` has shape ``(K_j,)`` or ``(K_j, m)``; the result has the
        leaf level's shape.
        """
        if len(per_level) != self.depth:
            raise ValueError("per_level must hold one array per level.")
        total = np.asarray(per_level[0], dtype=np.float64)
        for parent, values in zip(self.parent[1:], per_level[1:], strict=True):
            total = total[parent] + np.asarray(values, dtype=np.float64)
        return total

    def split(self, node_values: NDArray) -> tuple[NDArray, ...]:
        """Split node-order values ``(n_nodes, ...)`` into per-level views."""
        values = np.asarray(node_values)
        if values.shape[:1] != (self.n_nodes,):
            raise ValueError(f"node_values must have {self.n_nodes} rows.")
        return tuple(values[start:stop] for start, stop in zip(self.offsets[:-1], self.offsets[1:]))


@dataclass(frozen=True, eq=False)
class IndicatorCells:
    """One sparse indicator block of a nested border: the (leaf, level) cells its rows take.

    ``block`` indexes ``small_matrices`` and ``columns`` ``(k,)`` holds the
    block's border positions, level ``j`` at ``columns[j]``.  Cells are sorted
    by leaf, then level: ``start`` ``(K + 1,)`` is each leaf's first cell and
    ``level`` ``(cells,)`` each cell's level.  ``row_cell`` ``(n,)`` is the cell
    of each row of ``leaf_order``, -1 for a row with no level (a categorical's
    base level).
    """

    block: int
    columns: NDArray
    start: NDArray
    level: NDArray
    row_cell: NDArray


def _lineage_entry(
    cache: dict, name: str, sources: tuple, size: int, build: Callable[[], Any]
) -> Any:
    """``build()`` held in a design lineage's nesting cache for these integer code arrays.

    The sources are integer code arrays (bin and level codes) and the leaf
    order, whose leaf starts come with it from one tree entry; ``build``
    reads nothing else but ``size`` (a level count), which is part of the
    key.  Keyed by the sources' ids; the entry holds the sources themselves,
    so no id is reused while it lives.  Owner: the
    lineage's nesting cache (``selection.shared_nesting_cache``), beside the
    tree and leaf order it already holds on the same terms.  Lifetime: the
    first design built on these codes and all its lambda rebuilds, which pass
    code arrays through unchanged (``rebuild_design_matrix_with_lambdas``).
    Invalidation: none; the matrix constructors own their code arrays and
    nothing writes them after construction, so a different code array is a
    different key.  No weight, lambda, basis or penalty enters an entry.
    """
    key = (name, size, *(id(source) for source in sources))
    if key not in cache:
        cache[key] = (sources, build())
    return cache[key][1]


def _indicator_cells(codes: NDArray, order: NDArray, leaf: NDArray, leaves: int, k: int) -> tuple:
    """``(start, level, row_cell)`` of ``IndicatorCells`` for one-hot ``codes`` of ``k`` levels,
    ``leaf`` the leaf of each row in leaf order and ``leaves`` their number."""
    level = codes[order]
    has = level < k
    cells, row_cell = np.unique(leaf[has] * (k + 1) + level[has], return_inverse=True)
    start = np.searchsorted(cells // (k + 1), np.arange(leaves + 1))
    every_row = np.full(len(leaf), -1, dtype=np.intp)
    every_row[has] = row_cell
    return start, cells % (k + 1), every_row


def _leaf_codes(sources: tuple, order: NDArray) -> NDArray:
    """``(n, len(sources))`` int32: each code array's values in leaf order, one column each."""
    codes = np.empty((len(order), len(sources)), dtype=np.int32)
    for column, values in enumerate(sources):
        codes[:, column] = values[order]
    return codes


@dataclass(frozen=True, eq=False)
class LeafRows:
    """The dense border rows of a nested layout in leaf order, as the row pass forms them.

    ``codes`` ``(n, t + h)`` int32 holds each row's codes in leaf order: first
    the bins of ``t`` discretized splines, whose rows are rows of their tables
    (the support ``B_unique R_inv``, back to back in ``tables`` from
    ``table_start`` with ``table_width`` columns), then the levels of ``h``
    categorical blocks of ``one_hot_width`` levels, whose rows are one-hot (a
    code past the last level is the base level, a zero row).  ``gathered``
    ``(column, matrix)`` holds dense blocks read through the leaf order,
    ``sparse`` ``(column, B[leaf_order], R_inv)`` sparse splines, and
    ``generic`` ``(column, matrix)`` any other block, through its
    ``row_subset``.  Every ``column`` is a block's first column among the
    pass's dense columns.
    """

    codes: NDArray
    tables: NDArray
    table_start: NDArray
    table_width: NDArray
    table_column: NDArray
    one_hot_width: NDArray
    one_hot_column: NDArray
    gathered: tuple = ()
    sparse: tuple = ()
    generic: tuple = ()


@dataclass(frozen=True, eq=False)
class NestedStructuredLayout:
    """Design partition for one nested chain beside a dense border (§4).

    Built once per design and chain by ``layout.build_nested_structured_layout``
    and cached on the ``DesignMatrix`` like the single-level layouts.  Chain
    fields run coarsest to finest; ``chain_group_indices[-1]`` is the leaf
    group, the one whose ``RandomEffectGroupMatrix.codes`` are the leaf codes.
    ``level_indices[j]`` holds the global design coefficient indices of level
    ``j`` (its ``GroupSlice`` range) and ``structured_indices`` their
    concatenation in node order.  ``leaf_order`` holds the rows sorted by leaf
    code (stable) and ``leaf_starts`` ``(K + 1,)`` each leaf's start in it, so
    ``leaf_order[leaf_starts[l]:leaf_starts[l + 1]]`` are the rows of leaf
    ``l`` in row order; both are read-only and shared with the nesting cache
    entry that holds the tree.  The border fields (``_border_partition``'s)
    hold every group outside the chain, crossed random effects included.

    There is deliberately no ``dominant_group_index``: code written for one
    dominant group fails on this layout instead of treating the leaf as the
    only structured level.
    """

    chain_group_indices: tuple[int, ...]
    chain_group_names: tuple[str, ...]
    tree: NestedTree
    level_indices: tuple[NDArray, ...]
    leaf_order: NDArray
    leaf_starts: NDArray
    small_group_indices: tuple[int, ...]
    small_matrices: tuple[GroupMatrix, ...]
    local_groups: tuple[GroupSlice, ...]
    small_indices: NDArray
    dense_small_matrix: NDArray | None
    small_execution_plan: MatrixExecutionPlan | None
    structured_indices: NDArray = field(init=False)
    # The chain's nesting cache (selection.shared_nesting_cache), where the
    # leaf-ordered border codes outlive a lambda rebuild (``_lineage_entry``).
    lineage_cache: dict = field(default_factory=dict, repr=False)
    # The border centre and structural null generators per prior-weight
    # vector (``moments.nested_prior_statistics``).  Owner: this layout.
    # Lifetime: the layout.  Key: the prior weights themselves (held as a
    # read-only copy and compared exactly); at most two entries.  No working
    # weight, lambda or penalty enters an entry; the border matrices never
    # change, so nothing else invalidates it.
    prior_cache: list = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        depth = self.tree.depth
        if len(self.chain_group_indices) != depth or len(self.chain_group_names) != depth:
            raise ValueError("Chain group indices and names must match the tree depth.")
        levels = tuple(_frozen(values, np.intp) for values in self.level_indices)
        if tuple(len(values) for values in levels) != self.tree.sizes:
            raise ValueError("level_indices must match the tree level sizes.")
        object.__setattr__(self, "level_indices", levels)
        object.__setattr__(self, "small_indices", _frozen(self.small_indices, np.intp))
        object.__setattr__(self, "structured_indices", _frozen(np.concatenate(levels), np.intp))
        if self.dense_small_matrix is not None:
            object.__setattr__(
                self, "dense_small_matrix", _frozen(self.dense_small_matrix, np.float64)
            )

    @property
    def leaf_group_index(self) -> int:
        return self.chain_group_indices[-1]

    @property
    def leaf_group_name(self) -> str:
        return self.chain_group_names[-1]

    @cached_property
    def sparse_indicators(self) -> tuple[IndicatorCells, ...]:
        """The border's random-effect blocks, as the (leaf, level) cells their rows take.

        A row's centred entry ``x_j - m_lj`` of a one-hot block is exactly zero
        on every level ``j`` its leaf's rows never take (the border centre is
        0 there by type), so the row pass needs only ``sum_r L(r)`` entries, ``L(r)`` the
        number of levels of the row's leaf, where the dense pass forms ``n k``.
        The rule is the block's type: every ``RandomEffectGroupMatrix`` border
        block goes through its cells and every other block through the dense
        rows (``leaf_rows``), whatever its levels' fill.  Cache contract: owned
        by this immutable layout and living as long as it; the cells come from
        the lineage cache (``_lineage_entry``), valid for the very code array
        and leaf order they were built from.
        """
        if self.dense_small_matrix is not None:
            return ()
        sizes = np.diff(self.leaf_starts)
        leaf = np.repeat(np.arange(len(sizes)), sizes)
        found, offset = [], 0
        for block, matrix in enumerate(self.small_matrices):
            k = matrix.shape[1]
            columns, offset = np.arange(offset, offset + k), offset + k
            if not isinstance(matrix, RandomEffectGroupMatrix):
                continue
            start, level, every_row = _lineage_entry(
                self.lineage_cache,
                "indicator_cells",
                (matrix.codes, self.leaf_order),
                k,
                lambda matrix=matrix, k=k: _indicator_cells(
                    matrix.codes, self.leaf_order, leaf, len(sizes), k
                ),
            )
            found.append(IndicatorCells(block, columns, start, level, every_row))
        return tuple(found)

    @cached_property
    def indicator_columns(self) -> NDArray:
        """``(q,)`` bool: the columns of the sparse indicator blocks (``sparse_indicators``).

        Cache contract: owned by this immutable layout and living as long as it.
        """
        mask = np.zeros(len(self.small_indices), dtype=bool)
        for cells in self.sparse_indicators:
            mask[cells.columns] = True
        return _frozen(mask, bool)

    @cached_property
    def indicator_pattern(self) -> tuple[NDArray, NDArray] | None:
        """``(rows, positions)`` of every (leaf, level) cell, positions among the indicator columns.

        Row-major (by leaf, then position), the order ``np.nonzero`` gives: a
        superset of the nonzeros of the leaf means' indicator block, which the
        row pass writes only on these cells (``moments._indicator_means``), so
        a product over the pattern sums the same nonzero terms plus exact
        zeros.  ``None`` without a sparse indicator block.  Cache contract:
        owned by this immutable layout and living as long as it; design-only.
        """
        if not self.sparse_indicators:
            return None
        position = np.cumsum(self.indicator_columns) - 1
        rows, positions = [], []
        for cells in self.sparse_indicators:
            counts = np.diff(cells.start)
            rows.append(np.repeat(np.arange(len(counts)), counts))
            positions.append(position[cells.columns[cells.level]])
        row, column = np.concatenate(rows), np.concatenate(positions)
        order = np.lexsort((column, row))
        return _frozen(row[order], np.intp), _frozen(column[order], np.intp)

    @cached_property
    def leaf_rows(self) -> LeafRows:
        """How the row pass forms the dense border rows in leaf order (``LeafRows``).

        The sparse indicator blocks are left out.  Blocks are told apart by
        exact type, never by their data, and each form gives the matrix's own
        ``toarray`` rows.  Cache contract: owned by this immutable layout and
        living as long as it, like the border matrices it reads, which never
        change.  A spline table depends on ``R_inv`` (``n_bins x k`` values)
        and a sparse spline's leaf-ordered copy of ``B`` (about 12 bytes per
        stored entry) on ``B``'s values, so both are built here, per layout;
        the leaf-ordered codes, integer arrays a lambda rebuild passes through
        unchanged, come from the lineage cache (``_lineage_entry``).
        """
        order = self.leaf_order
        empty = np.zeros(0, dtype=np.intp)
        if self.dense_small_matrix is not None:
            gathered = ((0, self.dense_small_matrix),)
            codes = np.zeros((len(order), 0), dtype=np.int32)
            return LeafRows(codes, np.zeros(0), empty, empty, empty, empty, empty, gathered)
        skip = {cells.block for cells in self.sparse_indicators}
        tables, one_hot, gathered, sparse, generic, column = [], [], [], [], [], 0
        for block, matrix in enumerate(self.small_matrices):
            if block in skip:
                continue
            # isinstance narrows; the exact type keeps a subclass's own rows generic
            kind = type(matrix)
            if isinstance(matrix, DiscretizedSSPGroupMatrix) and kind in _TABLE_TYPES:
                table = np.asarray(matrix.B_unique @ matrix.R_inv, dtype=np.float64)
                tables.append((column, matrix.bin_idx, table))
            elif isinstance(matrix, CategoricalGroupMatrix) and kind in _ONE_HOT_TYPES:
                one_hot.append((column, matrix.codes, matrix.shape[1]))
            elif isinstance(matrix, DenseGroupMatrix) and kind is DenseGroupMatrix:
                gathered.append((column, matrix.M))
            elif isinstance(matrix, SparseSSPGroupMatrix) and kind is SparseSSPGroupMatrix:
                basis = np.ascontiguousarray(matrix.R_inv, dtype=np.float64)
                sparse.append((column, matrix.B[order], basis))
            else:
                generic.append((column, matrix))
            column += matrix.shape[1]
        sources = tuple(values for _, values, _ in tables + one_hot)
        codes = _lineage_entry(
            self.lineage_cache,
            "leaf_codes",
            (*sources, order),
            0,
            lambda: _leaf_codes(sources, order),
        )
        return LeafRows(
            codes=codes,
            tables=np.concatenate([np.ravel(table) for *_, table in tables] + [np.zeros(0)]),
            table_start=np.cumsum([0] + [table.size for *_, table in tables])[:-1].astype(np.intp),
            table_width=np.array([table.shape[1] for *_, table in tables], dtype=np.intp),
            table_column=np.array([column for column, *_ in tables], dtype=np.intp),
            one_hot_width=np.array([width for *_, width in one_hot], dtype=np.intp),
            one_hot_column=np.array([column for column, *_ in one_hot], dtype=np.intp),
            gathered=tuple(gathered),
            sparse=tuple(sparse),
            generic=tuple(generic),
        )


@dataclass(frozen=True, eq=False)
class NestedLeafStatistics:
    """Per-leaf statistics of one row-weight vector ``a`` (§3.4, §3.6, §4).

    Produced by one row pass (``moments._nested_pass``) in the
    border coordinates of the operator that carries them (``q`` columns), on
    the rows ``x - center`` so that no statistic carries a column's offset:

    - ``weight`` ``(K,)``: ``a_leaf[l] = sum_{r in l} a_r``.
    - ``mean`` ``(K, q)``: the DATA leaf centres ``mu_l - center`` of the
      factor these statistics belong to, in the shifted form ``x_ref(l) - c
      + sum_{r in l} |w_r| ((x_r - c) - (x_ref(l) - c)) / sum_{r in l} |w_r|``
      with the ABSOLUTE data weights (signed-rows note §4.1; never the
      operator's ``a``; a leaf cut by the row pass's chunks combines its
      pieces' centres the same way about the heaviest piece), and 0 for a
      leaf without a weighted row.  For rows ``w >= 0`` these are the
      weighted means.  A column that is constant within a leaf gives that
      constant (less ``c``) exactly.  Signed operators carry the same array
      their factor was built from.
    - ``within`` ``(q, q)``: ``sum_r a_r (x_r - m_l)(x_r - m_l)'`` by the centred
      row pass, never ``X'aX - sum_l ...`` (decision 1); exactly symmetric (the
      builder canonicalises ``0.5 (W + W')``).  A column constant within every
      leaf has an exactly zero row and column.
    - ``absolute`` ``(q,)``: ``sum_r e_r (x_rj - m_lj)^2`` with ``e_r >= |a_r|``
      the scale of row ``r``'s weight error (``e = a`` for Fisher rows, the
      observed rows' ``w0 (|u^2/V| + |(y - mu) factor|)``; design §3.3): the
      within-leaf curvature the rounding of the rows can move.
    - ``center`` ``(q,)``: the global centre ``c`` of the rows, the shifted
      prior-weighted mean of every column that is not one-hot and 0 on the
      one-hot ones and on an intercept column (``moments.nested_prior_statistics``,
      design §3.2).
    - ``deviation`` ``(K, q)`` or ``None``: ``dev_l = sum_{r in l} a_r (x_r -
      m_l)``; ``None`` means exactly zero.  The data pass carries it about the
      absolute-weight centres, ``-2 sum_{r in l, w_r < 0} |w_r| (x_r - m_l)``,
      an exact zero (``None``) when no row is negative.
    - ``indicator`` ``(q,)`` bool or ``None``: the columns of the border's
      random-effect blocks (the layout's ``indicator_columns``; False on an
      intercept column), whose node means vanish off each node's levels, so
      the factor sums their between-node scatter sparsely
      (``_weighted_scatter``); ``None`` means no such column.
    - ``generators`` ``BorderGenerators`` or ``None``: the exact null basis of
      the border's data part that the specification determines (design §3.6
      step 1), which the factor deflates structurally; ``None`` on a
      W-derivative operator, which is never factored, and where the border has
      no complete one-hot block.
    - ``error_mass`` ``(K,)`` or ``None``: ``sum_{r in l} e_r``, the leaf's
      weight-error mass (``None``: ``|weight|``, rows ``w >= 0`` with ``e = w``).
    - ``rounding`` ``(gamma_weight, gamma_scatter)`` or ``None``: the row
      pass's rounding constants for the leaf sums and the within scatter
      (``moments._nested_pass``); ``None`` lets the factor charge its own
      ``gamma_Q`` for both.
    - ``indicator_pattern`` ``(rows, positions)`` or ``None``: the layout's
      (leaf, level) cells (``NestedStructuredLayout.indicator_pattern``), a
      superset of the nonzeros of ``mean``'s ``indicator`` columns, which the
      factor's root scatter reads in place of a scan of the ``(K, q)`` block
      (perf scout F2).

    ``cross`` is the raw ``C_leaf[l] = sum_{r in l} a_r x_r = a_l (m_l + c) +
    dev_l``, derived for the raw-coordinate readers.  Non-finite values raise
    ``np.linalg.LinAlgError`` (the iterate's weights caused them);
    inconsistent shapes or an asymmetric ``within`` raise ``ValueError``.
    """

    weight: NDArray
    mean: NDArray
    within: NDArray
    absolute: NDArray
    center: NDArray
    deviation: NDArray | None = None
    indicator: NDArray | None = None
    generators: BorderGenerators | None = None
    error_mass: NDArray | None = None
    rounding: tuple[float, float] | None = None
    indicator_pattern: tuple[NDArray, NDArray] | None = None

    def __post_init__(self) -> None:
        weight = _frozen(self.weight, np.float64)
        mean = _frozen(self.mean, np.float64)
        within = _frozen(self.within, np.float64)
        absolute = _frozen(self.absolute, np.float64)
        center = _frozen(self.center, np.float64)
        deviation = None if self.deviation is None else _frozen(self.deviation, np.float64)
        indicator = None if self.indicator is None else _frozen(self.indicator, bool)
        error_mass = None if self.error_mass is None else _frozen(self.error_mass, np.float64)
        if error_mass is not None and error_mass.shape != weight.shape:
            raise ValueError("Leaf error mass must match the leaf weights (K,).")
        rounding = None if self.rounding is None else tuple(float(v) for v in self.rounding)
        if rounding is not None and (len(rounding) != 2 or min(rounding) < 0.0):
            raise ValueError("Leaf rounding must be two non-negative constants.")
        if weight.ndim != 1 or mean.ndim != 2 or mean.shape[0] != len(weight):
            raise ValueError("Leaf weight and mean must have shapes (K,) and (K, q).")
        q = mean.shape[1]
        if within.shape != (q, q) or absolute.shape != (q,) or center.shape != (q,):
            raise ValueError("Within-leaf scatter, absolute mass and centre must match q.")
        if deviation is not None and deviation.shape != mean.shape:
            raise ValueError("Leaf deviation must match the mean shape (K, q).")
        if indicator is not None and indicator.shape != (q,):
            raise ValueError("Leaf indicator columns must match q.")
        if self.generators is not None and self.generators.matrix.shape[0] != q:
            raise ValueError("Border generators must match q.")
        arrays = (
            [weight, mean, within, absolute, center]
            + ([] if deviation is None else [deviation])
            + ([] if error_mass is None else [error_mass])
        )
        if not all(np.all(np.isfinite(values)) for values in arrays):
            raise np.linalg.LinAlgError("Nested leaf statistics must be finite.")
        if not np.array_equal(within, within.T):
            raise ValueError("Within-leaf scatter must be exactly symmetric.")
        if self.indicator_pattern is not None:
            rows, positions = (_frozen(values, np.intp) for values in self.indicator_pattern)
            count = 0 if indicator is None else int(np.count_nonzero(indicator))
            valid = rows.ndim == 1 and rows.shape == positions.shape
            if valid and rows.size:
                valid = bool(
                    rows.min() >= 0
                    and rows.max() < len(weight)
                    and positions.min() >= 0
                    and positions.max() < count
                )
            if not valid:
                raise ValueError(
                    "The indicator pattern must index the leaves and indicator columns."
                )
            object.__setattr__(self, "indicator_pattern", (rows, positions))
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "within", within)
        object.__setattr__(self, "absolute", absolute)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "deviation", deviation)
        object.__setattr__(self, "indicator", indicator)
        object.__setattr__(self, "error_mass", error_mass)
        object.__setattr__(self, "rounding", rounding)

    @cached_property
    def cross(self) -> NDArray:
        """Raw ``C_leaf`` ``(K, q)``: ``a_l (m_l + c) + dev_l``; cached for this immutable object."""
        cross = self.weight[:, None] * (self.mean + self.center)
        return cross if self.deviation is None else cross + self.deviation

    @property
    def width(self) -> int:
        """Border width ``q``."""
        return int(self.mean.shape[1])


def _check_partition(small_indices: NDArray, structured_indices: NDArray) -> int:
    all_indices = np.concatenate((small_indices, structured_indices))
    if not np.array_equal(np.sort(all_indices), np.arange(len(all_indices))):
        raise ValueError("Structured index partitions must cover every coefficient once.")
    return len(all_indices)


def _assemble_leaf_statistics(**fields) -> NestedLeafStatistics:
    """``NestedLeafStatistics`` from arrays already frozen and exact copies of validated ones.

    For ``NestedDataOperator.augmented`` only, whose every entry is a copy of
    a validated statistic or an exact 0 or 1: no copy and no second
    finiteness or symmetry pass over the ``(K, q)`` means (perf scout F5).
    """
    statistics = object.__new__(NestedLeafStatistics)
    for spec in dataclasses.fields(NestedLeafStatistics):
        object.__setattr__(statistics, spec.name, fields[spec.name])
    return statistics


@dataclass(frozen=True, eq=False)
class NestedDataOperator:
    """Leaf-form symmetric operator ``O = X' diag(a) X`` of a nested design (§3.6).

    ``leaf`` carries ``a_leaf``, the factor's data leaf means about the global
    centre and the row-pass scatter; ``small_indices`` ``(q,)`` and
    ``structured_indices`` ``(n_nodes,)`` (node order) partition ``0..p-1``.
    The data operator of a fit (``a = w``) is ``NestedStructuredSystem.operator``;
    signed W-derivative operators use the same class, built about the same
    leaf means.  It is one member of ``CompactSymmetricOperator`` and may be
    wrapped by ``CenteredBlockOperator`` exactly like the single-level raw
    operators.
    """

    tree: NestedTree
    leaf: NestedLeafStatistics
    small_indices: NDArray
    structured_indices: NDArray
    shape: tuple[int, int] = field(init=False)

    def __post_init__(self) -> None:
        small_indices = _frozen(self.small_indices, np.intp)
        structured_indices = _frozen(self.structured_indices, np.intp)
        q = len(small_indices)
        if self.leaf.weight.shape != (self.tree.sizes[-1],) or self.leaf.width != q:
            raise ValueError("Leaf statistics do not match the tree leaves and border width.")
        if structured_indices.shape != (self.tree.n_nodes,):
            raise ValueError("Nested operator blocks do not match the tree and border.")
        p = _check_partition(small_indices, structured_indices)
        object.__setattr__(self, "small_indices", small_indices)
        object.__setattr__(self, "structured_indices", structured_indices)
        object.__setattr__(self, "shape", (p, p))

    @cached_property
    def A(self) -> NDArray:
        """Raw border block ``X_b' diag(a) X_b`` ``(q, q)``, exactly symmetric.

        ``within + sum_l [dev_l m_l' + m_l dev_l' + a_l m_l m_l']`` with the raw
        leaf means ``m = mean + center``: the chain path never forms it from
        rows (decision 1); only the dense-reference, diagonal and estimability
        readers ask for it.  Cached for the life of this immutable operator.
        """
        leaf = self.leaf
        mean = leaf.mean + leaf.center
        A = leaf.within + mean.T @ (leaf.weight[:, None] * mean)
        if leaf.deviation is not None:
            A = A + leaf.deviation.T @ mean + mean.T @ leaf.deviation
        return _frozen(0.5 * (A + A.T), np.float64)

    def matvec(self, rhs: NDArray) -> NDArray:
        """Apply ``O`` to ``(p,)`` or ``(p, m)`` in O(n_nodes m + q^2 m).

        Tree rows: ``subtree_sum(a_leaf * (M x_tree) + C_leaf x_b)``; border
        rows: ``C_leaf' (M x_tree) + A x_b``.  Dense-reference and test use only;
        no factor method materializes ``O``.
        """
        values = np.asarray(rhs, dtype=np.float64)
        columns = values[:, None] if values.ndim == 1 else values
        if columns.ndim != 2 or columns.shape[0] != self.shape[0]:
            raise ValueError(f"rhs must have shape ({self.shape[0]},) or ({self.shape[0]}, m).")
        border = columns[self.small_indices]
        path = self.tree.path_sum(self.tree.split(columns[self.structured_indices]))
        result = np.empty_like(columns)
        result[self.structured_indices] = np.concatenate(
            self.tree.subtree_sum(self.leaf.weight[:, None] * path + self.leaf.cross @ border)
        )
        result[self.small_indices] = self.leaf.cross.T @ path + self.A @ border
        return result[:, 0] if values.ndim == 1 else result

    def diagonal(self) -> NDArray:
        """Return ``diag(O)`` ``(p,)``: subtree sums of ``a_leaf`` on the tree, ``diag(A)`` on the border.

        ``operators.compact_operator_diagonal`` delegates here for a nested raw
        operator.
        """
        diagonal = np.empty(self.shape[0])
        diagonal[self.structured_indices] = np.concatenate(self.tree.subtree_sum(self.leaf.weight))
        diagonal[self.small_indices] = np.diag(self.A)
        return diagonal

    def augmented(self) -> NestedDataOperator:
        """Return the same operator on the coordinates ``[1, X]``, intercept first.

        The result has ``weight`` unchanged, ``mean = [1 | mean]`` (1 for every
        leaf), ``center = [0 | center]``, ``within = [[0, 0], [0, within]]``,
        ``absolute = [0 | absolute]``, ``deviation = None`` or ``[0 |
        deviation]``, ``indicator = None`` or ``[False | indicator]``,
        ``generators`` shifted by the intercept column,
        ``small_indices = [0, small_indices + 1]`` and ``structured_indices +
        1``, and shares ``tree``.  Every new entry is a
        copy or an exact zero or one: the intercept column of an operator whose
        rows all carry ``x_0 = 1`` has zero within-leaf deviation.
        """
        leaf = self.leaf
        q, leaves = leaf.width, len(leaf.weight)
        within = np.zeros((q + 1, q + 1))
        within[1:, 1:] = leaf.within
        deviation = leaf.deviation
        if deviation is not None:
            deviation = _sealed(np.column_stack((np.zeros(leaves), deviation)))
        # Every array is fresh (sealed, so the statistics keep it without a
        # copy) or already the source leaf's frozen array, and every entry is
        # a copy of validated statistics or an exact 0 or 1: the statistics
        # are assembled without repeating the finiteness and symmetry pass.
        augmented = _assemble_leaf_statistics(
            weight=leaf.weight,
            mean=_sealed(np.column_stack((np.ones(leaves), leaf.mean))),
            within=_sealed(within),
            absolute=_sealed(np.concatenate(([0.0], leaf.absolute))),
            center=_sealed(np.concatenate(([0.0], leaf.center))),
            deviation=deviation,
            indicator=None if leaf.indicator is None else _sealed(np.r_[False, leaf.indicator]),
            generators=None if leaf.generators is None else leaf.generators.shifted(1),
            error_mass=leaf.error_mass,
            rounding=leaf.rounding,
            indicator_pattern=leaf.indicator_pattern,
        )
        return NestedDataOperator(
            tree=self.tree,
            leaf=augmented,
            small_indices=np.concatenate(([0], self.small_indices + 1)),
            structured_indices=self.structured_indices + 1,
        )


@dataclass(frozen=True, eq=False)
class NestedPenalizedOperator:
    """``H = O_w + blockdiag(diag(lambda_node), S_b)`` for one nested system.

    ``node_penalty[j]`` ``(K_j,)`` holds ``lambda_u`` for every node of level
    ``j`` (per-node values admit an authoritative ``S_override`` diagonal,
    §3.7); every entry is finite and non-negative.  ``border_penalty``
    ``(q, q)`` is ``S_b`` in the border coordinates of ``data``, exactly
    symmetric.  Built by ``assembly.build_penalized_nested_operator``; the
    factor never adds ``S_b`` to ``A`` (it enters ``Q`` as its own PSD term,
    §3.4).  Negative, non-finite or misshapen penalties raise ``ValueError``.
    """

    data: NestedDataOperator
    node_penalty: tuple[NDArray, ...]
    border_penalty: NDArray
    shape: tuple[int, int] = field(init=False)

    def __post_init__(self) -> None:
        tree = self.data.tree
        node_penalty = tuple(_frozen(values, np.float64) for values in self.node_penalty)
        border_penalty = _frozen(self.border_penalty, np.float64)
        q = len(self.data.small_indices)
        if tuple(values.shape for values in node_penalty) != tuple((k,) for k in tree.sizes):
            raise ValueError("node_penalty must hold one (K_j,) array per level.")
        if border_penalty.shape != (q, q) or not np.array_equal(border_penalty, border_penalty.T):
            raise ValueError("border_penalty must be an exactly symmetric (q, q) matrix.")
        penalties = [*node_penalty, border_penalty]
        if not all(np.all(np.isfinite(values)) for values in penalties):
            raise ValueError("Nested penalties must be finite.")
        if any(np.any(values < 0.0) for values in node_penalty):
            raise ValueError("Nested node penalties must be non-negative.")
        object.__setattr__(self, "node_penalty", node_penalty)
        object.__setattr__(self, "border_penalty", border_penalty)
        object.__setattr__(self, "shape", self.data.shape)

    @property
    def tree(self) -> NestedTree:
        return self.data.tree

    @property
    def small_indices(self) -> NDArray:
        return self.data.small_indices

    @property
    def structured_indices(self) -> NDArray:
        return self.data.structured_indices

    def matvec(self, rhs: NDArray) -> NDArray:
        """Apply ``H`` to ``(p,)`` or ``(p, m)``: ``data.matvec`` plus the penalty."""
        values = np.asarray(rhs, dtype=np.float64)
        result = self.data.matvec(values)
        structured = self.structured_indices
        result[structured] += (np.concatenate(self.node_penalty) * values[structured].T).T
        result[self.small_indices] += self.border_penalty @ values[self.small_indices]
        return result

    def augmented(self) -> NestedPenalizedOperator:
        """Return ``H`` on ``[1, X]``: ``data.augmented()``, ``node_penalty``
        unchanged and ``border_penalty`` padded with an exactly zero intercept
        row and column."""
        q = len(self.small_indices)
        border_penalty = np.zeros((q + 1, q + 1))
        border_penalty[1:, 1:] = self.border_penalty
        return NestedPenalizedOperator(
            data=self.data.augmented(),
            node_penalty=self.node_penalty,
            border_penalty=border_penalty,
        )


def _to_parent(tree: NestedTree, level: int, values: NDArray) -> NDArray:
    """Sum the node values (rows) of ``level`` into their parents at ``level - 1``."""
    return tree._child_sums[level - 1] @ values


def _divide_rows(values: NDArray, weight: NDArray) -> NDArray:
    """Divide the rows of ``values`` ``(K, q)`` by ``weight`` ``(K,)``, exactly zero where it is zero."""
    return np.divide(values.T, weight, out=np.zeros(values.T.shape), where=weight != 0.0).T


def _weighted_scatter(
    values: NDArray,
    weight: NDArray,
    indicator: NDArray | None,
    center: NDArray | None = None,
    pattern: tuple[NDArray, NDArray] | None = None,
) -> tuple[NDArray, NDArray]:
    """``X' diag(weight) X`` ``(q, q)`` and ``|weight|' X**2`` ``(q,)`` for ``X = values - center``.

    ``center`` (``None`` for 0) must be 0 on the ``indicator`` columns, which
    keep their exact zeros.  The node means of a random-effect border column
    vanish on every node whose subtree lacks its level, so those columns
    (``indicator``, chosen by the block's type) go through sparse products on
    their exact nonzeros and the rest through BLAS; with no such column the
    product is the dense one.  Only exact zeros are skipped, so every entry
    sums the same nonzero terms as the dense product.  The centring touches
    only the dense columns' slice, never a copy of the whole ``(K, q)`` block.
    The second result is the absolute mass of each column, the diagonal of
    the same sum with ``|weight|``, for the componentwise rounding bound.
    ``pattern`` ``(rows, positions)``, row-major, is a superset of the
    indicator block's nonzeros known by construction (the leaf means' cells,
    ``NestedLeafStatistics.indicator_pattern``): it replaces the scan for
    them, and its extra entries hold exact zeros, which add nothing to any
    sum.
    """
    magnitude = np.abs(weight)
    if indicator is None or not indicator.any():
        centred = values if center is None else values - center
        return (weight[:, None] * centred).T @ centred, magnitude @ centred**2
    dense, sparse_columns = ~indicator, np.flatnonzero(indicator)
    if pattern is not None:
        rows, columns = pattern
    else:
        # a boolean mask, never a float copy of the (K, q) indicator block
        rows, columns = np.nonzero((values != 0.0)[:, indicator])
    entries, shape = values[rows, sparse_columns[columns]], (len(values), len(sparse_columns))
    sparse = scipy.sparse.csr_array((entries, (rows, columns)), shape=shape)
    weighted_sparse = scipy.sparse.csr_array((weight[rows] * entries, (rows, columns)), shape)
    dense_values = values[:, dense] if center is None else values[:, dense] - center[dense]
    weighted = weight[:, None] * dense_values
    scatter = np.empty((values.shape[1],) * 2)
    scatter[np.ix_(dense, dense)] = weighted.T @ dense_values
    cross = sparse.T @ weighted
    scatter[np.ix_(indicator, dense)] = cross
    scatter[np.ix_(dense, indicator)] = cross.T
    scatter[np.ix_(indicator, indicator)] = (sparse.T @ weighted_sparse).toarray()
    mass = np.empty(values.shape[1])
    mass[dense] = magnitude @ dense_values**2
    mass[indicator] = np.bincount(
        columns, weights=magnitude[rows] * entries**2, minlength=len(sparse_columns)
    )
    return scatter, mass


def _carried_deviation(
    tree: NestedTree,
    level: int,
    s: NDArray,
    rho: NDArray,
    d: NDArray,
    deviation: NDArray | None,
) -> NDArray | None:
    """The parents' carried deviation ``delta_p = sum_c (s_c d_c + rho_c delta_c)`` (note §4.2).

    ``sum_c |s_c| d_c = 0`` about the |s|-weighted parent centre, so the first
    sum is ``2 sum_{c: s_c < 0} s_c d_c``: an exact zero (``None``, with no
    child deviation) when no child weight is negative, never a cancellation.
    """
    negative = np.flatnonzero(s < 0.0)
    if not negative.size and deviation is None:
        return None
    values = np.zeros_like(d) if deviation is None else rho[:, None] * deviation
    if negative.size:
        values[negative] += 2.0 * s[negative, None] * d[negative]
    return _to_parent(tree, level, values)


def _carried_terms(
    d: NDArray, delta: NDArray, rho: NDArray, pivot: NDArray
) -> tuple[NDArray, NDArray]:
    """One level's carried terms of ``Q``, ``sum_u [rho_u (d_u delta_u' + delta_u d_u') -
    delta_u delta_u' / D_u]`` (note §4.3), and the diagonal majorant of their rounding.

    Only the nodes with a nonzero deviation contribute (exact zeros skipped).
    The majorant, before ``gamma_Q``, is ``sum_u [|rho_u| r2(d_u, delta_u) +
    delta_u^2 / D_u]`` with ``r2`` the balanced rank-two majorant
    (``_balanced_rank_two``).
    """
    width = d.shape[1]
    rows = np.flatnonzero(np.any(delta != 0.0, axis=1))
    if not rows.size:
        return np.zeros((width, width)), np.zeros(width)
    X, Y, r, D = d[rows], delta[rows], rho[rows], pivot[rows]
    cross = (r[:, None] * X).T @ Y
    term = cross + cross.T - (Y / D[:, None]).T @ Y
    mass = np.abs(r) @ _balanced_rank_two(X, Y) + (1.0 / D) @ (Y * Y)
    return term, mass


def _balanced_rank_two(a: NDArray, b: NDArray) -> NDArray:
    """Per-row diagonal majorants ``a_j^2 t + b_j^2 / t`` of ``|a b' + b a'|`` ``(K, q)``.

    ``|a_i b_j + b_i a_j| <= sqrt((a_i^2 t + b_i^2 / t)(a_j^2 t + b_j^2 / t))`` for
    any ``t > 0`` (Cauchy-Schwarz); ``t = ||b|| / ||a||`` per row balances the
    two, and a row with ``a = 0`` or ``b = 0`` is an exact zero.  Summed over
    rows the majorants keep ``|E_ij| <= sqrt(U_i U_j)`` (Cauchy-Schwarz again),
    which is what the border factorization's ``u_s`` needs (design §3.6).
    """
    size_a = np.sqrt(np.einsum("ij,ij->i", a, a))
    size_b = np.sqrt(np.einsum("ij,ij->i", b, b))
    live = (size_a > 0.0) & (size_b > 0.0)
    t = np.divide(size_b, size_a, out=np.zeros_like(size_a), where=live)
    inverse = np.divide(1.0, t, out=np.zeros_like(t), where=live)
    return a * a * t[:, None] + b * b * inverse[:, None]


def _profiled_square_sums(weight: NDArray, e: NDArray, center_effective: NDArray) -> NDArray:
    """``sum_u weight_u e~_uj^2`` bounded above, ``e~_u = e_u[1:] - c_eff e_u[0]``.

    ``(a - b)^2 <= 2 a^2 + 2 b^2`` per entry: ``2 (sum_u weight_u e_uj^2 +
    c_eff_j^2 sum_u weight_u e_u0^2)``, one pass over ``e`` with no ``(K, q)``
    temporary (``weight >= 0``).
    """
    squares = np.einsum("u,uj,uj->j", weight, e[:, 1:], e[:, 1:])
    intercept = float(np.einsum("u,u,u->", weight, e[:, 0], e[:, 0]))
    return 2.0 * (squares + center_effective**2 * intercept)


def _border_bound(
    *,
    tree: NestedTree,
    leaf: NestedLeafStatistics,
    error_mass: NDArray,
    gammas: tuple[float, float, float],
    e: Sequence[NDArray],
    center_effective: NDArray,
    rhos: Sequence[NDArray],
    carried: Sequence[NDArray | None],
    local: Sequence[NDArray | None],
    root_local: float,
    super_deviation: NDArray,
    super_carried: NDArray | None,
    D0: float,
    accumulation: NDArray,
    node_mass: NDArray,
) -> NDArray:
    """The running bound ``U`` ``(q - 1,)`` of the profiled border Schur complement.

    Signed-rows note §4.4 and design §3.6: ``|dQ_ij| <= sqrt(U_i U_j)`` to first
    order, every rounding charged where it is made with the exact first-order
    sensitivity of the profiled ``Q``.  The sensitivity of ``Q`` to a node's
    weight ``omega_u`` is ``e~_u e~_u'`` and to its carried deviation ``delta_u``
    is ``Delta e~_u' + e~_u Delta'`` (the node's leaf-like form after its subtree
    is eliminated: ``m_u - (MF)_u = e_u``), with ``e~_u = e_u[1:] - c_eff
    e_u[0]`` the profiled vector (``e~_0 = -delta_0 / D_0`` at the super-root).
    So, with ``gamma_Q = (n_nodes + q + 10) eps`` and the row pass's
    ``gamma_weight``, ``gamma_scatter`` (``moments._nested_pass``):

    - the accumulation of ``Q``: ``gamma_Q (sum_u |s_u| d_u^2 + the one-hot
      root form + sum_u [|rho_u| r2(d_u, delta_u) + delta_u^2 / D_u] +
      delta_0^2 / D_0 + |S_jj|)`` (``accumulation``);
    - the leaf rows: ``omega_l`` within ``gamma_weight sum_r e_r``, ``delta_l``
      within ``gamma_weight sum_r e_r |x_r - mu_l|`` and the within scatter
      within ``gamma_scatter sum_r e_r (x_r - mu_l)^2``, which by ``2ab <= a^2
      + b^2`` per row give ``(gamma_scatter + gamma_weight) absolute + 2
      gamma_weight sum_l E_l e~_l^2`` (``E_l = error_mass``);
    - every internal node and the super-root: the rounding of ``omega_p =
      sum_c s_c``, ``l_p = gamma_Q sum_c |s_c|`` (``local``), charged ``l_p
      e~_p^2``, and of ``delta_p``, componentwise within ``gamma_Q sum_c
      (|s_c| |d_c| + |rho_c| |delta_c|)``.  A rank-two term ``D e~' + e~ D'``
      with ``|D_j| <= eta_j`` is majorized by ``eta_j^2 / t + t e~_j^2`` for any
      ``t > 0`` (``_balanced_rank_two``); with ``t = l_p`` and, by
      Cauchy-Schwarz over the children, ``(sum_c |s_c| |d_cj|)^2 <= (sum_c
      |s_c|) (sum_c |s_c| d_cj^2)``, the ``|s| |d|`` part is at most ``gamma_Q
      sum_c |s_c| d_cj^2 + l_p e~_pj^2``: the node masses already summed
      (``node_mass``) and the weight term again.  The ``|rho| |delta|`` part
      is formed where a child carries a deviation (signed rows only).

    Every ``e~^2`` sum is bounded by ``_profiled_square_sums``, so the bound
    streams once over each level's ``e`` and forms no ``(K, q)`` temporary.
    For rows ``w >= 0`` (``delta = 0``) it is ``gamma_Q (accumulation +
    node_mass) + (gamma_scatter + gamma_weight) absolute + 2 gamma_weight sum
    E e~^2 + 2 sum_p l_p e~_p^2``, within a small constant of the non-negative
    construction's ``gamma_Q (absolute + sum |s| d^2)``.
    """
    gamma_Q, gamma_weight, gamma_scatter = gammas
    depth = tree.depth
    bound = gamma_Q * (accumulation + node_mass)
    bound = bound + (gamma_scatter + gamma_weight) * leaf.absolute[1:]
    bound = bound + 2.0 * gamma_weight * _profiled_square_sums(error_mass, e[-1], center_effective)
    for level in range(depth - 1):
        # node p at ``level`` with its children at ``level + 1``
        weight_local = local[level]
        assert weight_local is not None
        bound = bound + 2.0 * _profiled_square_sums(weight_local, e[level], center_effective)
        child_deviation = carried[level + 1]
        if child_deviation is not None:
            # the |rho| |delta| part of delta_p's rounding, exactly balanced
            eta = gamma_Q * _to_parent(
                tree, level + 1, np.abs(rhos[level + 1])[:, None] * np.abs(child_deviation[:, 1:])
            )
            node_e = e[level][:, 1:] - e[level][:, :1] * center_effective
            bound = bound + np.sum(_balanced_rank_two(node_e, eta), axis=0)
    super_e = -super_deviation / D0
    bound = bound + 2.0 * root_local * super_e**2
    if super_carried is not None:
        super_eta = gamma_Q * (np.abs(rhos[0]) @ np.abs(super_carried))
        bound = bound + _balanced_rank_two(super_e[None, :], super_eta[None, :])[0]
    return bound


def _descendant_products(rows: NDArray) -> NDArray:
    """Row ``i`` of the result is the product of the rows strictly after ``i`` (1 for the last)."""
    products = np.ones_like(rows)
    if len(rows) > 1:
        products[:-1] = np.cumprod(rows[:0:-1], axis=0)[::-1]
    return products


def _low_rank_trace(solve: Callable[[NDArray], NDArray], piece: LowRankSymmetricOperator) -> float:
    """Return ``tr(H^-1 U R U')`` through the solve ``H^-1 U``."""
    return float(np.sum(solve(piece.basis) * (piece.basis @ piece.core)))


def _low_rank_diagonal(
    solve: Callable[[NDArray], NDArray], piece: LowRankSymmetricOperator
) -> NDArray:
    """Return ``diag(H^-1 U R U')`` through the solve ``H^-1 U``."""
    return np.sum(solve(piece.basis) * (piece.basis @ piece.core), axis=1)


def _square_diagonal_by_columns(
    solve: Callable[[NDArray], NDArray],
    matvec: Callable[[NDArray], NDArray],
    width: int,
    block: int = 256,
) -> NDArray:
    """Return ``diag((H^-1 O)^2)`` from explicit columns, ``sum_v (O H^-1)_vi (H^-1 O)_vi``.

    O(p (n_nodes q + q^2)) in all: the route for an operator other than the
    factor's own data (decision 3), which no production caller passes; its
    accuracy is the forward error of the solves.
    """
    diagonal = np.empty(width)
    for start in range(0, width, block):
        stop = min(start + block, width)
        unit = np.zeros((width, stop - start))
        unit[np.arange(start, stop), np.arange(stop - start)] = 1.0
        diagonal[start:stop] = np.sum(solve(matvec(unit)) * matvec(solve(unit)), axis=0)
    return diagonal


@dataclass(frozen=True)
class _Piece:
    """Row-pass quantities of a leaf-form operator about the factor's leaf means (§3.6).

    ``a`` ``(K,)`` are the leaf weights, ``V = dev + a (.) e`` ``(K, q)`` and
    ``UOU = W_a + dev'e + e'dev + sum_l a_l e_l e_l'`` ``(q, q)``, exactly
    symmetric.  Pieces of one direction add.
    """

    a: NDArray
    V: NDArray
    UOU: NDArray

    def __add__(self, other: _Piece) -> _Piece:
        return _Piece(self.a + other.a, self.V + other.V, self.UOU + other.UOU)


@dataclass(frozen=True)
class _Prepared:
    """Per-direction products every pair reads (§3.6).

    ``zhat = M T^-1 M'V`` ``(K, q)``, ``VQ = V Q^-1``, ``P = Q^-1 (U'OU) Q^-1``,
    ``weight`` the ``A_x`` accumulation ``A_x = sum_c rho_c^2 A_c`` per level
    with ``a`` at the leaves, and ``level[I] = tr(H^-1 E_I H^-1 O)`` for every
    chain level, formed once from ``W = T^-1 M'V`` so that ``W`` need not stay.
    """

    zhat: NDArray
    VQ: NDArray
    P: NDArray
    weight: tuple[NDArray, ...]
    level: NDArray


@dataclass(frozen=True)
class _Direction:
    """One direction ``scale Omega + dH`` split into what each route reads.

    ``kind`` is ``"level"`` (``key`` the chain level), ``"border"`` (``key``
    the pair ``(J, Omega_JJ)`` of the component's border positions and its
    penalty block, ``None`` for an identity) or ``None``;
    ``piece`` merges every leaf-form part of ``dH``; ``operators`` keeps those
    parts for products with low-rank columns and ``low_rank`` the low-rank
    parts, which go through solves.
    """

    component: PenaltyComponent | None
    scale: float
    kind: str | None
    key: int | tuple | None
    piece: _Piece | None
    operators: tuple
    low_rank: tuple[LowRankSymmetricOperator, ...]

    def apply(self, values: NDArray, *, full: bool) -> NDArray:
        """Return ``(scale Omega + dH) values`` ``(p, r)``, the low-rank parts only when ``full``."""
        result = np.zeros_like(values)
        if self.component is not None:
            result += _penalty_product(self.component, self.scale, values)
        for operator in self.operators + (self.low_rank if full else ()):
            result += operator.matvec(values)
        return result


def _times_penalty(block: NDArray, omega: NDArray | None) -> NDArray:
    """``block @ Omega_JJ`` for a border penalty block, ``block`` itself for an identity."""
    return block if omega is None else block @ omega


def _penalty_trace(matrix: NDArray, key: tuple[NDArray, NDArray | None]) -> float:
    """``tr(matrix Omega)`` for a border penalty ``(J, Omega_JJ)``, reading ``matrix[J, J]`` only."""
    J, omega = key
    block = matrix[np.ix_(J, J)]
    return float(np.trace(block) if omega is None else np.sum(block * omega))


def _low_rank_terms(
    solve: Callable[[NDArray], NDArray], left: _Direction, right: _Direction
) -> float:
    """Return ``tr(H^-1 L_l H^-1 O_r) + tr(H^-1 X_l H^-1 L_r)`` for the low-rank parts ``L``.

    ``O = X + L`` with ``X`` the penalty and leaf-form parts.  The pair
    ``L_l, L_r`` is counted once, inside the first term, so with ``left is
    right`` the two loops give ``2 tr(H^-1 X H^-1 L) + tr(H^-1 L H^-1 L)``.
    """
    value = 0.0
    for piece in left.low_rank:
        columns = solve(piece.basis)
        value += float(np.sum((columns @ piece.core) * right.apply(columns, full=True)))
    for piece in right.low_rank:
        columns = solve(piece.basis)
        value += float(np.sum((columns @ piece.core) * left.apply(columns, full=False)))
    return value


def _low_rank_matrix(solve: Callable[[NDArray], NDArray], records: Sequence[_Direction]) -> NDArray:
    """Return the ``(m, m)`` low-rank cross terms of ``records``, zero without low-rank parts."""
    traces = np.zeros((len(records), len(records)))
    if any(record.low_rank for record in records):
        for i, j in itertools.combinations_with_replacement(range(len(records)), 2):
            traces[i, j] = traces[j, i] = _low_rank_terms(solve, records[i], records[j])
    return traces


def _super_root_inverse(rest_inverse: NDArray, center_star: NDArray, D0: float) -> NDArray:
    """``L0^-T diag(1 / D_0, rest_inverse) L0^-1`` with ``L0 = [[1, 0], [c*, I]]`` (§3.1)."""
    column = rest_inverse @ center_star
    size = len(center_star) + 1
    inverse = np.empty((size, size))
    inverse[0, 0] = 1.0 / D0 + float(center_star @ column)
    inverse[0, 1:] = inverse[1:, 0] = -column
    inverse[1:, 1:] = rest_inverse
    return inverse


class NestedSchurFactor:
    """Factorization of a nested chain beside a dense border (§3.2-§3.7; one-engine design §3.1, §3.6).

    ``H = [[T, C], [C', A]]`` with the tree block eliminated leaves first by the
    closed-form pivot recursion (no fill; Lean ``chain_ldl``, ``chain_det``),
    then the intercept as the tree's super-root (design §3.1): border column 0
    is the unpenalized parent of every root, with pivot ``D_0 = sum_r s_r``
    and between-child term the ``s``-weighted scatter of the root means about
    their centre ``c*``.  The rows may have any sign (design §3.3, signed-rows
    note §4): every node is centred on the ``|s|``-weighted (at the leaves
    ``|w|``-weighted) shifted mean of its children, and the deviation
    ``delta_u = sum a (x - m_u)`` that centre leaves, an exact zero for rows
    ``w >= 0``, is carried up the tree, ``delta_p = sum_c (s_c d_c + rho_c
    delta_c)``.  What remains is the intercept-profiled border Schur
    complement,

        Q = S_b + W_in + sum_u [s_u d_u d_u' + rho_u (d_u delta_u' + delta_u d_u')
            - delta_u delta_u' / D_u] - delta_0 delta_0' / D_0,

    ``d_u = m_u - m_p(u)`` (``m_root - c*`` at a root), from shifted means
    (§3.4): dense columns about the heaviest root, one-hot columns as ``sum s
    m m' - D_0 c* c*'`` on their sparse means with ``c* = sum_r s_r m_r /
    D_0``; for rows ``w >= 0`` every carried term is an exact zero and ``Q`` is
    the PSD sum.  ``sigma = omega / D`` (never ``1 - rho``); ``F`` and ``M F``
    by the top-down ``g/e`` recursion with the carried deviation, ``F_u =
    sigma_u g_u + delta_u / D_u``, ``e_u = rho_u g_u - delta_u / D_u`` (never
    ``c/D - sigma A``); ``H^-1`` on the pattern from the
    Takahashi scalars ``t, Z_uu, v, kappa`` (§3.5).  ``Q`` is factored by
    ``border.factor_border``: structural deflation of the exact null space
    the specification determines (``leaf.generators``), complete pivoting,
    Rump's verification of the retained block, a residual bound on the
    truncated subspace and disclosure of every null direction
    (``border_certificate``); every rank decision is taken there, on the
    Jacobi-scaled deflated matrix, never on the intercept.  The running bound
    ``U`` (``_border_bound``; signed-rows note §4.4) charges every rounding of
    the construction where it is made, with ``gamma_Q = (n_nodes + q + 10)
    eps`` and the row pass's constants, and with each row's weight-error scale
    ``e_r`` (``leaf.error_mass``) rather than ``|w_r|``: the rows' builder
    states where their weights come from a cancellation.  Non-negative Fisher
    weights (``e = w``) are charged componentwise, so only an exactly zero
    pivot is null and a tiny positive weight keeps its column.  The
    generalized inverse is ``Q^+ = L0^-T diag(1 / D_0, Q_rest^+) L0^-1``, ``L0
    = [[1, 0], [c_eff, I]]`` with the super-root's multiplier ``c_eff = c* +
    delta_0 / D_0``,
    with ``Q_rest^+`` the Moore-Penrose inverse in the scaled metric on the
    retained subspace; every null of ``Q`` then has the form ``[-c*'z; z]``
    and ``Q^+ Q e_0 = e_0``: the intercept is H-orthogonal to every null, the
    dense backend's weighted-mean-centred convention.  ``log|H| = sum_u log
    D_u + log D_0 + log pdet(Q_rest)`` is published as is, the convention
    identity ``det(T) pdet(Q) = sum_w pdet(H_c)``; no coupling term is added
    and no null space is refused for touching the tree.

    Construction takes a ``NestedPenalizedOperator`` on the coordinates ``[1,
    X]`` (``assembly.build_augmented_nested_factor``) and ``intercept=True``
    (``intercept=False``, the retired raw-coordinate coefficient factor, is a
    ``ValueError``).  It works in the centred coordinates ``[1, X - 1 c']`` of
    its leaf statistics (``c = leaf.center``, 0 on the intercept): ``Q``,
    ``F``, ``e`` and every closed form are free of the columns' offsets.  The
    change of basis ``R = I - e_0 c'`` touches only the intercept row, so the
    public methods return raw-coordinate values: ``solve`` maps ``R' r`` in
    and ``R x`` out, the intercept entry of ``diag(H^-1)`` is ``(e_0 - c)'
    Q^-1 (e_0 - c)``, ``inverse_operator_diagonal`` and
    ``coefficient_estimable`` add their intercept-column terms, and traces,
    ``diag(H^+ H)`` and ``logdet`` are invariant.

    Refusals at construction: ``LinAlgError`` when a tree pivot fails its
    certificate ``D_u > E_u + 4 eps (|omega_u| + lambda_u)``, ``E_u`` the
    running bound on its rounding (``E_leaf = gamma_weight sum_r e_r``,
    ``E_p = sum_c rho_c^2 E_c + gamma_Q sum_c |s_c|``; note §4.4, item 1): it
    never fires for ``w >= 0``, since then ``D_u >= lambda_u`` and ``E_u`` is
    a rounding of ``omega_u``, and it is the certificate that signed observed
    rows define a positive definite ``H`` there (design §6: a refused iterate,
    never another route); for a node with ``lambda_u = 0`` and ``omega_u =
    0``, a super-root pivot ``D_0`` within its own bound (for ``w >= 0``,
    ``D_0 == 0`` exactly: intercept aliasing), and the border refusals of
    ``factor_border`` (material negative curvature, which ``w >= 0`` cannot
    produce).

    Attributes read by callers (all set at construction):
    ``shape``, ``backend = "structured"``, ``rank`` (``n_nodes + rank(Q)``),
    ``rank_truncated``, ``schur_condition_estimate``
    (the retained squared-pivot ratio of the scaled border, ``inf`` when
    anything was truncated), ``minimum_local_diagonal`` (``min_u D_u``),
    ``border_certificate``, ``small_indices``, ``structured_indices`` (node
    order), ``max_structured_inverse_block``, ``chain_group_indices``,
    ``chain_group_names``, ``dominant_group_name`` (the leaf name, for
    reporting only: never use it to find a layout), ``intercept``,
    ``operator`` (the ``NestedPenalizedOperator`` it factors).

    Operators accepted by the operator methods, in this factor's coordinates:
    ``NestedDataOperator`` on the same forest (the same tree object, or equal
    sizes and parents), the same partitions and bitwise the same
    ``leaf.mean`` and ``leaf.center`` (else ``ValueError``);
    ``LowRankSymmetricOperator`` (handled
    by solves); and ``SumBlockOperator`` of those.  Anything else, including
    ``CenteredBlockOperator`` and the single-level operators, raises
    ``TypeError``.  ``operator.data`` itself is recognised by identity as the
    factor's own data operator (``H - S``).

    Caches (owner: the factor; lifetime: the factor; invalidation: none,
    because weights, penalties, basis, target and precision are all fixed at
    construction and any change builds a new factor): ``Q^-1``, ``diag(H^-1)``,
    the scaled border eigenvalues and, per chain level on first use, ``G_I =
    F_I' F_I``, ``h^I`` and the level-pair traces ``t1 + t2 + t3``.  ``T^-1 E_I
    F`` is recomputed per pair (one O(n_nodes q) tree solve).  They must not
    change a rank decision or a bound.
    """

    backend = "structured"
    shape: tuple[int, int]
    rank: int
    rank_truncated: bool
    schur_condition_estimate: float
    minimum_local_diagonal: float
    small_indices: NDArray
    structured_indices: NDArray
    max_structured_inverse_block: int
    chain_group_indices: tuple[int, ...]
    chain_group_names: tuple[str, ...]
    dominant_group_name: str
    intercept: bool
    operator: NestedPenalizedOperator

    def __init__(
        self,
        operator: NestedPenalizedOperator,
        *,
        chain_group_names: tuple[str, ...],
        chain_group_indices: tuple[int, ...],
        intercept: bool,
        max_structured_inverse_block: int = 256,
        excluded: tuple[int, ...] = (),
    ):
        # The whole construction under one BLAS thread when a wide fit released
        # the cap and the border is below the break-even: its dense kernels act
        # on the border width, and numpy's and scipy's OpenBLAS pools otherwise
        # contend between a GEMM and the next LAPACK call (perf scout F1:
        # dpotrf 140 ms after a threaded GEMM against 6 ms alone; the whole
        # construction 285 ms against 326 ms for the border kernels alone).
        with narrow_kernel_blas_threads(operator.data.leaf.width):
            self._construct(
                operator,
                chain_group_names=chain_group_names,
                chain_group_indices=chain_group_indices,
                intercept=intercept,
                max_structured_inverse_block=max_structured_inverse_block,
                excluded=excluded,
            )

    def _construct(
        self,
        operator: NestedPenalizedOperator,
        *,
        chain_group_names: tuple[str, ...],
        chain_group_indices: tuple[int, ...],
        intercept: bool,
        max_structured_inverse_block: int,
        excluded: tuple[int, ...],
    ) -> None:
        data = operator.data
        tree, leaf = data.tree, data.leaf
        depth, q = tree.depth, leaf.width
        names = tuple(chain_group_names)
        if len(names) != depth or len(chain_group_indices) != depth:
            raise ValueError("Chain group names and indices must match the tree depth.")
        if not intercept:
            raise ValueError(
                "The raw-coordinate nested factor (intercept=False) was retired: the "
                "slope covariance is the slope block of the augmented factor."
            )
        if q == 0:
            raise ValueError("An intercept factor needs border column 0.")
        self.operator = operator
        self.chain_group_names = names
        self.chain_group_indices = tuple(int(index) for index in chain_group_indices)
        self.dominant_group_name = names[-1]
        self.intercept = bool(intercept)
        self.max_structured_inverse_block = int(max_structured_inverse_block)
        self.shape = operator.shape
        self.small_indices = data.small_indices
        self.structured_indices = data.structured_indices
        self._tree, self._leaf, self._penalty = tree, leaf, operator.node_penalty
        p = self.shape[0]
        self._small_position = np.full(p, -1, dtype=np.intp)
        self._small_position[self.small_indices] = np.arange(q)
        self._node_position = np.full(p, -1, dtype=np.intp)
        self._node_position[self.structured_indices] = np.arange(tree.n_nodes)
        eps = np.finfo(np.float64).eps
        # The factor's coordinates: centred about leaf.center, with the
        # intercept to absorb the centre.  R = I - e_0 c' maps them.
        self._mean = leaf.mean
        self._center = leaf.center

        # Up pass, leaves first (§3.3; signed-rows note §4.2): pivots,
        # multipliers, shrunk weights, the |s|-weighted shifted node centres
        # and the carried deviation delta, with the running bound E_u on each
        # pivot's rounding (note §4.4).  For rows w >= 0 every delta is an
        # exact zero and |s| = s, so this is the non-negative recursion bit
        # for bit.
        gamma_Q = (tree.n_nodes + q + 10) * eps
        gamma_weight, gamma_scatter = (gamma_Q, gamma_Q) if leaf.rounding is None else leaf.rounding
        error_mass = np.abs(leaf.weight) if leaf.error_mass is None else leaf.error_mass
        slots: list[list[Any]] = [[None] * depth for _ in range(9)]
        omegas, means, pivots, rhos, sigmas, shrunk, deltas, carried, local = slots
        omega, mean, deviation = leaf.weight, self._mean, leaf.deviation
        pivot_error = gamma_weight * error_mass
        for level in reversed(range(depth)):
            lam = self._penalty[level]
            pivot = omega + lam
            if np.any((lam == 0.0) & (omega == 0.0)):
                raise np.linalg.LinAlgError(
                    f"Nested chain level {names[level]!r} has a node with zero penalty and "
                    "zero accumulated weight, a singular pivot."
                )
            # the tree pivot certificate (note §4.4, item 1): it cannot fail
            # for w >= 0, where D_u >= lambda_u and E_u << omega_u
            floor = pivot_error + 4.0 * eps * (np.abs(omega) + lam)
            if np.any(pivot <= floor):
                negative = pivot < -floor
                if np.any(negative):
                    # below minus the uncertainty: certified negative curvature
                    # (signed observed rows), not a pivot the rounding hides
                    worst = int(np.flatnonzero(negative)[np.argmin(pivot[negative])])
                    raise np.linalg.LinAlgError(
                        f"Nested chain level {names[level]!r} has materially negative "
                        f"curvature at a tree node: pivot {pivot[worst]:.6g}, below minus its "
                        f"certified uncertainty {floor[worst]:.3g}, so the Hessian is "
                        "indefinite there."
                    )
                worst = int(np.argmin(pivot - floor))
                raise np.linalg.LinAlgError(
                    f"Nested chain level {names[level]!r} has a tree pivot {pivot[worst]:.6g} "
                    f"within its certified uncertainty {floor[worst]:.3g}."
                )
            rho = lam / pivot
            sigma = omega / pivot
            s = omega * rho
            omegas[level], means[level], pivots[level] = omega, mean, pivot
            rhos[level], sigmas[level], shrunk[level] = rho, sigma, s
            carried[level] = deviation
            if level == 0:
                deltas[0] = mean
                break
            parent = tree.parent[level]
            omega_parent = _to_parent(tree, level, s)
            magnitude = np.abs(s)
            magnitude_parent = _to_parent(tree, level, magnitude)
            # Shifted parent centres about the child of largest |s|: a column
            # that is constant within the parent gives d = m_c - m_p = 0
            # exactly.  Any centre is exact algebra once the deviation it
            # leaves is carried (note §4.2); |s| keeps the divisor away from a
            # cancelling signed sum.
            order = np.lexsort((-magnitude, parent))
            head = order[np.r_[True, parent[order][1:] != parent[order][:-1]]]
            reference = np.zeros((tree.sizes[level - 1], q))
            reference[parent[head]] = mean[head]
            shift = _to_parent(tree, level, magnitude[:, None] * (mean - reference[parent]))
            mean_parent = np.where(
                magnitude_parent[:, None] != 0.0,
                reference + _divide_rows(shift, magnitude_parent),
                0.0,
            )
            deltas[level] = mean - mean_parent[parent]
            deviation = _carried_deviation(tree, level, s, rho, deltas[level], deviation)
            # the parent sum's own rounding, charged where it is made
            local[level - 1] = gamma_Q * magnitude_parent
            pivot_error = _to_parent(tree, level, rho**2 * pivot_error) + local[level - 1]
            omega, mean = omega_parent, mean_parent
        self._pivots, self._rho, self._sigma = tuple(pivots), tuple(rhos), tuple(sigmas)
        self.minimum_local_diagonal = float(min(np.min(pivot) for pivot in pivots))

        # F = T^-1 C and M F by the top-down g/e recursion (§3.3), with the
        # carried deviation (note §4.3): F_u = sigma_u g_u + delta_u / D_u,
        # e_u = rho_u g_u - delta_u / D_u, g_u = d_u + e_p(u).
        F: list[Any] = [None] * depth
        e: list[Any] = [None] * depth
        growth = deltas[0]
        for level in range(depth):
            if level:
                growth = deltas[level] + e[level - 1][tree.parent[level]]
            F[level], e[level] = sigmas[level][:, None] * growth, rhos[level][:, None] * growth
            if carried[level] is not None:
                share = carried[level] / pivots[level][:, None]
                F[level], e[level] = F[level] + share, e[level] - share
        self._F = tuple(F)
        self._e_leaf = e[-1]
        # the data operator's carried leaf deviation, which C_leaf = w m + delta
        # adds to every product with the tree (``_solve``); None when zero
        self._leaf_deviation = leaf.deviation

        # Takahashi scalars (§3.5), the ancestor tables and the rho path products.
        t = [np.zeros(tree.sizes[0])]
        for level in range(1, depth):
            variance = 1.0 / pivots[level - 1] + rhos[level - 1] ** 2 * t[level - 1]
            t.append(variance[tree.parent[level]])
        self._t = tuple(t)
        self._Z_diag = tuple(1.0 / D + sigma**2 * t_ for D, sigma, t_ in zip(pivots, sigmas, t))
        self._v = tuple(1.0 / D + rho**2 * t_ for D, rho, t_ in zip(pivots, rhos, t))
        self._kappa = tuple(
            1.0 / D - rho * sigma * t_ for D, rho, sigma, t_ in zip(pivots, rhos, sigmas, t)
        )
        # Node-order copies of the per-level scalars and, per level j, the
        # (j + 1, K_j) table of node-order indices of each node's ancestors (row
        # i the level-i ancestor, row j the node itself): every ancestor gather
        # is then one fancy index and the path products one cumprod.
        self._rho_all, self._sigma_all = np.concatenate(rhos), np.concatenate(sigmas)
        self._kappa_all, self._v_all = np.concatenate(self._kappa), np.concatenate(self._v)
        tables = [tree.offsets[0] + np.arange(tree.sizes[0])[None, :]]
        for level in range(1, depth):
            own = tree.offsets[level] + np.arange(tree.sizes[level])[None, :]
            tables.append(np.vstack((tables[-1][:, tree.parent[level]], own)))
        self._ancestor = tuple(tables)
        # _path[j][i]: pi_{p(u)}(z), the product of rho over levels i+1..j-1 for
        # the level-i ancestor z of a level-j node u; _pi[i]: pi_l(z) for the
        # leaves, which includes the leaf's own rho (1 at the leaf).
        self._path = tuple(_descendant_products(self._rho_all[table[:-1]]) for table in tables)
        self._pi = _descendant_products(self._rho_all[tables[-1]])

        # The super-root (design §3.1): border column 0, the intercept, is the
        # unpenalized parent of every root, eliminated last in the tree with
        # pivot D_0 = sum_r s_r.  Its between-child term is the s-weighted
        # scatter of the root means about c* (the |s|-weighted shifted mean on
        # the dense columns, the s-weighted mean on the one-hot ones), plus the
        # carried terms of the roots and the super-root's own -delta_0 delta_0'
        # / D_0, so the border matrix below is the intercept-profiled Schur
        # complement: log|H| = sum_u log D_u + log D_0 + log|Q|, and the
        # intercept never enters a rank decision.
        root_s, root_mean = shrunk[0], deltas[0][:, 1:]
        root_rho, root_magnitude = rhos[0], np.abs(shrunk[0])
        D0 = float(np.sum(root_s))
        root_local = gamma_Q * float(np.sum(root_magnitude))
        super_error = float(np.sum(root_rho**2 * pivot_error)) + root_local
        super_floor = super_error + 4.0 * eps * abs(D0)
        if D0 < -super_floor:
            raise np.linalg.LinAlgError(
                f"Nested chain {names!r} has materially negative curvature along the "
                f"intercept: the super-root pivot sum_r s_r = {D0:.6g} is below minus its "
                f"certified uncertainty {super_floor:.3g}, so the Hessian is indefinite there."
            )
        if not D0 > super_floor:
            raise np.linalg.LinAlgError(
                f"Nested chain {names!r} is aliased with the fitted intercept: the "
                "super-root pivot sum_r s_r is within its certified uncertainty."
            )
        indicator = None if leaf.indicator is None else leaf.indicator[1:]
        dense = np.ones(q - 1, dtype=bool) if indicator is None else ~indicator
        # Dense columns: the shifted two-pass form about the heaviest root.
        # One-hot columns: sum s m m' - D_0 c* c*' on their sparse means, whose
        # rounding the bound U below charges; the one direction where it
        # matters, the block's sum, is deflated structurally (§3.6 step 1).
        # (one product over the whole block for the one-hot columns, whose
        # reference is 0, and the shifted form on the dense columns' slice)
        dense_columns = np.flatnonzero(dense)
        reference = root_mean[int(np.argmax(root_magnitude)), dense_columns]
        center_star = (root_s @ root_mean) / D0
        center_star[dense_columns] = reference + (
            root_magnitude @ (root_mean[:, dense_columns] - reference)
        ) / float(np.sum(root_magnitude))
        # at depth 1 the roots are the leaves, whose one-hot means are nonzero
        # only on the layout's cells
        root_scatter, root_mass = _weighted_scatter(
            root_mean,
            root_s,
            indicator,
            center=np.where(dense, center_star, 0.0),
            pattern=leaf.indicator_pattern if depth == 1 else None,
        )
        if indicator is not None and indicator.any():
            one_hot = np.flatnonzero(indicator)
            root_scatter[np.ix_(one_hot, one_hot)] -= D0 * np.outer(
                center_star[one_hot], center_star[one_hot]
            )
            root_mass[one_hot] += D0 * center_star[one_hot] ** 2
        between, level_mass = root_scatter, root_mass
        for level in range(1, depth):
            scatter, mass = _weighted_scatter(deltas[level][:, 1:], shrunk[level], indicator)
            between, level_mass = between + scatter, level_mass + mass
        root_deviation = None if carried[0] is None else carried[0][:, 1:]
        # the super-root's carried deviation delta_0 = sum_r (s_r d_r + rho_r
        # delta_r): -2 sum_{s_r < 0} |s_r| d_r on the |s|-centred dense
        # columns, 0 on the s-centred one-hot ones (exact algebra, note §4.2)
        super_deviation = np.zeros(q - 1)
        carried_Q, carried_mass = np.zeros((q - 1, q - 1)), np.zeros(q - 1)
        negative_roots = np.flatnonzero(root_s < 0.0)
        if negative_roots.size:
            # sum_r s_r d_r on the dense columns, which the |s| centre does not
            # zero once a root weight is negative
            root_sum = 2.0 * (
                root_s[negative_roots]
                @ (root_mean[np.ix_(negative_roots, dense_columns)] - center_star[dense_columns])
            )
            super_deviation[dense] = root_sum
            if indicator is not None and indicator.any():
                # the dense x one-hot block of sum_r s_r d_r d_r' with the two
                # centres: the root scatter formed sum s (m_d - c_d) m_o' and
                # misses -(sum s d_d) c_o' (zero when no root weight is negative)
                correction = np.zeros((q - 1, q - 1))
                correction[np.ix_(dense_columns, one_hot)] = -np.outer(
                    root_sum, center_star[one_hot]
                )
                carried_Q += correction + correction.T
        if root_deviation is not None:
            super_deviation += root_rho @ root_deviation
        for level in range(depth):
            if carried[level] is None:
                continue
            # the roots' d about c* (dense on the one-hot columns), formed only
            # for signed rows
            d_level = root_mean - center_star if level == 0 else deltas[level][:, 1:]
            term, mass = _carried_terms(d_level, carried[level][:, 1:], rhos[level], pivots[level])
            carried_Q += term
            carried_mass += mass
        if np.any(super_deviation):
            carried_Q -= np.outer(super_deviation, super_deviation) / D0
            carried_mass += super_deviation**2 / D0
        Q_data = leaf.within[1:, 1:] + between + carried_Q
        Q_data = 0.5 * (Q_data + Q_data.T)
        S_rest = operator.border_penalty[1:, 1:]
        # the super-root's multiplier F_0 = c* + delta_0 / D_0 (sigma_0 = 1)
        center_effective = center_star + super_deviation / D0
        # The running bound of every term (signed-rows note §4.4, design
        # §3.6): |dQ_ij| <= sqrt(U_i U_j), each rounding charged where it is
        # made with the exact first-order sensitivity of Q (``_border_bound``).
        # sum_c |s_c| d_c^2 over every child of an internal node and over the
        # roots about c*, the rank-two majorants' share (``_border_bound``): the
        # scatter masses, with the one-hot root form's (m - c)^2 <= 2 m^2 + 2 c^2
        node_mass = level_mass - root_mass
        root_node_mass = root_mass.copy()
        if indicator is not None and indicator.any():
            root_node_mass[one_hot] = 2.0 * (
                root_mass[one_hot]
                - D0 * center_star[one_hot] ** 2
                + float(np.sum(root_magnitude)) * center_star[one_hot] ** 2
            )
        bound = _border_bound(
            tree=tree,
            leaf=leaf,
            error_mass=error_mass,
            gammas=(gamma_Q, gamma_weight, gamma_scatter),
            e=e,
            center_effective=center_effective,
            rhos=rhos,
            carried=carried,
            local=local,
            root_local=root_local,
            super_deviation=super_deviation,
            super_carried=root_deviation,
            D0=D0,
            accumulation=level_mass + carried_mass + np.abs(np.diag(S_rest)),
            node_mass=node_mass + root_node_mass,
        )
        generators = None if leaf.generators is None else leaf.generators.dropped(1)
        # Coordinates excluded from the Laplace approximation (design §3.9,
        # ``reml.identified``): removing border columns from H leaves the
        # other columns' Schur complement Q[M, M], so their rows and columns
        # of Q, S and U are set to exact zero and the border takes them as
        # exact nulls; Q^+ is then (Q[M, M])^-1 padded with zeros and log
        # pdet(Q) = log|Q[M, M]|, the identified part of H.
        self.excluded = tuple(int(index) for index in excluded)
        if self.excluded:
            rest = self._small_position[np.asarray(self.excluded, dtype=np.intp)] - 1
            if np.any(rest < 0):
                raise ValueError("Only border slope columns can be excluded from a nested factor.")
            if generators is not None and np.any(generators.matrix[rest]):
                raise ValueError("An excluded border column cannot carry a structural generator.")
            Q_data, S_rest, bound = Q_data.copy(), S_rest.copy(), bound.copy()
            Q_data[rest, :] = Q_data[:, rest] = 0.0
            S_rest[rest, :] = S_rest[:, rest] = 0.0
            bound[rest] = 0.0
        border = factor_border(Q_data, S_rest, bound, generators, term_name=names[-1])
        center_star = center_effective
        self._center_star = center_star
        self._intercept_pivot = D0
        self._border = border
        self.border_certificate: BorderCertificate = border.certificate
        # §3.6 step 5 / §3.9: coefficients a truncated direction with certified
        # penalty curvature touches are weakly identified, not aliased
        # (augmented-global indices; the rest columns sit one after the intercept).
        certificate = border.certificate
        self.weakly_identified_coefficients: tuple[int, ...] = tuple(
            sorted(
                {
                    int(self.small_indices[column + 1])
                    for direction, weak in zip(
                        certificate.directions, certificate.weak, strict=True
                    )
                    if weak
                    for column in direction
                }
            )
        )
        # Q^+ and its data side (``_Q_inverse``, ``_Q_inverse_data``) are formed
        # on first read: a PIRLS iterate only solves (``solve_data``).
        self._scaled_matrix = border.scaled_matrix
        self._scaled_eigenvalues_cache: NDArray | None = None
        self.schur_condition_estimate = border.condition
        # Nulls of Q in the factor's coordinates: [-c*'z; z] for every null z
        # of the profiled Q (W's intercept entry is 0, so Q^+ Q e_0 = e_0 and
        # the intercept is H-orthogonal to every null: gram's weighted-mean-
        # centred convention, §3.6).  diag(Q^+ Q) = 1 - sum_k V_jk W_jk.
        null = border.null
        self._null_border = np.vstack((-(center_star @ null)[None, :], null))
        self._retained_own = np.concatenate(
            ([1.0], 1.0 - np.einsum("ij,ij->i", null, border.null_left))
        )
        self._retained_diagonal = self._retained_own
        # sqrt(diag Q) in the factor's coordinates, the scaling of the
        # estimability null basis (``coefficient_estimable``)
        self._border_root = np.sqrt(
            np.maximum(
                np.r_[D0, np.diag(Q_data) + np.diag(S_rest) + D0 * center_star**2],
                np.finfo(np.float64).tiny,
            )
        )
        # a column null by its own bound has scale 1, as in the rank decision
        self._border_root[1 + border.exact_null_columns] = 1.0
        # The published log|H|: det(T) pdet(Q), the convention identity of §3.6
        # (= sum_w pdet(H_c), gram's weighted-mean-centred pseudo-determinant).
        self._logdet = float(
            sum(np.sum(np.log(pivot)) for pivot in pivots) + np.log(D0) + border.logdet
        )
        self.rank = int(tree.n_nodes + q - null.shape[1])
        self.rank_truncated = self.rank < p
        self._diagonal_cache: NDArray | None = None
        self._level_gram: dict[int, NDArray] = {}
        self._level_h: dict[int, NDArray] = {}
        self._level_pairs: dict[tuple[int, int], float] = {}
        self._state_format = _NESTED_STATE_FORMAT

    def __getstate__(self) -> dict:
        return _pending_nested_state(self)

    def __setstate__(self, state: dict) -> None:
        _restore_nested_state(self, state)

    def __getattr__(self, name: str):
        return _rebuild_on_first_use(self, name, self._rebuild)

    def _rebuild(self, state: dict) -> None:
        """Rebuild a factor pickled by another build from the operator it factored (§3.12).

        The saved ``NestedPenalizedOperator`` is the linear system at the saved
        coefficients and smoothing parameters (its leaf statistics carry the
        final working weights), so this is one factorization of the fitted
        ``H``.  Leaf statistics from before a field existed read that field's
        default.  The retired raw-coordinate factor (``intercept=False``),
        which an old state retains but nothing reads, is restored inert.
        """
        operator = state.get("operator")
        if operator is None or not state.get("intercept", True):
            self.__dict__.update(state)
            return
        warnings.warn(_REBUILT_NOTICE, UserWarning, stacklevel=4)
        NestedSchurFactor.__init__(
            self,
            operator,
            chain_group_names=tuple(state["chain_group_names"]),
            chain_group_indices=tuple(state["chain_group_indices"]),
            intercept=True,
            max_structured_inverse_block=int(state.get("max_structured_inverse_block", 256)),
        )

    @cached_property
    def _Q_inverse(self) -> NDArray:
        """``Q^+ = L0^-T diag(1 / D_0, Q_rest^+) L0^-1``, ``L0 = [[1, 0], [c*, I]]``, formed on first read.

        Cache owner: the factor; lifetime: the factor; invalidation: none (its
        inputs are fixed at construction).  Under the construction's BLAS cap.
        """
        with narrow_kernel_blas_threads(self._leaf.width):
            return _super_root_inverse(
                self._border.inverse, self._center_star, self._intercept_pivot
            )

    @cached_property
    def _Q_inverse_data(self) -> NDArray:
        """The same map of the data-side border inverse (``BorderFactor.inverse_data``).

        Equal to ``Q^+`` on every data-derived vector, whose component along the
        deflated structural nulls is exactly zero, and free of the ``1 / a_NN``
        entries that would cancel in it.  Formed on first read, as ``_Q_inverse``.
        """
        if not self._border.deflated:
            return self._Q_inverse
        with narrow_kernel_blas_threads(self._leaf.width):
            return _super_root_inverse(
                self._border.inverse_data, self._center_star, self._intercept_pivot
            )

    # -- tree kernels -------------------------------------------------------
    def _tree_solve(self, rhs: Sequence[NDArray]) -> list[NDArray]:
        """Return ``T^-1 rhs`` for per-level ``(K_j,)`` or ``(K_j, m)`` right-hand sides.

        ``T = L D L'`` with ``L_{a,u} = sigma_u`` for ``a`` a strict ancestor of
        ``u`` (Lean ``chain_ldl``): a forward pass leaves to roots, the pivot
        division, and a backward pass roots to leaves.
        """
        tree, depth = self._tree, self._tree.depth
        y: list[Any] = [None] * depth
        carried = np.zeros_like(np.asarray(rhs[-1], dtype=np.float64))
        for level in reversed(range(depth)):
            y[level] = rhs[level] - carried
            if level:
                carried = _to_parent(tree, level, (self._sigma[level] * y[level].T).T + carried)
        x: list[Any] = [None] * depth
        above = np.zeros_like(y[0])
        for level in range(depth):
            if level:
                above = (above + x[level - 1])[tree.parent[level]]
            x[level] = ((y[level].T / self._pivots[level]) - self._sigma[level] * above.T).T
        return x

    def _selector_h(self, weights: Sequence) -> tuple[NDArray, ...]:
        """Return ``h_a = sum_{c in ch(a)} (weight_c sigma_c^2 + rho_c^2 h_c)`` bottom-up.

        ``weights[j]`` is a scalar or ``(K_j,)`` per level: the level selector
        ``1_I`` for ``penalty_cross_trace``, the node penalties for the penalty
        sandwich.
        """
        tree = self._tree
        h: list[Any] = [None] * tree.depth
        h[-1] = np.zeros(tree.sizes[-1])
        for level in reversed(range(1, tree.depth)):
            summand = weights[level] * self._sigma[level] ** 2 + self._rho[level] ** 2 * h[level]
            h[level - 1] = _to_parent(tree, level, summand)
        return tuple(h)

    def _level_selector_h(self, level: int) -> NDArray:
        """``h^I`` of the level-``level`` selector in node order, cached per level."""
        cached = self._level_h.get(level)
        if cached is None:
            selector = [float(j == level) for j in range(self._tree.depth)]
            cached = self._level_h[level] = np.concatenate(self._selector_h(selector))
        return cached

    def _gram(self, level: int) -> NDArray:
        """``G_I = F_I' F_I``, cached per level for the life of the factor."""
        cached = self._level_gram.get(level)
        if cached is None:
            cached = self._level_gram[level] = self._F[level].T @ self._F[level]
        return cached

    def _inverse_diagonal(self) -> NDArray:
        """``diag(H^-1)``: ``Z_uu + F_u Q^-1 F_u'`` on the tree, ``diag(Q^-1)`` on the border; cached."""
        if self._diagonal_cache is None:
            Q_inverse = self._Q_inverse
            # F = T^-1 C is data: F v = 0 on the deflated structural nulls, so
            # the tree entries read the data-side inverse (BorderFactor)
            Q_data = self._Q_inverse_data
            diagonal = np.empty(self.shape[0])
            diagonal[self.structured_indices] = np.concatenate(
                [
                    Z + np.sum((F @ Q_data) * F, axis=1)
                    for Z, F in zip(self._Z_diag, self._F, strict=True)
                ]
            )
            diagonal[self.small_indices] = np.diag(Q_inverse)
            # the intercept entry of R Q^-1 R': (e_0 - c)' Q^-1 (e_0 - c)
            intercept = -self._center
            intercept[:1] += 1.0
            diagonal[self.small_indices[:1]] = intercept @ Q_inverse @ intercept
            self._diagonal_cache = diagonal
        return self._diagonal_cache

    # -- protocol: solves, determinant, selected inverse ------------------
    def solve(
        self,
        rhs: NDArray,
        *,
        centred: bool = False,
        border_centred: NDArray | None = None,
    ) -> NDArray:
        """Return ``H^-1 rhs`` for ``(p,)`` or ``(p, m)`` (§3.6); see ``_solve``.

        ``border_centred``, when given, is the border slopes' part of ``rhs``
        already in the factor's centred coordinates (as
        ``SumToZeroTreeFactor.solve``'s; ``solve_data`` takes the intercept's
        row with it, which here is ``rhs``'s own).

        With ``centred`` the intercept entry stays the centred ``alpha``, as in
        ``solve_data``.
        """
        if border_centred is not None:
            head = np.asarray(rhs, dtype=np.float64)[:1].reshape(1, -1)
            tail = np.asarray(border_centred, dtype=np.float64).reshape(len(self._center) - 1, -1)
            border_centred = np.vstack((head, tail))
        return self._solve(
            rhs,
            lambda border: self._Q_inverse @ border,
            border_centred=border_centred,
            centred=centred,
        )

    def _border_apply_data(self, values: NDArray) -> NDArray:
        """``Q^+ values`` for data-derived border vectors ``(q,)`` or ``(q, r)``, factored.

        The super-root first (``z_rest = y_rest - c* y_0``, ``x_0 = y_0 / D_0 -
        c*' x_rest``), then ``BorderFactor.apply_data`` on the profiled rest.
        """
        values = np.asarray(values, dtype=np.float64)
        matrix = values[:, None] if values.ndim == 1 else values
        rest = matrix[1:] - self._center_star[:, None] * matrix[:1]
        x_rest = self._border.apply_data(rest)
        x_0 = matrix[:1] / self._intercept_pivot - self._center_star @ x_rest
        result = np.vstack((x_0, x_rest))
        return result[:, 0] if values.ndim == 1 else result

    def _retained_border(self, border: NDArray) -> NDArray:
        """``(I - V_t W_t') border`` ``(q, r)``: the difference of two retained-subspace solves kept on it.

        Each of the two border solves of ``_solve`` lands on the retained
        subspace to the rounding of its own (projected) result, but they
        cancel to a much smaller solution, which would then carry that
        rounding along a truncated null direction: the Moore-Penrose solution
        has none.  ``V_t = [-c*' v; v]`` are the truncated (not exact) border
        nulls in the factor's coordinates and ``W_t = [0; w]`` their left
        partners (``Q^+ Q = I - V W'``, ``W'V = I``); ``O(q t r)``.  Exact
        nulls (zero rows of ``Q``) carry exact zeros already.
        """
        start = self._border.certificate.exact_null
        if self._null_border.shape[1] == start:
            return border
        coefficients = self._border.null_left[:, start:].T @ border[1:]
        return border - self._null_border[:, start:] @ coefficients

    def _border_quadratic_data(self, rows: NDArray) -> NDArray:
        """``y' Q^+ y`` per row of data-derived border rows ``(r, q)``: the super-root, then the rest."""
        rest = rows[:, 1:] - rows[:, :1] * self._center_star[None, :]
        return rows[:, 0] ** 2 / self._intercept_pivot + self._border.quadratic_data(rest)

    def solve_data(
        self,
        rhs: NDArray,
        *,
        border_centred: NDArray | None = None,
        centred: bool = False,
    ) -> NDArray:
        """Return ``H^-1 rhs`` for a data-derived ``rhs``, such as the normal equations' ``X'Wz``.

        A right-hand side in the range of the data operator's transpose is
        exactly orthogonal to the deflated structural nulls, so this reads the
        data-side border inverse (``BorderFactor.inverse_data``): the same
        solution in exact arithmetic, without the cancellation of ``1 / a_NN``
        sized products along a direction only the penalty identifies (one-engine
        design §3.6 step 1).  PIRLS solves its Newton system through it.
        ``border_centred`` ``(q,)`` or ``(q, m)``, when given, is the border
        part of ``rhs`` already in the factor's centred coordinates (``R'``
        applied), formed from centred rows (``NestedStructuredSystem.
        xtwz_small_centred`` behind the intercept's sum): it replaces the
        subtraction ``r_b - c r_0``, which loses a column's offset-to-spread
        ratio in digits (§3.2).  With ``centred`` the intercept entry of the
        solution stays the centred ``alpha`` (no ``R`` map out), the PIRLS
        state of one-engine design §3.8.
        """
        return self._solve(
            rhs, self._border_apply_data, border_centred=border_centred, centred=centred
        )

    def _solve(
        self,
        rhs: NDArray,
        apply: Callable[[NDArray], NDArray],
        *,
        border_centred: NDArray | None = None,
        centred: bool = False,
    ) -> NDArray:
        """Return ``H^-1 rhs`` for ``(p,)`` or ``(p, m)`` (§3.6).

        ``u = T^-1 r_t`` by the tree forward, diagonal and backward passes,
        ``x_b = Q^+ (r_b - C_leaf' (M u)_leaf)`` with ``C_leaf = w m + delta``
        in the factor's coordinates, ``x_t = u - F x_b``, between ``R'`` in and ``R``
        out; ``Q^+`` is the retained-subspace inverse on a truncated factor.
        O(n_nodes q + q^2) per column.  A non-finite solution raises
        ``LinAlgError``.
        """
        values = np.asarray(rhs, dtype=np.float64)
        columns = values[:, None] if values.ndim == 1 else values
        if columns.ndim != 2 or columns.shape[0] != self.shape[0]:
            raise ValueError(
                f"rhs must have shape ({self.shape[0]},) or ({self.shape[0]}, m), "
                f"got {values.shape}."
            )
        tree = self._tree
        u = self._tree_solve(tree.split(columns[self.structured_indices]))
        if border_centred is None:
            border_rhs = columns[self.small_indices]
            # R' r: the centred border rows lose c times the intercept row.
            border_rhs = border_rhs - self._center[:, None] * border_rhs[:1]
        else:
            border_rhs = np.asarray(border_centred, dtype=np.float64).reshape(len(self._center), -1)
        reached = tree.path_sum(u)
        # C_leaf = w m + delta: the carried deviation of signed rows (note §4.3)
        share = self._mean.T @ (self._leaf.weight[:, None] * reached)
        if self._leaf_deviation is not None:
            share = share + self._leaf_deviation.T @ reached
        # the tree's share C'T^-1 r_t is data (C v = 0 on the structural nulls):
        # it takes the data-side inverse whatever the right-hand side
        border = self._retained_border(apply(border_rhs) - self._border_apply_data(share))
        solution = np.empty_like(columns)
        solution[self.structured_indices] = np.concatenate(
            [u_level - F @ border for u_level, F in zip(u, self._F, strict=True)]
        )
        # R x: only the intercept entry moves back to raw coordinates.
        if not centred:
            border[:1] -= self._center @ border
        solution[self.small_indices] = border
        if not np.all(np.isfinite(solution)):
            raise np.linalg.LinAlgError(
                f"Nested chain {self.dominant_group_name!r} solve is not representable."
            )
        return solution[:, 0] if values.ndim == 1 else solution

    def logdet(self) -> float:
        """Return ``sum_u log D_u + log|Q|``.

        On a truncated factor ``log|Q|`` is ``log det(B' Q B)`` with ``B`` an
        orthonormal basis of the complement of the certified null space, the
        pseudo-determinant the single-level factor reports; with an intercept
        it is taken in the dense backend's weighted-mean-centred coordinates.
        """
        return self._logdet

    def _validate_selected_indices(self, indices: NDArray) -> NDArray:
        selected = np.asarray(indices, dtype=np.intp)
        if selected.ndim != 1:
            raise ValueError("Selected inverse indices must be one-dimensional.")
        if np.any((selected < 0) | (selected >= self.shape[0])):
            raise IndexError("Selected inverse index is outside the factor dimensions.")
        if len(np.unique(selected)) != len(selected):
            raise ValueError("Selected inverse indices must be unique.")
        return selected

    def selected_inverse_diagonal(self, indices: NDArray) -> NDArray:
        """Return ``diag(H^-1)[indices]`` in the requested order.

        Tree node ``u``: ``Z_uu + F_u Q^-1 F_u'``; border: ``diag(Q^-1)``.  The
        full diagonal is formed once, O(n_nodes q^2), and cached.  Indices must
        be one-dimensional, unique and in range (``ValueError``/``IndexError``).
        """
        diagonal = self._inverse_diagonal()[self._validate_selected_indices(indices)]
        if not np.all(np.isfinite(diagonal)):
            raise np.linalg.LinAlgError(
                f"Nested chain {self.dominant_group_name!r} selected inverse is not representable."
            )
        return diagonal

    def selected_inverse_block(self, indices: NDArray) -> NDArray:
        """Return the principal block ``H^-1[indices][:, indices]`` by solves.

        Refuses with ``ValueError`` when more than ``max_structured_inverse_block``
        of the requested indices are tree nodes (any chain level), with the
        single-level factor's "request its diagonal instead" message.
        """
        selected = self._validate_selected_indices(indices)
        nodes = int(np.count_nonzero(self._node_position[selected] >= 0))
        if nodes > self.max_structured_inverse_block:
            raise ValueError(
                f"Refusing to materialize a {nodes} x {nodes} inverse block for structured "
                f"term {self.dominant_group_name!r}; request its diagonal instead."
            )
        unit = np.zeros((self.shape[0], len(selected)))
        unit[selected, np.arange(len(selected))] = 1.0
        block = self.solve(unit)[selected]
        return 0.5 * (block + block.T)

    def row_quadratic_forms(self, rows: NDArray) -> NDArray:
        """Return ``x_i' H^+ x_i`` for each row of ``rows`` ``(m, p)``, forming no ``K x K`` block.

        In the factor's coordinates ``H^+ = [[T^-1 + F Q^+ F', -F Q^+], [-Q^+ F',
        Q^+]]``, so a row with tree part ``b`` and border part ``a`` (``R'``
        applied) gives ``||D^-1/2 L^-1 b||^2 + y' Q^+ y`` with ``y = a - F' b``:
        the random-effect and fixed-effect halves of the hat diagonal of Bates et
        al. (2015, eqs. 63-65).  ``L^-1 e_u`` lives on the reach of ``u`` (Gilbert
        and Peierls 1988), its ancestors-or-self: 1 at ``u`` and ``-sigma_u`` times
        the ``rho`` product strictly between at each ancestor (``_path``).  Each
        nonzero of ``b`` adds at most ``depth`` terms, summed per row and node
        before squaring, so a row whose levels are not one root-to-leaf path is
        exact too.  O(nnz(b) depth + m q^2).  The rows must be data rows (a fit
        row ``[1, x_i]``, the only caller being leverage): ``y`` then has no
        component along the deflated structural nulls and ``y' Q^+ y`` reads the
        data side (``BorderFactor.quadratic_data``).
        """
        values = scipy.sparse.csr_array(rows, dtype=np.float64)
        if values.shape[1] != self.shape[0]:
            raise ValueError(f"rows must have shape (m, {self.shape[0]}), got {values.shape}.")
        tree = self._tree
        border = values[:, self.small_indices].toarray()
        # R' x: the centred border columns lose c times the intercept entry.
        border -= border[:, :1] * self._center
        keys, terms = [], []
        for level, (start, stop) in enumerate(zip(tree.offsets[:-1], tree.offsets[1:])):
            level_rows = values[:, self.structured_indices[start:stop]].tocoo()
            border -= level_rows @ self._F[level]
            row, local = (np.asarray(index, dtype=np.intp) for index in level_rows.coords)
            reach = np.vstack(
                (-self._sigma[level][local] * self._path[level][:, local], np.ones(len(local)))
            )
            keys.append((row * tree.n_nodes + self._ancestor[level][:, local]).ravel())
            terms.append((level_rows.data * reach).ravel())
        reached, slot = np.unique(np.concatenate(keys), return_inverse=True)
        solved = np.bincount(slot, weights=np.concatenate(terms))
        pivots = np.concatenate(self._pivots)[reached % tree.n_nodes]
        tree_part = np.bincount(
            reached // tree.n_nodes, weights=solved * solved / pivots, minlength=values.shape[0]
        )
        # a row [1, x] is data: exactly orthogonal to the structural nulls
        return tree_part + self._border_quadratic_data(border)

    # -- penalty components --------------------------------------------------
    def _classify(self, component: PenaltyComponent) -> tuple[str, Any]:
        """Return ``("level", I)`` for an identity penalty on exactly level ``I``, else ``("border", (J, Omega_JJ))``.

        ``J`` holds the component's border positions and ``Omega_JJ`` its
        penalty block on them (``None`` for an identity): every border trace
        reads ``Q^-1`` on the component's own rows and columns only, the rest
        of the embedded ``(q, q)`` penalty being exact zeros.  Anything else (a
        dense penalty on the chain, part of a level, or a component straddling
        chain and border) is ``ValueError``.
        """
        indices = _component_indices(component, self.shape[0])
        nodes = self._node_position[indices]
        if np.all(nodes >= 0):
            offsets = self._tree.offsets
            level = int(np.searchsorted(offsets, nodes.min(), side="right") - 1)
            whole_level = np.arange(offsets[level], offsets[level + 1])
            if component.penalty_kind != "identity" or not np.array_equal(
                np.sort(nodes), whole_level
            ):
                raise ValueError(
                    f"Penalty component {component.name!r} must be an identity penalty on "
                    "exactly one nested chain level."
                )
            return "level", level
        positions = self._small_position[indices]
        if np.any(positions < 0):
            raise ValueError(
                f"Penalty component {component.name!r} straddles the nested chain and the border."
            )
        if component.penalty_kind == "identity":
            return "border", (positions, None)
        return "border", (positions, _component_omega(component, self.shape[0]))

    def _level_pair(self, left_level: int, right_level: int) -> float:
        """``||(H^-1)_IJ||_F^2 = t1 + t2 + t3`` for two chain levels (§3.6)."""
        depth = self._tree.depth
        h_I = self._tree.split(self._level_selector_h(left_level))
        h_J = self._tree.split(self._level_selector_h(right_level))
        t1 = 0.0
        for level in range(depth):
            D, rho, sigma, t = (
                self._pivots[level],
                self._rho[level],
                self._sigma[level],
                self._t[level],
            )
            diag_I, diag_J = (
                h_I[level] + float(level == left_level),
                h_J[level] + float(level == right_level),
            )
            off_I = rho * h_I[level] - float(level == left_level) * sigma
            off_J = rho * h_J[level] - float(level == right_level) * sigma
            t1 += float(np.sum(diag_I * diag_J / D**2) + 2.0 * np.sum(off_I * off_J * t / D))
        selected = [
            self._F[left_level] if level == left_level else np.zeros_like(F)
            for level, F in enumerate(self._F)
        ]
        Y = self._tree_solve(selected)
        # every Q^+ here sits beside F-derived blocks: the data side
        Q_inverse = self._Q_inverse_data
        t2 = 2.0 * float(np.sum(Q_inverse * (self._F[right_level].T @ Y[right_level])))
        t3 = float(
            np.sum((Q_inverse @ self._gram(right_level)) * (Q_inverse @ self._gram(left_level)).T)
        )
        return t1 + t2 + t3

    def _penalty_pair(self, left: tuple, right: tuple) -> float:
        """Unscaled ``tr(H^-1 Omega_l H^-1 Omega_r)`` for two classified components.

        A border penalty is zero outside its own rows and columns ``J``, so each
        product reads the ``Q^-1`` rows and columns on ``J`` only: the same
        nonzero terms as the embedded ``(q, q)`` products, in O(k q^2) for a
        level and O(k_l k_r (k_l + k_r)) for two border components.
        """
        Q_inverse = self._Q_inverse
        if left[0] == "level" and right[0] == "level":
            # coarser level first: the order the §8 verification used, and
            # bitwise symmetric under the cancellation of t1 + t2 + t3
            pair = (min(left[1], right[1]), max(left[1], right[1]))
            if pair not in self._level_pairs:
                self._level_pairs[pair] = self._level_pair(*pair)
            return self._level_pairs[pair]
        if left[0] == "level" or right[0] == "level":
            level, (J, omega) = (left[1], right[1]) if left[0] == "level" else (right[1], left[1])
            # (Q^+ G_I Q^+)_JJ: both Q^+ beside G_I = F_I'F_I, the data side
            Q_data = self._Q_inverse_data
            rows = Q_data[J] @ self._gram(level)
            return float(np.sum(rows * _times_penalty(Q_data[:, J], omega).T))
        (J_l, omega_l), (J_r, omega_r) = left[1], right[1]
        return float(
            np.sum(
                _times_penalty(Q_inverse[np.ix_(J_r, J_l)], omega_l)
                * _times_penalty(Q_inverse[np.ix_(J_l, J_r)], omega_r).T
            )
        )

    def trace_inverse_penalty(self, component: PenaltyComponent) -> float:
        """Return ``tr(H^-1 Omega)`` for one penalty component.

        An identity component whose indices are exactly one chain level ``I``
        gives ``sum_{u in I} diag(H^-1)_u``; a component wholly in the border
        (identity or dense) gives ``tr(Q^-1_JJ Omega)``.  A component that
        straddles the chain and the border, or covers part of a level, raises
        ``ValueError``.
        """
        kind, key = self._classify(component)
        if kind == "level":
            offsets = self._tree.offsets
            nodes = self.structured_indices[offsets[key] : offsets[key + 1]]
            return float(np.sum(self._inverse_diagonal()[nodes]))
        return _penalty_trace(self._Q_inverse, key)

    def penalty_cross_trace(
        self,
        left: PenaltyComponent,
        right: PenaltyComponent,
        left_scale: float,
        right_scale: float,
    ) -> float:
        """Return ``left_scale right_scale tr(H^-1 Omega_l H^-1 Omega_r)`` (§3.6).

        Two levels: ``t1 + t2 + t3`` with ``t1`` the O(n_nodes) ``h/e`` closed
        form, ``t2 = 2 tr(Q^-1 F_J' (T^-1 E_I F)_J)`` and ``t3 = tr(Q^-1 G_J Q^-1
        G_I)``.  Level and border: ``tr((Q^-1 G_I Q^-1)_JJ Omega_J)``.  Border
        pair: ``Q^-1`` blocks.  Component rules as in ``trace_inverse_penalty``.
        """
        pair = self._penalty_pair(self._classify(left), self._classify(right))
        return float(left_scale * right_scale * pair)

    # -- operators ------------------------------------------------------------
    def _check_operator(self, operator: NestedDataOperator) -> None:
        tree = self._tree
        same_forest = operator.tree is tree or (
            operator.tree.sizes == tree.sizes
            and all(
                np.array_equal(a, b) for a, b in zip(operator.tree.parent, tree.parent, strict=True)
            )
        )
        if not same_forest:
            raise ValueError("Nested operator describes another forest than the factor.")
        if not (
            np.array_equal(operator.small_indices, self.small_indices)
            and np.array_equal(operator.structured_indices, self.structured_indices)
        ):
            raise ValueError("Nested operator partitions differ from the factor's.")
        if not (
            np.array_equal(operator.leaf.mean, self._leaf.mean)
            and np.array_equal(operator.leaf.center, self._leaf.center)
        ):
            raise ValueError("Nested operator was built about other leaf means than the factor's.")

    def _pieces(self, operator) -> tuple[tuple, tuple[LowRankSymmetricOperator, ...]]:
        """Split an accepted operator into leaf-form and low-rank parts (``TypeError`` otherwise)."""
        if operator is None:
            return (), ()
        if isinstance(operator, SumBlockOperator):
            parts = [self._pieces(item) for item in operator.operators]
            return (
                tuple(itertools.chain.from_iterable(leaf for leaf, _ in parts)),
                tuple(itertools.chain.from_iterable(low for _, low in parts)),
            )
        if operator.shape != self.shape:
            raise ValueError("Operator and factor dimensions must match.")
        if isinstance(operator, LowRankSymmetricOperator):
            return (), (operator,)
        if isinstance(operator, NestedDataOperator):
            self._check_operator(operator)
            return (operator,), ()
        raise TypeError(
            f"NestedSchurFactor does not represent {type(operator).__name__} operators."
        )

    def _piece(self, operator: NestedDataOperator) -> _Piece:
        """Row-pass quantities of a leaf-form operator about the factor's leaf means (§3.6)."""
        leaf, e = operator.leaf, self._e_leaf
        weighted = leaf.weight[:, None] * e
        V = weighted if leaf.deviation is None else leaf.deviation + weighted
        UOU = leaf.within + weighted.T @ e
        if leaf.deviation is not None:
            UOU = UOU + leaf.deviation.T @ e + e.T @ leaf.deviation
        return _Piece(leaf.weight, V, 0.5 * (UOU + UOU.T))

    def _direction(
        self,
        component: PenaltyComponent | None,
        scale: float,
        operators: tuple,
        low_rank: tuple[LowRankSymmetricOperator, ...],
    ) -> _Direction:
        kind, key = (None, None) if component is None else self._classify(component)
        piece = self._merged_piece(operators) if operators else None
        return _Direction(component, float(scale), kind, key, piece, operators, low_rank)

    def _merged_piece(self, operators: Sequence[NestedDataOperator]) -> _Piece:
        """The summed row-pass quantities of one or more accepted leaf-form operators."""
        return sum((self._piece(item) for item in operators[1:]), start=self._piece(operators[0]))

    def _direction_of(self, component, scale, operator) -> _Direction:
        return self._direction(component, scale, *self._pieces(operator))

    def _weight_delta(self, weights: NDArray) -> list[NDArray]:
        """``delta_l = a_l``, ``delta_u = sum_c rho_c delta_c`` per level, for ``(K,)`` or ``(K, n)``."""
        tree, depth = self._tree, self._tree.depth
        delta: list[Any] = [None] * depth
        delta[-1] = weights
        for level in reversed(range(1, depth)):
            delta[level - 1] = _to_parent(tree, level, (self._rho[level] * delta[level].T).T)
        return delta

    def _tree_weight_solve(self, weights: NDArray) -> list[NDArray]:
        """Return ``T^-1 M'a`` per level from the tree covariances, never by the tree solve.

        ``(T^-1 M'a)_u = sum_l Cov(beta_u, S_l) a_l = kappa_u delta_u - sigma_u
        sum_{x strict ancestor} v_x pi_{p(u)}(x) (delta_x - rho_c delta_c)``, ``c``
        the child of ``x`` toward ``u``.  With ``lambda << omega`` the forward
        pass of the tree solve cancels ``M'a`` against the eliminated mass at
        the internal nodes; this form does not.
        """
        delta = self._weight_delta(weights)
        flat = np.concatenate(delta).reshape(self._tree.n_nodes, -1)
        values = []
        for level, table in enumerate(self._ancestor):
            above, below, own = table[:-1], table[1:], table[-1]
            mass = flat[above] - self._rho_all[below][..., None] * flat[below]
            weight = (self._v_all[above] * self._path[level])[..., None]
            ancestors = self._sigma_all[own][:, None] * np.sum(weight * mass, axis=0)
            value = self._kappa_all[own][:, None] * flat[own] - ancestors
            values.append(value.reshape(delta[level].shape))
        return values

    def _prepare(self, piece: _Piece) -> _Prepared:
        """Form the per-direction products, with ``tr(H^-1 E_I H^-1 O)`` per level.

        The level values are ``sum_{i in I} [(Z M'diag(a) M Z)_ii - 2 W_i Q^-1 F_i'
        + F_i P F_i']`` with the tree-covariance closed form for the first term
        (never explicit inverse columns) and ``W = T^-1 M'V``.
        """
        tree, depth = self._tree, self._tree.depth
        # every product here has Q^+ beside a data operator's rows
        Q_inverse = self._Q_inverse_data
        tree_solve = self._tree_solve(tree.subtree_sum(piece.V))
        P = Q_inverse @ piece.UOU @ Q_inverse
        table = self._ancestor[-1]
        values = np.empty(depth)
        for level in range(depth):
            h = self._level_selector_h(level)
            above, below = table[:level], table[1 : level + 1]
            mass = h[above] - self._rho_all[below] ** 2 * h[below]
            if level:
                mass[-1] -= self._sigma_all[below[-1]] ** 2
            covariance = self._kappa_all[table[level]] ** 2 * self._pi[level] ** 2 + np.sum(
                self._v_all[above] ** 2 * self._pi[:level] ** 2 * mass, axis=0
            )
            F = self._F[level]
            values[level] = (
                piece.a @ covariance
                - 2.0 * np.sum((tree_solve[level] @ Q_inverse) * F)
                + np.sum((F @ P) * F)
            )
        weight: list[Any] = [None] * (depth - 1) + [piece.a]
        for level in reversed(range(1, depth)):
            weight[level - 1] = _to_parent(tree, level, self._rho[level] ** 2 * weight[level])
        return _Prepared(
            zhat=tree.path_sum(tree_solve),
            VQ=piece.V @ Q_inverse,
            P=P,
            weight=tuple(weight),
            level=values,
        )

    def _operator_pair(
        self, left: _Piece, left_prep: _Prepared, right: _Piece, right_prep: _Prepared
    ) -> float:
        """``tr(H^-1 O_l H^-1 O_r)`` for two leaf-form pieces (§3.6)."""
        tree, depth = self._tree, self._tree.depth
        zz = 0.0
        for level in range(depth):
            inner = left_prep.weight[level] * right_prep.weight[level]
            if level < depth - 1:
                below = (
                    self._rho[level + 1] ** 4
                    * left_prep.weight[level + 1]
                    * right_prep.weight[level + 1]
                )
                inner = inner - _to_parent(tree, level + 1, below)
            zz += float(np.sum(self._v[level] ** 2 * inner))
        return (
            zz
            + float(np.sum(right_prep.VQ * left_prep.zhat))
            + float(np.sum(left_prep.VQ * right_prep.zhat))
            + float(np.sum(left_prep.P * right.UOU))
        )

    def _penalty_operator(self, kind: str, key, prepared: _Prepared) -> float:
        if kind == "level":
            return float(prepared.level[key])
        return _penalty_trace(prepared.P, key)

    def _pair(self, left: _Direction, left_prep, right: _Direction, right_prep) -> float:
        """The closed-form part of ``tr(H^-1 O_l H^-1 O_r)`` (penalty and leaf-form parts)."""
        value = 0.0
        if left.kind and right.kind:
            value += (
                left.scale
                * right.scale
                * self._penalty_pair((left.kind, left.key), (right.kind, right.key))
            )
        if left.kind and right.piece is not None:
            value += left.scale * self._penalty_operator(left.kind, left.key, right_prep)
        if right.kind and left.piece is not None:
            value += right.scale * self._penalty_operator(right.kind, right.key, left_prep)
        if left.piece is not None and right.piece is not None:
            value += self._operator_pair(left.piece, left_prep, right.piece, right_prep)
        return value

    def _cross_matrix(self, records: Sequence[_Direction]) -> NDArray:
        """The closed-form ``(m, m)`` matrix over ``records`` (low-rank parts excluded)."""
        prepared = [
            None if record.piece is None else self._prepare(record.piece) for record in records
        ]
        traces = np.empty((len(records), len(records)))
        for i, j in itertools.combinations_with_replacement(range(len(records)), 2):
            traces[i, j] = traces[j, i] = self._pair(
                records[i], prepared[i], records[j], prepared[j]
            )
        return traces

    def _intercept_column_solve(self, piece: _Piece) -> NDArray:
        """Return ``H^-1 O e_0`` in the factor's coordinates, border column 0 the intercept.

        ``O e_0 = [M'a; A_{:,0}]`` and ``U'(O e_0) = sum_l V_l =: g`` from the
        row pass (never ``A_{:,0} - (MF)'a``): tree rows ``T^-1 M'a - F Q^-1 g``,
        border rows ``Q^-1 g``.
        """
        g = self._border_apply_data(piece.V.sum(axis=0))
        nodes = self._tree_weight_solve(piece.a)
        column = np.empty(self.shape[0])
        column[self.structured_indices] = np.concatenate(
            [level - F @ g for level, F in zip(nodes, self._F, strict=True)]
        )
        column[self.small_indices] = g
        return column

    def _intercept_column(self, piece: _Piece) -> NDArray:
        """``O e_0 = [M'a; A_{:,0}]`` in the factor's coordinates, ``A_{:,0} = sum_l V_l + (MF)'a``."""
        column = np.empty(self.shape[0])
        column[self.structured_indices] = np.concatenate(self._tree.subtree_sum(piece.a))
        column[self.small_indices] = piece.V.sum(axis=0) + (self._mean - self._e_leaf).T @ piece.a
        return column

    def _piece_trace(self, piece: _Piece) -> float:
        return float(piece.a @ self._v[-1] + np.sum(self._Q_inverse_data * piece.UOU))

    def _piece_diagonal(self, piece: _Piece) -> NDArray:
        """``diag(H^-1 O)`` by the row-pass closed form (§3.6)."""
        tree, depth = self._tree, self._tree.depth
        delta: list[Any] = [None] * depth
        delta[-1] = piece.a
        for level in reversed(range(1, depth)):
            delta[level - 1] = _to_parent(tree, level, self._rho[level] * delta[level])
        MV = tree.subtree_sum(piece.V)
        Q_inverse = self._Q_inverse_data
        diagonal = np.empty(self.shape[0])
        diagonal[self.structured_indices] = np.concatenate(
            [
                kappa * d - np.sum((F @ Q_inverse) * MV_level, axis=1)
                for kappa, d, F, MV_level in zip(self._kappa, delta, self._F, MV, strict=True)
            ]
        )
        diagonal[self.small_indices] = np.einsum(
            "ij,ji->i", Q_inverse, piece.UOU + piece.V.T @ (self._mean - self._e_leaf)
        )
        return diagonal

    def _identity_diagonal(self, retained: NDArray) -> NDArray:
        """``diag(H^+ (H - S)) = diag(H^+ H) - diag(H^+ S)`` for the factor's own data operator.

        ``diag(H^+ H)`` is 1 on the tree and ``retained`` on the border (1
        unless truncated; ``_retained`` in the coordinates reported).
        """
        diagonal = self._inverse_diagonal()
        edf = np.empty(self.shape[0])
        edf[self.structured_indices] = (
            1.0 - np.concatenate(self._penalty) * diagonal[self.structured_indices]
        )
        edf[self.small_indices] = retained - np.sum(
            self._Q_inverse * self.operator.border_penalty, axis=1
        )
        return edf

    def _identity_square_diagonal(self, retained: NDArray) -> NDArray:
        """``diag((H^+ (H - S))^2)`` through the penalty sandwich ``H^+ S H^+`` (§3.6).

        ``(P - H^+ S)^2 = P - 2 H^+ S + H^+ S H^+ S`` with the projector ``P =
        H^+ H`` (``H^+ S P = H^+ S`` because a truncated direction is
        unpenalized), so the border reads ``retained`` in place of 1.
        Tree entries ``(H^-1 S H^-1)_uu = N_u + 2 W_u Q^-1 F_u' + F_u P F_u'`` with
        ``N_u = sum_v lambda_v Z_uv^2`` from the tree covariances (own,
        descendant, ancestor and incomparable nodes, O(n_nodes depth)), ``W =
        T^-1 Lambda F`` and ``P = Q^-1 (F'Lambda F + S_b) Q^-1``; border entries
        are ``P``.
        """
        lam = self._penalty
        h, lam_all = np.concatenate(self._selector_h(lam)), np.concatenate(lam)
        Q_inverse = self._Q_inverse
        weighted = self._tree_solve(
            [lam_level[:, None] * F for lam_level, F in zip(lam, self._F, strict=True)]
        )
        core = self.operator.border_penalty + sum(
            (F.T @ (lam_level[:, None] * F) for lam_level, F in zip(lam, self._F, strict=True)),
            start=np.zeros_like(self.operator.border_penalty),
        )
        P = Q_inverse @ core @ Q_inverse
        # beside F and W = T^-1 Lambda F (data: both vanish on the deflated
        # structural nulls) the tree entries read the data-side inverse
        Q_data = self._Q_inverse_data
        P_data = P if Q_data is Q_inverse else Q_data @ core @ Q_data
        sandwich = []
        for level, table in enumerate(self._ancestor):
            above, below = table[:-1], table[1:]
            mass = (
                h[above]
                - lam_all[below] * self._sigma_all[below] ** 2
                - self._rho_all[below] ** 2 * h[below]
            )
            ancestors = np.sum(
                self._path[level] ** 2
                * (lam_all[above] * self._kappa_all[above] ** 2 + self._v_all[above] ** 2 * mass),
                axis=0,
            )
            F = self._F[level]
            sandwich.append(
                lam[level] * self._Z_diag[level] ** 2
                + self._kappa[level] ** 2 * h[table[-1]]
                + self._sigma[level] ** 2 * ancestors
                + 2.0 * np.sum((weighted[level] @ Q_data) * F, axis=1)
                + np.sum((F @ P_data) * F, axis=1)
            )
        diagonal = self._inverse_diagonal()[self.structured_indices]
        S_b = self.operator.border_penalty
        result = np.empty(self.shape[0])
        result[self.structured_indices] = (
            1.0 - 2.0 * lam_all * diagonal + lam_all * np.concatenate(sandwich)
        )
        result[self.small_indices] = (
            retained - 2.0 * np.sum(Q_inverse * S_b, axis=1) + np.sum(P * S_b, axis=1)
        )
        return result

    def trace_inverse_operator(self, operator: CompactSymmetricOperator) -> float:
        """Return ``tr(H^-1 O)``: ``sum_l a_l v_l + <Q^-1, U'OU>`` (§3.6).

        ``U'OU = W_a + dev'e + e'dev + sum_l a_l e_l e_l'`` from the row-pass
        quantities about the factor's own leaf means, never ``A_O`` minus
        path-sum products.  For ``operator.data`` it returns ``p - tr(H^-1 S)``,
        the sum of the identity-route ``inverse_operator_diagonal``.
        """
        if operator is self.operator.data:
            return float(np.sum(self._identity_diagonal(self._retained_diagonal)))
        record = self._direction_of(None, 0.0, operator)
        value = 0.0 if record.piece is None else self._piece_trace(record.piece)
        return value + sum(_low_rank_trace(self.solve, piece) for piece in record.low_rank)

    def inverse_operator_diagonal(self, operator: CompactSymmetricOperator) -> NDArray:
        """Return ``diag(H^-1 O)`` ``(p,)`` (§3.6, decision 3).

        The factor's own data operator takes the identity route: ``1 -
        lambda_u (H^-1)_uu`` on the tree, ``1 - diag(Q^-1 S_b)`` on the border.
        Other leaf-form operators take the row-pass closed form: tree ``u``:
        ``kappa_u delta_u - F_u Q^-1 (M'V)_u'`` with ``V = dev + a (.) e``,
        ``delta_l = a_l``, ``delta_u = sum_c rho_c delta_c``; border
        ``diag(Q^-1 (U'OU + V'(MF)))``.  With an intercept these are centred
        values ``N = diag(H_c^-1 O_c)``; ``R N R^-1`` adds ``c_j N_j0`` to border
        entry ``j`` and ``-c' N_b0`` to the intercept.
        """
        if operator is self.operator.data:
            return self._identity_diagonal(self._retained_diagonal)
        record = self._direction_of(None, 0.0, operator)
        diagonal = np.zeros(self.shape[0])
        if record.piece is not None:
            diagonal = self._piece_diagonal(record.piece)
            if self.intercept:
                column = self._intercept_column_solve(record.piece)[self.small_indices]
                diagonal[self.small_indices] += self._center * column
                diagonal[self.small_indices[0]] -= self._center @ column
        for piece in record.low_rank:
            diagonal += _low_rank_diagonal(self.solve, piece)
        return diagonal

    def inverse_operator_square_diagonal(self, operator: CompactSymmetricOperator) -> NDArray:
        """Return ``diag((H^-1 O)^2)`` ``(p,)`` (§3.6, decision 3).

        Own data operator: ``1 - 2 lambda_u Z^H_uu + lambda_u (H^-1 S H^-1)_uu``
        on the tree and ``1 - 2 (Q^-1 S_b)_jj + ((H^-1 S H^-1)_bb S_b)_jj`` on the
        border, from the penalty sandwich; O(n_nodes d^2 + n_nodes q^2).  Other
        operators take ``sum_v (O H^-1)_vi (H^-1 O)_vi`` from explicit columns
        (``_square_diagonal_by_columns``), O(p (n_nodes q + q^2)) and limited to
        the forward error of the solves; the exact-arithmetic decomposition
        ``sum_v (H^-1 O H^-1)_uv O_vu`` it evaluates has terms of size
        ``1/lambda`` that cancel to a result of size ``1/w`` when ``lambda <<
        omega`` (§10, measured 1.9e-3).  No production caller passes another
        operator.
        """
        if operator is self.operator.data:
            return self._identity_square_diagonal(self._retained_diagonal)
        self._pieces(operator)
        return _square_diagonal_by_columns(self.solve, operator.matvec, self.shape[0])

    def operator_cross_trace(
        self,
        left: CompactSymmetricOperator,
        right: CompactSymmetricOperator,
    ) -> float:
        """Return ``tr(H^-1 O_l H^-1 O_r)`` (§3.6).

        ``tr(Zhat diag(a_l) Zhat diag(a_r)) + <Q^-1, V_r' Zhat V_l> + <Q^-1, V_l'
        Zhat V_r> + <Q^-1 (U'O_lU) Q^-1, U'O_rU>`` with ``Zhat = M Z M'``, the
        first term by the O(n_nodes) ``A_x`` recursion and ``Zhat V`` by one
        tree solve with ``q`` right-hand sides.
        """
        records = [self._direction_of(None, 0.0, left), self._direction_of(None, 0.0, right)]
        return float((self._cross_matrix(records) + _low_rank_matrix(self.solve, records))[0, 1])

    def penalty_operator_cross_trace(
        self,
        component: PenaltyComponent,
        scale: float,
        operator: CompactSymmetricOperator,
    ) -> float:
        """Return ``tr(H^-1 (scale Omega) H^-1 O)`` (§3.6).

        Level ``I``: ``scale sum_{i in I} [(Z M'diag(a)M Z)_ii - 2 W_i Q^-1 F_i' +
        F_i P F_i']`` with the tree-covariance closed form for the first term
        (never explicit inverse columns).  Border: ``scale <Q^-1 Omega Q^-1,
        U'OU>``.
        """
        records = [
            self._direction(component, scale, (), ()),
            self._direction_of(None, 0.0, operator),
        ]
        return float((self._cross_matrix(records) + _low_rank_matrix(self.solve, records))[0, 1])

    def derivative_cross_traces(self, directions: Sequence[DerivativeDirection]) -> NDArray:
        """Return the symmetric ``(m, m)`` matrix ``tr(H^-1 O_i H^-1 O_j)``.

        ``O_i = scale_i Omega_i + dH_i``: ``Omega_i`` an identity component on
        exactly one chain level or a component wholly in the border, ``dH_i``
        ``None`` or an accepted operator.  By bilinearity each entry is a sum of
        the four method formulas above; per direction ``Zhat V_i``, ``V_i
        Q^-1``, ``P_i = Q^-1 (U'O_iU) Q^-1``, ``W_i = T^-1 M'V_i`` and the
        ``A_x`` accumulations are formed once, and per level ``G_I``,
        ``T^-1 E_I F`` and ``h^I``.  O(m (K q^2 + q^3) + m^2 (K q + q^2 +
        n_nodes)).  This is the ``DerivativeCrossTraceFactor`` method that
        ``reml_direct_hessian`` dispatches on.
        """
        records = [
            self._direction_of(component, scale, operator)
            for component, scale, operator in directions
        ]
        return self._cross_matrix(records) + _low_rank_matrix(self.solve, records)

    def coefficient_estimable(self) -> NDArray:
        """Return per-coefficient estimability ``(p,)`` after the border truncation.

        All ``True`` without a null direction; otherwise the null basis of ``H``
        in raw coordinates, ``[-F z; z]`` with the intercept entry mapped by
        ``R = I - e_0 c'``, its border rows scaled by ``sqrt(diag Q)`` (the
        Jacobi scaling the rank decisions use, so they carry the scaled null
        vectors' eps-level noise), passed to
        ``geometry._coefficient_estimable_from_null_basis``.  A coordinate
        touches the null space in either scaling.
        """
        from superglm.solvers._structured.geometry import _coefficient_estimable_from_null_basis

        null = self._null_border
        if not null.shape[1]:
            return np.ones(self.shape[0], dtype=bool)
        raw = null.copy()
        raw[0] -= self._center @ null
        null_basis = np.zeros((self.shape[0], null.shape[1]))
        null_basis[self.small_indices] = self._border_root[:, None] * raw
        null_basis[self.structured_indices] = -np.concatenate(self._F) @ null
        return _coefficient_estimable_from_null_basis(self.shape[0], null_basis)

    def scaled_schur_eigenvalues(self) -> NDArray:
        """Return the ascending eigenvalues of the scaled deflated border matrix, cached.

        ``Q_s = D_s Q''' D_s`` after the super-root and the structural
        deflation (§3.6), exact-null rows and columns zero.  The public
        curvature check the observed-geometry build applies in place of reading
        a private ``Q`` (§6): it refuses the iterate when the smallest is below
        ``-1e-10 max(|eigenvalues|)``.
        """
        if self._scaled_eigenvalues_cache is None:
            self._scaled_eigenvalues_cache = np.linalg.eigvalsh(self._scaled_matrix)
        return self._scaled_eigenvalues_cache


class ProfiledNestedSchurFactor:
    """Slope inverse of a nested fit with the intercept profiled out (§3.6, §6).

    ``augmented_factor`` factors ``H_aug`` on ``[1, X]`` (``intercept=True``);
    this adapter exposes ``H_c^-1``, the slope block of ``H_aug^-1``, through the
    Hessian-factor protocol.  With ``e0`` the intercept coordinate, ``u = (1,
    mean_x)`` and ``P = [-mean_x'; I]`` these identities hold exactly and are
    the whole adapter:

        H_aug^-1 = e0 e0' / sum_w + P H_c^-1 P',   H_aug^-1 u = e0 / sum_w,
        G := H_aug^-1 - e0 e0' / sum_w = P H_c^-1 P',   H_c^-1 P' = E' G,
        O_c = P' O_aug P for an operator centred on mean_x.

    So for a slope penalty ``Omega`` (embedded with a zero intercept row) and
    centred operators ``O_c = P' O_aug P`` with ``y = O_aug e0`` and ``t =
    (O_aug)_00``:

        tr(H_c^-1 Omega) and all penalty pairs: the augmented values (index shift);
        tr(H_c^-1 O_c) = tr(H_aug^-1 O_aug) - t / sum_w;
        tr(H_c^-1 Omega H_c^-1 O_c) = tr(H_aug^-1 Omega H_aug^-1 O_aug);
        tr(H_c^-1 O_c H_c^-1 O'_c) = tr(H_aug^-1 O_aug H_aug^-1 O'_aug)
                                      - 2 y' H_aug^-1 y' / sum_w + t t' / sum_w^2;
        diag(H_c^-1 O_c)_j = (H_aug^-1 O_aug)_(j+1, j+1) - mean_x[j] (H_aug^-1 y)_(j+1).

    ``O_aug`` is ``raw.augmented()`` for a ``CenteredBlockOperator`` whose
    ``raw`` is a ``NestedDataOperator`` in slope coordinates.  The augmented
    factor evaluates every piece in its centred coordinates (centre ``c``,
    0 on the tree), where the last identity reads ``N_(j+1, j+1) - (mean_x[j]
    - c[j]) N_(j+1, 0)`` with ``N = H_aug^-1 O_aug`` centred: both terms are
    free of the columns' offsets.  The corrections carry no ``1/lambda`` term: the near-null
    direction of ``H_aug`` (intercept against the roots) is annihilated by
    ``O_aug`` and orthogonal to ``y``.  ``tests/test_nested_schur_factor.py``
    verifies these routes against an exact centred reference (the §8
    verification checked the augmented factor only).

    Accepted operators, in slope coordinates: a ``CenteredBlockOperator`` whose
    ``center`` is bitwise ``mean_x`` and whose ``raw`` is a ``NestedDataOperator``
    accepted by the augmented factor after augmentation; ``data_operator``
    itself passed raw, which is wrapped as ``CenteredBlockOperator(raw,
    cross=xtw, total=sum_w, center=mean_x)``, the implicit centring
    ``irls_direct``'s ``p_eff`` relies on; ``LowRankSymmetricOperator`` pieces through profiled solves; and
    ``SumBlockOperator`` of those.  Any other raw ``NestedDataOperator``, or a
    centred operator about any other center, raises ``ValueError`` (its
    correction would carry ``1/lambda`` terms); other kinds raise
    ``TypeError``.  The centred ``data_operator``
    (``StructuredLinearSystemState.centered_data_operator``, the ``edf`` and
    ``edf1`` caller in ``state_ops``) takes the identity routes, recognised by
    ``raw is data_operator`` with ``center``, ``cross`` and ``total`` equal to
    ``mean_x``, ``xtw`` and ``sum_w``.

    Attributes (the named contracts of §6): ``augmented_factor``, ``sum_w``,
    ``xtw`` ``(p,)``, ``mean_x = xtw / sum_w`` (``w_derivatives`` reads
    ``mean_x`` and ``sum_w`` by ``getattr`` and silently uncentres without
    them), ``data_operator``, ``shape = (p, p)``, ``backend = "structured"``,
    ``rank = max(augmented rank - 1, 0)``, ``rank_truncated``,
    ``schur_condition_estimate``, ``minimum_local_diagonal``, ``dominant_group_name``,
    ``chain_group_indices``, ``chain_group_names``, ``small_indices`` and
    ``structured_indices`` (slope coordinates, ``augmented - 1`` without the
    intercept), ``max_structured_inverse_block``.  ``logdet()`` is
    ``augmented logdet - log(sum_w)``.  ``ValueError`` when ``sum_w`` is not
    positive and finite or the widths disagree.
    """

    backend = "structured"
    shape: tuple[int, int]
    augmented_factor: NestedSchurFactor
    sum_w: float
    xtw: NDArray
    mean_x: NDArray
    data_operator: NestedDataOperator
    rank: int
    rank_truncated: bool
    schur_condition_estimate: float
    minimum_local_diagonal: float
    dominant_group_name: str
    chain_group_indices: tuple[int, ...]
    chain_group_names: tuple[str, ...]
    small_indices: NDArray
    structured_indices: NDArray
    max_structured_inverse_block: int

    def __init__(
        self,
        *,
        augmented_factor: NestedSchurFactor,
        sum_w: float,
        xtw: NDArray,
        data_operator: NestedDataOperator,
    ):
        self.augmented_factor = augmented_factor
        self.sum_w = float(sum_w)
        self.xtw = _frozen(xtw, np.float64)
        if not np.isfinite(self.sum_w) or self.sum_w <= 0.0:
            raise ValueError("sum_w must be positive and finite.")
        p = len(self.xtw)
        if augmented_factor.shape != (p + 1, p + 1) or data_operator.shape != (p, p):
            raise ValueError("Augmented factor, data operator and xtw widths disagree.")
        if not augmented_factor.intercept:
            raise ValueError("The augmented factor must carry the intercept in border column 0.")
        self.data_operator = data_operator
        self.shape = (p, p)
        self.mean_x = _frozen(self.xtw / self.sum_w, np.float64)
        self.rank = max(int(augmented_factor.rank) - 1, 0)
        self.rank_truncated = self.rank < p
        self.schur_condition_estimate = augmented_factor.schur_condition_estimate
        self.minimum_local_diagonal = augmented_factor.minimum_local_diagonal
        self.dominant_group_name = augmented_factor.dominant_group_name
        self.chain_group_indices = augmented_factor.chain_group_indices
        self.chain_group_names = augmented_factor.chain_group_names
        self.small_indices = augmented_factor.small_indices[1:] - 1
        self.structured_indices = augmented_factor.structured_indices - 1
        self.max_structured_inverse_block = augmented_factor.max_structured_inverse_block
        self.border_certificate = augmented_factor.border_certificate
        self.weakly_identified_slopes = tuple(
            index - 1 for index in augmented_factor.weakly_identified_coefficients if index > 0
        )
        self._centered_data = CenteredBlockOperator(
            raw=data_operator, cross=self.xtw, total=self.sum_w, center=self.mean_x
        )
        # the augmented factor's centre in slope coordinates (0 on the tree)
        self._center = np.zeros(p)
        self._center[self.small_indices] = augmented_factor._center[1:]
        # diag(H_aug^+ H_aug), whose slope entries are diag(H_c^+ H_c): the
        # super-root keeps the intercept H-orthogonal to every null
        # (H_aug^+ H_aug e0 = e0), so the diagonal does not depend on the centre.
        self._retained_about_mean = augmented_factor._retained_own
        self._state_format = _NESTED_STATE_FORMAT

    def __getstate__(self) -> dict:
        return _pending_nested_state(self)

    def __setstate__(self, state: dict) -> None:
        _restore_nested_state(self, state)

    def __getattr__(self, name: str):
        return _rebuild_on_first_use(self, name, self._rebuild)

    def _rebuild(self, state: dict) -> None:
        """Rebuild an adapter pickled by another build around its (rebuilt) augmented factor (§3.12)."""
        ProfiledNestedSchurFactor.__init__(
            self,
            augmented_factor=state["augmented_factor"],
            sum_w=state["sum_w"],
            xtw=state["xtw"],
            data_operator=state["data_operator"],
        )

    @staticmethod
    def _shift_component(component: PenaltyComponent) -> PenaltyComponent:
        start, stop = component.group_sl.start, component.group_sl.stop
        if start is None or stop is None:
            raise ValueError("Penalty component slices must have explicit bounds.")
        return replace(component, group_sl=slice(start + 1, stop + 1, component.group_sl.step))

    def _augment(self, operator: CenteredBlockOperator) -> NestedDataOperator:
        """``O_aug`` of a centred slope operator, checked like any augmented-factor operator.

        The augmented factor's forest, partitions and bitwise leaf means must
        match (``ValueError`` otherwise): a raw operator built about other leaf
        means would give silently wrong row-pass quantities.
        """
        augmented = cast(NestedDataOperator, operator.raw).augmented()
        self.augmented_factor._check_operator(augmented)
        return augmented

    def _pieces(self, operator) -> tuple[tuple[CenteredBlockOperator, ...], tuple]:
        """Split a slope operator into centred leaf-form and low-rank parts."""
        if operator is None:
            return (), ()
        if isinstance(operator, SumBlockOperator):
            parts = [self._pieces(item) for item in operator.operators]
            return (
                tuple(itertools.chain.from_iterable(leaf for leaf, _ in parts)),
                tuple(itertools.chain.from_iterable(low for _, low in parts)),
            )
        if operator.shape != self.shape:
            raise ValueError("Operator and factor dimensions must match.")
        if isinstance(operator, LowRankSymmetricOperator):
            return (), (operator,)
        if isinstance(operator, NestedDataOperator):
            if operator is not self.data_operator:
                raise ValueError(
                    "Only the factor's own data operator may be passed raw; centre other "
                    "nested operators on mean_x."
                )
            return (self._centered_data,), ()
        if isinstance(operator, CenteredBlockOperator) and isinstance(
            operator.raw, NestedDataOperator
        ):
            if not np.array_equal(operator.center, self.mean_x):
                raise ValueError("Centred nested operator must be centred on the factor's mean_x.")
            return (operator,), ()
        raise TypeError(
            f"ProfiledNestedSchurFactor does not represent {type(operator).__name__} operators."
        )

    def _is_centered_data(self, centered: tuple, low_rank: tuple) -> bool:
        if len(centered) != 1 or low_rank:
            return False
        operator = centered[0]
        return (
            operator.raw is self.data_operator
            and operator.total == self.sum_w
            and np.array_equal(operator.cross, self.xtw)
        )

    def _direction(self, component, scale, operator) -> tuple[_Direction, _Direction, float]:
        """Return the augmented record, the slope record and ``t = sum (O_aug)_00``."""
        centered, low_rank = self._pieces(operator)
        shifted = None if component is None else self._shift_component(component)
        augmented = tuple(self._augment(item) for item in centered)
        # the augmented operators have given the record its piece, and only the
        # slope record's operators meet low-rank columns (``_cross_matrix``): a
        # record that kept them would hold each direction's [1 | X] leaf copies
        # through the whole cross matrix
        record = replace(
            self.augmented_factor._direction(shifted, scale, augmented, ()), operators=()
        )
        slope = _Direction(component, float(scale), None, None, None, centered, low_rank)
        return record, slope, float(sum(item.total for item in centered))

    def _cross_matrix(self, records: Sequence[tuple]) -> NDArray:
        """``tr(H_c^-1 O_i H_c^-1 O_j)`` for parsed directions by the class-docstring identity."""
        augmented = [record for record, _, _ in records]
        slopes = [slope for _, slope, _ in records]
        totals = np.array([total for _, _, total in records])
        factor = self.augmented_factor
        traces = factor._cross_matrix(augmented)
        zero = np.zeros(factor.shape[0])
        pieces = [record.piece for record in augmented]
        Y = np.column_stack([zero if p is None else factor._intercept_column(p) for p in pieces])
        HY = np.column_stack(
            [zero if p is None else factor._intercept_column_solve(p) for p in pieces]
        )
        quadratic = Y.T @ HY
        traces = traces - (quadratic + quadratic.T) / self.sum_w
        # (each total over sum_w first: sum_w^2 overflows past ~1.3e154)
        means = totals / self.sum_w
        traces = traces + np.outer(means, means)
        return 0.5 * (traces + traces.T) + _low_rank_matrix(self.solve, slopes)

    def solve(self, rhs: NDArray) -> NDArray:
        """Return ``H_c^-1 rhs``: the augmented solve of ``[0; rhs]``, intercept row dropped."""
        values = np.asarray(rhs, dtype=np.float64)
        columns = values[:, None] if values.ndim == 1 else values
        if columns.ndim != 2 or columns.shape[0] != self.shape[0]:
            raise ValueError(
                f"rhs must have shape ({self.shape[0]},) or ({self.shape[0]}, m), "
                f"got {values.shape}."
            )
        augmented = np.zeros((self.shape[0] + 1, columns.shape[1]))
        augmented[1:] = columns
        solution = self.augmented_factor.solve(augmented)[1:]
        return solution[:, 0] if values.ndim == 1 else solution

    def logdet(self) -> float:
        """Return ``augmented_factor.logdet() - log(sum_w)``."""
        return float(self.augmented_factor.logdet() - np.log(self.sum_w))

    def selected_inverse_diagonal(self, indices: NDArray) -> NDArray:
        """Augmented ``selected_inverse_diagonal`` at ``indices + 1``."""
        return self.augmented_factor.selected_inverse_diagonal(
            np.asarray(indices, dtype=np.intp) + 1
        )

    def selected_inverse_block(self, indices: NDArray) -> NDArray:
        """Augmented ``selected_inverse_block`` at ``indices + 1`` (same cap)."""
        return self.augmented_factor.selected_inverse_block(np.asarray(indices, dtype=np.intp) + 1)

    def trace_inverse_penalty(self, component: PenaltyComponent) -> float:
        """Augmented ``trace_inverse_penalty`` of the component shifted by one."""
        return self.augmented_factor.trace_inverse_penalty(self._shift_component(component))

    def penalty_cross_trace(
        self,
        left: PenaltyComponent,
        right: PenaltyComponent,
        left_scale: float,
        right_scale: float,
    ) -> float:
        """Augmented ``penalty_cross_trace`` of the shifted components."""
        return self.augmented_factor.penalty_cross_trace(
            self._shift_component(left), self._shift_component(right), left_scale, right_scale
        )

    def trace_inverse_operator(self, operator: CompactSymmetricOperator) -> float:
        """Return ``tr(H_c^-1 O_c)`` by the class-docstring identity."""
        centered, low_rank = self._pieces(operator)
        if self._is_centered_data(centered, low_rank):
            return float(
                np.sum(self.augmented_factor._identity_diagonal(self._retained_about_mean)[1:])
            )
        value = sum(_low_rank_trace(self.solve, piece) for piece in low_rank)
        if centered:
            piece = self.augmented_factor._merged_piece([self._augment(item) for item in centered])
            total = sum(item.total for item in centered)
            value += self.augmented_factor._piece_trace(piece) - total / self.sum_w
        return float(value)

    def inverse_operator_diagonal(self, operator: CompactSymmetricOperator) -> NDArray:
        """Return ``diag(H_c^-1 O_c)`` ``(p,)``; identity route for the centred data operator."""
        centered, low_rank = self._pieces(operator)
        if self._is_centered_data(centered, low_rank):
            return self.augmented_factor._identity_diagonal(self._retained_about_mean)[1:]
        diagonal = np.zeros(self.shape[0])
        for piece in low_rank:
            diagonal += _low_rank_diagonal(self.solve, piece)
        if centered:
            factor = self.augmented_factor
            piece = factor._merged_piece([self._augment(item) for item in centered])
            diagonal += factor._piece_diagonal(piece)[1:]
            diagonal -= (self.mean_x - self._center) * factor._intercept_column_solve(piece)[1:]
        return diagonal

    def inverse_operator_square_diagonal(self, operator: CompactSymmetricOperator) -> NDArray:
        """Return ``diag((H_c^-1 O_c)^2)`` ``(p,)``.

        The centred data operator takes the augmented identity route restricted
        to the slopes (``H_c^-1 S H_c^-1 S = E' H_aug^-1 S H_aug^-1 S E``); other
        operators the explicit-column route with its documented caveat.
        """
        centered, low_rank = self._pieces(operator)
        if self._is_centered_data(centered, low_rank):
            return self.augmented_factor._identity_square_diagonal(self._retained_about_mean)[1:]
        return _square_diagonal_by_columns(self.solve, operator.matvec, self.shape[0])

    def operator_cross_trace(
        self,
        left: CompactSymmetricOperator,
        right: CompactSymmetricOperator,
    ) -> float:
        """Return ``tr(H_c^-1 O_l H_c^-1 O_r)`` by the class-docstring identity."""
        records = [self._direction(None, 0.0, left), self._direction(None, 0.0, right)]
        return float(self._cross_matrix(records)[0, 1])

    def penalty_operator_cross_trace(
        self,
        component: PenaltyComponent,
        scale: float,
        operator: CompactSymmetricOperator,
    ) -> float:
        """Return ``tr(H_c^-1 (scale Omega) H_c^-1 O_c)``: the augmented value."""
        records = [self._direction(component, scale, None), self._direction(None, 0.0, operator)]
        return float(self._cross_matrix(records)[0, 1])

    def derivative_cross_traces(self, directions: Sequence[DerivativeDirection]) -> NDArray:
        """Return ``tr(H_c^-1 O_i H_c^-1 O_j)`` for every pair of slope directions.

        The augmented factor's matrix for the augmented directions, minus
        ``2 Y' H_aug^-1 Y / sum_w`` and plus ``t t' / sum_w^2``, with ``Y`` the
        columns ``y_i = O_aug,i e0`` (zero for penalty-only directions) and
        ``t_i`` their intercept entries; ``H_aug^-1 y_i`` is the closed-form
        column of ``_intercept_column_solve``, no solve.
        """
        records = [
            self._direction(component, scale, operator) for component, scale, operator in directions
        ]
        return self._cross_matrix(records)

    def coefficient_estimable(self) -> NDArray:
        """Augmented ``coefficient_estimable`` without the intercept entry."""
        return self.augmented_factor.coefficient_estimable()[1:]

    def scaled_schur_eigenvalues(self) -> NDArray:
        """Augmented ``scaled_schur_eigenvalues``."""
        return self.augmented_factor.scaled_schur_eigenvalues()
