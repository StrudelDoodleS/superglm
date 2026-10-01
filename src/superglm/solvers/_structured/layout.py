"""Coefficient layouts and design products for structured systems."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cached_property

import numpy as np
from numpy.typing import NDArray

from superglm._group_matrix._group_matrix_execution import MatrixExecutionPlan
from superglm.group_matrix import (
    CategoricalGroupMatrix,
    DenseGroupMatrix,
    DesignMatrix,
    DiscretizedSSPGroupMatrix,
    FactorSmoothGroupMatrix,
    GroupMatrix,
    RandomEffectGroupMatrix,
    SparseSSPGroupMatrix,
)
from superglm.solvers._structured.nested import (
    _TABLE_TYPES,
    LeafRows,
    NestedStructuredLayout,
    NestedTree,
    _leaf_codes,
    _lineage_entry,
)
from superglm.solvers._structured.selection import (
    cached_nested_parent_codes,
    shared_nesting_cache,
)
from superglm.types import GroupSlice

# Border blocks whose rows an ``fs`` leaf pass writes as one-hot codes, by
# exact type: a categorical (its base level a zero row) and a random effect.
_FS_ONE_HOT_TYPES = (CategoricalGroupMatrix, RandomEffectGroupMatrix)


@dataclass(frozen=True, eq=False)
class FactorSmoothLeafLayout:
    """Design partition of one ``fs`` FactorSmooth term beside its border (one-engine design §3.4).

    A depth-1 block tree under the super-root: each FactorSmooth level is a
    leaf carrying ``k`` coefficients, and every other group joins the border
    (``_border_partition``).  ``leaf_order`` holds the rows sorted by level
    (stable) and ``leaf_starts`` ``(K + 1,)`` each level's start in it, so a
    level's rows are contiguous in the leaf pass.  ``leaf_rows`` writes every
    border block as dense rows in that order (one-hot blocks from their
    codes, so no border block is densified whole), and ``basis_table`` is a
    discrete term's support rows in the natural basis.  The border fields
    (``small_*``, ``local_groups``, ``dense_small_matrix``) are
    ``_border_partition``'s; ``indicator_columns`` is all
    ``False`` (no border block takes cells here) and ``sparse_indicators`` is
    empty, the attributes the shared border-row helpers read.

    Caches (the cached properties and the lineage entries): the cached
    properties are owned by this immutable layout, which lives in its design's
    layout cache and is never pickled; the lineage entries by the shared
    nesting cache.  They depend on the design only (codes, basis, border
    matrices; prior weights for the centres), never on a working weight,
    lambda or penalty; the leaf memo alone is keyed by the exact working rows.
    """

    dominant_group_index: int
    dominant_group_name: str
    dominant: FactorSmoothGroupMatrix
    small_group_indices: tuple[int, ...]
    small_matrices: tuple[GroupMatrix, ...]
    local_groups: tuple[GroupSlice, ...]
    small_indices: NDArray
    structured_indices: NDArray
    dense_small_matrix: NDArray | None
    small_execution_plan: MatrixExecutionPlan | None
    leaf_order: NDArray
    leaf_starts: NDArray
    # The lineage's nesting cache (``selection.shared_nesting_cache``): the level
    # order, the sorted basis, the support table, the prior-weight centres and
    # the leaf memo live there (``block_leaves._lineage_slot``), matched on the
    # objects they depend on (``lineage_sources``), so a lambda rebuild of the
    # design that keeps them shares them.
    lineage_cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        for name in ("small_indices", "structured_indices", "leaf_order", "leaf_starts"):
            values = np.array(getattr(self, name), dtype=np.intp, copy=True)
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        if self.dense_small_matrix is not None:
            dense = np.asarray(self.dense_small_matrix, dtype=np.float64)
            dense.setflags(write=False)
            object.__setattr__(self, "dense_small_matrix", dense)

    @property
    def n_levels(self) -> int:
        return int(self.dominant.coefficient_levels)

    @property
    def leaf_count(self) -> int:
        """Every level of the term (``sz``: ``K``, one more than its public levels)."""
        return int(self.dominant.n_levels)

    @property
    def block_size(self) -> int:
        return int(self.dominant.block_size)

    @property
    def width(self) -> int:
        return len(self.small_indices)

    @property
    def sparse_indicators(self) -> tuple:
        return ()

    def thin_levels(self, prior_weights: NDArray | None) -> tuple:
        """The ``sz`` levels with fewer than ``m`` distinct weighted ``x`` values (design decision 4).

        Beside the required global Spline such a level makes an exact alias of
        the main-effect polynomial (the penalty's null space, dimension ``m``)
        with its deviation: flagged and kept, never refused.  Distinct ``x``
        values are distinct basis rows among the level's rows of positive
        prior weight.  A disclosure only: it routes nothing.  Held in the
        lineage cache, keyed by the sources and the prior weights.
        """
        levels, _ = self.thin_level_counts(prior_weights)
        return tuple(self.dominant.levels[level] for level in levels)

    def thin_level_counts(self, prior_weights: NDArray | None) -> tuple[NDArray, NDArray]:
        """``(levels, distinct)``: the thin levels' codes and their distinct weighted ``x`` counts.

        The structure ``thin_levels`` names, by code: the balance tree builds
        each thin level's exact data-null directions from it (``balance_tree``,
        the penalized aliases).  Held in the lineage cache, keyed by the
        sources and the prior weights.
        """
        dominant = self.dominant
        n = len(self.leaf_order)
        weights = (
            np.ones(n) if prior_weights is None else np.asarray(prior_weights, dtype=np.float64)
        )
        slot = self.lineage_cache.setdefault(("sz_slot", "thin_counts"), [])
        sources = self.lineage_sources
        for held_sources, held, value in slot:
            if (
                len(held_sources) == len(sources)
                and all(a is b for a, b in zip(held_sources, sources, strict=True))
                and np.array_equal(held, weights)
            ):
                return value
        nullity = _sz_penalty_null_space(dominant).shape[1]
        ordered = np.ascontiguousarray(weights[self.leaf_order])
        distinct = _distinct_level_counts(
            dominant, self.sorted_basis, self.leaf_starts, ordered, nullity
        )
        thin = np.flatnonzero(distinct < nullity).astype(np.intp)
        value = (thin, np.asarray(distinct[thin], dtype=np.intp))
        for array in value:
            array.setflags(write=False)
        slot.insert(0, (sources, np.array(weights, copy=True), value))
        del slot[2:]
        return value

    @cached_property
    def indicator_columns(self) -> NDArray:
        mask = np.zeros(len(self.small_indices), dtype=bool)
        mask.setflags(write=False)
        return mask

    @cached_property
    def one_hot_columns(self) -> NDArray:
        """``(q,)`` bool: the border columns of one-hot blocks (``_FS_ONE_HOT_TYPES``), by type."""
        mask = np.zeros(len(self.small_indices), dtype=bool)
        offset = 0
        for matrix in self.small_matrices:
            width = matrix.shape[1]
            if type(matrix) in _FS_ONE_HOT_TYPES:
                mask[offset : offset + width] = True
            offset += width
        mask.setflags(write=False)
        return mask

    @cached_property
    def basis_table(self) -> NDArray | None:
        """A discrete term's support rows in the natural basis ``B_unique @ natural_map``; else ``None``.

        Held in the lineage cache (keyed by the two arrays), so a lambda
        rebuild of the design that keeps them reuses it.
        """
        if not self.dominant.is_discrete:
            return None
        support, natural = self.dominant.B_unique, self.dominant.natural_map

        def build() -> NDArray:
            table = np.ascontiguousarray(np.asarray(support, dtype=np.float64) @ natural)
            table.setflags(write=False)
            return table

        return _lineage_entry(self.lineage_cache, "fs_basis_table", (support, natural), 0, build)

    @property
    def lineage_sources(self) -> tuple:
        """The objects the leaf data depend on besides the working rows (lineage-cache keys).

        The term's codes and basis arrays and every border matrix: a lambda
        rebuild of the design that passes them through unchanged shares every
        design-only cache and the leaf memo; one that re-creates any of them
        (an SSP border block re-parameterized) misses, correctly.
        """
        dominant = self.dominant
        basis = (dominant.B_unique, dominant.bin_idx) if dominant.is_discrete else (dominant.B,)
        return (dominant.codes, dominant.natural_map, *basis, *self.small_matrices)

    @cached_property
    def leaf_levels(self) -> NDArray:
        """``(n,)`` intp: each row's level in level order (the leaf pass's segment codes)."""
        counts = np.diff(self.leaf_starts)
        levels = np.repeat(np.arange(len(counts), dtype=np.intp), counts)
        levels.setflags(write=False)
        return levels

    @cached_property
    def sorted_basis(self) -> tuple[NDArray, ...]:
        """The term's basis in level order: ``(data, indices, indptr)`` of the exact CSR
        rows, or ``(bins,)`` of a discrete term; a permuted copy of the stored basis
        (same size), read sequentially by the leaf pass."""
        order = self.leaf_order
        dominant = self.dominant
        if dominant.is_discrete:
            source = dominant.bin_idx

            def build_bins() -> tuple[NDArray, ...]:
                bins = np.ascontiguousarray(np.asarray(source, dtype=np.intp)[order])
                bins.setflags(write=False)
                return (bins,)

            return _lineage_entry(
                self.lineage_cache, "fs_sorted_bins", (source, order), 0, build_bins
            )
        basis = dominant.B
        if basis is None:  # pragma: no cover - an exact term always holds its CSR basis
            raise RuntimeError("An exact FactorSmooth term has no CSR basis.")

        def build_csr() -> tuple[NDArray, ...]:
            csr = basis[order]
            arrays = (
                np.ascontiguousarray(csr.data, dtype=np.float64),
                np.ascontiguousarray(csr.indices, dtype=np.int64),
                np.ascontiguousarray(csr.indptr, dtype=np.int64),
            )
            for array in arrays:
                array.setflags(write=False)
            return arrays

        return _lineage_entry(self.lineage_cache, "fs_sorted_csr", (basis, order), 0, build_csr)

    @cached_property
    def leaf_rows(self) -> LeafRows:
        """How the leaf pass forms the dense border rows in level order (``nested.LeafRows``).

        Every border block is included; blocks are told apart by exact type,
        never by their data, as ``NestedStructuredLayout.leaf_rows`` does.
        """
        order = self.leaf_order
        empty = np.zeros(0, dtype=np.intp)
        if self.dense_small_matrix is not None:
            gathered = ((0, self.dense_small_matrix),)
            codes = np.zeros((len(order), 0), dtype=np.int32)
            return LeafRows(codes, np.zeros(0), empty, empty, empty, empty, empty, gathered)
        tables, one_hot, gathered, sparse, generic, column = [], [], [], [], [], 0
        for matrix in self.small_matrices:
            kind = type(matrix)
            if isinstance(matrix, DiscretizedSSPGroupMatrix) and kind in _TABLE_TYPES:
                table = np.asarray(matrix.B_unique @ matrix.R_inv, dtype=np.float64)
                tables.append((column, matrix.bin_idx, table))
            elif isinstance(matrix, CategoricalGroupMatrix) and kind in _FS_ONE_HOT_TYPES:
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


_MAX_FUSED_DENSE_SMALL_WIDTH = 32


def _validate_structured_inputs(
    group_matrices: list[GroupMatrix],
    groups: list[GroupSlice],
    W: NDArray,
    Wz: NDArray,
    dominant_group_index: int,
) -> tuple[NDArray, NDArray, RandomEffectGroupMatrix]:
    if len(group_matrices) != len(groups):
        raise ValueError("group_matrices and groups must have the same length.")
    if not 0 <= dominant_group_index < len(group_matrices):
        raise IndexError("dominant_group_index is outside group_matrices.")
    dominant = group_matrices[dominant_group_index]
    if not isinstance(dominant, RandomEffectGroupMatrix):
        raise ValueError("The dominant structured group must be a RandomEffectGroupMatrix.")
    weights = np.asarray(W, dtype=np.float64)
    weighted_rhs = np.asarray(Wz, dtype=np.float64)
    if weights.ndim != 1 or weighted_rhs.shape != weights.shape:
        raise ValueError("W and Wz must be one-dimensional arrays with identical shape.")
    if len(weights) != dominant.shape[0] or any(
        matrix.shape[0] != len(weights) for matrix in group_matrices
    ):
        raise ValueError("All group matrices, W, and Wz must have the same row count.")
    return weights, weighted_rhs, dominant


def _border_partition(
    group_matrices: list[GroupMatrix] | tuple[GroupMatrix, ...],
    groups: list[GroupSlice],
    structured_group_indices: tuple[int, ...],
    n: int,
) -> dict:
    """Border (small-block) fields shared by every structured layout.

    Every group outside ``structured_group_indices`` joins the border, in
    design order, with one fused dense matrix when it is narrow and all dense
    and one reusable moment plan otherwise.
    """
    small_group_indices = tuple(
        index for index in range(len(group_matrices)) if index not in structured_group_indices
    )
    small_matrices = tuple(group_matrices[index] for index in small_group_indices)
    small_ranges = tuple(
        np.arange(groups[index].start, groups[index].end, dtype=np.intp)
        for index in small_group_indices
    )
    small_indices = np.concatenate(small_ranges) if small_ranges else np.empty(0, dtype=np.intp)
    local_groups: list[GroupSlice] = []
    local_start = 0
    for index in small_group_indices:
        group = groups[index]
        local_end = local_start + group.size
        local_groups.append(replace(group, start=local_start, end=local_end))
        local_start = local_end

    dense_small_matrix = None
    small_execution_plan = None
    if (
        small_matrices
        and local_start <= _MAX_FUSED_DENSE_SMALL_WIDTH
        and all(type(matrix) is DenseGroupMatrix for matrix in small_matrices)
    ):
        dense_small_matrix = np.ascontiguousarray(
            np.column_stack([matrix.M for matrix in small_matrices]),
            dtype=np.float64,
        )
    elif small_matrices:
        small_execution_plan = MatrixExecutionPlan(
            small_matrices,
            n=n,
            ordinary_tabmat=True,
        )
        small_execution_plan.validate_group_spans(local_groups)
    return {
        "small_group_indices": small_group_indices,
        "small_matrices": small_matrices,
        "local_groups": tuple(local_groups),
        "small_indices": small_indices,
        "dense_small_matrix": dense_small_matrix,
        "small_execution_plan": small_execution_plan,
    }


def build_nested_structured_layout(
    group_matrices: list[GroupMatrix] | tuple[GroupMatrix, ...],
    groups: list[GroupSlice],
    *,
    chain_group_indices: tuple[int, ...],
    nesting_cache: dict | None = None,
) -> NestedStructuredLayout:
    """Build the tree and border partitions of one nested RandomEffect chain (§4).

    ``chain_group_indices`` runs coarsest to finest.  Every consecutive pair
    must pass the strict-nesting count test (§3.7); a level reused under
    several parents is refused with the instruction to build the interaction
    code, as lme4 documents for implicit nesting.  Unobserved child levels
    point at parent 0.  ``nesting_cache`` is the nesting cache (see
    ``shared_nesting_cache``); the tree and the leaf row order read only the
    chain's codes, so they are one of its entries and outlive a lambda
    rebuild, while the border is rebuilt.
    """
    chain = tuple(int(index) for index in chain_group_indices)
    if not chain:
        raise ValueError("A nested chain needs at least one RandomEffect group.")
    if len(group_matrices) != len(groups):
        raise ValueError("group_matrices and groups must have the same length.")
    members: dict[int, RandomEffectGroupMatrix] = {}
    for index in chain:
        group, matrix = groups[index], group_matrices[index]
        if not isinstance(matrix, RandomEffectGroupMatrix):
            raise ValueError(f"Nested chain group {group.name!r} is not a RandomEffect term.")
        if group.size != matrix.n_levels:
            raise ValueError(f"Nested chain group {group.name!r} slice does not match its levels.")
        if group.constraints is not None or group.scop_reparameterization is not None:
            raise ValueError(f"Nested chain group {group.name!r} carries constraints or SCOP.")
        members[index] = matrix
    cache = {} if nesting_cache is None else nesting_cache
    key = ("nested_tree", chain, tuple((group.name, group.start, group.end) for group in groups))
    if key not in cache:
        cache[key] = _nested_tree(members, groups, chain, cache)
    tree, leaf_order, leaf_starts = cache[key]
    return NestedStructuredLayout(
        chain_group_indices=chain,
        chain_group_names=tuple(groups[index].name for index in chain),
        tree=tree,
        level_indices=tuple(np.arange(groups[index].start, groups[index].end) for index in chain),
        leaf_order=leaf_order,
        leaf_starts=leaf_starts,
        **_border_partition(group_matrices, groups, chain, members[chain[-1]].shape[0]),
        lineage_cache=cache,
    )


def _nested_tree(
    members: dict[int, RandomEffectGroupMatrix],
    groups: list[GroupSlice],
    chain: tuple[int, ...],
    nesting_cache: dict,
) -> tuple[NestedTree, NDArray, NDArray]:
    """Return the chain's tree after the strict-nesting tests, the rows in stable
    leaf order and each leaf's start in that order ``(K + 1,)``."""
    parents = [np.full(members[chain[0]].n_levels, -1, dtype=np.intp)]
    for coarse, fine in zip(chain[:-1], chain[1:], strict=True):
        codes = cached_nested_parent_codes(members, groups, fine, coarse, nesting_cache)
        if codes is None:
            coarse_name, fine_name = groups[coarse].name, groups[fine].name
            raise ValueError(
                f"RandomEffect {fine_name!r} is not strictly nested in {coarse_name!r}: a "
                f"{fine_name!r} level occurs under several {coarse_name!r} levels. Build the "
                f"interaction code (for example '{coarse_name}:{fine_name}') and use it as "
                "the finer term."
            )
        parents.append(codes)
    tree = NestedTree(tuple(members[index].n_levels for index in chain), tuple(parents))
    codes = members[chain[-1]].codes
    order = np.argsort(codes, kind="stable")
    starts = np.concatenate(([0], np.cumsum(np.bincount(codes, minlength=tree.sizes[-1]))))
    for array in (order, starts):
        array.setflags(write=False)
    return tree, order, starts


def build_factor_smooth_leaf_layout(
    group_matrices: list[GroupMatrix] | tuple[GroupMatrix, ...],
    groups: list[GroupSlice],
    *,
    dominant_group_index: int,
    nesting_cache: dict | None = None,
) -> FactorSmoothLeafLayout:
    """Build the ``fs`` leaf layout: border partition, level order and level starts.

    The level order reads only the term's codes, so it is held in the
    lineage's nesting cache (``_lineage_entry``) and outlives a lambda rebuild.
    """
    if len(group_matrices) != len(groups):
        raise ValueError("group_matrices and groups must have the same length.")
    if not 0 <= dominant_group_index < len(group_matrices):
        raise IndexError("dominant_group_index is outside group_matrices.")
    dominant = group_matrices[dominant_group_index]
    if not isinstance(dominant, FactorSmoothGroupMatrix):
        raise ValueError("The leaf layout needs a FactorSmoothGroupMatrix.")
    dominant_group = groups[dominant_group_index]
    if dominant_group.size != dominant.coefficient_levels * dominant.block_size:
        raise ValueError("The dominant group slice does not match its factor-smooth width.")
    cache = {} if nesting_cache is None else nesting_cache
    codes = dominant.codes
    order, starts = _lineage_entry(
        cache,
        "fs_level_order",
        (codes,),
        dominant.n_levels,
        lambda: _level_order(codes, dominant.n_levels),
    )
    structured_indices = np.arange(dominant_group.start, dominant_group.end, dtype=np.intp).reshape(
        dominant.coefficient_levels, dominant.block_size
    )
    return FactorSmoothLeafLayout(
        dominant_group_index=dominant_group_index,
        dominant_group_name=dominant_group.name,
        dominant=dominant,
        structured_indices=structured_indices,
        leaf_order=order,
        leaf_starts=starts,
        lineage_cache=cache,
        **_border_partition(group_matrices, groups, (dominant_group_index,), dominant.shape[0]),
    )


def _sz_penalty_null_space(dominant: FactorSmoothGroupMatrix) -> NDArray:
    """``N_P`` ``(k, m)``: the null space of an ``sz`` term's summed level penalty.

    Its eigenvectors at or below ``k eps`` of the largest eigenvalue; in the
    natural parameterization (a diagonal penalty) the unpenalized polynomial
    coordinates, exactly.
    """
    omega = sum(
        np.asarray(matrix, dtype=np.float64) for _, matrix in dominant.repeated_penalty_components
    )
    values, vectors = np.linalg.eigh(0.5 * (omega + np.transpose(omega)))
    return vectors[:, values <= len(values) * np.finfo(np.float64).eps * values[-1]]


def _distinct_level_counts(
    dominant: FactorSmoothGroupMatrix,
    sorted_source: tuple[NDArray, ...],
    starts: NDArray,
    ordered_weights: NDArray,
    cap: int,
) -> NDArray:
    """Each level's count of distinct basis rows of positive weight, stopped at ``cap``.

    ``sorted_source`` is the basis in level order (``sorted_basis``): an exact
    term's CSR arrays or a discrete term's bins, one bin a row.
    """
    from superglm.solvers._structured.leaf_kernels import _distinct_level_rows

    if dominant.is_discrete:
        bins = sorted_source[0]
        data, indices, indptr = np.zeros(len(bins)), bins, np.arange(len(bins) + 1)
    else:
        data, indices, indptr = sorted_source
    return _distinct_level_rows(data, indices, indptr, starts, ordered_weights, cap)


def sz_unidentified_levels(
    dominant: FactorSmoothGroupMatrix, prior_weights: NDArray | None
) -> tuple[tuple[int, ...], NDArray]:
    """The ``sz`` levels the data leave unidentified, and the penalty's null space ``N_P``.

    A level whose rows of positive weight hold fewer distinct ``x`` values than
    the dimension ``m`` of ``N_P`` (none, for a level without weight) cannot
    tell part of its polynomial deviation from the main effect: the rule
    ``thin_level_counts`` applies, on the same kernel.  Backend independent:
    it reads the design and the prior weights alone, so a gram fit records
    the levels an ``sz`` balance tree names.
    """
    null_space = _sz_penalty_null_space(dominant)
    null_space.setflags(write=False)
    nullity = null_space.shape[1]
    if not nullity:
        return (), null_space
    codes = dominant.codes
    weights = (
        np.ones(len(codes)) if prior_weights is None else np.asarray(prior_weights, np.float64)
    )
    order, starts = _level_order(codes, dominant.n_levels)
    if dominant.is_discrete:
        source: tuple[NDArray, ...] = (np.ascontiguousarray(dominant.bin_idx[order]),)
    else:
        csr = dominant.B[order]
        source = (
            np.ascontiguousarray(csr.data, dtype=np.float64),
            np.ascontiguousarray(csr.indices, dtype=np.int64),
            np.ascontiguousarray(csr.indptr, dtype=np.int64),
        )
    distinct = _distinct_level_counts(
        dominant, source, starts, np.ascontiguousarray(weights[order]), nullity
    )
    return tuple(int(level) for level in np.flatnonzero(distinct < nullity)), null_space


def _level_order(codes: NDArray, n_levels: int) -> tuple[NDArray, NDArray]:
    """The rows in stable level order and each level's start ``(K + 1,)``."""
    order = np.argsort(codes, kind="stable")
    starts = np.concatenate(([0], np.cumsum(np.bincount(codes, minlength=n_levels))))
    for array in (order, starts):
        array.setflags(write=False)
    return order, starts


def get_factor_smooth_leaf_layout(
    dm: DesignMatrix,
    groups: list[GroupSlice],
    *,
    dominant_group_index: int,
) -> FactorSmoothLeafLayout:
    """Return the DesignMatrix-owned ``fs`` leaf layout reused across REML trials."""
    signature = (
        "fs_leaf",
        dominant_group_index,
        tuple((group.name, group.start, group.end) for group in groups),
    )
    cache = dm._structured_layout_cache
    layout = cache.get(signature)
    if layout is None:
        layout = build_factor_smooth_leaf_layout(
            dm.group_matrices,
            groups,
            dominant_group_index=dominant_group_index,
            nesting_cache=shared_nesting_cache(cache),
        )
        cache[signature] = layout
    return layout


def get_nested_structured_layout(
    dm: DesignMatrix,
    groups: list[GroupSlice],
    *,
    chain_group_indices: tuple[int, ...],
) -> NestedStructuredLayout:
    """Return the DesignMatrix-owned layout of one nested chain.

    Cached under the chain and the group spans in the design's layout cache;
    the chain's nesting tests and tree live in its shared nesting cache.
    """
    signature = (
        "nested",
        tuple(chain_group_indices),
        tuple((group.name, group.start, group.end) for group in groups),
    )
    cache = dm._structured_layout_cache
    layout = cache.get(signature)
    if layout is None:
        layout = build_nested_structured_layout(
            dm.group_matrices,
            groups,
            chain_group_indices=chain_group_indices,
            nesting_cache=shared_nesting_cache(cache),
        )
        cache[signature] = layout
    return layout


def get_structured_layout(
    dm: DesignMatrix,
    groups: list[GroupSlice],
    *,
    dominant_group_index: int,
    chain_group_indices: tuple[int, ...] = (),
) -> FactorSmoothLeafLayout | NestedStructuredLayout:
    """Dispatch layout construction by dominant matrix type.

    A FactorSmooth term takes its leaf layout (``fs`` block leaves, or the
    ``sz`` balance tree over the same leaves); a RandomEffect leaf takes the
    layout of its nested chain, which ``resolve_structured_backend`` resolves
    (a lone level is a chain of one).
    """
    dominant = dm.group_matrices[dominant_group_index]
    if isinstance(dominant, FactorSmoothGroupMatrix):
        # fs: a depth-1 block tree; sz: the balance tree over the same leaves (§3.4, §3.5)
        return get_factor_smooth_leaf_layout(dm, groups, dominant_group_index=dominant_group_index)
    if not chain_group_indices:
        raise ValueError(
            "A RandomEffect structured layout needs its nested chain; a lone random "
            "effect is a chain of one (resolve_structured_backend)."
        )
    return get_nested_structured_layout(dm, groups, chain_group_indices=chain_group_indices)


def _structured_row_matrix(
    layout: FactorSmoothLeafLayout | NestedStructuredLayout,
    group_matrices: list[GroupMatrix] | tuple[GroupMatrix, ...],
) -> RandomEffectGroupMatrix | FactorSmoothGroupMatrix:
    """Return the structured group whose codes touch rows (a nested chain's leaf)."""
    index = (
        layout.leaf_group_index
        if isinstance(layout, NestedStructuredLayout)
        else layout.dominant_group_index
    )
    dominant = group_matrices[index]
    if not isinstance(dominant, RandomEffectGroupMatrix | FactorSmoothGroupMatrix):
        raise ValueError("Structured layout no longer points to a structured group.")
    return dominant


def structured_design_matvec(
    layout: FactorSmoothLeafLayout | NestedStructuredLayout,
    group_matrices: list[GroupMatrix] | tuple[GroupMatrix, ...],
    beta: NDArray,
) -> NDArray:
    """Apply a grouped design while fusing a cached dense-small partition.

    A nested chain applies the leaf matrix to the path sums ``M beta_tree``
    (§6), one O(n + n_nodes) pass.
    """
    values = np.asarray(beta, dtype=np.float64)
    width = len(layout.small_indices) + layout.structured_indices.size
    if values.shape != (width,):
        raise ValueError(f"beta must have shape ({width},).")
    dominant = _structured_row_matrix(layout, group_matrices)

    if layout.dense_small_matrix is not None:
        result = layout.dense_small_matrix @ values[layout.small_indices]
    else:
        result = np.zeros(dominant.shape[0], dtype=np.float64)
        local_beta = values[layout.small_indices]
        for matrix, group in zip(
            layout.small_matrices,
            layout.local_groups,
            strict=True,
        ):
            result += matrix.matvec(local_beta[group.sl])
    structured_values = values[layout.structured_indices]
    if isinstance(layout, NestedStructuredLayout):
        result += dominant.matvec(layout.tree.path_sum(layout.tree.split(structured_values)))
    else:
        result += dominant.matvec(structured_values.ravel())
    return result


def structured_design_rmatvec(
    layout: FactorSmoothLeafLayout | NestedStructuredLayout,
    group_matrices: list[GroupMatrix] | tuple[GroupMatrix, ...],
    rows: NDArray,
) -> NDArray:
    """Apply a grouped design transpose with one cached dense-small product.

    A nested chain takes the subtree sums ``M'`` of the leaf transpose (§6).
    """
    values = np.asarray(rows, dtype=np.float64)
    dominant = _structured_row_matrix(layout, group_matrices)
    if values.shape != (dominant.shape[0],):
        raise ValueError(f"rows must have shape ({dominant.shape[0]},).")

    width = len(layout.small_indices) + layout.structured_indices.size
    result = np.empty(width, dtype=np.float64)
    if layout.dense_small_matrix is not None:
        result[layout.small_indices] = layout.dense_small_matrix.T @ values
    elif layout.small_matrices:
        result[layout.small_indices] = np.concatenate(
            [matrix.rmatvec(values) for matrix in layout.small_matrices]
        )
    else:
        result[layout.small_indices] = np.empty(0, dtype=np.float64)
    if isinstance(layout, NestedStructuredLayout):
        result[layout.structured_indices] = np.concatenate(
            layout.tree.subtree_sum(dominant.rmatvec(values))
        )
    else:
        result[layout.structured_indices] = dominant.rmatvec(values).reshape(
            layout.structured_indices.shape
        )
    return result
