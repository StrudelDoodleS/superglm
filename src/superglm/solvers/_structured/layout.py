"""Coefficient layouts and design products for structured systems."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from superglm._group_matrix._group_matrix_execution import MatrixExecutionPlan
from superglm.group_matrix import (
    DenseGroupMatrix,
    DesignMatrix,
    FactorSmoothGroupMatrix,
    GroupMatrix,
    RandomEffectGroupMatrix,
)
from superglm.solvers._structured.nested import NestedStructuredLayout, NestedTree
from superglm.solvers._structured.selection import cached_nested_parent_codes
from superglm.types import GroupSlice


@dataclass(frozen=True)
class ScalarStructuredLayout:
    """Cached coefficient partitions and small-block execution plan."""

    dominant_group_index: int
    dominant_group_name: str
    small_group_indices: tuple[int, ...]
    small_matrices: tuple[GroupMatrix, ...]
    local_groups: tuple[GroupSlice, ...]
    small_indices: NDArray
    structured_indices: NDArray
    dense_small_matrix: NDArray | None
    small_execution_plan: MatrixExecutionPlan | None

    def __post_init__(self) -> None:
        for name in ("small_indices", "structured_indices"):
            values = np.array(getattr(self, name), dtype=np.intp, copy=True)
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        if self.dense_small_matrix is not None:
            dense = np.asarray(self.dense_small_matrix, dtype=np.float64)
            dense.setflags(write=False)
            object.__setattr__(self, "dense_small_matrix", dense)


@dataclass(frozen=True)
class BlockStructuredLayout:
    """Cached coefficient partitions for one dominant factor-smooth block."""

    dominant_group_index: int
    dominant_group_name: str
    small_group_indices: tuple[int, ...]
    small_matrices: tuple[GroupMatrix, ...]
    local_groups: tuple[GroupSlice, ...]
    small_indices: NDArray
    structured_indices: NDArray
    dense_small_matrix: NDArray | None
    small_execution_plan: MatrixExecutionPlan | None

    def __post_init__(self) -> None:
        for name in ("small_indices", "structured_indices"):
            values = np.array(getattr(self, name), dtype=np.intp, copy=True)
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        if self.dense_small_matrix is not None:
            dense = np.asarray(self.dense_small_matrix, dtype=np.float64)
            dense.setflags(write=False)
            object.__setattr__(self, "dense_small_matrix", dense)


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


def build_scalar_structured_layout(
    group_matrices: list[GroupMatrix] | tuple[GroupMatrix, ...],
    groups: list[GroupSlice],
    *,
    dominant_group_index: int,
) -> ScalarStructuredLayout:
    """Build immutable partitions and one reusable small-block moment plan."""
    if len(group_matrices) != len(groups):
        raise ValueError("group_matrices and groups must have the same length.")
    if not 0 <= dominant_group_index < len(group_matrices):
        raise IndexError("dominant_group_index is outside group_matrices.")
    dominant = group_matrices[dominant_group_index]
    if not isinstance(dominant, RandomEffectGroupMatrix):
        raise ValueError("The dominant structured group must be a RandomEffectGroupMatrix.")

    dominant_group = groups[dominant_group_index]
    if dominant_group.size != dominant.n_levels:
        raise ValueError("The dominant group slice does not match its random-effect width.")
    structured_indices = np.arange(
        dominant_group.start,
        dominant_group.end,
        dtype=np.intp,
    )
    return ScalarStructuredLayout(
        dominant_group_index=dominant_group_index,
        dominant_group_name=dominant_group.name,
        structured_indices=structured_indices,
        **_border_partition(group_matrices, groups, (dominant_group_index,), dominant.shape[0]),
    )


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
    point at parent 0.  ``reference_row`` is each leaf's first training row.
    """
    chain = tuple(int(index) for index in chain_group_indices)
    if len(chain) < 2:
        raise ValueError("A nested chain needs at least two RandomEffect groups.")
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
    leaf = members[chain[-1]]
    n = leaf.shape[0]
    first_row = np.full(leaf.n_levels, n, dtype=np.intp)
    np.minimum.at(first_row, leaf.codes, np.arange(n, dtype=np.intp))
    return NestedStructuredLayout(
        chain_group_indices=chain,
        chain_group_names=tuple(groups[index].name for index in chain),
        tree=tree,
        level_indices=tuple(np.arange(groups[index].start, groups[index].end) for index in chain),
        reference_row=np.where(first_row < n, first_row, -1),
        **_border_partition(group_matrices, groups, chain, n),
    )


def get_scalar_structured_layout(
    dm: DesignMatrix,
    groups: list[GroupSlice],
    *,
    dominant_group_index: int,
) -> ScalarStructuredLayout:
    """Return a DesignMatrix-owned layout reused across REML candidate fits."""
    signature = (
        dominant_group_index,
        tuple((group.name, group.start, group.end) for group in groups),
    )
    cache = dm._scalar_structured_layout_cache
    layout = cache.get(signature)
    if layout is None:
        layout = build_scalar_structured_layout(
            dm.group_matrices,
            groups,
            dominant_group_index=dominant_group_index,
        )
        cache[signature] = layout
    return layout


def build_block_structured_layout(
    group_matrices: list[GroupMatrix] | tuple[GroupMatrix, ...],
    groups: list[GroupSlice],
    *,
    dominant_group_index: int,
) -> BlockStructuredLayout:
    """Build immutable partitions for one dominant factor-smooth term."""
    if len(group_matrices) != len(groups):
        raise ValueError("group_matrices and groups must have the same length.")
    if not 0 <= dominant_group_index < len(group_matrices):
        raise IndexError("dominant_group_index is outside group_matrices.")
    dominant = group_matrices[dominant_group_index]
    if not isinstance(dominant, FactorSmoothGroupMatrix):
        raise ValueError("The dominant block group must be a FactorSmoothGroupMatrix.")
    dominant_group = groups[dominant_group_index]
    if dominant_group.size != dominant.coefficient_levels * dominant.block_size:
        raise ValueError("The dominant group slice does not match its factor-smooth width.")

    structured_indices = np.arange(
        dominant_group.start,
        dominant_group.end,
        dtype=np.intp,
    ).reshape(dominant.coefficient_levels, dominant.block_size)
    return BlockStructuredLayout(
        dominant_group_index=dominant_group_index,
        dominant_group_name=dominant_group.name,
        structured_indices=structured_indices,
        **_border_partition(group_matrices, groups, (dominant_group_index,), dominant.shape[0]),
    )


def get_block_structured_layout(
    dm: DesignMatrix,
    groups: list[GroupSlice],
    *,
    dominant_group_index: int,
) -> BlockStructuredLayout:
    """Return a DesignMatrix-owned block layout reused across REML trials."""
    signature = (
        "block",
        dominant_group_index,
        tuple((group.name, group.start, group.end) for group in groups),
    )
    cache = dm._scalar_structured_layout_cache
    layout = cache.get(signature)
    if layout is None:
        layout = build_block_structured_layout(
            dm.group_matrices,
            groups,
            dominant_group_index=dominant_group_index,
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

    Cached under the chain and the group spans in the design's layout cache,
    which also holds the chain's nesting tests.
    """
    signature = (
        "nested",
        tuple(chain_group_indices),
        tuple((group.name, group.start, group.end) for group in groups),
    )
    cache = dm._scalar_structured_layout_cache
    layout = cache.get(signature)
    if layout is None:
        layout = build_nested_structured_layout(
            dm.group_matrices,
            groups,
            chain_group_indices=chain_group_indices,
            nesting_cache=cache,
        )
        cache[signature] = layout
    return layout


def get_structured_layout(
    dm: DesignMatrix,
    groups: list[GroupSlice],
    *,
    dominant_group_index: int,
    chain_group_indices: tuple[int, ...] = (),
) -> ScalarStructuredLayout | BlockStructuredLayout | NestedStructuredLayout:
    """Dispatch layout construction by dominant matrix type or nested chain."""
    if len(chain_group_indices) >= 2:
        return get_nested_structured_layout(dm, groups, chain_group_indices=chain_group_indices)
    dominant = dm.group_matrices[dominant_group_index]
    if isinstance(dominant, FactorSmoothGroupMatrix):
        return get_block_structured_layout(
            dm,
            groups,
            dominant_group_index=dominant_group_index,
        )
    return get_scalar_structured_layout(
        dm,
        groups,
        dominant_group_index=dominant_group_index,
    )


def _structured_row_matrix(
    layout: ScalarStructuredLayout | BlockStructuredLayout | NestedStructuredLayout,
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
    layout: ScalarStructuredLayout | BlockStructuredLayout | NestedStructuredLayout,
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
    layout: ScalarStructuredLayout | BlockStructuredLayout | NestedStructuredLayout,
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
