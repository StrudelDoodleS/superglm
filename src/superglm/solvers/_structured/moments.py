"""Compact sufficient-statistic assembly for structured systems."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from superglm._group_matrix._group_matrix_algebra import (
    _BlockWeightCache,
    _cross_gram,
    _random_effect_cross_gram,
)
from superglm._group_matrix._group_matrix_centered import _compensated_add
from superglm._group_matrix._group_matrix_kernels import (
    _dense_small_weighted_moments,
    _random_effect_sufficient_stats,
)
from superglm.factor_smooth_geometry import adjoint_sum_to_zero_blocks
from superglm.group_matrix import (
    DenseGroupMatrix,
    FactorSmoothGroupMatrix,
    GroupMatrix,
    RandomEffectGroupMatrix,
)
from superglm.solvers._structured.layout import (
    BlockStructuredLayout,
    ScalarStructuredLayout,
    _validate_structured_inputs,
    build_block_structured_layout,
    build_scalar_structured_layout,
)
from superglm.solvers._structured.nested import (
    NestedDataOperator,
    NestedLeafStatistics,
    NestedStructuredLayout,
)
from superglm.solvers._structured.operators import (
    BlockSymmetricOperator,
    SumToZeroBlockOperator,
    SymmetricBlockOperator,
)
from superglm.types import GroupSlice


@dataclass(frozen=True)
class ScalarStructuredSystem:
    """Unpenalized coefficient blocks and working sufficient statistics."""

    operator: SymmetricBlockOperator
    xtw_small: NDArray
    xtw_structured: NDArray
    xtwz_small: NDArray
    xtwz_structured: NDArray
    sum_w: float
    sum_wz: float
    dominant_group_index: int
    dominant_group_name: str


@dataclass(frozen=True)
class BlockStructuredSystem:
    """Unpenalized block-Schur geometry and working sufficient statistics."""

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


@dataclass(frozen=True)
class SumToZeroBlockStructuredSystem:
    """Raw all-level SZ moments with public ``K - 1`` transpose products."""

    operator: SumToZeroBlockOperator
    xtw_small: NDArray
    xtw_structured: NDArray
    xtwz_small: NDArray
    xtwz_structured: NDArray
    raw_xtw_structured: NDArray
    raw_xtwz_structured: NDArray
    sum_w: float
    sum_wz: float
    dominant_group_index: int
    dominant_group_name: str
    level_labels: tuple[object, ...]


def _small_block_moments(
    layout: ScalarStructuredLayout | NestedStructuredLayout,
    weights: NDArray,
    weighted_rhs: NDArray,
    weight_cache: _BlockWeightCache,
) -> tuple[NDArray, NDArray, NDArray]:
    """Return the border Gram ``X_b'WX_b``, ``X_b'W`` and ``X_b'Wz``."""
    if not len(layout.small_indices):
        return np.empty((0, 0), dtype=np.float64), np.empty(0), np.empty(0)
    if layout.dense_small_matrix is not None:
        return _dense_small_weighted_moments(layout.dense_small_matrix, weights, weighted_rhs)
    if layout.small_execution_plan is None:  # pragma: no cover - layout invariant
        raise RuntimeError("Structured small block has no execution plan.")
    small_moments = layout.small_execution_plan._moments_prevalidated(
        weights,
        rhs=(weighted_rhs,),
        include_xtw=True,
        signed=bool(np.any(weights < 0.0)),
        _cache=weight_cache,
    )
    if small_moments.xtw is None:  # pragma: no cover - requested above
        raise RuntimeError("Structured small moment plan omitted X'W.")
    return small_moments.gram, small_moments.xtw, small_moments.xt_rhs[0]


def _random_effect_cross(
    layout: ScalarStructuredLayout | NestedStructuredLayout,
    random_effect: RandomEffectGroupMatrix,
    weights: NDArray,
    weight_cache: _BlockWeightCache,
) -> NDArray:
    """Return the level-by-border cross ``Z_re' W X_b`` from the level kernels."""
    if not layout.small_matrices:
        return np.empty((random_effect.n_levels, 0), dtype=np.float64)
    return np.concatenate(
        [
            _random_effect_cross_gram(random_effect, matrix, weights, weight_cache)
            for matrix in layout.small_matrices
        ],
        axis=1,
    )


def _border_rows(layout: NestedStructuredLayout, rows: NDArray) -> NDArray:
    """Materialize the border rows ``X_b[rows]`` of one chunk (q columns)."""
    if layout.dense_small_matrix is not None:
        return layout.dense_small_matrix[rows]
    blocks = [
        np.asarray(matrix.row_subset(rows).toarray(), dtype=np.float64)
        for matrix in layout.small_matrices
    ]
    return np.hstack(blocks) if blocks else np.empty((len(rows), 0), dtype=np.float64)


def _centered_leaf_pass(
    layout: NestedStructuredLayout,
    codes: NDArray,
    weights: NDArray,
    leaf_weight: NDArray,
    mean: NDArray | None,
    chunk_size: int,
) -> tuple[NDArray, NDArray, NDArray | None]:
    """Return leaf means, the centred within-leaf scatter and the deviations.

    The exact centred row pass of §3.4 and §3.6 (decision 1), in bounded row
    chunks.  Without ``mean`` it first forms the data leaf means in the
    shifted form ``x_ref + sum_r w_r (x_r - x_ref) / w_l`` about each leaf's
    reference row (0 where ``w_l == 0``), so a column constant within a leaf
    gives that constant exactly; the deviations are then zero by
    construction and returned as ``None``.  With ``mean`` (a signed operator
    about its factor's data means) it also forms ``dev_l = sum_r a_r (x_r -
    m_l)``.  The scatter ``sum_r a_r (x_r - m_l)(x_r - m_l)'`` never subtracts
    raw moments, so such a column has an exactly zero row and column; chunks
    accumulate with compensated addition and the result is symmetrized.
    """
    n, q = len(weights), len(layout.small_indices)
    starts = range(0, n, chunk_size)
    deviation = None
    if mean is None:
        observed = layout.reference_row >= 0
        reference = np.zeros((len(leaf_weight), q), dtype=np.float64)
        reference[observed] = _border_rows(layout, layout.reference_row[observed])
        shift = np.zeros_like(reference)
        for start in starts:
            rows = np.arange(start, min(start + chunk_size, n))
            leaves = codes[rows]
            difference = _border_rows(layout, rows) - reference[leaves]
            np.add.at(shift, leaves, weights[rows, None] * difference)
        active = leaf_weight != 0.0
        mean = np.zeros_like(reference)
        mean[active] = reference[active] + shift[active] / leaf_weight[active, None]
    else:
        mean = np.asarray(mean, dtype=np.float64)
        deviation = np.zeros_like(mean)
    within = np.zeros((q, q), dtype=np.float64)
    compensation = np.zeros_like(within)
    for start in starts:
        rows = np.arange(start, min(start + chunk_size, n))
        leaves = codes[rows]
        centered = _border_rows(layout, rows) - mean[leaves]
        weighted = weights[rows, None] * centered
        _compensated_add(within, compensation, centered.T @ weighted)
        if deviation is not None:
            np.add.at(deviation, leaves, weighted)
    return mean, 0.5 * (within + within.T), deviation


def build_nested_leaf_statistics(
    layout: NestedStructuredLayout,
    group_matrices: list[GroupMatrix],
    weights: NDArray,
    *,
    mean: NDArray | None = None,
    chunk_size: int = 8192,
) -> NestedLeafStatistics:
    """One row pass of a nested chain: per-leaf statistics of the row weights.

    ``weight`` and ``cross`` come from the existing leaf kernels; ``mean``,
    ``within`` and ``deviation`` from ``_centered_leaf_pass``.  ``mean=None``
    is the data pass (shifted data means, deviation ``None``); a given
    ``mean`` is a signed pass about those means.
    """
    leaf = group_matrices[layout.leaf_group_index]
    if not isinstance(leaf, RandomEffectGroupMatrix):
        raise ValueError("The nested leaf group must be a RandomEffectGroupMatrix.")
    values = np.asarray(weights, dtype=np.float64)
    leaf_weight = leaf.rmatvec(values)
    leaf_mean, within, deviation = _centered_leaf_pass(
        layout, leaf.codes, values, leaf_weight, mean, chunk_size
    )
    return NestedLeafStatistics(
        weight=leaf_weight,
        cross=_random_effect_cross(layout, leaf, values, _BlockWeightCache()),
        mean=leaf_mean,
        within=within,
        deviation=deviation,
    )


def build_nested_structured_system(
    group_matrices: list[GroupMatrix],
    groups: list[GroupSlice],
    W: NDArray,
    Wz: NDArray,
    *,
    layout: NestedStructuredLayout,
    mean: NDArray | None = None,
) -> NestedStructuredSystem:
    """Build the nested data operator and sufficient statistics from one set of rows.

    ``mean=None`` is the data system of a PIRLS iterate; a signed W-derivative
    operator passes its factor's ``data_operator.leaf.mean`` (§6).  Only the
    leaf group touches rows: its level sums come from the leaf kernels and
    every parent level is their subtree sum.
    """
    weights, weighted_rhs, leaf = _validate_structured_inputs(
        group_matrices,
        groups,
        W,
        Wz,
        layout.leaf_group_index,
    )
    if any(
        matrix is not group_matrices[index]
        for matrix, index in zip(layout.small_matrices, layout.small_group_indices, strict=True)
    ) or layout.chain_group_names != tuple(groups[i].name for i in layout.chain_group_indices):
        raise ValueError("Nested layout does not match the supplied grouped design.")
    A, xtw_small, xtwz_small = _small_block_moments(
        layout, weights, weighted_rhs, _BlockWeightCache()
    )
    leaf_statistics = build_nested_leaf_statistics(layout, group_matrices, weights, mean=mean)
    operator = NestedDataOperator(
        tree=layout.tree,
        leaf=leaf_statistics,
        A=0.5 * (A + A.T),
        small_indices=layout.small_indices,
        structured_indices=layout.structured_indices,
    )
    return NestedStructuredSystem(
        operator=operator,
        xtw_small=xtw_small,
        xtw_structured=np.concatenate(layout.tree.subtree_sum(leaf_statistics.weight)),
        xtwz_small=xtwz_small,
        xtwz_structured=np.concatenate(layout.tree.subtree_sum(leaf.rmatvec(weighted_rhs))),
        sum_w=float(np.sum(weights)),
        sum_wz=float(np.sum(weighted_rhs)),
        chain_group_indices=layout.chain_group_indices,
        chain_group_names=layout.chain_group_names,
        dominant_group_name=layout.leaf_group_name,
    )


def build_scalar_structured_system(
    group_matrices: list[GroupMatrix],
    groups: list[GroupSlice],
    W: NDArray,
    Wz: NDArray,
    *,
    dominant_group_index: int,
    tabmat_split=None,
    layout: ScalarStructuredLayout | None = None,
) -> ScalarStructuredSystem:
    """Build exact scalar-Schur blocks without a full coefficient Gram matrix."""
    del tabmat_split
    weights, weighted_rhs, dominant = _validate_structured_inputs(
        group_matrices,
        groups,
        W,
        Wz,
        dominant_group_index,
    )
    if layout is None:
        layout = build_scalar_structured_layout(
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
        raise ValueError("Structured layout does not match the supplied grouped design.")

    weight_cache = _BlockWeightCache()
    A, xtw_small, xtwz_small = _small_block_moments(layout, weights, weighted_rhs, weight_cache)
    C = _random_effect_cross(layout, dominant, weights, weight_cache)
    level_W, level_Wz = _random_effect_sufficient_stats(
        dominant.codes,
        weights,
        weighted_rhs,
        dominant.n_levels,
    )
    operator = SymmetricBlockOperator(
        A=A,
        C=C,
        d=level_W,
        small_indices=layout.small_indices,
        structured_indices=layout.structured_indices,
    )
    return ScalarStructuredSystem(
        operator=operator,
        xtw_small=xtw_small,
        xtw_structured=level_W,
        xtwz_small=xtwz_small,
        xtwz_structured=level_Wz,
        sum_w=float(np.sum(weights)),
        sum_wz=float(np.sum(weighted_rhs)),
        dominant_group_index=dominant_group_index,
        dominant_group_name=layout.dominant_group_name,
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
    tabmat_split=None,
    layout: BlockStructuredLayout | None = None,
) -> BlockStructuredSystem | SumToZeroBlockStructuredSystem:
    """Build exact block-Schur moments without a full coefficient Gram matrix."""
    del tabmat_split
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
        layout = build_block_structured_layout(
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
                signed=bool(np.any(weights < 0.0)),
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
        return SumToZeroBlockStructuredSystem(
            operator=operator,
            xtw_small=xtw_small,
            xtw_structured=xtw_structured,
            xtwz_small=xtwz_small,
            xtwz_structured=xtwz_structured,
            raw_xtw_structured=raw_xtw_structured,
            raw_xtwz_structured=raw_xtwz_structured,
            sum_w=float(np.sum(weights)),
            sum_wz=float(np.sum(weighted_rhs)),
            dominant_group_index=dominant_group_index,
            dominant_group_name=layout.dominant_group_name,
            level_labels=dominant.levels,
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
    return BlockStructuredSystem(
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
    tabmat_split=None,
    layout: ScalarStructuredLayout | BlockStructuredLayout | NestedStructuredLayout | None = None,
) -> (
    ScalarStructuredSystem
    | BlockStructuredSystem
    | SumToZeroBlockStructuredSystem
    | NestedStructuredSystem
):
    """Dispatch sufficient-statistic construction by layout or dominant matrix type."""
    if isinstance(layout, NestedStructuredLayout):
        return build_nested_structured_system(group_matrices, groups, W, Wz, layout=layout)
    dominant = group_matrices[dominant_group_index]
    if isinstance(dominant, FactorSmoothGroupMatrix):
        if layout is not None and not isinstance(layout, BlockStructuredLayout):
            raise TypeError("FactorSmooth structured builds require a block layout.")
        return build_block_structured_system(
            group_matrices,
            groups,
            W,
            Wz,
            dominant_group_index=dominant_group_index,
            tabmat_split=tabmat_split,
            layout=layout,
        )
    if layout is not None and not isinstance(layout, ScalarStructuredLayout):
        raise TypeError("RandomEffect structured builds require a scalar layout.")
    return build_scalar_structured_system(
        group_matrices,
        groups,
        W,
        Wz,
        dominant_group_index=dominant_group_index,
        tabmat_split=tabmat_split,
        layout=layout,
    )
