"""Compact sufficient-statistic assembly for structured systems."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from superglm._group_matrix._group_matrix_algebra import (
    _cross_gram,
    _random_effect_cross_gram,
)
from superglm._group_matrix._group_matrix_kernels import (
    _dense_small_weighted_moments,
    _random_effect_sufficient_stats,
)
from superglm.factor_smooth_geometry import adjoint_sum_to_zero_blocks
from superglm.group_matrix import (
    DenseGroupMatrix,
    FactorSmoothGroupMatrix,
    GroupMatrix,
)
from superglm.solvers._structured.layout import (
    BlockStructuredLayout,
    ScalarStructuredLayout,
    _validate_structured_inputs,
    build_block_structured_layout,
    build_scalar_structured_layout,
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

    if len(layout.small_indices):
        if layout.dense_small_matrix is not None:
            A, xtw_small, xtwz_small = _dense_small_weighted_moments(
                layout.dense_small_matrix,
                weights,
                weighted_rhs,
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
        C = np.concatenate(
            [
                _random_effect_cross_gram(dominant, matrix, weights)
                for matrix in layout.small_matrices
            ],
            axis=1,
        )
    else:
        A = np.empty((0, 0), dtype=np.float64)
        C = np.empty((dominant.n_levels, 0), dtype=np.float64)
        xtw_small = np.empty(0, dtype=np.float64)
        xtwz_small = np.empty(0, dtype=np.float64)

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
    layout: ScalarStructuredLayout | BlockStructuredLayout | None = None,
) -> ScalarStructuredSystem | BlockStructuredSystem | SumToZeroBlockStructuredSystem:
    """Dispatch sufficient-statistic construction by dominant matrix type."""
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
