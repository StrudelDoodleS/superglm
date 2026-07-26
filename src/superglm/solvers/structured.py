"""Structured linear algebra for dominant random-effect blocks."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from superglm._group_matrix._group_matrix_algebra import (
    _cross_gram,
    _random_effect_cross_gram,
)
from superglm._group_matrix._group_matrix_execution import MatrixExecutionPlan
from superglm._group_matrix._group_matrix_kernels import (
    _dense_small_weighted_moments,
    _random_effect_sufficient_stats,
)
from superglm.factor_smooth_geometry import adjoint_sum_to_zero_blocks
from superglm.group_matrix import (
    DenseGroupMatrix,
    DesignMatrix,
    FactorSmoothGroupMatrix,
    GroupMatrix,
    RandomEffectGroupMatrix,
)
from superglm.solvers._structured.factors import (
    BlockSchurFactor,
    ProfiledBlockSchurFactor,
    ProfiledScalarSchurFactor,
    ScalarSchurFactor,
)
from superglm.solvers._structured.geometry import (
    _MAX_DENSE_CENTERED_ESTIMABILITY_WIDTH as _MAX_DENSE_CENTERED_ESTIMABILITY_WIDTH,
)
from superglm.solvers._structured.geometry import (
    _augmented_small_data_block as _augmented_small_data_block,
)
from superglm.solvers._structured.geometry import (
    _bounded_centered_estimability as _bounded_centered_estimability,
)
from superglm.solvers._structured.geometry import (
    _centered_operator_column_scale as _centered_operator_column_scale,
)
from superglm.solvers._structured.geometry import (
    _certified_local_null_basis as _certified_local_null_basis,
)
from superglm.solvers._structured.geometry import (
    _certified_reduced_schur_null_basis as _certified_reduced_schur_null_basis,
)
from superglm.solvers._structured.geometry import (
    _certified_ritz_discarded as _certified_ritz_discarded,
)
from superglm.solvers._structured.geometry import (
    _certified_sum_to_zero_centered_estimability as _certified_sum_to_zero_centered_estimability,
)
from superglm.solvers._structured.geometry import (
    _coefficient_estimable_from_null_basis as _coefficient_estimable_from_null_basis,
)
from superglm.solvers._structured.geometry import (
    _coefficient_estimable_from_scaled_null_basis as _coefficient_estimable_from_scaled_null_basis,
)
from superglm.solvers._structured.geometry import (
    _independent_block_centered_estimability as _independent_block_centered_estimability,
)
from superglm.solvers._structured.geometry import (
    _lifted_null_row_norms as _lifted_null_row_norms,
)
from superglm.solvers._structured.geometry import (
    _local_range_inverse_and_null_projector as _local_range_inverse_and_null_projector,
)
from superglm.solvers._structured.geometry import (
    _null_basis_with_inherited_gram_scale as _null_basis_with_inherited_gram_scale,
)
from superglm.solvers._structured.geometry import (
    _orthonormal_column_span as _orthonormal_column_span,
)
from superglm.solvers._structured.geometry import (
    _orthonormal_scaled_parameter_null_span as _orthonormal_scaled_parameter_null_span,
)
from superglm.solvers._structured.geometry import (
    _ritz_rank_masks as _ritz_rank_masks,
)
from superglm.solvers._structured.geometry import (
    _ritz_residual_norms as _ritz_residual_norms,
)
from superglm.solvers._structured.geometry import (
    _sum_to_zero_centered_estimability as _sum_to_zero_centered_estimability,
)
from superglm.solvers._structured.geometry import (
    _sum_to_zero_inherent_null_row_norms as _sum_to_zero_inherent_null_row_norms,
)
from superglm.solvers._structured.geometry import (
    _sum_to_zero_normalized_structured_matvec as _sum_to_zero_normalized_structured_matvec,
)
from superglm.solvers._structured.geometry import (
    _sum_to_zero_public_null_geometry as _sum_to_zero_public_null_geometry,
)
from superglm.solvers._structured.geometry import (
    _sum_to_zero_public_spectral_bound as _sum_to_zero_public_spectral_bound,
)
from superglm.solvers._structured.geometry import (
    _sum_to_zero_public_spectral_estimability as _sum_to_zero_public_spectral_estimability,
)
from superglm.solvers._structured.geometry import (
    _sum_to_zero_public_weak_bases as _sum_to_zero_public_weak_bases,
)
from superglm.solvers._structured.geometry import (
    _sum_to_zero_retained_constraint_row_space as _sum_to_zero_retained_constraint_row_space,
)
from superglm.solvers._structured.geometry import (
    _sum_to_zero_scaled_basis_null_row_norms as _sum_to_zero_scaled_basis_null_row_norms,
)
from superglm.solvers._structured.geometry import (
    _sum_to_zero_scaled_null_constraint_geometry as _sum_to_zero_scaled_null_constraint_geometry,
)
from superglm.solvers._structured.geometry import (
    _SumToZeroPublicNullGeometry as _SumToZeroPublicNullGeometry,
)
from superglm.solvers._structured.geometry import (
    centered_operator_coefficient_estimable as centered_operator_coefficient_estimable,
)
from superglm.solvers._structured.operators import (
    BlockSymmetricOperator,
    CenteredBlockOperator,
    SumToZeroBlockOperator,
    SymmetricBlockOperator,
)
from superglm.solvers._structured.operators import (
    CompactSymmetricOperator as CompactSymmetricOperator,
)
from superglm.solvers._structured.operators import (
    LowRankSymmetricOperator as LowRankSymmetricOperator,
)
from superglm.solvers._structured.operators import (
    SumBlockOperator as SumBlockOperator,
)
from superglm.solvers._structured.operators import (
    _apply_local_blocks as _apply_local_blocks,
)
from superglm.solvers._structured.operators import (
    _block_operator_bdlr as _block_operator_bdlr,
)
from superglm.solvers._structured.operators import (
    _block_operator_dlr as _block_operator_dlr,
)
from superglm.solvers._structured.operators import (
    _BlockDiagonalLowRank as _BlockDiagonalLowRank,
)
from superglm.solvers._structured.operators import (
    _DiagonalLowRank as _DiagonalLowRank,
)
from superglm.solvers._structured.operators import (
    _empty_block_part as _empty_block_part,
)
from superglm.solvers._structured.operators import (
    _general_bdlr_diagonal as _general_bdlr_diagonal,
)
from superglm.solvers._structured.operators import (
    _general_bdlr_square_diagonal as _general_bdlr_square_diagonal,
)
from superglm.solvers._structured.operators import (
    _general_dlr_diagonal as _general_dlr_diagonal,
)
from superglm.solvers._structured.operators import (
    _general_dlr_square_diagonal as _general_dlr_square_diagonal,
)
from superglm.solvers._structured.operators import (
    _GeneralBlockDiagonalLowRank as _GeneralBlockDiagonalLowRank,
)
from superglm.solvers._structured.operators import (
    _GeneralDiagonalLowRank as _GeneralDiagonalLowRank,
)
from superglm.solvers._structured.operators import (
    _merge_bdlr as _merge_bdlr,
)
from superglm.solvers._structured.operators import (
    _merge_dlr as _merge_dlr,
)
from superglm.solvers._structured.operators import (
    _multiply_symmetric_bdlr as _multiply_symmetric_bdlr,
)
from superglm.solvers._structured.operators import (
    _multiply_symmetric_bdlr_coalesced as _multiply_symmetric_bdlr_coalesced,
)
from superglm.solvers._structured.operators import (
    _multiply_symmetric_dlr as _multiply_symmetric_dlr,
)
from superglm.solvers._structured.operators import (
    _operator_bdlr as _operator_bdlr,
)
from superglm.solvers._structured.operators import (
    _operator_dlr as _operator_dlr,
)
from superglm.solvers._structured.operators import (
    _sum_to_zero_operator_bdlr as _sum_to_zero_operator_bdlr,
)
from superglm.solvers._structured.operators import (
    _trace_general_bdlr_product as _trace_general_bdlr_product,
)
from superglm.solvers._structured.operators import (
    _trace_general_product as _trace_general_product,
)
from superglm.solvers._structured.operators import (
    _trace_symmetric_bdlr as _trace_symmetric_bdlr,
)
from superglm.solvers._structured.operators import (
    _trace_symmetric_dlr as _trace_symmetric_dlr,
)
from superglm.solvers._structured.operators import (
    compact_operator_diagonal as compact_operator_diagonal,
)
from superglm.solvers._structured.operators import (
    materialize_compact_operator as materialize_compact_operator,
)
from superglm.solvers._structured.selection import (
    _AUTO_MAX_STRUCTURED_COST_RATIO as _AUTO_MAX_STRUCTURED_COST_RATIO,
)
from superglm.solvers._structured.selection import (
    _AUTO_MIN_COEFFICIENT_WIDTH as _AUTO_MIN_COEFFICIENT_WIDTH,
)
from superglm.solvers._structured.selection import (
    StructuredBackendDecision as StructuredBackendDecision,
)
from superglm.solvers._structured.selection import (
    StructuredGroupSelection as StructuredGroupSelection,
)
from superglm.solvers._structured.selection import (
    _block_structured_auto_is_beneficial as _block_structured_auto_is_beneficial,
)
from superglm.solvers._structured.selection import (
    _factor_smooth_singular_local_level as _factor_smooth_singular_local_level,
)
from superglm.solvers._structured.selection import (
    _selection_failure as _selection_failure,
)
from superglm.solvers._structured.selection import (
    _structured_auto_is_beneficial as _structured_auto_is_beneficial,
)
from superglm.solvers._structured.selection import (
    _sum_to_zero_structured_auto_is_beneficial as _sum_to_zero_structured_auto_is_beneficial,
)
from superglm.solvers._structured.selection import (
    resolve_structured_backend as resolve_structured_backend,
)
from superglm.solvers._structured.selection import (
    select_structured_group as select_structured_group,
)
from superglm.solvers.hessian_factor import _component_indices
from superglm.types import GroupSlice, PenaltyComponent

if TYPE_CHECKING:
    from superglm.solvers.sum_to_zero import (
        ProfiledSumToZeroBlockFactor,
        SumToZeroBlockFactor,
    )


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


@dataclass(frozen=True)
class CachedScalarStructuredSolution:
    """One lambda-only solve against cached structured working moments."""

    beta: NDArray
    intercept: float
    factor: ProfiledScalarSchurFactor
    penalized_operator: SymmetricBlockOperator
    log_det_H: float  # noqa: N815
    hessian_rank: int


@dataclass(frozen=True)
class CachedBlockStructuredSolution:
    """One lambda-only solve against cached factor-smooth working moments."""

    beta: NDArray
    intercept: float
    factor: ProfiledBlockSchurFactor
    penalized_operator: BlockSymmetricOperator
    log_det_H: float  # noqa: N815
    hessian_rank: int


@dataclass(frozen=True)
class CachedSumToZeroStructuredSolution:
    """One lambda-only solve against cached constrained SZ moments."""

    beta: NDArray
    intercept: float
    factor: ProfiledSumToZeroBlockFactor
    penalized_operator: SumToZeroBlockOperator
    log_det_H: float  # noqa: N815
    hessian_rank: int


@dataclass(frozen=True)
class StructuredLevelSupport:
    """Compact training support retained for one all-level structured term."""

    count: NDArray
    fit_weight: NDArray
    information: NDArray
    unpooled_effect: NDArray | None = None

    def __post_init__(self) -> None:
        expected_shape: tuple[int, ...] | None = None
        for name, dtype in (
            ("count", np.int64),
            ("fit_weight", np.float64),
            ("information", np.float64),
        ):
            values = np.array(getattr(self, name), dtype=dtype, copy=True)
            if values.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional.")
            if expected_shape is None:
                expected_shape = values.shape
            elif values.shape != expected_shape:
                raise ValueError("Structured support arrays must have identical shapes.")
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        if self.unpooled_effect is not None:
            unpooled = np.array(self.unpooled_effect, dtype=np.float64, copy=True)
            if unpooled.shape != expected_shape:
                raise ValueError("unpooled_effect must match the structured support shape.")
            unpooled.setflags(write=False)
            object.__setattr__(self, "unpooled_effect", unpooled)


@dataclass(frozen=True)
class FactorSmoothLevelSupport:
    """Compact row support and local Fisher information for a factor smooth."""

    count: NDArray
    fit_weight: NDArray
    information: NDArray

    def __post_init__(self) -> None:
        count = np.array(self.count, dtype=np.int64, copy=True)
        fit_weight = np.array(self.fit_weight, dtype=np.float64, copy=True)
        information = np.array(self.information, dtype=np.float64, copy=True)
        if count.ndim != 1 or fit_weight.shape != count.shape:
            raise ValueError("FactorSmooth count and fit_weight must be aligned vectors.")
        if (
            information.ndim != 3
            or information.shape[0] != len(count)
            or information.shape[1] != information.shape[2]
        ):
            raise ValueError(
                "FactorSmooth information must have shape (n_levels, block_size, block_size)."
            )
        if not np.allclose(
            information,
            information.transpose(0, 2, 1),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("FactorSmooth local information blocks must be symmetric.")
        for values in (count, fit_weight, information):
            values.setflags(write=False)
        object.__setattr__(self, "count", count)
        object.__setattr__(self, "fit_weight", fit_weight)
        object.__setattr__(self, "information", information)


@dataclass(frozen=True)
class StructuredLinearSystemState:
    """Authoritative compact factors and moments retained after a fit."""

    coefficient_factor: ScalarSchurFactor | BlockSchurFactor | SumToZeroBlockFactor
    profiled_factor: (
        ProfiledScalarSchurFactor | ProfiledBlockSchurFactor | ProfiledSumToZeroBlockFactor
    )
    augmented_factor: ScalarSchurFactor | BlockSchurFactor | SumToZeroBlockFactor
    system: ScalarStructuredSystem | BlockStructuredSystem | SumToZeroBlockStructuredSystem
    penalized_operator: SymmetricBlockOperator | BlockSymmetricOperator | SumToZeroBlockOperator
    centered_data_operator: CenteredBlockOperator
    support_totals: dict[
        str,
        StructuredLevelSupport | FactorSmoothLevelSupport,
    ]
    backend: str = "structured"
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        if self.coefficient_factor.shape != self.system.operator.shape:
            raise ValueError("Coefficient factor does not match the structured system.")
        if self.profiled_factor.shape != self.system.operator.shape:
            raise ValueError("Profiled factor does not match the structured system.")
        expected_augmented = self.system.operator.shape[0] + 1
        if self.augmented_factor.shape != (expected_augmented, expected_augmented):
            raise ValueError("Augmented factor does not match the structured system.")
        if self.penalized_operator.shape != self.system.operator.shape:
            raise ValueError("Penalized operator does not match the structured system.")
        if self.centered_data_operator.shape != self.system.operator.shape:
            raise ValueError("Centered data operator does not match the structured system.")
        object.__setattr__(self, "support_totals", dict(self.support_totals))


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
    small_group_indices = tuple(
        index for index in range(len(group_matrices)) if index != dominant_group_index
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
            n=dominant.shape[0],
            ordinary_tabmat=True,
        )
        small_execution_plan.validate_group_spans(local_groups)
    return ScalarStructuredLayout(
        dominant_group_index=dominant_group_index,
        dominant_group_name=dominant_group.name,
        small_group_indices=small_group_indices,
        small_matrices=small_matrices,
        local_groups=tuple(local_groups),
        small_indices=small_indices,
        structured_indices=structured_indices,
        dense_small_matrix=dense_small_matrix,
        small_execution_plan=small_execution_plan,
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
    small_group_indices = tuple(
        index for index in range(len(group_matrices)) if index != dominant_group_index
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
            n=dominant.shape[0],
            ordinary_tabmat=True,
        )
        small_execution_plan.validate_group_spans(local_groups)
    return BlockStructuredLayout(
        dominant_group_index=dominant_group_index,
        dominant_group_name=dominant_group.name,
        small_group_indices=small_group_indices,
        small_matrices=small_matrices,
        local_groups=tuple(local_groups),
        small_indices=small_indices,
        structured_indices=structured_indices,
        dense_small_matrix=dense_small_matrix,
        small_execution_plan=small_execution_plan,
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


def get_structured_layout(
    dm: DesignMatrix,
    groups: list[GroupSlice],
    *,
    dominant_group_index: int,
) -> ScalarStructuredLayout | BlockStructuredLayout:
    """Dispatch layout construction by dominant structured matrix type."""
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


def structured_design_matvec(
    layout: ScalarStructuredLayout | BlockStructuredLayout,
    group_matrices: list[GroupMatrix] | tuple[GroupMatrix, ...],
    beta: NDArray,
) -> NDArray:
    """Apply a grouped design while fusing a cached dense-small partition."""
    values = np.asarray(beta, dtype=np.float64)
    width = len(layout.small_indices) + layout.structured_indices.size
    if values.shape != (width,):
        raise ValueError(f"beta must have shape ({width},).")
    dominant = group_matrices[layout.dominant_group_index]
    if not isinstance(dominant, RandomEffectGroupMatrix | FactorSmoothGroupMatrix):
        raise ValueError("Structured layout no longer points to a structured group.")

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
    result += dominant.matvec(values[layout.structured_indices].ravel())
    return result


def structured_design_rmatvec(
    layout: ScalarStructuredLayout | BlockStructuredLayout,
    group_matrices: list[GroupMatrix] | tuple[GroupMatrix, ...],
    rows: NDArray,
) -> NDArray:
    """Apply a grouped design transpose with one cached dense-small product."""
    values = np.asarray(rows, dtype=np.float64)
    dominant = group_matrices[layout.dominant_group_index]
    if not isinstance(dominant, RandomEffectGroupMatrix | FactorSmoothGroupMatrix):
        raise ValueError("Structured layout no longer points to a structured group.")
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
    result[layout.structured_indices] = dominant.rmatvec(values).reshape(
        layout.structured_indices.shape
    )
    return result


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


def _lambda_for_component(
    lambda2: float | dict[str, float],
    name: str,
) -> float:
    return float(lambda2[name]) if isinstance(lambda2, dict) else float(lambda2)


def _dense_component_omega(
    component: PenaltyComponent,
    group_matrix: GroupMatrix,
) -> NDArray:
    if component.omega_ssp is not None:
        return np.asarray(component.omega_ssp, dtype=np.float64)
    if component.omega_raw is None or not hasattr(group_matrix, "R_inv"):
        raise ValueError(f"Dense penalty component {component.name!r} has no solver-space matrix.")
    return np.asarray(
        group_matrix.R_inv.T @ component.omega_raw @ group_matrix.R_inv,
        dtype=np.float64,
    )


def build_penalized_scalar_operator(
    system: ScalarStructuredSystem,
    group_matrices: list[GroupMatrix],
    groups: list[GroupSlice],
    lambda2: float | dict[str, float],
    *,
    reml_penalties: list[PenaltyComponent] | None = None,
    S_override: NDArray | None = None,
) -> SymmetricBlockOperator:
    """Add block penalties to a structured Gram without forming a full penalty matrix."""
    operator = system.operator
    p = operator.shape[0]
    A = np.array(operator.A, copy=True)
    d = np.array(operator.d, copy=True)
    small_position = np.full(p, -1, dtype=np.intp)
    small_position[operator.small_indices] = np.arange(len(operator.small_indices))
    structured_position = np.full(p, -1, dtype=np.intp)
    structured_position[operator.structured_indices] = np.arange(len(operator.structured_indices))

    if S_override is not None:
        penalty = np.asarray(S_override, dtype=np.float64)
        if penalty.shape != (p, p):
            raise ValueError(f"S_override must have shape ({p}, {p}).")
        cross = penalty[np.ix_(operator.structured_indices, operator.small_indices)]
        if np.any(np.abs(cross) > 1e-12):
            raise ValueError("S_override couples the dominant and dense-small blocks.")
        A += penalty[np.ix_(operator.small_indices, operator.small_indices)]
        d += np.diag(penalty)[operator.structured_indices]
        return SymmetricBlockOperator(
            A=A,
            C=operator.C,
            d=d,
            small_indices=operator.small_indices,
            structured_indices=operator.structured_indices,
        )

    if reml_penalties is not None:
        for component in reml_penalties:
            lam = _lambda_for_component(lambda2, component.name)
            if lam == 0.0:
                continue
            indices = _component_indices(component, p)
            local_small = small_position[indices]
            local_structured = structured_position[indices]
            wholly_small = np.all(local_small >= 0)
            wholly_structured = np.all(local_structured >= 0)
            if not wholly_small and not wholly_structured:
                raise ValueError(
                    f"Penalty component {component.name!r} crosses structured partitions."
                )
            if component.penalty_kind == "identity":
                if wholly_small:
                    A[local_small, local_small] += lam
                else:
                    d[local_structured] += lam
                continue

            omega = _dense_component_omega(
                component,
                group_matrices[component.group_index],
            )
            if omega.shape != (len(indices), len(indices)):
                raise ValueError(
                    f"Penalty component {component.name!r} has shape {omega.shape}; "
                    f"expected ({len(indices)}, {len(indices)})."
                )
            if wholly_small:
                A[np.ix_(local_small, local_small)] += lam * omega
                continue
            off_diagonal = omega - np.diag(np.diag(omega))
            if np.any(np.abs(off_diagonal) > 1e-12):
                raise ValueError(f"Dominant penalty component {component.name!r} is not diagonal.")
            d[local_structured] += lam * np.diag(omega)
    else:
        for group_index, (matrix, group) in enumerate(zip(group_matrices, groups, strict=True)):
            if not group.penalized:
                continue
            lam = (
                float(lambda2.get(group.name, 0.0)) if isinstance(lambda2, dict) else float(lambda2)
            )
            if lam == 0.0:
                continue
            indices = np.arange(group.start, group.end, dtype=np.intp)
            local_small = small_position[indices]
            local_structured = structured_position[indices]
            if isinstance(matrix, RandomEffectGroupMatrix):
                if np.all(local_small >= 0):
                    A[local_small, local_small] += lam
                elif np.all(local_structured >= 0):
                    d[local_structured] += lam
                else:
                    raise ValueError(
                        f"RandomEffect group {group.name!r} crosses structured partitions."
                    )
                continue
            omega_raw = getattr(matrix, "omega", None)
            if omega_raw is None or not hasattr(matrix, "R_inv"):
                continue
            omega = np.asarray(
                matrix.R_inv.T @ omega_raw @ matrix.R_inv,
                dtype=np.float64,
            )
            if not np.all(local_small >= 0):
                raise ValueError(
                    f"Penalty geometry for dominant group index {group_index} is unsupported."
                )
            A[np.ix_(local_small, local_small)] += lam * omega

    return SymmetricBlockOperator(
        A=A,
        C=operator.C,
        d=d,
        small_indices=operator.small_indices,
        structured_indices=operator.structured_indices,
    )


def build_penalized_block_operator(
    system: BlockStructuredSystem,
    group_matrices: list[GroupMatrix],
    groups: list[GroupSlice],
    lambda2: float | dict[str, float],
    *,
    reml_penalties: list[PenaltyComponent] | None = None,
    S_override: NDArray | None = None,
) -> BlockSymmetricOperator:
    """Add compact penalties to a factor-smooth block Gram."""
    operator = system.operator
    p = operator.shape[0]
    A = np.array(operator.A, copy=True)
    D = np.array(operator.D, copy=True)
    small_position = np.full(p, -1, dtype=np.intp)
    small_position[operator.small_indices] = np.arange(len(operator.small_indices))
    structured_position = np.full(p, -1, dtype=np.intp)
    structured_position[operator.structured_indices.ravel()] = np.arange(
        operator.n_levels * operator.block_size
    )

    if S_override is not None:
        penalty = np.asarray(S_override, dtype=np.float64)
        if penalty.shape != (p, p):
            raise ValueError(f"S_override must have shape ({p}, {p}).")
        flat_structured = operator.structured_indices.ravel()
        cross = penalty[np.ix_(flat_structured, operator.small_indices)]
        if np.any(np.abs(cross) > 1e-12):
            raise ValueError("S_override couples the dominant and dense-small blocks.")
        A += penalty[np.ix_(operator.small_indices, operator.small_indices)]
        structured_penalty = penalty[np.ix_(flat_structured, flat_structured)]
        residual = np.array(structured_penalty, copy=True)
        for level in range(operator.n_levels):
            local = slice(
                level * operator.block_size,
                (level + 1) * operator.block_size,
            )
            D[level] += structured_penalty[local, local]
            residual[local, local] = 0.0
        if np.any(np.abs(residual) > 1e-12):
            raise ValueError("S_override couples distinct factor-smooth levels.")
        return BlockSymmetricOperator(
            A=A,
            C=operator.C,
            D=D,
            small_indices=operator.small_indices,
            structured_indices=operator.structured_indices,
        )

    if reml_penalties is not None:
        for component in reml_penalties:
            lam = _lambda_for_component(lambda2, component.name)
            if lam == 0.0:
                continue
            indices = _component_indices(component, p)
            local_small = small_position[indices]
            local_structured = structured_position[indices]
            wholly_small = np.all(local_small >= 0)
            wholly_structured = np.all(local_structured >= 0)
            if not wholly_small and not wholly_structured:
                raise ValueError(
                    f"Penalty component {component.name!r} crosses structured partitions."
                )
            if component.penalty_kind == "identity":
                if wholly_small:
                    A[local_small, local_small] += lam
                else:
                    levels = local_structured // operator.block_size
                    coordinates = local_structured % operator.block_size
                    D[levels, coordinates, coordinates] += lam
                continue
            if component.penalty_kind == "repeated":
                if not wholly_structured:
                    raise ValueError(
                        f"Repeated penalty component {component.name!r} must lie in "
                        "the dominant factor-smooth block."
                    )
                if (
                    component.repeat_count != operator.n_levels
                    or component.block_width != operator.block_size
                    or not np.array_equal(
                        indices.reshape(operator.n_levels, operator.block_size),
                        operator.structured_indices,
                    )
                ):
                    raise ValueError(
                        f"Repeated penalty component {component.name!r} does not match "
                        "the dominant factor-smooth geometry."
                    )
                omega = np.asarray(component.omega_ssp, dtype=np.float64)
                if omega.shape != (operator.block_size, operator.block_size):
                    raise ValueError(
                        f"Repeated penalty component {component.name!r} has shape "
                        f"{omega.shape}; expected "
                        f"({operator.block_size}, {operator.block_size})."
                    )
                D += lam * omega[None, :, :]
                continue

            omega = _dense_component_omega(
                component,
                group_matrices[component.group_index],
            )
            if omega.shape != (len(indices), len(indices)):
                raise ValueError(
                    f"Penalty component {component.name!r} has shape {omega.shape}; "
                    f"expected ({len(indices)}, {len(indices)})."
                )
            if not wholly_small:
                raise ValueError(
                    f"Dense penalty component {component.name!r} cannot span the "
                    "dominant factor-smooth block."
                )
            A[np.ix_(local_small, local_small)] += lam * omega
    else:
        for group_index, (matrix, group) in enumerate(zip(group_matrices, groups, strict=True)):
            if not group.penalized:
                continue
            indices = np.arange(group.start, group.end, dtype=np.intp)
            local_small = small_position[indices]
            local_structured = structured_position[indices]
            if isinstance(matrix, FactorSmoothGroupMatrix):
                if not np.all(local_structured >= 0):
                    raise ValueError(
                        f"FactorSmooth group {group.name!r} is not the dominant block."
                    )
                for suffix, omega in matrix.repeated_penalty_components:
                    if isinstance(lambda2, dict):
                        lam = float(
                            lambda2.get(
                                f"{group.name}:{suffix}",
                                lambda2.get(group.name, 0.0),
                            )
                        )
                    else:
                        lam = float(lambda2)
                    D += lam * np.asarray(omega, dtype=np.float64)[None, :, :]
                continue
            lam = (
                float(lambda2.get(group.name, 0.0)) if isinstance(lambda2, dict) else float(lambda2)
            )
            if lam == 0.0:
                continue
            if isinstance(matrix, RandomEffectGroupMatrix):
                if not np.all(local_small >= 0):
                    raise ValueError(
                        f"RandomEffect group {group.name!r} crosses structured partitions."
                    )
                A[local_small, local_small] += lam
                continue
            omega_raw = getattr(matrix, "omega", None)
            if omega_raw is None or not hasattr(matrix, "R_inv"):
                continue
            if not np.all(local_small >= 0):
                raise ValueError(
                    f"Penalty geometry for dominant group index {group_index} is unsupported."
                )
            omega = np.asarray(
                matrix.R_inv.T @ omega_raw @ matrix.R_inv,
                dtype=np.float64,
            )
            A[np.ix_(local_small, local_small)] += lam * omega

    return BlockSymmetricOperator(
        A=A,
        C=operator.C,
        D=D,
        small_indices=operator.small_indices,
        structured_indices=operator.structured_indices,
    )


def build_penalized_sum_to_zero_operator(
    system: SumToZeroBlockStructuredSystem,
    group_matrices: list[GroupMatrix],
    groups: list[GroupSlice],
    lambda2: float | dict[str, float],
    *,
    reml_penalties: list[PenaltyComponent] | None = None,
    S_override: NDArray | None = None,
) -> SumToZeroBlockOperator:
    """Add public penalties to their symmetric raw all-level SZ geometry."""
    operator = system.operator
    p = operator.shape[0]
    A = np.array(operator.A, copy=True)
    D = np.array(operator.D, copy=True)
    small_position = np.full(p, -1, dtype=np.intp)
    small_position[operator.small_indices] = np.arange(len(operator.small_indices))
    structured_position = np.full(p, -1, dtype=np.intp)
    structured_position[operator.structured_indices.ravel()] = np.arange(
        (operator.n_levels - 1) * operator.block_size
    )

    if S_override is not None:
        penalty = np.asarray(S_override, dtype=np.float64)
        if penalty.shape != (p, p):
            raise ValueError(f"S_override must have shape ({p}, {p}).")
        flat_structured = operator.structured_indices.ravel()
        cross = penalty[np.ix_(flat_structured, operator.small_indices)]
        if np.any(np.abs(cross) > 1e-12):
            raise ValueError("S_override couples the SZ and dense-small blocks.")
        A += penalty[np.ix_(operator.small_indices, operator.small_indices)]
        public = penalty[np.ix_(flat_structured, flat_structured)]
        free_levels = operator.n_levels - 1
        k = operator.block_size
        blocks = public.reshape(free_levels, k, free_levels, k)
        local = 0.5 * blocks[0, :, 0, :] if free_levels == 1 else blocks[0, :, 1, :]
        expected = np.empty_like(blocks)
        for left in range(free_levels):
            for right in range(free_levels):
                expected[left, :, right, :] = (2.0 if left == right else 1.0) * local
        if not np.allclose(blocks, expected, rtol=0.0, atol=1e-12):
            raise ValueError("S_override has noncanonical sum-to-zero penalty geometry.")
        D += local[None, :, :]
        return SumToZeroBlockOperator(
            A=A,
            C=operator.C,
            D=D,
            small_indices=operator.small_indices,
            structured_indices=operator.structured_indices,
        )

    if reml_penalties is not None:
        for component in reml_penalties:
            lam = _lambda_for_component(lambda2, component.name)
            if lam == 0.0:
                continue
            indices = _component_indices(component, p)
            local_small = small_position[indices]
            local_structured = structured_position[indices]
            wholly_small = np.all(local_small >= 0)
            wholly_structured = np.all(local_structured >= 0)
            if not wholly_small and not wholly_structured:
                raise ValueError(
                    f"Penalty component {component.name!r} crosses structured partitions."
                )
            if component.penalty_kind == "identity":
                if not wholly_small:
                    raise ValueError("The dominant SZ block accepts only a sum-to-zero penalty.")
                A[local_small, local_small] += lam
                continue
            if component.penalty_kind == "sum_to_zero":
                if (
                    not wholly_structured
                    or component.repeat_count != operator.n_levels
                    or component.block_width != operator.block_size
                    or not np.array_equal(
                        indices.reshape(operator.n_levels - 1, operator.block_size),
                        operator.structured_indices,
                    )
                ):
                    raise ValueError(
                        f"Sum-to-zero penalty component {component.name!r} does not "
                        "match the dominant SZ geometry."
                    )
                omega = np.asarray(component.omega_ssp, dtype=np.float64)
                if omega.shape != (operator.block_size, operator.block_size):
                    raise ValueError(
                        f"Sum-to-zero penalty component {component.name!r} has "
                        "the wrong local shape."
                    )
                D += lam * omega[None, :, :]
                continue
            if wholly_structured:
                raise ValueError("The dominant SZ block accepts only penalty_kind='sum_to_zero'.")
            omega = _dense_component_omega(
                component,
                group_matrices[component.group_index],
            )
            if omega.shape != (len(indices), len(indices)):
                raise ValueError(
                    f"Penalty component {component.name!r} has shape {omega.shape}; "
                    f"expected ({len(indices)}, {len(indices)})."
                )
            A[np.ix_(local_small, local_small)] += lam * omega
    else:
        for group_index, (matrix, group) in enumerate(zip(group_matrices, groups, strict=True)):
            if not group.penalized:
                continue
            indices = np.arange(group.start, group.end, dtype=np.intp)
            local_small = small_position[indices]
            local_structured = structured_position[indices]
            if isinstance(matrix, FactorSmoothGroupMatrix):
                if (
                    matrix.factor_basis != "sz"
                    or not np.all(local_structured >= 0)
                    or len(matrix.repeated_penalty_components) != 1
                ):
                    raise ValueError(
                        f"FactorSmooth group {group.name!r} does not match the dominant SZ block."
                    )
                suffix, omega = matrix.repeated_penalty_components[0]
                lam = (
                    float(
                        lambda2.get(
                            f"{group.name}:{suffix}",
                            lambda2.get(group.name, 0.0),
                        )
                    )
                    if isinstance(lambda2, dict)
                    else float(lambda2)
                )
                D += lam * np.asarray(omega, dtype=np.float64)[None, :, :]
                continue
            lam = (
                float(lambda2.get(group.name, 0.0)) if isinstance(lambda2, dict) else float(lambda2)
            )
            if lam == 0.0:
                continue
            if isinstance(matrix, RandomEffectGroupMatrix):
                if not np.all(local_small >= 0):
                    raise ValueError(
                        f"RandomEffect group {group.name!r} crosses structured partitions."
                    )
                A[local_small, local_small] += lam
                continue
            omega_raw = getattr(matrix, "omega", None)
            if omega_raw is None or not hasattr(matrix, "R_inv"):
                continue
            if not np.all(local_small >= 0):
                raise ValueError(
                    f"Penalty geometry for dominant group index {group_index} is unsupported."
                )
            omega = np.asarray(
                matrix.R_inv.T @ omega_raw @ matrix.R_inv,
                dtype=np.float64,
            )
            A[np.ix_(local_small, local_small)] += lam * omega

    return SumToZeroBlockOperator(
        A=A,
        C=operator.C,
        D=D,
        small_indices=operator.small_indices,
        structured_indices=operator.structured_indices,
    )


def build_penalized_structured_operator(
    system: (ScalarStructuredSystem | BlockStructuredSystem | SumToZeroBlockStructuredSystem),
    group_matrices: list[GroupMatrix],
    groups: list[GroupSlice],
    lambda2: float | dict[str, float],
    *,
    reml_penalties: list[PenaltyComponent] | None = None,
    S_override: NDArray | None = None,
) -> SymmetricBlockOperator | BlockSymmetricOperator | SumToZeroBlockOperator:
    """Dispatch compact penalty assembly by structured-system geometry."""
    if isinstance(system, SumToZeroBlockStructuredSystem):
        return build_penalized_sum_to_zero_operator(
            system,
            group_matrices,
            groups,
            lambda2,
            reml_penalties=reml_penalties,
            S_override=S_override,
        )
    if isinstance(system, BlockStructuredSystem):
        return build_penalized_block_operator(
            system,
            group_matrices,
            groups,
            lambda2,
            reml_penalties=reml_penalties,
            S_override=S_override,
        )
    return build_penalized_scalar_operator(
        system,
        group_matrices,
        groups,
        lambda2,
        reml_penalties=reml_penalties,
        S_override=S_override,
    )


def build_augmented_scalar_factor(
    system: ScalarStructuredSystem,
    penalized_operator: SymmetricBlockOperator,
) -> tuple[ScalarSchurFactor, NDArray]:
    """Add the unpenalized intercept and return its Schur factor and global RHS."""
    operator = system.operator
    if not np.array_equal(
        penalized_operator.small_indices,
        operator.small_indices,
    ) or not np.array_equal(
        penalized_operator.structured_indices,
        operator.structured_indices,
    ):
        raise ValueError("Penalized and unpenalized operators must use identical partitions.")

    q = len(operator.small_indices)
    p = operator.shape[0]
    A_augmented = np.empty((q + 1, q + 1), dtype=np.float64)
    A_augmented[0, 0] = system.sum_w
    A_augmented[0, 1:] = system.xtw_small
    A_augmented[1:, 0] = system.xtw_small
    A_augmented[1:, 1:] = penalized_operator.A
    C_augmented = np.empty((len(operator.structured_indices), q + 1))
    C_augmented[:, 0] = system.xtw_structured
    C_augmented[:, 1:] = operator.C
    small_indices = np.concatenate(
        [
            np.array([0], dtype=np.intp),
            operator.small_indices + 1,
        ]
    )
    structured_indices = operator.structured_indices + 1
    factor = ScalarSchurFactor(
        A=A_augmented,
        C=C_augmented,
        d=penalized_operator.d,
        small_indices=small_indices,
        structured_indices=structured_indices,
        term_name=system.dominant_group_name,
    )
    rhs = np.empty(p + 1, dtype=np.float64)
    rhs[0] = system.sum_wz
    rhs[operator.small_indices + 1] = system.xtwz_small
    rhs[operator.structured_indices + 1] = system.xtwz_structured
    return factor, rhs


def build_augmented_block_factor(
    system: BlockStructuredSystem,
    penalized_operator: BlockSymmetricOperator,
) -> tuple[BlockSchurFactor, NDArray]:
    """Add the intercept to a factor-smooth block system and factor it."""
    operator = system.operator
    if not np.array_equal(
        penalized_operator.small_indices,
        operator.small_indices,
    ) or not np.array_equal(
        penalized_operator.structured_indices,
        operator.structured_indices,
    ):
        raise ValueError("Penalized and unpenalized operators must use identical partitions.")

    q = len(operator.small_indices)
    p = operator.shape[0]
    A_augmented = np.empty((q + 1, q + 1), dtype=np.float64)
    A_augmented[0, 0] = system.sum_w
    A_augmented[0, 1:] = system.xtw_small
    A_augmented[1:, 0] = system.xtw_small
    A_augmented[1:, 1:] = penalized_operator.A
    C_augmented = np.empty(
        (operator.n_levels, operator.block_size, q + 1),
        dtype=np.float64,
    )
    C_augmented[:, :, 0] = system.xtw_structured
    C_augmented[:, :, 1:] = operator.C
    small_indices = np.concatenate(
        [
            np.array([0], dtype=np.intp),
            operator.small_indices + 1,
        ]
    )
    structured_indices = operator.structured_indices + 1
    factor = BlockSchurFactor(
        A=A_augmented,
        C=C_augmented,
        D=penalized_operator.D,
        small_indices=small_indices,
        structured_indices=structured_indices,
        term_name=system.dominant_group_name,
    )
    rhs = np.empty(p + 1, dtype=np.float64)
    rhs[0] = system.sum_wz
    rhs[operator.small_indices + 1] = system.xtwz_small
    rhs[operator.structured_indices + 1] = system.xtwz_structured
    return factor, rhs


def build_augmented_sum_to_zero_factor(
    system: SumToZeroBlockStructuredSystem,
    penalized_operator: SumToZeroBlockOperator,
):
    """Add the intercept and factor an SZ system in raw symmetric geometry."""
    from superglm.solvers.sum_to_zero import SumToZeroBlockFactor

    operator = system.operator
    if not np.array_equal(
        penalized_operator.small_indices,
        operator.small_indices,
    ) or not np.array_equal(
        penalized_operator.structured_indices,
        operator.structured_indices,
    ):
        raise ValueError("Penalized and unpenalized operators must use identical partitions.")
    q = len(operator.small_indices)
    p = operator.shape[0]
    A_augmented = np.empty((q + 1, q + 1), dtype=np.float64)
    A_augmented[0, 0] = system.sum_w
    A_augmented[0, 1:] = system.xtw_small
    A_augmented[1:, 0] = system.xtw_small
    A_augmented[1:, 1:] = penalized_operator.A
    C_augmented = np.empty(
        (operator.n_levels, operator.block_size, q + 1),
        dtype=np.float64,
    )
    C_augmented[:, :, 0] = system.raw_xtw_structured
    C_augmented[:, :, 1:] = operator.C
    small_indices = np.concatenate(
        (
            np.array([0], dtype=np.intp),
            operator.small_indices + 1,
        )
    )
    structured_indices = operator.structured_indices + 1
    factor = SumToZeroBlockFactor(
        A=A_augmented,
        C=C_augmented,
        D=penalized_operator.D,
        small_indices=small_indices,
        structured_indices=structured_indices,
        term_name=system.dominant_group_name,
        level_labels=system.level_labels,
    )
    rhs = np.empty(p + 1, dtype=np.float64)
    rhs[0] = system.sum_wz
    rhs[operator.small_indices + 1] = system.xtwz_small
    rhs[operator.structured_indices + 1] = system.xtwz_structured
    return factor, rhs


def build_augmented_structured_factor(
    system: (ScalarStructuredSystem | BlockStructuredSystem | SumToZeroBlockStructuredSystem),
    penalized_operator: (SymmetricBlockOperator | BlockSymmetricOperator | SumToZeroBlockOperator),
):
    """Dispatch intercept augmentation and Schur factorization."""
    if isinstance(system, SumToZeroBlockStructuredSystem):
        if not isinstance(penalized_operator, SumToZeroBlockOperator):
            raise TypeError("SZ structured systems require a sum-to-zero operator.")
        return build_augmented_sum_to_zero_factor(system, penalized_operator)
    if isinstance(system, BlockStructuredSystem):
        if not isinstance(penalized_operator, BlockSymmetricOperator):
            raise TypeError("Block structured systems require a block penalized operator.")
        return build_augmented_block_factor(system, penalized_operator)
    if not isinstance(penalized_operator, SymmetricBlockOperator):
        raise TypeError("Scalar structured systems require a scalar penalized operator.")
    return build_augmented_scalar_factor(system, penalized_operator)


def solve_cached_scalar_structured(
    system: ScalarStructuredSystem,
    group_matrices: list[GroupMatrix],
    groups: list[GroupSlice],
    lambdas: float | dict[str, float],
    *,
    reml_penalties: list[PenaltyComponent] | None = None,
) -> CachedScalarStructuredSolution:
    """Solve a lambda trial from cached working sufficient statistics."""
    penalized = build_penalized_scalar_operator(
        system,
        group_matrices,
        groups,
        lambdas,
        reml_penalties=reml_penalties,
    )
    augmented_factor, rhs = build_augmented_scalar_factor(system, penalized)
    coefficients = augmented_factor.solve(rhs)
    xtw = np.empty(system.operator.shape[0], dtype=np.float64)
    xtw[system.operator.small_indices] = system.xtw_small
    xtw[system.operator.structured_indices] = system.xtw_structured
    factor = ProfiledScalarSchurFactor(
        augmented_factor=augmented_factor,
        sum_w=system.sum_w,
        xtw=xtw,
    )
    return CachedScalarStructuredSolution(
        beta=coefficients[1:],
        intercept=float(coefficients[0]),
        factor=factor,
        penalized_operator=penalized,
        log_det_H=augmented_factor.logdet(),
        hessian_rank=augmented_factor.rank,
    )


def solve_cached_block_structured(
    system: BlockStructuredSystem,
    group_matrices: list[GroupMatrix],
    groups: list[GroupSlice],
    lambdas: float | dict[str, float],
    *,
    reml_penalties: list[PenaltyComponent] | None = None,
) -> CachedBlockStructuredSolution:
    """Solve a factor-smooth lambda trial from cached working moments."""
    penalized = build_penalized_block_operator(
        system,
        group_matrices,
        groups,
        lambdas,
        reml_penalties=reml_penalties,
    )
    augmented_factor, rhs = build_augmented_block_factor(system, penalized)
    coefficients = augmented_factor.solve(rhs)
    xtw = np.empty(system.operator.shape[0], dtype=np.float64)
    xtw[system.operator.small_indices] = system.xtw_small
    xtw[system.operator.structured_indices] = system.xtw_structured
    factor = ProfiledBlockSchurFactor(
        augmented_factor=augmented_factor,
        sum_w=system.sum_w,
        xtw=xtw,
    )
    return CachedBlockStructuredSolution(
        beta=coefficients[1:],
        intercept=float(coefficients[0]),
        factor=factor,
        penalized_operator=penalized,
        log_det_H=augmented_factor.logdet(),
        hessian_rank=augmented_factor.rank,
    )


def solve_cached_sum_to_zero_structured(
    system: SumToZeroBlockStructuredSystem,
    group_matrices: list[GroupMatrix],
    groups: list[GroupSlice],
    lambdas: float | dict[str, float],
    *,
    reml_penalties: list[PenaltyComponent] | None = None,
) -> CachedSumToZeroStructuredSolution:
    """Solve an SZ lambda trial from cached raw/public sufficient statistics."""
    from superglm.solvers.sum_to_zero import ProfiledSumToZeroBlockFactor

    penalized = build_penalized_sum_to_zero_operator(
        system,
        group_matrices,
        groups,
        lambdas,
        reml_penalties=reml_penalties,
    )
    augmented_factor, rhs = build_augmented_sum_to_zero_factor(system, penalized)
    coefficients = augmented_factor.solve(rhs)
    xtw = np.empty(system.operator.shape[0], dtype=np.float64)
    xtw[system.operator.small_indices] = system.xtw_small
    xtw[system.operator.structured_indices] = system.xtw_structured
    factor = ProfiledSumToZeroBlockFactor(
        augmented_factor=augmented_factor,
        sum_w=system.sum_w,
        xtw=xtw,
    )
    return CachedSumToZeroStructuredSolution(
        beta=coefficients[1:],
        intercept=float(coefficients[0]),
        factor=factor,
        penalized_operator=penalized,
        log_det_H=augmented_factor.logdet(),
        hessian_rank=augmented_factor.rank,
    )


def solve_cached_structured(
    system: (ScalarStructuredSystem | BlockStructuredSystem | SumToZeroBlockStructuredSystem),
    group_matrices: list[GroupMatrix],
    groups: list[GroupSlice],
    lambdas: float | dict[str, float],
    *,
    reml_penalties: list[PenaltyComponent] | None = None,
) -> (
    CachedScalarStructuredSolution
    | CachedBlockStructuredSolution
    | CachedSumToZeroStructuredSolution
):
    """Dispatch a cached lambda-only solve by dominant structured geometry."""
    if isinstance(system, SumToZeroBlockStructuredSystem):
        return solve_cached_sum_to_zero_structured(
            system,
            group_matrices,
            groups,
            lambdas,
            reml_penalties=reml_penalties,
        )
    if isinstance(system, BlockStructuredSystem):
        return solve_cached_block_structured(
            system,
            group_matrices,
            groups,
            lambdas,
            reml_penalties=reml_penalties,
        )
    return solve_cached_scalar_structured(
        system,
        group_matrices,
        groups,
        lambdas,
        reml_penalties=reml_penalties,
    )
