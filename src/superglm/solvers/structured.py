"""Structured linear algebra for dominant random-effect blocks."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

import numpy as np
import scipy.linalg
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
    CompactSymmetricOperator,
    SumToZeroBlockOperator,
    SymmetricBlockOperator,
    _BlockDiagonalLowRank,
    _DiagonalLowRank,
    _general_bdlr_diagonal,
    _general_bdlr_square_diagonal,
    _general_dlr_diagonal,
    _general_dlr_square_diagonal,
    _multiply_symmetric_bdlr,
    _multiply_symmetric_dlr,
    _operator_bdlr,
    _operator_dlr,
    _trace_general_bdlr_product,
    _trace_general_product,
    _trace_symmetric_bdlr,
    _trace_symmetric_dlr,
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
    _empty_block_part as _empty_block_part,
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
    _multiply_symmetric_bdlr_coalesced as _multiply_symmetric_bdlr_coalesced,
)
from superglm.solvers._structured.operators import (
    _sum_to_zero_operator_bdlr as _sum_to_zero_operator_bdlr,
)
from superglm.solvers._structured.operators import (
    compact_operator_diagonal as compact_operator_diagonal,
)
from superglm.solvers._structured.operators import (
    materialize_compact_operator as materialize_compact_operator,
)
from superglm.solvers.hessian_factor import _component_indices, _component_omega
from superglm.types import GroupSlice, PenaltyComponent

if TYPE_CHECKING:
    from superglm.solvers.sum_to_zero import (
        ProfiledSumToZeroBlockFactor,
        SumToZeroBlockFactor,
    )


@dataclass(frozen=True)
class StructuredGroupSelection:
    """Dominant structured group choice or a recorded dense-fallback reason."""

    group_index: int | None
    group_name: str | None
    fallback_reason: str | None


@dataclass(frozen=True)
class StructuredBackendDecision:
    """Resolved direct backend and the selected dominant block."""

    use_structured: bool
    group_index: int | None
    group_name: str | None
    fallback_reason: str | None


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


_AUTO_MIN_COEFFICIENT_WIDTH = 32
_AUTO_MAX_STRUCTURED_COST_RATIO = 0.75
_MAX_FUSED_DENSE_SMALL_WIDTH = 32


def _structured_auto_is_beneficial(
    dominant_size: int,
    small_size: int,
) -> tuple[bool, float]:
    """Apply the measured scalar-Schur crossover and return its cost ratio.

    July 2026 end-to-end REML profiles found dense wins below 32 slope
    coefficients, while scalar Schur wins materially at and above that width
    when its leading factor/solve work is no more than 75% of dense work.
    The intercept is included in both algebra estimates.
    """
    if dominant_size < 1 or small_size < 0:
        raise ValueError("Structured auto dimensions must be non-negative with a dominant block.")
    coefficient_width = dominant_size + small_size
    dense_dimension = coefficient_width + 1
    schur_small_dimension = small_size + 1
    dense_cost = float(dense_dimension**3)
    structured_cost = float(schur_small_dimension**3 + dominant_size * schur_small_dimension**2)
    cost_ratio = structured_cost / dense_cost
    return (
        coefficient_width >= _AUTO_MIN_COEFFICIENT_WIDTH
        and cost_ratio <= _AUTO_MAX_STRUCTURED_COST_RATIO,
        cost_ratio,
    )


def _block_structured_auto_is_beneficial(
    n_levels: int,
    block_size: int,
    small_size: int,
) -> tuple[bool, float]:
    """Estimate the block-Schur crossover from the actual ``K``, ``k``, and ``q``.

    The estimate counts local factorizations, local solves against the
    dense-small block, Schur accumulation, and the final dense-small
    factorization.  It intentionally ignores shared row-moment work, so auto
    selection only chooses the block backend when its linear algebra alone has
    a material cubic-cost advantage.
    """
    if n_levels < 1 or block_size < 1 or small_size < 0:
        raise ValueError(
            "Block structured auto dimensions require positive K and k and non-negative q."
        )
    coefficient_width = n_levels * block_size + small_size
    dense_dimension = coefficient_width + 1
    schur_small_dimension = small_size + 1
    dense_cost = float(dense_dimension**3)
    structured_cost = float(
        n_levels * block_size**3
        + n_levels * block_size**2 * schur_small_dimension
        + n_levels * block_size * schur_small_dimension**2
        + schur_small_dimension**3
    )
    cost_ratio = structured_cost / dense_cost
    return (
        coefficient_width >= _AUTO_MIN_COEFFICIENT_WIDTH
        and cost_ratio <= _AUTO_MAX_STRUCTURED_COST_RATIO,
        cost_ratio,
    )


def _sum_to_zero_structured_auto_is_beneficial(
    n_levels: int,
    block_size: int,
    small_size: int,
) -> tuple[bool, float]:
    """Estimate constrained SZ work from local blocks and its dense border."""
    if n_levels < 2 or block_size < 1 or small_size < 0:
        raise ValueError(
            "SZ structured auto dimensions require K >= 2, positive k, and non-negative q."
        )
    coefficient_width = (n_levels - 1) * block_size + small_size
    dense_dimension = coefficient_width + 1
    border_width = small_size + block_size + 1
    dense_cost = float(dense_dimension**3)
    structured_cost = float(
        n_levels * block_size**3 + n_levels * block_size**2 * border_width + border_width**3
    )
    cost_ratio = structured_cost / dense_cost
    return (
        coefficient_width >= _AUTO_MIN_COEFFICIENT_WIDTH
        and cost_ratio <= _AUTO_MAX_STRUCTURED_COST_RATIO,
        cost_ratio,
    )


def _selection_failure(
    reason: str,
    mode: Literal["auto", "structured"],
) -> StructuredGroupSelection:
    if mode == "structured":
        raise ValueError(f"direct_solve='structured' is ineligible: {reason}")
    return StructuredGroupSelection(
        group_index=None,
        group_name=None,
        fallback_reason=reason,
    )


def select_structured_group(
    group_matrices: list[GroupMatrix],
    groups: list[GroupSlice],
    *,
    mode: Literal["auto", "structured"],
) -> StructuredGroupSelection:
    """Select the largest eligible random-effect or factor-smooth block."""
    if mode not in ("auto", "structured"):
        raise ValueError("Structured selection mode must be 'auto' or 'structured'.")
    if len(group_matrices) != len(groups):
        raise ValueError("group_matrices and groups must have the same length.")

    for group in groups:
        if group.constraints is not None:
            return _selection_failure(
                f"group {group.name!r} has coefficient constraints",
                mode,
            )
        if group.scop_reparameterization is not None:
            return _selection_failure(
                f"group {group.name!r} has unsupported SCOP geometry",
                mode,
            )

    candidates = [
        index
        for index, matrix in enumerate(group_matrices)
        if isinstance(matrix, RandomEffectGroupMatrix | FactorSmoothGroupMatrix)
    ]
    if not candidates:
        return _selection_failure("the model has no RandomEffect or FactorSmooth term", mode)

    dominant_index = max(candidates, key=lambda index: group_matrices[index].shape[1])
    dominant_group = groups[dominant_index]
    dominant_matrix = group_matrices[dominant_index]
    if dominant_group.size != dominant_matrix.shape[1]:
        term_kind = (
            "FactorSmooth"
            if isinstance(dominant_matrix, FactorSmoothGroupMatrix)
            else "RandomEffect"
        )
        return _selection_failure(
            f"{term_kind} group {dominant_group.name!r} has inconsistent coefficient geometry",
            mode,
        )
    return StructuredGroupSelection(
        group_index=dominant_index,
        group_name=dominant_group.name,
        fallback_reason=None,
    )


def _factor_smooth_singular_local_level(
    matrix: FactorSmoothGroupMatrix,
    group_name: str,
    row_weights: NDArray,
    lambda2: float | dict[str, float],
) -> int | None:
    """Return the first structurally singular penalized local block, if any."""
    weights = np.asarray(row_weights, dtype=np.float64)
    if weights.shape != (matrix.shape[0],):
        raise ValueError("row_weights must match the structured design row count.")
    active_components: list[str] = []
    local_penalty = np.zeros((matrix.block_size, matrix.block_size), dtype=np.float64)
    for suffix, omega in matrix.repeated_penalty_components:
        if isinstance(lambda2, dict):
            lam = float(
                lambda2.get(
                    f"{group_name}:{suffix}",
                    lambda2.get(group_name, 0.0),
                )
            )
        else:
            lam = float(lambda2)
        if lam:
            active_components.append(suffix)
            local_penalty += np.asarray(omega, dtype=np.float64)

    penalty_eigenvalues = np.linalg.eigvalsh(local_penalty)
    penalty_scale = max(float(np.max(np.abs(penalty_eigenvalues), initial=0.0)), 1.0)
    penalty_threshold = np.finfo(np.float64).eps * max(matrix.block_size, 1) * penalty_scale * 10.0
    if penalty_eigenvalues[0] > penalty_threshold:
        return None

    positive_rows = weights > 0.0
    packed_support = np.packbits(positive_rows)
    support_digest = hashlib.blake2b(
        packed_support.data,
        digest_size=16,
    ).digest()
    cache_key = (tuple(active_components), support_digest)
    if getattr(matrix, "_structured_feasibility_key", None) == cache_key:
        return getattr(matrix, "_structured_feasibility_level", None)

    information, _xtw, _rhs = matrix.factor_smooth_sufficient_stats(
        positive_rows.astype(np.float64),
        np.zeros_like(weights),
    )
    local_blocks = np.asarray(information, dtype=np.float64) + local_penalty[None, :, :]

    eigenvalues = np.linalg.eigvalsh(local_blocks)
    scales = np.maximum(np.max(np.abs(eigenvalues), axis=1), 1.0)
    thresholds = np.finfo(np.float64).eps * max(matrix.block_size, 1) * scales * 10.0
    singular = eigenvalues[:, 0] <= thresholds
    singular_level = int(np.flatnonzero(singular)[0]) if np.any(singular) else None
    matrix._structured_feasibility_key = cache_key
    matrix._structured_feasibility_level = singular_level
    return singular_level


def resolve_structured_backend(
    group_matrices: list[GroupMatrix],
    groups: list[GroupSlice],
    *,
    direct_solve: str,
    coefficient_width: int,
    row_weights: NDArray | None = None,
    lambda2: float | dict[str, float] | None = None,
) -> StructuredBackendDecision:
    """Resolve forced/automatic scalar Schur use once for a direct fit."""
    if direct_solve not in ("auto", "structured"):
        return StructuredBackendDecision(
            use_structured=False,
            group_index=None,
            group_name=None,
            fallback_reason=None,
        )

    mode: Literal["auto", "structured"] = "structured" if direct_solve == "structured" else "auto"
    selection = select_structured_group(group_matrices, groups, mode=mode)
    if selection.group_index is None:
        return StructuredBackendDecision(
            use_structured=False,
            group_index=None,
            group_name=None,
            fallback_reason=selection.fallback_reason,
        )
    group_name = selection.group_name
    if group_name is None:  # pragma: no cover - StructuredGroupSelection invariant
        raise RuntimeError("structured group selection omitted its group name")
    if mode == "structured":
        return StructuredBackendDecision(
            use_structured=True,
            group_index=selection.group_index,
            group_name=group_name,
            fallback_reason=None,
        )

    dominant_matrix = group_matrices[selection.group_index]
    dominant_size = dominant_matrix.shape[1]
    small_size = coefficient_width - dominant_size
    if (
        isinstance(dominant_matrix, RandomEffectGroupMatrix)
        and row_weights is not None
        and lambda2 is not None
    ):
        if isinstance(lambda2, dict):
            dominant_lambda = lambda2.get(group_name)
        else:
            dominant_lambda = float(lambda2)
        if dominant_lambda is not None and float(dominant_lambda) == 0.0:
            weights = np.asarray(row_weights, dtype=np.float64)
            if weights.shape != (dominant_matrix.shape[0],):
                raise ValueError("row_weights must match the structured design row count.")
            level_weight = np.bincount(
                dominant_matrix.codes,
                weights=weights,
                minlength=dominant_matrix.n_levels,
            )
            if np.any(level_weight <= 0.0):
                return StructuredBackendDecision(
                    use_structured=False,
                    group_index=selection.group_index,
                    group_name=group_name,
                    fallback_reason=(
                        f"RandomEffect group {group_name!r} has a level with "
                        "zero total weight and zero penalty"
                    ),
                )
    if (
        isinstance(dominant_matrix, FactorSmoothGroupMatrix)
        and row_weights is not None
        and lambda2 is not None
    ):
        singular_level = _factor_smooth_singular_local_level(
            dominant_matrix,
            group_name,
            row_weights,
            lambda2,
        )
        if singular_level is not None:
            level_label = dominant_matrix.levels[singular_level]
            return StructuredBackendDecision(
                use_structured=False,
                group_index=selection.group_index,
                group_name=group_name,
                fallback_reason=(
                    f"FactorSmooth group {group_name!r} has a singular local block "
                    f"for level {level_label!r} under the requested weights and penalties"
                ),
            )
    if isinstance(dominant_matrix, FactorSmoothGroupMatrix):
        if dominant_matrix.factor_basis == "sz":
            use_structured, cost_ratio = _sum_to_zero_structured_auto_is_beneficial(
                dominant_matrix.n_levels,
                dominant_matrix.block_size,
                small_size,
            )
        else:
            use_structured, cost_ratio = _block_structured_auto_is_beneficial(
                dominant_matrix.n_levels,
                dominant_matrix.block_size,
                small_size,
            )
    else:
        use_structured, cost_ratio = _structured_auto_is_beneficial(
            dominant_size,
            small_size,
        )
    fallback_reason = None
    if not use_structured:
        geometry_name = (
            "FactorSmooth"
            if isinstance(
                group_matrices[selection.group_index],
                FactorSmoothGroupMatrix,
            )
            else "RandomEffect"
        )
        if isinstance(dominant_matrix, FactorSmoothGroupMatrix):
            dimensions = (
                f"K={dominant_matrix.n_levels}, k={dominant_matrix.block_size}, q={small_size}"
            )
        else:
            dimensions = f"K={dominant_size}, q={small_size}"
        fallback_reason = (
            f"{geometry_name} geometry is below the measured structured crossover "
            f"(p={coefficient_width}, {dimensions}, estimated_cost_ratio={cost_ratio:.3f}; "
            f"require p >= {_AUTO_MIN_COEFFICIENT_WIDTH} and ratio <= "
            f"{_AUTO_MAX_STRUCTURED_COST_RATIO:.2f})"
        )
    return StructuredBackendDecision(
        use_structured=use_structured,
        group_index=selection.group_index,
        group_name=selection.group_name,
        fallback_reason=fallback_reason,
    )


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


class ScalarSchurFactor:
    """Factorization of one diagonal random-effect block and a dense remainder."""

    backend = "structured"

    def __init__(
        self,
        *,
        A: NDArray,
        C: NDArray,
        d: NDArray,
        small_indices: NDArray,
        structured_indices: NDArray,
        term_name: str,
        max_structured_inverse_block: int = 256,
    ):
        self.A = np.asarray(A, dtype=np.float64)
        self.C = np.asarray(C, dtype=np.float64)
        self.d = np.asarray(d, dtype=np.float64)
        self.small_indices = np.asarray(small_indices, dtype=np.intp)
        self.structured_indices = np.asarray(structured_indices, dtype=np.intp)
        self.term_name = term_name
        self.dominant_group_name = term_name
        self.max_structured_inverse_block = int(max_structured_inverse_block)

        q = len(self.small_indices)
        k = len(self.structured_indices)
        if self.A.shape != (q, q):
            raise ValueError(f"A shape {self.A.shape} does not match ({q}, {q}).")
        if self.C.shape != (k, q):
            raise ValueError(f"C shape {self.C.shape} does not match ({k}, {q}).")
        if self.d.shape != (k,):
            raise ValueError(f"d shape {self.d.shape} does not match ({k},).")
        self.minimum_local_diagonal = float(np.min(self.d)) if k else float("inf")
        if np.any(self.d <= 0) or not np.all(np.isfinite(self.d)):
            raise np.linalg.LinAlgError(
                f"Structured term {term_name!r} has an invalid minimum local diagonal "
                f"{self.minimum_local_diagonal:.17g}; all local diagonals must be "
                "positive and finite."
            )

        all_indices = np.concatenate([self.small_indices, self.structured_indices])
        if len(np.unique(all_indices)) != len(all_indices):
            raise ValueError("small_indices and structured_indices must be disjoint.")
        if not np.array_equal(np.sort(all_indices), np.arange(len(all_indices))):
            raise ValueError("Structured index partitions must cover every coefficient once.")

        self.shape = (len(all_indices), len(all_indices))
        self._small_position = np.full(self.shape[0], -1, dtype=np.intp)
        self._small_position[self.small_indices] = np.arange(q)
        self._structured_position = np.full(self.shape[0], -1, dtype=np.intp)
        self._structured_position[self.structured_indices] = np.arange(k)
        self._d_inv = 1.0 / self.d
        self._F = self._d_inv[:, None] * self.C
        Q = self.A - self.C.T @ self._F
        self._Q = 0.5 * (Q + Q.T)
        self._Q_cholesky: NDArray | None = None
        self._Q_svd: tuple[NDArray, NDArray, NDArray] | None = None
        self.used_dense_fallback = False
        self.fallback_reason: str | None = None

        if q == 0:
            self.schur_condition_estimate = 1.0
            logdet_Q = 0.0
            self._Q_rank = 0
        else:
            try:
                self._Q_cholesky = scipy.linalg.cholesky(
                    self._Q,
                    lower=True,
                    check_finite=False,
                )
                probe_rhs = np.zeros(q)
                probe_rhs[0] = 1.0
                probe_solution = scipy.linalg.cho_solve(
                    (self._Q_cholesky, True),
                    probe_rhs,
                    check_finite=False,
                )
                residual = np.linalg.norm(self._Q @ probe_solution - probe_rhs)
                if not np.isfinite(residual) or residual >= 1e-6:
                    raise np.linalg.LinAlgError(
                        f"Schur Cholesky residual {residual:.3g} exceeds 1e-6"
                    )
                diagonal = np.abs(np.diag(self._Q_cholesky))
                self.schur_condition_estimate = float(
                    (diagonal.max() / max(diagonal.min(), 1e-300)) ** 2
                )
                logdet_Q = 2.0 * float(np.sum(np.log(diagonal)))
                self._Q_rank = q
            except (np.linalg.LinAlgError, ValueError) as error:
                self._Q_cholesky = None
                self.used_dense_fallback = True
                self.fallback_reason = f"Schur Cholesky fallback: {error}"
                U, singular_values, Vh = np.linalg.svd(self._Q, full_matrices=False)
                threshold = singular_values[0] * 1e-10 if len(singular_values) else 0.0
                positive = singular_values > threshold
                inverse_singular_values = np.zeros_like(singular_values)
                np.divide(
                    1.0,
                    singular_values,
                    out=inverse_singular_values,
                    where=positive,
                )
                self._Q_svd = (U, inverse_singular_values, Vh)
                self._Q_rank = int(np.count_nonzero(positive))
                if not len(singular_values) or singular_values[-1] <= threshold:
                    self.schur_condition_estimate = float("inf")
                else:
                    self.schur_condition_estimate = float(singular_values[0] / singular_values[-1])
                logdet_Q = float(np.sum(np.log(singular_values[positive])))

        self._logdet = float(np.sum(np.log(self.d)) + logdet_Q)
        self.rank = int(k + self._Q_rank)
        self.rank_truncated = self.rank < self.shape[0]
        self._Q_inverse_cache: NDArray | None = None
        self._inverse_dlr_cache: _DiagonalLowRank | None = None

    def _Q_solve(self, rhs: NDArray) -> NDArray:
        """Solve the dense-small Schur system using the cached robust factor."""
        values = np.asarray(rhs, dtype=np.float64)
        if self._Q.shape[0] == 0:
            return np.zeros_like(values)
        if self._Q_cholesky is not None:
            return scipy.linalg.cho_solve(
                (self._Q_cholesky, True),
                values,
                check_finite=False,
            )
        if self._Q_svd is None:
            raise RuntimeError("Structured Schur factor has no usable dense-small factor.")
        U, inverse_singular_values, Vh = self._Q_svd
        return (Vh.T * inverse_singular_values) @ (U.T @ values)

    def solve(self, rhs: NDArray) -> NDArray:
        """Solve the globally indexed block system for one or many right-hand sides."""
        values = np.asarray(rhs, dtype=np.float64)
        vector_rhs = values.ndim == 1
        if vector_rhs:
            values = values[:, None]
        if values.ndim != 2 or values.shape[0] != self.shape[0]:
            raise ValueError(
                f"rhs must have shape ({self.shape[0]},) or ({self.shape[0]}, m), "
                f"got {np.asarray(rhs).shape}."
            )

        rhs_a = values[self.small_indices]
        rhs_b = values[self.structured_indices]
        d_inv_rhs_b = self._d_inv[:, None] * rhs_b
        schur_rhs = rhs_a - self.C.T @ d_inv_rhs_b
        solution_a = self._Q_solve(schur_rhs)
        solution_b = d_inv_rhs_b - self._F @ solution_a
        solution = np.empty_like(values)
        solution[self.small_indices] = solution_a
        solution[self.structured_indices] = solution_b
        return solution[:, 0] if vector_rhs else solution

    def coefficient_estimable(self) -> NDArray:
        """Return coordinate estimability after dense-small rank truncation."""
        if not self.rank_truncated or self._Q_svd is None:
            return np.ones(self.shape[0], dtype=bool)
        _, inverse_singular_values, Vh = self._Q_svd
        null_small = Vh[inverse_singular_values == 0.0].T
        null_basis = np.zeros((self.shape[0], null_small.shape[1]))
        null_basis[self.small_indices] = null_small
        null_basis[self.structured_indices] = -self._F @ null_small
        return _coefficient_estimable_from_null_basis(self.shape[0], null_basis)

    def logdet(self) -> float:
        """Return the exact positive-definite log determinant."""
        return self._logdet

    def _Q_inverse(self) -> NDArray:
        if self._Q_inverse_cache is None:
            self._Q_inverse_cache = self._Q_solve(np.eye(self._Q.shape[0]))
        return self._Q_inverse_cache

    def _validate_selected_indices(self, indices: NDArray) -> NDArray[np.intp]:
        selected = np.asarray(indices, dtype=np.intp)
        if selected.ndim != 1:
            raise ValueError("Selected inverse indices must be one-dimensional.")
        if np.any((selected < 0) | (selected >= self.shape[0])):
            raise IndexError("Selected inverse index is outside the factor dimensions.")
        if len(np.unique(selected)) != len(selected):
            raise ValueError("Selected inverse indices must be unique.")
        return selected

    def selected_inverse_block(self, indices: NDArray) -> NDArray:
        """Return one requested principal inverse block without forming the full inverse."""
        selected = self._validate_selected_indices(indices)
        small_mask = self._small_position[selected] >= 0
        structured_mask = ~small_mask
        small_output = np.flatnonzero(small_mask)
        structured_output = np.flatnonzero(structured_mask)
        small_position = self._small_position[selected[small_mask]]
        structured_position = self._structured_position[selected[structured_mask]]
        if len(structured_position) > self.max_structured_inverse_block:
            raise ValueError(
                f"Refusing to materialize a {len(structured_position)} x "
                f"{len(structured_position)} inverse block for structured term "
                f"{self.term_name!r}; request its diagonal instead."
            )

        inverse = np.empty((len(selected), len(selected)), dtype=np.float64)
        Q_inverse = self._Q_inverse()
        if len(small_position):
            inverse[np.ix_(small_output, small_output)] = Q_inverse[
                np.ix_(small_position, small_position)
            ]
        if len(structured_position):
            F_selected = self._F[structured_position]
            structured_block = F_selected @ Q_inverse @ F_selected.T + np.diag(
                self._d_inv[structured_position]
            )
            inverse[np.ix_(structured_output, structured_output)] = structured_block
        if len(small_position) and len(structured_position):
            structured_small = -self._F[structured_position] @ Q_inverse[:, small_position]
            inverse[np.ix_(structured_output, small_output)] = structured_small
            inverse[np.ix_(small_output, structured_output)] = structured_small.T
        return inverse

    def selected_inverse_diagonal(self, indices: NDArray) -> NDArray:
        """Return requested inverse diagonal entries in global index order."""
        selected = self._validate_selected_indices(indices)
        diagonal = np.empty(len(selected), dtype=np.float64)
        small_mask = self._small_position[selected] >= 0
        if np.any(small_mask):
            small_position = self._small_position[selected[small_mask]]
            diagonal[small_mask] = np.diag(self._Q_inverse())[small_position]
        if np.any(~small_mask):
            structured_position = self._structured_position[selected[~small_mask]]
            F_selected = self._F[structured_position]
            diagonal[~small_mask] = self._d_inv[structured_position] + np.sum(
                (F_selected @ self._Q_inverse()) * F_selected,
                axis=1,
            )
        return diagonal

    def trace_inverse_penalty(self, component: PenaltyComponent) -> float:
        """Return ``trace(H^-1 Omega)`` without expanding identity penalties."""
        indices = _component_indices(component, self.shape[0])
        if component.penalty_kind == "identity":
            return float(np.sum(self.selected_inverse_diagonal(indices)))
        inverse_block = self.selected_inverse_block(indices)
        return float(np.trace(inverse_block @ _component_omega(component, self.shape[0])))

    def penalty_cross_trace(
        self,
        left: PenaltyComponent,
        right: PenaltyComponent,
        left_scale: float,
        right_scale: float,
    ) -> float:
        """Return a scaled ``trace(H^-1 Omega_l H^-1 Omega_r)``."""
        left_indices = _component_indices(left, self.shape[0])
        right_indices = _component_indices(right, self.shape[0])
        left_is_dominant = left.penalty_kind == "identity" and np.array_equal(
            np.sort(left_indices), np.sort(self.structured_indices)
        )
        right_is_dominant = right.penalty_kind == "identity" and np.array_equal(
            np.sort(right_indices), np.sort(self.structured_indices)
        )
        scale = float(left_scale * right_scale)

        if left_is_dominant and right_is_dominant:
            Q_inverse = self._Q_inverse()
            G = self._F.T @ self._F
            low_rank_diagonal = np.sum((self._F @ Q_inverse) * self._F, axis=1)
            trace_value = (
                self._d_inv @ self._d_inv
                + 2.0 * (self._d_inv @ low_rank_diagonal)
                + np.trace(Q_inverse @ G @ Q_inverse @ G)
            )
            return float(scale * trace_value)

        if right_is_dominant:
            return self.penalty_cross_trace(
                right,
                left,
                right_scale,
                left_scale,
            )

        if left_is_dominant:
            if np.any(self._structured_position[right_indices] >= 0):
                raise ValueError(
                    "A dominant identity cross-trace currently requires the other "
                    "penalty to lie wholly in the dense-small block."
                )
            right_positions = self._small_position[right_indices]
            inverse_b_right = -self._F @ self._Q_inverse()[:, right_positions]
            cross_product = inverse_b_right.T @ inverse_b_right
            if right.penalty_kind != "identity":
                cross_product = cross_product @ _component_omega(right, self.shape[0])
            return float(scale * np.trace(cross_product))

        selected = np.unique(np.concatenate([left_indices, right_indices]))
        inverse_selected = self.selected_inverse_block(selected)
        positions = np.full(self.shape[0], -1, dtype=np.intp)
        positions[selected] = np.arange(len(selected))
        left_positions = positions[left_indices]
        right_positions = positions[right_indices]
        right_left = inverse_selected[np.ix_(right_positions, left_positions)]
        left_right = inverse_selected[np.ix_(left_positions, right_positions)]
        if left.penalty_kind != "identity":
            right_left = right_left @ _component_omega(left, self.shape[0])
        if right.penalty_kind != "identity":
            left_right = left_right @ _component_omega(right, self.shape[0])
        return float(scale * np.trace(right_left @ left_right))

    def _inverse_dlr(self) -> _DiagonalLowRank:
        cached = self._inverse_dlr_cache
        if cached is not None:
            return cached
        basis = np.zeros((self.shape[0], len(self.small_indices)))
        if len(self.small_indices):
            basis[self.small_indices] = np.eye(len(self.small_indices))
            basis[self.structured_indices] = -self._F
        diagonal = np.zeros(self.shape[0])
        diagonal[self.structured_indices] = self._d_inv
        cached = _DiagonalLowRank(
            diagonal=diagonal,
            basis=basis,
            core=self._Q_inverse(),
        )
        self._inverse_dlr_cache = cached
        return cached

    def _penalty_operator(
        self,
        component: PenaltyComponent,
        scale: float,
    ) -> SymmetricBlockOperator:
        indices = _component_indices(component, self.shape[0])
        local_small = self._small_position[indices]
        local_structured = self._structured_position[indices]
        A = np.zeros_like(self.A)
        C = np.zeros_like(self.C)
        d = np.zeros_like(self.d)
        if component.penalty_kind == "identity":
            if np.all(local_small >= 0):
                A[local_small, local_small] = scale
            elif np.all(local_structured >= 0):
                d[local_structured] = scale
            else:
                raise ValueError("Identity penalty crosses structured partitions.")
        else:
            if not np.all(local_small >= 0):
                raise ValueError("A dense structured-operator penalty must lie in the small block.")
            A[np.ix_(local_small, local_small)] = scale * _component_omega(
                component,
                self.shape[0],
            )
        return SymmetricBlockOperator(
            A=A,
            C=C,
            d=d,
            small_indices=self.small_indices,
            structured_indices=self.structured_indices,
        )

    def trace_inverse_operator(self, operator: CompactSymmetricOperator) -> float:
        """Return ``trace(H^-1 O)`` from matching compact geometry."""
        if operator.shape != self.shape:
            raise ValueError("Operator and factor dimensions must match.")
        return _trace_symmetric_dlr(self._inverse_dlr(), _operator_dlr(operator))

    def inverse_operator_diagonal(
        self,
        operator: CompactSymmetricOperator,
    ) -> NDArray:
        """Return ``diag(H^-1 O)`` in O(Kq + q²) memory."""
        if operator.shape != self.shape:
            raise ValueError("Operator and factor dimensions must match.")
        return _general_dlr_diagonal(
            _multiply_symmetric_dlr(
                self._inverse_dlr(),
                _operator_dlr(operator),
            )
        )

    def inverse_operator_square_diagonal(
        self,
        operator: CompactSymmetricOperator,
    ) -> NDArray:
        """Return ``diag((H^-1 O)^2)`` compactly."""
        if operator.shape != self.shape:
            raise ValueError("Operator and factor dimensions must match.")
        return _general_dlr_square_diagonal(
            _multiply_symmetric_dlr(
                self._inverse_dlr(),
                _operator_dlr(operator),
            )
        )

    def operator_cross_trace(
        self,
        left: CompactSymmetricOperator,
        right: CompactSymmetricOperator,
    ) -> float:
        """Return ``trace(H^-1 O_left H^-1 O_right)`` compactly."""
        if left.shape != self.shape or right.shape != self.shape:
            raise ValueError("Operators and factor dimensions must match.")
        inverse = self._inverse_dlr()
        return _trace_general_product(
            _multiply_symmetric_dlr(inverse, _operator_dlr(left)),
            _multiply_symmetric_dlr(inverse, _operator_dlr(right)),
        )

    def penalty_operator_cross_trace(
        self,
        component: PenaltyComponent,
        scale: float,
        operator: CompactSymmetricOperator,
    ) -> float:
        """Return ``trace(H^-1 lambda*Omega H^-1 O)`` compactly."""
        return self.operator_cross_trace(
            self._penalty_operator(component, scale),
            operator,
        )


class BlockSchurFactor:
    """Factorization of repeated dense local blocks and a dense-small remainder."""

    backend = "structured"

    def __init__(
        self,
        *,
        A: NDArray,
        C: NDArray,
        D: NDArray,
        small_indices: NDArray,
        structured_indices: NDArray,
        term_name: str,
        max_structured_inverse_block: int = 256,
    ):
        self.A = np.asarray(A, dtype=np.float64)
        self.C = np.asarray(C, dtype=np.float64)
        self.D = np.asarray(D, dtype=np.float64)
        self.small_indices = np.asarray(small_indices, dtype=np.intp)
        self.structured_indices = np.asarray(structured_indices, dtype=np.intp)
        self.term_name = term_name
        self.dominant_group_name = term_name
        self.max_structured_inverse_block = int(max_structured_inverse_block)

        if self.C.ndim != 3:
            raise ValueError("C must have shape (n_levels, block_size, small_size).")
        self.n_levels, self.block_size, q = self.C.shape
        if self.A.shape != (q, q):
            raise ValueError(f"A shape {self.A.shape} does not match ({q}, {q}).")
        if self.D.shape != (self.n_levels, self.block_size, self.block_size):
            raise ValueError(
                f"D shape {self.D.shape} does not match "
                f"({self.n_levels}, {self.block_size}, {self.block_size})."
            )
        if self.small_indices.shape != (q,):
            raise ValueError("small_indices width does not match A.")
        if self.structured_indices.shape != (self.n_levels, self.block_size):
            raise ValueError("structured_indices shape does not match C and D.")
        if not np.all(np.isfinite(self.D)):
            raise np.linalg.LinAlgError(
                f"Structured term {term_name!r} has non-finite local blocks."
            )
        if not np.allclose(self.D, self.D.transpose(0, 2, 1), rtol=0.0, atol=1e-13):
            raise ValueError("Every local D block must be symmetric.")

        all_indices = np.concatenate([self.small_indices, self.structured_indices.ravel()])
        if len(np.unique(all_indices)) != len(all_indices):
            raise ValueError("small_indices and structured_indices must be disjoint.")
        if not np.array_equal(np.sort(all_indices), np.arange(len(all_indices))):
            raise ValueError("Structured index partitions must cover every coefficient once.")
        self.shape = (len(all_indices), len(all_indices))
        self._small_position = np.full(self.shape[0], -1, dtype=np.intp)
        self._small_position[self.small_indices] = np.arange(q)
        self._structured_position = np.full(self.shape[0], -1, dtype=np.intp)
        self._structured_position[self.structured_indices.ravel()] = np.arange(
            self.n_levels * self.block_size
        )

        local_eigenvalues = np.linalg.eigvalsh(self.D)
        minimum_flat = int(np.argmin(local_eigenvalues))
        minimum_level = minimum_flat // self.block_size
        self.minimum_local_eigenvalue = float(local_eigenvalues.ravel()[minimum_flat])
        self.minimum_local_diagonal = self.minimum_local_eigenvalue
        if self.minimum_local_eigenvalue <= 0.0:
            raise np.linalg.LinAlgError(
                f"Structured term {term_name!r} level {minimum_level} local block is not "
                f"positive definite (minimum eigenvalue "
                f"{self.minimum_local_eigenvalue:.17g})."
            )
        try:
            self._D_cholesky = np.linalg.cholesky(self.D)
            self._D_inv = np.linalg.inv(self.D)
        except np.linalg.LinAlgError as error:  # pragma: no cover - eigenvalue guard above
            raise np.linalg.LinAlgError(
                f"Structured term {term_name!r} failed local block factorization: {error}"
            ) from error
        local_residual = np.max(
            np.linalg.norm(
                np.einsum("kij,kjl->kil", self.D, self._D_inv, optimize=True)
                - np.eye(self.block_size)[None, :, :],
                axis=(1, 2),
            )
        )
        if not np.isfinite(local_residual) or local_residual >= 1e-6:
            raise np.linalg.LinAlgError(
                f"Structured term {term_name!r} local inverse residual "
                f"{local_residual:.3g} exceeds 1e-6."
            )

        self._F = np.einsum("kij,kjq->kiq", self._D_inv, self.C, optimize=True)
        eliminated = np.einsum("kiq,kir->qr", self.C, self._F, optimize=True)
        Q = self.A - eliminated
        self._Q = 0.5 * (Q + Q.T)
        schur_reference_scale = max(
            float(np.linalg.norm(self.A, ord=2)) if q else 0.0,
            float(np.linalg.norm(eliminated, ord=2)) if q else 0.0,
            1.0,
        )
        self._Q_cholesky: NDArray | None = None
        self._Q_svd: tuple[NDArray, NDArray, NDArray] | None = None
        self.used_dense_fallback = False
        self.fallback_reason: str | None = None

        if q == 0:
            self.schur_condition_estimate = 1.0
            logdet_Q = 0.0
            self._Q_rank = 0
        else:
            try:
                self._Q_cholesky = scipy.linalg.cholesky(
                    self._Q,
                    lower=True,
                    check_finite=False,
                )
                probe_rhs = np.zeros(q)
                probe_rhs[0] = 1.0
                probe_solution = scipy.linalg.cho_solve(
                    (self._Q_cholesky, True),
                    probe_rhs,
                    check_finite=False,
                )
                residual = np.linalg.norm(self._Q @ probe_solution - probe_rhs)
                if not np.isfinite(residual) or residual >= 1e-6:
                    raise np.linalg.LinAlgError(
                        f"Schur Cholesky residual {residual:.3g} exceeds 1e-6"
                    )
                diagonal = np.abs(np.diag(self._Q_cholesky))
                self.schur_condition_estimate = float(
                    (diagonal.max() / max(diagonal.min(), 1e-300)) ** 2
                )
                logdet_Q = 2.0 * float(np.sum(np.log(diagonal)))
                self._Q_rank = q
            except (np.linalg.LinAlgError, ValueError) as error:
                self._Q_cholesky = None
                self.used_dense_fallback = True
                self.fallback_reason = f"Schur Cholesky fallback: {error}"
                U, singular_values, Vh = np.linalg.svd(self._Q, full_matrices=False)
                threshold = (
                    max(
                        singular_values[0] * 1e-10,
                        np.finfo(np.float64).eps * schur_reference_scale * max(q, 1) * 10.0,
                    )
                    if len(singular_values)
                    else 0.0
                )
                positive = singular_values > threshold
                inverse_singular_values = np.zeros_like(singular_values)
                np.divide(
                    1.0,
                    singular_values,
                    out=inverse_singular_values,
                    where=positive,
                )
                self._Q_svd = (U, inverse_singular_values, Vh)
                self._Q_rank = int(np.count_nonzero(positive))
                if not len(singular_values) or singular_values[-1] <= threshold:
                    self.schur_condition_estimate = float("inf")
                else:
                    self.schur_condition_estimate = float(singular_values[0] / singular_values[-1])
                logdet_Q = float(np.sum(np.log(singular_values[positive])))

        local_logdet = 2.0 * float(np.sum(np.log(np.diagonal(self._D_cholesky, axis1=1, axis2=2))))
        self._logdet = local_logdet + logdet_Q
        self.rank = int(self.n_levels * self.block_size + self._Q_rank)
        self.rank_truncated = self.rank < self.shape[0]
        self._Q_inverse_cache: NDArray | None = None
        self._inverse_bdlr_cache: _BlockDiagonalLowRank | None = None

    def _Q_solve(self, rhs: NDArray) -> NDArray:
        values = np.asarray(rhs, dtype=np.float64)
        if self._Q.shape[0] == 0:
            return np.zeros_like(values)
        if self._Q_cholesky is not None:
            return scipy.linalg.cho_solve(
                (self._Q_cholesky, True),
                values,
                check_finite=False,
            )
        if self._Q_svd is None:
            raise RuntimeError("Block Schur factor has no usable dense-small factor.")
        U, inverse_singular_values, Vh = self._Q_svd
        return (Vh.T * inverse_singular_values) @ (U.T @ values)

    def _Q_inverse(self) -> NDArray:
        if self._Q_inverse_cache is None:
            self._Q_inverse_cache = self._Q_solve(np.eye(self._Q.shape[0]))
        return self._Q_inverse_cache

    def solve(self, rhs: NDArray) -> NDArray:
        """Solve the globally indexed block system for one or many RHS columns."""
        values = np.asarray(rhs, dtype=np.float64)
        vector_rhs = values.ndim == 1
        if vector_rhs:
            values = values[:, None]
        if values.ndim != 2 or values.shape[0] != self.shape[0]:
            raise ValueError(
                f"rhs must have shape ({self.shape[0]},) or ({self.shape[0]}, m), "
                f"got {np.asarray(rhs).shape}."
            )
        rhs_small = values[self.small_indices]
        rhs_structured = values[self.structured_indices]
        D_inv_rhs = np.einsum(
            "kij,kjm->kim",
            self._D_inv,
            rhs_structured,
            optimize=True,
        )
        schur_rhs = rhs_small - np.einsum(
            "kiq,kim->qm",
            self.C,
            D_inv_rhs,
            optimize=True,
        )
        solution_small = self._Q_solve(schur_rhs)
        solution_structured = D_inv_rhs - np.einsum(
            "kiq,qm->kim",
            self._F,
            solution_small,
            optimize=True,
        )
        solution = np.empty_like(values)
        solution[self.small_indices] = solution_small
        solution[self.structured_indices] = solution_structured
        return solution[:, 0] if vector_rhs else solution

    def coefficient_estimable(self) -> NDArray:
        """Return coordinate estimability after dense-small rank truncation."""
        if not self.rank_truncated or self._Q_svd is None:
            return np.ones(self.shape[0], dtype=bool)
        _, inverse_singular_values, Vh = self._Q_svd
        null_small = Vh[inverse_singular_values == 0.0].T
        null_basis = np.zeros((self.shape[0], null_small.shape[1]))
        null_basis[self.small_indices] = null_small
        null_basis[self.structured_indices] = -(self._F @ null_small)
        return _coefficient_estimable_from_null_basis(self.shape[0], null_basis)

    def logdet(self) -> float:
        return self._logdet

    def _validate_selected_indices(self, indices: NDArray) -> NDArray[np.intp]:
        selected = np.asarray(indices, dtype=np.intp)
        if selected.ndim != 1:
            raise ValueError("Selected inverse indices must be one-dimensional.")
        if np.any((selected < 0) | (selected >= self.shape[0])):
            raise IndexError("Selected inverse index is outside the factor dimensions.")
        if len(np.unique(selected)) != len(selected):
            raise ValueError("Selected inverse indices must be unique.")
        return selected

    def selected_inverse_block(self, indices: NDArray) -> NDArray:
        selected = self._validate_selected_indices(indices)
        small_mask = self._small_position[selected] >= 0
        structured_output = np.flatnonzero(~small_mask)
        small_output = np.flatnonzero(small_mask)
        small_position = self._small_position[selected[small_mask]]
        structured_position = self._structured_position[selected[~small_mask]]
        if len(structured_position) > self.max_structured_inverse_block:
            raise ValueError(
                f"Refusing to materialize a {len(structured_position)} x "
                f"{len(structured_position)} inverse block for structured term "
                f"{self.term_name!r}; request its diagonal instead."
            )

        inverse = np.empty((len(selected), len(selected)), dtype=np.float64)
        Q_inverse = self._Q_inverse()
        if len(small_position):
            inverse[np.ix_(small_output, small_output)] = Q_inverse[
                np.ix_(small_position, small_position)
            ]
        if len(structured_position):
            F_flat = self._F.reshape(self.n_levels * self.block_size, -1)
            F_selected = F_flat[structured_position]
            structured_block = F_selected @ Q_inverse @ F_selected.T
            levels = structured_position // self.block_size
            coordinates = structured_position % self.block_size
            for row in range(len(structured_position)):
                same_level = np.flatnonzero(levels == levels[row])
                structured_block[row, same_level] += self._D_inv[
                    levels[row],
                    coordinates[row],
                    coordinates[same_level],
                ]
            inverse[np.ix_(structured_output, structured_output)] = structured_block
        if len(small_position) and len(structured_position):
            F_flat = self._F.reshape(self.n_levels * self.block_size, -1)
            structured_small = -F_flat[structured_position] @ Q_inverse[:, small_position]
            inverse[np.ix_(structured_output, small_output)] = structured_small
            inverse[np.ix_(small_output, structured_output)] = structured_small.T
        return inverse

    def selected_inverse_diagonal(self, indices: NDArray) -> NDArray:
        selected = self._validate_selected_indices(indices)
        diagonal = np.empty(len(selected), dtype=np.float64)
        small_mask = self._small_position[selected] >= 0
        Q_inverse = self._Q_inverse()
        if np.any(small_mask):
            small_position = self._small_position[selected[small_mask]]
            diagonal[small_mask] = np.diag(Q_inverse)[small_position]
        if np.any(~small_mask):
            positions = self._structured_position[selected[~small_mask]]
            levels = positions // self.block_size
            coordinates = positions % self.block_size
            F_flat = self._F.reshape(self.n_levels * self.block_size, -1)
            F_selected = F_flat[positions]
            diagonal[~small_mask] = self._D_inv[levels, coordinates, coordinates] + np.sum(
                (F_selected @ Q_inverse) * F_selected, axis=1
            )
        return diagonal

    def _inverse_bdlr(self) -> _BlockDiagonalLowRank:
        cached = self._inverse_bdlr_cache
        if cached is not None:
            return cached
        q = len(self.small_indices)
        basis = np.zeros((self.shape[0], q), dtype=np.float64)
        if q:
            basis[self.small_indices] = np.eye(q)
            basis[self.structured_indices] = -self._F
        cached = _BlockDiagonalLowRank(
            blocks=self._D_inv,
            structured_indices=self.structured_indices,
            basis=basis,
            core=self._Q_inverse(),
            shape=self.shape,
        )
        self._inverse_bdlr_cache = cached
        return cached

    def _penalty_operator(
        self,
        component: PenaltyComponent,
        scale: float,
    ) -> BlockSymmetricOperator:
        indices = _component_indices(component, self.shape[0])
        local_small = self._small_position[indices]
        local_structured = self._structured_position[indices]
        A = np.zeros_like(self.A)
        C = np.zeros_like(self.C)
        D = np.zeros_like(self.D)
        if component.penalty_kind == "identity":
            if np.all(local_small >= 0):
                A[local_small, local_small] = scale
            elif np.all(local_structured >= 0):
                for position in local_structured:
                    level = position // self.block_size
                    coordinate = position % self.block_size
                    D[level, coordinate, coordinate] = scale
            else:
                raise ValueError("Identity penalty crosses structured partitions.")
        elif component.penalty_kind == "repeated":
            if not np.all(local_structured >= 0):
                raise ValueError("Repeated penalty must lie in the structured block.")
            if component.repeat_count != self.n_levels or component.block_width != self.block_size:
                raise ValueError("Repeated penalty geometry does not match the block factor.")
            if not np.array_equal(
                indices.reshape(self.n_levels, self.block_size), self.structured_indices
            ):
                raise ValueError("Repeated penalty ordering does not match the block factor.")
            omega = np.asarray(component.omega_ssp, dtype=np.float64)
            if omega.shape != (self.block_size, self.block_size):
                raise ValueError("Repeated penalty local matrix has the wrong shape.")
            D[:] = scale * omega
        else:
            if not np.all(local_small >= 0):
                raise ValueError("Dense penalties must lie in the block factor's small block.")
            A[np.ix_(local_small, local_small)] = scale * _component_omega(
                component,
                self.shape[0],
            )
        return BlockSymmetricOperator(
            A=A,
            C=C,
            D=D,
            small_indices=self.small_indices,
            structured_indices=self.structured_indices,
        )

    def trace_inverse_penalty(self, component: PenaltyComponent) -> float:
        return self.trace_inverse_operator(self._penalty_operator(component, 1.0))

    def penalty_cross_trace(
        self,
        left: PenaltyComponent,
        right: PenaltyComponent,
        left_scale: float,
        right_scale: float,
    ) -> float:
        return self.operator_cross_trace(
            self._penalty_operator(left, left_scale),
            self._penalty_operator(right, right_scale),
        )

    def trace_inverse_operator(self, operator: CompactSymmetricOperator) -> float:
        if operator.shape != self.shape:
            raise ValueError("Operator and factor dimensions must match.")
        return _trace_symmetric_bdlr(
            self._inverse_bdlr(),
            _operator_bdlr(operator, self.structured_indices),
        )

    def inverse_operator_diagonal(
        self,
        operator: CompactSymmetricOperator,
    ) -> NDArray:
        if operator.shape != self.shape:
            raise ValueError("Operator and factor dimensions must match.")
        return _general_bdlr_diagonal(
            _multiply_symmetric_bdlr(
                self._inverse_bdlr(),
                _operator_bdlr(operator, self.structured_indices),
            )
        )

    def inverse_operator_square_diagonal(
        self,
        operator: CompactSymmetricOperator,
    ) -> NDArray:
        if operator.shape != self.shape:
            raise ValueError("Operator and factor dimensions must match.")
        return _general_bdlr_square_diagonal(
            _multiply_symmetric_bdlr(
                self._inverse_bdlr(),
                _operator_bdlr(operator, self.structured_indices),
            )
        )

    def operator_cross_trace(
        self,
        left: CompactSymmetricOperator,
        right: CompactSymmetricOperator,
    ) -> float:
        if left.shape != self.shape or right.shape != self.shape:
            raise ValueError("Operators and factor dimensions must match.")
        inverse = self._inverse_bdlr()
        return _trace_general_bdlr_product(
            _multiply_symmetric_bdlr(
                inverse,
                _operator_bdlr(left, self.structured_indices),
            ),
            _multiply_symmetric_bdlr(
                inverse,
                _operator_bdlr(right, self.structured_indices),
            ),
        )

    def penalty_operator_cross_trace(
        self,
        component: PenaltyComponent,
        scale: float,
        operator: CompactSymmetricOperator,
    ) -> float:
        return self.operator_cross_trace(
            self._penalty_operator(component, scale),
            operator,
        )


class ProfiledBlockSchurFactor:
    """Profiled slope view of an augmented block-Schur factor."""

    backend = "structured"

    def __init__(
        self,
        *,
        augmented_factor: BlockSchurFactor,
        sum_w: float,
        xtw: NDArray,
    ):
        self.augmented_factor = augmented_factor
        self.sum_w = float(sum_w)
        self.xtw = np.asarray(xtw, dtype=np.float64)
        if not np.isfinite(self.sum_w) or self.sum_w <= 0.0:
            raise ValueError("sum_w must be positive and finite.")
        if augmented_factor.shape != (len(self.xtw) + 1, len(self.xtw) + 1):
            raise ValueError("Augmented factor width does not match xtw.")
        if 0 not in augmented_factor.small_indices:
            raise ValueError("The augmented intercept must belong to the dense-small block.")
        self.shape = (len(self.xtw), len(self.xtw))
        self.mean_x = self.xtw / self.sum_w
        self.rank = max(int(augmented_factor.rank) - 1, 0)
        self.rank_truncated = self.rank < self.shape[0]
        self.used_dense_fallback = augmented_factor.used_dense_fallback
        self.schur_condition_estimate = augmented_factor.schur_condition_estimate
        self.minimum_local_diagonal = augmented_factor.minimum_local_diagonal
        self.minimum_local_eigenvalue = augmented_factor.minimum_local_eigenvalue
        self.fallback_reason = augmented_factor.fallback_reason
        self.dominant_group_name = augmented_factor.dominant_group_name
        self.n_levels = augmented_factor.n_levels
        self.block_size = augmented_factor.block_size
        self.small_indices = augmented_factor.small_indices[1:] - 1
        self.structured_indices = augmented_factor.structured_indices - 1
        self._inverse_bdlr_cache: _BlockDiagonalLowRank | None = None

    @staticmethod
    def _shift_indices(indices: NDArray) -> NDArray[np.intp]:
        return np.asarray(indices, dtype=np.intp) + 1

    @staticmethod
    def _shift_component(component: PenaltyComponent) -> PenaltyComponent:
        start = component.group_sl.start
        stop = component.group_sl.stop
        if start is None or stop is None:
            raise ValueError("Penalty component slices must have explicit bounds.")
        return replace(
            component,
            group_sl=slice(start + 1, stop + 1, component.group_sl.step),
        )

    def solve(self, rhs: NDArray) -> NDArray:
        values = np.asarray(rhs, dtype=np.float64)
        vector_rhs = values.ndim == 1
        if vector_rhs:
            values = values[:, None]
        if values.ndim != 2 or values.shape[0] != self.shape[0]:
            raise ValueError(
                f"rhs must have shape ({self.shape[0]},) or ({self.shape[0]}, m), "
                f"got {np.asarray(rhs).shape}."
            )
        augmented_rhs = np.zeros((self.shape[0] + 1, values.shape[1]))
        augmented_rhs[1:] = values
        solution = self.augmented_factor.solve(augmented_rhs)[1:]
        return solution[:, 0] if vector_rhs else solution

    def logdet(self) -> float:
        return float(self.augmented_factor.logdet() - np.log(self.sum_w))

    def selected_inverse_block(self, indices: NDArray) -> NDArray:
        return self.augmented_factor.selected_inverse_block(self._shift_indices(indices))

    def selected_inverse_diagonal(self, indices: NDArray) -> NDArray:
        return self.augmented_factor.selected_inverse_diagonal(self._shift_indices(indices))

    def trace_inverse_penalty(self, component: PenaltyComponent) -> float:
        return self.augmented_factor.trace_inverse_penalty(self._shift_component(component))

    def penalty_cross_trace(
        self,
        left: PenaltyComponent,
        right: PenaltyComponent,
        left_scale: float,
        right_scale: float,
    ) -> float:
        return self.augmented_factor.penalty_cross_trace(
            self._shift_component(left),
            self._shift_component(right),
            left_scale,
            right_scale,
        )

    def _inverse_bdlr(self) -> _BlockDiagonalLowRank:
        cached = self._inverse_bdlr_cache
        if cached is not None:
            return cached
        augmented = self.augmented_factor
        q_augmented = len(augmented.small_indices)
        basis = np.zeros((self.shape[0], q_augmented), dtype=np.float64)
        if len(self.small_indices):
            basis[self.small_indices, 1:] = np.eye(len(self.small_indices))
        basis[self.structured_indices] = -augmented._F
        cached = _BlockDiagonalLowRank(
            blocks=augmented._D_inv,
            structured_indices=self.structured_indices,
            basis=basis,
            core=augmented._Q_inverse(),
            shape=self.shape,
        )
        self._inverse_bdlr_cache = cached
        return cached

    def trace_inverse_operator(self, operator: CompactSymmetricOperator) -> float:
        if operator.shape != self.shape:
            raise ValueError("Operator and factor dimensions must match.")
        return _trace_symmetric_bdlr(
            self._inverse_bdlr(),
            _operator_bdlr(operator, self.structured_indices),
        )

    def inverse_operator_diagonal(
        self,
        operator: CompactSymmetricOperator,
    ) -> NDArray:
        if operator.shape != self.shape:
            raise ValueError("Operator and factor dimensions must match.")
        return _general_bdlr_diagonal(
            _multiply_symmetric_bdlr(
                self._inverse_bdlr(),
                _operator_bdlr(operator, self.structured_indices),
            )
        )

    def inverse_operator_square_diagonal(
        self,
        operator: CompactSymmetricOperator,
    ) -> NDArray:
        if operator.shape != self.shape:
            raise ValueError("Operator and factor dimensions must match.")
        return _general_bdlr_square_diagonal(
            _multiply_symmetric_bdlr(
                self._inverse_bdlr(),
                _operator_bdlr(operator, self.structured_indices),
            )
        )

    def operator_cross_trace(
        self,
        left: CompactSymmetricOperator,
        right: CompactSymmetricOperator,
    ) -> float:
        if left.shape != self.shape or right.shape != self.shape:
            raise ValueError("Operators and factor dimensions must match.")
        inverse = self._inverse_bdlr()
        return _trace_general_bdlr_product(
            _multiply_symmetric_bdlr(
                inverse,
                _operator_bdlr(left, self.structured_indices),
            ),
            _multiply_symmetric_bdlr(
                inverse,
                _operator_bdlr(right, self.structured_indices),
            ),
        )

    def penalty_operator_cross_trace(
        self,
        component: PenaltyComponent,
        scale: float,
        operator: CompactSymmetricOperator,
    ) -> float:
        shifted = self._shift_component(component)
        penalty = self.augmented_factor._penalty_operator(shifted, scale)
        slope_penalty = BlockSymmetricOperator(
            A=penalty.A[1:, 1:],
            C=penalty.C[:, :, 1:],
            D=penalty.D,
            small_indices=self.small_indices,
            structured_indices=self.structured_indices,
        )
        return self.operator_cross_trace(slope_penalty, operator)


class ProfiledScalarSchurFactor:
    """Slope inverse induced by profiling an intercept from a scalar Schur factor.

    ``ScalarSchurFactor`` factors the raw augmented coefficient system
    ``[1, X]' W [1, X] + diag(0, S)``.  The lower-right block of its inverse is
    exactly the inverse of the centered slope Hessian.  This adapter exposes
    that block through the common Hessian-factor protocol without materializing
    a coefficient-by-coefficient matrix.
    """

    backend = "structured"

    def __init__(
        self,
        *,
        augmented_factor: ScalarSchurFactor,
        sum_w: float,
        xtw: NDArray,
    ):
        self.augmented_factor = augmented_factor
        self.sum_w = float(sum_w)
        self.xtw = np.asarray(xtw, dtype=np.float64)
        if not np.isfinite(self.sum_w) or self.sum_w <= 0.0:
            raise ValueError("sum_w must be positive and finite.")
        if augmented_factor.shape != (len(self.xtw) + 1, len(self.xtw) + 1):
            raise ValueError("Augmented factor width does not match xtw.")
        self.shape = (len(self.xtw), len(self.xtw))
        self.mean_x = self.xtw / self.sum_w
        self.rank = max(int(augmented_factor.rank) - 1, 0)
        self.rank_truncated = self.rank < self.shape[0]
        self.used_dense_fallback = augmented_factor.used_dense_fallback
        self.schur_condition_estimate = augmented_factor.schur_condition_estimate
        self.minimum_local_diagonal = augmented_factor.minimum_local_diagonal
        self.fallback_reason = augmented_factor.fallback_reason
        self.dominant_group_name = augmented_factor.dominant_group_name
        self.small_indices = augmented_factor.small_indices[1:] - 1
        self.structured_indices = augmented_factor.structured_indices - 1
        self._inverse_dlr_cache: _DiagonalLowRank | None = None

    @staticmethod
    def _shift_indices(indices: NDArray) -> NDArray[np.intp]:
        return np.asarray(indices, dtype=np.intp) + 1

    @staticmethod
    def _shift_component(component: PenaltyComponent) -> PenaltyComponent:
        start = component.group_sl.start
        stop = component.group_sl.stop
        if start is None or stop is None:
            raise ValueError("Penalty component slices must have explicit bounds.")
        return replace(
            component,
            group_sl=slice(start + 1, stop + 1, component.group_sl.step),
        )

    def solve(self, rhs: NDArray) -> NDArray:
        """Apply the profiled slope inverse to one or many right-hand sides."""
        values = np.asarray(rhs, dtype=np.float64)
        vector_rhs = values.ndim == 1
        if vector_rhs:
            values = values[:, None]
        if values.ndim != 2 or values.shape[0] != self.shape[0]:
            raise ValueError(
                f"rhs must have shape ({self.shape[0]},) or ({self.shape[0]}, m), "
                f"got {np.asarray(rhs).shape}."
            )
        augmented_rhs = np.zeros((self.shape[0] + 1, values.shape[1]))
        augmented_rhs[1:] = values
        solution = self.augmented_factor.solve(augmented_rhs)[1:]
        return solution[:, 0] if vector_rhs else solution

    def logdet(self) -> float:
        """Return the profiled centered-slope log determinant."""
        return float(self.augmented_factor.logdet() - np.log(self.sum_w))

    def selected_inverse_block(self, indices: NDArray) -> NDArray:
        return self.augmented_factor.selected_inverse_block(self._shift_indices(indices))

    def selected_inverse_diagonal(self, indices: NDArray) -> NDArray:
        return self.augmented_factor.selected_inverse_diagonal(self._shift_indices(indices))

    def trace_inverse_penalty(self, component: PenaltyComponent) -> float:
        return self.augmented_factor.trace_inverse_penalty(self._shift_component(component))

    def penalty_cross_trace(
        self,
        left: PenaltyComponent,
        right: PenaltyComponent,
        left_scale: float,
        right_scale: float,
    ) -> float:
        return self.augmented_factor.penalty_cross_trace(
            self._shift_component(left),
            self._shift_component(right),
            left_scale,
            right_scale,
        )

    def _inverse_dlr(self) -> _DiagonalLowRank:
        cached = self._inverse_dlr_cache
        if cached is not None:
            return cached
        q_augmented = len(self.augmented_factor.small_indices)
        basis = np.zeros((self.shape[0], q_augmented), dtype=np.float64)
        if len(self.small_indices):
            basis[self.small_indices, 1:] = np.eye(len(self.small_indices))
        basis[self.structured_indices] = -self.augmented_factor._F
        diagonal = np.zeros(self.shape[0], dtype=np.float64)
        diagonal[self.structured_indices] = self.augmented_factor._d_inv
        cached = _DiagonalLowRank(
            diagonal=diagonal,
            basis=basis,
            core=self.augmented_factor._Q_inverse(),
        )
        self._inverse_dlr_cache = cached
        return cached

    def _penalty_operator(
        self,
        component: PenaltyComponent,
        scale: float,
    ) -> SymmetricBlockOperator:
        indices = _component_indices(component, self.shape[0])
        small_positions = np.full(self.shape[0], -1, dtype=np.intp)
        small_positions[self.small_indices] = np.arange(len(self.small_indices))
        structured_positions = np.full(self.shape[0], -1, dtype=np.intp)
        structured_positions[self.structured_indices] = np.arange(len(self.structured_indices))
        local_small = small_positions[indices]
        local_structured = structured_positions[indices]
        A = np.zeros((len(self.small_indices), len(self.small_indices)))
        C = np.zeros((len(self.structured_indices), len(self.small_indices)))
        d = np.zeros(len(self.structured_indices))
        if component.penalty_kind == "identity":
            if np.all(local_small >= 0):
                A[local_small, local_small] = scale
            elif np.all(local_structured >= 0):
                d[local_structured] = scale
            else:
                raise ValueError("Identity penalty crosses structured partitions.")
        else:
            if not np.all(local_small >= 0):
                raise ValueError("A dense structured-operator penalty must lie in the small block.")
            A[np.ix_(local_small, local_small)] = scale * _component_omega(
                component,
                self.shape[0],
            )
        return SymmetricBlockOperator(
            A=A,
            C=C,
            d=d,
            small_indices=self.small_indices,
            structured_indices=self.structured_indices,
        )

    def trace_inverse_operator(self, operator: CompactSymmetricOperator) -> float:
        """Return ``trace(Hc^-1 Gc)`` for a matching raw data operator."""
        if operator.shape != self.shape:
            raise ValueError("Operator and factor dimensions must match.")
        if isinstance(operator, SymmetricBlockOperator):
            operator = CenteredBlockOperator(
                raw=operator,
                cross=self.xtw,
                total=self.sum_w,
                center=self.mean_x,
            )
        return _trace_symmetric_dlr(self._inverse_dlr(), _operator_dlr(operator))

    def inverse_operator_diagonal(
        self,
        operator: CompactSymmetricOperator,
    ) -> NDArray:
        """Return ``diag(Hc^-1 O)`` in O(Kq + q²) memory."""
        if operator.shape != self.shape:
            raise ValueError("Operator and factor dimensions must match.")
        if isinstance(operator, SymmetricBlockOperator):
            operator = CenteredBlockOperator(
                raw=operator,
                cross=self.xtw,
                total=self.sum_w,
                center=self.mean_x,
            )
        return _general_dlr_diagonal(
            _multiply_symmetric_dlr(
                self._inverse_dlr(),
                _operator_dlr(operator),
            )
        )

    def inverse_operator_square_diagonal(
        self,
        operator: CompactSymmetricOperator,
    ) -> NDArray:
        """Return ``diag((Hc^-1 O)^2)`` compactly."""
        if operator.shape != self.shape:
            raise ValueError("Operator and factor dimensions must match.")
        if isinstance(operator, SymmetricBlockOperator):
            operator = CenteredBlockOperator(
                raw=operator,
                cross=self.xtw,
                total=self.sum_w,
                center=self.mean_x,
            )
        return _general_dlr_square_diagonal(
            _multiply_symmetric_dlr(
                self._inverse_dlr(),
                _operator_dlr(operator),
            )
        )

    def operator_cross_trace(
        self,
        left: CompactSymmetricOperator,
        right: CompactSymmetricOperator,
    ) -> float:
        """Return ``trace(H^-1 O_left H^-1 O_right)`` compactly."""
        if left.shape != self.shape or right.shape != self.shape:
            raise ValueError("Operators and factor dimensions must match.")
        inverse = self._inverse_dlr()
        left_product = _multiply_symmetric_dlr(inverse, _operator_dlr(left))
        right_product = _multiply_symmetric_dlr(inverse, _operator_dlr(right))
        return _trace_general_product(left_product, right_product)

    def penalty_operator_cross_trace(
        self,
        component: PenaltyComponent,
        scale: float,
        operator: CompactSymmetricOperator,
    ) -> float:
        """Return ``trace(H^-1 lambda*Omega H^-1 O)`` compactly."""
        return self.operator_cross_trace(
            self._penalty_operator(component, scale),
            operator,
        )


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
