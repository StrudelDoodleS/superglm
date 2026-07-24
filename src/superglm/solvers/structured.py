"""Structured linear algebra for dominant random-effect blocks."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from superglm._group_matrix._group_matrix_algebra import _random_effect_cross_gram
from superglm._group_matrix._group_matrix_execution import MatrixExecutionPlan
from superglm._group_matrix._group_matrix_kernels import (
    _dense_small_weighted_moments,
    _random_effect_sufficient_stats,
)
from superglm.group_matrix import (
    DenseGroupMatrix,
    DesignMatrix,
    GroupMatrix,
    RandomEffectGroupMatrix,
)
from superglm.solvers.hessian_factor import _component_indices, _component_omega
from superglm.types import GroupSlice, PenaltyComponent


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
class CachedScalarStructuredSolution:
    """One lambda-only solve against cached structured working moments."""

    beta: NDArray
    intercept: float
    factor: ProfiledScalarSchurFactor
    penalized_operator: SymmetricBlockOperator
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
class StructuredLinearSystemState:
    """Authoritative compact factors and moments retained after a fit."""

    coefficient_factor: ScalarSchurFactor
    profiled_factor: ProfiledScalarSchurFactor
    augmented_factor: ScalarSchurFactor
    system: ScalarStructuredSystem
    penalized_operator: SymmetricBlockOperator
    centered_data_operator: CenteredBlockOperator
    support_totals: dict[str, StructuredLevelSupport]
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
    """Select the largest eligible random-effect block for scalar Schur elimination."""
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
        if isinstance(matrix, RandomEffectGroupMatrix)
    ]
    if not candidates:
        return _selection_failure("the model has no RandomEffect term", mode)

    dominant_index = max(candidates, key=lambda index: group_matrices[index].shape[1])
    dominant_group = groups[dominant_index]
    dominant_matrix = group_matrices[dominant_index]
    if dominant_group.size != dominant_matrix.shape[1]:
        return _selection_failure(
            f"RandomEffect group {dominant_group.name!r} has inconsistent coefficient geometry",
            mode,
        )
    return StructuredGroupSelection(
        group_index=dominant_index,
        group_name=dominant_group.name,
        fallback_reason=None,
    )


def resolve_structured_backend(
    group_matrices: list[GroupMatrix],
    groups: list[GroupSlice],
    *,
    direct_solve: str,
    coefficient_width: int,
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
    if mode == "structured":
        return StructuredBackendDecision(
            use_structured=True,
            group_index=selection.group_index,
            group_name=selection.group_name,
            fallback_reason=None,
        )

    dominant_size = group_matrices[selection.group_index].shape[1]
    small_size = coefficient_width - dominant_size
    use_structured, cost_ratio = _structured_auto_is_beneficial(
        dominant_size,
        small_size,
    )
    fallback_reason = None
    if not use_structured:
        fallback_reason = (
            "RandomEffect geometry is below the measured structured crossover "
            f"(p={coefficient_width}, K={dominant_size}, q={small_size}, "
            f"estimated_cost_ratio={cost_ratio:.3f}; require p >= "
            f"{_AUTO_MIN_COEFFICIENT_WIDTH} and ratio <= "
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


def structured_design_matvec(
    layout: ScalarStructuredLayout,
    group_matrices: list[GroupMatrix] | tuple[GroupMatrix, ...],
    beta: NDArray,
) -> NDArray:
    """Apply a grouped design while fusing a cached dense-small partition."""
    values = np.asarray(beta, dtype=np.float64)
    width = len(layout.small_indices) + len(layout.structured_indices)
    if values.shape != (width,):
        raise ValueError(f"beta must have shape ({width},).")
    dominant = group_matrices[layout.dominant_group_index]
    if not isinstance(dominant, RandomEffectGroupMatrix):
        raise ValueError("Structured layout no longer points to a random-effect group.")

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
    result += dominant.matvec(values[layout.structured_indices])
    return result


def structured_design_rmatvec(
    layout: ScalarStructuredLayout,
    group_matrices: list[GroupMatrix] | tuple[GroupMatrix, ...],
    rows: NDArray,
) -> NDArray:
    """Apply a grouped design transpose with one cached dense-small product."""
    values = np.asarray(rows, dtype=np.float64)
    dominant = group_matrices[layout.dominant_group_index]
    if not isinstance(dominant, RandomEffectGroupMatrix):
        raise ValueError("Structured layout no longer points to a random-effect group.")
    if values.shape != (dominant.shape[0],):
        raise ValueError(f"rows must have shape ({dominant.shape[0]},).")

    width = len(layout.small_indices) + len(layout.structured_indices)
    result = np.empty(width, dtype=np.float64)
    if layout.dense_small_matrix is not None:
        result[layout.small_indices] = layout.dense_small_matrix.T @ values
    elif layout.small_matrices:
        result[layout.small_indices] = np.concatenate(
            [matrix.rmatvec(values) for matrix in layout.small_matrices]
        )
    else:
        result[layout.small_indices] = np.empty(0, dtype=np.float64)
    result[layout.structured_indices] = dominant.rmatvec(values)
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


@dataclass(frozen=True)
class SymmetricBlockOperator:
    """Symmetric matrix represented by dense-small, cross, and diagonal blocks."""

    A: NDArray
    C: NDArray
    d: NDArray
    small_indices: NDArray
    structured_indices: NDArray
    shape: tuple[int, int] = field(init=False)

    def __post_init__(self):
        for name, dtype in (
            ("A", np.float64),
            ("C", np.float64),
            ("d", np.float64),
            ("small_indices", np.intp),
            ("structured_indices", np.intp),
        ):
            values = np.array(getattr(self, name), dtype=dtype, copy=True)
            values.setflags(write=False)
            object.__setattr__(self, name, values)

        q = len(self.small_indices)
        k = len(self.structured_indices)
        if self.A.shape != (q, q):
            raise ValueError(f"A shape {self.A.shape} does not match ({q}, {q}).")
        if self.C.shape != (k, q):
            raise ValueError(f"C shape {self.C.shape} does not match ({k}, {q}).")
        if self.d.shape != (k,):
            raise ValueError(f"d shape {self.d.shape} does not match ({k},).")
        all_indices = np.concatenate([self.small_indices, self.structured_indices])
        if len(np.unique(all_indices)) != len(all_indices):
            raise ValueError("small_indices and structured_indices must be disjoint.")
        if not np.array_equal(np.sort(all_indices), np.arange(len(all_indices))):
            raise ValueError("Structured index partitions must cover every coefficient once.")
        object.__setattr__(self, "shape", (len(all_indices), len(all_indices)))

    def matvec(self, rhs: NDArray) -> NDArray:
        """Apply the compact symmetric operator to one or many RHS columns."""
        values = np.asarray(rhs, dtype=np.float64)
        vector_rhs = values.ndim == 1
        if vector_rhs:
            values = values[:, None]
        if values.ndim != 2 or values.shape[0] != self.shape[0]:
            raise ValueError(f"rhs must have shape ({self.shape[0]},) or ({self.shape[0]}, m).")
        small_rhs = values[self.small_indices]
        structured_rhs = values[self.structured_indices]
        result = np.empty_like(values)
        result[self.small_indices] = self.A @ small_rhs + self.C.T @ structured_rhs
        result[self.structured_indices] = self.C @ small_rhs + self.d[:, None] * structured_rhs
        return result[:, 0] if vector_rhs else result


@dataclass(frozen=True)
class CenteredBlockOperator:
    """A block operator centered around a fixed weighted design mean."""

    raw: SymmetricBlockOperator
    cross: NDArray
    total: float
    center: NDArray
    shape: tuple[int, int] = field(init=False)

    def __post_init__(self):
        p = self.raw.shape[0]
        cross = np.array(self.cross, dtype=np.float64, copy=True)
        center = np.array(self.center, dtype=np.float64, copy=True)
        if cross.shape != (p,) or center.shape != (p,):
            raise ValueError("Centered operator vectors must match its coefficient width.")
        cross.setflags(write=False)
        center.setflags(write=False)
        object.__setattr__(self, "cross", cross)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "total", float(self.total))
        object.__setattr__(self, "shape", self.raw.shape)

    def matvec(self, rhs: NDArray) -> NDArray:
        values = np.asarray(rhs, dtype=np.float64)
        result = self.raw.matvec(values)
        if values.ndim == 1:
            center_projection = float(self.center @ values)
            cross_projection = float(self.cross @ values)
            return (
                result
                - self.cross * center_projection
                - self.center * cross_projection
                + self.total * self.center * center_projection
            )
        center_projection = self.center @ values
        cross_projection = self.cross @ values
        return (
            result
            - self.cross[:, None] * center_projection
            - self.center[:, None] * cross_projection
            + self.total * self.center[:, None] * center_projection
        )


@dataclass(frozen=True)
class LowRankSymmetricOperator:
    """A symmetric low-rank update ``U R U.T``."""

    basis: NDArray
    core: NDArray
    shape: tuple[int, int] = field(init=False)

    def __post_init__(self):
        basis = np.array(self.basis, dtype=np.float64, copy=True)
        core = np.array(self.core, dtype=np.float64, copy=True)
        if basis.ndim != 2 or core.shape != (basis.shape[1], basis.shape[1]):
            raise ValueError("Low-rank operator basis and core shapes are inconsistent.")
        if not np.allclose(core, core.T, rtol=0.0, atol=1e-14):
            raise ValueError("Low-rank operator core must be symmetric.")
        basis.setflags(write=False)
        core.setflags(write=False)
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "core", core)
        object.__setattr__(self, "shape", (basis.shape[0], basis.shape[0]))

    def matvec(self, rhs: NDArray) -> NDArray:
        values = np.asarray(rhs, dtype=np.float64)
        if values.ndim not in (1, 2) or values.shape[0] != self.shape[0]:
            raise ValueError("rhs does not match the low-rank operator width.")
        return self.basis @ (self.core @ (self.basis.T @ values))


@dataclass(frozen=True)
class SumBlockOperator:
    """A small sum of compact symmetric operators."""

    operators: tuple[
        SymmetricBlockOperator | CenteredBlockOperator | LowRankSymmetricOperator,
        ...,
    ]
    shape: tuple[int, int] = field(init=False)

    def __post_init__(self):
        if not self.operators:
            raise ValueError("A compact operator sum cannot be empty.")
        shape = self.operators[0].shape
        if any(operator.shape != shape for operator in self.operators[1:]):
            raise ValueError("All compact operators in a sum must have the same shape.")
        object.__setattr__(self, "shape", shape)

    def matvec(self, rhs: NDArray) -> NDArray:
        return sum(
            (operator.matvec(rhs) for operator in self.operators),
            start=np.zeros_like(np.asarray(rhs, dtype=np.float64)),
        )


CompactSymmetricOperator = (
    SymmetricBlockOperator | CenteredBlockOperator | LowRankSymmetricOperator | SumBlockOperator
)


@dataclass(frozen=True)
class _DiagonalLowRank:
    """Internal exact ``diag(d) + U R U.T`` representation."""

    diagonal: NDArray
    basis: NDArray
    core: NDArray


@dataclass(frozen=True)
class _GeneralDiagonalLowRank:
    """Internal exact ``diag(d) + L M R.T`` representation."""

    diagonal: NDArray
    left: NDArray
    core: NDArray
    right: NDArray


def _block_operator_dlr(operator: SymmetricBlockOperator) -> _DiagonalLowRank:
    p = operator.shape[0]
    q = len(operator.small_indices)
    diagonal = np.zeros(p, dtype=np.float64)
    diagonal[operator.structured_indices] = operator.d
    if q == 0:
        return _DiagonalLowRank(
            diagonal=diagonal,
            basis=np.empty((p, 0)),
            core=np.empty((0, 0)),
        )
    small_basis = np.zeros((p, q), dtype=np.float64)
    small_basis[operator.small_indices] = np.eye(q)
    cross_basis = np.zeros((p, q), dtype=np.float64)
    cross_basis[operator.structured_indices] = operator.C
    basis = np.column_stack((small_basis, cross_basis))
    core = np.block(
        [
            [operator.A, np.eye(q)],
            [np.eye(q), np.zeros((q, q))],
        ]
    )
    return _DiagonalLowRank(diagonal=diagonal, basis=basis, core=core)


def _merge_dlr(parts: tuple[_DiagonalLowRank, ...]) -> _DiagonalLowRank:
    if not parts:
        raise ValueError("At least one diagonal-low-rank part is required.")
    diagonal = sum(
        (part.diagonal for part in parts),
        start=np.zeros_like(parts[0].diagonal),
    )
    ranks = [part.core.shape[0] for part in parts]
    if not any(ranks):
        return _DiagonalLowRank(
            diagonal=diagonal,
            basis=np.empty((len(diagonal), 0)),
            core=np.empty((0, 0)),
        )
    basis = np.column_stack([part.basis for part in parts if part.core.shape[0]])
    core = scipy.linalg.block_diag(*[part.core for part in parts if part.core.shape[0]])
    return _DiagonalLowRank(diagonal=diagonal, basis=basis, core=core)


def _operator_dlr(operator: CompactSymmetricOperator) -> _DiagonalLowRank:
    if isinstance(operator, SumBlockOperator):
        return _merge_dlr(tuple(_operator_dlr(item) for item in operator.operators))
    if isinstance(operator, LowRankSymmetricOperator):
        return _DiagonalLowRank(
            diagonal=np.zeros(operator.shape[0]),
            basis=operator.basis,
            core=operator.core,
        )
    base = _block_operator_dlr(
        operator.raw if isinstance(operator, CenteredBlockOperator) else operator
    )
    if not isinstance(operator, CenteredBlockOperator):
        return base
    update_basis = np.column_stack((operator.cross, operator.center))
    update_core = np.array(
        [
            [0.0, -1.0],
            [-1.0, operator.total],
        ]
    )
    return _merge_dlr(
        (
            base,
            _DiagonalLowRank(
                diagonal=np.zeros(operator.shape[0]),
                basis=update_basis,
                core=update_core,
            ),
        )
    )


def _trace_symmetric_dlr(left: _DiagonalLowRank, right: _DiagonalLowRank) -> float:
    value = float(left.diagonal @ right.diagonal)
    if right.core.size:
        value += float(
            np.trace(right.core @ (right.basis.T @ (left.diagonal[:, None] * right.basis)))
        )
    if left.core.size:
        value += float(
            np.trace(left.core @ (left.basis.T @ (right.diagonal[:, None] * left.basis)))
        )
    if left.core.size and right.core.size:
        overlap = left.basis.T @ right.basis
        value += float(np.trace(left.core @ overlap @ right.core @ overlap.T))
    return value


def _multiply_symmetric_dlr(
    left: _DiagonalLowRank,
    right: _DiagonalLowRank,
) -> _GeneralDiagonalLowRank:
    diagonal = left.diagonal * right.diagonal
    left_parts: list[NDArray] = []
    core_parts: list[NDArray] = []
    right_parts: list[NDArray] = []
    if right.core.size:
        left_parts.append(left.diagonal[:, None] * right.basis)
        core_parts.append(right.core)
        right_parts.append(right.basis)
    if left.core.size:
        left_parts.append(left.basis)
        core_parts.append(left.core)
        right_parts.append(right.diagonal[:, None] * left.basis)
    if left.core.size and right.core.size:
        left_parts.append(left.basis)
        core_parts.append(left.core @ (left.basis.T @ right.basis) @ right.core)
        right_parts.append(right.basis)
    if not core_parts:
        empty = np.empty((len(diagonal), 0))
        return _GeneralDiagonalLowRank(
            diagonal=diagonal,
            left=empty,
            core=np.empty((0, 0)),
            right=empty,
        )
    return _GeneralDiagonalLowRank(
        diagonal=diagonal,
        left=np.column_stack(left_parts),
        core=scipy.linalg.block_diag(*core_parts),
        right=np.column_stack(right_parts),
    )


def _general_dlr_diagonal(operator: _GeneralDiagonalLowRank) -> NDArray:
    """Return the diagonal of a general diagonal-plus-low-rank operator."""
    diagonal = np.array(operator.diagonal, dtype=np.float64, copy=True)
    if operator.core.size:
        diagonal += np.sum((operator.left @ operator.core) * operator.right, axis=1)
    return diagonal


def _general_dlr_square_diagonal(operator: _GeneralDiagonalLowRank) -> NDArray:
    """Return the diagonal of the square of a general DLR operator."""
    diagonal = np.square(operator.diagonal)
    if not operator.core.size:
        return diagonal
    low_diagonal = np.sum((operator.left @ operator.core) * operator.right, axis=1)
    diagonal += 2.0 * operator.diagonal * low_diagonal
    square_left = operator.left @ operator.core @ (operator.right.T @ operator.left) @ operator.core
    diagonal += np.sum(square_left * operator.right, axis=1)
    return diagonal


def _trace_general_product(
    left: _GeneralDiagonalLowRank,
    right: _GeneralDiagonalLowRank,
) -> float:
    value = float(left.diagonal @ right.diagonal)
    if right.core.size:
        value += float(
            np.trace(right.core @ (right.right.T @ (left.diagonal[:, None] * right.left)))
        )
    if left.core.size:
        value += float(np.trace(left.core @ (left.right.T @ (right.diagonal[:, None] * left.left))))
    if left.core.size and right.core.size:
        value += float(
            np.trace(
                left.core @ (left.right.T @ right.left) @ right.core @ (right.right.T @ left.left)
            )
        )
    return value


def materialize_compact_operator(operator: CompactSymmetricOperator) -> NDArray:
    """Materialize a compact operator for dense-reference paths only."""
    return operator.matvec(np.eye(operator.shape[0]))


def compact_operator_diagonal(
    operator: CompactSymmetricOperator,
) -> NDArray:
    """Return an exact compact-operator diagonal in O(Kq + q²) memory."""
    if isinstance(operator, SumBlockOperator):
        return sum(
            (compact_operator_diagonal(item) for item in operator.operators),
            start=np.zeros(operator.shape[0]),
        )
    if isinstance(operator, LowRankSymmetricOperator):
        return np.sum((operator.basis @ operator.core) * operator.basis, axis=1)
    raw = operator.raw if isinstance(operator, CenteredBlockOperator) else operator
    diagonal = np.empty(raw.shape[0], dtype=np.float64)
    diagonal[raw.small_indices] = np.diag(raw.A)
    diagonal[raw.structured_indices] = raw.d
    if isinstance(operator, CenteredBlockOperator):
        diagonal = (
            diagonal - 2.0 * operator.cross * operator.center + operator.total * operator.center**2
        )
    return diagonal


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
