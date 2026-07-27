"""Penalty assembly and cached solves for structured systems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from superglm.group_matrix import (
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
from superglm.solvers._structured.moments import (
    BlockStructuredSystem,
    ScalarStructuredSystem,
    SumToZeroBlockStructuredSystem,
)
from superglm.solvers._structured.operators import (
    BlockSymmetricOperator,
    SumToZeroBlockOperator,
    SymmetricBlockOperator,
)
from superglm.solvers._structured.overrides import (
    _factor_smooth_override_local_blocks,
    _structured_override_incompatibility,
)
from superglm.solvers.hessian_factor import _component_indices
from superglm.types import GroupSlice, PenaltyComponent

if TYPE_CHECKING:
    from superglm.solvers.sum_to_zero import ProfiledSumToZeroBlockFactor


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
        incompatibility = _structured_override_incompatibility(
            penalty,
            small_indices=operator.small_indices,
            structured_indices=operator.structured_indices,
            geometry="random_effect",
        )
        if incompatibility is not None:
            raise ValueError(incompatibility)
        A += penalty[np.ix_(operator.small_indices, operator.small_indices)]
        structured_penalty = penalty[
            np.ix_(operator.structured_indices, operator.structured_indices)
        ]
        d += np.diag(structured_penalty)
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
        incompatibility = _structured_override_incompatibility(
            penalty,
            small_indices=operator.small_indices,
            structured_indices=operator.structured_indices,
            geometry="factor_smooth",
        )
        if incompatibility is not None:
            raise ValueError(incompatibility)
        A += penalty[np.ix_(operator.small_indices, operator.small_indices)]
        D += _factor_smooth_override_local_blocks(
            penalty,
            operator.structured_indices,
            sum_to_zero=False,
        )
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
        incompatibility = _structured_override_incompatibility(
            penalty,
            small_indices=operator.small_indices,
            structured_indices=operator.structured_indices,
            geometry="sum_to_zero",
        )
        if incompatibility is not None:
            raise ValueError(incompatibility)
        A += penalty[np.ix_(operator.small_indices, operator.small_indices)]
        D += _factor_smooth_override_local_blocks(
            penalty,
            operator.structured_indices,
            sum_to_zero=True,
        )
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
