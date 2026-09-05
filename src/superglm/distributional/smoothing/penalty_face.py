"""Certified coefficient subspaces for smoothing parameters at infinity.

This module is internal.  It turns a set of disjoint penalty components into
the exact coefficient face on which every selected quadratic penalty is zero.
The solver continues to evaluate predictors in the ordinary stacked layout;
only coefficient updates are represented in the face's reduced coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from superglm.distributional.layout import StackedLayout
from superglm.reml.penalty_algebra import penalty_component_dense_matrix
from superglm.solvers.rank import RankDecomposition
from superglm.types import PenaltyComponent


class PenaltyFaceError(ValueError):
    """Raised when a requested infinity face cannot be certified."""


def _readonly(values: NDArray) -> NDArray[np.float64]:
    result = np.array(values, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _readonly_typed(values: NDArray, *, dtype: np.dtype | type) -> NDArray:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _spectral_norm(values: NDArray) -> float:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.size == 0:
        return 0.0
    return float(np.linalg.norm(matrix, ord=2))


def _component_slice(component: PenaltyComponent, width: int) -> slice:
    block = component.group_sl
    if (
        block.step not in (None, 1)
        or not isinstance(block.start, int)
        or not isinstance(block.stop, int)
        or block.start < 0
        or block.stop <= block.start
        or block.stop > width
    ):
        raise PenaltyFaceError(
            f"penalty component {component.name!r} has an invalid solver-space slice"
        )
    return block


def _declared_rank(component: PenaltyComponent, block_width: int) -> int:
    try:
        numeric = float(component.rank)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PenaltyFaceError(
            f"penalty component {component.name!r} has no finite structural rank"
        ) from exc
    rounded = int(round(numeric)) if np.isfinite(numeric) else -1
    if numeric != float(rounded) or rounded < 0 or rounded > block_width:
        raise PenaltyFaceError(
            f"penalty component {component.name!r} has no finite integer structural rank"
        )
    return rounded


def _expanded_components(
    layout: StackedLayout,
    component_names: tuple[str, ...],
) -> tuple[NDArray[np.float64], int]:
    width = layout.n_coefficients
    available = {component.name: component for component in layout.penalties}
    missing = [name for name in component_names if name not in available]
    if missing:
        raise PenaltyFaceError(f"unknown penalty face components: {missing}")

    occupied = np.zeros(width, dtype=bool)
    constraint = np.zeros((width, width), dtype=np.float64)
    expected_rank = 0
    eps = np.finfo(np.float64).eps
    for name in component_names:
        component = available[name]
        block = _component_slice(component, width)
        for other in layout.penalties:
            if other.name == component.name:
                continue
            other_block = _component_slice(other, width)
            if max(block.start, other_block.start) < min(block.stop, other_block.stop):
                raise PenaltyFaceError(
                    f"penalty component {component.name!r} belongs to a shared or "
                    "overlapping coefficient block"
                )
        if np.any(occupied[block]):
            raise PenaltyFaceError(
                "penalty face components overlap in coefficient space; "
                "shared and multi-penalty blocks are not supported"
            )
        occupied[block] = True
        try:
            expanded = np.asarray(
                penalty_component_dense_matrix(component),
                dtype=np.float64,
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise PenaltyFaceError(
                f"penalty component {component.name!r} cannot be expanded in solver space"
            ) from exc
        block_width = block.stop - block.start
        if expanded.shape != (block_width, block_width) or not np.all(np.isfinite(expanded)):
            raise PenaltyFaceError(
                f"penalty component {component.name!r} has invalid solver-space geometry"
            )
        entry_scale = float(np.max(np.abs(expanded), initial=0.0))
        if entry_scale == 0.0:
            normalized = np.array(expanded, copy=True)
        else:
            normalized = expanded / entry_scale
            spectral_scale = float(np.linalg.norm(normalized, ord=2))
            if spectral_scale == 0.0 or not np.isfinite(spectral_scale):
                raise PenaltyFaceError(
                    f"penalty component {component.name!r} has invalid solver-space geometry"
                )
            normalized /= spectral_scale
        symmetry_error = float(np.linalg.norm(normalized - normalized.T, ord=2))
        symmetry_bound = 32.0 * max(block_width, 1) * eps
        if symmetry_error > symmetry_bound:
            raise PenaltyFaceError(
                f"penalty component {component.name!r} is not numerically symmetric"
            )
        normalized = 0.5 * normalized + 0.5 * normalized.T
        # Only the null space of each disjoint component defines the face.
        # Normalize blocks independently so arbitrary positive penalty units do
        # not enter the later global rank certificate.
        constraint[block, block] = normalized
        expected_rank += _declared_rank(component, block_width)

    return _readonly(constraint), expected_rank


@dataclass(frozen=True)
class PenaltyFace:
    """Immutable certified null space of disjoint penalty components."""

    component_names: tuple[str, ...]
    coefficient_names: tuple[str, ...]
    constraint_matrix: NDArray[np.float64]
    null_basis: NDArray[np.float64]
    constraint_basis: NDArray[np.float64]
    constraint_rank: int
    rank_resolution: float = field(init=False)
    null_residual_bound: float = field(init=False)
    orthogonality_bound: float = field(init=False)

    def __post_init__(self) -> None:
        component_names = tuple(self.component_names)
        coefficient_names = tuple(self.coefficient_names)
        constraint = _readonly(self.constraint_matrix)
        basis = _readonly(self.null_basis)
        constraint_basis = _readonly(self.constraint_basis)
        width = len(coefficient_names)
        if not component_names or len(set(component_names)) != len(component_names):
            raise PenaltyFaceError("a penalty face requires unique component names")
        if constraint.shape != (width, width):
            raise PenaltyFaceError("penalty face constraint shape does not match its layout")
        if self.constraint_rank < 1 or self.constraint_rank > width:
            raise PenaltyFaceError("penalty face must impose at least one resolved constraint")
        if basis.shape != (width, width - self.constraint_rank):
            raise PenaltyFaceError("penalty face null basis has the wrong certified dimension")
        if constraint_basis.shape != (width, self.constraint_rank):
            raise PenaltyFaceError("penalty face constraint basis has the wrong dimension")
        if not (
            np.all(np.isfinite(constraint))
            and np.all(np.isfinite(basis))
            and np.all(np.isfinite(constraint_basis))
        ):
            raise PenaltyFaceError("penalty face geometry must be finite")
        scale = _spectral_norm(constraint)
        if scale == 0.0:
            raise PenaltyFaceError("penalty face constraint has zero numerical scale")
        eps = np.finfo(np.float64).eps
        rank_resolution = 64.0 * max(width, 1) * eps * scale
        null_residual_bound = rank_resolution
        orthogonality_bound = 64.0 * max(width, basis.shape[1], 1) * eps
        if _spectral_norm(constraint - constraint.T) > rank_resolution:
            raise PenaltyFaceError("penalty face constraint must be numerically symmetric")
        if _spectral_norm(constraint @ basis) > null_residual_bound:
            raise PenaltyFaceError("penalty face null basis failed its residual certificate")
        complete_basis = np.column_stack((basis, constraint_basis))
        orthogonality_error = _spectral_norm(complete_basis.T @ complete_basis - np.eye(width))
        if orthogonality_error > orthogonality_bound:
            raise PenaltyFaceError("penalty face null basis failed orthogonality certification")
        object.__setattr__(self, "component_names", component_names)
        object.__setattr__(self, "coefficient_names", coefficient_names)
        object.__setattr__(self, "constraint_matrix", constraint)
        object.__setattr__(self, "null_basis", basis)
        object.__setattr__(self, "constraint_basis", constraint_basis)
        object.__setattr__(self, "rank_resolution", rank_resolution)
        object.__setattr__(self, "null_residual_bound", null_residual_bound)
        object.__setattr__(self, "orthogonality_bound", orthogonality_bound)

    @property
    def width(self) -> int:
        return len(self.coefficient_names)

    @property
    def reduced_width(self) -> int:
        return self.null_basis.shape[1]

    @property
    def projector(self) -> NDArray[np.float64]:
        return _readonly(self.null_basis @ self.null_basis.T)

    def project(self, coefficients: NDArray) -> NDArray[np.float64]:
        values = np.asarray(coefficients, dtype=np.float64)
        if values.shape != (self.width,) or not np.all(np.isfinite(values)):
            raise ValueError("coefficients do not match the penalty face")
        return _readonly(self.null_basis @ (self.null_basis.T @ values))

    def reduce_vector(self, values: NDArray) -> NDArray[np.float64]:
        vector = np.asarray(values, dtype=np.float64)
        if vector.shape != (self.width,) or not np.all(np.isfinite(vector)):
            raise ValueError("vector does not match the penalty face")
        return _readonly(self.null_basis.T @ vector)

    def lift_vector(self, values: NDArray) -> NDArray[np.float64]:
        vector = np.asarray(values, dtype=np.float64)
        if vector.shape != (self.reduced_width,) or not np.all(np.isfinite(vector)):
            raise ValueError("reduced vector does not match the penalty face")
        return _readonly(self.null_basis @ vector)

    def reduce_matrix(self, values: NDArray) -> NDArray[np.float64]:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.shape != (self.width, self.width) or not np.all(np.isfinite(matrix)):
            raise ValueError("matrix does not match the penalty face")
        reduced = self.null_basis.T @ matrix @ self.null_basis
        return _readonly(0.5 * (reduced + reduced.T))

    def validate_layout(self, layout: StackedLayout) -> None:
        if not isinstance(layout, StackedLayout):
            raise TypeError("layout must be a StackedLayout")
        if tuple(layout.coefficient_names) != self.coefficient_names:
            raise PenaltyFaceError("penalty face belongs to a different coefficient layout")
        constraint, expected_rank = _expanded_components(layout, self.component_names)
        matrix_scale = max(
            float(np.linalg.norm(constraint, ord=2)),
            float(np.linalg.norm(self.constraint_matrix, ord=2)),
            np.finfo(np.float64).tiny,
        )
        matrix_error = float(np.linalg.norm(constraint - self.constraint_matrix, ord=2))
        matrix_bound = 1024.0 * max(self.width, 1) * np.finfo(np.float64).eps * matrix_scale
        if expected_rank != self.constraint_rank or matrix_error > matrix_bound:
            raise PenaltyFaceError("penalty face geometry does not match the coefficient layout")

    def lift_rank_decomposition(
        self,
        reduced: RankDecomposition,
    ) -> RankDecomposition:
        """Lift a reduced curvature decomposition without re-decomposing Q H Qᵀ."""
        if not isinstance(reduced, RankDecomposition):
            raise TypeError("reduced must be a RankDecomposition")
        if reduced.width != self.reduced_width:
            raise ValueError("reduced decomposition does not match the penalty face")

        rank = reduced.rank
        if reduced.cholesky_factor is not None:
            inverse_transpose = scipy.linalg.solve_triangular(
                reduced.cholesky_factor.T,
                np.eye(rank),
                lower=False,
                check_finite=False,
            )
            reduced_inverse_factor = np.zeros((self.reduced_width, rank), dtype=np.float64)
            scale = reduced.column_scale[reduced.active_columns]
            reduced_inverse_factor[reduced.active_columns, :] = inverse_transpose / scale[:, None]
        elif rank:
            if reduced.solution_basis is None or reduced.retained_values is None:
                raise PenaltyFaceError("reduced decomposition has no retained inverse factor")
            retained = np.asarray(reduced.retained_values, dtype=np.float64)
            if retained.shape != (rank,) or np.any(retained <= 0.0):
                raise PenaltyFaceError("reduced decomposition has invalid retained curvature")
            reduced_inverse_factor = reduced.solution_basis / np.sqrt(retained)
        else:
            reduced_inverse_factor = np.zeros((self.reduced_width, 0), dtype=np.float64)
        solution_basis = self.null_basis @ reduced_inverse_factor
        retained_values = np.ones(rank, dtype=np.float64)

        if reduced.estimable_functional_basis is not None:
            reduced_estimable = np.asarray(
                reduced.estimable_functional_basis,
                dtype=np.float64,
            )
            if reduced_estimable.shape != (self.reduced_width, rank) or not np.all(
                np.isfinite(reduced_estimable)
            ):
                raise PenaltyFaceError(
                    "reduced decomposition has invalid estimable-functional metadata"
                )
        elif reduced.method == "cholesky":
            active = np.asarray(reduced.active_columns, dtype=np.intp)
            if (
                active.shape != (rank,)
                or np.any(active < 0)
                or np.any(active >= self.reduced_width)
            ):
                raise PenaltyFaceError("reduced Cholesky decomposition has invalid active columns")
            reduced_estimable = np.zeros(
                (self.reduced_width, rank),
                dtype=np.float64,
            )
            reduced_estimable[active, np.arange(rank)] = 1.0
        elif rank == 0:
            reduced_estimable = np.zeros((self.reduced_width, 0), dtype=np.float64)
        else:
            raise PenaltyFaceError(
                "rank-deficient reduced decomposition has no estimable-functional basis"
            )
        estimable_basis = self.null_basis @ reduced_estimable

        reduced_null = reduced.null_basis()
        lifted_reduced_null = self.null_basis @ reduced_null
        full_null = np.column_stack((self.constraint_basis, lifted_reduced_null))
        return RankDecomposition(
            policy_version=reduced.policy_version,
            method="gram_eigh",
            column_scale=_readonly(np.ones(self.width, dtype=np.float64)),
            active_columns=_readonly_typed(
                np.arange(self.width, dtype=np.intp),
                dtype=np.intp,
            ),
            rank=rank,
            pre_truncation_condition=reduced.pre_truncation_condition,
            cutoff=reduced.cutoff,
            rank_truncated=rank < self.width,
            used_svd_fallback=reduced.used_svd_fallback,
            resolution_limited=reduced.resolution_limited,
            log_pdet=reduced.log_pdet,
            solution_basis=_readonly(solution_basis),
            parameter_null_basis=_readonly(full_null),
            estimable_functional_basis=_readonly(estimable_basis),
            structural_aliases=_readonly_typed(
                np.zeros(self.width, dtype=bool),
                dtype=bool,
            ),
            retained_values=_readonly(retained_values),
        )


def build_penalty_face(
    layout: StackedLayout,
    component_names: tuple[str, ...] | list[str],
) -> PenaltyFace:
    """Build the certified global null basis for disjoint penalty components."""
    if not isinstance(layout, StackedLayout):
        raise TypeError("layout must be a StackedLayout")
    try:
        names = tuple(component_names)
    except TypeError as exc:
        raise TypeError("component_names must be an iterable of names") from exc
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise PenaltyFaceError("a penalty face requires at least one component name")
    if len(set(names)) != len(names):
        raise PenaltyFaceError("penalty face component names must be unique")

    constraint, expected_rank = _expanded_components(layout, names)
    width = layout.n_coefficients
    if expected_rank < 1:
        raise PenaltyFaceError("selected penalty components impose no resolved constraint")
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(constraint)
    except np.linalg.LinAlgError as exc:
        raise PenaltyFaceError("penalty face rank eigendecomposition failed") from exc
    scale = float(np.max(np.abs(eigenvalues), initial=0.0))
    if scale == 0.0:
        raise PenaltyFaceError("selected penalty components have zero numerical scale")
    eps = np.finfo(np.float64).eps
    rank_resolution = 64.0 * max(width, 1) * eps * scale
    if float(eigenvalues[0]) < -rank_resolution:
        raise PenaltyFaceError("selected penalty constraint is materially indefinite")
    first_retained = width - expected_rank
    if first_retained < 0:
        raise PenaltyFaceError("declared penalty rank exceeds the coefficient width")
    discarded_max = float(np.max(np.abs(eigenvalues[:first_retained]), initial=0.0))
    retained_min = float(eigenvalues[first_retained]) if expected_rank else float("inf")
    if discarded_max > rank_resolution or retained_min <= 8.0 * rank_resolution:
        raise PenaltyFaceError(
            "penalty face rank cannot be resolved from its declared structural rank"
        )

    basis = np.asarray(eigenvectors[:, :first_retained], dtype=np.float64)
    orthogonality_bound = 64.0 * max(width, first_retained, 1) * eps
    orthogonality_error = float(np.linalg.norm(basis.T @ basis - np.eye(first_retained), ord=2))
    if orthogonality_error > orthogonality_bound:
        raise PenaltyFaceError("penalty face null basis failed orthogonality certification")
    null_residual_bound = 64.0 * max(width, 1) * eps * scale
    null_residual = float(np.linalg.norm(constraint @ basis, ord=2))
    if null_residual > null_residual_bound:
        raise PenaltyFaceError("penalty face null basis failed its residual certificate")

    return PenaltyFace(
        component_names=names,
        coefficient_names=tuple(layout.coefficient_names),
        constraint_matrix=constraint,
        null_basis=basis,
        constraint_basis=eigenvectors[:, first_retained:],
        constraint_rank=expected_rank,
    )


__all__ = ["PenaltyFace", "PenaltyFaceError", "build_penalty_face"]
