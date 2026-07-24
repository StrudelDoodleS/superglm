"""Structured linear algebra for dominant random-effect blocks."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from superglm._group_matrix._group_matrix_algebra import _random_effect_cross_gram
from superglm._group_matrix._group_matrix_kernels import (
    _random_effect_sufficient_stats,
)
from superglm.group_matrix import (
    GroupMatrix,
    RandomEffectGroupMatrix,
    _block_xtwx_signed,
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


def build_scalar_structured_system(
    group_matrices: list[GroupMatrix],
    groups: list[GroupSlice],
    W: NDArray,
    Wz: NDArray,
    *,
    dominant_group_index: int,
    tabmat_split=None,
) -> ScalarStructuredSystem:
    """Build exact scalar-Schur blocks without a full coefficient Gram matrix."""
    weights, weighted_rhs, dominant = _validate_structured_inputs(
        group_matrices,
        groups,
        W,
        Wz,
        dominant_group_index,
    )
    dominant_group = groups[dominant_group_index]
    structured_indices = np.arange(
        dominant_group.start,
        dominant_group.end,
        dtype=np.intp,
    )

    small_group_indices = [
        index for index in range(len(group_matrices)) if index != dominant_group_index
    ]
    small_matrices = [group_matrices[index] for index in small_group_indices]
    small_ranges = [
        np.arange(groups[index].start, groups[index].end, dtype=np.intp)
        for index in small_group_indices
    ]
    small_indices = np.concatenate(small_ranges) if small_ranges else np.empty(0, dtype=np.intp)
    local_groups: list[GroupSlice] = []
    local_start = 0
    for index in small_group_indices:
        group = groups[index]
        local_end = local_start + group.size
        local_groups.append(replace(group, start=local_start, end=local_end))
        local_start = local_end

    if len(small_indices):
        if tabmat_split is not None:
            A = np.asarray(
                tabmat_split.sandwich(weights, cols=small_indices),
                dtype=np.float64,
            )
        else:
            A = _block_xtwx_signed(small_matrices, local_groups, weights)
        C = np.concatenate(
            [_random_effect_cross_gram(dominant, matrix, weights) for matrix in small_matrices],
            axis=1,
        )
        xtw_small = np.concatenate([matrix.rmatvec(weights) for matrix in small_matrices])
        xtwz_small = np.concatenate([matrix.rmatvec(weighted_rhs) for matrix in small_matrices])
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
        small_indices=small_indices,
        structured_indices=structured_indices,
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
        dominant_group_name=dominant_group.name,
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

    def trace_inverse_operator(self, operator: SymmetricBlockOperator) -> float:
        """Return ``trace(H^-1 O)`` from matching compact block geometry."""
        if not np.array_equal(operator.small_indices, self.small_indices) or not np.array_equal(
            operator.structured_indices,
            self.structured_indices,
        ):
            raise ValueError("Operator and factor must use identical structured partitions.")
        Q_inverse = self._Q_inverse()
        inverse_ba = -self._F @ Q_inverse
        inverse_bb_diagonal = self._d_inv + np.sum(
            (self._F @ Q_inverse) * self._F,
            axis=1,
        )
        return float(
            np.trace(Q_inverse @ operator.A)
            + 2.0 * np.sum(inverse_ba * operator.C)
            + inverse_bb_diagonal @ operator.d
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

    def trace_inverse_operator(self, operator: SymmetricBlockOperator) -> float:
        """Return ``trace(Hc^-1 Gc)`` for a matching raw data operator."""
        if operator.shape != self.shape:
            raise ValueError("Operator and factor dimensions must match.")
        q = len(operator.small_indices)
        augmented_operator = SymmetricBlockOperator(
            A=np.pad(operator.A, ((1, 0), (1, 0))),
            C=np.pad(operator.C, ((0, 0), (1, 0))),
            d=operator.d,
            small_indices=np.concatenate(
                (np.array([0], dtype=np.intp), operator.small_indices + 1)
            ),
            structured_indices=operator.structured_indices + 1,
        )
        if augmented_operator.A.shape != (q + 1, q + 1):  # pragma: no cover - invariant
            raise RuntimeError("Augmented compact operator has inconsistent shape.")
        raw_trace = self.augmented_factor.trace_inverse_operator(augmented_operator)
        centered_correction = float(self.xtw @ self.solve(self.xtw) / self.sum_w)
        return float(raw_trace - centered_correction)
