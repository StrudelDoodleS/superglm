"""Scalar and block Schur factorizations for structured systems."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from superglm.solvers._structured.geometry import (
    _coefficient_estimable_from_null_basis,
)
from superglm.solvers._structured.operators import (
    BlockSymmetricOperator,
    CenteredBlockOperator,
    CompactSymmetricOperator,
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
from superglm.solvers.hessian_factor import _component_indices, _component_omega
from superglm.types import PenaltyComponent


def _schur_absolute_cutoff(reference_scale: float, width: int) -> float:
    """Return the absolute rank floor for a cancellation-prone Schur block."""
    return np.finfo(np.float64).eps * float(reference_scale) * max(int(width), 1) * 10.0


def _reject_coupled_schur_null_space(
    F: NDArray,
    Vh: NDArray,
    positive: NDArray,
    *,
    term_name: str,
) -> None:
    """Reject singular Schur geometry whose null space couples to local blocks.

    The compact determinant and generalized-inverse formulae remain valid for
    an uncoupled Schur null space.  A coupled null direction is changed by the
    non-orthogonal block elimination, so multiplying local determinants by a
    Schur pseudo-determinant would publish the wrong REML geometry.
    """
    if np.all(positive):
        return
    null_basis = Vh[~positive].T
    flat_F = np.asarray(F, dtype=np.float64).reshape(-1, F.shape[-1])
    coupling = flat_F @ null_basis
    reference = max(float(np.linalg.norm(flat_F, ord=2)), 1.0)
    tolerance = (
        np.finfo(np.float64).eps * max(flat_F.shape[0], flat_F.shape[1], 1) * reference * 10.0
    )
    if float(np.linalg.norm(coupling, ord=2)) > tolerance:
        raise np.linalg.LinAlgError(
            f"Structured term {term_name!r} has a coupled rank-deficient Schur null space."
        )


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
        # Only `d` was checked here. A non-finite A or C reaches no guard at
        # all: `np.linalg.norm(..., ord=2)` returns nan for an inf input rather
        # than raising, so the factor builds and carries a nan scale into the
        # REML criterion. A refusal with the wrong class still stops something;
        # silent acceptance distorts smoothing-parameter selection while every
        # accuracy metric stays flat.
        if not np.all(np.isfinite(self.A)) or not np.all(np.isfinite(self.C)):
            raise np.linalg.LinAlgError(
                f"Structured term {term_name!r} has non-finite ordinary or cross blocks."
            )
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
        eliminated = self.C.T @ self._F
        Q = self.A - eliminated
        self._Q = 0.5 * (Q + Q.T)
        schur_reference_scale = max(
            float(np.linalg.norm(self.A, ord=2)) if q else 0.0,
            float(np.linalg.norm(eliminated, ord=2)) if q else 0.0,
            1.0,
        )
        absolute_cutoff = _schur_absolute_cutoff(schur_reference_scale, q)
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
                if float(np.min(diagonal * diagonal)) <= absolute_cutoff:
                    raise np.linalg.LinAlgError(
                        "Schur Cholesky pivot is below the absolute cancellation floor"
                    )
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
                    max(singular_values[0] * 1e-10, absolute_cutoff)
                    if len(singular_values)
                    else absolute_cutoff
                )
                positive = singular_values > threshold
                _reject_coupled_schur_null_space(
                    self._F,
                    Vh,
                    positive,
                    term_name=term_name,
                )
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
        # A and C join D for the reason given in ScalarSchurFactor above: an inf
        # in either passes every remaining check and only shows up later as a
        # nan logdet.
        if (
            not np.all(np.isfinite(self.A))
            or not np.all(np.isfinite(self.C))
            or not np.all(np.isfinite(self.D))
        ):
            raise np.linalg.LinAlgError(
                f"Structured term {term_name!r} has non-finite ordinary, cross, or local blocks."
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
        absolute_cutoff = _schur_absolute_cutoff(schur_reference_scale, q)
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
                if float(np.min(diagonal * diagonal)) <= absolute_cutoff:
                    raise np.linalg.LinAlgError(
                        "Schur Cholesky pivot is below the absolute cancellation floor"
                    )
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
                    max(singular_values[0] * 1e-10, absolute_cutoff)
                    if len(singular_values)
                    else absolute_cutoff
                )
                positive = singular_values > threshold
                _reject_coupled_schur_null_space(
                    self._F,
                    Vh,
                    positive,
                    term_name=term_name,
                )
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
