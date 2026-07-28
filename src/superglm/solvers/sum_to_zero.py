"""Rank-aware structured factorization for sum-to-zero factor smooths."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from superglm.solvers._structured.operators import (
    BlockSymmetricOperator,
    CompactSymmetricOperator,
    SumToZeroBlockOperator,
    _BlockDiagonalLowRank,
    _general_bdlr_diagonal,
    _general_bdlr_square_diagonal,
    _multiply_symmetric_bdlr_coalesced,
    _operator_bdlr,
    _trace_general_bdlr_product,
    _trace_symmetric_bdlr,
)
from superglm.solvers.hessian_factor import _component_indices, _component_omega
from superglm.types import PenaltyComponent


class SumToZeroIdentifiabilityError(np.linalg.LinAlgError):
    """Raised when the globally constrained SZ system is not identifiable."""


@dataclass(frozen=True)
class _LocalPSD:
    """Positive-range inverse and null basis for one symmetric local block."""

    pinv: NDArray
    null: NDArray
    positive_eigenvalues: NDArray
    rank: int


def _decompose_local_psd(
    block: NDArray,
    *,
    term_name: str,
    level_label: Any,
) -> _LocalPSD:
    """Decompose a PSD block without requiring every level to be full rank."""
    locals_, _minimum = _decompose_local_psd_batch(
        np.asarray(block, dtype=np.float64)[None, :, :],
        term_name=term_name,
        level_labels=(level_label,),
    )
    return locals_[0]


def _decompose_local_psd_batch(
    blocks: NDArray,
    *,
    term_name: str,
    level_labels: tuple[Any, ...],
) -> tuple[tuple[_LocalPSD, ...], float]:
    """Decompose every local PSD block in one batched LAPACK dispatch."""
    values = np.asarray(blocks, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError("Local blocks must have shape (K, k, k).")
    if values.shape[0] != len(level_labels):
        raise ValueError("level_labels length must equal the number of local blocks.")
    symmetric = 0.5 * (values + values.transpose(0, 2, 1))
    finite = np.all(np.isfinite(symmetric), axis=(1, 2))
    if not np.all(finite):
        level = int(np.flatnonzero(~finite)[0])
        raise np.linalg.LinAlgError(
            f"Structured term {term_name!r} level {level_labels[level]!r} has non-finite curvature."
        )

    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scales = np.maximum(np.max(np.abs(eigenvalues), axis=1), 1.0)
    # A large smoothing parameter can put O(1) data curvature beside an
    # O(1e11) wiggle penalty.  The usual dimension-scaled roundoff threshold
    # retains that real null-space information; eps**(2/3) would incorrectly
    # discard eigenvalues as large as about five at lambda=1e10.
    thresholds = np.finfo(np.float64).eps * symmetric.shape[1] * scales * 10.0
    negative = eigenvalues[:, 0] < -thresholds
    if np.any(negative):
        level = int(np.flatnonzero(negative)[0])
        raise np.linalg.LinAlgError(
            f"Structured term {term_name!r} level {level_labels[level]!r} "
            f"has negative local curvature ({eigenvalues[level, 0]:.17g})."
        )

    positive = eigenvalues > thresholds[:, None]
    full_rank = np.all(positive, axis=1)
    full_inverses = (
        np.linalg.inv(symmetric[full_rank])
        if np.any(full_rank)
        else np.empty((0, symmetric.shape[1], symmetric.shape[2]), dtype=np.float64)
    )
    full_inverse_index = 0
    locals_list = []
    for level in range(values.shape[0]):
        positive_values = np.asarray(
            eigenvalues[level][positive[level]],
            dtype=np.float64,
        )
        positive_vectors = np.asarray(
            eigenvectors[level][:, positive[level]],
            dtype=np.float64,
        )
        if full_rank[level]:
            # Divide-and-conquer eigenvectors can lose relative inverse
            # accuracy when O(1) data curvature shares a block with an
            # O(1e11) smoothing penalty.  Once the spectrum has certified
            # full rank, a batched direct inverse is both more accurate and
            # still compact.  Rank-deficient blocks retain the spectral
            # pseudo-inverse and null geometry below.
            pinv = full_inverses[full_inverse_index]
            full_inverse_index += 1
        elif positive_values.size:
            pinv = (positive_vectors / positive_values) @ positive_vectors.T
        else:
            pinv = np.zeros_like(symmetric[level])
        locals_list.append(
            _LocalPSD(
                pinv=0.5 * (pinv + pinv.T),
                null=np.asarray(
                    eigenvectors[level][:, ~positive[level]],
                    dtype=np.float64,
                ),
                positive_eigenvalues=positive_values,
                rank=int(positive_values.size),
            )
        )
    locals_ = tuple(locals_list)
    return locals_, float(np.min(eigenvalues))


def _constraint_equilibration(matrix: NDArray) -> NDArray:
    """Whiten the positive range of a small constraint covariance."""
    symmetric = 0.5 * (
        np.asarray(matrix, dtype=np.float64) + np.asarray(matrix, dtype=np.float64).T
    )
    if symmetric.shape == (0, 0):
        return symmetric
    eigenvalues, eigenvectors = scipy.linalg.eigh(
        symmetric,
        driver="evr",
        check_finite=False,
    )
    scale = max(float(np.max(np.abs(eigenvalues), initial=0.0)), 1.0)
    threshold = np.finfo(np.float64).eps * max(symmetric.shape[0], 1) * scale * 10.0
    factors = np.ones_like(eigenvalues)
    positive = eigenvalues > threshold
    factors[positive] = 1.0 / np.sqrt(eigenvalues[positive])
    return (eigenvectors * factors) @ eigenvectors.T


class _SymmetricBorderFactor:
    """LDL factor with a residual-checked SVD fallback for a small border."""

    def __init__(self, matrix: NDArray):
        border = np.asarray(matrix, dtype=np.float64)
        if border.ndim != 2 or border.shape[0] != border.shape[1]:
            raise ValueError("The constrained border must be square.")
        if not np.all(np.isfinite(border)):
            raise np.linalg.LinAlgError("The constrained border contains non-finite values.")
        self.matrix = 0.5 * (border + border.T)
        self.size = self.matrix.shape[0]
        scale = max(float(np.linalg.norm(self.matrix, ord=np.inf)), 1.0)
        self.threshold = max(
            np.finfo(np.float64).eps ** (2 / 3) * scale,
            np.finfo(np.float64).eps * max(self.size, 1) * scale * 10.0,
        )
        self.used_fallback = False
        self.fallback_reason: str | None = None
        self._triangular: NDArray | None = None
        self._permutation: NDArray | None = None
        self._inverse_pivots: tuple[tuple[slice, NDArray], ...] = ()
        self._svd: tuple[NDArray, NDArray, NDArray] | None = None

        try:
            lu, diagonal, permutation = scipy.linalg.ldl(
                self.matrix,
                lower=True,
                hermitian=True,
                check_finite=False,
            )
            triangular = lu[permutation, :]
            residual = np.linalg.norm(lu @ diagonal @ lu.T - self.matrix, ord=np.inf) / scale
            if not np.isfinite(residual) or residual > 1e-9:
                raise np.linalg.LinAlgError(
                    f"LDL reconstruction residual {residual:.3g} exceeds 1e-9"
                )
            pivots, inverse_pivots = self._analyze_ldl_pivots(diagonal)
            self._triangular = triangular
            self._permutation = np.asarray(permutation, dtype=np.intp)
            self._inverse_pivots = inverse_pivots
            self._set_spectrum(pivots)
            if self.zero_count == 0:
                probe = np.zeros((self.size, min(self.size, 2)))
                for column in range(probe.shape[1]):
                    probe[column * (self.size - 1) // max(probe.shape[1] - 1, 1), column] = 1.0
                if probe.size:
                    solution = self.solve(probe)
                    solve_residual = np.linalg.norm(
                        self.matrix @ solution - probe, ord=np.inf
                    ) / max(np.linalg.norm(probe, ord=np.inf), 1.0)
                    if not np.isfinite(solve_residual) or solve_residual > 1e-8:
                        raise np.linalg.LinAlgError(
                            f"LDL solve residual {solve_residual:.3g} exceeds 1e-8"
                        )
        except (np.linalg.LinAlgError, ValueError) as error:
            self.used_fallback = True
            self.fallback_reason = f"constrained-border LDL fallback: {error}"
            self._triangular = None
            self._permutation = None
            self._inverse_pivots = ()
            left, singular_values, right = np.linalg.svd(self.matrix, full_matrices=False)
            inverse_values = np.zeros_like(singular_values)
            active = singular_values > self.threshold
            np.divide(1.0, singular_values, out=inverse_values, where=active)
            self._svd = (left, inverse_values, right)
            self._set_spectrum(np.linalg.eigvalsh(self.matrix))

    def _analyze_ldl_pivots(
        self,
        diagonal: NDArray,
    ) -> tuple[NDArray, tuple[tuple[slice, NDArray], ...]]:
        eigenvalues: list[float] = []
        inverse_blocks: list[tuple[slice, NDArray]] = []
        index = 0
        while index < self.size:
            if index + 1 < self.size and diagonal[index, index + 1] != 0.0:
                block_slice = slice(index, index + 2)
                block = diagonal[block_slice, block_slice]
                values = np.linalg.eigvalsh(block)
                eigenvalues.extend(float(value) for value in values)
                if np.all(np.abs(values) > self.threshold):
                    inverse_blocks.append((block_slice, np.linalg.inv(block)))
                index += 2
            else:
                block_slice = slice(index, index + 1)
                value = float(diagonal[index, index])
                eigenvalues.append(value)
                if abs(value) > self.threshold:
                    inverse_blocks.append(
                        (block_slice, np.array([[1.0 / value]], dtype=np.float64))
                    )
                index += 1
        return np.asarray(eigenvalues), tuple(inverse_blocks)

    def _set_spectrum(self, eigenvalues: NDArray) -> None:
        values = np.asarray(eigenvalues, dtype=np.float64)
        positive = values > self.threshold
        negative = values < -self.threshold
        self.positive_count = int(np.count_nonzero(positive))
        self.negative_count = int(np.count_nonzero(negative))
        self.zero_count = int(values.size - self.positive_count - self.negative_count)
        active = np.abs(values) > self.threshold
        self.logabsdet = float(np.sum(np.log(np.abs(values[active])))) if np.any(active) else 0.0
        absolute = np.abs(values[active])
        self.condition_estimate = (
            float(absolute.max() / absolute.min()) if absolute.size else float("inf")
        )

    def solve(self, rhs: NDArray) -> NDArray:
        values = np.asarray(rhs, dtype=np.float64)
        vector_rhs = values.ndim == 1
        if vector_rhs:
            values = values[:, None]
        if values.ndim != 2 or values.shape[0] != self.size:
            raise ValueError(f"border rhs must have shape ({self.size},) or ({self.size}, m)")
        if self.zero_count:
            raise np.linalg.LinAlgError("The constrained border is singular.")
        if self._triangular is not None and self._permutation is not None:
            permuted_rhs = values[self._permutation]
            forward = scipy.linalg.solve_triangular(
                self._triangular,
                permuted_rhs,
                lower=True,
                unit_diagonal=True,
                check_finite=False,
            )
            middle = np.zeros_like(forward)
            for block_slice, inverse in self._inverse_pivots:
                middle[block_slice] = inverse @ forward[block_slice]
            permuted_solution = scipy.linalg.solve_triangular(
                self._triangular.T,
                middle,
                lower=False,
                unit_diagonal=True,
                check_finite=False,
            )
            solution = np.empty_like(permuted_solution)
            solution[self._permutation] = permuted_solution
        elif self._svd is not None:
            left, inverse_values, right = self._svd
            solution = (right.T * inverse_values) @ (left.T @ values)
        else:  # pragma: no cover - guarded by construction
            raise RuntimeError("The constrained border has no usable factor.")
        return solution[:, 0] if vector_rhs else solution


class SumToZeroBlockFactor:
    """Factor an all-level PSD block system under an exact sum-to-zero constraint."""

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
        level_labels: tuple[Any, ...] | None = None,
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
            raise ValueError("C must have shape (K, k, q).")
        self.n_levels, self.block_size, q = self.C.shape
        if self.n_levels < 2:
            raise ValueError("Sum-to-zero factors require at least two levels.")
        if self.A.shape != (q, q):
            raise ValueError(f"A shape {self.A.shape} does not match ({q}, {q}).")
        if self.D.shape != (self.n_levels, self.block_size, self.block_size):
            raise ValueError("D must have shape (K, k, k).")
        if self.small_indices.shape != (q,):
            raise ValueError("small_indices width does not match A.")
        if self.structured_indices.shape != (self.n_levels - 1, self.block_size):
            raise ValueError("structured_indices must have shape (K - 1, k).")
        if not np.all(np.isfinite(self.A)) or not np.all(np.isfinite(self.C)):
            raise np.linalg.LinAlgError(
                f"Structured SZ term {term_name!r} has non-finite ordinary or cross blocks."
            )
        if not np.allclose(self.A, self.A.T, rtol=0.0, atol=1e-13):
            raise ValueError("A must be symmetric.")
        if not np.allclose(self.D, self.D.transpose(0, 2, 1), rtol=0.0, atol=1e-13):
            raise ValueError("Every local D block must be symmetric.")
        all_indices = np.concatenate((self.small_indices, self.structured_indices.ravel()))
        if len(np.unique(all_indices)) != len(all_indices):
            raise ValueError("small_indices and structured_indices must be disjoint.")
        if not np.array_equal(np.sort(all_indices), np.arange(len(all_indices))):
            raise ValueError("Structured index partitions must cover every coefficient once.")
        self.shape = (len(all_indices), len(all_indices))
        if level_labels is None:
            self.level_labels = tuple(range(self.n_levels))
        else:
            self.level_labels = tuple(level_labels)
            if len(self.level_labels) != self.n_levels:
                raise ValueError("level_labels length must equal K.")

        self._locals, self.minimum_local_eigenvalue = _decompose_local_psd_batch(
            self.D,
            term_name=term_name,
            level_labels=self.level_labels,
        )
        self._pinv = np.stack([local.pinv for local in self._locals])
        self._positive_rank = sum(local.rank for local in self._locals)
        self.deficient_levels = tuple(
            label
            for label, local in zip(self.level_labels, self._locals, strict=True)
            if local.rank < self.block_size
        )
        self.minimum_local_diagonal = self.minimum_local_eigenvalue

        null_widths = [local.null.shape[1] for local in self._locals]
        self._null_width = int(sum(null_widths))
        gamma_slices: list[slice] = []
        offset = 0
        for width in null_widths:
            gamma_slices.append(slice(offset, offset + width))
            offset += width
        self._gamma_slices = tuple(gamma_slices)

        Q = np.array(self.A, copy=True)
        R = np.zeros((self.block_size, q))
        M = np.zeros((self.block_size, self.block_size))
        E = np.zeros((q, self._null_width))
        N = np.zeros((self.block_size, self._null_width))
        self._pinv_cross = np.empty_like(self.C)
        for level, (local, gamma_slice) in enumerate(
            zip(self._locals, self._gamma_slices, strict=True)
        ):
            pinv_cross = local.pinv @ self.C[level]
            self._pinv_cross[level] = pinv_cross
            Q -= self.C[level].T @ pinv_cross
            R += pinv_cross
            M += local.pinv
            E[:, gamma_slice] = self.C[level].T @ local.null
            N[:, gamma_slice] = local.null
        Q = 0.5 * (Q + Q.T)
        border = np.block(
            [
                [Q, E, -R.T],
                [
                    E.T,
                    np.zeros((self._null_width, self._null_width)),
                    N.T,
                ],
                [-R, N, -M],
            ]
        )
        self._border = border
        constraint_transform = _constraint_equilibration(M)
        border_transform = np.eye(border.shape[0], dtype=np.float64)
        border_transform[-self.block_size :, -self.block_size :] = constraint_transform
        scaled_border = border_transform.T @ border @ border_transform
        self._border_transform = border_transform
        self._border_factor = _SymmetricBorderFactor(scaled_border)
        expected_positive = q + self._null_width
        expected_negative = self.block_size
        if (
            self._border_factor.positive_count != expected_positive
            or self._border_factor.negative_count != expected_negative
            or self._border_factor.zero_count
        ):
            raise SumToZeroIdentifiabilityError(
                f"Structured SZ term {term_name!r} is globally unidentifiable after "
                f"enforcing sum-to-zero; deficient fitted levels={self.deficient_levels!r}. "
                "Use basis='fs', reduce k, or provide more numeric support."
            )

        local_logdet = sum(
            float(np.sum(np.log(local.positive_eigenvalues)))
            for local in self._locals
            if local.positive_eigenvalues.size
        )
        transform_logdet = np.linalg.slogdet(border_transform)[1]
        self._logdet = local_logdet + self._border_factor.logabsdet - 2.0 * float(transform_logdet)
        self.rank = self.shape[0]
        self.rank_truncated = False
        self.public_positive_definite = True
        self.used_dense_fallback = self._border_factor.used_fallback
        self.fallback_reason = self._border_factor.fallback_reason
        self.schur_condition_estimate = self._border_factor.condition_estimate
        self._small_position = np.full(self.shape[0], -1, dtype=np.intp)
        self._small_position[self.small_indices] = np.arange(q)
        self._structured_position = np.full(self.shape[0], -1, dtype=np.intp)
        self._structured_position[self.structured_indices.ravel()] = np.arange(
            (self.n_levels - 1) * self.block_size
        )
        self._border_inverse_cache: NDArray | None = None
        self._inverse_bdlr_cache: _BlockDiagonalLowRank | None = None
        self._raw_border_basis = self._build_raw_border_basis()
        self._public_border_basis = np.zeros((self.shape[0], border.shape[0]))
        self._public_border_basis[self.small_indices, :q] = np.eye(q)
        self._public_border_basis[self.structured_indices] = self._raw_border_basis[:-1]

    def _build_raw_border_basis(self) -> NDArray:
        q = len(self.small_indices)
        basis = np.zeros(
            (
                self.n_levels,
                self.block_size,
                q + self._null_width + self.block_size,
            )
        )
        basis[:, :, :q] = -self._pinv_cross
        for level, (local, gamma_slice) in enumerate(
            zip(self._locals, self._gamma_slices, strict=True)
        ):
            basis[level, :, q + gamma_slice.start : q + gamma_slice.stop] = local.null
        basis[:, :, q + self._null_width :] = -self._pinv
        return basis

    def _border_inverse(self) -> NDArray:
        if self._border_inverse_cache is None:
            self._border_inverse_cache = self._solve_border(np.eye(self._border.shape[0]))
            self._border_inverse_cache = 0.5 * (
                self._border_inverse_cache + self._border_inverse_cache.T
            )
        return self._border_inverse_cache

    def _solve_border(self, rhs: NDArray) -> NDArray:
        transformed_rhs = self._border_transform.T @ np.asarray(rhs, dtype=np.float64)
        scaled_solution = self._border_factor.solve(transformed_rhs)
        return self._border_transform @ scaled_solution

    def solve(self, rhs: NDArray) -> NDArray:
        """Solve the public ``K - 1`` coordinate system without materializing it."""
        values = np.asarray(rhs, dtype=np.float64)
        vector_rhs = values.ndim == 1
        if vector_rhs:
            values = values[:, None]
        if values.ndim != 2 or values.shape[0] != self.shape[0]:
            raise ValueError(
                f"rhs must have shape ({self.shape[0]},) or ({self.shape[0]}, m), "
                f"got {np.asarray(rhs).shape}."
            )
        q = len(self.small_indices)
        raw_rhs = np.zeros((self.n_levels, self.block_size, values.shape[1]))
        raw_rhs[:-1] = values[self.structured_indices]
        pinv_rhs = np.einsum("kij,kjm->kim", self._pinv, raw_rhs, optimize=True)
        border_small = values[self.small_indices] - np.einsum(
            "kiq,kim->qm",
            self.C,
            pinv_rhs,
            optimize=True,
        )
        border_null = np.empty((self._null_width, values.shape[1]))
        for level, (local, gamma_slice) in enumerate(
            zip(self._locals, self._gamma_slices, strict=True)
        ):
            border_null[gamma_slice] = local.null.T @ raw_rhs[level]
        border_multiplier = -np.sum(pinv_rhs, axis=0)
        border_rhs = np.vstack((border_small, border_null, border_multiplier))
        border_solution = self._solve_border(border_rhs)
        small_solution = border_solution[:q]
        gamma = border_solution[q : q + self._null_width]
        multiplier = border_solution[q + self._null_width :]
        raw_solution = (
            pinv_rhs
            - np.einsum(
                "kiq,qm->kim",
                self._pinv_cross,
                small_solution,
                optimize=True,
            )
            - np.einsum("kij,jm->kim", self._pinv, multiplier, optimize=True)
        )
        for level, (local, gamma_slice) in enumerate(
            zip(self._locals, self._gamma_slices, strict=True)
        ):
            raw_solution[level] += local.null @ gamma[gamma_slice]
        solution = np.empty_like(values)
        solution[self.small_indices] = small_solution
        solution[self.structured_indices] = raw_solution[:-1]
        return solution[:, 0] if vector_rhs else solution

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

    def _selected_base_covariance(self, selected: NDArray) -> NDArray:
        base = np.zeros((len(selected), len(selected)))
        structured = self._structured_position[selected]
        for row, left_position in enumerate(structured):
            if left_position < 0:
                continue
            left_level, left_coordinate = divmod(left_position, self.block_size)
            for column, right_position in enumerate(structured):
                if right_position < 0:
                    continue
                right_level, right_coordinate = divmod(right_position, self.block_size)
                if left_level == right_level:
                    base[row, column] = self._pinv[
                        left_level,
                        left_coordinate,
                        right_coordinate,
                    ]
        return base

    def selected_inverse_block(self, indices: NDArray) -> NDArray:
        selected = self._validate_selected_indices(indices)
        structured_count = int(np.count_nonzero(self._structured_position[selected] >= 0))
        if structured_count > self.max_structured_inverse_block:
            raise ValueError(
                f"Refusing to materialize a {structured_count} x {structured_count} "
                f"inverse block for structured term {self.term_name!r}; "
                "request its diagonal instead."
            )
        basis = self._public_border_basis[selected]
        return self._selected_base_covariance(selected) + basis @ self._border_inverse() @ basis.T

    def selected_inverse_diagonal(self, indices: NDArray) -> NDArray:
        selected = self._validate_selected_indices(indices)
        diagonal = np.zeros(len(selected))
        structured = self._structured_position[selected]
        structured_mask = structured >= 0
        if np.any(structured_mask):
            levels, coordinates = np.divmod(structured[structured_mask], self.block_size)
            diagonal[structured_mask] = self._pinv[levels, coordinates, coordinates]
        basis = self._public_border_basis[selected]
        diagonal += np.sum((basis @ self._border_inverse()) * basis, axis=1)
        return diagonal

    def raw_level_inverse_block(self, level: int) -> NDArray:
        if isinstance(level, bool) or not isinstance(level, (int, np.integer)):
            raise TypeError("level must be an integer index.")
        level = int(level)
        if level < 0 or level >= self.n_levels:
            raise IndexError("raw level index is outside the fitted level range.")
        basis = self._raw_border_basis[level]
        covariance = self._pinv[level] + basis @ self._border_inverse() @ basis.T
        return 0.5 * (covariance + covariance.T)

    def _inverse_bdlr(self) -> _BlockDiagonalLowRank:
        cached = self._inverse_bdlr_cache
        if cached is None:
            cached = _BlockDiagonalLowRank(
                blocks=self._pinv[:-1],
                structured_indices=self.structured_indices,
                basis=self._public_border_basis,
                core=self._border_inverse(),
                shape=self.shape,
            )
            self._inverse_bdlr_cache = cached
        return cached

    def _penalty_operator(
        self,
        component: PenaltyComponent,
        scale: float,
    ) -> CompactSymmetricOperator:
        indices = _component_indices(component, self.shape[0])
        local_small = self._small_position[indices]
        local_structured = self._structured_position[indices]
        A = np.zeros_like(self.A)
        C = np.zeros_like(self.C)
        if component.penalty_kind == "sum_to_zero":
            if not np.all(local_structured >= 0):
                raise ValueError("Sum-to-zero penalty must lie in the structured block.")
            if (
                component.repeat_count != self.n_levels
                or component.block_width != self.block_size
                or not np.array_equal(
                    indices.reshape(self.n_levels - 1, self.block_size),
                    self.structured_indices,
                )
            ):
                raise ValueError("Sum-to-zero penalty geometry does not match the block factor.")
            omega = np.asarray(component.omega_ssp, dtype=np.float64)
            if omega.shape != (self.block_size, self.block_size):
                raise ValueError("Sum-to-zero penalty local matrix has the wrong shape.")
            return SumToZeroBlockOperator(
                A=A,
                C=C,
                D=np.broadcast_to(scale * omega, self.D.shape),
                small_indices=self.small_indices,
                structured_indices=self.structured_indices,
            )
        if component.penalty_kind == "identity":
            if np.all(local_small >= 0):
                A[local_small, local_small] = scale
                return SumToZeroBlockOperator(
                    A=A,
                    C=C,
                    D=np.zeros_like(self.D),
                    small_indices=self.small_indices,
                    structured_indices=self.structured_indices,
                )
            if np.all(local_structured >= 0):
                public_D = np.zeros((self.n_levels - 1, self.block_size, self.block_size))
                for position in local_structured:
                    level, coordinate = divmod(position, self.block_size)
                    public_D[level, coordinate, coordinate] = scale
                return BlockSymmetricOperator(
                    A=A,
                    C=np.zeros((self.n_levels - 1, self.block_size, len(self.small_indices))),
                    D=public_D,
                    small_indices=self.small_indices,
                    structured_indices=self.structured_indices,
                )
            raise ValueError("Identity penalty crosses structured partitions.")
        if not np.all(local_small >= 0):
            raise ValueError("Dense penalties must lie in the factor's small block.")
        A[np.ix_(local_small, local_small)] = scale * _component_omega(
            component,
            self.shape[0],
        )
        return SumToZeroBlockOperator(
            A=A,
            C=C,
            D=np.zeros_like(self.D),
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
            _multiply_symmetric_bdlr_coalesced(
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
            _multiply_symmetric_bdlr_coalesced(
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
            _multiply_symmetric_bdlr_coalesced(
                inverse,
                _operator_bdlr(left, self.structured_indices),
            ),
            _multiply_symmetric_bdlr_coalesced(
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


class ProfiledSumToZeroBlockFactor:
    """Profiled slope view of an augmented sum-to-zero factor."""

    backend = "structured"

    def __init__(
        self,
        *,
        augmented_factor: SumToZeroBlockFactor,
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
        if not len(augmented_factor.small_indices) or augmented_factor.small_indices[0] != 0:
            raise ValueError("The augmented intercept must be the first dense-small coefficient.")
        self.shape = (len(self.xtw), len(self.xtw))
        self.mean_x = self.xtw / self.sum_w
        self.small_indices = augmented_factor.small_indices[1:] - 1
        self.structured_indices = augmented_factor.structured_indices - 1
        self.n_levels = augmented_factor.n_levels
        self.block_size = augmented_factor.block_size
        self.rank = max(augmented_factor.rank - 1, 0)
        self.rank_truncated = self.rank < self.shape[0]
        self.used_dense_fallback = augmented_factor.used_dense_fallback
        self.fallback_reason = augmented_factor.fallback_reason
        self.schur_condition_estimate = augmented_factor.schur_condition_estimate
        self.minimum_local_eigenvalue = augmented_factor.minimum_local_eigenvalue
        self.minimum_local_diagonal = augmented_factor.minimum_local_diagonal
        self.dominant_group_name = augmented_factor.dominant_group_name
        self.deficient_levels = augmented_factor.deficient_levels
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

    def raw_level_inverse_block(self, level: int) -> NDArray:
        return self.augmented_factor.raw_level_inverse_block(level)

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
        if cached is None:
            augmented = self.augmented_factor._inverse_bdlr()
            cached = _BlockDiagonalLowRank(
                blocks=augmented.blocks,
                structured_indices=self.structured_indices,
                basis=augmented.basis[1:],
                core=augmented.core,
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
            _multiply_symmetric_bdlr_coalesced(
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
            _multiply_symmetric_bdlr_coalesced(
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
            _multiply_symmetric_bdlr_coalesced(
                inverse,
                _operator_bdlr(left, self.structured_indices),
            ),
            _multiply_symmetric_bdlr_coalesced(
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
        if isinstance(penalty, SumToZeroBlockOperator):
            slope_penalty: CompactSymmetricOperator = SumToZeroBlockOperator(
                A=penalty.A[1:, 1:],
                C=penalty.C[:, :, 1:],
                D=penalty.D,
                small_indices=self.small_indices,
                structured_indices=self.structured_indices,
            )
        elif isinstance(penalty, BlockSymmetricOperator):
            slope_penalty = BlockSymmetricOperator(
                A=penalty.A[1:, 1:],
                C=penalty.C[:, :, 1:],
                D=penalty.D,
                small_indices=self.small_indices,
                structured_indices=self.structured_indices,
            )
        else:  # pragma: no cover - _penalty_operator contract
            raise TypeError("Profiled SZ penalties must use block-compatible operators.")
        return self.operator_cross_trace(slope_penalty, operator)


__all__ = [
    "ProfiledSumToZeroBlockFactor",
    "SumToZeroBlockFactor",
    "SumToZeroIdentifiabilityError",
]
