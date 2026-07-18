"""Versioned numerical-rank policy and retained-subspace operations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import scipy.linalg
from numpy.typing import NDArray


@dataclass(frozen=True)
class RankPolicy:
    version: int
    factor_rcond: float
    gram_rcond: float
    certification_band: float
    warning_condition: float
    severe_condition: float


_EPS = np.finfo(float).eps
SHARED_RANK_POLICY = RankPolicy(
    version=1,
    factor_rcond=float(np.sqrt(_EPS)),
    gram_rcond=float(_EPS),
    certification_band=32.0,
    warning_condition=float(1.0 / np.sqrt(_EPS)),
    severe_condition=float(1.0 / _EPS),
)


def diagonal_of_square(matrix: NDArray) -> NDArray:
    """Return ``diag(matrix @ matrix)`` with an O(p²) contraction."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    return np.einsum("ij,ji->i", matrix, matrix, optimize=True)


def streamed_weighted_factor(
    chunks: Iterable[tuple[int, int, NDArray]],
    weights: NDArray,
    *,
    center: NDArray | None = None,
) -> NDArray:
    """Build a compact QR factor from bounded weighted row chunks."""
    weights = np.asarray(weights, dtype=float)
    factor: NDArray | None = None
    width = 0 if center is None else len(center)
    for start, stop, values in chunks:
        block = np.asarray(values, dtype=float)
        width = block.shape[1]
        if center is not None:
            block = block - center
        block = np.sqrt(weights[start:stop])[:, None] * block
        stacked = block if factor is None else np.vstack((factor, block))
        factor = np.linalg.qr(stacked, mode="r")
    return np.empty((0, width)) if factor is None else np.asarray(factor)


def streamed_weighted_factor_rhs(
    chunks: Iterable[tuple[int, int, NDArray]],
    weights: NDArray,
    response: NDArray,
    *,
    center: NDArray | None = None,
) -> tuple[NDArray, NDArray]:
    """Build a compact weighted QR factor and its consistently transformed RHS.

    Appending the response to every bounded design chunk preserves ``Q.T @ b``
    without retaining either the observation matrix or the observation-length
    orthogonal factor.  The returned factor has at most ``p + 1`` rows.
    """
    weights = np.asarray(weights, dtype=float)
    response = np.asarray(response, dtype=float)
    if weights.ndim != 1 or response.shape != weights.shape:
        raise ValueError("weights and response must be matching vectors")
    joint_factor: NDArray | None = None
    width = 0 if center is None else len(center)
    for start, stop, values in chunks:
        block = np.asarray(values, dtype=float)
        width = block.shape[1]
        if center is not None:
            block = block - center
        sqrt_weights = np.sqrt(weights[start:stop])
        joint_block = np.column_stack(
            (sqrt_weights[:, None] * block, sqrt_weights * response[start:stop])
        )
        stacked = joint_block if joint_factor is None else np.vstack((joint_factor, joint_block))
        joint_factor = np.linalg.qr(stacked, mode="r")
    if joint_factor is None:
        return np.empty((0, width)), np.empty(0)
    return np.asarray(joint_factor[:, :width]), np.asarray(joint_factor[:, width])


def needs_factor_certification(
    decomposition: RankDecomposition,
    *,
    policy: RankPolicy = SHARED_RANK_POLICY,
) -> bool:
    """Whether Gram geometry lies inside a band requiring factor certification.

    A certificate governs the retained subspace as well as the integer rank.
    Normal equations can erase a factor-scale direction at the numerical
    boundary, or retain a different direction while reporting the same rank.
    """
    if decomposition.method == "qr_svd":
        # A factor decomposition is already the authoritative certificate;
        # never stream and factor the same rows again merely because the
        # factor policy itself truncated a nonzero singular value.
        return False
    certification_condition = policy.warning_condition / np.sqrt(policy.certification_band)
    return bool(
        decomposition.width > 0
        and (
            (
                decomposition.rank == decomposition.width
                and decomposition.pre_truncation_condition >= certification_condition
            )
            or (decomposition.rank < decomposition.width and decomposition.resolution_limited)
        )
    )


def _freeze(values: NDArray, *, dtype=float) -> NDArray:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class RankDecomposition:
    policy_version: int
    method: Literal["empty", "cholesky", "pivoted_cholesky", "gram_eigh", "qr_svd"]
    column_scale: NDArray
    active_columns: NDArray
    rank: int
    pre_truncation_condition: float
    cutoff: float
    rank_truncated: bool
    used_svd_fallback: bool
    resolution_limited: bool
    log_pdet: float
    cholesky_factor: NDArray | None = None
    pivots: NDArray | None = None
    solution_basis: NDArray | None = None
    parameter_null_basis: NDArray | None = None
    estimable_functional_basis: NDArray | None = None
    structural_aliases: NDArray | None = None
    retained_values: NDArray | None = None
    factor_rhs_left_basis: NDArray | None = None
    factor_rhs_triangular: NDArray | None = None

    @property
    def width(self) -> int:
        return int(self.column_scale.size)

    def solve(self, rhs: NDArray) -> NDArray:
        rhs = np.asarray(rhs, dtype=float)
        if rhs.shape != (self.width,):
            raise ValueError("rhs width does not match decomposition")
        if self.rank == 0:
            return np.zeros_like(rhs)
        if self.cholesky_factor is not None:
            active_rhs = rhs[self.active_columns] / self.column_scale[self.active_columns]
            active_solution = scipy.linalg.cho_solve(
                (self.cholesky_factor, True), active_rhs, check_finite=False
            )
            result = np.zeros(self.width)
            result[self.active_columns] = active_solution / self.column_scale[self.active_columns]
            return result
        if self.solution_basis is None or self.retained_values is None:
            raise RuntimeError("retained spectral basis is unavailable")
        return self.solution_basis @ ((self.solution_basis.T @ rhs) / self.retained_values)

    def solve_factor_rhs(self, transformed_rhs: NDArray) -> NDArray:
        """Solve from a response transformed with the certified factor's QR.

        This path avoids re-forming normal equations at the factor-rank
        boundary.  It is available only when ``decompose_factor`` was asked to
        retain the bounded factor solve.
        """
        if self.factor_rhs_left_basis is None:
            raise RuntimeError("factor-RHS solve was not retained")
        transformed_rhs = np.asarray(transformed_rhs, dtype=float)
        if transformed_rhs.shape != (self.factor_rhs_left_basis.shape[0],):
            raise ValueError("transformed RHS length does not match the certified factor")
        if self.rank == 0:
            return np.zeros(self.width)
        projected_rhs = self.factor_rhs_left_basis.T @ transformed_rhs
        if self.factor_rhs_triangular is not None:
            active_solution = scipy.linalg.solve_triangular(
                self.factor_rhs_triangular,
                projected_rhs,
                lower=False,
                check_finite=False,
            )
            result = np.zeros(self.width)
            result[self.active_columns] = active_solution
            return result
        if self.solution_basis is None:
            raise RuntimeError("retained factor solution basis is unavailable")
        return self.solution_basis @ projected_rhs

    def pseudo_inverse(self) -> NDArray:
        if self.rank == 0:
            return np.zeros((self.width, self.width))
        if self.cholesky_factor is not None:
            inverse_equilibrated = scipy.linalg.cho_solve(
                (self.cholesky_factor, True),
                np.eye(len(self.active_columns)),
                check_finite=False,
            )
            inverse = np.zeros((self.width, self.width))
            scale = self.column_scale[self.active_columns]
            inverse[np.ix_(self.active_columns, self.active_columns)] = (
                inverse_equilibrated / np.outer(scale, scale)
            )
            return 0.5 * (inverse + inverse.T)
        if self.solution_basis is None or self.retained_values is None:
            raise RuntimeError("retained spectral basis is unavailable")
        inverse = (self.solution_basis / self.retained_values) @ self.solution_basis.T
        return 0.5 * (inverse + inverse.T)

    def retained_parameter_basis(self) -> NDArray:
        if self.solution_basis is not None:
            return self.solution_basis.copy()
        basis = np.zeros((self.width, self.rank))
        if self.rank:
            basis[self.active_columns, :] = np.diag(1.0 / self.column_scale[self.active_columns])
        return basis

    def null_basis(self) -> NDArray:
        if self.parameter_null_basis is None:
            return np.zeros((self.width, 0))
        return self.parameter_null_basis.copy()

    def is_estimable(self, contrast: NDArray) -> bool:
        contrast = np.asarray(contrast, dtype=float)
        if contrast.shape != (self.width,):
            raise ValueError("contrast width does not match decomposition")
        scaled_columns = self.column_scale > 0.0
        contrast_norm = float(np.linalg.norm(contrast))
        structural_tolerance = SHARED_RANK_POLICY.factor_rcond * max(
            contrast_norm,
            np.finfo(float).tiny,
        )
        if np.linalg.norm(contrast[~scaled_columns]) > structural_tolerance:
            return False
        null = self.null_basis()
        if null.shape[1] == 0:
            return True

        # Test orthogonality in the equilibrated dual coordinates used by the
        # rank decision.  Comparing ``contrast @ parameter_null_basis`` against
        # an unscaled absolute tolerance makes exact aliases appear estimable
        # when one design column is multiplied by a large constant.
        scaled_contrast = contrast[scaled_columns] / self.column_scale[scaled_columns]
        equilibrated_null = null[scaled_columns, :] * self.column_scale[scaled_columns, None]
        null_norms = np.linalg.norm(equilibrated_null, axis=0)
        retained_null = null_norms > np.finfo(float).eps
        if not np.any(retained_null):
            return True
        normalized_null = equilibrated_null[:, retained_null] / null_norms[retained_null]
        projection = scaled_contrast @ normalized_null
        tolerance = SHARED_RANK_POLICY.factor_rcond * max(
            float(np.linalg.norm(scaled_contrast)),
            np.finfo(float).tiny,
        )
        return bool(np.linalg.norm(projection) <= tolerance)

    def coefficient_estimable(self) -> NDArray:
        """Return all unit-coordinate estimability decisions in one projection."""
        scaled_columns = self.column_scale > 0.0
        result = np.zeros(self.width, dtype=bool)
        null = self.null_basis()
        if null.shape[1] == 0:
            result[scaled_columns] = True
            return result

        equilibrated_null = null[scaled_columns, :] * self.column_scale[scaled_columns, None]
        null_norms = np.linalg.norm(equilibrated_null, axis=0)
        retained_null = null_norms > np.finfo(float).eps
        if not np.any(retained_null):
            result[scaled_columns] = True
            return result
        normalized_null = equilibrated_null[:, retained_null] / null_norms[retained_null]
        result[scaled_columns] = (
            np.linalg.norm(normalized_null, axis=1) <= SHARED_RANK_POLICY.factor_rcond
        )
        return result


@dataclass(frozen=True)
class RankInfo:
    """Compact fitted-subspace metadata in solver coefficient coordinates."""

    policy_version: int
    coordinate_space: Literal["solver"]
    selected_columns: NDArray
    selected_group_names: Sequence[str]
    sum_w: float
    mean_x: NDArray
    intercept_edf: float
    data: RankDecomposition
    augmented: RankDecomposition
    coefficient: RankDecomposition
    feature_edf: NDArray
    group_edf: Mapping[str, float]
    objective_loss: float | None

    @property
    def total_edf(self) -> float:
        return self.intercept_edf + float(np.sum(self.feature_edf))

    def solve(self, rhs: NDArray) -> NDArray:
        rhs = np.asarray(rhs, dtype=float)
        if rhs.shape != self.mean_x.shape:
            raise ValueError("rhs width does not match fitted coefficient space")
        result = np.zeros_like(rhs)
        result[self.selected_columns] = self.augmented.solve(rhs[self.selected_columns])
        return result

    def pseudo_inverse(self) -> NDArray:
        width = len(self.mean_x)
        result = np.zeros((width, width))
        result[np.ix_(self.selected_columns, self.selected_columns)] = (
            self.augmented.pseudo_inverse()
        )
        return result

    def is_estimable(self, contrast: NDArray) -> bool:
        contrast = np.asarray(contrast, dtype=float)
        if contrast.shape != self.mean_x.shape:
            raise ValueError("contrast width does not match fitted coefficient space")
        unselected = np.ones(len(contrast), dtype=bool)
        unselected[self.selected_columns] = False
        tolerance = SHARED_RANK_POLICY.factor_rcond * max(1.0, float(np.linalg.norm(contrast)))
        if np.linalg.norm(contrast[unselected]) > tolerance:
            return False
        return self.data.is_estimable(contrast[self.selected_columns])

    def coefficient_estimable(self) -> NDArray:
        result = np.zeros(len(self.mean_x), dtype=bool)
        result[self.selected_columns] = self.data.coefficient_estimable()
        return result


def _equilibrate_gram(
    matrix: NDArray, *, allow_indefinite: bool = False
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("matrix must be square")
    if not np.all(np.isfinite(values)):
        raise ValueError("matrix must be finite")
    symmetric = 0.5 * (values + values.T)
    diagonal = np.diag(symmetric)
    scale_reference = max(float(np.max(np.abs(diagonal), initial=0.0)), 1.0)
    if not allow_indefinite and np.any(diagonal < -100.0 * _EPS * scale_reference):
        raise ValueError("matrix has a materially negative diagonal")
    if allow_indefinite:
        row_scale = np.max(np.abs(symmetric), axis=1, initial=0.0)
        diagonal_scale = np.maximum(np.abs(diagonal), _EPS * row_scale)
    else:
        diagonal_scale = np.maximum(diagonal, 0.0)
    active_columns = np.flatnonzero(diagonal_scale > 0.0)
    column_scale = np.zeros(len(diagonal))
    column_scale[active_columns] = np.sqrt(diagonal_scale[active_columns])
    if active_columns.size:
        active_scale = column_scale[active_columns]
        equilibrated = symmetric[np.ix_(active_columns, active_columns)] / np.outer(
            active_scale, active_scale
        )
        equilibrated = 0.5 * (equilibrated + equilibrated.T)
    else:
        equilibrated = np.zeros((0, 0))
    return equilibrated, column_scale, active_columns, symmetric


def _null_basis(
    width: int,
    active_columns: NDArray,
    active_scale: NDArray,
    discarded_vectors: NDArray,
) -> NDArray:
    pieces: list[NDArray] = []
    if discarded_vectors.shape[1]:
        discarded = np.zeros((width, discarded_vectors.shape[1]))
        discarded[active_columns, :] = discarded_vectors / active_scale[:, None]
        pieces.append(discarded)
    inactive = np.setdiff1d(np.arange(width), active_columns, assume_unique=True)
    if inactive.size:
        inactive_basis = np.zeros((width, inactive.size))
        inactive_basis[inactive, np.arange(inactive.size)] = 1.0
        pieces.append(inactive_basis)
    return np.column_stack(pieces) if pieces else np.zeros((width, 0))


def _scaled_subspace_logdet(coordinates: NDArray) -> float:
    """Return ``log(det(coordinates.T @ coordinates))`` across extreme row scales."""
    width = coordinates.shape[1]
    if width == 0:
        return 0.0

    # Ordinary QR/SVD only provides absolute accuracy.  DGEJSV's 'F' mode
    # applies full row and column pivoting so diagonal scaling cannot erase a
    # genuine retained direction.  Ask for the unrestricted singular-value
    # range because rank has already been decided in equilibrated coordinates.
    singular_values, _, _, scaling, _, info = scipy.linalg.lapack.dgejsv(
        np.asfortranarray(coordinates),
        joba=2,  # 'F': full-pivoting, high-relative-accuracy preprocessing
        jobu=3,  # 'N': singular values only
        jobv=3,  # 'N': singular values only
        jobr=0,  # 'N': do not truncate the requested singular-value range
    )
    if info != 0:
        raise np.linalg.LinAlgError(f"high-accuracy retained SVD failed with info={info}")
    if np.any(singular_values <= 0.0) or np.any(scaling[:2] <= 0.0):
        raise ValueError("retained coordinate basis is not full rank")
    log_scale = float(np.log(scaling[0]) - np.log(scaling[1]))
    return 2.0 * (float(np.sum(np.log(singular_values))) + width * log_scale)


def _retained_log_pdet(
    active_scale: NDArray,
    retained_vectors: NDArray,
    discarded_vectors: NDArray,
    retained_values: NDArray,
) -> float:
    """Return the retained pseudo-logdet without forming a coordinate Gram."""
    if retained_values.size == 0:
        return 0.0

    # V (retained) and N (discarded) form an orthogonal basis.  Jacobi's
    # complementary-minor identity gives
    #
    # det(V.T D^2 V) = det(D)^2 det(N.T D^-2 N).
    #
    # Evaluate whichever side has fewer columns; this is both cheaper and more
    # accurate for the common one-alias case.
    if retained_vectors.shape[1] <= discarded_vectors.shape[1]:
        coordinate_logdet = _scaled_subspace_logdet(active_scale[:, None] * retained_vectors)
    else:
        coordinate_logdet = 2.0 * float(np.sum(np.log(active_scale)))
        coordinate_logdet += _scaled_subspace_logdet(discarded_vectors / active_scale[:, None])
    return coordinate_logdet + float(np.sum(np.log(np.abs(retained_values))))


def decompose_gram(
    matrix: NDArray,
    *,
    policy: RankPolicy = SHARED_RANK_POLICY,
    residual_tol: float = 1e-6,
    fallback_factor: NDArray | None = None,
    allow_indefinite: bool = False,
) -> RankDecomposition:
    """Equilibrate and decompose a symmetric positive-semidefinite matrix."""
    equilibrated, column_scale, active_columns, _ = _equilibrate_gram(
        matrix, allow_indefinite=allow_indefinite
    )
    width = len(column_scale)
    structural_aliases = column_scale == 0.0
    if active_columns.size == 0:
        return RankDecomposition(
            policy_version=policy.version,
            method="empty",
            column_scale=_freeze(column_scale),
            active_columns=_freeze(active_columns, dtype=int),
            rank=0,
            pre_truncation_condition=float("inf"),
            cutoff=0.0,
            rank_truncated=width > 0,
            used_svd_fallback=False,
            resolution_limited=False,
            log_pdet=0.0,
            parameter_null_basis=_freeze(np.eye(width)),
            structural_aliases=_freeze(structural_aliases, dtype=bool),
            retained_values=_freeze(np.array([])),
        )

    if not allow_indefinite:
        try:
            factor = scipy.linalg.cholesky(equilibrated, lower=True, check_finite=False)
            matrix_norm = float(np.linalg.norm(equilibrated, ord=1))
            trtri = scipy.linalg.get_lapack_funcs("trtri", (factor,))
            inverse_factor, inverse_info = trtri(
                factor,
                lower=1,
                unitdiag=0,
                overwrite_c=0,
            )
            if inverse_info != 0:
                raise np.linalg.LinAlgError("triangular inverse failed during rank certification")
            inverse_factor_frobenius = float(np.linalg.norm(inverse_factor, ord="fro"))
            min_eigenvalue_lower_bound = 1.0 / inverse_factor_frobenius**2
            pocon = scipy.linalg.get_lapack_funcs("pocon", (factor,))
            reciprocal_condition, info = pocon(factor, matrix_norm, uplo="L")
            safely_full_rank = (
                np.isfinite(min_eigenvalue_lower_bound)
                and min_eigenvalue_lower_bound
                > policy.certification_band * policy.gram_rcond * matrix_norm
            )
            if safely_full_rank:
                probe = np.arange(1.0, len(active_columns) + 1.0)
                solved = scipy.linalg.cho_solve((factor, True), probe, check_finite=False)
                residual = np.linalg.norm(equilibrated @ solved - probe) / max(
                    np.linalg.norm(probe), 1e-300
                )
                if residual <= residual_tol:
                    log_pdet = 2.0 * float(np.sum(np.log(np.diag(factor)))) + 2.0 * float(
                        np.sum(np.log(column_scale[active_columns]))
                    )
                    null = _null_basis(
                        width,
                        active_columns,
                        column_scale[active_columns],
                        np.zeros((len(active_columns), 0)),
                    )
                    return RankDecomposition(
                        policy_version=policy.version,
                        method="cholesky",
                        column_scale=_freeze(column_scale),
                        active_columns=_freeze(active_columns, dtype=int),
                        rank=len(active_columns),
                        pre_truncation_condition=float(
                            np.sqrt(1.0 / reciprocal_condition)
                            if info == 0
                            and np.isfinite(reciprocal_condition)
                            and reciprocal_condition > 0.0
                            else np.sqrt(matrix_norm / min_eigenvalue_lower_bound)
                        ),
                        cutoff=policy.gram_rcond * matrix_norm,
                        rank_truncated=len(active_columns) < width,
                        used_svd_fallback=False,
                        resolution_limited=False,
                        log_pdet=log_pdet,
                        cholesky_factor=_freeze(factor),
                        parameter_null_basis=_freeze(null),
                        structural_aliases=_freeze(structural_aliases, dtype=bool),
                    )
        except (np.linalg.LinAlgError, ValueError):
            pass

    eigenvalues, eigenvectors = np.linalg.eigh(equilibrated)
    raw_eigenvalues = eigenvalues
    max_eigenvalue = max(float(eigenvalues[-1]), 0.0)
    max_abs_eigenvalue = float(np.max(np.abs(eigenvalues), initial=0.0))
    negative_tolerance = 100.0 * _EPS * max(max_abs_eigenvalue, 1.0)
    materially_indefinite = bool(eigenvalues[0] < -negative_tolerance)
    if not allow_indefinite and materially_indefinite:
        raise ValueError(
            "matrix is materially indefinite "
            f"(min equilibrated eigenvalue={eigenvalues[0]:.3e}, "
            f"scale={max_abs_eigenvalue:.3e})"
        )
    psd_semantics = not materially_indefinite
    if psd_semantics:
        eigenvalues = np.maximum(eigenvalues, 0.0)
        max_abs_eigenvalue = max_eigenvalue
    cutoff = policy.gram_rcond * max_abs_eigenvalue
    retained_mask = eigenvalues > cutoff if psd_semantics else np.abs(eigenvalues) > cutoff
    rank = int(np.count_nonzero(retained_mask))
    positive = np.abs(eigenvalues[np.abs(eigenvalues) > 0.0])
    condition = (
        float(np.sqrt(max_abs_eigenvalue / np.min(positive)))
        if positive.size and max_abs_eigenvalue > 0.0
        else float("inf")
    )

    if rank == len(active_columns) and np.all(eigenvalues > 0.0):
        try:
            factor = scipy.linalg.cholesky(equilibrated, lower=True, check_finite=False)
            probe = np.arange(1.0, len(active_columns) + 1.0)
            solved = scipy.linalg.cho_solve((factor, True), probe, check_finite=False)
            residual = np.linalg.norm(equilibrated @ solved - probe) / max(
                np.linalg.norm(probe), 1e-300
            )
            if residual <= residual_tol:
                log_pdet = 2.0 * float(np.sum(np.log(np.diag(factor)))) + 2.0 * float(
                    np.sum(np.log(column_scale[active_columns]))
                )
                null = _null_basis(
                    width,
                    active_columns,
                    column_scale[active_columns],
                    np.zeros((len(active_columns), 0)),
                )
                return RankDecomposition(
                    policy_version=policy.version,
                    method="cholesky",
                    column_scale=_freeze(column_scale),
                    active_columns=_freeze(active_columns, dtype=int),
                    rank=rank,
                    pre_truncation_condition=condition,
                    cutoff=cutoff,
                    rank_truncated=rank < width,
                    used_svd_fallback=False,
                    resolution_limited=False,
                    log_pdet=log_pdet,
                    cholesky_factor=_freeze(factor),
                    parameter_null_basis=_freeze(null),
                    structural_aliases=_freeze(structural_aliases, dtype=bool),
                    retained_values=_freeze(eigenvalues),
                )
        except (np.linalg.LinAlgError, ValueError):
            pass

    retained_vectors = eigenvectors[:, retained_mask]
    discarded_vectors = eigenvectors[:, ~retained_mask]
    solution_basis = np.zeros((width, rank))
    estimable_basis = np.zeros((width, rank))
    active_scale = column_scale[active_columns]
    solution_basis[active_columns, :] = retained_vectors / active_scale[:, None]
    estimable_basis[active_columns, :] = retained_vectors * active_scale[:, None]
    null = _null_basis(width, active_columns, active_scale, discarded_vectors)
    retained_values = eigenvalues[retained_mask]
    # Normal equations cannot distinguish an exact active-column alias from a
    # full-rank factor direction whose squared singular value rounded to zero.
    # Structural zero columns were removed above; every other PSD truncation
    # therefore needs observation-factor certification when one is available.
    resolution_limited = bool(
        (psd_semantics and rank < len(active_columns))
        or np.any((np.abs(raw_eigenvalues) > 0.0) & ~retained_mask)
        or (fallback_factor is not None and decompose_factor(fallback_factor).rank > rank)
    )
    log_pdet = (
        2.0 * float(np.sum(np.log(active_scale))) + float(np.sum(np.log(np.abs(retained_values))))
        if rank == width
        else _retained_log_pdet(
            active_scale,
            retained_vectors,
            discarded_vectors,
            retained_values,
        )
    )
    if psd_semantics and 0 < rank < len(active_columns):
        # Choose the earliest original-coordinate representative whose
        # principal system has the certified rank. This gives exact aliases a
        # reproducible zero coefficient while estimability still uses the true
        # spectral null space above.
        selected_local: list[int] = []
        for candidate in range(len(active_columns)):
            trial = selected_local + [candidate]
            principal = equilibrated[np.ix_(trial, trial)]
            principal_rank = int(np.count_nonzero(np.linalg.eigvalsh(principal) > cutoff))
            if principal_rank > len(selected_local):
                selected_local.append(candidate)
            if len(selected_local) == rank:
                break
        if len(selected_local) == rank:
            selected_local_array = np.asarray(selected_local, dtype=int)
            representative_columns = active_columns[selected_local_array]
            representative = equilibrated[np.ix_(selected_local_array, selected_local_array)]
            try:
                representative_factor = scipy.linalg.cholesky(
                    representative, lower=True, check_finite=False
                )
                representative_basis = np.zeros((width, rank))
                representative_basis[representative_columns, np.arange(rank)] = (
                    1.0 / column_scale[representative_columns]
                )
                representative_aliases = np.ones(width, dtype=bool)
                representative_aliases[representative_columns] = False
                return RankDecomposition(
                    policy_version=policy.version,
                    method="pivoted_cholesky",
                    column_scale=_freeze(column_scale),
                    active_columns=_freeze(representative_columns, dtype=int),
                    rank=rank,
                    pre_truncation_condition=condition,
                    cutoff=cutoff,
                    rank_truncated=True,
                    used_svd_fallback=False,
                    resolution_limited=resolution_limited,
                    log_pdet=log_pdet,
                    cholesky_factor=_freeze(representative_factor),
                    pivots=_freeze(representative_columns, dtype=int),
                    solution_basis=_freeze(representative_basis),
                    parameter_null_basis=_freeze(null),
                    estimable_functional_basis=_freeze(estimable_basis),
                    structural_aliases=_freeze(representative_aliases, dtype=bool),
                    retained_values=_freeze(retained_values),
                )
            except (np.linalg.LinAlgError, ValueError):
                pass
    return RankDecomposition(
        policy_version=policy.version,
        method="gram_eigh",
        column_scale=_freeze(column_scale),
        active_columns=_freeze(active_columns, dtype=int),
        rank=rank,
        pre_truncation_condition=condition,
        cutoff=cutoff,
        rank_truncated=rank < width,
        used_svd_fallback=False,
        resolution_limited=resolution_limited,
        log_pdet=log_pdet,
        solution_basis=_freeze(solution_basis),
        parameter_null_basis=_freeze(null),
        estimable_functional_basis=_freeze(estimable_basis),
        structural_aliases=_freeze(structural_aliases, dtype=bool),
        retained_values=_freeze(retained_values),
    )


def decompose_symmetric(
    matrix: NDArray,
    *,
    policy: RankPolicy = SHARED_RANK_POLICY,
    residual_tol: float = 1e-6,
) -> RankDecomposition:
    """Decompose symmetric full-Newton curvature that may be indefinite."""
    return decompose_gram(
        matrix,
        policy=policy,
        residual_tol=residual_tol,
        allow_indefinite=True,
    )


def decompose_factor(
    factor: NDArray,
    *,
    policy: RankPolicy = SHARED_RANK_POLICY,
    retain_factor_solve: bool = False,
) -> RankDecomposition:
    """Decompose a weighted/augmented factor using the factor-space rule."""
    factor = np.asarray(factor, dtype=float)
    if factor.ndim != 2 or not np.all(np.isfinite(factor)):
        raise ValueError("factor must be a finite matrix")
    width = factor.shape[1]
    column_scale = np.linalg.norm(factor, axis=0)
    active_columns = np.flatnonzero(column_scale > 0.0)
    if active_columns.size == 0:
        decomposition = decompose_gram(np.zeros((width, width)), policy=policy)
        if retain_factor_solve:
            decomposition = replace(
                decomposition,
                factor_rhs_left_basis=_freeze(np.zeros((factor.shape[0], 0))),
            )
        return decomposition
    active_scale = column_scale[active_columns]
    equilibrated = factor[:, active_columns] / active_scale
    # A tall observation factor needs only its thin left singular vectors;
    # requesting a full U would allocate O(n²) memory.  A wide factor still
    # needs full right vectors so exact row-rank null directions are retained.
    full_matrices = equilibrated.shape[0] < equilibrated.shape[1]
    left_vectors, singular_values, Vh = np.linalg.svd(
        equilibrated,
        full_matrices=full_matrices,
    )
    cutoff = policy.factor_rcond * singular_values[0]
    retained_mask = singular_values > cutoff
    rank = int(np.count_nonzero(retained_mask))
    retained_vectors = Vh[: len(singular_values), :].T[:, retained_mask]
    discarded_vectors = Vh.T[:, rank:]
    solution_basis = np.zeros((width, rank))
    estimable_basis = np.zeros((width, rank))
    solution_basis[active_columns, :] = retained_vectors / active_scale[:, None]
    estimable_basis[active_columns, :] = retained_vectors * active_scale[:, None]
    null = _null_basis(width, active_columns, active_scale, discarded_vectors)
    retained_values = singular_values[retained_mask] ** 2
    factor_rhs_left_basis = None
    if retain_factor_solve:
        retained_left = left_vectors[:, : len(singular_values)][:, retained_mask]
        factor_rhs_left_basis = retained_left / singular_values[retained_mask]
    log_pdet = (
        2.0 * float(np.sum(np.log(active_scale))) + float(np.sum(np.log(np.abs(retained_values))))
        if rank == width
        else _retained_log_pdet(
            active_scale,
            retained_vectors,
            discarded_vectors,
            retained_values,
        )
    )
    condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values[-1] > 0.0
        else float("inf")
    )
    if 0 < rank < len(active_columns):
        equilibrated_gram = equilibrated.T @ equilibrated
        selected_local: list[int] = []
        gram_cutoff = cutoff**2
        for candidate in range(len(active_columns)):
            trial = selected_local + [candidate]
            principal = equilibrated_gram[np.ix_(trial, trial)]
            principal_rank = int(np.count_nonzero(np.linalg.eigvalsh(principal) > gram_cutoff))
            if principal_rank > len(selected_local):
                selected_local.append(candidate)
            if len(selected_local) == rank:
                break
        if len(selected_local) == rank:
            selected_local_array = np.asarray(selected_local, dtype=int)
            representative_columns = active_columns[selected_local_array]
            representative = equilibrated_gram[np.ix_(selected_local_array, selected_local_array)]
            try:
                representative_factor = scipy.linalg.cholesky(
                    representative, lower=True, check_finite=False
                )
                representative_basis = np.zeros((width, rank))
                representative_basis[representative_columns, np.arange(rank)] = (
                    1.0 / column_scale[representative_columns]
                )
                representative_aliases = np.ones(width, dtype=bool)
                representative_aliases[representative_columns] = False
                representative_rhs_left_basis = None
                representative_rhs_triangular = None
                if retain_factor_solve:
                    selected_factor = factor[:, representative_columns]
                    representative_rhs_left_basis, representative_rhs_triangular = np.linalg.qr(
                        selected_factor,
                        mode="reduced",
                    )
                return RankDecomposition(
                    policy_version=policy.version,
                    method="qr_svd",
                    column_scale=_freeze(column_scale),
                    active_columns=_freeze(representative_columns, dtype=int),
                    rank=rank,
                    pre_truncation_condition=condition,
                    cutoff=cutoff,
                    rank_truncated=True,
                    used_svd_fallback=True,
                    resolution_limited=bool(np.any((singular_values > 0.0) & ~retained_mask)),
                    log_pdet=log_pdet,
                    cholesky_factor=_freeze(representative_factor),
                    pivots=_freeze(representative_columns, dtype=int),
                    solution_basis=_freeze(representative_basis),
                    parameter_null_basis=_freeze(null),
                    estimable_functional_basis=_freeze(estimable_basis),
                    structural_aliases=_freeze(representative_aliases, dtype=bool),
                    retained_values=_freeze(retained_values),
                    factor_rhs_left_basis=(
                        None
                        if representative_rhs_left_basis is None
                        else _freeze(representative_rhs_left_basis)
                    ),
                    factor_rhs_triangular=(
                        None
                        if representative_rhs_triangular is None
                        else _freeze(representative_rhs_triangular)
                    ),
                )
            except (np.linalg.LinAlgError, ValueError):
                pass
    return RankDecomposition(
        policy_version=policy.version,
        method="qr_svd",
        column_scale=_freeze(column_scale),
        active_columns=_freeze(active_columns, dtype=int),
        rank=rank,
        pre_truncation_condition=condition,
        cutoff=cutoff,
        rank_truncated=rank < width,
        used_svd_fallback=True,
        resolution_limited=bool(np.any((singular_values > 0.0) & ~retained_mask)),
        log_pdet=log_pdet,
        solution_basis=_freeze(solution_basis),
        parameter_null_basis=_freeze(null),
        estimable_functional_basis=_freeze(estimable_basis),
        structural_aliases=_freeze(column_scale == 0.0, dtype=bool),
        retained_values=_freeze(retained_values),
        factor_rhs_left_basis=(
            None if factor_rhs_left_basis is None else _freeze(factor_rhs_left_basis)
        ),
    )


def selected_group_name_set(result, groups: Sequence, *, penalty=None) -> set[str]:
    """Return explicit solver selection, with a legacy coefficient fallback.

    Legacy results predate explicit rank/selection metadata.  When the fitted
    penalty is available, preserve every group that was not subject to a
    positive nonsmooth penalty; a valid zero estimate is not deselection.
    """
    if getattr(result, "rank_info", None) is not None:
        return set(result.rank_info.selected_group_names)
    if penalty is not None:
        from superglm.penalties.base import penalty_can_zero_groups, penalty_targets_group

        can_zero_groups = penalty_can_zero_groups(penalty)
        return {
            group.name
            for group in groups
            if not can_zero_groups
            or not penalty_targets_group(penalty, group)
            or np.linalg.norm(result.beta[group.sl]) > 1e-12
        }
    return {group.name for group in groups if np.linalg.norm(result.beta[group.sl]) > 1e-12}
