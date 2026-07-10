"""Versioned numerical-rank policy and retained-subspace operations."""

from __future__ import annotations

from dataclasses import dataclass
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
        null = self.null_basis()
        if null.shape[1] == 0:
            return True
        projection = contrast @ null
        tolerance = SHARED_RANK_POLICY.factor_rcond * max(1.0, float(np.linalg.norm(contrast)))
        return bool(np.linalg.norm(projection) <= tolerance)


def _equilibrate_gram(matrix: NDArray) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("matrix must be square")
    if not np.all(np.isfinite(values)):
        raise ValueError("matrix must be finite")
    symmetric = 0.5 * (values + values.T)
    diagonal = np.diag(symmetric)
    scale_reference = max(float(np.max(np.abs(diagonal), initial=0.0)), 1.0)
    if np.any(diagonal < -100.0 * _EPS * scale_reference):
        raise ValueError("matrix has a materially negative diagonal")
    diagonal = np.maximum(diagonal, 0.0)
    active_columns = np.flatnonzero(diagonal > 0.0)
    column_scale = np.zeros(len(diagonal))
    column_scale[active_columns] = np.sqrt(diagonal[active_columns])
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


def decompose_gram(
    matrix: NDArray,
    *,
    policy: RankPolicy = SHARED_RANK_POLICY,
    residual_tol: float = 1e-6,
    fallback_factor: NDArray | None = None,
) -> RankDecomposition:
    """Equilibrate and decompose a symmetric positive-semidefinite matrix."""
    equilibrated, column_scale, active_columns, _ = _equilibrate_gram(matrix)
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

    eigenvalues, eigenvectors = np.linalg.eigh(equilibrated)
    max_eigenvalue = max(float(eigenvalues[-1]), 0.0)
    negative_tolerance = 100.0 * _EPS * max(max_eigenvalue, 1.0)
    if eigenvalues[0] < -negative_tolerance:
        raise ValueError("matrix is materially indefinite")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    cutoff = policy.gram_rcond * max_eigenvalue
    retained_mask = eigenvalues > cutoff
    rank = int(np.count_nonzero(retained_mask))
    positive = eigenvalues[eigenvalues > 0.0]
    condition = (
        float(np.sqrt(max_eigenvalue / positive[0]))
        if positive.size and max_eigenvalue > 0.0
        else float("inf")
    )

    if rank == len(active_columns):
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
    resolution_limited = bool(
        np.any((eigenvalues > 0.0) & ~retained_mask)
        or (fallback_factor is not None and decompose_factor(fallback_factor).rank > rank)
    )
    log_pdet = (
        float(np.sum(np.log(retained_values))) + 2.0 * float(np.sum(np.log(active_scale)))
        if rank
        else 0.0
    )
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


def decompose_factor(
    factor: NDArray,
    *,
    policy: RankPolicy = SHARED_RANK_POLICY,
) -> RankDecomposition:
    """Decompose a weighted/augmented factor using the factor-space rule."""
    factor = np.asarray(factor, dtype=float)
    if factor.ndim != 2 or not np.all(np.isfinite(factor)):
        raise ValueError("factor must be a finite matrix")
    width = factor.shape[1]
    column_scale = np.linalg.norm(factor, axis=0)
    active_columns = np.flatnonzero(column_scale > 0.0)
    if active_columns.size == 0:
        return decompose_gram(np.zeros((width, width)), policy=policy)
    active_scale = column_scale[active_columns]
    equilibrated = factor[:, active_columns] / active_scale
    _, singular_values, Vh = np.linalg.svd(equilibrated, full_matrices=True)
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
    condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values[-1] > 0.0
        else float("inf")
    )
    log_pdet = (
        2.0 * float(np.sum(np.log(singular_values[retained_mask])))
        + 2.0 * float(np.sum(np.log(active_scale)))
        if rank
        else 0.0
    )
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
    )
