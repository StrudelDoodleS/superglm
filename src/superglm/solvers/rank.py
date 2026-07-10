"""Versioned numerical-rank policy and retained-subspace operations."""

from __future__ import annotations

from collections.abc import Sequence
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
    group_edf: dict[str, float]
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
        for column in self.selected_columns:
            contrast = np.zeros(len(self.mean_x))
            contrast[column] = 1.0
            result[column] = self.is_estimable(contrast)
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

    eigenvalues, eigenvectors = np.linalg.eigh(equilibrated)
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
    resolution_limited = bool(
        np.any((np.abs(eigenvalues) > 0.0) & ~retained_mask)
        or (fallback_factor is not None and decompose_factor(fallback_factor).rank > rank)
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
                representative_logdet = 2.0 * float(
                    np.sum(np.log(np.diag(representative_factor)))
                ) + 2.0 * float(np.sum(np.log(column_scale[representative_columns])))
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
                    log_pdet=representative_logdet,
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
    if rank:
        retained_coordinates = active_scale[:, None] * retained_vectors
        sign, coordinate_logdet = np.linalg.slogdet(retained_coordinates.T @ retained_coordinates)
        if sign <= 0:
            raise ValueError("retained Gram pseudo-determinant is not positive")
        log_pdet = float(coordinate_logdet + np.sum(np.log(np.abs(retained_values))))
    else:
        log_pdet = 0.0
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
                representative_logdet = 2.0 * float(
                    np.sum(np.log(np.diag(representative_factor)))
                ) + 2.0 * float(np.sum(np.log(column_scale[representative_columns])))
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
                    log_pdet=representative_logdet,
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
    if rank:
        retained_factor = (
            active_scale[:, None] * retained_vectors * singular_values[retained_mask][None, :]
        )
        sign, log_pdet = np.linalg.slogdet(retained_factor.T @ retained_factor)
        if sign <= 0:
            raise ValueError("retained factor pseudo-determinant is not positive")
        log_pdet = float(log_pdet)
    else:
        log_pdet = 0.0
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


def selected_group_name_set(result, groups: Sequence) -> set[str]:
    """Return explicit solver selection, with a legacy coefficient fallback."""
    if getattr(result, "rank_info", None) is not None:
        return set(result.rank_info.selected_group_names)
    return {group.name for group in groups if np.linalg.norm(result.beta[group.sl]) > 1e-12}
