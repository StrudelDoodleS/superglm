"""Penalised (X'WX + S)^{-1} covariance utilities.

Extracted from ``metrics.py`` to break the inference <-> metrics circular
dependency.  These functions depend only on numpy, scipy, group_matrix, and
types — they have no dependency on inference results or metrics classes.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

from superglm._group_matrix._group_matrix_execution import MatrixExecutionPlan
from superglm.group_matrix import (
    DesignMatrix,
    DiscretizedSplineCategoricalGroupMatrix,
    DiscretizedSSPGroupMatrix,
    GroupMatrix,
    SparseSSPGroupMatrix,
    SplineCategoricalGroupMatrix,
)
from superglm.solvers.centered_system import iter_grouped_design_chunks, penalty_factor
from superglm.solvers.hessian_factor import HessianFactor, _expanded_component_omega
from superglm.solvers.rank import (
    decompose_factor,
    decompose_gram_if_authoritative,
    streamed_weighted_factor,
)
from superglm.solvers.structured import (
    ProfiledBlockSchurFactor,
    ProfiledScalarSchurFactor,
)
from superglm.solvers.sum_to_zero import ProfiledSumToZeroBlockFactor
from superglm.types import GroupSlice


def _selector_indices(selector, size: int) -> tuple[NDArray[np.intp], bool]:
    """Normalize one NumPy-style selector to explicit integer indices."""
    if isinstance(selector, slice):
        return np.arange(size, dtype=np.intp)[selector], False
    values = np.asarray(selector)
    scalar = values.ndim == 0
    if values.dtype == bool:
        if values.ndim != 1 or values.shape[0] != size:
            raise IndexError(
                "boolean covariance index must be one-dimensional and match "
                f"the indexed dimension of {size}"
            )
        values = np.flatnonzero(values)
    else:
        values = values.astype(np.intp, copy=False).ravel()
    values = np.where(values < 0, values + size, values).astype(np.intp, copy=False)
    if np.any((values < 0) | (values >= size)):
        raise IndexError("covariance index is outside the matrix dimensions")
    return values, scalar


class StructuredSlopeCovarianceAccessor:
    """Selected access to a compact profiled slope covariance."""

    backend = "structured"

    def __init__(
        self,
        factor: HessianFactor,
        *,
        scale: float = 1.0,
    ):
        self.factor = factor
        self.scale = float(scale)
        self.shape = factor.shape
        self.ndim = 2

    def scaled(self, scale: float) -> StructuredSlopeCovarianceAccessor:
        return StructuredSlopeCovarianceAccessor(
            self.factor,
            scale=self.scale * float(scale),
        )

    def selected_block(self, indices: NDArray) -> NDArray:
        selected = np.asarray(indices, dtype=np.intp)
        return self.scale * self.factor.selected_inverse_block(selected)

    def selected_diagonal(self, indices: NDArray) -> NDArray:
        selected = np.asarray(indices, dtype=np.intp)
        return self.scale * self.factor.selected_inverse_diagonal(selected)

    def solve(self, rhs: NDArray) -> NDArray:
        return self.scale * self.factor.solve(rhs)

    def trace(self) -> float:
        indices = np.arange(self.shape[0], dtype=np.intp)
        return float(np.sum(self.selected_diagonal(indices)))

    def quadratic_form(self, contrast: NDArray) -> float:
        values = np.asarray(contrast, dtype=np.float64)
        if values.shape != (self.shape[0],):
            raise ValueError(f"contrast must have shape ({self.shape[0]},).")
        return float(values @ self.solve(values))

    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) != 2:
            raise IndexError("covariance access requires row and column selectors")
        rows, row_scalar = _selector_indices(key[0], self.shape[0])
        columns, column_scalar = _selector_indices(key[1], self.shape[1])
        selected = np.unique(np.concatenate((rows, columns)))
        block = self.selected_block(selected)
        positions = np.full(self.shape[0], -1, dtype=np.intp)
        positions[selected] = np.arange(len(selected))
        result = block[np.ix_(positions[rows], positions[columns])]
        if row_scalar and column_scalar:
            return float(result[0, 0])
        if row_scalar:
            return result[0]
        if column_scalar:
            return result[:, 0]
        return result

    def __array__(self, dtype=None, copy=None) -> NDArray:
        del copy
        values = self.selected_block(np.arange(self.shape[0], dtype=np.intp))
        return values.astype(dtype, copy=False) if dtype is not None else values


class StructuredCovarianceAccessor:
    """Selected access to an augmented structured coefficient covariance.

    The retained factor lives in solver coordinates. ``intercept_shift`` maps
    its intercept into the canonical public coefficient state without
    materializing the full augmented inverse.
    """

    backend = "structured"

    def __init__(
        self,
        factor: (
            ProfiledScalarSchurFactor | ProfiledBlockSchurFactor | ProfiledSumToZeroBlockFactor
        ),
        *,
        intercept_shift: NDArray | None = None,
        scale: float = 1.0,
    ):
        self.factor = factor
        self.augmented_factor = factor.augmented_factor
        p = factor.shape[0]
        if intercept_shift is None:
            intercept_shift = np.zeros(p, dtype=np.float64)
        shift = np.array(intercept_shift, dtype=np.float64, copy=True)
        if shift.shape != (p,):
            raise ValueError(f"intercept_shift must have shape ({p},).")
        shift.setflags(write=False)
        self.intercept_shift = shift
        self.scale = float(scale)
        self.shape = (p + 1, p + 1)
        self.ndim = 2
        self.slopes = StructuredSlopeCovarianceAccessor(factor, scale=self.scale)

    def scaled(self, scale: float) -> StructuredCovarianceAccessor:
        return StructuredCovarianceAccessor(
            self.factor,
            intercept_shift=self.intercept_shift,
            scale=self.scale * float(scale),
        )

    def _solver_contrasts(self, public_indices: NDArray) -> NDArray:
        selected = np.asarray(public_indices, dtype=np.intp)
        if selected.ndim != 1 or np.any((selected < 0) | (selected >= self.shape[0])):
            raise IndexError("selected augmented covariance index is out of bounds")
        contrasts = np.zeros((self.shape[0], len(selected)), dtype=np.float64)
        for column, index in enumerate(selected):
            if index == 0:
                contrasts[0, column] = 1.0
                contrasts[1:, column] = self.intercept_shift
            else:
                contrasts[index, column] = 1.0
        return contrasts

    def _check_block_request(self, selected: NDArray) -> None:
        slope_indices = selected[selected > 0] - 1
        structured = np.intersect1d(
            slope_indices,
            np.asarray(self.factor.structured_indices, dtype=np.intp).ravel(),
            assume_unique=False,
        )
        limit = self.augmented_factor.max_structured_inverse_block
        if len(structured) > limit:
            raise RuntimeError(
                "Refusing to materialize a full dominant structured covariance "
                f"block with {len(structured)} coefficients; request its diagonal or "
                "a bounded selected block instead."
            )

    def selected_block(self, indices: NDArray) -> NDArray:
        selected = np.asarray(indices, dtype=np.intp)
        self._check_block_request(selected)
        contrasts = self._solver_contrasts(selected)
        solved = self.augmented_factor.solve(contrasts)
        return self.scale * (contrasts.T @ solved)

    def selected_diagonal(self, indices: NDArray) -> NDArray:
        selected = np.asarray(indices, dtype=np.intp)
        if selected.ndim != 1 or np.any((selected < 0) | (selected >= self.shape[0])):
            raise IndexError("selected augmented covariance index is out of bounds")
        result = np.empty(len(selected), dtype=np.float64)
        intercept_mask = selected == 0
        if np.any(intercept_mask):
            result[intercept_mask] = self.intercept_variance()
        slope_mask = ~intercept_mask
        if np.any(slope_mask):
            result[slope_mask] = self.slope_selected_diagonal(selected[slope_mask] - 1)
        return result

    def slope_selected_block(self, indices: NDArray) -> NDArray:
        return self.slopes.selected_block(indices)

    def slope_selected_diagonal(self, indices: NDArray) -> NDArray:
        return self.slopes.selected_diagonal(indices)

    def intercept_variance(self) -> float:
        contrast = np.concatenate(([1.0], self.intercept_shift))
        return float(self.scale * (contrast @ self.augmented_factor.solve(contrast)))

    def intercept_cross(self, slope_indices: NDArray) -> NDArray:
        selected = np.asarray(slope_indices, dtype=np.intp)
        if selected.ndim != 1:
            raise ValueError("slope_indices must be one-dimensional")
        contrast = np.concatenate(([1.0], self.intercept_shift))
        solved = self.augmented_factor.solve(contrast)
        return self.scale * solved[selected + 1]

    def solve(self, rhs: NDArray) -> NDArray:
        values = np.asarray(rhs, dtype=np.float64)
        vector_rhs = values.ndim == 1
        if vector_rhs:
            values = values[:, None]
        if values.ndim != 2 or values.shape[0] != self.shape[0]:
            raise ValueError(f"rhs must have shape ({self.shape[0]},) or ({self.shape[0]}, m).")
        solver_rhs = np.array(values, copy=True)
        solver_rhs[1:] += self.intercept_shift[:, None] * values[0]
        solver_solution = self.augmented_factor.solve(solver_rhs)
        public_solution = np.array(solver_solution, copy=True)
        public_solution[0] += self.intercept_shift @ solver_solution[1:]
        public_solution *= self.scale
        return public_solution[:, 0] if vector_rhs else public_solution

    def quadratic_form(self, contrast: NDArray) -> float:
        values = np.asarray(contrast, dtype=np.float64)
        if values.shape != (self.shape[0],):
            raise ValueError(f"contrast must have shape ({self.shape[0]},).")
        return float(values @ self.solve(values))

    def trace(self) -> float:
        return float(
            self.intercept_variance()
            + np.sum(
                self.slope_selected_diagonal(
                    np.arange(self.factor.shape[0], dtype=np.intp),
                )
            )
        )

    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) != 2:
            raise IndexError("covariance access requires row and column selectors")
        rows, row_scalar = _selector_indices(key[0], self.shape[0])
        columns, column_scalar = _selector_indices(key[1], self.shape[1])
        selected = np.unique(np.concatenate((rows, columns)))
        block = self.selected_block(selected)
        positions = np.full(self.shape[0], -1, dtype=np.intp)
        positions[selected] = np.arange(len(selected))
        result = block[np.ix_(positions[rows], positions[columns])]
        if row_scalar and column_scalar:
            return float(result[0, 0])
        if row_scalar:
            return result[0]
        if column_scalar:
            return result[:, 0]
        return result

    def __array__(self, dtype=None, copy=None) -> NDArray:
        del copy
        values = self.selected_block(np.arange(self.shape[0], dtype=np.intp))
        return values.astype(dtype, copy=False) if dtype is not None else values


def covariance_selected_block(covariance, indices: NDArray) -> NDArray:
    """Return one principal block from a dense or compact covariance."""
    selected = np.asarray(indices, dtype=np.intp)
    if hasattr(covariance, "selected_block"):
        return np.asarray(covariance.selected_block(selected), dtype=np.float64)
    return np.asarray(covariance[np.ix_(selected, selected)], dtype=np.float64)


def covariance_selected_diagonal(covariance, indices: NDArray) -> NDArray:
    """Return selected diagonal entries from a dense or compact covariance."""
    selected = np.asarray(indices, dtype=np.intp)
    if hasattr(covariance, "selected_diagonal"):
        return np.asarray(covariance.selected_diagonal(selected), dtype=np.float64)
    return np.diag(np.asarray(covariance))[selected]


def covariance_factor_smooth_raw_level_block(
    covariance,
    public_group_indices: NDArray,
    *,
    level: int,
    n_levels: int,
    block_size: int,
    term_name: str | None = None,
) -> NDArray:
    """Return one raw-level SZ covariance from public ``K - 1`` coordinates."""
    public = np.asarray(public_group_indices, dtype=np.intp)
    expected_width = (n_levels - 1) * block_size
    if public.shape != (expected_width,):
        raise ValueError(
            "public_group_indices must contain the complete "
            f"(K - 1)k SZ block ({expected_width} entries)."
        )
    if isinstance(level, bool) or not isinstance(level, (int, np.integer)):
        raise TypeError("level must be an integer index")
    level = int(level)
    if level < 0 or level >= n_levels:
        raise IndexError("raw level index is outside the fitted level range")

    if (
        isinstance(covariance, StructuredCovarianceAccessor)
        and isinstance(
            covariance.factor,
            ProfiledSumToZeroBlockFactor,
        )
        and (term_name is None or covariance.factor.dominant_group_name == term_name)
    ):
        return covariance.scale * covariance.factor.raw_level_inverse_block(level)

    if level < n_levels - 1:
        start = level * block_size
        return covariance_selected_block(
            covariance,
            public[start : start + block_size],
        )

    public_covariance = covariance_selected_block(covariance, public)
    final_contrast = np.tile(
        -np.eye(block_size, dtype=np.float64),
        (1, n_levels - 1),
    )
    result = final_contrast @ public_covariance @ final_contrast.T
    return 0.5 * (result + result.T)


def covariance_quadratic_form(covariance, contrast: NDArray) -> float:
    """Evaluate one dense or compact covariance quadratic form."""
    values = np.asarray(contrast, dtype=np.float64)
    if hasattr(covariance, "quadratic_form"):
        return float(covariance.quadratic_form(values))
    return float(values @ covariance @ values)


def covariance_slope_view(covariance, *, scale: float = 1.0):
    """Return a phi-scaled slope covariance without forcing materialization."""
    if isinstance(covariance, StructuredCovarianceAccessor):
        return covariance.scaled(scale).slopes
    return float(scale) * np.asarray(covariance)[1:, 1:]


def _second_diff_penalty(p: int) -> NDArray:
    """Second-difference penalty matrix D2'D2 for p basis functions."""
    D2 = np.diff(np.eye(p), n=2, axis=0)
    return D2.T @ D2


def _active_penalty_matrix(
    group_matrices: list,
    groups: list[GroupSlice],
    active_groups: list[GroupSlice],
    lambda2: float | dict[str, float],
    *,
    S_override: NDArray | None = None,
    reml_penalties: list | None = None,
) -> NDArray:
    """Build ``S`` directly in compact active coordinates."""
    p_active = sum(group.size for group in active_groups)
    if p_active == 0:
        return np.empty((0, 0), dtype=np.float64)

    active_by_name = {group.name: group for group in active_groups}
    if S_override is not None:
        selected_columns = np.asarray(
            [
                column
                for group in groups
                if group.name in active_by_name
                for column in range(group.start, group.end)
            ],
            dtype=np.intp,
        )
        return np.asarray(S_override[np.ix_(selected_columns, selected_columns)], dtype=np.float64)

    S = np.zeros((p_active, p_active), dtype=np.float64)
    if reml_penalties is not None:
        represented_group_indices = {component.group_index for component in reml_penalties}
        for component in reml_penalties:
            active_group = active_by_name.get(component.group_name)
            if active_group is None:
                continue
            gm = group_matrices[component.group_index]
            lam = lambda2[component.name] if isinstance(lambda2, dict) else lambda2
            if lam == 0:
                continue
            if component.penalty_kind == "identity":
                indices = np.arange(
                    active_group.start,
                    active_group.end,
                    dtype=np.intp,
                )
                S[indices, indices] += lam
                continue
            omega = (
                component.omega_ssp
                if component.omega_ssp is not None
                else gm.R_inv.T @ component.omega_raw @ gm.R_inv
            )
            omega = _expanded_component_omega(
                component,
                omega,
                active_group.size,
            )
            S[active_group.sl, active_group.sl] += lam * omega

        for group_index, group in enumerate(groups):
            if group_index in represented_group_indices:
                continue
            active_group = active_by_name.get(group.name)
            if active_group is None or group.scop_reparameterization is None or not group.penalized:
                continue
            lam = lambda2.get(group.name, 0.0) if isinstance(lambda2, dict) else lambda2
            if lam > 0:
                S[active_group.sl, active_group.sl] += (
                    lam * group.scop_reparameterization.penalty_matrix()
                )
        return S

    for gm, group in zip(group_matrices, groups, strict=True):
        active_group = active_by_name.get(group.name)
        if active_group is None or not group.penalized:
            continue
        if isinstance(
            gm,
            SparseSSPGroupMatrix
            | SplineCategoricalGroupMatrix
            | DiscretizedSplineCategoricalGroupMatrix
            | DiscretizedSSPGroupMatrix,
        ):
            omega_components = getattr(gm, "omega_components", None)
            if omega_components is not None:
                from superglm.reml.penalty_algebra import resolve_component_lambda

                for suffix, omega_j in omega_components:
                    lam_j = resolve_component_lambda(lambda2, group.name, suffix)
                    if lam_j == 0:
                        continue
                    S[active_group.sl, active_group.sl] += lam_j * (gm.R_inv.T @ omega_j @ gm.R_inv)
                continue
            lam = lambda2.get(group.name, 0.0) if isinstance(lambda2, dict) else lambda2
            if lam == 0:
                continue
            omega = gm.omega
            if omega is None:
                omega = _second_diff_penalty(gm.R_inv.shape[0])
            S[active_group.sl, active_group.sl] += lam * gm.R_inv.T @ omega @ gm.R_inv
        elif group.scop_reparameterization is not None:
            lam = lambda2.get(group.name, 0.0) if isinstance(lambda2, dict) else lambda2
            if lam == 0:
                continue
            S[active_group.sl, active_group.sl] += (
                lam * group.scop_reparameterization.penalty_matrix()
            )
    return S


def _penalised_xtwx_inv(
    beta: NDArray,
    W: NDArray,
    group_matrices: list,
    groups: list[GroupSlice],
    lambda2: float | dict[str, float],
    S_override: NDArray | None = None,
    selected_group_names: set[str] | None = None,
) -> tuple[NDArray, NDArray, NDArray, list[GroupSlice], list]:
    """Compute (X'WX + S)^{-1} via augmented QR + truncated SVD.

    Shared by ``model._coef_covariance`` and ``ModelMetrics._active_info``.

    Parameters
    ----------
    beta : (p,) array — full coefficient vector.
    W : (n,) array — working weights (dmu_deta² / V, sample_weight-scaled).
    group_matrices : list of GroupMatrix — design matrices per group.
    groups : list of GroupSlice — slices into beta for each group.
    lambda2 : float or dict[str, float]
        Smoothing penalty weight. A scalar applies to all groups; a dict
        maps group names to per-group lambdas (REML).

    Returns
    -------
    X_a : (n, p_active) dense active design matrix.
    XtWX_S_inv : (p_active, p_active) inverse of (X'WX + S).
    XtWX_S_inv_aug : (p_active+1, p_active+1) inverse of augmented system
        including intercept row/column. Element [0,0] is intercept variance,
        [1:,1:] are feature marginal variances (accounting for intercept).
    active_groups : list of GroupSlice re-indexed to X_a columns.
    active_gms : list of GroupMatrix for active groups.
    """
    active_cols: list[NDArray] = []
    active_groups_out: list[GroupSlice] = []
    active_gms: list = []
    active_group_names: list[str] = []
    col = 0
    for gm, g in zip(group_matrices, groups):
        if (
            g.name in selected_group_names
            if selected_group_names is not None
            else np.linalg.norm(beta[g.sl]) > 1e-12
        ):
            arr = gm.toarray()
            active_cols.append(arr)
            active_gms.append(gm)
            active_group_names.append(g.name)
            p_g = arr.shape[1]
            active_groups_out.append(
                GroupSlice(
                    name=g.name,
                    start=col,
                    end=col + p_g,
                    weight=g.weight,
                    penalty_dim=g.penalty_dim,
                    penalized=g.penalized,
                    feature_name=g.feature_name,
                    subgroup_type=g.subgroup_type,
                    constraints=g.constraints,
                    monotone_engine=g.monotone_engine,
                    scop_reparameterization=g.scop_reparameterization,
                )
            )
            col += p_g

    if not active_cols:
        n = len(W)
        # Augmented inverse is 1×1: just the intercept variance
        w_sum = float(np.sum(W))
        aug_inv = np.array([[1.0 / w_sum]]) if w_sum > 0 else np.array([[0.0]])
        return np.empty((n, 0)), np.empty((0, 0)), aug_inv, [], []

    X_a = np.hstack(active_cols)
    p_a = X_a.shape[1]

    # Build sqrt(S) factor: L such that L'L = S (block-diagonal penalty)
    # Unpenalized groups (e.g. select=True null-space) get no penalty contribution.
    if S_override is not None:
        # S_override is full (p x p) — slice to active columns, then sqrt
        active_idx = []
        for ag, gname in zip(active_groups_out, active_group_names):
            orig_g = next(g for g in groups if g.name == gname)
            active_idx.extend(range(orig_g.start, orig_g.end))
        active_idx_arr = np.array(active_idx)
        S_active = S_override[np.ix_(active_idx_arr, active_idx_arr)]
        eigvals_s, eigvecs_s = np.linalg.eigh(S_active)
        eigvals_s = np.maximum(eigvals_s, 0.0)
        S_rows = np.sqrt(eigvals_s)[:, None] * eigvecs_s.T  # sqrt(S)
    else:
        S_rows = np.zeros((p_a, p_a))
        for gm_orig, ag, gname in zip(active_gms, active_groups_out, active_group_names):
            if not ag.penalized:
                continue

            if isinstance(
                gm_orig,
                SparseSSPGroupMatrix
                | SplineCategoricalGroupMatrix
                | DiscretizedSplineCategoricalGroupMatrix
                | DiscretizedSSPGroupMatrix,
            ):
                R_inv = gm_orig.R_inv
                omega_components = getattr(gm_orig, "omega_components", None)
                if omega_components is not None:
                    from superglm.reml.penalty_algebra import resolve_component_lambda

                    S_g = np.zeros((ag.size, ag.size))
                    for suffix, omega_j in omega_components:
                        lam_j = resolve_component_lambda(lambda2, gname, suffix)
                        if lam_j == 0:
                            continue
                        S_g += lam_j * (R_inv.T @ omega_j @ R_inv)
                    if not np.any(S_g):
                        continue
                else:
                    lam_g = lambda2.get(gname, 0.0) if isinstance(lambda2, dict) else lambda2
                    if lam_g == 0:
                        continue
                    omega = gm_orig.omega
                    if omega is None:
                        p_b = R_inv.shape[0]
                        omega = _second_diff_penalty(p_b)
                    S_g = lam_g * R_inv.T @ omega @ R_inv
            elif ag.scop_reparameterization is not None:
                lam_g = lambda2.get(gname, 0.0) if isinstance(lambda2, dict) else lambda2
                if lam_g == 0:
                    continue
                S_g = lam_g * ag.scop_reparameterization.penalty_matrix()
            else:
                continue

            eigvals_g, eigvecs_g = np.linalg.eigh(S_g)
            eigvals_g = np.maximum(eigvals_g, 0.0)
            L_g = np.sqrt(eigvals_g)[:, None] * eigvecs_g.T
            S_rows[ag.sl, ag.sl] = L_g

    # Augmented QR: [sqrt(W)*X; sqrt(S)] → R'R = X'WX + S
    A = np.vstack([X_a * np.sqrt(W)[:, None], S_rows])
    XtWX_S_inv = decompose_factor(A).pseudo_inverse()

    # Augmented (p+1)×(p+1) inverse including intercept row/column.
    # The augmented Fisher information is:
    #   F_aug = [[sum(W),  X'W1], [X'W1,  X'WX + S]]
    # where X'W1 = X_a' @ W (cross-product of features with intercept).
    # This is needed for correct SEs that account for intercept estimation.
    sqrtW = np.sqrt(W)
    X_aug = np.hstack([np.ones((len(W), 1)), X_a])
    S_aug_rows = np.zeros((p_a + 1, p_a + 1))
    S_aug_rows[1:, 1:] = S_rows  # no penalty on intercept
    A_aug = np.vstack([X_aug * sqrtW[:, None], S_aug_rows])
    XtWX_S_inv_aug = decompose_factor(A_aug).pseudo_inverse()

    return X_a, XtWX_S_inv, XtWX_S_inv_aug, active_groups_out, active_gms


def _intercept_prefixed_chunks(
    chunks: Iterator[tuple[int, int, NDArray]],
) -> Iterator[tuple[int, int, NDArray]]:
    """Prepend the intercept column to each bounded design chunk."""
    for start, stop, block in chunks:
        yield start, stop, np.column_stack((np.ones(len(block)), block))


def _certified_penalised_inverse(
    active_gms: list[GroupMatrix],
    W: NDArray,
    S: NDArray,
    *,
    intercept: bool,
) -> NDArray:
    """``(X'WX + S)^{-1}`` from the observation factor, for an uncertifiable Gram.

    ``needs_factor_certification`` says the normal equations cannot certify
    their own retained subspace, and a certificate governs that subspace as
    well as the integer rank -- so this rebuilds the same operator as
    ``A'A`` with ``A = [sqrt(W) X ; sqrt(S)]`` and decomposes ``A``.  It is
    the destination ``_penalised_xtwx_inv`` already uses at every one of its
    own inversions; the difference is that ``A`` is accumulated here in
    bounded row chunks, so the ``(n, p)`` dense block that
    ``_penalised_xtwx_inv_gram`` exists to avoid is never materialised.  That
    is the same shape ``inference/metrics.py``'s ``_certified_*_rank`` helpers
    use for the same verdict.

    ``intercept`` prepends the intercept column, giving the augmented system
    ``[[sum W, X'W1], [X'W1, X'WX + S]]`` whose factor is ``A`` with a leading
    ``sqrt(W)`` column and an unpenalised leading column in ``sqrt(S)``.

    **The verdict has two arms and this route serves both.**  Neither is "the"
    trigger, and the second is the likelier one in production:

    ``rank < width and resolution_limited`` -- the Gram has ERASED a direction
    the factor resolves.  Measured on issue #356's aliased-pair fit with the
    ridge that places the alias's residual eigenvalue between the two rank
    cuts: the Gram reports rank 17 of 18 and ``||pinv||_2 = 1.5422``, this
    route reports rank 18 and ``2.0000e+11``, and the largest published
    standard error moves from 1.18 to 2.58e+05.

    ``rank == width and pre_truncation_condition >= certification_condition``
    -- NOTHING is discarded; the whole retained subspace is simply recomputed
    from a factor that never squared the condition.  That is the ordinary
    ill-conditioned rating design, and the correction it buys is the larger of
    the two: on the full-rank collinear fixture in
    ``test_factor_certification_authority.py`` the published standard errors
    move 5.74e-03 to 4.15e-02 relative against the Gram's answer over 7
    ``OPENBLAS_CORETYPE`` microkernels, against 1.78e-10 on the aliased one.

    Outside both arms the two agree and this route is not taken.
    """
    width = S.shape[0]
    design = DesignMatrix(active_gms, n=len(W), p=width)
    chunks = iter_grouped_design_chunks(design)
    factor = streamed_weighted_factor(
        _intercept_prefixed_chunks(chunks) if intercept else chunks,
        W,
    )
    smooth_factor = penalty_factor(S)
    if intercept:
        # The intercept carries no penalty, exactly as ``S_aug_rows`` above.
        # Unconditional, including on the zero-penalty row count that
        # ``penalty_factor`` returns: the padding is what makes that empty
        # block the augmented width, so the stack below has one shape.
        smooth_factor = np.hstack(
            (np.zeros((smooth_factor.shape[0], 1)), smooth_factor),
        )
    return decompose_factor(np.vstack((factor, smooth_factor))).pseudo_inverse()


def _penalised_xtwx_inv_gram(
    beta: NDArray,
    W: NDArray,
    group_matrices: list,
    groups: list[GroupSlice],
    lambda2: float | dict[str, float],
    S_override: NDArray | None = None,
    selected_group_names: set[str] | None = None,
    reml_penalties: list | None = None,
    compute_augmented: bool = True,
) -> tuple[NDArray, NDArray, list[GroupSlice], NDArray | None, NDArray | None]:
    """Fast (X'WX + S)^{-1} via per-group gram matrices.

    Same result as ``_penalised_xtwx_inv`` but avoids forming the dense
    (n, p) matrix. Computes X'WX block-by-block using the group gram
    kernels, then inverts the (p_active, p_active) system directly.
    Cost is O(n · sum(p_g²)) + O(p³) instead of O(n · p²) -- on the
    certified fast path only. When a consult falls back to the certified
    factor (``needs_factor_certification``), the design is streamed in
    bounded dense chunks and the cost is at least O(n · p²): inside the
    band, neither half of the fast-path claim holds.

    Does NOT return X_a (not needed for REML). For leverage/hat matrix
    diagnostics, use ``_penalised_xtwx_inv`` instead.

    **Not covariance-only.**  Inside the certification band the two inversions
    below leave the Gram for the observation factor, and this helper has three
    consumers, not one: the published covariance here, ``reml/runner.py``'s
    Fellner-Schall trace term, and ``inference/_term_covariance.py``'s Bayesian
    term covariance.  The route change reaches all three by construction.  No
    REML fit in the suite reaches the band -- an instrumented run of the whole
    suite records the fallback only from this module's own tests -- and a
    ``fit_reml`` on the aliased-pair design records none either, so the REML
    consumer is unpinned rather than known-safe.  Tracked on #356.

    Returns
    -------
    XtWX_S_inv : (p_active, p_active) inverse of (X'WX + S).
    XtWX_S_inv_aug : (p_active+1, p_active+1) inverse of augmented system
        including intercept row/column.
    active_groups : list of GroupSlice re-indexed to active columns.
    XtWX : (p_active, p_active) X'WX gram matrix, or None if p_active == 0.
    S : (p_active, p_active) penalty matrix, or None if p_active == 0.
    """
    # Identify active groups
    active_gms: list = []
    active_groups_out: list[GroupSlice] = []
    active_group_names: list[str] = []
    col = 0
    for gm, g in zip(group_matrices, groups):
        if (
            g.name in selected_group_names
            if selected_group_names is not None
            else np.linalg.norm(beta[g.sl]) > 1e-12
        ):
            active_gms.append(gm)
            active_group_names.append(g.name)
            p_g = gm.shape[1]
            active_groups_out.append(
                GroupSlice(
                    name=g.name,
                    start=col,
                    end=col + p_g,
                    weight=g.weight,
                    penalty_dim=g.penalty_dim,
                    penalized=g.penalized,
                    feature_name=g.feature_name,
                    subgroup_type=g.subgroup_type,
                    constraints=g.constraints,
                    monotone_engine=g.monotone_engine,
                    scop_reparameterization=g.scop_reparameterization,
                )
            )
            col += p_g

    p_a = col
    if p_a == 0:
        w_sum = float(np.sum(W))
        aug_inv = np.array([[1.0 / w_sum]]) if w_sum > 0 else np.array([[0.0]])
        return np.empty((0, 0)), aug_inv, [], None, None

    # The active group subset has its own solver coordinates, so use one
    # ephemeral plan and fuse X'W with its Gram when the augmented covariance
    # is requested.
    active_plan = MatrixExecutionPlan(active_gms, n=len(W))
    active_moments = active_plan.moments(W, include_xtw=compute_augmented)
    XtWX = active_moments.gram

    S = _active_penalty_matrix(
        group_matrices,
        groups,
        active_groups_out,
        lambda2,
        S_override=S_override,
        reml_penalties=reml_penalties,
    )

    M = XtWX + S
    # The pseudo-inverse here IS the published covariance, so it may only be
    # taken from a Gram that can certify its own retained subspace.
    # ``decompose_gram_if_authoritative`` returns ``None`` for exactly the
    # ``needs_factor_certification`` band, on EITHER of its two arms -- the
    # Gram having erased a direction the factor resolves, or the Gram being
    # full rank but reached through a squared condition.  The factor is the
    # authority in both.
    authoritative_gram = decompose_gram_if_authoritative(M)
    XtWX_S_inv = (
        authoritative_gram.pseudo_inverse()
        if authoritative_gram is not None
        else _certified_penalised_inverse(active_gms, W, S, intercept=False)
    )

    if not compute_augmented:
        return XtWX_S_inv, np.empty((0, 0)), active_groups_out, XtWX, S

    # Augmented (p+1)×(p+1) inverse including intercept row/column.
    XtW1 = active_moments.xtw
    if XtW1 is None:  # pragma: no cover - guaranteed by compute_augmented
        raise RuntimeError("execution plan did not return X'W")
    sum_W = float(np.sum(W))

    M_aug = np.empty((p_a + 1, p_a + 1))
    M_aug[0, 0] = sum_W
    M_aug[0, 1:] = XtW1
    M_aug[1:, 0] = XtW1
    M_aug[1:, 1:] = M  # XtWX + S
    # Its own verdict, not the unaugmented one's: bordering with the intercept
    # changes the width, the spectrum and hence the certification band.
    authoritative_augmented_gram = decompose_gram_if_authoritative(M_aug)
    XtWX_S_inv_aug = (
        authoritative_augmented_gram.pseudo_inverse()
        if authoritative_augmented_gram is not None
        else _certified_penalised_inverse(active_gms, W, S, intercept=True)
    )

    return XtWX_S_inv, XtWX_S_inv_aug, active_groups_out, XtWX, S
