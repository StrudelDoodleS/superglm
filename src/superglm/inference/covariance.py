"""Covariance accessors and penalised-Hessian penalty assembly.

Extracted from ``metrics.py`` to break the inference <-> metrics circular
dependency.  These functions depend only on numpy, scipy, group_matrix, and
types — they have no dependency on inference results or metrics classes.

The module no longer assembles or decomposes a penalised Hessian.
``_penalised_xtwx_inv`` and ``_penalised_xtwx_inv_gram`` used to build
``(X'WX + S)`` here and take its pseudo-inverse; both were dead in production
and were removed.  The accessors below still apply an inverse, but only
through a factorisation handed to them.  Coefficient covariance is published by
``model/state_ops.py`` and ``inference/metrics.py``, which consult
``decompose_gram_if_authoritative`` and fall back to a streamed observation
factor themselves.  ``_active_penalty_matrix`` -- the ``S`` half of that
assembly -- stayed, because those two are its callers.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from superglm.group_matrix import (
    DiscretizedSplineCategoricalGroupMatrix,
    DiscretizedSSPGroupMatrix,
    SparseSSPGroupMatrix,
    SplineCategoricalGroupMatrix,
)
from superglm.solvers.hessian_factor import HessianFactor, _expanded_component_omega
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
