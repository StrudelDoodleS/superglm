"""Private identifiability helpers for spline feature specs."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def build_identifiability_projection(
    *,
    inverse: NDArray,
    basis: NDArray,
    constraint_projection: NDArray | None,
    n_basis: int,
    geometry_weight: NDArray | None = None,
) -> NDArray | None:
    """Compute the projection that removes the intercept-confounded direction."""
    n_cols = constraint_projection.shape[1] if constraint_projection is not None else n_basis
    if n_cols <= 1:
        return constraint_projection

    if geometry_weight is None:
        counts = np.bincount(inverse, minlength=basis.shape[0]).astype(np.float64)
    else:
        weights = np.asarray(geometry_weight, dtype=np.float64).ravel()
        if weights.shape != inverse.shape:
            raise ValueError("geometry_weight must be one-dimensional and match the spline rows")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("geometry_weight must be finite and non-negative")
        if not np.any(weights > 0.0):
            raise ValueError("geometry_weight must retain at least one spline row")
        counts = np.bincount(
            inverse,
            weights=weights,
            minlength=basis.shape[0],
        )
    constraint = counts @ basis
    if np.linalg.norm(constraint) < 1e-12:
        return constraint_projection

    q, _ = np.linalg.qr(constraint.reshape(-1, 1), mode="complete")
    z = q[:, 1:]
    return z if constraint_projection is None else constraint_projection @ z


def apply_identifiability(
    *,
    omega: NDArray,
    projection: NDArray | None,
    projection_ident: NDArray | None,
    absorbs_intercept: bool,
) -> tuple[NDArray, int, NDArray | None]:
    """Apply identifiability projection to the penalty matrix when needed."""
    if not absorbs_intercept:
        return omega, omega.shape[0], projection

    if projection_ident is projection:
        return omega, omega.shape[0], projection

    if projection is not None:
        z = projection.T @ projection_ident
    else:
        z = projection_ident
    omega_ident = z.T @ omega @ z
    return omega_ident, omega_ident.shape[0], projection_ident


def build_identifiability_projection_for_spec(
    spec: Any,
    x: NDArray,
    constraint_projection: NDArray | None,
    geometry_weight: NDArray | None = None,
) -> NDArray | None:
    """Compute the identifiability projection for a spline spec."""
    x = np.asarray(x, dtype=np.float64).ravel()
    support, inverse = np.unique(x, return_inverse=True)
    basis = spec._basis_matrix(support).toarray()
    if constraint_projection is not None:
        basis = basis @ constraint_projection

    return build_identifiability_projection(
        inverse=inverse,
        basis=basis,
        constraint_projection=constraint_projection,
        n_basis=spec._n_basis,
        geometry_weight=geometry_weight,
    )


def apply_identifiability_for_spec(
    spec: Any,
    x: NDArray,
    omega: NDArray,
    projection: NDArray | None,
    geometry_weight: NDArray | None = None,
) -> tuple[NDArray, int, NDArray | None]:
    """Apply identifiability projection for a spline spec."""
    projection_ident = build_identifiability_projection_for_spec(
        spec,
        x,
        projection,
        geometry_weight,
    )
    return apply_identifiability(
        omega=omega,
        projection=projection,
        projection_ident=projection_ident,
        absorbs_intercept=spec.absorbs_intercept,
    )


__all__ = [
    "apply_identifiability",
    "apply_identifiability_for_spec",
    "build_identifiability_projection",
    "build_identifiability_projection_for_spec",
]
