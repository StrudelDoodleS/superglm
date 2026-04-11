"""Runtime evaluation helpers for spline feature specs."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline as BSpl

from superglm.features import _spline_extrapolation, _spline_knots


def prepare_eval_points(spec: Any, x: NDArray) -> tuple[NDArray, bool]:
    """Apply the configured extrapolation policy for basis evaluation."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if spec.extrapolation == "clip":
        return np.clip(x, spec._lo, spec._hi), False
    if spec.extrapolation == "extend":
        return x, True

    scale = max(1.0, abs(spec._lo), abs(spec._hi), abs(spec._hi - spec._lo))
    tol = 1e-12 * scale
    lo_mask = x < (spec._lo - tol)
    hi_mask = x > (spec._hi + tol)
    if np.any(lo_mask) or np.any(hi_mask):
        raise ValueError(
            f"Spline received values outside training range "
            f"[{spec._lo:.6g}, {spec._hi:.6g}] with extrapolation='error'."
        )
    return x, False


def basis_matrix(spec: Any, x: NDArray):
    """Evaluate the raw B-spline basis under the extrapolation policy."""
    x_eval, extrapolate = prepare_eval_points(spec, x)
    return BSpl.design_matrix(x_eval, spec._knots, spec.degree, extrapolate=extrapolate)


def raw_basis_matrix(spec: Any, x: NDArray) -> NDArray:
    """Evaluate the raw (pre-projection) basis at points clipped to training range."""
    x = np.asarray(x, dtype=np.float64).ravel()
    x_clip = np.clip(x, spec._lo, spec._hi)
    return BSpl.design_matrix(x_clip, spec._knots, spec.degree, extrapolate=False).toarray()


def basis_value_and_slope_at(spec: Any, x0: float) -> tuple[NDArray, NDArray]:
    """Return the raw basis row and its first derivative at ``x0``."""
    return cast(
        tuple[NDArray, NDArray],
        _spline_extrapolation.basis_value_and_slope_at(
            x0,
            knots=spec._knots,
            degree=spec.degree,
            n_basis=spec._n_basis,
        ),
    )


def boundary_linear_rows(spec: Any) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Cache basis value/slope rows for linear continuation at the boundaries."""
    if spec._basis_lo is None or spec._basis_d1_lo is None:
        (
            spec._basis_lo,
            spec._basis_d1_lo,
            spec._basis_hi,
            spec._basis_d1_hi,
        ) = _spline_extrapolation.boundary_linear_rows(
            knots=spec._knots,
            degree=spec.degree,
            n_basis=spec._n_basis,
            lo=spec._lo,
            hi=spec._hi,
        )
        return spec._basis_lo, spec._basis_d1_lo, spec._basis_hi, spec._basis_d1_hi
    if spec._basis_hi is None or spec._basis_d1_hi is None:
        spec._basis_hi, spec._basis_d1_hi = basis_value_and_slope_at(spec, spec._hi)
    return spec._basis_lo, spec._basis_d1_lo, spec._basis_hi, spec._basis_d1_hi


def linear_tail_basis_matrix(spec: Any, x: NDArray):
    """Evaluate the raw basis with explicit linear continuation outside the fit range."""
    basis_lo, slope_lo, basis_hi, slope_hi = boundary_linear_rows(spec)
    return _spline_extrapolation.linear_tail_basis_matrix(
        x,
        knots=spec._knots,
        degree=spec.degree,
        n_basis=spec._n_basis,
        lo=spec._lo,
        hi=spec._hi,
        basis_lo=basis_lo,
        slope_lo=slope_lo,
        basis_hi=basis_hi,
        slope_hi=slope_hi,
    )


def place_knots(spec: Any, x: NDArray) -> None:
    """Place interior knots and build the full knot vector."""
    if spec._explicit_boundary is not None:
        spec._lo, spec._hi = spec._explicit_boundary
    else:
        spec._lo, spec._hi = float(x.min()), float(x.max())
    spec._basis_lo = None
    spec._basis_hi = None
    spec._basis_d1_lo = None
    spec._basis_d1_hi = None

    interior, spec._knot_strategy_actual = _spline_knots.resolve_interior_knots(
        x,
        lo=spec._lo,
        hi=spec._hi,
        n_knots=spec.n_knots,
        knot_strategy=spec.knot_strategy,
        knot_alpha=spec.knot_alpha,
        explicit_knots=spec._explicit_knots,
        explicit_boundary=spec._explicit_boundary,
    )
    spec._assemble_knot_vector(interior)


def assemble_clamped_knot_vector(spec: Any, interior: NDArray) -> None:
    """Build the default clamped knot vector from interior knots."""
    pad = (spec._hi - spec._lo) * 1e-6
    spec._knots = np.concatenate(
        [
            np.repeat(spec._lo - pad, spec.degree + 1),
            interior,
            np.repeat(spec._hi + pad, spec.degree + 1),
        ]
    )
    spec._n_basis = len(spec._knots) - spec.degree - 1


def transform(spec: Any, x: NDArray) -> NDArray:
    """Build the design matrix using knots learned during build()."""
    basis = spec._basis_matrix(x).toarray()
    scop_sigma = getattr(spec, "_scop_Sigma", None)
    if scop_sigma is not None:
        x_sigma = basis @ scop_sigma
        return x_sigma[:, 1:] - getattr(spec, "_scop_col_means")
    if spec._R_inv is not None:
        basis = basis @ spec._R_inv
    return basis


def reconstruct(spec: Any, beta: NDArray, n_points: int = 200) -> dict[str, Any]:
    """Reconstruct a spline curve on a regular grid."""
    x_grid = np.linspace(spec._lo, spec._hi, n_points)
    scop_sigma = getattr(spec, "_scop_Sigma", None)
    if scop_sigma is not None:
        basis_grid = transform(spec, x_grid)
        log_rels = basis_grid @ beta
        beta_orig = beta
    else:
        beta_orig = spec._R_inv @ beta if spec._R_inv is not None else beta
        basis_grid = spec._basis_matrix(x_grid).toarray()
        log_rels = basis_grid @ beta_orig
    return {
        "x": x_grid,
        "log_relativity": log_rels,
        "relativity": np.exp(log_rels),
        "knots_interior": spec._knots[spec.degree + 1 : -(spec.degree + 1)],
        "coefficients_original": beta_orig,
    }
