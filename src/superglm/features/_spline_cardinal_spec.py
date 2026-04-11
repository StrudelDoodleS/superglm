"""Cardinal CR spline spec helpers."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from superglm.features import _spline_cardinal, _spline_knots, _spline_runtime


def place_knots(spec: Any, x: NDArray) -> None:
    """Place K = n_knots + 2 cardinal knots and build the penalty matrices."""
    x = np.asarray(x, dtype=np.float64).ravel()
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

    spec._cr_knots = np.concatenate([[spec._lo], interior, [spec._hi]])
    spec._n_basis = len(spec._cr_knots)
    spec._knots = spec._cr_knots
    build_cr_matrices(spec)


def build_cr_matrices(spec: Any) -> None:
    """Build and store the tridiagonal system matrices for the cardinal CR spline."""
    spec._cr_M, spec._cr_S = _spline_cardinal.build_cr_penalty_matrices(spec._cr_knots)


def cardinal_boundary_slopes(spec: Any) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Return basis value and slope at the boundary knots for linear extrapolation."""
    return cast(
        tuple[NDArray, NDArray, NDArray, NDArray],
        _spline_cardinal.cardinal_boundary_slopes(spec._cr_knots, spec._cr_M),
    )


def basis_matrix(spec: Any, x: NDArray) -> sp.csr_matrix:
    """Evaluate the cardinal CR basis at data points."""
    x_eval, extrapolate = _spline_runtime.prepare_eval_points(spec, x)
    if not extrapolate:
        return eval_cardinal_basis(spec, x_eval)
    return linear_tail_cardinal_basis(spec, x_eval)


def raw_basis_matrix(spec: Any, x: NDArray) -> NDArray:
    """Evaluate the raw cardinal basis at points clipped to training range."""
    x = np.asarray(x, dtype=np.float64).ravel()
    x_clip = np.clip(x, spec._lo, spec._hi)
    return eval_cardinal_basis(spec, x_clip).toarray()


def linear_tail_cardinal_basis(spec: Any, x: NDArray) -> sp.csr_matrix:
    """Evaluate cardinal basis with natural-spline linear tails outside range."""
    return _spline_cardinal.linear_tail_cardinal_basis(
        x,
        lo=spec._lo,
        hi=spec._hi,
        knots=spec._cr_knots,
        M=spec._cr_M,
    )


def eval_cardinal_basis(spec: Any, x: NDArray) -> sp.csr_matrix:
    """Evaluate the cardinal cubic regression spline basis."""
    return _spline_cardinal.eval_cardinal_basis(x, spec._cr_knots, spec._cr_M)


def reconstruct(spec: Any, beta: NDArray, n_points: int = 200) -> dict[str, Any]:
    """Reconstruct a cardinal spline curve on a regular grid."""
    beta_orig = spec._R_inv @ beta if spec._R_inv is not None else beta
    x_grid = np.linspace(spec._lo, spec._hi, n_points)
    basis_grid = basis_matrix(spec, x_grid).toarray()
    log_rels = basis_grid @ beta_orig
    assert spec._cr_knots is not None
    return {
        "x": x_grid,
        "log_relativity": log_rels,
        "relativity": np.exp(log_rels),
        "knots_interior": spec._cr_knots[1:-1],
        "coefficients_original": beta_orig,
    }
