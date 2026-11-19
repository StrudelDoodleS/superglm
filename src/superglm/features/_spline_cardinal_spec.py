"""Cardinal CR spline spec helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from superglm.features import _spline_cardinal, _spline_knots, _spline_runtime


def place_knots(
    spec: Any,
    x: NDArray,
    sample_weight: NDArray | None = None,
) -> None:
    """Place K = n_knots + 2 cardinal knots and build the penalty matrices."""
    x, sample_weight = _spline_knots.knot_geometry_data(x, sample_weight)
    if spec._explicit_boundary is not None:
        spec._lo, spec._hi = spec._explicit_boundary
    else:
        spec._lo, spec._hi = float(x.min()), float(x.max())
    spec._require_range()
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
        sample_weight=sample_weight,
    )
    # The cardinal system needs distinct knots: a quantile knot on the boundary
    # (all rows at one end of an explicit range) falls back to uniform knots.
    if spec._knot_strategy_actual in _spline_knots.QUANTILE_STRATEGIES and not np.all(
        (interior > spec._lo) & (interior < spec._hi)
    ):
        interior = np.linspace(spec._lo, spec._hi, spec.n_knots + 2)[1:-1]
        spec._knot_strategy_actual = "uniform"

    spec._cr_knots = np.concatenate([[spec._lo], interior, [spec._hi]])
    spec._n_basis = len(spec._cr_knots)
    spec._knots = spec._cr_knots
    build_cr_matrices(spec)


def build_cr_matrices(spec: Any) -> None:
    """Build and store the tridiagonal system matrices for the cardinal CR spline."""
    spec._cr_M, spec._cr_S = _spline_cardinal.build_cr_penalty_matrices(spec._cr_knots)


def structural_penalty(spec: Any) -> NDArray:
    """The cardinal penalty's null space without its knot-spacing spread."""
    return _spline_cardinal.structural_cr_penalty(spec._cr_knots)


def basis_matrix(spec: Any, x: NDArray) -> sp.csr_matrix:
    """Evaluate the cardinal CR basis at data points."""
    x_eval, extrapolate = _spline_runtime.prepare_eval_points(spec, x)
    if not extrapolate:
        return eval_cardinal_basis(spec, x_eval)
    return linear_tail_cardinal_basis(spec, x_eval)


def raw_basis_matrix(spec: Any, x: NDArray) -> NDArray:
    """Evaluate the raw cardinal basis under the extrapolation policy, dense."""
    x_eval, extrapolate = _spline_runtime.prepare_eval_points(spec, x)
    if not extrapolate:
        return _spline_cardinal.eval_cardinal_dense(x_eval, spec._cr_knots, spec._cr_M)
    return linear_tail_cardinal_basis(spec, x_eval).toarray()


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
