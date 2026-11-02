"""Subclass-specific spline helper operations."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm.features import _spline_constraints
from superglm.types import LinearConstraintSet


def assemble_open_knot_vector(spec: Any, interior: NDArray) -> None:
    """Build the padded open knot vector used by unconstrained B-splines."""
    x_range = spec._hi - spec._lo
    lo_effective = spec._lo - 0.001 * x_range
    hi_effective = spec._hi + 0.001 * x_range

    inner = np.concatenate([[lo_effective], interior, [hi_effective]])
    dx_lo = inner[1] - inner[0]
    dx_hi = inner[-1] - inner[-2]

    lower = lo_effective - dx_lo * np.arange(spec.degree, 0, -1)
    upper = hi_effective + dx_hi * np.arange(1, spec.degree + 1)

    spec._knots = np.concatenate([lower, inner, upper])
    spec._n_basis = len(spec._knots) - spec.degree - 1


def assemble_clamped_knot_vector(spec: Any, interior: NDArray) -> None:
    """Build the exact-boundary clamped knot vector used by cubic regression splines."""
    spec._knots = np.concatenate(
        [
            np.repeat(spec._lo, spec.degree + 1),
            interior,
            np.repeat(spec._hi, spec.degree + 1),
        ]
    )
    spec._n_basis = len(spec._knots) - spec.degree - 1


def build_shape_constraints_raw(spec: Any):
    """Build fit-time shape constraints on raw spline coefficients."""
    kind = spec.constraint_kind
    if kind in {"increasing", "decreasing"}:
        return _spline_constraints.build_monotone_difference_constraints(spec._n_basis, kind)
    if kind in {"convex", "concave"}:
        constraints = _spline_constraints.build_curvature_difference_constraints(
            spec._knots,
            spec.degree,
            kind,
            domain=(spec._lo, spec._hi),
        )
        if getattr(spec, "_Z", None) is not None:
            # Natural boundary conditions already impose the first and last
            # exact curvature rows as equalities.  Keeping their O(eps)
            # projections as QP inequalities creates scale-dependent fake
            # active constraints.
            interior = constraints.A[1:-1]
            return LinearConstraintSet(A=interior, b=np.zeros(interior.shape[0]))
        return constraints
    raise ValueError(f"Unsupported fit-time shape kind: {kind!r}")
