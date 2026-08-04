"""Private constraint helpers for spline feature specs."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline as BSpl

from superglm.types import LinearConstraintSet


def curvature_difference_operator(
    knots: NDArray,
    degree: int,
    *,
    domain: tuple[float, float] | None = None,
    normalize: bool = True,
) -> NDArray:
    """Return the linear operator controlling B-spline curvature.

    The first-derivative B-spline coefficients are proportional to
    ``(c[i + 1] - c[i]) / (t[i + degree + 1] - t[i + 1])``.  Their successive
    differences control the represented spline's second derivative.

    When ``domain`` is supplied, the rows are exact for convexity/concavity on
    that closed interval for degrees one through three: slope jumps for a
    piecewise-linear spline, one probe per constant-curvature span for degree
    two, and every endpoint of the piecewise-linear second derivative for
    cubic splines.  Without ``domain``, the derivative-coefficient difference
    operator over the complete active knot interval is returned.

    The knot geometry is normalized before differentiation, and each result
    row is normalized with a max-first norm.  Both operations preserve signs
    while avoiding overflow/underflow for valid extreme predictor scales.
    """
    knots = np.asarray(knots, dtype=np.float64)
    if knots.ndim != 1 or not np.all(np.isfinite(knots)):
        raise ValueError("Curvature constraints require a finite one-dimensional knot vector")
    if not isinstance(degree, int) or degree < 1:
        raise ValueError("Curvature constraints require degree >= 1")
    n_basis = len(knots) - degree - 1
    if n_basis < 2:
        raise ValueError("Curvature constraints require at least 2 basis functions")
    if np.any(np.diff(knots) < 0.0):
        raise ValueError("Curvature constraints require a non-decreasing knot vector")

    active_lo = float(knots[degree])
    active_hi = float(knots[n_basis])
    active_span = active_hi - active_lo
    if not np.isfinite(active_span) or active_span <= 0.0:
        raise ValueError("Curvature constraints require a positive active knot interval")
    normalized_knots = (knots - active_lo) / active_span

    spans = normalized_knots[degree + 1 : degree + n_basis] - normalized_knots[1:n_basis]
    # Degree-two and cubic domain rows are exact second-derivative probes and
    # never divide by these spans.  A knot placed on the fitted boundary of a
    # clamped vector repeats it ``degree + 2`` times and so zeroes a span, but
    # that only invalidates the derivative-coefficient operator, which those
    # paths do not build.
    uses_derivative_coefficients = domain is None or degree == 1
    if spans.shape != (n_basis - 1,) or (uses_derivative_coefficients and np.any(spans <= 0.0)):
        raise ValueError("Curvature constraints require positive derivative knot spans")

    if uses_derivative_coefficients:
        first_derivative = np.diff(np.eye(n_basis), axis=0) / spans[:, None]
        curvature = np.diff(first_derivative, axis=0)
    else:
        curvature = np.zeros((0, n_basis), dtype=np.float64)

    if domain is not None:
        if degree > 3:
            raise NotImplementedError(
                "Exact fit-time curvature constraints are implemented only for degree <= 3"
            )
        lo, hi = (float(value) for value in domain)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            raise ValueError("Curvature constraint domain must be finite and increasing")
        lo_normalized = (lo - active_lo) / active_span
        hi_normalized = (hi - active_lo) / active_span
        tolerance = 64.0 * np.finfo(np.float64).eps
        if lo_normalized < -tolerance or hi_normalized > 1.0 + tolerance:
            raise ValueError("Curvature constraint domain must lie in the active knot interval")

        interior = np.unique(
            normalized_knots[
                (normalized_knots > lo_normalized) & (normalized_knots < hi_normalized)
            ]
        )
        if degree == 1:
            jump_points = normalized_knots[2:n_basis]
            keep = (jump_points > lo_normalized) & (jump_points < hi_normalized)
            curvature = curvature[keep]
        else:
            breakpoints = np.concatenate(([lo_normalized], interior, [hi_normalized]))
            points = 0.5 * (breakpoints[:-1] + breakpoints[1:]) if degree == 2 else breakpoints
            curvature = np.asarray(
                BSpl(
                    normalized_knots,
                    np.eye(n_basis, dtype=np.float64),
                    degree,
                    extrapolate=False,
                )(points, nu=2),
                dtype=np.float64,
            )

    if not normalize or curvature.shape[0] == 0:
        return curvature

    row_max = np.max(np.abs(curvature), axis=1)
    if np.any(~np.isfinite(row_max)) or np.any(row_max <= 0.0):
        raise ValueError("Curvature constraint rows must be finite and non-zero")
    max_scaled = curvature / row_max[:, None]
    relative_norms = np.linalg.norm(max_scaled, axis=1)
    if np.any(~np.isfinite(relative_norms)) or np.any(relative_norms <= 0.0):
        raise ValueError("Curvature constraint rows must be finite and non-zero")
    return max_scaled / relative_norms[:, None]


def build_monotone_difference_constraints(
    n_basis: int, monotone: str | None
) -> LinearConstraintSet:
    """Build first-difference monotonicity constraints on raw spline coefficients."""
    D = np.diff(np.eye(n_basis), axis=0)
    if monotone == "decreasing":
        D = -D
    return LinearConstraintSet(A=D, b=np.zeros(n_basis - 1))


def build_curvature_difference_constraints(
    knots: NDArray,
    degree: int,
    kind: str,
    *,
    domain: tuple[float, float] | None = None,
) -> LinearConstraintSet:
    """Build knot-spacing-aware curvature constraints on raw coefficients.

    Adjacent first-derivative coefficients are proportional to
    ``(c[i + 1] - c[i]) / (t[i + degree + 1] - t[i + 1])``. Requiring their
    successive differences to have one sign therefore constrains curvature
    correctly for non-uniform as well as uniform knot vectors. For degree-one
    splines the same rows constrain the slope jumps at interior knots.  A
    supplied ``domain`` replaces padded endpoint-control rows with exact
    public-boundary rows for degree-two and cubic splines.
    """
    n_basis = len(np.asarray(knots)) - degree - 1
    if n_basis < 3:
        raise ValueError("Curvature constraints require degree >= 1 and at least 3 basis functions")
    D2 = curvature_difference_operator(knots, degree, domain=domain)
    if kind == "concave":
        D2 = -D2
    return LinearConstraintSet(A=D2, b=np.zeros(D2.shape[0]))


def build_natural_constraint_null_space(
    knots: NDArray,
    degree: int,
    *,
    lo: float,
    hi: float,
) -> NDArray:
    """Compute the null space of natural-boundary spline constraints."""
    n_basis = len(knots) - degree - 1
    C = np.zeros((2, n_basis))
    for j in range(n_basis):
        c = np.zeros(n_basis)
        c[j] = 1.0
        spl = BSpl(knots, c, degree)
        C[0, j] = spl(lo, nu=2)
        C[1, j] = spl(hi, nu=2)
    Q, _ = np.linalg.qr(C.T, mode="complete")
    return Q[:, 2:]


__all__ = [
    "build_curvature_difference_constraints",
    "build_monotone_difference_constraints",
    "build_natural_constraint_null_space",
    "curvature_difference_operator",
]
