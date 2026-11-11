"""Independent optimality check for a post-fit shape projection (a test oracle).

A post-fit repair minimises ``||W^(1/2) (B beta - t)||^2`` over the coefficients
whose spline obeys the shape on the whole fitted range. That is a convex program
with one linear inequality per point of the range, so ``beta`` is its solution
exactly when it is feasible and satisfies the KKT conditions of convex
semi-infinite programming (Hettich & Kortanek, *Semi-infinite programming*, SIAM
Review 35(3), 1993):

    H beta - g = sum_i mu_i r(x_i),   mu_i >= 0,

where ``H = B'WB``, ``g = B'Wt`` and ``r(x_i)`` is the row of the constrained
derivative at a point ``x_i`` where that derivative touches zero. Feasibility is
read off the exact piecewise polynomial (``scipy.interpolate.PPoly`` roots), and
the multipliers come from ``scipy.optimize.nnls``; neither shares code with the
repair's own QP.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline, PPoly
from scipy.optimize import nnls


@dataclass(frozen=True)
class ProjectionCertificate:
    """What the oracle read off one repaired term."""

    #: Lowest signed derivative over the range, relative to its largest magnitude.
    minimum_derivative: float
    #: KKT residual ``||H beta - g - sum mu_i r_i|| / ||g||`` at the best ``mu >= 0``.
    kkt_residual: float
    #: Weighted squared distance of the repaired curve from the fitted one.
    loss: float


def _order_and_sign(kind: str) -> tuple[int, float]:
    order = 1 if kind in {"increasing", "decreasing"} else 2
    return order, -1.0 if kind in {"decreasing", "concave"} else 1.0


def _raw(spec, beta):
    return spec._R_inv @ beta if spec._R_inv is not None else beta


def projection_certificate(spec, kind, x_grid, target, weights, beta, touch=1e-7):
    """Certify ``beta`` as the weighted projection of ``target`` onto the ``kind`` cone.

    ``touch`` is the relative derivative level below which a point counts as one
    where the constraint holds with equality.
    """
    order, sign = _order_and_sign(kind)
    beta = np.asarray(beta, dtype=np.float64)
    derivative = PPoly.from_spline(
        BSpline(spec._knots, _raw(spec, beta), spec.degree, extrapolate=False)
    ).derivative(order)
    lo, hi = float(spec._lo), float(spec._hi)
    breaks = derivative.x[(derivative.x >= lo) & (derivative.x <= hi)]
    # Where the derivative is stationary or exactly zero: a constraint held at a
    # point within rounding of a span's minimum touches at its zeros, not the minimum.
    special = np.concatenate(
        (
            derivative.derivative().roots(discontinuity=False, extrapolate=False),
            derivative.roots(discontinuity=False, extrapolate=False),
        )
    )
    special = special[np.isfinite(special) & (special >= lo) & (special <= hi)]
    candidates = np.unique(np.concatenate(([lo, hi], breaks, special)))
    values = sign * derivative(candidates)
    scale = float(np.max(np.abs(sign * derivative(np.linspace(lo, hi, 2001)))))
    scale = scale if scale > 0.0 else 1.0

    basis = np.asarray(spec.transform(x_grid), dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    H = basis.T @ (basis * w[:, None])
    g = basis.T @ (w * np.asarray(target, dtype=np.float64))
    touching = candidates[values <= touch * scale]
    # On a span the projection holds flat every point touches zero, and the
    # multipliers may sit anywhere in it, so the span is sampled densely.
    spans = np.unique(np.concatenate(([lo, hi], breaks)))
    for start, end in zip(spans[:-1], spans[1:], strict=True):
        span = np.linspace(start, end, 129)
        if end > start and np.max(np.abs(derivative(span))) <= touch * scale:
            touching = np.concatenate((touching, span))
    n_basis = len(spec._knots) - spec.degree - 1
    rows = sign * BSpline(spec._knots, np.eye(n_basis), spec.degree, extrapolate=False).derivative(
        order
    )(touching)
    rows = rows @ spec._R_inv if spec._R_inv is not None else rows
    if touching.size:
        _, residual = nnls(rows.T, H @ beta - g, maxiter=100 * touching.size)
    else:
        residual = float(np.linalg.norm(H @ beta - g))
    curve = basis @ beta
    return ProjectionCertificate(
        minimum_derivative=float(np.min(values)) / scale,
        kkt_residual=float(residual) / float(np.linalg.norm(g)),
        loss=float(w @ (curve - target) ** 2),
    )
