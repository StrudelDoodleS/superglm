"""Post-fit shape repair for 1-D spline terms."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from superglm._roundoff import gamma

if TYPE_CHECKING:
    from superglm.features.spline import _SplineBase
    from superglm.types import GroupSlice


@dataclass
class MonotoneRepairResult:
    """Result of a post-fit shape repair for one spline feature."""

    feature_name: str
    direction: str  # "increasing" | "decreasing" | "convex" | "concave"
    grid: NDArray  # (n_grid,) evaluation points
    original_log_effect: NDArray  # (n_grid,) pre-repair curve
    repaired_log_effect: NDArray  # (n_grid,) post-repair curve
    repaired_beta_reparam: NDArray  # full feature beta in reparametrized space
    max_violation_before: float
    max_violation_after: float
    projection_residual: float  # maximum curve change induced by coefficient projection

    @property
    def kind(self) -> str:
        """Canonical shape token for compatibility with generalized postfit repair."""
        return self.direction


@dataclass(frozen=True)
class ShapeConstraintCertificate:
    """Span-wise certificate for one fitted spline shape constraint."""

    kind: str
    minimum_signed_derivative: float
    minimum_scaled_slack: float
    worst_x: float
    maximum_row_norm: float

    @property
    def violation(self) -> float:
        return max(0.0, -self.minimum_signed_derivative)


def _shape_order_and_sign(kind: str) -> tuple[int, float]:
    if kind in {"increasing", "decreasing"}:
        return 1, -1.0 if kind == "decreasing" else 1.0
    if kind in {"convex", "concave"}:
        return 2, -1.0 if kind == "concave" else 1.0
    raise ValueError(f"Unsupported shape kind: {kind!r}")


def _raw_shape_coefficients(spec: _SplineBase, beta: NDArray) -> NDArray:
    beta_arr = np.asarray(beta, dtype=np.float64)
    scop_sigma = getattr(spec, "_scop_Sigma", None)
    if scop_sigma is not None:
        drop_dim = int(getattr(spec, "_scop_null_dim", 1))
        return np.asarray(
            np.asarray(scop_sigma, dtype=np.float64)[:, drop_dim:] @ beta_arr,
            dtype=np.float64,
        )
    r_inv = getattr(spec, "_R_inv", None)
    return beta_arr if r_inv is None else np.asarray(r_inv @ beta_arr, dtype=np.float64)


def _shape_polynomial(spec: _SplineBase, beta: NDArray):
    """Return the exact piecewise polynomial represented by fitted coefficients."""
    from scipy.interpolate import BSpline, CubicSpline, PPoly

    raw_beta = _raw_shape_coefficients(spec, beta)
    cardinal_knots = getattr(spec, "_cr_knots", None)
    if cardinal_knots is not None:
        return CubicSpline(
            np.asarray(cardinal_knots, dtype=np.float64),
            raw_beta,
            bc_type="natural",
            extrapolate=False,
        )
    spline = BSpline(
        np.asarray(spec._knots, dtype=np.float64),
        raw_beta,
        int(spec.degree),
        extrapolate=False,
    )
    return PPoly.from_spline(spline)


def _shape_breakpoints(spec: _SplineBase) -> NDArray:
    cardinal_knots = getattr(spec, "_cr_knots", None)
    knots = cardinal_knots if cardinal_knots is not None else spec._knots
    knots_arr = np.asarray(knots, dtype=np.float64)
    return np.unique(knots_arr[(knots_arr >= spec._lo) & (knots_arr <= spec._hi)])


def _uses_span_jump_constraints(spec: _SplineBase, kind: str) -> bool:
    order, _ = _shape_order_and_sign(kind)
    if getattr(spec, "_cr_knots", None) is not None:
        return False
    return (order == 1 and int(spec.degree) == 0) or (order == 2 and int(spec.degree) == 1)


def _has_natural_curvature_boundaries(spec: _SplineBase, kind: str) -> bool:
    """Whether endpoint curvature is structurally fixed to zero."""
    from superglm.features._spline_subclass_ops import has_natural_ends

    return kind in {"convex", "concave"} and (
        has_natural_ends(spec) or getattr(spec, "_cr_M", None) is not None
    )


def _raw_basis(spec: _SplineBase, exponent: int = 0):
    """The raw basis as a spline in ``x * 2**exponent``, exact for a power of two."""
    from scipy.interpolate import BSpline, CubicSpline

    cardinal_knots = getattr(spec, "_cr_knots", None)
    if cardinal_knots is not None:
        return CubicSpline(
            np.ldexp(np.asarray(cardinal_knots, dtype=np.float64), exponent),
            np.eye(int(spec._n_basis), dtype=np.float64),
            axis=0,
            bc_type="natural",
            extrapolate=False,
        )
    return BSpline(
        np.ldexp(np.asarray(spec._knots, dtype=np.float64), exponent),
        np.eye(int(spec._n_basis), dtype=np.float64),
        int(spec.degree),
        extrapolate=False,
    )


def _raw_derivative(spec: _SplineBase, basis, points: NDArray, order: int) -> NDArray:
    if getattr(spec, "_cr_knots", None) is None and order > int(spec.degree):
        return np.zeros((points.size, int(spec._n_basis)), dtype=np.float64)
    return np.asarray(basis(points, nu=order), dtype=np.float64)


def shape_derivative_matrix(spec: _SplineBase, x: NDArray, order: int) -> NDArray:
    """Evaluate exact fitted-basis derivatives at ``x``."""
    points = np.asarray(x, dtype=np.float64).ravel()
    return _to_fitted(spec, _raw_derivative(spec, _raw_basis(spec), points, order))


def _to_fitted(spec: _SplineBase, raw: NDArray) -> NDArray:
    """Raw-basis rows in the fitted coefficients."""
    scop_sigma = getattr(spec, "_scop_Sigma", None)
    if scop_sigma is not None:
        drop_dim = int(getattr(spec, "_scop_null_dim", 1))
        fitted = raw @ np.asarray(scop_sigma, dtype=np.float64)[:, drop_dim:]
    else:
        r_inv = getattr(spec, "_R_inv", None)
        fitted = raw if r_inv is None else raw @ r_inv
    return np.asarray(fitted, dtype=np.float64)


def _shape_span_jump_matrix(spec: _SplineBase, points: NDArray, order: int) -> NDArray:
    """Return exact one-sided derivative jumps from adjacent polynomial spans.

    ``nextafter`` cannot represent an interior probe when two knots are adjacent
    floating-point numbers.  Evaluating the derivative spline at each span's
    left endpoint instead gives its exact right-continuous polynomial piece and
    therefore preserves even a one-ULP-wide span.
    """
    from scipy.interpolate import BSpline

    knots = np.asarray(spec._knots, dtype=np.float64)
    n_basis = len(knots) - int(spec.degree) - 1
    raw_basis = BSpline(
        knots,
        np.eye(n_basis, dtype=np.float64),
        int(spec.degree),
        extrapolate=False,
    )
    if order:
        raw_basis = raw_basis.derivative(order)

    rows = np.zeros((len(points), n_basis), dtype=np.float64)
    for row_index, point in enumerate(np.asarray(points, dtype=np.float64)):
        if not np.any(knots == point):
            continue
        previous_knots = knots[knots < point]
        following_knots = knots[knots > point]
        if previous_knots.size == 0 or following_knots.size == 0:
            continue
        previous_span_start = float(previous_knots[-1])
        left = np.asarray(raw_basis(previous_span_start), dtype=np.float64)
        right = np.asarray(raw_basis(float(point)), dtype=np.float64)
        rows[row_index] = right - left

    scop_sigma = getattr(spec, "_scop_Sigma", None)
    if scop_sigma is not None:
        drop_dim = int(getattr(spec, "_scop_null_dim", 1))
        fitted = rows @ np.asarray(scop_sigma, dtype=np.float64)[:, drop_dim:]
    else:
        r_inv = getattr(spec, "_R_inv", None)
        fitted = rows if r_inv is None else rows @ r_inv
    return np.asarray(fitted, dtype=np.float64)


def _shape_constraint_rows(spec: _SplineBase, points: NDArray, kind: str) -> NDArray:
    order, sign = _shape_order_and_sign(kind)
    if order == 1 and int(spec.degree) == 0 and getattr(spec, "_cr_knots", None) is None:
        # A degree-zero spline is piecewise constant.  Its classical first
        # derivative vanishes inside every span, so monotonicity lives entirely
        # in the one-sided jumps at the knots.
        return sign * _shape_span_jump_matrix(spec, points, 0)
    if order == 2 and int(spec.degree) == 1 and getattr(spec, "_cr_knots", None) is None:
        # Piecewise-linear convexity/concavity lives in the slope jumps;
        # the classical second derivative is identically zero inside spans.
        return sign * _shape_span_jump_matrix(spec, points, 1)
    rows = sign * shape_derivative_matrix(spec, points, order)
    if order == 2 and _has_natural_curvature_boundaries(spec, kind):
        # Natural spline boundary curvature is structurally zero. The dense
        # null-space projection can leave an O(eps) residual in these rows;
        # normalizing that numerical zero would turn it into an O(1) signed
        # certificate slack. Keep the adjacent interior spans in the
        # certificate, but represent the exact endpoint equalities as zero.
        lo = float(getattr(spec, "_lo"))
        hi = float(getattr(spec, "_hi"))
        boundary = (points == lo) | (points == hi)
        if np.any(boundary):
            rows = rows.copy()
            rows[boundary] = 0.0
    return rows


def _certificate_candidates(spec: _SplineBase, beta: NDArray, kind: str) -> NDArray:
    """Return every possible within-span extremum of the constrained derivative."""
    order, _ = _shape_order_and_sign(kind)
    if _uses_span_jump_constraints(spec, kind):
        return _shape_breakpoints(spec)
    polynomial = _shape_polynomial(spec, beta)
    derivative = polynomial.derivative(order)
    stationary = derivative.derivative().roots(
        discontinuity=False,
        extrapolate=False,
    )
    stationary = np.asarray(stationary, dtype=np.float64).ravel()
    stationary = stationary[
        np.isfinite(stationary) & (stationary >= spec._lo) & (stationary <= spec._hi)
    ]
    breakpoints = _shape_breakpoints(spec)
    # Include both one-sided limits.  Cubic public splines are C2, but this
    # also certifies lower-degree and repeated-knot bases without assuming
    # derivative continuity at an interior breakpoint.
    one_sided: list[float] = []
    for point in breakpoints[1:-1]:
        left = np.nextafter(point, spec._lo)
        right = np.nextafter(point, spec._hi)
        if left >= spec._lo:
            one_sided.append(float(left))
        if right <= spec._hi:
            one_sided.append(float(right))
    return np.unique(
        np.concatenate(
            (
                [spec._lo, spec._hi],
                breakpoints,
                np.asarray(one_sided, dtype=np.float64),
                stationary,
            )
        )
    )


def _normalized_nonzero_shape_rows(
    rows: NDArray,
) -> tuple[NDArray, NDArray, NDArray]:
    """Normalize meaningful constraint rows without imposing predictor units.

    Derivative rows carry inverse powers of the predictor's units.  An absolute
    or cross-span relative floor can therefore erase valid geometry.  Retain
    every locally nonzero row and normalize in two stages so squaring very small
    row entries cannot underflow.
    """
    rows_arr = np.asarray(rows, dtype=np.float64)
    if rows_arr.ndim != 2:
        raise RuntimeError("Shape constraint geometry must be a two-dimensional matrix")
    if not np.all(np.isfinite(rows_arr)):
        raise RuntimeError("Shape constraint geometry contains non-finite values")
    if rows_arr.shape[0] == 0:
        return rows_arr.copy(), np.empty(0, dtype=np.float64), np.zeros(0, dtype=bool)

    row_max = np.max(np.abs(rows_arr), axis=1)
    geometry_scale = float(np.max(row_max))
    if geometry_scale == 0.0:
        return (
            np.empty((0, rows_arr.shape[1]), dtype=np.float64),
            np.zeros(rows_arr.shape[0], dtype=np.float64),
            np.zeros(rows_arr.shape[0], dtype=bool),
        )

    # Each polynomial span has its own natural derivative scale.  A narrow
    # but valid span can differ from its neighbour by many orders of magnitude,
    # so rows may only be classified as structural zero locally.
    keep = row_max > 0.0
    max_scaled = rows_arr[keep] / row_max[keep, None]
    relative_norms = np.linalg.norm(max_scaled, axis=1)
    normalized = max_scaled / relative_norms[:, None]
    row_norms = np.zeros(rows_arr.shape[0], dtype=np.float64)
    row_norms[keep] = row_max[keep] * relative_norms
    return normalized, row_norms, keep


def shape_constraint_certificate(
    spec: _SplineBase,
    beta: NDArray,
    kind: str,
) -> ShapeConstraintCertificate:
    """Certify derivative sign continuously on every polynomial knot span."""
    beta_arr = np.asarray(beta, dtype=np.float64)
    candidates = _certificate_candidates(spec, beta_arr, kind)
    rows = _shape_constraint_rows(spec, candidates, kind)
    normalized_rows, row_norms, keep = _normalized_nonzero_shape_rows(rows)
    if not np.any(keep):
        # A constant/linear fitted span can have an identically zero second
        # derivative.  That is a valid curvature certificate.
        return ShapeConstraintCertificate(
            kind=kind,
            minimum_signed_derivative=0.0,
            minimum_scaled_slack=0.0,
            worst_x=float(spec._lo),
            maximum_row_norm=0.0,
        )
    rows_kept = rows[keep]
    values = rows_kept @ beta_arr
    scaled = normalized_rows @ beta_arr
    raw_worst_index = int(np.argmin(values))
    scaled_worst_index = int(np.argmin(scaled))
    kept_points = candidates[keep]
    if _has_natural_curvature_boundaries(spec, kind) and scaled[scaled_worst_index] >= 0.0:
        # The exact minimum over the closed fitted interval is attained at
        # both natural boundaries, where the constrained second derivative is
        # zero. The nonzero rows above certify every interior span.
        return ShapeConstraintCertificate(
            kind=kind,
            minimum_signed_derivative=0.0,
            minimum_scaled_slack=0.0,
            worst_x=float(getattr(spec, "_lo")),
            maximum_row_norm=float(np.max(row_norms[keep])),
        )
    return ShapeConstraintCertificate(
        kind=kind,
        minimum_signed_derivative=float(values[raw_worst_index]),
        minimum_scaled_slack=float(scaled[scaled_worst_index]),
        worst_x=float(kept_points[raw_worst_index]),
        maximum_row_norm=float(np.max(row_norms[keep])),
    )


def shape_constraint_is_roundoff_feasible(
    spec: _SplineBase,
    beta: NDArray,
    kind: str,
) -> bool:
    """Return whether any violation is no larger than the certificate's own rounding."""
    beta_arr = np.asarray(beta, dtype=np.float64)
    certificate = shape_constraint_certificate(spec, beta_arr, kind)
    tolerance = shape_slack_tolerance(beta_arr)
    return certificate.minimum_scaled_slack >= -tolerance


#: Roundings behind one certified slack, per coefficient.  The slack is a unit row
#: times ``beta``, the row comes from a conditioned basis derivative, and the
#: projection's QP accepts a linear constraint violated by up to about
#: ``1e-12 ||beta||``; the
#: count keeps a factor of ten over that for a term of ten coefficients.
_SLACK_STEPS = 2**14
#: Roundings behind one row of a deviance, merit or predictor sum, beyond the
#: summation: the predictor, the link and the unit deviance.
_ROW_STEPS = 16


def shape_sum_tolerance(scale: float, terms: int) -> float:
    """Higham's ``gamma_n * scale`` for a sum of ``terms`` rows, ``scale`` being their absolute total."""
    return float(gamma(_ROW_STEPS * terms)) * scale


def shape_slack_tolerance(beta: NDArray, curve_scale: float = 0.0) -> float:
    """Higham's ``gamma_n * scale`` for a scaled slack, a unit row times ``beta``.

    The slack carries the units of the coefficients and ``||beta||`` bounds its
    terms.  ``curve_scale``, the magnitude of the curve ``beta`` approximates,
    keeps the allowance when an optimizer's result collapses towards zero.
    """
    beta_arr = np.asarray(beta, dtype=np.float64)
    scale = max(float(np.linalg.norm(beta_arr)), curve_scale)
    return float(gamma(_SLACK_STEPS * beta_arr.size)) * scale


def _repair_shape(
    shape: str,
    spec: _SplineBase,
    beta_reparam: NDArray,
    groups: list[GroupSlice],
    weights: NDArray | None,
    n_grid: int,
    metric: NDArray | None = None,
) -> MonotoneRepairResult:
    """Project a fitted spline curve onto the ``shape`` cone, in the fitted basis.

    The body behind ``MonotoneRepairer.repair`` and ``CurvatureRepairer.repair``.
    ``metric``, when given, is the term's penalized Hessian in its fitted
    coefficients, and the projection is taken in it rather than on the grid.
    """
    beta_combined = np.concatenate([beta_reparam[g.sl] for g in groups])
    recon = spec.reconstruct(beta_combined, n_points=n_grid)
    x_grid = recon["x"]
    log_rels = recon["log_relativity"]
    certificate_before = shape_constraint_certificate(spec, beta_combined, shape)
    viol_before = certificate_before.violation

    if certificate_before.minimum_scaled_slack >= -shape_slack_tolerance(beta_combined):
        return MonotoneRepairResult(
            feature_name="",  # filled by caller
            direction=shape,
            grid=x_grid,
            original_log_effect=log_rels.copy(),
            repaired_log_effect=log_rels.copy(),
            repaired_beta_reparam=beta_reparam.copy(),
            max_violation_before=viol_before,
            max_violation_after=0.0,
            projection_residual=0.0,
        )

    w_grid = np.ones(n_grid) if weights is None else weights
    beta_reparam_new, repaired_check, proj_residual, certificate_after = (
        _project_shape_in_fitted_basis(
            spec,
            x_grid,
            log_rels,
            w_grid,
            shape,
            **({} if metric is None else {"metric": (metric, beta_combined)}),
        )
    )
    beta_out = beta_reparam.copy()
    offset = 0
    for g in groups:
        g_size = g.size
        beta_out[g.sl] = beta_reparam_new[offset : offset + g_size]
        offset += g_size
    viol_after = certificate_after.violation

    return MonotoneRepairResult(
        feature_name="",  # filled by caller
        direction=shape,
        grid=x_grid,
        original_log_effect=log_rels.copy(),
        repaired_log_effect=repaired_check.copy(),
        repaired_beta_reparam=beta_out,
        max_violation_before=viol_before,
        max_violation_after=viol_after,
        projection_residual=proj_residual,
    )


class MonotoneRepairer:
    """Constrained fitted-basis projection for monotone spline curves."""

    def __init__(self, direction: str = "increasing"):
        if direction not in ("increasing", "decreasing"):
            raise ValueError(f"direction must be 'increasing' or 'decreasing', got {direction!r}")
        self.direction = direction

    def repair(
        self,
        spec: _SplineBase,
        beta_reparam: NDArray,
        groups: list[GroupSlice],
        weights: NDArray | None = None,
        n_grid: int = 500,
        metric: NDArray | None = None,
    ) -> MonotoneRepairResult:
        """Project the fitted spline curve onto the requested monotone cone.

        Parameters
        ----------
        spec : _SplineBase
            The fitted spline spec (with knots, R_inv, etc. already set).
        beta_reparam : NDArray
            Full model beta (reparametrised space). Slices for this feature
            are extracted via ``groups``.
        groups : list[GroupSlice]
            All groups belonging to this feature (1 for non-select, 2 for select=True).
        weights : NDArray or None
            Grid weights (from training data histogram). If None, uniform.
        n_grid : int
            Number of grid points for curve reconstruction.
        metric : NDArray or None
            The term's penalized Hessian ``X'WX + S_lambda`` in its fitted
            coefficients, with the intercept profiled, as
            ``apply_shape_postfit`` passes it: the fitted coefficients are
            projected in it. If None, the curve is projected on the grid,
            weighted by ``weights``.

        Returns
        -------
        MonotoneRepairResult
        """
        return _repair_shape(self.direction, spec, beta_reparam, groups, weights, n_grid, metric)


#: Local-reduction steps the cone projection takes before it brackets instead.
_PROJECTION_STEPS = 12
#: Exchange steps between the inner and outer cones once it brackets.
_BRACKET_STEPS = 40
#: Gap between the inner and outer cones' optima within which the inner optimum is
#: the projection. The objective is scaled so that twice it, plus one, is the
#: projection's loss over the dropped term's; this is the objective tolerance SLSQP
#: was given on the same scaled problem before.
_OBJECTIVE_GAP = 1e-12
#: Largest curvature term one span may add to the step's Hessian, relative to
#: ``||H||``.  Beyond it the step's saddle solve would lose more digits than float64
#: holds, so that span takes a plain cutting-plane step instead.
_CURVATURE_CONDITION_LIMIT = 1e8
#: Move of a span's minimum, as a fraction of its span, within which a curvature step's
#: local model holds and the minimum's cut replaces the span's older ones.
_LOCAL_STEP = 0.25
#: Width, as a fraction of its span, within which a new checkpoint repeats a kept one.
_CHECKPOINT_SEPARATION = 1e-9
#: Unit rows closer than this in every entry are one constraint stated twice: the
#: two one-sided values of a continuous derivative at a knot, read from two pieces.
_SAME_ROW = 2.0**-40
#: Entries of row differences one block of candidate pairs may hold (2 MiB of
#: binary64); a typical repair compares all its pairs in one block.
_PAIR_BLOCK = 2**18


@dataclass(frozen=True)
class _ShapePiece:
    """One polynomial piece of the constrained derivative ``sign * f^(order)``.

    ``rows @ beta`` are the piece's monomial coefficients in ``tau``, the position
    across its span of ``width`` scaled to ``[0, 1]``, so the derivative there is
    ``sum_k (rows @ beta)[k] tau^k``.
    """

    width: float
    rows: NDArray


def _shape_pieces(spec: _SplineBase, kind: str) -> tuple[list[_ShapePiece], int]:
    """The polynomial pieces of the constrained derivative over the fitted range.

    Each piece is the Taylor expansion at its span's left end, the representable
    point from which the spline evaluates that piece (as in
    ``_shape_span_jump_matrix``), so even a one-ULP span keeps its own polynomial.
    The predictor is rescaled by a power of two, exactly, so the span widths are
    near one and no derivative of order up to the degree under- or overflows; each
    piece's rows are rescaled by a power of two as well, which leaves its cone
    unchanged. Returns the pieces and their degree ``p``.
    """
    from math import factorial

    order, sign = _shape_order_and_sign(kind)
    degree = 3 if getattr(spec, "_cr_knots", None) is not None else int(spec.degree)
    p = degree - order
    breaks = np.unique(np.concatenate(([spec._lo], _shape_breakpoints(spec), [spec._hi])))
    lo, hi = breaks[:-1], breaks[1:]
    exponent = 1 - int(np.frexp(float(breaks[-1] - breaks[0]))[1])
    basis = _raw_basis(spec, exponent)
    left = np.ldexp(lo, exponent)
    width = np.ldexp(hi - lo, exponent)
    terms = [
        (sign / factorial(k))
        * width[:, None] ** k
        * _to_fitted(spec, _raw_derivative(spec, basis, left, order + k))
        for k in range(p + 1)
    ]
    rows = np.stack(terms, axis=1)
    largest = np.max(np.abs(rows), axis=(1, 2))
    rows *= np.ldexp(1.0, -np.frexp(np.where(largest > 0.0, largest, 1.0))[1])[:, None, None]
    pieces = [
        _ShapePiece(float(b - a), piece_rows) for a, b, piece_rows in zip(lo, hi, rows, strict=True)
    ]
    return pieces, p


def _tau_powers(tau: float, p: int, nu: int = 0) -> NDArray:
    """The ``nu``-th ``tau`` derivative of ``(1, tau, ..., tau^p)``."""
    k = np.arange(p + 1)
    falling = np.ones(p + 1)
    for j in range(nu):
        falling *= np.maximum(k - j, 0)
    return falling * tau ** np.maximum(k - nu, 0)


def _piece_minimum(piece: _ShapePiece, p: int, beta: NDArray) -> tuple[float, NDArray] | None:
    """The piece's lowest interior local minimum: its ``tau`` and the row of its value."""
    if p < 2:
        return None
    from numpy.polynomial import polynomial

    coefficients = piece.rows @ beta
    if p == 2:
        # The vertex, in the float64 operations the general path below performs on a
        # quadratic: its slope's one root, a minimum where the slope rises.
        bend = 2.0 * coefficients[2]
        vertex = -coefficients[1] / bend if bend > 0.0 else np.nan
        if not 0.0 < vertex < 1.0:
            return None
        return float(vertex), _tau_powers(float(vertex), p) @ piece.rows
    slope = polynomial.polyder(coefficients)
    roots = polynomial.polyroots(slope)
    real = np.real(roots[np.abs(np.imag(roots)) <= 1e-12 * np.maximum(1.0, np.abs(roots))])
    candidates = real[(real > 0.0) & (real < 1.0)]
    candidates = candidates[polynomial.polyval(candidates, polynomial.polyder(slope)) > 0.0]
    if candidates.size == 0:
        return None
    tau = float(candidates[np.argmin(polynomial.polyval(candidates, coefficients))])
    return tau, _tau_powers(tau, p) @ piece.rows


def _distinct_unit_rows(rows: NDArray) -> tuple[NDArray, list[list[int]]]:
    """Unit-norm nonzero rows with repeats dropped, and the input rows each one stands for.

    A row repeats the first earlier kept row within ``_SAME_ROW`` in every entry.
    Only rows whose entry sums lie within ``sqrt(width)`` such widths of each other
    can match, so they are compared in sorted order, a block of rows at a time:
    memory linear in the rows, even when many sums tie.
    """
    unit, _, keep = _normalized_nonzero_shape_rows(rows)
    sources = np.flatnonzero(keep)
    count, width = unit.shape
    # max_k |a_k - b_k| <= s implies |sum(a - b)| <= width * s; each sum of unit
    # entries also carries up to gamma_width * width of rounding.
    sums = unit.sum(axis=1)
    radius = width * (_SAME_ROW + 2.0 * float(gamma(width)))
    order = np.argsort(sums, kind="stable")
    ordered = sums[order]
    first = np.searchsorted(ordered, ordered - radius, side="left")
    span = np.searchsorted(ordered, ordered + radius, side="right") - first
    # Every row with each one in its window, as (row, candidate) pairs, for the sorted
    # rows start:stop, whose pairs' differences hold at most _PAIR_BLOCK entries or
    # one row's window: evenly spaced jump rows share a sum, putting every pair in it.
    ends = np.cumsum(span)
    partner: dict[int, list[int]] = {}
    start = 0
    while start < count:
        budget = ends[start] - span[start] + _PAIR_BLOCK // width
        stop = max(start + 1, int(np.searchsorted(ends, budget, side="right")))
        block = span[start:stop]
        left = np.repeat(order[start:stop], block)
        right = order[
            np.arange(block.sum()) - np.repeat(np.cumsum(block) - block - first[start:stop], block)
        ]
        earlier = right < left
        left, right = left[earlier], right[earlier]
        close = np.max(np.abs(unit[left] - unit[right]), axis=1, initial=0.0) <= _SAME_ROW
        for index, other in zip(left[close].tolist(), right[close].tolist(), strict=True):
            partner.setdefault(index, []).append(other)
        start = stop
    kept: list[int] = []
    representative: dict[int, int] = {}
    merged: list[list[int]] = []
    for index in range(count):
        match = min(
            (other for other in partner.get(index, ()) if other in representative), default=None
        )
        if match is None:
            representative[index] = len(kept)
            kept.append(index)
            merged.append([int(sources[index])])
        else:
            merged[representative[match]].append(int(sources[index]))
    return unit[kept], merged


def _end_value_rows(
    pieces: list[_ShapePiece], p: int, natural: bool
) -> tuple[NDArray, list[tuple[int, float]]]:
    """Each piece's derivative at both ends of its span, and the (piece, ``tau``) of each row.

    The rows are necessary, and exact for ``p <= 1``. A natural spline's boundary
    curvature is an equality the basis already holds, so its round-off is not read
    as an inequality.
    """
    rows, ends = [], []
    for index, piece in enumerate(pieces):
        for tau in (0.0, 1.0):
            if natural and (index, tau) in {(0, 0.0), (len(pieces) - 1, 1.0)}:
                continue
            rows.append(_tau_powers(tau, p) @ piece.rows)
            ends.append((index, tau))
    return np.asarray(rows, dtype=np.float64).reshape(len(rows), -1), ends


def _bernstein_rows(piece: _ShapePiece, p: int, t0: float, t1: float) -> NDArray:
    """Rows of the derivative's Bernstein coefficients on ``[t0, t1]`` of the piece."""
    from scipy.special import comb

    nodes = (np.arange(p + 1) + 0.5) / (p + 1)
    k = np.arange(p + 1)
    bernstein = comb(p, k) * nodes[:, None] ** k * (1.0 - nodes[:, None]) ** (p - k)
    values = np.stack([_tau_powers(t0 + (t1 - t0) * s, p) for s in nodes]) @ piece.rows
    return np.linalg.solve(bernstein, values)


def _inner_cone_rows(
    pieces: list[_ShapePiece], p: int, checkpoints: list[list[float]], natural: bool
) -> NDArray:
    """Bernstein coefficients of every sub-span, split at the checkpoints.

    Non-negative Bernstein coefficients bound the derivative from below on their
    sub-span, so these rows are sufficient (Papp & Alizadeh 2014, section 3.2). A
    quadratic piece that is non-negative has non-negative coefficients on both sides
    of its minimum, so a split there loses nothing. The value at a split inside a
    piece is a convex combination of its two neighbouring coefficients (the joint
    of one polynomial is smooth), so it is implied and left out.
    """
    rows = []
    for index, (piece, points) in enumerate(zip(pieces, checkpoints, strict=True)):
        splits = np.unique(np.concatenate(([0.0], points, [1.0])))
        for start, end in zip(splits[:-1], splits[1:], strict=True):
            block = _bernstein_rows(piece, p, float(start), float(end))
            first = start == 0.0 and not (natural and index == 0)
            last = end == 1.0 and not (natural and index == len(pieces) - 1)
            rows.append(block[(0 if first else 1) : (p + 1 if last else p)])
            if p == 0 and (first or last):
                rows[-1] = block
    return np.vstack(rows)


def _cut_rows(
    pieces: list[_ShapePiece], p: int, cuts: list[list[float]], width: int
) -> tuple[NDArray, NDArray, list[tuple[int, float]]]:
    """Unit rows of the derivative at each cut, their norms, and each cut's (piece, ``tau``)."""
    owners = [(index, t) for index, points in enumerate(cuts) for t in points]
    raw = np.asarray(
        [_tau_powers(t, p) @ pieces[index].rows for index, t in owners], dtype=np.float64
    ).reshape(len(owners), width)
    normalized, norms, keep = _normalized_nonzero_shape_rows(raw)
    unit = np.zeros_like(raw)
    unit[keep] = normalized
    return unit, norms, owners


def _lowest_minima(
    pieces: list[_ShapePiece], p: int, beta: NDArray
) -> list[tuple[float, NDArray] | None]:
    return [_piece_minimum(piece, p, beta) for piece in pieces]


def _minima_feasible(minima: list[tuple[float, NDArray] | None], beta: NDArray) -> bool:
    """Whether every span's lowest interior value is non-negative to the certificate's rounding."""
    rows = [row for _, row in filter(None, minima)]
    if not rows:
        return True
    unit, _, _ = _normalized_nonzero_shape_rows(np.asarray(rows, dtype=np.float64))
    return float(np.min(unit @ beta, initial=0.0)) >= -shape_slack_tolerance(beta, 1.0)


#: Slack, in the scaled problem's units (the target's RMS is one), within which a unit
#: row counts as held when an uncertified QP point is checked for its KKT conditions.
_HELD_SLACK = 1e-8
#: Relative KKT residual an uncertified QP point may leave and still be the optimum.
_KKT_RESIDUAL = 1e-9


def _jacobi_scale(H: NDArray) -> NDArray:
    """Powers of two near ``1 / sqrt(diag(H))``, one per coefficient; 1 where the diagonal is 0."""
    diagonal = np.sqrt(np.maximum(np.diag(H), 0.0))
    return np.ldexp(1.0, -np.frexp(np.where(diagonal > 0.0, diagonal, 1.0))[1])


#: Iterations, at least, of a solve warm-started from a working set: a right guess
#: certifies within about as many iterations as there are coefficients.
_WARM_STEPS = 50
#: Ridge on a semidefinite projection Hessian's null space, relative to its largest
#: eigenvalue: sqrt(u), far above the rank authority's cutoff of eps, so the ridged
#: Hessian is full rank, and small enough that the grid loss it trades is at the
#: square root of the rounding level.
_NULL_RIDGE = 2.0**-26


def _null_space_ridge(H: NDArray, rank: int) -> NDArray:
    """``H`` with ``_NULL_RIDGE * ||H||`` added on the eigenvectors past its rank.

    A grid of fewer well-placed points than coefficients leaves the projection's
    objective flat along the null space of ``H``; the ridge picks, among the
    coefficients closest on the grid, those with the least null-space part.
    """
    values, vectors = np.linalg.eigh(H)
    null = vectors[:, : H.shape[0] - rank]
    return H + (_NULL_RIDGE * float(values[-1])) * (null @ null.T)


class _ScaledResult:
    """A QP result in the caller's coefficients, ``beta = scale * z``."""

    def __init__(self, result, scale: NDArray, truncated: bool = False) -> None:
        self.beta = scale * np.asarray(result.beta, dtype=np.float64)
        self.active_set = list(result.active_set)
        self.converged = bool(result.converged)
        self.n_iter = int(result.n_iter)
        # Whether the rank authority truncated the Hessian given, before any ridge.
        self.truncated = truncated


# Every solve runs in Jacobi-scaled coefficients, beta = D z with D the power of two
# nearest 1 / sqrt(diag(H)): exact, so the QP and its answer are unchanged, while
# cond(D H D) is within a factor p of the best any diagonal scaling gives (van der
# Sluis, Numer. Math. 14, 1969). A grid with sparse data weighs some basis columns
# 1e-5 of the rest, which squared in H puts cond(H) past 1 / u: the active-set path
# then turns on how the kernel rounded H, and one kernel's solve certifies where
# another's cycles. Scaled, the same solves certify on every kernel.
#
# A primal active-set solve needs a feasible start (Nocedal & Wright 2006, section
# 16.5). Every row here is homogeneous, so a cone: the previous step's point moved
# along a ray inside the cone just far enough to meet the new rows is one, next to
# the answer (_shifted_start), and beta = 0, its apex, always is. With a warm
# working set, the rows the caller expects to bind, a right guess certifies in a
# few iterations; a wrong one on a degenerate face, where most of a large basis's
# rows bind, can cycle, so that attempt gets a short budget. Then beta = 0, the
# shifted start alone and the solver's own start are tried, with a budget that visits
# every row twice beyond the solver's default, as #621 gives its own solves, and a
# null-space ridge if the Hessian is semidefinite.
def _solve_cone_qp(
    H: NDArray,
    g: NDArray,
    rows: NDArray,
    active: list[int] | None = None,
    start: NDArray | None = None,
    *,
    feasible: bool = False,
    regular: bool = False,
):
    """The repository QP on unit ``rows @ beta >= 0``; ``None`` when no solution is certified.

    The QP certifies its own result, except on a face whose held rows leave no step,
    as when every jump of a curve projected flat is held. Such a point is accepted
    when it is feasible and ``H beta - g`` lies in the cone of its held rows, which
    for a convex QP is optimality. ``active`` is a warm working set and ``start`` a
    point meeting every row (``_shifted_start``); without one, ``beta = 0`` is used.
    ``feasible`` returns, when nothing is certified, the attempt of least objective
    that meets every row and improves on the cone's apex, instead of ``None``: a
    bound, not an optimum. ``regular`` returns ``None`` at once where the QP would
    truncate ``H``'s rank.
    """
    from scipy.optimize import nnls

    from superglm.solvers.constrained_qp import solve_constrained_qp
    from superglm.solvers.rank import _symmetric_part, decompose_gram

    scale = _jacobi_scale(H)
    budget = max(200, 2 * (rows.shape[0] + H.shape[0]))
    bounds = np.zeros(rows.shape[0])
    problem = (H * scale[:, None] * scale[None, :], g * scale, rows * scale[None, :], bounds)
    zero = np.zeros(H.shape[0])
    shifted = zero if start is None else start / scale
    starts: list[tuple[NDArray | None, list[int] | None, int]] = [
        (zero, None, budget),
        (None, None, budget),
    ]
    if start is not None:
        starts.insert(1, (shifted, None, budget))
    if active:
        starts.insert(0, (shifted, active, max(_WARM_STEPS, H.shape[0])))
    result = None
    attempts: list[_ScaledResult] = []
    truncated = False
    for ridged in (False, True):
        try:
            # One decomposition of the round's Hessian serves every start's solve.
            decomposition = decompose_gram(_symmetric_part(problem[0]))
        except ValueError:
            return _least_feasible(H, g, rows, attempts) if feasible else None
        truncated = truncated or decomposition.rank < decomposition.width
        if regular and truncated:
            return None
        for point, working, steps in starts:
            try:
                solved = solve_constrained_qp(
                    *problem,
                    active_set_init=working,
                    max_iter=steps,
                    beta_start=point,
                    _decomposition=decomposition,
                )
            except ValueError:
                return _least_feasible(H, g, rows, attempts) if feasible else None
            result = _ScaledResult(solved, scale, truncated)
            if result.converged:
                return result
            attempts.append(result)
        if ridged or solved.rank == solved.width:
            break
        problem = (_null_space_ridge(problem[0], solved.rank), *problem[1:])
    assert result is not None
    beta = result.beta
    slack = rows @ beta
    if np.all(np.isfinite(beta)) and np.min(slack, initial=0.0) >= -shape_slack_tolerance(
        beta, 1.0
    ):
        gradient = H @ beta - g
        held = rows[slack <= _HELD_SLACK]
        residual = nnls(held.T, gradient)[1] if held.shape[0] else float(np.linalg.norm(gradient))
        if residual <= _KKT_RESIDUAL * max(1.0, float(np.linalg.norm(g))):
            return result
    return _least_feasible(H, g, rows, attempts) if feasible else None


def _least_feasible(H: NDArray, g: NDArray, rows: NDArray, attempts: list) -> _ScaledResult | None:
    """The attempt of least ``0.5 b'Hb - g'b`` below the apex's 0 among those meeting every row."""
    best, least = None, 0.0
    for attempt in attempts:
        beta = attempt.beta
        if not np.all(np.isfinite(beta)) or np.min(
            rows @ beta, initial=0.0
        ) < -shape_slack_tolerance(beta, 1.0):
            continue
        value = float(0.5 * beta @ H @ beta - g @ beta)
        if value < least:
            best, least = attempt, value
    return best


def _broken_rows(rows: NDArray, beta: NDArray) -> list[int]:
    """The rows ``beta`` breaks: a working set to start a solve from."""
    return np.flatnonzero(rows @ beta < 0.0).tolist()


def _working_rows(
    held: set, ends: int, owners: list[tuple[int, float]], cut_rows: NDArray, beta: NDArray
) -> list[int]:
    """A warm working set: the rows the last solve held, and the new cuts ``beta`` breaks.

    ``held`` names end rows by index and cuts by their (piece, ``tau``), so a
    cut keeps its place when the rows are rebuilt around it.
    """
    broken = cut_rows @ beta < 0.0
    return [row for row in range(ends) if row in held] + [
        ends + index for index, owner in enumerate(owners) if owner in held or bool(broken[index])
    ]


def _shifted_start(rows: NDArray, beta: NDArray, inside: NDArray | None) -> NDArray | None:
    """``beta`` moved along ``inside`` just far enough to meet every row, or ``None``.

    ``inside`` is a ray strictly inside the cone (``_inside_ray``); the shift is
    widened by ``2**-20`` of itself, far above the rounding of the slacks, so the
    start meets each row with room. ``None`` when the ray fails a row.
    """
    if inside is None:
        return None
    reach = rows @ inside
    if not np.all(reach > 0.0):
        return None
    shift = float(np.max(-(rows @ beta) / reach, initial=0.0))
    return beta + (shift * (1.0 + 2.0**-20)) * inside if shift > 0.0 else beta


def _inside_ray(basis: NDArray, x: NDArray, weights: NDArray, kind: str) -> NDArray:
    """Coefficients of a curve strictly inside the ``kind`` cone, where the basis holds one.

    The weighted least-squares fit, with a free constant the centred basis lacks,
    of ``t`` (monotone) or ``(t - 1/2)**2`` (curvature) in the dimensionless
    position ``t`` on the grid: a spline basis reproduces those polynomials, so its
    derivative is constant. ``_shifted_start`` checks every row it is used on.
    """
    order, sign = _shape_order_and_sign(kind)
    t = (x - x[0]) / (x[-1] - x[0])
    target = sign * (t if order == 1 else (t - 0.5) ** 2)
    design = np.column_stack([basis, np.ones(x.size)])
    gram = design.T @ (design * weights[:, None])
    return np.linalg.lstsq(gram, design.T @ (weights * target), rcond=None)[0][:-1]


def _bracketed_projection(
    H: NDArray,
    g: NDArray,
    pieces: list[_ShapePiece],
    p: int,
    end_rows: NDArray,
    checkpoints: list[list[float]],
    natural: bool,
    point: NDArray,
    inside: NDArray | None,
) -> tuple[NDArray | None, float]:
    """Exchange between the Bernstein inner cone and the cutting-plane outer cone.

    Every inner point is shape-feasible and every outer optimum bounds the
    projection's objective from below, so the best inner point is certified once
    the two meet. A span the projection holds flat sits at its cone's apex, which the
    inner cone holds exactly. Each solve starts from ``point``, the last outer
    optimum, shifted along ``inside``. An inner solve that does not certify still
    gives a feasible point when one improves on the apex; with no outer optimum the
    gap is left from the unconstrained minimum. ``None`` when no inner point is found.
    """

    def objective(beta: NDArray) -> float:
        return float(0.5 * beta @ H @ beta - g @ beta)

    best, upper, lower = None, np.inf, -np.inf
    cuts = [list(points) for points in checkpoints]
    for _ in range(_BRACKET_STEPS):
        inner_rows, _ = _distinct_unit_rows(_inner_cone_rows(pieces, p, cuts, natural))
        inner = _solve_cone_qp(
            H,
            g,
            inner_rows,
            _broken_rows(inner_rows, point),
            _shifted_start(inner_rows, point, inside),
            feasible=True,
        )
        cut_rows, _, owners = _cut_rows(pieces, p, cuts, H.shape[0])
        outer_rows = np.vstack([end_rows, cut_rows])
        outer = _solve_cone_qp(
            H,
            g,
            outer_rows,
            list(range(end_rows.shape[0], outer_rows.shape[0])),
            _shifted_start(outer_rows, point, inside),
        )
        if inner is not None and objective(inner.beta) < upper:
            best, upper = np.asarray(inner.beta, dtype=np.float64), objective(inner.beta)
        if outer is None:
            break
        point = outer.beta
        lower = max(lower, objective(outer.beta))
        if upper - lower <= _OBJECTIVE_GAP:
            break
        active = {row - end_rows.shape[0] for row in outer.active_set}
        cuts = [[] for _ in pieces]
        for position, (index, t) in enumerate(owners):
            if position in active:
                cuts[index].append(t)
        for index, piece in enumerate(pieces):
            for beta, violated_only in (
                (outer.beta, True),
                *([] if inner is None else [(inner.beta, False)]),
            ):
                minimum = _piece_minimum(piece, p, beta)
                if (
                    minimum is not None
                    and (minimum[1] @ beta < 0.0 or not violated_only)
                    and all(abs(minimum[0] - t) > _CHECKPOINT_SEPARATION for t in cuts[index])
                ):
                    cuts[index].append(minimum[0])
    if lower == -np.inf:
        lower = objective(np.linalg.lstsq(H, g, rcond=None)[0])
    return best, upper - lower


# The monotone or curvature cone a repair projects onto. A derivative of degree
# p <= 1 on each span is non-negative exactly when its span-end values are, so one
# QP is exact. A higher degree makes the cone semi-infinite (Papp & Alizadeh 2014:
# second-order-cone representable, not polyhedral), solved by exchange (Blankenship
# & Falk 1976; Arai & Kitahara 2026). Each QP holds the span-end values and a cut at
# every span's lowest interior minimum, keeps the cuts it holds active and drops the
# rest, which does not move a strictly convex QP's optimum. A binding cut's minimum
# moves with the coefficients; adding that curvature, lambda d d' / q'', to the
# Hessian makes the step a local-reduction SQP one, quadratically convergent
# (Hettich & Kortanek 1993). The curvature only places the cuts: the answer is the
# plain relaxation at them, a QP optimum whose own KKT conditions hold, and it is
# the projection once every span's minimum is feasible to the certificate's
# rounding. Should that not happen within _PROJECTION_STEPS, the cuts seed a bracket
# between the Bernstein inner cone and the cutting-plane outer cone
# (_bracketed_projection), shape-feasible by construction and disclosed if the
# bracket does not close.
def _cone_projection(
    spec: _SplineBase,
    H: NDArray,
    g: NDArray,
    kind: str,
    label: str,
    inside: NDArray | None = None,
) -> NDArray:
    """Minimise ``0.5 b'Hb - g'b`` over coefficients whose curve has ``kind`` everywhere.

    Exact on spans whose derivative is at most linear; elsewhere a local-reduction SQP
    places the cuts. The comment above gives the method and its sources. ``inside`` is
    a ray strictly inside the cone, from which each solve's start is found.
    """
    # The unconstrained optimum is the fitted curve itself; the rows it breaks are
    # the first working set, and the first cuts sit where it breaks the rule.
    beta = np.linalg.lstsq(H, g, rcond=None)[0]
    if _uses_span_jump_constraints(spec, kind):
        rows, _ = _distinct_unit_rows(_shape_constraint_rows(spec, _shape_breakpoints(spec), kind))
        result = _solve_cone_qp(
            H, g, rows, _broken_rows(rows, beta), _shifted_start(rows, beta, inside)
        )
        if result is None:
            raise RuntimeError(f"{label} projection failed to converge")
        return np.asarray(result.beta, dtype=np.float64)

    pieces, p = _shape_pieces(spec, kind)
    natural = _has_natural_curvature_boundaries(spec, kind)
    end_values, ends = _end_value_rows(pieces, p, natural)
    end_rows, end_sources = _distinct_unit_rows(end_values)
    end_norms = np.linalg.norm(end_values, axis=1)
    width = H.shape[0]
    h_norm = None  # ||H||_2, formed when a curvature step first needs it
    held: set = set(_broken_rows(end_rows, beta))
    checkpoints = [
        [] if minimum is None or minimum[1] @ beta >= 0.0 else [minimum[0]]
        for minimum in _lowest_minima(pieces, p, beta)
    ]
    curvature = np.zeros_like(H)
    # Whether the QP truncates H itself, read off the first step, whose curvature is 0.
    truncated = None
    for _ in range(_PROJECTION_STEPS):
        cut_rows, cut_norms, owners = _cut_rows(pieces, p, checkpoints, width)
        rows = np.vstack([end_rows, cut_rows])
        working = _working_rows(held, end_rows.shape[0], owners, cut_rows, beta)
        start = _shifted_start(rows, beta, inside)
        # A curvature step whose Hessian the QP would truncate, where H's it does not,
        # is not taken: the truncated solve's least-squares steps do not certify.
        result = _solve_cone_qp(
            H + curvature,
            g + curvature @ beta,
            rows,
            working,
            start,
            regular=bool(curvature.any()) and truncated is False,
        )
        if truncated is None and result is not None:
            truncated = result.truncated
        if result is None and curvature.any():
            curvature = np.zeros_like(H)
            result = _solve_cone_qp(H, g, rows, working, start)
        if result is None:
            break
        held = {
            row if row < end_rows.shape[0] else owners[row - end_rows.shape[0]]
            for row in result.active_set
        }
        corrected = curvature.any()
        previous, beta = beta, np.asarray(result.beta, dtype=np.float64)
        minima = _lowest_minima(pieces, p, beta)
        if _minima_feasible(minima, beta):
            if not corrected:
                return beta
            # The relaxation at these cuts and every span's minimum.
            certified_cuts = [
                points + ([] if minimum is None else [minimum[0]])
                for points, minimum in zip(checkpoints, minima, strict=True)
            ]
            certified_cut_rows, _, certified_owners = _cut_rows(pieces, p, certified_cuts, width)
            certified_rows = np.vstack([end_rows, certified_cut_rows])
            certified = _solve_cone_qp(
                H,
                g,
                certified_rows,
                _working_rows(held, end_rows.shape[0], certified_owners, certified_cut_rows, beta),
                _shifted_start(certified_rows, beta, inside),
            )
            if certified is not None and _minima_feasible(
                _lowest_minima(pieces, p, certified.beta), certified.beta
            ):
                return np.asarray(certified.beta, dtype=np.float64)

        active = list(result.active_set)
        weight = np.zeros(len(pieces))
        kept: list[list[float]] = [[] for _ in pieces]
        if active:
            multipliers = np.linalg.lstsq(
                rows[active].T, (H + curvature) @ beta - g - curvature @ previous, rcond=None
            )[0]
            for row_index, multiplier in zip(active, multipliers, strict=True):
                cut = row_index - end_rows.shape[0]
                if cut < 0:
                    # A span end held at zero beside an interior minimum brackets the
                    # minimum with the span's cut; the force both hold belongs to the
                    # minimum, so it counts toward that span's curvature.
                    source = end_sources[row_index][0]
                    near = [
                        (abs(minima[i][0] - tau) * pieces[i].width, i)
                        for i, tau in (ends[j] for j in end_sources[row_index])
                        if minima[i] is not None
                    ]
                    if multiplier > 0.0 and near:
                        weight[min(near)[1]] += multiplier / end_norms[source]
                    continue
                owner, tau = owners[cut]
                kept[owner].append(tau)
                if multiplier > 0.0 and cut_norms[cut] > 0.0:
                    weight[owner] += multiplier / cut_norms[cut]
        curvature = np.zeros_like(H)
        for index, (piece, minimum) in enumerate(zip(pieces, minima, strict=True)):
            if minimum is None:
                continue
            tau, row = minimum
            if (row @ beta < 0.0 or weight[index] > 0.0) and all(
                abs(tau - other) > _CHECKPOINT_SEPARATION for other in kept[index]
            ):
                kept[index].append(tau)
            if weight[index] > 0.0:
                slope_row = _tau_powers(tau, p, 1) @ piece.rows
                bend = float(_tau_powers(tau, p, 2) @ piece.rows @ beta)
                size = weight[index] * float(slope_row @ slope_row)
                if h_norm is None:
                    h_norm = float(np.linalg.norm(H, 2))
                if bend > 0.0 and size <= _CURVATURE_CONDITION_LIMIT * h_norm * bend:
                    curvature += (weight[index] / bend) * np.outer(slope_row, slope_row)
                    # The curvature carries the minimum's motion, so near its old cut
                    # the span's one cut is at its minimum: older cuts would pin a
                    # bracket round it. A minimum that jumped keeps the bracket.
                    if all(abs(tau - other) <= _LOCAL_STEP for other in kept[index]):
                        kept[index] = [tau]
        checkpoints = kept

    best, gap = _bracketed_projection(H, g, pieces, p, end_rows, checkpoints, natural, beta, inside)
    if best is None:
        raise RuntimeError(f"{label} projection failed to converge")
    if gap > _OBJECTIVE_GAP:
        warnings.warn(
            f"{label} repair stopped before it certified the closest {kind} curve; the "
            f"repaired curve is {kind} everywhere, and its distance from the fitted curve "
            f"exceeds the closest one's by at most {2.0 * gap:.1e} of the dropped term's.",
            RuntimeWarning,
            stacklevel=4,
        )
    return best


def _project_shape_in_fitted_basis(
    spec: _SplineBase,
    x_grid: NDArray,
    original_curve: NDArray,
    weights: NDArray,
    kind: str,
    metric: tuple[NDArray, NDArray] | None = None,
) -> tuple[NDArray, NDArray, float, ShapeConstraintCertificate]:
    """Project a curve onto a shape cone in fitted coefficient coordinates.

    The fitted spline basis already contains the model's identifiability projection.
    Solving in that basis therefore keeps the repaired term centred under the training
    measure and avoids the lossy raw-basis -> fitted-basis round trip.  Zero coefficients
    are a feasible point and provide an objective certificate for the returned solution.
    ``metric``, a pair ``(H, fitted)``, projects ``fitted`` in ``H`` instead of the
    curve on the grid; ``H`` is the term's penalized Hessian (``shape_ops``). Returns the
    coefficients, the repaired curve, its largest change and the coefficients' certificate.
    """
    label = "Curvature" if kind in {"convex", "concave"} else "Monotone"

    fitted_basis = np.asarray(spec.transform(x_grid), dtype=np.float64)
    target = np.asarray(original_curve, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if fitted_basis.ndim != 2 or fitted_basis.shape[0] != target.size:
        raise RuntimeError(f"{label} projection received incoherent fitted-basis geometry")
    if w.shape != target.shape or not np.all(np.isfinite(w)) or np.any(w <= 0.0):
        raise RuntimeError(f"{label} projection requires positive finite grid weights")
    if not np.all(np.isfinite(fitted_basis)) or not np.all(np.isfinite(target)):
        raise RuntimeError(f"{label} projection requires finite fitted-basis geometry")

    total_weight = float(np.sum(w))
    # The optimizer works in units of the target's weighted RMS, so its
    # tolerances and step control do not depend on the response's units.
    # Taken in units of a power of two near the target's largest value, which is exact and
    # keeps ``target * target`` normal for a response far from 1.
    target_largest = float(np.max(np.abs(target))) if target.size else 0.0
    target_unit = (
        float(np.ldexp(1.0, int(np.frexp(target_largest)[1]))) if target_largest > 0.0 else 1.0
    )
    curve_scale = (
        target_unit * float(np.sqrt(float(np.dot(w, (target / target_unit) ** 2)) / total_weight))
        or 1.0
    )
    if metric is None:
        H = fitted_basis.T @ (fitted_basis * w[:, None])
        g = fitted_basis.T @ (target * w)
        zero_loss = float(np.dot(w, target * target))
        H_scaled = H / total_weight
        g_scaled = g / (total_weight * curve_scale)
    else:
        # The fitted coefficients' distance in the penalized Hessian: the second-order
        # model of the penalized objective the publication rule judges.
        H = np.asarray(metric[0], dtype=np.float64)
        fitted = np.asarray(metric[1], dtype=np.float64)
        # The quadratic forms are taken on the coefficients in units of a power of two near
        # their largest entry: exact to rescale, and normal for a response of any size, where
        # ``fitted @ H @ fitted`` itself can underflow.
        largest = float(np.max(np.abs(fitted))) if fitted.size else 0.0
        unit = float(np.ldexp(1.0, int(np.frexp(largest)[1]))) if largest > 0.0 else 1.0
        g_unit = H @ (fitted / unit)
        zero_unit = float((fitted / unit) @ g_unit)
        if not np.all(np.isfinite(H)) or not np.isfinite(zero_unit) or zero_unit <= 0.0:
            raise RuntimeError(f"{label} projection requires a positive definite metric")
        # Scaled, as on the grid, so the dropped term's objective is 0 and the fitted
        # one's is -1/2.
        H_scaled = H * ((curve_scale / unit) ** 2 / zero_unit)
        g_scaled = g_unit * (curve_scale / unit / zero_unit)

    inside = _inside_ray(fitted_basis, np.asarray(x_grid, dtype=np.float64), w / total_weight, kind)
    beta_projected = curve_scale * _cone_projection(
        spec, H_scaled, g_scaled, kind, label, inside=inside
    )

    repaired_curve = fitted_basis @ beta_projected
    if not np.all(np.isfinite(beta_projected)) or not np.all(np.isfinite(repaired_curve)):
        raise RuntimeError(f"{label} projection returned non-finite fitted state")

    feasibility_tolerance = shape_slack_tolerance(beta_projected, curve_scale)
    certificate = shape_constraint_certificate(spec, beta_projected, kind)
    if certificate.minimum_scaled_slack < -feasibility_tolerance:
        raise RuntimeError(
            f"{label} projection returned an infeasible fitted state: "
            f"minimum scaled slack={certificate.minimum_scaled_slack:.3e}"
        )

    residual = repaired_curve - target
    if metric is None:
        projection_loss = float(np.dot(w, residual * residual))
        objective_tolerance = shape_sum_tolerance(zero_loss, w.size)
    else:
        step = (beta_projected - fitted) / unit
        projection_loss = float(step @ H @ step)
        magnitude = float(np.abs(step) @ np.abs(H) @ np.abs(step)) + zero_unit
        objective_tolerance = shape_sum_tolerance(magnitude, H.shape[0])
        # Both losses in the coefficient units above, where neither can underflow.
        zero_loss = zero_unit
    if not np.isfinite(projection_loss) or projection_loss > zero_loss + objective_tolerance:
        raise RuntimeError(
            f"{label} projection failed its feasible-zero objective certificate: "
            f"projected={projection_loss:.6g}, zero={zero_loss:.6g}"
        )
    max_curve_change = float(np.max(np.abs(residual))) if residual.size else 0.0
    return beta_projected, repaired_curve, max_curve_change, certificate


class CurvatureRepairer:
    """Convex/concave repair for spline curves on the linear predictor scale."""

    def __init__(self, kind: str = "convex"):
        if kind not in ("convex", "concave"):
            raise ValueError(f"kind must be 'convex' or 'concave', got {kind!r}")
        self.kind = kind

    def repair(
        self,
        spec: _SplineBase,
        beta_reparam: NDArray,
        groups: list[GroupSlice],
        weights: NDArray | None = None,
        n_grid: int = 500,
        metric: NDArray | None = None,
    ) -> MonotoneRepairResult:
        return _repair_shape(self.kind, spec, beta_reparam, groups, weights, n_grid, metric)
