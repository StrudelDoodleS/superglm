"""Post-fit shape repair for 1-D spline terms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

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
    return kind in {"convex", "concave"} and (
        getattr(spec, "_Z", None) is not None or getattr(spec, "_cr_M", None) is not None
    )


def _shape_probe_points(spec: _SplineBase, kind: str) -> NDArray:
    breakpoints = _shape_breakpoints(spec)
    if _uses_span_jump_constraints(spec, kind):
        return breakpoints
    if breakpoints.size < 2:
        return np.asarray([spec._lo, spec._hi], dtype=np.float64)
    midpoints = 0.5 * (breakpoints[:-1] + breakpoints[1:])
    return np.unique(np.concatenate((breakpoints, midpoints, [spec._lo, spec._hi])))


def shape_derivative_matrix(spec: _SplineBase, x: NDArray, order: int) -> NDArray:
    """Evaluate exact fitted-basis derivatives at ``x``."""
    from scipy.interpolate import BSpline, CubicSpline

    points = np.asarray(x, dtype=np.float64).ravel()
    cardinal_knots = getattr(spec, "_cr_knots", None)
    if cardinal_knots is not None:
        raw = CubicSpline(
            np.asarray(cardinal_knots, dtype=np.float64),
            np.eye(int(spec._n_basis), dtype=np.float64),
            axis=0,
            bc_type="natural",
            extrapolate=False,
        )(points, nu=order)
    elif order > int(spec.degree):
        raw = np.zeros((points.size, int(spec._n_basis)), dtype=np.float64)
    else:
        raw = BSpline(
            np.asarray(spec._knots, dtype=np.float64),
            np.eye(int(spec._n_basis), dtype=np.float64),
            int(spec.degree),
            extrapolate=False,
        )(points, nu=order)
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


def _violating_shape_constraint_points(
    spec: _SplineBase,
    beta: NDArray,
    kind: str,
    tolerance: float,
) -> NDArray:
    """Return every analytically located derivative extremum below tolerance."""
    beta_arr = np.asarray(beta, dtype=np.float64)
    candidates = _certificate_candidates(spec, beta_arr, kind)
    rows = _shape_constraint_rows(spec, candidates, kind)
    normalized_rows, _, keep = _normalized_nonzero_shape_rows(rows)
    if not np.any(keep):
        return np.empty(0, dtype=np.float64)
    scaled = normalized_rows @ beta_arr
    return candidates[keep][scaled < -tolerance]


def shape_constraint_is_roundoff_feasible(
    spec: _SplineBase,
    beta: NDArray,
    kind: str,
) -> bool:
    """Return whether any violation is no larger than floating-point roundoff."""
    beta_arr = np.asarray(beta, dtype=np.float64)
    certificate = shape_constraint_certificate(spec, beta_arr, kind)
    tolerance = _shape_roundoff_tolerance(beta_arr)
    return certificate.minimum_scaled_slack >= -tolerance


def _shape_roundoff_tolerance(beta: NDArray) -> float:
    return (
        256.0
        * np.finfo(np.float64).eps
        * (1.0 + float(np.linalg.norm(np.asarray(beta, dtype=np.float64))))
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

        Returns
        -------
        MonotoneRepairResult
        """
        beta_combined = np.concatenate([beta_reparam[g.sl] for g in groups])
        recon = spec.reconstruct(beta_combined, n_points=n_grid)
        x_grid = recon["x"]
        log_rels = recon["log_relativity"]
        certificate_before = shape_constraint_certificate(spec, beta_combined, self.direction)
        viol_before = certificate_before.violation

        if certificate_before.minimum_scaled_slack >= -_shape_roundoff_tolerance(beta_combined):
            return MonotoneRepairResult(
                feature_name="",  # filled by caller
                direction=self.direction,
                grid=x_grid,
                original_log_effect=log_rels.copy(),
                repaired_log_effect=log_rels.copy(),
                repaired_beta_reparam=beta_reparam.copy(),
                max_violation_before=viol_before,
                max_violation_after=0.0,
                projection_residual=0.0,
            )

        w_grid = np.ones(n_grid) if weights is None else weights
        beta_reparam_new, repaired_check, proj_residual = _project_shape_in_fitted_basis(
            spec,
            x_grid,
            log_rels,
            w_grid,
            self.direction,
        )
        beta_out = beta_reparam.copy()
        offset = 0
        for g in groups:
            g_size = g.size
            beta_out[g.sl] = beta_reparam_new[offset : offset + g_size]
            offset += g_size
        certificate_after = shape_constraint_certificate(
            spec,
            beta_reparam_new,
            self.direction,
        )
        viol_after = certificate_after.violation

        return MonotoneRepairResult(
            feature_name="",  # filled by caller
            direction=self.direction,
            grid=x_grid,
            original_log_effect=log_rels.copy(),
            repaired_log_effect=repaired_check.copy(),
            repaired_beta_reparam=beta_out,
            max_violation_before=viol_before,
            max_violation_after=viol_after,
            projection_residual=proj_residual,
        )


def monotonicity_violation(values: NDArray, direction: str) -> float:
    """Maximum monotonicity violation: max backwards step size."""
    diffs = np.diff(values)
    if direction == "increasing":
        violations = np.maximum(0.0, -diffs)
    else:
        violations = np.maximum(0.0, diffs)
    return float(np.max(violations)) if len(violations) > 0 else 0.0


def curvature_violation(values: NDArray, kind: str) -> float:
    diffs2 = np.diff(values, n=2)
    if kind == "convex":
        bad = np.maximum(0.0, -diffs2)
    elif kind == "concave":
        bad = np.maximum(0.0, diffs2)
    else:
        raise ValueError(f"Unsupported curvature kind: {kind!r}")
    return float(np.max(bad)) if len(bad) else 0.0


def _project_shape_in_fitted_basis(
    spec: _SplineBase,
    x_grid: NDArray,
    original_curve: NDArray,
    weights: NDArray,
    kind: str,
) -> tuple[NDArray, NDArray, float]:
    """Project a curve onto a shape cone in fitted coefficient coordinates.

    The fitted spline basis already contains the model's identifiability projection.
    Solving in that basis therefore keeps the repaired term centred under the training
    measure and avoids the lossy raw-basis -> fitted-basis round trip.  Zero coefficients
    are a feasible point and provide an objective certificate for the returned solution.
    """
    from scipy.optimize import LinearConstraint, minimize

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

    H = fitted_basis.T @ (fitted_basis * w[:, None])
    g = fitted_basis.T @ (target * w)
    zero_loss = float(np.dot(w, target * target))
    objective_scale = max(zero_loss, 1.0)
    H_scaled = H / objective_scale
    g_scaled = g / objective_scale

    def objective(beta: NDArray) -> float:
        return float(0.5 * beta @ H_scaled @ beta - g_scaled @ beta)

    def gradient(beta: NDArray) -> NDArray:
        return H_scaled @ beta - g_scaled

    constraint_points = list(_shape_probe_points(spec, kind))
    beta_projected = np.zeros(fitted_basis.shape[1], dtype=np.float64)
    scaled_constraints = np.empty((0, fitted_basis.shape[1]), dtype=np.float64)
    max_refinements = max(8, 2 * fitted_basis.shape[1])
    for _ in range(max_refinements):
        constraint_matrix = _shape_constraint_rows(
            spec,
            np.asarray(constraint_points, dtype=np.float64),
            kind,
        )
        normalized_rows, _, keep = _normalized_nonzero_shape_rows(constraint_matrix)
        if not np.any(keep):
            # Identically zero derivatives make the zero coefficient vector
            # (and every candidate) shape-feasible.
            scaled_constraints = np.empty((0, fitted_basis.shape[1]), dtype=np.float64)
        else:
            scaled_constraints = normalized_rows

        constraints = (
            []
            if scaled_constraints.size == 0
            else [LinearConstraint(scaled_constraints, 0.0, np.inf)]
        )
        result = minimize(
            objective,
            beta_projected,
            jac=gradient,
            method="SLSQP",
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 500},
        )
        if not result.success:
            raise RuntimeError(
                f"{label} projection failed to converge: "
                f"status={result.status}, message={result.message}"
            )

        beta_projected = np.asarray(result.x, dtype=np.float64)
        feasibility_tolerance = 2e-12 * (1.0 + float(np.linalg.norm(beta_projected)))
        violating_points = _violating_shape_constraint_points(
            spec,
            beta_projected,
            kind,
            feasibility_tolerance,
        )
        if violating_points.size == 0:
            break

        new_points = [
            float(new_point)
            for new_point in violating_points
            if not any(
                abs(new_point - old_point)
                <= 16.0 * np.finfo(np.float64).eps * max(1.0, abs(new_point))
                for old_point in constraint_points
            )
        ]
        if not new_points:
            certificate = shape_constraint_certificate(spec, beta_projected, kind)
            raise RuntimeError(
                f"{label} projection could not certify a constrained derivative span: "
                f"minimum scaled slack={certificate.minimum_scaled_slack:.3e}"
            )
        constraint_points.extend(new_points)
    else:
        raise RuntimeError(f"{label} projection exhausted its derivative-span refinements")

    repaired_curve = fitted_basis @ beta_projected
    if not np.all(np.isfinite(beta_projected)) or not np.all(np.isfinite(repaired_curve)):
        raise RuntimeError(f"{label} projection returned non-finite fitted state")

    feasibility_tolerance = 2e-12 * (1.0 + float(np.linalg.norm(beta_projected)))
    certificate = shape_constraint_certificate(spec, beta_projected, kind)
    if certificate.minimum_scaled_slack < -feasibility_tolerance:
        raise RuntimeError(
            f"{label} projection returned an infeasible fitted state: "
            f"minimum scaled slack={certificate.minimum_scaled_slack:.3e}"
        )

    residual = repaired_curve - target
    projection_loss = float(np.dot(w, residual * residual))
    objective_tolerance = 1e-8 * (1.0 + zero_loss)
    if not np.isfinite(projection_loss) or projection_loss > zero_loss + objective_tolerance:
        raise RuntimeError(
            f"{label} projection failed its feasible-zero objective certificate: "
            f"projected={projection_loss:.6g}, zero={zero_loss:.6g}"
        )
    max_curve_change = float(np.max(np.abs(residual))) if residual.size else 0.0
    return beta_projected, repaired_curve, max_curve_change


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
    ) -> MonotoneRepairResult:
        beta_combined = np.concatenate([beta_reparam[g.sl] for g in groups])

        recon = spec.reconstruct(beta_combined, n_points=n_grid)
        x_grid = recon["x"]
        log_rels = recon["log_relativity"]

        certificate_before = shape_constraint_certificate(spec, beta_combined, self.kind)
        viol_before = certificate_before.violation

        if certificate_before.minimum_scaled_slack >= -_shape_roundoff_tolerance(beta_combined):
            return MonotoneRepairResult(
                feature_name="",
                direction=self.kind,
                grid=x_grid,
                original_log_effect=log_rels.copy(),
                repaired_log_effect=log_rels.copy(),
                repaired_beta_reparam=beta_reparam.copy(),
                max_violation_before=viol_before,
                max_violation_after=0.0,
                projection_residual=0.0,
            )

        w_grid = np.ones(n_grid) if weights is None else weights
        beta_reparam_new, repaired_check, proj_residual = _project_shape_in_fitted_basis(
            spec,
            x_grid,
            log_rels,
            w_grid,
            self.kind,
        )
        certificate_after = shape_constraint_certificate(spec, beta_reparam_new, self.kind)
        viol_after = certificate_after.violation

        beta_out = beta_reparam.copy()
        offset = 0
        for g in groups:
            g_size = g.size
            beta_out[g.sl] = beta_reparam_new[offset : offset + g_size]
            offset += g_size

        return MonotoneRepairResult(
            feature_name="",
            direction=self.kind,
            grid=x_grid,
            original_log_effect=log_rels.copy(),
            repaired_log_effect=repaired_check.copy(),
            repaired_beta_reparam=beta_out,
            max_violation_before=viol_before,
            max_violation_after=viol_after,
            projection_residual=proj_residual,
        )


def derivative_grid_matrix(spec: _SplineBase, n_grid: int = 200) -> NDArray:
    """B-spline first derivatives at grid points. Reserved for future constrained IRLS."""
    raise NotImplementedError(
        "Fit-time monotone constraints via derivative grid are not yet implemented."
    )
