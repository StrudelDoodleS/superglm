"""Post-fit monotone repair for 1-D spline terms.

Implements pyGAM-style isotonic regression repair: reconstruct the fitted
curve on a grid, apply weighted isotonic regression, then project back to
spline coefficients via weighted least squares.
"""

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
    """Result of a post-fit monotone repair for one spline feature."""

    feature_name: str
    direction: str  # "increasing" | "decreasing"
    grid: NDArray  # (n_grid,) evaluation points
    original_log_effect: NDArray  # (n_grid,) pre-repair curve
    repaired_log_effect: NDArray  # (n_grid,) post-repair curve
    repaired_beta_reparam: NDArray  # full feature beta in reparametrized space
    max_violation_before: float
    max_violation_after: float
    projection_residual: float  # ||B @ beta_proj - repaired_curve||


class MonotoneRepairer:
    """Isotonic regression repair for spline curves."""

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
        """Full repair pipeline: reconstruct -> isotonic -> project -> recenter.

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
        # 1. Combine beta from all subgroups
        beta_combined = np.concatenate([beta_reparam[g.sl] for g in groups])

        # 2. Reconstruct curve on grid
        recon = spec.reconstruct(beta_combined, n_points=n_grid)
        x_grid = recon["x"]
        log_rels = recon["log_relativity"]

        # 3. Compute violation before repair
        viol_before = monotonicity_violation(log_rels, self.direction)

        if viol_before < 1e-12:
            # Already monotone — return identity
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

        # 4. Grid weights
        if weights is None:
            w_grid = np.ones(n_grid)
        else:
            w_grid = weights

        # 5. Isotonic regression
        repaired = _weighted_isotonic(log_rels, w_grid, self.direction)

        # 6. Project back to coefficients
        beta_orig_new = _project_to_coefficients(spec, x_grid, repaired, w_grid)

        # 7. Recenter: subtract weighted mean to preserve identifiability
        beta_orig_new = _recenter(spec, x_grid, beta_orig_new, w_grid)

        # 8. Compute projection residual
        B_grid = spec._basis_matrix(x_grid).toarray()
        proj_residual = float(np.max(np.abs(B_grid @ beta_orig_new - repaired)))

        # 9. Convert to reparametrised space
        beta_reparam_new = _to_reparam(spec, beta_orig_new, groups)

        # 10. Patch into full beta vector
        beta_out = beta_reparam.copy()
        offset = 0
        for g in groups:
            g_size = g.size
            beta_out[g.sl] = beta_reparam_new[offset : offset + g_size]
            offset += g_size

        # 11. Verify violation after
        repaired_check = B_grid @ beta_orig_new
        viol_after = monotonicity_violation(repaired_check, self.direction)

        return MonotoneRepairResult(
            feature_name="",  # filled by caller
            direction=self.direction,
            grid=x_grid,
            original_log_effect=log_rels.copy(),
            repaired_log_effect=repaired.copy(),
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


def _weighted_isotonic(y: NDArray, w: NDArray, direction: str) -> NDArray:
    """Apply weighted isotonic regression via sklearn."""
    from sklearn.isotonic import IsotonicRegression

    increasing = direction == "increasing"
    ir = IsotonicRegression(increasing=increasing, out_of_bounds="clip")
    x_dummy = np.arange(len(y), dtype=np.float64)
    return ir.fit_transform(x_dummy, y, sample_weight=w)


def _project_to_coefficients(
    spec: _SplineBase,
    x_grid: NDArray,
    repaired_curve: NDArray,
    weights: NDArray,
) -> NDArray:
    """Weighted least-squares projection: find beta_orig minimizing ||B @ beta - curve||_W.

    Returns beta in the original (unprojected) basis space.
    """
    B_grid = spec._basis_matrix(x_grid).toarray()
    # Weighted normal equations: (B'WB) beta = B'W curve
    W_diag = weights
    BtWB = B_grid.T @ (B_grid * W_diag[:, None])
    BtWy = B_grid.T @ (repaired_curve * W_diag)
    # Regularize slightly for numerical stability
    BtWB += 1e-10 * np.eye(BtWB.shape[0])
    beta_orig = np.linalg.solve(BtWB, BtWy)
    return beta_orig


def _recenter(
    spec: _SplineBase,
    x_grid: NDArray,
    beta_orig: NDArray,
    weights: NDArray,
) -> NDArray:
    """Subtract weighted mean to preserve identifiability (intercept absorbs the mean)."""
    B_grid = spec._basis_matrix(x_grid).toarray()
    curve = B_grid @ beta_orig
    wmean = np.average(curve, weights=weights)
    # Subtract mean from curve, re-project
    curve_centered = curve - wmean
    W_diag = weights
    BtWB = B_grid.T @ (B_grid * W_diag[:, None])
    BtWy = B_grid.T @ (curve_centered * W_diag)
    BtWB += 1e-10 * np.eye(BtWB.shape[0])
    return np.linalg.solve(BtWB, BtWy)


def _to_reparam(
    spec: _SplineBase,
    beta_orig: NDArray,
    groups: list[GroupSlice],
) -> NDArray:
    """Convert original-space beta back to reparametrised space.

    For select=True, splits via U_null/U_range projections.
    For non-select, uses pinv(R_inv).
    """
    if spec.select and spec._U_null is not None:
        # select=True: beta_orig lives in raw K-dimensional space
        # Linear subgroup: project onto U_null
        beta_linear = spec._U_null.T @ beta_orig  # (1,)
        # Range subgroup: project onto U_range, then apply pinv(R_inv_range)
        beta_range_raw = spec._U_range.T @ beta_orig  # (n_range,)
        # The range subgroup has its own R_inv from SSP reparametrisation
        # We need to find the reparametrised coefficients
        # The R_inv for range is stored as part of the combined R_inv
        # For select=True, each subgroup has its own transform
        # Linear subgroup: no reparametrisation (reparametrize=False)
        # Range subgroup: has R_inv applied during fit
        # We need to invert the R_inv for the range subgroup
        # The combined R_inv is [U_null | U_range @ R_inv_local]
        # So for range: beta_reparam = pinv(R_inv_local) @ beta_range_raw
        R_inv = spec._R_inv
        if R_inv is not None:
            # R_inv combines both subgroups: cols = [1 null col | n_range reparam cols]
            n_null = spec._U_null.shape[1]
            R_inv_range_combined = R_inv[:, n_null:]  # (K, n_range_reparam)
            # Range raw = U_range.T @ beta_orig, range_reparam via pinv
            R_inv_range_local = spec._U_range.T @ R_inv_range_combined
            beta_range_reparam = np.linalg.lstsq(R_inv_range_local, beta_range_raw, rcond=None)[0]
        else:
            beta_range_reparam = beta_range_raw
        return np.concatenate([beta_linear, beta_range_reparam])
    else:
        # Non-select: pinv(R_inv) @ beta_orig
        R_inv = spec._R_inv
        if R_inv is not None:
            return np.linalg.lstsq(R_inv, beta_orig, rcond=None)[0]
        return beta_orig


def derivative_grid_matrix(spec: _SplineBase, n_grid: int = 200) -> NDArray:
    """B-spline first derivatives at grid points. Reserved for future constrained IRLS."""
    raise NotImplementedError(
        "Fit-time monotone constraints via derivative grid are not yet implemented."
    )
