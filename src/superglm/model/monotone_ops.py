"""Post-fit monotone repair operations for SuperGLM."""

from __future__ import annotations

import numpy as np


def apply_monotone_postfit(
    model,
    X,
    exposure=None,
    offset=None,
    *,
    sample_weight=None,
    n_grid: int = 500,
):
    """Find all monotone-annotated splines, repair them, store results.

    Iterates ``model._specs`` for ``_SplineBase`` instances with
    ``monotone is not None``. Idempotent: skips features already repaired.

    Parameters
    ----------
    model : SuperGLM
        A fitted model.
    X : DataFrame
        Training data (used for grid weight computation).
    exposure, sample_weight : array-like, optional
        Frequency weights.
    offset : array-like, optional
        Offset term.
    n_grid : int
        Grid resolution for isotonic regression.

    Returns
    -------
    SuperGLM
        The model (self), with ``_monotone_repairs`` populated.
    """
    from superglm.constraints import MonotoneRepairer
    from superglm.features.spline import _SplineBase
    from superglm.model.base import resolve_sample_weight_alias

    exposure = resolve_sample_weight_alias(
        exposure, sample_weight, method_name="apply_monotone_postfit()"
    )

    if model._result is None:
        raise RuntimeError("Model must be fitted before calling apply_monotone_postfit().")

    # Initialize storage if needed
    if not hasattr(model, "_monotone_repairs"):
        model._monotone_repairs = {}

    repaired_any = False

    for name, spec in model._specs.items():
        if not isinstance(spec, _SplineBase):
            continue
        if spec.monotone is None:
            continue
        if name in model._monotone_repairs:
            continue  # idempotent

        groups = [g for g in model._groups if g.feature_name == name]
        if not groups:
            continue

        # Check if any group is active
        beta = model.result.beta
        active = any(np.linalg.norm(beta[g.sl]) > 1e-12 for g in groups)
        if not active:
            continue

        # Compute grid weights from training data distribution
        x_col = np.asarray(X[name], dtype=np.float64)
        hist_counts, bin_edges = np.histogram(x_col, bins=n_grid)
        # Interpolate to grid centers
        x_grid = np.linspace(spec._lo, spec._hi, n_grid)
        grid_weights = np.interp(
            x_grid,
            0.5 * (bin_edges[:-1] + bin_edges[1:]),
            hist_counts.astype(np.float64) + 1.0,  # +1 smoothing
        )
        grid_weights = np.maximum(grid_weights, 1e-6)

        repairer = MonotoneRepairer(direction=spec.monotone)
        repair_result = repairer.repair(spec, beta, groups, weights=grid_weights, n_grid=n_grid)
        repair_result.feature_name = name

        # Patch beta directly (PIRLSResult is a mutable dataclass)
        model._result.beta = repair_result.repaired_beta_reparam

        model._monotone_repairs[name] = repair_result
        repaired_any = True

    # Invalidate cached properties if any repair was done
    if repaired_any:
        for attr in ("_coef_covariance", "_fit_active_info", "_group_edf"):
            try:
                delattr(model, attr)
            except AttributeError:
                pass

    return model
