"""Post-fit shape repair operations for SuperGLM."""

from __future__ import annotations

import numpy as np


def _constraint_kind(spec) -> str | None:
    return getattr(spec, "constraint_kind", getattr(spec, "monotone", None))


def _constraint_mode(spec) -> str:
    return getattr(spec, "constraint_mode", getattr(spec, "monotone_mode", "postfit"))


def _grid_weights(spec, x_col, sample_weight, n_grid: int):
    hist_counts, bin_edges = np.histogram(x_col, bins=n_grid, weights=sample_weight)
    x_grid = np.linspace(spec._lo, spec._hi, n_grid)
    grid_weights = np.interp(
        x_grid,
        0.5 * (bin_edges[:-1] + bin_edges[1:]),
        hist_counts.astype(np.float64) + 1.0,
    )
    return np.maximum(grid_weights, 1e-6)


def _repairer(kind: str):
    from superglm.constraints import CurvatureRepairer, MonotoneRepairer

    if kind in {"increasing", "decreasing"}:
        return MonotoneRepairer(direction=kind)
    if kind in {"convex", "concave"}:
        return CurvatureRepairer(kind=kind)
    raise ValueError(f"Unsupported postfit shape kind: {kind!r}")


def _invalidate_repair_caches(model) -> None:
    for attr in ("_coef_covariance", "_fit_active_info", "_fit_inference_info", "_group_edf"):
        try:
            delattr(model, attr)
        except AttributeError:
            pass


def apply_shape_postfit(model, X, sample_weight=None, offset=None, *, n_grid: int = 500):
    from superglm.features.spline import _SplineBase

    del offset

    if model._result is None:
        raise RuntimeError("Model must be fitted before calling apply_shape_postfit().")

    if not hasattr(model, "_shape_repairs"):
        model._shape_repairs = {}
    if not hasattr(model, "_monotone_repairs"):
        model._monotone_repairs = {}

    sample_weight_arr = None
    if sample_weight is not None:
        sample_weight_arr = np.asarray(sample_weight, dtype=np.float64)

    repaired_any = False

    for name, spec in model._specs.items():
        if not isinstance(spec, _SplineBase):
            continue

        kind = _constraint_kind(spec)
        if kind is None or _constraint_mode(spec) != "postfit":
            continue
        if name in model._shape_repairs:
            if kind in {"increasing", "decreasing"}:
                model._monotone_repairs[name] = model._shape_repairs[name]
            continue

        groups = [g for g in model._groups if g.feature_name == name]
        if not groups:
            continue

        beta = model.result.beta
        if not any(np.linalg.norm(beta[g.sl]) > 1e-12 for g in groups):
            continue

        x_col = np.asarray(X[name], dtype=np.float64)
        grid_weights = _grid_weights(spec, x_col, sample_weight_arr, n_grid)

        repair_result = _repairer(kind).repair(
            spec,
            beta,
            groups,
            weights=grid_weights,
            n_grid=n_grid,
        )
        repair_result.feature_name = name

        model._result.beta = repair_result.repaired_beta_reparam
        model._shape_repairs[name] = repair_result
        if kind in {"increasing", "decreasing"}:
            model._monotone_repairs[name] = repair_result
        repaired_any = True

    if repaired_any:
        _invalidate_repair_caches(model)

    return model
