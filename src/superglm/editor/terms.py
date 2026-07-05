"""Term extraction, weighting, and offset scoring for editor sessions."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from superglm.editor._types import EditableTerm


def term_from_inference(ti) -> EditableTerm:
    # TermInference is plotting/inference oriented. EditableTerm is the editor's
    # mutable log-effect grid plus enough metadata to reconstruct display and
    # control handles.
    log_effect = np.asarray(ti.log_relativity, dtype=np.float64).ravel()
    x = None if ti.x is None else np.asarray(ti.x, dtype=np.float64).ravel()
    levels = None if ti.levels is None else [str(level) for level in ti.levels]
    metadata = {
        "active": bool(ti.active),
        "centering_mode": ti.centering_mode,
        "edf": ti.edf,
        "smoothing_lambda": ti.smoothing_lambda,
        "monotone": ti.monotone,
    }
    if ti.spline is not None:
        metadata["spline"] = {
            "kind": ti.spline.kind,
            "boundary": tuple(float(v) for v in ti.spline.boundary),
            "n_basis": ti.spline.n_basis,
            "degree": ti.spline.degree,
        }
    if ti.smooth_curve is not None:
        metadata["smooth_curve"] = {
            "x": np.asarray(ti.smooth_curve.x, dtype=np.float64).tolist(),
            "log_relativity": np.asarray(ti.smooth_curve.log_relativity, dtype=np.float64).tolist(),
        }
    return EditableTerm(
        name=ti.name,
        kind=ti.kind,
        x=x,
        levels=levels,
        original_log_effect=log_effect.copy(),
        edited_log_effect=log_effect.copy(),
        weights=np.ones_like(log_effect, dtype=np.float64),
        ci_lower_log_effect=_ci_to_log_effect(ti.ci_lower, log_effect),
        ci_upper_log_effect=_ci_to_log_effect(ti.ci_upper, log_effect),
        metadata=metadata,
    )


def term_type_from_spec(spec) -> str:
    from superglm.features.categorical import Categorical
    from superglm.features.numeric import Numeric
    from superglm.features.ordered_categorical import OrderedCategorical
    from superglm.features.polynomial import Polynomial
    from superglm.features.spline import _SplineBase

    if isinstance(spec, _SplineBase):
        return "spline"
    if isinstance(spec, OrderedCategorical):
        return "ordered categorical"
    if isinstance(spec, Categorical):
        return "categorical"
    if isinstance(spec, Polynomial):
        return "polynomial"
    if isinstance(spec, Numeric):
        return "numeric"
    return type(spec).__name__


def term_weights_from_fit(model, name: str, term: EditableTerm) -> NDArray:
    # Exposure bars/densities use retained fit weights. If fit data was not
    # retained, fall back to equal weights so the editor remains usable.
    fallback = np.ones(term.size, dtype=np.float64)
    X_ref = getattr(model, "_fit_X_ref", None)
    fit_weights = getattr(model, "_fit_weights", None)
    if X_ref is None or fit_weights is None or name not in X_ref:
        return fallback

    values = X_ref[name]
    weights = np.asarray(fit_weights, dtype=np.float64).ravel()
    if weights.size != len(values):
        return fallback

    if term.levels is not None:
        raw_str = np.asarray([str(v) for v in np.asarray(values, dtype=object)], dtype=object)
        return np.asarray(
            [np.sum(weights[raw_str == level]) for level in term.levels], dtype=np.float64
        )

    if term.x is None or term.x.size <= 1:
        return np.array([float(np.sum(weights))], dtype=np.float64)

    try:
        raw_x = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return fallback
    if raw_x.size != weights.size:
        return fallback

    return np.histogram(raw_x, bins=grid_edges(term.x), weights=weights)[0].astype(np.float64)


def term_offset_values(term: EditableTerm, values) -> NDArray:
    # Score edited terms back onto raw rows for fixed-offset refits. Categoricals
    # map by label; continuous terms interpolate over the editor grid.
    effects = np.asarray(term.edited_log_effect, dtype=np.float64).ravel()
    if term.levels is not None:
        mapping = {level: float(effects[i]) for i, level in enumerate(term.levels)}
        raw = [str(v) for v in np.asarray(values, dtype=object).ravel()]
        missing = sorted({v for v in raw if v not in mapping})
        if missing:
            raise KeyError(f"Offset data contains unseen level(s) for {term.name!r}: {missing}")
        return np.asarray([mapping[v] for v in raw], dtype=np.float64)

    x_values = np.asarray(values, dtype=np.float64).ravel()
    if effects.size == 1:
        return x_values * float(effects[0])
    if term.x is None:
        raise TypeError(f"Term {term.name!r} does not expose an x grid for offset scoring.")

    x_grid = np.asarray(term.x, dtype=np.float64).ravel()
    order = np.argsort(x_grid)
    x_sorted = x_grid[order]
    y_sorted = effects[order]
    return np.interp(x_values, x_sorted, y_sorted, left=y_sorted[0], right=y_sorted[-1])


def resolve_refit_method(model, method: str) -> str:
    if method == "auto":
        meta = getattr(model, "_last_fit_meta", None) or {}
        return "fit_reml" if meta.get("method") == "fit_reml" else "fit"
    return method


def grid_edges(x: NDArray) -> NDArray:
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return np.array([0.0, 1.0], dtype=np.float64)
    if x.size == 1:
        return np.array([x[0] - 0.5, x[0] + 0.5], dtype=np.float64)
    mids = (x[:-1] + x[1:]) / 2.0
    left = x[0] - (mids[0] - x[0])
    right = x[-1] + (x[-1] - mids[-1])
    return np.concatenate([[left], mids, [right]]).astype(np.float64)


def _ci_to_log_effect(values, log_effect: NDArray) -> NDArray | None:
    # Term inference may provide intervals on relativity or link scale depending
    # on source path. Store them on the same log-effect scale as edits.
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.shape != log_effect.shape:
        return None
    if np.all(arr > 0):
        return np.log(arr)
    return arr
