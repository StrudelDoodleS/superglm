"""Spline control-handle helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from superglm.editor._types import EditableTerm

# Term types whose control handles are recovered from a fitted basis rather than
# drawn as a display-only fallback.  Kept here, where the recovery lives, so the
# two callers that gate on it (`EditorSession._require_control_term` and
# `payloads._controls_payload`) cannot drift apart.
CONTROL_HANDLE_TERM_TYPES = ("spline", "piecewise")


def control_points(model, term: EditableTerm, n_handles: int | None = None) -> dict:
    # Prefer the fitted spline basis when it is available. Moving one displayed
    # handle then changes one basis coefficient and preserves the spline's
    # native continuity behavior.
    raw = raw_control_components(model, term, n_handles=n_handles)
    if raw is not None:
        basis, basis_indices, x_ctrl, coeff = raw
        min_handles, max_handles = _control_handle_limits(basis.shape[1])
        return {
            "x": x_ctrl.copy(),
            "log_effect": coeff[basis_indices].copy(),
            "basis_index": basis_indices.copy(),
            "basis": np.asarray(basis[:, basis_indices].T, dtype=np.float64),
            "build_basis": np.asarray(basis.T, dtype=np.float64),
            "build_log_effect": coeff.copy(),
            "min_handles": min_handles,
            "max_handles": max_handles,
        }

    # Fallback handles are display controls only. They are useful for editable
    # curves without recoverable basis details, but they are not model
    # coefficients.
    x_ctrl = fallback_control_x(term, n_handles=n_handles)
    min_handles, max_handles = fallback_control_handle_limits(term)
    return {
        "x": x_ctrl.copy(),
        "log_effect": interp_log_effect(term, x_ctrl),
        "basis_index": np.arange(x_ctrl.size, dtype=np.intp),
        "min_handles": min_handles,
        "max_handles": max_handles,
    }


def control_curve_after_move(
    model,
    term: EditableTerm,
    handle_index: int,
    log_effect: float,
    *,
    n_handles: int | None = None,
) -> tuple[NDArray, dict]:
    raw = raw_control_components(model, term, n_handles=n_handles)
    if raw is not None:
        basis, basis_indices, x_ctrl, coeff = raw
        if handle_index < 0 or handle_index >= x_ctrl.size:
            raise IndexError(f"Control handle index out of range for term {term.name!r}.")
        basis_index = int(basis_indices[handle_index])
        coeff[basis_index] = float(log_effect)
        return np.asarray(basis @ coeff, dtype=np.float64), {
            "basis": "raw_b_spline",
            "basis_index": basis_index,
            "x": float(x_ctrl[handle_index]),
        }

    # The fallback path rebuilds the curve through fixed-x controls. PCHIP keeps
    # the preview local and shape-preserving without pretending to be the
    # original fitted spline basis.
    x_ctrl = fallback_control_x(term, n_handles=n_handles)
    if handle_index < 0 or handle_index >= x_ctrl.size:
        raise IndexError(f"Control handle index out of range for term {term.name!r}.")
    target = interp_log_effect(term, x_ctrl)
    target[handle_index] = float(log_effect)
    return pchip_control_curve(term, x_ctrl, target), {"x": float(x_ctrl[handle_index])}


def raw_control_components(
    model,
    term: EditableTerm,
    *,
    n_handles: int | None = None,
) -> tuple[NDArray, NDArray[np.intp], NDArray, NDArray] | None:
    # Some spline specs expose their raw design matrix. When they do, we recover
    # the current displayed coefficient vector by least squares against the
    # edited curve, then expose a readable subset of basis coefficients.
    spec = None if model is None else getattr(model, "_specs", {}).get(term.name)
    if spec is None or not hasattr(spec, "_raw_basis_matrix") or term.x is None:
        return None
    try:
        basis = _as_dense_matrix(spec._raw_basis_matrix(term.x))
    except Exception:
        return None
    if basis.ndim != 2 or basis.shape[1] < 3:
        return None
    coeff = np.linalg.lstsq(
        basis,
        np.asarray(term.edited_log_effect, dtype=np.float64),
        rcond=None,
    )[0]
    x_ctrl = _basis_support_centers(basis, term)
    if x_ctrl is None:
        x_ctrl = _greville_abscissae(spec, basis.shape[1], term)
    if n_handles is None and getattr(spec, "_editor_wants_all_handles", False):
        # Opt-in: one handle per basis column instead of the 12-handle default.
        # A spec sets this when every column is a reported coefficient rather
        # than one sample of a dense curve -- subsampling would then hide model
        # parameters from the editor, not just thin the display. The hard cap in
        # `_control_handle_limits` still applies, so a spec with more than 24
        # columns does subsample and has to say so.
        n_handles = basis.shape[1]
    basis_indices = _control_basis_indices(basis.shape[1], n_handles=n_handles)
    return basis, basis_indices, x_ctrl[basis_indices], np.asarray(coeff, dtype=np.float64)


def fallback_control_handle_limits(term: EditableTerm) -> tuple[int, int]:
    spline_meta = term.metadata.get("spline", {})
    n_basis = int(spline_meta.get("n_basis", 9)) if isinstance(spline_meta, dict) else 9
    max_handles = min(max(n_basis, 3), 12)
    return min(3, max_handles), max_handles


def fallback_control_x(term: EditableTerm, n_handles: int | None = None) -> NDArray:
    raw = term.metadata.get("control_x")
    if raw is not None:
        values = np.asarray(raw, dtype=np.float64).ravel()
        if values.size >= 3 and (n_handles is None or values.size == int(n_handles)):
            return values

    if term.x is None:
        raise TypeError(f"Term {term.name!r} does not expose an x grid.")
    spline_meta = term.metadata.get("spline", {})
    n_basis = int(spline_meta.get("n_basis", 9)) if isinstance(spline_meta, dict) else 9
    min_handles, max_handles = fallback_control_handle_limits(term)
    if n_handles is None:
        n_controls = int(np.clip(n_basis, 6, max_handles))
    else:
        n_controls = int(np.clip(int(n_handles), min_handles, max_handles))
    x = np.asarray(term.x, dtype=np.float64).ravel()
    values = np.linspace(float(np.min(x)), float(np.max(x)), n_controls)
    if n_handles is None:
        term.metadata["control_x"] = values.tolist()
    return values


def interp_log_effect(term: EditableTerm, x_values: NDArray) -> NDArray:
    if term.x is None:
        raise TypeError(f"Term {term.name!r} does not expose an x grid.")
    x_grid = np.asarray(term.x, dtype=np.float64).ravel()
    y_grid = np.asarray(term.edited_log_effect, dtype=np.float64).ravel()
    order = np.argsort(x_grid)
    return np.interp(
        np.asarray(x_values, dtype=np.float64),
        x_grid[order],
        y_grid[order],
        left=float(y_grid[order][0]),
        right=float(y_grid[order][-1]),
    ).astype(np.float64)


def pchip_control_curve(
    term: EditableTerm,
    x_ctrl: NDArray,
    target_ctrl: NDArray,
) -> NDArray:
    from scipy.interpolate import PchipInterpolator

    if term.x is None:
        raise TypeError(f"Term {term.name!r} does not expose an x grid.")
    x_grid = np.asarray(term.x, dtype=np.float64).ravel()
    x = np.asarray(x_ctrl, dtype=np.float64).ravel()
    y = np.asarray(target_ctrl, dtype=np.float64).ravel()
    order = np.argsort(x)
    return np.asarray(PchipInterpolator(x[order], y[order])(x_grid), dtype=np.float64)


def _basis_support_centers(basis: NDArray, term: EditableTerm) -> NDArray | None:
    # Basis-weighted centers usually place handles where each basis function has
    # visible influence, which is more intuitive than uniformly spaced controls.
    if term.x is None:
        return None
    x_grid = np.asarray(term.x, dtype=np.float64).ravel()
    weights = np.maximum(np.asarray(basis, dtype=np.float64), 0.0)
    totals = np.sum(weights, axis=0)
    if np.any(totals <= 1e-14):
        return None
    return np.asarray((weights.T @ x_grid) / totals, dtype=np.float64)


def _greville_abscissae(spec, n_basis: int, term: EditableTerm) -> NDArray:
    # Greville abscissae are a standard x-location proxy for B-spline control
    # coefficients. Use them when basis support centers are unavailable.
    knots = getattr(spec, "_knots", None)
    degree = int(getattr(spec, "degree", 3))
    if knots is not None and degree > 0:
        knots_arr = np.asarray(knots, dtype=np.float64)
        if knots_arr.size >= n_basis + degree + 1:
            x = np.array(
                [np.mean(knots_arr[i + 1 : i + degree + 1]) for i in range(n_basis)],
                dtype=np.float64,
            )
            if term.x is not None:
                x_grid = np.asarray(term.x, dtype=np.float64)
                x = np.clip(x, float(np.min(x_grid)), float(np.max(x_grid)))
            return x

    if term.x is None:
        return np.arange(n_basis, dtype=np.float64)
    x_grid = np.asarray(term.x, dtype=np.float64).ravel()
    return np.linspace(float(np.min(x_grid)), float(np.max(x_grid)), n_basis)


def _control_handle_limits(n_basis: int) -> tuple[int, int]:
    max_handles = min(max(int(n_basis), 3), 24)
    min_handles = min(3, max_handles)
    return min_handles, max_handles


def _control_handle_count(n_basis: int, n_handles: int | None) -> int:
    min_handles, max_handles = _control_handle_limits(n_basis)
    if n_handles is None:
        return min(max_handles, 12)
    return int(np.clip(int(n_handles), min_handles, max_handles))


def _control_basis_indices(n_basis: int, n_handles: int | None = None) -> NDArray[np.intp]:
    count = _control_handle_count(n_basis, n_handles)
    if count >= n_basis:
        return np.arange(n_basis, dtype=np.intp)
    return np.unique(np.linspace(0, n_basis - 1, count, dtype=np.intp)).astype(np.intp)


def _as_dense_matrix(matrix) -> NDArray:
    if hasattr(matrix, "toarray"):
        return np.asarray(matrix.toarray(), dtype=np.float64)
    return np.asarray(matrix, dtype=np.float64)
