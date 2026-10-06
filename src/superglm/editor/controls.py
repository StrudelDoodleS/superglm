"""Spline control-handle helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from superglm.editor._types import EditableTerm
from superglm.editor.errors import EditorIndexError, EditorTypeError

# Term types whose control handles are recovered from a fitted basis rather than
# drawn as a display-only fallback.  Kept here, where the recovery lives, so the
# two callers that gate on it (`EditorSession._require_control_term` and
# `payloads._controls_payload`) cannot drift apart.  An ordered categorical is
# not listed: only one with a spline basis has handles, and
# `ordered_spline_geometry` is the gate both callers ask about it.
CONTROL_HANDLE_TERM_TYPES = ("spline", "piecewise")

# Points per gap between adjacent levels on the grid an ordered spline is drawn
# on; the grid also holds every level position exactly.
ORDERED_SPLINE_GRID_STEPS = 24

# Why an ordered term with a spline basis shows no handles: one fixed sentence
# per reason, shown on the disabled Handles tool.
ORDERED_SPLINE_GROUPED = (
    "Handles are off while levels are grouped. Ungroup them to edit the spline."
)
ORDERED_SPLINE_SHAPED = "Handles are off once a band is shaped. Undo the shape to edit the spline."
ORDERED_SPLINE_POSITIONS = "Handles need each level at its own place on the spline's axis."
ORDERED_SPLINE_UNAVAILABLE = "Handles are not available for this spline."

_UNIT_ROUNDOFF = 2.0**-53


_SUBNORMAL_SPACING = 2.0**-1074


def _gamma(count: int) -> float:
    """Higham's ``gamma_k = k u / (1 - k u)`` for ``k`` roundings, ``u = 2^-53``."""
    product = count * _UNIT_ROUNDOFF
    return product / (1.0 - product) if product < 1.0 else float("inf")


@dataclass(frozen=True)
class OrderedSplineGeometry:
    """An ordered term's fitted spline on its level axis, in display coordinates.

    Smooth level ``i`` sits at display x ``i``; between levels the spline's own
    positions (``linspace(0, 1)`` or the user's ``values=``) map linearly onto
    the display axis.  ``level_basis`` ``(S, K)`` and ``grid_basis`` ``(G, K)``
    are the inner spline's raw basis at the level positions and on a grid of
    ``ORDERED_SPLINE_GRID_STEPS`` points per gap; grid row
    ``ORDERED_SPLINE_GRID_STEPS * i`` is evaluated at level ``i``'s position
    itself.  ``fitted`` are the in-force fit's raw coefficients less the curve
    at the reporting base, so ``level_basis @ fitted`` is the displayed level
    effects: every basis row sums to one, so subtracting a constant from every
    coefficient subtracts it from the curve.  ``live`` lists, in order, the
    columns with positive mass on the grid; only they get handles.
    """

    level_index: NDArray[np.intp]
    level_basis: NDArray
    grid_x: NDArray
    grid_basis: NDArray
    handle_x: NDArray
    live: NDArray[np.intp]
    fitted: NDArray
    n_points: int


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
            raise EditorIndexError(f"Control handle index out of range for term {term.name!r}.")
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
        raise EditorIndexError(f"Control handle index out of range for term {term.name!r}.")
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
        raise EditorTypeError(f"Term {term.name!r} does not expose an x grid.")
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


def ordered_spline_geometry(model, term: EditableTerm) -> OrderedSplineGeometry | str | None:
    """The fitted spline of an ordered term with a spline basis, on its level axis.

    ``None`` for every other term.  A fixed sentence instead of the geometry
    when the term's handles are off: a grouping or a shaped band changes what
    the coefficients mean, and a fitted curve that does not reproduce the
    reported level effects to round-off is refused rather than drawn.
    """
    from superglm.features.ordered_categorical import OrderedCategorical

    spec = None if model is None else getattr(model, "_specs", {}).get(term.name)
    if not isinstance(spec, OrderedCategorical) or spec.basis_kind != "spline":
        return None
    if spec._grouping is not None:
        return ORDERED_SPLINE_GROUPED
    inner = spec._basis_spline
    if inner.polynomial_ranges:
        return ORDERED_SPLINE_SHAPED
    smooth = list(spec._smooth_levels)
    if term.levels is None or term.levels[: len(smooth)] != [str(level) for level in smooth]:
        return ORDERED_SPLINE_UNAVAILABLE
    positions = np.asarray([spec._level_to_value[level] for level in smooth], dtype=np.float64)
    if positions.size < 2 or not np.all(np.diff(positions) > 0.0):
        return ORDERED_SPLINE_POSITIONS
    steps = ORDERED_SPLINE_GRID_STEPS
    fractions = np.arange(steps, dtype=np.float64) / steps
    gaps = np.diff(positions)
    grid_positions = np.concatenate(
        [(positions[:-1, None] + gaps[:, None] * fractions).ravel(), positions[-1:]]
    )
    starts = np.arange(positions.size - 1, dtype=np.float64)
    grid_x = np.concatenate([(starts[:, None] + fractions).ravel(), [positions.size - 1.0]])
    base = np.array([spec._level_to_value[spec._base_level]], dtype=np.float64)
    beta = np.concatenate(
        [
            np.asarray(model.result.beta, dtype=np.float64)[group.sl]
            for group in model._groups
            if group.feature_name == term.name
        ]
    )
    spline_beta, _ = spec._split_beta(beta)
    try:
        level_basis = _as_dense_matrix(inner._basis_matrix(positions))
        grid_basis = _as_dense_matrix(inner._basis_matrix(grid_positions))
        base_row = _as_dense_matrix(inner._basis_matrix(base))[0]
    except ValueError:
        # extrapolation="error" and a declared level outside the fitted range.
        return ORDERED_SPLINE_UNAVAILABLE
    coefficient_map = _raw_coefficient_map(inner, spline_beta.size)
    if coefficient_map.shape != (level_basis.shape[1], spline_beta.size):
        return ORDERED_SPLINE_UNAVAILABLE
    raw = coefficient_map @ spline_beta
    fitted = raw - float(base_row @ raw)
    effects = np.asarray(term.original_log_effect, dtype=np.float64)[: positions.size]
    bound = _certification_bound(level_basis, base_row, coefficient_map, spline_beta, inner)
    # A bound that overflowed (|M| |beta| on an ill-scaled fit) certifies nothing.
    if not (np.all(np.isfinite(bound)) and np.all(np.isfinite(fitted))):
        return ORDERED_SPLINE_UNAVAILABLE
    if not np.all(np.abs(level_basis @ fitted - effects) <= bound):
        return ORDERED_SPLINE_UNAVAILABLE
    centres = _handle_centres(grid_basis, grid_x)
    live = np.flatnonzero(np.isfinite(centres)).astype(np.intp)
    if live.size < 3:
        return ORDERED_SPLINE_UNAVAILABLE
    return OrderedSplineGeometry(
        level_index=np.arange(positions.size, dtype=np.intp),
        level_basis=level_basis,
        grid_x=grid_x,
        grid_basis=grid_basis,
        handle_x=centres,
        live=live,
        fitted=fitted,
        n_points=int(term.size),
    )


def least_change_coefficients(
    geometry: OrderedSplineGeometry,
    effects: NDArray,
    prior: NDArray | list[float] | None = None,
) -> NDArray:
    """The coefficients behind ``effects``: the least change from ``prior`` that reproduces them.

    ``prior`` is the vector the latest handle move wrote, else the fit's.  The
    level points alone cannot say which coefficients they came from -- the
    basis usually has more columns than there are levels -- so the
    minimum-norm correction ``lstsq`` returns keeps every coefficient the
    edits do not need to move.  With no more levels than the basis can
    interpolate, the corrected spline passes through every edited level;
    otherwise it is their least-squares spline.
    """
    start = geometry.fitted if prior is None else np.asarray(prior, dtype=np.float64)
    if start.shape != geometry.fitted.shape:
        start = geometry.fitted
    smooth = np.asarray(effects, dtype=np.float64)[geometry.level_index]
    residual = smooth - geometry.level_basis @ start
    correction = np.linalg.lstsq(geometry.level_basis, residual, rcond=None)[0]
    return np.asarray(start + correction, dtype=np.float64)


def ordered_handle_columns(
    geometry: OrderedSplineGeometry, n_handles: int | None = None
) -> NDArray[np.intp]:
    """The basis columns that carry a handle, thinned like a numeric spline's."""
    return geometry.live[_control_basis_indices(geometry.live.size, n_handles=n_handles)]


def ordered_control_points(
    geometry: OrderedSplineGeometry,
    coefficients: NDArray,
    *,
    n_handles: int | None = None,
) -> dict:
    """Handles for an ordered spline, in the shape ``control_points`` returns.

    ``basis`` rows run over the term's display points (levels, then specials
    at zero), so the browser's drag preview moves the level dots exactly as
    for a numeric spline.  ``build_basis`` rows run over the drawing grid
    ``grid_x``, and ``basis_index`` indexes them.
    """
    handles = ordered_handle_columns(geometry, n_handles)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    basis = np.zeros((handles.size, geometry.n_points), dtype=np.float64)
    basis[:, geometry.level_index] = geometry.level_basis[:, handles].T
    min_handles, max_handles = _control_handle_limits(geometry.live.size)
    return {
        "x": geometry.handle_x[handles].copy(),
        "log_effect": coefficients[handles].copy(),
        "basis_index": np.searchsorted(geometry.live, handles).astype(np.intp),
        "basis": basis,
        "build_basis": np.asarray(geometry.grid_basis[:, geometry.live].T, dtype=np.float64),
        "build_log_effect": coefficients[geometry.live].copy(),
        "grid_x": geometry.grid_x.copy(),
        "min_handles": min_handles,
        "max_handles": max_handles,
    }


def ordered_control_after_move(
    geometry: OrderedSplineGeometry,
    coefficients: NDArray,
    handle_index: int,
    log_effect: float,
    *,
    n_handles: int | None = None,
) -> tuple[NDArray, int]:
    """The coefficients with one handle's set to ``log_effect``, and its column."""
    handles = ordered_handle_columns(geometry, n_handles)
    if handle_index < 0 or handle_index >= handles.size:
        raise EditorIndexError("Control handle index out of range for this term.")
    column = int(handles[handle_index])
    moved = np.asarray(coefficients, dtype=np.float64).copy()
    moved[column] = float(log_effect)
    return moved, column


def spline_fits_levels(geometry: OrderedSplineGeometry, term: EditableTerm, records) -> bool:
    """Whether the spline the coefficients draw passes through the edited levels.

    Decided by construction, not by a tolerance: it does when the basis
    interpolates any level values (full row rank at the level positions), when
    no smooth level is edited, or when the latest edit touching the smooth
    levels was a handle move that wrote all of them.  ``matrix_rank`` uses
    numpy's default cutoff; the answer only picks what the chart draws.
    """
    smooth = geometry.level_index
    if np.linalg.matrix_rank(geometry.level_basis) == smooth.size:
        return True
    if np.array_equal(term.edited_log_effect[smooth], term.original_log_effect[smooth]):
        return True
    for record in reversed(records):
        if record.term != term.name or not np.intersect1d(record.indices, smooth).size:
            continue
        return "coefficients" in record.params and record.indices.size == smooth.size
    return False


def _raw_coefficient_map(inner, width: int) -> NDArray:
    """The matrix taking the fitted coefficients to the raw basis coefficients.

    ``transform`` evaluates ``B @ R_inv`` (or, for a SCOP monotone fit,
    ``(B @ Sigma)[:, null_dim:]`` less a column-mean constant), so the raw
    coefficients are this matrix times beta, up to that constant, which the
    base shift removes.
    """
    sigma = getattr(inner, "_scop_Sigma", None)
    if sigma is not None:
        drop = int(getattr(inner, "_scop_null_dim", 1))
        return np.asarray(sigma, dtype=np.float64)[:, drop:]
    r_inv = getattr(inner, "_R_inv", None)
    if r_inv is None:
        return np.eye(width, dtype=np.float64)
    return np.asarray(r_inv, dtype=np.float64)


def _certification_bound(level_basis, base_row, coefficient_map, beta, inner) -> NDArray:
    """Per level, how far two float64 evaluations of the base-relative curve can differ.

    The reported effect is ``fl(fl(b_i M) beta) - fl(fl(b_0 M) beta)``; ours is
    ``fl(b_i fl(M beta) - fl(b_0 fl(M beta)))``.  Each is within
    ``gamma_{K+p+3} (1 + |b_i|_1) (a_i + a_0)`` of the exact value, where
    ``a_i = |b_i| |M| |beta|`` (Higham 2002, sections 3.1 and 3.5), plus the
    SCOP column-mean constant's magnitude when the fit carries one.  Twice
    that, and a ``gamma`` whose count also covers forming ``a_i`` from
    non-negative terms, keeps the computed bound an upper bound.

    That bound is relative, and rounds to zero when the effects are
    subnormal. Under gradual underflow each product also carries an absolute
    error of at most ``eta = 2^-1075``, half the subnormal spacing, while a
    sum that underflows is exact (Demmel, *Underflow and the Reliability of
    Numerical Software*, SIAM J. Sci. Stat. Comput. 5(4), 1984). The four
    evaluations form at most ``K + p + 1`` products a level, whose errors the
    later sums and products carry at most ``(1 + |beta|_1)(1 + |b|_1)`` times,
    so ``count (1 + |beta|_1)(1 + |b|_1)`` subnormal spacings ``2 eta``, each
    product and sum of it rounded outward, bound the absolute part.
    """
    columns = level_basis.shape[1]
    weights = np.abs(coefficient_map) @ np.abs(beta)
    constant = 0.0
    means = getattr(inner, "_scop_col_means", None)
    if getattr(inner, "_scop_Sigma", None) is not None and means is not None:
        constant = float(np.abs(np.asarray(means, dtype=np.float64)) @ np.abs(beta))
    level_magnitude = np.abs(level_basis) @ weights + constant
    base_magnitude = float(np.abs(base_row) @ weights) + constant
    row_norm = np.maximum(1.0, np.sum(np.abs(level_basis), axis=1))
    count = 2 * (columns + beta.size) + 6
    relative = 4.0 * _gamma(count) * row_norm * (level_magnitude + base_magnitude)
    up = np.inf
    beta_mass = np.nextafter(1.0 + _sum_up(np.abs(beta)), up)
    basis_mass = np.nextafter(
        1.0 + max(_sum_up(np.abs(level_basis), axis=1).max(), _sum_up(np.abs(base_row))), up
    )
    spacings = np.nextafter(np.nextafter(count * beta_mass, up) * basis_mass, up)
    absolute = np.nextafter(spacings * _SUBNORMAL_SPACING, up)
    return np.nextafter(relative + absolute, up)


def _sum_up(values: NDArray, axis: int | None = None):
    """An upper bound on the sum of non-negative ``values``: the float64 sum times ``1 + gamma``.

    ``gamma``'s count covers the ``n - 1`` roundings of the sum, forming and
    applying the factor (two), and the two of ``gamma`` itself.
    """
    n = values.shape[-1] if axis is not None else values.size
    return np.sum(values, axis=axis) * (1.0 + _gamma(n + 3))


def _handle_centres(grid_basis: NDArray, grid_x: NDArray) -> NDArray:
    """Each column's basis-weighted centre on the display axis; NaN without visible mass.

    The rule a numeric spline's handles use (``_basis_support_centers``), taken
    on the uniform display grid.  The B-spline collocation kernel is totally
    positive (de Boor, "Total positivity of the spline collocation matrix",
    Indiana Univ. Math. J. 25, 1976), so these centres never decrease from one
    column to the next; a cardinal basis's negative lobes are left out of the
    mass.  Greville abscissae are not the fallback here: a P-spline's open knot
    vector puts its end columns' abscissae outside the level range, which
    stacked two handles on each end level of the browser fixture's ``age_band``.
    """
    mass = np.maximum(grid_basis, 0.0)
    totals = np.sum(mass, axis=0)
    centres = np.full(totals.size, np.nan, dtype=np.float64)
    visible = totals > 0.0
    centres[visible] = (mass[:, visible].T @ grid_x) / totals[visible]
    return centres
