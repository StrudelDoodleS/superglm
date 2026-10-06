"""Shaped ranges: pin a chosen range of a spline term to a low-degree polynomial."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any

import numpy as np

from superglm._frame import as_eager_frame
from superglm.dm_builder import resolve_discrete_n_bins, should_discretize
from superglm.editor.errors import EditorValueError
from superglm.features._spline_ranges import NARROWEST_GAP, SHAPE_NAMES, PolynomialRange
from superglm.features._spline_runtime import fit_support
from superglm.features.ordered_categorical import OrderedCategorical
from superglm.features.rebuild import (
    TOO_FEW_POINTS,
    RangePlacementError,
    band_edges,
    base_names_level,
    current_ranges,
    edge_text,
    merged_ranges,
    pristine_basis,
    rebuilt_ordered_spec,
    shape_unavailable_reason,
    shaped_spline,
    source_spline,
)
from superglm.features.spline import _SplineBase

# Read by model/report_ops.py BY NAME, so the model layer never imports the
# editor: a basis carrying it had its shape chosen in the editor from this
# data, so its tests are conditional on it.
EDITOR_CHOSEN_SHAPE_ATTRIBUTE = "_editor_chosen_shape"


def shape_availability(model, name: str) -> tuple[bool, str | None]:
    """Whether ``name`` can take a shaped range, and the hover reason when it can't."""
    reason = shape_unavailable_reason(model, name)
    return reason is None, reason


# The joins the editor offers: Tangent, and Corner (the library's kink).
EDITOR_JOINS = ("tangent", "kink")
_LINEAR_TANGENT = "A degree-1 spline cannot join a range along its tangent; choose Corner."


def shape_payload(model, name: str, support: dict[str, list[int]] | None) -> dict[str, Any]:
    """The palette's state for one term: availability, the ranges in force, ``support``.

    ``specials`` names an ordered term's special levels as the axis shows them.
    """
    spec = model._specs[name]
    available, reason = shape_availability(model, name)
    ranges = [
        {"lo": r.lo, "hi": r.hi, "degree": r.degree, "label": r.label, "join": r.join}
        for r in current_ranges(spec)
    ]
    specials = spec._special_display if isinstance(spec, OrderedCategorical) else ()
    linear = available and source_spline(spec).degree < 2
    return {
        "available": available,
        "reason": reason,
        "ranges": ranges,
        "support": support,
        "specials": [str(level) for level in specials],
        "joins": ["kink"] if linear else list(EDITOR_JOINS),
        "join_reason": _LINEAR_TANGENT if linear else None,
    }


def waiting_ranges(draft, fitted) -> list[dict[str, Any]]:
    """The ranges ``draft`` adds or changes against the fitted spec, in axis order.

    Each is listed as the palette lists a range in force.
    """
    in_force = {(r.lo, r.hi, r.degree, r.join) for r in current_ranges(fitted)}
    ranges = current_ranges(draft)
    if not isinstance(draft, OrderedCategorical):
        ranges = sorted(ranges, key=lambda r: r.lo)
    return [
        {"lo": r.lo, "hi": r.hi, "degree": r.degree, "label": r.label, "join": r.join}
        for r in ranges
        if (r.lo, r.hi, r.degree, r.join) not in in_force
    ]


def shape_support(model, name: str, grid, X, sample_weight) -> dict[str, list[int]] | None:
    """How many values a numeric term's refit sees below and up to each grid point's edges.

    ``below[k]`` counts those under the lower edge a selection starting at
    grid point k snaps to, and ``through[k]`` those up to the upper edge one
    ending there snaps to, so a run i..j holds ``through[j] - below[i]``: the
    count the library certifies a range with. The values are the refit's:
    distinct positive-weight ones, or occupied bin centres when it bins. None
    unless the term is a numeric spline that can take a shape and its data
    was retained.
    """
    spec = model._specs[name]
    if X is None or not isinstance(spec, _SplineBase) or shape_unavailable_reason(model, name):
        return None
    x = np.asarray(as_eager_frame(X).column_array(name), dtype=np.float64)
    if sample_weight is not None:
        x = x[np.asarray(sample_weight, dtype=np.float64) > 0.0]
    binned = should_discretize(spec, model._discrete)
    n_bins = resolve_discrete_n_bins(name, spec, model._n_bins) if binned else None
    support = fit_support(x, n_bins)
    boundary = spec.fitted_boundary
    lows = [_snapped_edge(boundary, value, -1) for value in grid]
    highs = [_snapped_edge(boundary, value, 1) for value in grid]
    return {
        "below": np.searchsorted(support, lows).tolist(),
        "through": np.searchsorted(support, highs, side="right").tolist(),
    }


def shaped_feature_spec(
    model, name: str, *, lo, hi, degree: int, join: str = "tangent", X, draft_spec=None
) -> tuple[Any, dict[str, Any]]:
    """A fresh spec for ``name`` with ``[lo, hi]`` pinned to a ``degree`` polynomial.

    ``join`` is ``"tangent"`` (the curve leaves the range along its slope) or
    ``"kink"`` (the slope may change at the edge). Ranges already in force are
    kept; the same range with a new degree or join replaces the old one, and
    any other overlap is refused by name. A
    numeric term keeps its fitted base knots and boundary, so the free
    part's knots never move; an ordered term rebuilds from its declaration,
    whose placement is deterministic on the same level axis.

    ``draft_spec`` is the term's spec as waiting changes leave it (None: the
    fitted spec); a numeric draft is the unfitted spline an earlier waiting
    shape built, and keeps the fitted knots and boundary it states.
    """
    reason = shape_unavailable_reason(model, name)
    if reason is not None:
        raise EditorValueError(reason)
    if not _is_shape_degree(degree):
        raise EditorValueError("Choose a shape: Flat, Line, Quadratic or Cubic.")
    if join not in EDITOR_JOINS:
        raise EditorValueError("Choose a join: Tangent or Corner.")
    spec = model._specs[name] if draft_spec is None else draft_spec
    if join == "tangent" and source_spline(spec).degree < 2:
        raise EditorValueError(_LINEAR_TANGENT)
    ordered = isinstance(spec, OrderedCategorical)
    position = spec._range_edge_value if ordered else float
    try:
        lo, hi = band_edges(spec, name, lo, hi) if ordered else _numeric_edges(spec, lo, hi)
        new = PolynomialRange(lo, hi, degree, join)
        ranges = merged_ranges(current_ranges(spec), new, position)
    except RangePlacementError as exc:
        raise EditorValueError(str(exc)) from exc
    if ordered:
        source = pristine_basis(spec)
        knots = source._named_knots or source._explicit_knots
        boundary = source._explicit_boundary
    else:
        source = spec
        knots, boundary = _free_geometry(spec)
    basis = shaped_spline(source, ranges, knots=knots, boundary=boundary)
    setattr(basis, EDITOR_CHOSEN_SHAPE_ATTRIBUTE, True)
    replacement = _hosted(spec, basis, name, X) if ordered else basis
    span = f"{edge_text(lo)}–{edge_text(hi)}"
    return replacement, {
        "format": "superglm.editor.shaped_range.v1",
        "term": name,
        "lo": lo,
        "hi": hi,
        "degree": degree,
        "join": join,
        "label": f"{new.label} {span} in {name}",
        "message": f"{name} was given a {new.label} range {span} in the editor "
        "and the full model was refit.",
    }


def snap_edge(value: float, span: float, direction: int) -> float:
    """``value`` on the grid of three significant figures of ``span``, rounded outward.

    ``direction`` is -1 for a lower edge and +1 for an upper one, so the
    snapped range still holds every selected point. Rounding the grid
    multiple to its decimal places drops the binary residue of ``k * step``.
    """
    exponent = math.floor(math.log10(span)) - 2
    step = 10.0**exponent
    places = max(0, -exponent)
    k = round(value / step)
    snapped = round(k * step, places)
    if (snapped - value) * direction < 0:
        snapped = round((k + direction) * step, places)
    return snapped


def _numeric_edges(spec, lo, hi) -> tuple[float, float]:
    """Snap selected values outward onto the fitted span's grid, clipped to the boundary.

    An edge snapped onto (or past) the boundary is the boundary exactly, so
    it is never inserted as a knot a round-off away from the end.
    """
    if not (_is_finite(lo) and _is_finite(hi)):
        raise EditorValueError("Range edges on a numeric term must be finite numbers.")
    if not lo < hi:
        raise EditorValueError(TOO_FEW_POINTS)
    boundary = _free_geometry(spec)[1]
    return _snapped_edge(boundary, float(lo), -1), _snapped_edge(boundary, float(hi), 1)


def _free_geometry(spec) -> tuple[Any, tuple[float, float]]:
    """The base knots and boundary a numeric term keeps when it is shaped.

    A fitted spline reports them. A draft, the unfitted spline an earlier
    waiting shape built (``shaped_spline``), states the fitted ones it kept.
    """
    if spec.fitted_boundary is not None:
        return spec.fitted_base_knots, spec.fitted_boundary
    return spec._explicit_knots, spec._explicit_boundary


def _snapped_edge(boundary: tuple[float, float], value: float, direction: int) -> float:
    """``snap_edge`` on the fitted span, clipped to the boundary before and after snapping.

    Clipping first keeps an edge far past the boundary from overflowing the grid.
    A boundary off the grid can leave a snapped edge a hair inside it, a free
    end the library refuses as too narrow, so an edge that close goes onto the
    boundary: still outward, so the range still holds every selected point.
    """
    b_lo, b_hi = boundary
    snapped = snap_edge(min(max(value, b_lo), b_hi), b_hi - b_lo, direction)
    edge = max(b_lo, snapped) if direction < 0 else min(b_hi, snapped)
    end = b_lo if direction < 0 else b_hi
    return end if abs(edge - end) < NARROWEST_GAP * (b_hi - b_lo) else edge


def _hosted(spec: OrderedCategorical, basis, name: str, X) -> OrderedCategorical:
    """A fresh ordered term around ``basis``, keeping order, specials, grouping, base."""
    frame = as_eager_frame(X)
    frame.require_columns((name,))
    return rebuilt_ordered_spec(
        spec,
        grouping=getattr(spec, "_grouping", None),
        base=spec.base,
        data=frame.column_array(name),
        basis=basis,
        level=base_names_level(spec),
    )


def _is_finite(value) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _is_shape_degree(value) -> bool:
    """An integer naming a shape; a bool or a float such as 2.9 names none."""
    integer = isinstance(value, Integral) and not isinstance(value, bool)
    return integer and value in range(len(SHAPE_NAMES))
