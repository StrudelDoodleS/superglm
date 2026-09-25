"""Shaped ranges: pin a chosen range of a spline term to a low-degree polynomial."""

from __future__ import annotations

import math
from collections.abc import Callable
from numbers import Integral, Real
from typing import Any

import numpy as np

from superglm._frame import as_eager_frame
from superglm.dm_builder import resolve_discrete_n_bins, should_discretize
from superglm.editor.collapse import (
    _pristine_basis,
    interaction_users,
    rebuilt_ordered_spec,
    special_labels,
)
from superglm.editor.errors import EditorValueError
from superglm.features._spline_ranges import SHAPE_NAMES, PolynomialRange
from superglm.features._spline_runtime import fit_support
from superglm.features.ordered_categorical import OrderedCategorical, _spline_kind_name
from superglm.features.spline import CardinalCRSpline, Spline, _SplineBase

# Read by model/report_ops.py BY NAME, so the model layer never imports the
# editor: a basis carrying it had its shape chosen in the editor from this
# data, so its tests are conditional on it.
EDITOR_CHOSEN_SHAPE_ATTRIBUTE = "_editor_chosen_shape"
_TOO_FEW_POINTS = "Select at least two points to shape a range."


def shape_availability(model, name: str) -> tuple[bool, str | None]:
    """Whether ``name`` can take a shaped range, and the hover reason when it can't."""
    reason = _unavailable_reason(model, name)
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
        for r in _current_ranges(spec)
    ]
    specials = spec._special_display if isinstance(spec, OrderedCategorical) else ()
    linear = available and _source_spline(spec).degree < 2
    return {
        "available": available,
        "reason": reason,
        "ranges": ranges,
        "support": support,
        "specials": [str(level) for level in specials],
        "joins": ["kink"] if linear else list(EDITOR_JOINS),
        "join_reason": _LINEAR_TANGENT if linear else None,
    }


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
    if X is None or not isinstance(spec, _SplineBase) or _unavailable_reason(model, name):
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
    model, name: str, *, lo, hi, degree: int, join: str = "tangent", X
) -> tuple[Any, dict[str, Any]]:
    """A fresh spec for ``name`` with ``[lo, hi]`` pinned to a ``degree`` polynomial.

    ``join`` is ``"tangent"`` (the curve leaves the range along its slope) or
    ``"kink"`` (the slope may change at the edge). Ranges already in force are
    kept; the same range with a new degree or join replaces the old one, and
    any other overlap is refused by name. A
    numeric term keeps its fitted base knots and boundary, so the free
    part's knots never move; an ordered term rebuilds from its declaration,
    whose placement is deterministic on the same level axis.
    """
    reason = _unavailable_reason(model, name)
    if reason is not None:
        raise EditorValueError(reason)
    if not _is_shape_degree(degree):
        raise EditorValueError("Choose a shape: Flat, Line, Quadratic or Cubic.")
    if join not in EDITOR_JOINS:
        raise EditorValueError("Choose a join: Tangent or Corner.")
    if join == "tangent" and _source_spline(model._specs[name]).degree < 2:
        raise EditorValueError(_LINEAR_TANGENT)
    spec = model._specs[name]
    ordered = isinstance(spec, OrderedCategorical)
    position = spec._range_edge_value if ordered else float
    lo, hi = _band_edges(spec, name, lo, hi) if ordered else _numeric_edges(spec, lo, hi)
    new = PolynomialRange(lo, hi, degree, join)
    ranges = _merged_ranges(_current_ranges(spec), new, position)
    if ordered:
        source = _pristine_basis(spec)
        knots = source._named_knots or source._explicit_knots
        boundary = source._explicit_boundary
    else:
        source, knots, boundary = spec, spec.fitted_base_knots, spec.fitted_boundary
    basis = _shaped_spline(source, ranges, knots=knots, boundary=boundary)
    setattr(basis, EDITOR_CHOSEN_SHAPE_ATTRIBUTE, True)
    replacement = _hosted(spec, basis, name, X) if ordered else basis
    span = f"{_edge_text(lo)}–{_edge_text(hi)}"
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


def _unavailable_reason(model, name: str) -> str | None:
    source = _source_spline(model._specs[name])
    if source is None:
        return "Shapes need a spline term."
    if isinstance(source, CardinalCRSpline):
        return "Shapes are not available for cardinal cubic regression splines."
    if source.constraint_kind is not None:
        return "Remove the term's shape constraint to add shaped ranges."
    if source.select:
        return "Remove select=True from the term to add shaped ranges."
    if max(source._m_orders) > source.degree:
        # A shaped term is rebuilt with a derivative penalty, whose order the
        # degree bounds; a difference penalty (ps) is not bounded so.
        return "Shapes need a penalty order no higher than the spline's degree."
    if interaction_users(model, name):
        return "A term used by an interaction cannot be reshaped."
    return None


def _source_spline(spec) -> _SplineBase | None:
    """The spline a term is declared with: its own spec, or an ordered term's basis."""
    basis = getattr(spec, "_spline_obj", None) if isinstance(spec, OrderedCategorical) else spec
    return basis if isinstance(basis, _SplineBase) else None


def _current_ranges(spec) -> tuple[PolynomialRange, ...]:
    """The ranges in force, in axis order: band names on an ordered term, values otherwise."""
    source = _source_spline(spec)
    if source is None:
        return ()
    if not isinstance(spec, OrderedCategorical):
        return source.polynomial_ranges
    return tuple(sorted(source.polynomial_ranges, key=lambda r: spec._range_edge_value(r.lo)))


def _numeric_edges(spec, lo, hi) -> tuple[float, float]:
    """Snap selected values outward onto the fitted span's grid, clipped to the boundary.

    An edge snapped onto (or past) the boundary is the boundary exactly, so
    it is never inserted as a knot a round-off away from the end.
    """
    if not (_is_finite(lo) and _is_finite(hi)):
        raise EditorValueError("Range edges on a numeric term must be finite numbers.")
    if not lo < hi:
        raise EditorValueError(_TOO_FEW_POINTS)
    boundary = spec.fitted_boundary
    return _snapped_edge(boundary, float(lo), -1), _snapped_edge(boundary, float(hi), 1)


def _snapped_edge(boundary: tuple[float, float], value: float, direction: int) -> float:
    """``snap_edge`` on the fitted span, clipped to the boundary before and after snapping.

    Clipping first keeps an edge far past the boundary from overflowing the grid.
    """
    b_lo, b_hi = boundary
    snapped = snap_edge(min(max(value, b_lo), b_hi), b_hi - b_lo, direction)
    return max(b_lo, snapped) if direction < 0 else min(b_hi, snapped)


def _band_edges(spec: OrderedCategorical, name: str, lo, hi) -> tuple[str, str]:
    """Two single bands in axis order; a group, a special or an unknown label refuses."""
    lo, hi = str(lo), str(hi)
    if {lo, hi} & special_labels(spec):
        raise EditorValueError(
            f"A shaped range covers only the bands of {name}; "
            "leave its special levels out of the selection."
        )
    try:
        at = {lo: spec._range_edge_value(lo), hi: spec._range_edge_value(hi)}
    except ValueError as exc:
        raise EditorValueError(
            f"A shaped range must start and end on single bands of {name}; "
            "ungroup the bands at its ends first."
        ) from exc
    if at[lo] == at[hi]:
        raise EditorValueError(_TOO_FEW_POINTS)
    lo, hi = sorted((lo, hi), key=at.__getitem__)
    return lo, hi


def _merged_ranges(
    existing: tuple[PolynomialRange, ...],
    new: PolynomialRange,
    position: Callable[[Any], float],
) -> list[PolynomialRange]:
    """``existing`` plus ``new``: the same range is replaced, any other overlap refused."""
    span = (position(new.lo), position(new.hi))
    kept = []
    for current in existing:
        at = (position(current.lo), position(current.hi))
        if at == span:
            continue
        if at[0] < span[1] and span[0] < at[1]:
            raise EditorValueError(
                f"This range overlaps the {current.label} range "
                f"{_edge_text(current.lo)}–{_edge_text(current.hi)}. "
                "Undo it or choose a range outside it."
            )
        kept.append(current)
    return [*kept, new]


def _shaped_spline(source: _SplineBase, ranges, *, knots, boundary) -> _SplineBase:
    """``source``'s settings with ``ranges``; a ``ps``/``ns`` source becomes ``bs``.

    Range edges repeat knots, which the equal-spacing difference penalties
    cannot take; a ``bs`` with the same knots, degree and penalty order is
    the derivative-penalty spline whose penalty can skip the pinned ranges.
    """
    return Spline(
        kind="cr" if _spline_kind_name(source) == "cr" else "bs",
        n_knots=source.n_knots,
        knots=knots,
        boundary=boundary,
        degree=source.degree,
        knot_strategy=source.knot_strategy,
        knot_alpha=source.knot_alpha,
        penalty=source.penalty,
        extrapolation=source.extrapolation,
        discrete=source.discrete,
        n_bins=source.n_bins,
        m=source._m_orders,
        lambda_policy=source._lambda_policy,
        polynomial_ranges=ranges,
    )


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
    )


def _edge_text(edge) -> str:
    return edge if isinstance(edge, str) else f"{edge:g}"


def _is_finite(value) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _is_shape_degree(value) -> bool:
    """An integer naming a shape; a bool or a float such as 2.9 names none."""
    integer = isinstance(value, Integral) and not isinstance(value, bool)
    return integer and value in range(len(SHAPE_NAMES))
