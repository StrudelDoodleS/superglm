"""Shaped ranges: pin a chosen range of a spline term to a low-degree polynomial."""

from __future__ import annotations

import math
from collections.abc import Callable
from numbers import Real
from typing import Any

from superglm._frame import as_eager_frame
from superglm.editor.collapse import _pristine_basis, interaction_users, rebuilt_ordered_spec
from superglm.editor.errors import EditorValueError
from superglm.features._spline_ranges import SHAPE_NAMES, PolynomialRange
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


def shape_payload(model, name: str) -> dict[str, Any]:
    """The palette's state for one term: availability and the ranges in force."""
    available, reason = shape_availability(model, name)
    ranges = [
        {"lo": r.lo, "hi": r.hi, "degree": r.degree, "label": r.label}
        for r in _current_ranges(model._specs[name])
    ]
    return {"available": available, "reason": reason, "ranges": ranges}


def shaped_feature_spec(model, name: str, *, lo, hi, degree: int, X) -> tuple[Any, dict[str, Any]]:
    """A fresh spec for ``name`` with ``[lo, hi]`` pinned to a ``degree`` polynomial.

    Ranges already in force are kept; the same range with a new degree
    replaces the old one, and any other overlap is refused by name. A
    numeric term keeps its fitted base knots and boundary, so the free
    part's knots never move; an ordered term rebuilds from its declaration,
    whose placement is deterministic on the same level axis.
    """
    reason = _unavailable_reason(model, name)
    if reason is not None:
        raise EditorValueError(reason)
    if degree not in range(len(SHAPE_NAMES)):
        raise EditorValueError("Choose a shape: Flat, Line, Quadratic or Cubic.")
    spec = model._specs[name]
    ordered = isinstance(spec, OrderedCategorical)
    position = spec._range_edge_value if ordered else float
    lo, hi = _band_edges(spec, name, lo, hi) if ordered else _numeric_edges(spec, lo, hi)
    new = PolynomialRange(lo, hi, degree)
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
    b_lo, b_hi = spec.fitted_boundary
    span = b_hi - b_lo
    return max(b_lo, snap_edge(float(lo), span, -1)), min(b_hi, snap_edge(float(hi), span, 1))


def _band_edges(spec: OrderedCategorical, name: str, lo, hi) -> tuple[str, str]:
    """Two single bands in axis order; a group, a special or an unknown label refuses."""
    lo, hi = str(lo), str(hi)
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
                "Restore it or choose a range outside it."
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
