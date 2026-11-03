"""Subclass-specific spline helper operations."""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm.features import _shape_rule_rows
from superglm.features.constraint import CONSTRAINT_KINDS, rules_of
from superglm.types import LinearConstraintSet


def assemble_open_knot_vector(spec: Any, interior: NDArray) -> None:
    """Build the padded open knot vector used by unconstrained B-splines."""
    x_range = spec._hi - spec._lo
    lo_effective = spec._lo - 0.001 * x_range
    hi_effective = spec._hi + 0.001 * x_range

    inner = np.concatenate([[lo_effective], interior, [hi_effective]])
    dx_lo = inner[1] - inner[0]
    dx_hi = inner[-1] - inner[-2]

    lower = lo_effective - dx_lo * np.arange(spec.degree, 0, -1)
    upper = hi_effective + dx_hi * np.arange(1, spec.degree + 1)

    spec._knots = np.concatenate([lower, inner, upper])
    spec._n_basis = len(spec._knots) - spec.degree - 1


def assemble_clamped_knot_vector(spec: Any, interior: NDArray) -> None:
    """Build the exact-boundary clamped knot vector used by cubic regression splines."""
    spec._knots = np.concatenate(
        [
            np.repeat(spec._lo, spec.degree + 1),
            interior,
            np.repeat(spec._hi, spec.degree + 1),
        ]
    )
    spec._n_basis = len(spec._knots) - spec.degree - 1


def has_natural_ends(spec: Any) -> bool:
    """Whether the spec's equality rows hold ``f''`` at zero at its boundary.

    True for a cubic or higher ``NaturalSpline`` and for a
    ``CubicRegressionSpline`` with an end no polynomial range reaches.
    """
    from superglm.features.spline import CubicRegressionSpline, NaturalSpline

    if isinstance(spec, NaturalSpline):
        return spec.degree >= 3
    return isinstance(spec, CubicRegressionSpline) and any(spec._free_ends())


def _is_rule_collection(spec: Any) -> bool:
    """Whether the spec's fit-time rules are anything but one full-range shape.

    Reads the legacy ``monotone`` fields of a spec saved before ``constraint_kind``.
    """
    kind = getattr(spec, "constraint_kind", getattr(spec, "monotone", None))
    mode = getattr(spec, "constraint_mode", getattr(spec, "monotone_mode", "postfit"))
    return kind is not None and mode == "fit" and kind not in CONSTRAINT_KINDS


def refuse_max_step(spec: Any) -> None:
    """Refuse a step limit on a spline term, which has no levels to step between."""
    for rule in rules_of(spec):
        if rule.kind == "max_step":
            raise ValueError(
                f"{rule.describe()} limits the step between adjacent levels, and a spline term "
                "has no levels; band the feature with OrderedCategorical to limit its steps, "
                "or drop the rule"
            )


def _collection_pieces(spec: Any, *, stacklevel: int) -> list:
    """The spec's rules as pieces on its range, refusing rules a spline cannot fit."""
    refuse_max_step(spec)
    return _shape_rule_rows.shape_pieces(
        rules_of(spec), spec._lo, spec._hi, stacklevel=stacklevel + 1
    )


def _fmt_end(x: float) -> str:
    return f"{x:.6g}"


def _shown_and_held(spec: Any, rule: Any, lo: float, hi: float) -> tuple[Any, str]:
    """A flat rule and the ground it holds on [lo, hi], in the term's own words.

    An ordered term's basis (``_axis_levels``) names the rule's ends and the levels it
    holds rather than their values.
    """
    levels = spec._axis_levels
    if levels is None:
        return rule, f"[{_fmt_end(lo)}, {_fmt_end(hi)}]"
    label = dict(levels)
    shown = replace(rule, lo=label.get(rule.lo, rule.lo), hi=label.get(rule.hi, rule.hi))
    covered = [name for value, name in levels if lo <= value <= hi]
    return shown, f"the levels {covered[0]!r} to {covered[-1]!r}"


def _warn_widened_flats(spec: Any, pieces: list, *, stacklevel: int) -> None:
    """Warn once when flat stretches hold on more than they state, naming where they hold.

    They do when an end is not a knot (the whole knot spans it touches), when a
    natural end holds the boundary span flat beside them, and when two sit fewer
    than ``degree`` knot spans apart, which holds the gap between them flat
    (``held_flat_regions``). An end at the feature's boundary is exact: the
    curve is read on its range. An ordered term's basis (``_axis_levels``)
    names its ends and the levels the stretch holds on rather than their values.
    """
    levels = spec._axis_levels
    label = {} if levels is None else dict(levels)
    flat_rules = [rule for rule in rules_of(spec) if rule.kind == "flat"]
    flat_pieces = [piece for piece in pieces if piece[0] == "flat"]
    zero = _shape_rule_rows.flat_zero_regions(spec._knots, pieces)
    regions = _shape_rule_rows.held_flat_regions(
        spec._knots, spec.degree, pieces, natural_ends=has_natural_ends(spec)
    )
    widened = []
    for s, e in regions:
        inside = [z for z in zero if s <= z[0] and z[1] <= e]
        held_flats = [
            (rule, a, b)
            for rule, (_, a, b) in zip(flat_rules, flat_pieces, strict=True)
            if s <= a and b <= e
        ]
        extended = s < inside[0][0] or e > inside[-1][1]
        ends = dict.fromkeys(end for _, a, b in held_flats for end in (a, b))
        missing = [
            repr(label[end]) if end in label else _fmt_end(end)
            for end in ends
            if end not in (spec._lo, spec._hi) and not np.any(spec._knots == end)
        ]
        if not missing and not extended and len(inside) == 1:
            continue
        reason = "the whole knot spans it touches"
        fixes = []
        if missing:
            where = (
                "a knot at " + missing[0]
                if len(missing) == 1
                else "knots at " + " and ".join(missing)
            )
            fixes.append(f"put {where} for exact edges")
        if extended:
            reason += " and the boundary span its natural end holds flat beside them"
            fixes.append("put a knot inside the boundary span to keep it free")
        if len(inside) > 1:
            reason += f" and the gap between them, under {spec.degree} knot spans wide"
            fixes.append(f"leave at least {spec.degree} knot spans between flat stretches")
        lo, hi = max(s, spec._lo), min(e, spec._hi)
        shown = [_shown_and_held(spec, rule, lo, hi)[0].describe() for rule, _, _ in held_flats]
        held = _shown_and_held(spec, held_flats[0][0], lo, hi)[1]
        widened.append(f"{' & '.join(shown)} holds on {held}, {reason}; {'; '.join(fixes)}")
    if widened:
        warnings.warn("; ".join(widened), UserWarning, stacklevel=stacklevel + 1)


def _refuse_flat_over_curve(spec: Any, pieces: list) -> None:
    """Refuse flat stretches that hold the whole curve constant.

    Every first-derivative coefficient is then held at zero, which leaves one
    column, the constant, confounded with the intercept: the term has no effect
    to report, and its rounding would print a varying one. A flat rule needs a
    stretch for the same reason.
    """
    natural = has_natural_ends(spec)
    if not _shape_rule_rows.flat_covers_curve(
        spec._knots, spec.degree, pieces, natural_ends=natural
    ):
        return
    regions = _shape_rule_rows.held_flat_regions(
        spec._knots, spec.degree, pieces, natural_ends=natural
    )
    lo, hi = max(regions[0][0], spec._lo), min(regions[-1][1], spec._hi)
    flat_rules = [rule for rule in rules_of(spec) if rule.kind == "flat"]
    shown, held = zip(*(_shown_and_held(spec, rule, lo, hi) for rule in flat_rules), strict=True)
    zero = _shape_rule_rows.flat_zero_regions(spec._knots, pieces)
    natural = regions[0][0] < zero[0][0] or regions[-1][1] > zero[-1][1]
    gaps = len(zero) > 1
    raise ValueError(
        f"{' & '.join(rule.describe() for rule in shown)} holds the curve flat on {held[0]}, "
        "the whole knot spans it touches"
        + (" and the boundary spans its natural ends hold flat beside them" if natural else "")
        + (f" and the gaps between them, under {spec.degree} knot spans wide" if gaps else "")
        + ", which is the feature's whole range and leaves the term with no effect; remove the "
        "term, or narrow the stretch to leave part of the range free"
        + (", or put a knot inside a boundary span" if natural else "")
        + (f", or leave at least {spec.degree} knot spans between flat stretches" if gaps else "")
    )


def prepare_shape_rules(spec: Any, *, stacklevel: int = 1) -> None:
    """Resolve the spec's rule collection on its placed knots, once per build.

    Stores the pieces (``_shape_pieces``) the inequality rows are built from and
    the flat equality rows (``_flat_rows``) the main effect absorbs, so every
    later rebuild of the rows (the REML restore, the design compiler) reads the
    same pieces without warning again. Refusals and warnings are raised here,
    at the build. ``stacklevel`` counts frames above this function, 1 being its
    caller.
    """
    spec._shape_pieces = None
    spec._flat_rows = None
    if not _is_rule_collection(spec):
        return
    pieces = _collection_pieces(spec, stacklevel=stacklevel + 1)
    _refuse_flat_over_curve(spec, pieces)
    _warn_widened_flats(spec, pieces, stacklevel=stacklevel + 1)
    _, flat_rows = _shape_rule_rows.shape_rows(
        spec._knots,
        spec.degree,
        pieces,
        domain=(spec._lo, spec._hi),
        natural_ends=has_natural_ends(spec),
    )
    spec._shape_pieces = pieces
    spec._flat_rows = flat_rows


def build_shape_constraints_raw(spec: Any) -> LinearConstraintSet:
    """Build fit-time shape constraints on raw spline coefficients.

    A single full-range rule uses the full-range builders. A rule collection
    uses the pieces ``prepare_shape_rules`` resolved at the build; its flat
    rows are equalities the main effect absorbs, not rows here.
    """
    kind = spec.constraint_kind
    if kind in CONSTRAINT_KINDS:
        rows = _shape_rule_rows.full_range_rows(
            spec._knots,
            spec.degree,
            kind,
            domain=(spec._lo, spec._hi),
            natural_ends=has_natural_ends(spec),
        )
        return LinearConstraintSet(A=rows, b=np.zeros(rows.shape[0]))
    if not _is_rule_collection(spec):
        raise ValueError(f"Unsupported fit-time shape kind: {kind!r}")
    pieces = spec._shape_pieces
    if pieces is None:
        pieces = _collection_pieces(spec, stacklevel=1)
    rows, _ = _shape_rule_rows.shape_rows(
        spec._knots,
        spec.degree,
        pieces,
        domain=(spec._lo, spec._hi),
        natural_ends=has_natural_ends(spec),
    )
    cap = _cap_rows(spec)
    if cap is None:
        return rows
    return LinearConstraintSet(A=np.vstack([rows.A, cap.A]), b=np.concatenate([rows.b, cap.b]))


def _cap_rows(spec: Any) -> LinearConstraintSet | None:
    """The rows of the spec's cap on its printed curve, or ``None`` without a cap.

    The curve is monotone on its whole range when a direction rule covers it,
    and its cap is then stated at the two ends.
    """
    rules = rules_of(spec)
    cap = next((rule for rule in rules if rule.kind == "cap"), None)
    if cap is None:
        return None
    mean = spec._training_basis_mean
    if mean is None:
        raise RuntimeError(
            f"{cap.describe()} needs the training basis mean stored when the term was built; "
            "build the term on its training rows before building its constraint rows"
        )
    monotone = any(
        rule.kind in ("increasing", "decreasing") and rule.stretch == (None, None) for rule in rules
    )
    ends = (spec._lo, spec._hi) if monotone else None
    return _shape_rule_rows.spline_cap_rows(spec._knots, spec.degree, mean, cap.bounds, ends=ends)
