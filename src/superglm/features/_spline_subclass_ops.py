"""Subclass-specific spline helper operations."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm.features import _shape_rule_rows, _spline_constraints
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


def _collection_pieces(spec: Any, *, stacklevel: int) -> list:
    """The spec's rules as pieces on its range, refusing rules a spline cannot fit yet."""
    rules = rules_of(spec)
    for rule in rules:
        if rule.kind in ("cap", "max_step"):
            raise ValueError(
                f"{rule.describe()} is not supported on spline terms yet: caps and step "
                "limits cannot be fitted on this term type; drop the rule"
            )
    return _shape_rule_rows.shape_pieces(rules, spec._lo, spec._hi, stacklevel=stacklevel + 1)


def _fmt_end(x: float) -> str:
    return f"{x:.6g}"


def _warn_widened_flats(spec: Any, pieces: list, *, stacklevel: int) -> None:
    """Warn once when a flat stretch's ends are not knots, naming where it holds.

    An end at the feature's boundary is exact: the curve is read on its range.
    """
    flat_rules = [rule for rule in rules_of(spec) if rule.kind == "flat"]
    flat_pieces = [piece for piece in pieces if piece[0] == "flat"]
    widened = []
    for rule, piece in zip(flat_rules, flat_pieces, strict=True):
        (lo, hi) = _shape_rule_rows.flat_zero_regions(spec._knots, [piece])[0]
        lo, hi = max(lo, spec._lo), min(hi, spec._hi)
        _, a, b = piece
        missing = [
            _fmt_end(end)
            for end in (a, b)
            if end not in (spec._lo, spec._hi) and not np.any(spec._knots == end)
        ]
        if not missing:
            continue
        where = (
            "a knot at " + missing[0] if len(missing) == 1 else "knots at " + " and ".join(missing)
        )
        widened.append(
            f"{rule.describe()} holds on [{_fmt_end(lo)}, {_fmt_end(hi)}], the whole knot spans "
            f"it touches; put {where} for exact edges"
        )
    if widened:
        warnings.warn("; ".join(widened), UserWarning, stacklevel=stacklevel + 1)


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
    _warn_widened_flats(spec, pieces, stacklevel=stacklevel + 1)
    _, flat_rows = _shape_rule_rows.shape_rows(
        spec._knots, spec.degree, pieces, natural_ends=has_natural_ends(spec)
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
    if kind in {"increasing", "decreasing"}:
        return _spline_constraints.build_monotone_difference_constraints(spec._n_basis, kind)
    if kind in {"convex", "concave"}:
        constraints = _spline_constraints.build_curvature_difference_constraints(
            spec._knots,
            spec.degree,
            kind,
            domain=(spec._lo, spec._hi),
        )
        if has_natural_ends(spec):
            # Natural boundary conditions already impose the first and last
            # exact curvature rows as equalities.  Keeping their O(eps)
            # projections as QP inequalities creates scale-dependent fake
            # active constraints.
            interior = constraints.A[1:-1]
            return LinearConstraintSet(A=interior, b=np.zeros(interior.shape[0]))
        return constraints
    if not _is_rule_collection(spec):
        raise ValueError(f"Unsupported fit-time shape kind: {kind!r}")
    pieces = spec._shape_pieces
    if pieces is None:
        pieces = _collection_pieces(spec, stacklevel=1)
    rows, _ = _shape_rule_rows.shape_rows(
        spec._knots, spec.degree, pieces, natural_ends=has_natural_ends(spec)
    )
    return rows
