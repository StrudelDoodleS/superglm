"""OrderedCategorical feature: ordered categories with an inner shape basis.

Continuous variables often arrive pre-binned into ordered bands (age bands
"18-25", "26-35", ...). This feature maps the band labels to numeric values
and fits a shape over them, so adjacent levels borrow strength. ``basis=``
takes the shape: ``Spline(...)`` (the penalized smooth, the default),
``Piecewise(...)`` (stated kinks, breaks named by band) or ``Polynomial(...)``
(weighted orthogonal ordinal contrasts on the level positions). A pickled
step-mode spec fails at first use (see ``_basis_spline``).
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.typing import NDArray

from superglm.features.categorical import (
    _grouping_labels,
    _validate_categorical_levels,
    _validate_missing_only,
)
from superglm.features.constraint import (
    _CURVATURE_KINDS,
    _DIRECTION_KINDS,
    ConstraintRules,
    rules_of,
)
from superglm.features.piecewise import Piecewise
from superglm.features.polynomial import Polynomial
from superglm.types import GroupInfo, LinearConstraintSet

if TYPE_CHECKING:  # Spline imports this module at runtime; keep the cycle type-only.
    from superglm.features.spline import _SplineBase


# One migration sentence for every path that refuses a step-mode spec, so the
# editor's refusal and ``_basis_spline``'s name the same replacement.
_STEP_MODE_REMOVED_MESSAGE = (
    "Step mode was removed in 0.24.0, so a spec built or pickled "
    "before the removal cannot be used; rebuild it with "
    "basis=Spline(...) for a smoothed ordinal term, or "
    "Categorical(...) for independent level effects."
)

# Stable leading clause of the knot-clamp warning; editor/collapse.py filters
# its internal clone's warning by this prefix.
_CLAMP_WARNING_PREFIX = "OrderedCategorical: n_knots exceeds the level count"

# The rules an ordered term holds at its levels rather than on its basis spline.
_LEVEL_RULE_KINDS = frozenset({"cap", "max_step"})


def _spline_kind_name(spline: Any) -> str:
    """Return the public factory kind for a spline specification."""
    from superglm.features.spline import (
        BSplineSmooth,
        CardinalCRSpline,
        CubicRegressionSpline,
        NaturalSpline,
        PSpline,
    )

    kind_by_type = (
        (PSpline, "ps"),
        (BSplineSmooth, "bs"),
        (NaturalSpline, "ns"),
        (CubicRegressionSpline, "cr"),
        (CardinalCRSpline, "cr_cardinal"),
    )
    for spline_type, kind in kind_by_type:
        if isinstance(spline, spline_type):
            return kind
    return type(spline).__name__


def _declared_matcher(declared_levels: list[Any], *, numeric_strings: bool = False):
    """Return a function mapping a raw column value onto the level it denotes.

    The declaration is canonical: data spelled differently (``9.0`` against a
    declared ``9``) is mapped onto it once, here. Numeric matching applies only
    to non-string raw values, so ``"001"`` is never claimed by a declared ``1``;
    leading zeros are identity in policy and vehicle codes.
    """
    by_text: dict[str, Any] = {}
    by_value: dict[float, Any] = {}
    ambiguous: set[float] = set()
    for level in declared_levels:
        by_text.setdefault(str(level), level)
        try:
            value = float(level)
        except (TypeError, ValueError):
            continue
        prior = by_value.get(value)
        if prior is not None and str(prior) != str(level):
            ambiguous.add(value)
        by_value.setdefault(value, level)

    def match(raw: Any) -> Any:
        text = str(raw)
        if text in by_text:
            return by_text[text]
        if isinstance(raw, str) and not numeric_strings:
            # A raw string is its own identity. Only the grouping path opts in,
            # because a LevelGrouping's labels are stringified data.
            return raw
        # Only a value equal to the same number counts ("9" and 9.0 match).
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return raw
        if value not in by_value:
            return raw
        # Ambiguity matters only for a raw value that matches no spelling exactly
        # yet equals two numerically (`order=["1", "1.0"]`); refuse that value,
        # not the declaration.
        if value in ambiguous:
            raise ValueError(
                f"Level {raw!r} is ambiguous: the declared levels spell the value {value} "
                "more than one way and this value matches neither exactly. Declare the "
                "levels consistently, or supply the level under one of its exact spellings."
            )
        matched = by_value[value]
        # float() is lossy past 2**53; when both sides are real numbers, confirm
        # the match exactly (Python compares int to float exactly).
        if not isinstance(raw, str) and not isinstance(matched, str) and raw != matched:
            return raw
        return matched

    return match


def group_axis_position(member_values: Sequence[float]) -> float:
    """Where a merged group sits on the numeric level axis: the MEAN of its members.

    The one definition, shared by the spec (placing the group on the basis
    axis) and the display collapse in ``plotting/group_display.py`` (placing
    its marker). The mean is label-invariant, so renaming a group never moves
    the fit (issue #326). To place a group elsewhere, declare ``values=`` with
    equal values for its members. Not used for a ``Piecewise``/``Polynomial``
    inner basis, whose axis is level positions and where a group is one band.
    """
    return float(np.mean(np.asarray(member_values, dtype=np.float64)))


def _regroup_to_declared(grouping: Any, declared_levels: list[Any]):
    """Return ``grouping`` with its ORIGINAL levels re-spelled as declared.

    Group LABELS are new names the caller chose and are left alone; only the
    original levels are references to levels this spec already knows.
    """
    if grouping is None:
        return None
    from dataclasses import replace as _replace

    match = _declared_matcher(declared_levels, numeric_strings=True)

    def respell(level: Any) -> str:
        return str(match(level))

    def respell_label(label: Any, members: list) -> str:
        """A group LABEL is a new name the caller chose -- unless the group is the
        identity, in which case the label IS the level and shares its spelling."""
        if [str(m) for m in members] == [str(label)]:
            return respell(label)
        return str(label)

    labels = {
        label: respell_label(label, members)
        for label, members in grouping.group_to_originals.items()
    }
    # Re-spelling can make two group labels collide -- an identity level "1.0"
    # becomes "1" while the caller already named a group "1". Left alone, the
    # dict comprehensions below drop one entry and every original silently maps
    # to the survivor, folding a level into a group it was never in.
    collisions = [v for v in set(labels.values()) if list(labels.values()).count(v) > 1]
    if collisions:
        raise ValueError(
            f"OrderedCategorical grouping label(s) {sorted(collisions)!r} collide once level "
            "names are matched against the declaration. Rename the group so it does not "
            "clash with a level label."
        )
    return _replace(
        grouping,
        original_to_group={
            respell(k): labels.get(v, v) for k, v in grouping.original_to_group.items()
        },
        group_to_originals={
            labels.get(label, label): [respell(m) for m in members]
            for label, members in grouping.group_to_originals.items()
        },
        all_original_levels=[respell(level) for level in grouping.all_original_levels],
        grouped_levels=[labels.get(label, label) for label in grouping.grouped_levels],
    )


def _require_two_smooth_levels(smooth_levels: list[str], special_set: set[str]) -> None:
    """The final smooth-level list must retain at least two levels to smooth."""
    if special_set and len(smooth_levels) < 2:
        raise ValueError(
            "OrderedCategorical needs at least two non-special levels to fit a "
            f"smooth; got {smooth_levels!r} after removing {sorted(special_set)!r}. "
            "Use Categorical(...) for independent level effects."
        )


def _require_no_grouped_specials(grouping: Any, special_set: set[str]) -> None:
    """Refuse a grouping that merges a special into any other level.

    Merging a special into an ordered group would smooth it while ``_specials``
    still reports it free. Merging two specials, or renaming one under a new
    label, is refused because ``specials=`` would no longer name it. Only a
    group whose label equals its one member is exempt.
    """
    if not special_set or grouping is None:
        return
    # A grouping must also cover every special: one that omits it maps those
    # rows to NaN, which would otherwise fail deep in scipy without naming the
    # grouping.
    covered = {
        str(member) for originals in grouping.group_to_originals.values() for member in originals
    }
    uncovered = sorted(special_set - covered)
    if uncovered:
        raise ValueError(
            f"OrderedCategorical grouping does not cover free level(s) {uncovered!r}. "
            "Every special must appear in the grouping as its own single-member "
            "group; rows of an omitted level map to no group and reach the spline "
            "as NaN."
        )
    for label, originals in grouping.group_to_originals.items():
        members = [str(member) for member in originals]
        if members == [str(label)]:
            continue
        merged = [member for member in members if member in special_set]
        if merged:
            joined = ", ".join(repr(member) for member in merged)
            raise ValueError(
                f"OrderedCategorical grouping merges free level(s) {joined} into group "
                f"{label!r}. Specials are fitted outside the smooth and may not be "
                "grouped; group the ordered levels only."
            )


class GroupNamedAsSpecialError(ValueError):
    """A group of other levels is named as a free level is spelled."""


def _require_no_group_named_as_special(grouping: Any, spellings: dict[str, Any]) -> None:
    """Refuse a group of other levels whose label is a spelling of a special.

    ``spellings`` maps each text a special's indicator matches, the one
    ``specials=`` declares (``"9"``) and the one the term reports (``"9.0"``),
    to the special as reported. A group named either way would have its rows
    claimed by that indicator and fitted as the special.
    """
    if grouping is None:
        return
    for label, originals in grouping.group_to_originals.items():
        text = str(label)
        if text in spellings and [str(member) for member in originals] != [str(spellings[text])]:
            raise GroupNamedAsSpecialError(
                f"OrderedCategorical grouping names a group of other levels {text!r}, a "
                f"spelling of the free level {spellings[text]!r}, so that level's indicator would "
                "claim the group's rows; give the group another name."
            )


class OrderedCategorical:
    """Ordered categorical feature smoothed by a spline over its level values.

    Designed for continuous variables that arrive pre-binned into ordered
    categories (e.g. age bands, mileage bands).  Maps category labels to
    numeric values and fits a smooth function through them, borrowing
    strength between adjacent levels.  With ``fit_reml()``, REML selects the
    smoothing parameter automatically — the effective degrees of freedom will
    typically be much less than the number of levels.

    The smooth is configured by passing a :func:`Spline` specification as
    ``basis``::

        OrderedCategorical(
            order=["low", "medium", "high"],
            basis=Spline(kind="ps", k=6),
        )

    Parameters
    ----------
    values : dict[str, float] or None
        Explicit mapping from category labels to numeric values (e.g.
        midpoints: ``{"18-25": 21.5, "26-35": 30.5, ...}``).
        Mutually exclusive with ``order``.
    order : list[str] or None
        Ordered list of category labels.  Numeric values are generated as
        ``linspace(0, 1, len(order))``.  Mutually exclusive with ``values``.
    basis : Spline, Piecewise, Polynomial object, or None
        The shape fitted over the ordered levels.  Deep-copied at
        construction, so mutating the passed object afterwards changes
        nothing here.

        ``Spline(...)`` gives the penalized smooth::

            OrderedCategorical(order=[...], basis=Spline(kind="cr", k=6))

        Omitting ``basis`` gives the default cubic regression spline,
        ``Spline(kind="cr", n_knots=5, penalty="ssp", select=False)``.  In
        either form ``n_knots`` is clamped to ``n_levels - 1`` with a
        warning.  ``Spline(knots=[...])`` may state knots as band names
        (``knots=["Mi060", "Mi066"]``), each resolving to that level's value
        on the smooth's axis; numeric entries stay axis values.

        ``Piecewise(breaks=[...])`` gives stated kinks with no smoothing
        penalty, an unpenalized main block like the ``specials=`` block.  It
        evaluates on level positions ``0..L-1`` (``values=`` sets the order
        but not the spacing).  Breaks are band names, with integer positions
        as the escape hatch; ``degrees=[...]`` states one polynomial degree per
        segment (``0`` = flat tail), value-continuous at every seam.  Rating
        tables stay one row per band at any degree.  ``extrapolation`` is
        inert here, since every level lies inside ``[0, L-1]``.

        ``Polynomial(powers=[...])`` gives orthogonal ordinal contrasts (the
        ``contr.poly`` device) on the level positions, orthonormalized against
        the training exposure (SAS ``ORPOL``'s weighted construction).
        ``powers=`` may be a non-contiguous subset, and each stated power
        reports its own z row in the summary.
    base : str
        Reporting reference level. ``"most_exposed"`` (default), ``"first"``,
        or a specific level name. This changes only the reported relativities
        and reference-adjusted intercept, not the fitted smooth.
    specials : list[str] or None
        Level labels held out of the smooth and fitted as free, unpenalized
        level effects — one indicator column and one coefficient each. Use for
        levels that are structurally different rather than merely sparse (a
        ``MISSING`` band, a structural zero); the penalty already handles
        sparse bands better than free levels do. A label listed here is
        removed from ``order``/``values`` if also present there, and never
        receives a numeric position on the smooth's axis.

    Examples
    --------
    Using ordered level names (auto-spaced 0 to 1) with an explicit smooth::

        OrderedCategorical(
            order=["18-25", "26-35", "36-45", "46-55", "56+"],
            basis=Spline(kind="ps", k=6),
        )

    Using explicit midpoints::

        OrderedCategorical(
            values={"18-25": 21.5, "26-35": 30.5, "36-45": 40.5},
            basis=Spline(kind="cr", k=4),
        )

    Independent, unsmoothed level effects should use ``Categorical``::

        Categorical(base="most_exposed")
    """

    # Whether `base=` names a band although it reads "first" or "most_exposed";
    # set only by a rebuild. A class attribute so older pickles read False.
    _base_is_level: bool = False
    # The cap and step limit the last build holds at the levels; a spec pickled
    # before level rules existed reads none.
    _level_rules: ConstraintRules = ConstraintRules(())

    def __setstate__(self, state: dict) -> None:
        """Restore a pickle; a spec saved before ``specials=`` existed has none.

        Its ``_ordered_levels`` were its smooth levels and no level was held
        out: the state ``specials=None`` builds.
        """
        if "_specials" not in state:
            state["_specials"] = []
            state["_special_raw"] = []
            state["_special_display"] = []
            state["_smooth_levels"] = list(state.get("_ordered_levels", ()))
        self.__dict__.update(state)

    def __init__(
        self,
        values: dict[str, float] | None = None,
        order: list[str] | None = None,
        basis: _SplineBase | None = None,
        base: Literal["most_exposed", "first"] | str = "most_exposed",
        grouping: Any = None,
        specials: list[str] | None = None,
    ):
        from superglm.features.spline import Spline, _SplineBase

        if values is not None and order is not None:
            raise ValueError("Specify exactly one of 'values' or 'order', not both.")
        if values is None and order is None:
            raise ValueError("Must specify either 'values' or 'order'.")

        self._specials: list[str] = []
        # The declarations as given, aligned with `_specials`: `_special_mask`
        # needs them to match a non-str label against a numeric column.
        self._special_raw: list[Any] = []
        if specials is not None:
            # Level labels live in `str` space throughout this file, so a
            # non-str special must be coerced to pop from order=/values=.
            for raw_lev in specials:
                lev = str(raw_lev)
                if lev in self._specials:
                    raise ValueError(f"Duplicate special level {lev!r} in 'specials'.")
                # "9" and "9.0" differ as strings but `_special_mask` also
                # matches raw values, and 9 == 9.0: both indicators would claim
                # the same rows, two collinear unpenalized coefficients.
                clash = next((prev for prev in self._special_raw if prev == raw_lev), None)
                if clash is not None:
                    raise ValueError(
                        f"Special level {raw_lev!r} in 'specials' duplicates {clash!r}: they "
                        "are spelled differently but compare equal, so both would claim the "
                        "same rows and their free coefficients would be unidentifiable."
                    )
                self._specials.append(lev)
                self._special_raw.append(raw_lev)
        # How each special is reported: the label its domain spells it with, in
        # the same namespace as the smooth levels. Refined below.
        self._special_display: list[Any] = list(self._special_raw)
        special_set = set(self._specials)
        # A special declared as `9` must be admissible in both the raw and the
        # string form its mask matches on, or it is rejected as unseen.
        known_special_labels: set[Any] = set(special_set)
        if specials is not None:
            known_special_labels |= set(specials)

        if special_set:
            # Match on both views, as `_special_mask` does, or the level ends up
            # both smoothed and free.
            raw_specials = list(self._special_raw)

            def _is_special(label: Any) -> bool:
                if str(label) in special_set:
                    return True
                return any(
                    not isinstance(raw, str) and not isinstance(label, str) and label == raw
                    for raw in raw_specials
                )

            # Report a special under its domain's label, not the coerced string:
            # rating tables, editor weights and exposure bars look rows up by the
            # column's own labels.
            domain = list(values) if values is not None else list(order)
            for matched in domain:
                if not _is_special(matched):
                    continue
                for j, coerced in enumerate(self._specials):
                    if coerced == str(matched) or (
                        not isinstance(self._special_raw[j], str)
                        and not isinstance(matched, str)
                        and matched == self._special_raw[j]
                    ):
                        self._special_display[j] = matched
                        break

            if values is not None:
                values = {k: v for k, v in values.items() if not _is_special(k)}
            else:
                order = [lev for lev in order if not _is_special(lev)]

        # Whether the caller named a basis; the clamp warning spells out the
        # remedy when there is no `Spline` in the caller's source to point at.
        basis_was_explicit = basis is not None
        if basis is None:
            # The default smooth: a cubic regression spline, as Spline's default is.
            basis = Spline(kind="cr", n_knots=5, penalty="ssp", select=False)
        elif isinstance(basis, str) and basis in ("spline", "step"):
            raise ValueError(
                f"OrderedCategorical no longer accepts basis={basis!r}: the legacy "
                "string modes were removed in 0.24.0. Pass the smooth itself with "
                "basis=Spline(...) -- or omit basis for the default cubic regression spline. For "
                "independent, unsmoothed level effects use Categorical(...)."
            )
        elif not isinstance(basis, _SplineBase | Piecewise | Polynomial):
            raise ValueError(
                f"basis must be a Spline, Piecewise or Polynomial object, or None; got {basis!r}"
            )
        # Deep-copy: `_spline_obj` is the pristine declaration editor/collapse.py
        # rebuilds from when a grouping is undone, so a caller mutating their
        # object afterwards must not change it.
        self._spline_obj: _SplineBase | Piecewise | Polynomial = copy.deepcopy(basis)
        self._basis_was_explicit = basis_was_explicit
        # Vestigial: dm_builder, screening, report, export and editor still
        # branch on this string. An instance attribute so a step-mode pickle
        # keeps "step" and `_basis_spline` names it.
        self.basis = "spline"

        self.base = base
        self._smooth_levels: list[str] = []
        # Levels as reported, in block order: smooth levels under their
        # `order=`/`values=` labels, then specials under `_special_display`.
        # Downstream row names, weights and membership tests join on equality.
        self._ordered_levels: list[Any] = []

        # Derive smooth levels and numeric values
        if values is not None:
            sorted_items = sorted(values.items(), key=lambda kv: kv[1])
            self._smooth_levels = [k for k, _ in sorted_items]
            self._level_to_value = dict(values)
        else:
            self._smooth_levels = list(order)
            n = len(order)
            vals = np.linspace(0.0, 1.0, n) if n > 1 else np.array([0.0])
            self._level_to_value = dict(zip(order, vals.tolist()))
        self._ordered_levels = list(self._smooth_levels)
        # The declared smooth levels before grouping: the namespace named
        # breaks/knots resolve against, stable across a collapse.
        self._declared_smooth_levels: list[Any] = list(self._smooth_levels)

        # Level labels must be distinct as strings: grouping and every report
        # join on str(), so `order=[1, "1", 2]` cannot be represented.
        _texts = [str(level) for level in self._ordered_levels]
        _dupes = sorted({t for t in _texts if _texts.count(t) > 1})
        if _dupes:
            raise ValueError(
                f"OrderedCategorical level label(s) {_dupes!r} appear more than once once "
                "rendered as text. Levels are matched and reported by their string form, so "
                "two levels cannot share one; declare them with distinct labels."
            )

        # A grouping is built from data, so re-spell it once into the
        # declaration's namespace. A special is spelled as the term reports it
        # (9.0 beside 1.0 for specials=[9]), since rows canonicalise to that.
        grouping = _regroup_to_declared(grouping, self._ordered_levels + self._special_display)
        grouped_specials = {str(level) for level in self._special_display}
        # Penalized collapse (L1 fusion of adjacent levels; Gertheiss & Tutz,
        # Ann. Appl. Stat. 4(4), 2010) would attach here as its own mode beside
        # `basis=`; the stated-structure bases must not grow a penalty path.
        self._grouping = grouping
        # Before the numeric-position check: merging or renaming a special has
        # a more specific error than 'no numeric position'.
        _require_no_grouped_specials(grouping, grouped_specials)
        _require_no_group_named_as_special(
            grouping,
            {
                text: shown
                for coerced, shown in zip(self._specials, self._special_display)
                for text in (coerced, str(shown))
            },
        )
        self._original_level_to_value: dict[str, float] | None = None
        if grouping is not None:
            # Original level->value map for plot expansion, keyed by the
            # grouping's string level names as `_expand_grouped_term` reads it.
            _match = _declared_matcher(
                self._ordered_levels + list(self._specials), numeric_strings=True
            )
            orig_ltv = {str(_match(k)): v for k, v in self._level_to_value.items()}
            self._original_level_to_value = orig_ltv
            # One position per group, `group_axis_position` over its members.
            # A LevelGrouping is string-keyed while `values=`/`order=` keys are
            # the declared objects, so join on text.
            by_text = {str(k): v for k, v in orig_ltv.items()}
            grouped_ltv = {}
            for glev in grouping.grouped_levels:
                # ``.get``: a hand-built LevelGrouping may miss a label, which
                # should reach the named error below, not a bare KeyError.
                originals = grouping.group_to_originals.get(glev, [glev])
                vals = [by_text[str(o)] for o in originals if str(o) in by_text]
                if vals:
                    grouped_ltv[glev] = group_axis_position(vals)
                # A group sits at its members' mean only if every member has a
                # declared value; averaging a subset would fit at a coordinate
                # the partition never asks for, so refuse here.
                undeclared = [str(o) for o in originals if str(o) not in by_text]
                if vals and undeclared and str(glev) not in grouped_specials:
                    raise ValueError(
                        f"OrderedCategorical grouping puts undeclared level(s) "
                        f"{undeclared!r} in group {glev!r}, whose other members are "
                        f"declared. A group's numeric position is the mean of its "
                        f"members' positions, so a partly-declared group has no "
                        f"well-defined position; declared levels are "
                        f"{sorted(by_text)}. Declare those levels, or leave them out "
                        "of the grouping."
                    )
            # Specials are exempt: a free level has no coordinate on the axis.
            missing = [
                g
                for g in grouping.grouped_levels
                if g not in grouped_ltv and str(g) not in grouped_specials
            ]
            if missing:
                raise ValueError(
                    f"OrderedCategorical grouping produced level(s) {missing!r} with no "
                    f"numeric position; known levels are {sorted(by_text)}. The grouping "
                    "and the declaration disagree about how levels are named."
                )
            self._level_to_value = grouped_ltv
            self._smooth_levels = [
                lev for lev in grouping.grouped_levels if lev not in grouped_specials
            ]
            # _known_levels includes all *original* levels (for predict-time validation)
            self._known_levels = set(grouping.all_original_levels) | known_special_labels
        else:
            self._known_levels = set(self._smooth_levels) | known_special_labels
        self._ordered_levels = list(self._smooth_levels) + list(self._special_display)
        self._n_levels = len(self._smooth_levels)

        # Piecewise/Polynomial bases evaluate on level positions 0..L-1, since
        # band vocabulary is positional; values= still fixes the order.
        if not isinstance(self._spline_obj, _SplineBase):
            self._install_position_axis()

        _require_two_smooth_levels(self._smooth_levels, special_set)
        if str(base) in special_set:
            raise ValueError(
                f"OrderedCategorical reporting base {base!r} is a special level. The base "
                "anchors every reported relativity and must lie on the smooth; choose one "
                f"of {self._smooth_levels!r}."
            )

        # Reporting state, populated by _choose_base at build time.
        self._base_is_level: bool = False
        self._base_level: str = ""
        self._non_base: list[str] = []

        # A declared special with no effective training rows is pinned: no
        # indicator column, its identity kept, its rows scoring zero.
        # `_active_specials` indexes `_specials` in emitted column order;
        # `_pinned_specials` holds the display labels of the rest.
        self._pinned_specials: list[Any] = []
        self._active_specials: list[int] = list(range(len(self._specials)))

        # Stamped per build by the design-matrix builder: True under the prior
        # contract, where hosted Piecewise geometry follows physical rows and
        # the build derives the positive-row indicator from its own weights.
        self._inner_geometry_physical_rows: bool = False

        # The inner spline this wrapper owns and delegates to (deferred until
        # n_levels is known, because the knot count clamps against it).
        self._spline: _SplineBase | None = None
        self._init_spline()

    @property
    def _basis_spline(self) -> _SplineBase:
        """The inner spline, narrowed to non-None.

        ``None`` means the instance bypassed ``__init__`` (a pickle of the
        removed step mode). Every numeric path funnels through this property,
        so such a spec fails at first use. ``getattr(spec, name, default)``
        swallows the ``AttributeError``, so do not wrap readers in a
        defaulting ``getattr``: completeness of the funnel is the guarantee.
        """
        spline = self._spline
        if spline is None:
            raise AttributeError(
                f"OrderedCategorical(basis={self.basis!r}) has no inner spline. "
                f"{_STEP_MODE_REMOVED_MESSAGE}"
            )
        return spline

    # ── Derived spline metadata ────────────────────────────────────
    # Read-only views of the inner spline's configuration, kept as public API
    # for external readers; no superglm code path reads them.

    @property
    def kind(self) -> str:
        """Public factory kind of the inner spline (``"ps"``, ``"cr"``, ...)."""
        return _spline_kind_name(self._basis_spline)

    @property
    def n_knots(self) -> int:
        """Knot count of the inner spline, after the ``n_levels - 1`` clamp."""
        return self._basis_spline.n_knots

    @property
    def degree(self) -> int:
        """B-spline degree of the inner spline."""
        return self._basis_spline.degree

    @property
    def select(self) -> bool:
        """Whether the inner spline carries double-penalty selection."""
        return self._basis_spline.select

    @property
    def penalty(self) -> str | None:
        """Penalty type of the inner spline."""
        return self._basis_spline.penalty

    # ── Declared shape constraint ──────────────────────────────────
    # Forwarded because shape engines call ``rules_of(spec)`` on the feature
    # spec, which for an ordered term is this wrapper. ``getattr`` with a
    # default, because a Piecewise or Polynomial inner basis takes no constraint.
    # They forward the basis as declared (``_declared_basis``), not the built
    # inner spline: a build places the declaration (``_place_basis_rules``),
    # moving stretch ends to level values on the inner spline and leaving the
    # cap and step limit on this wrapper, where they bind at the levels.
    # The legacy ``monotone`` spelling is not forwarded: ``rules_of`` reads
    # the rule objects first and a saved model's kind and mode second; a new
    # reader calls ``rules_of``. One reader spells the kind itself:
    # ``_spline_subclass_ops.build_shape_constraints_raw`` reads a bare
    # ``spec.constraint_kind``, safe because ``spec`` there is always a spline.

    @property
    def constraint_kind(self) -> str | None:
        """Shape token declared on the inner basis (``"increasing"``, ...).

        The constraint binds on the whole level axis, not only at the ``L``
        level values: conservative, since published monotone-ordinal methods
        constrain the level effects (Rufibach, CSDA 54(6), 2010, Sec. 5) and
        sign conditions are sufficient but not necessary (Hofner, Kneib &
        Hothorn, Stat. Comput. 26, 2016, Sec. 3.3). A cap and a step limit
        bind at the levels, which are what gets priced (``_level_rows``). See
        ``docs/how-to/constrain-a-smooth.md``.
        """
        declared = self._declared_basis
        return getattr(declared, "constraint_kind", getattr(declared, "monotone", None))

    @property
    def constraint_mode(self) -> str:
        """Whether the declared constraint binds at fit time or post-fit."""
        declared = self._declared_basis
        return getattr(declared, "constraint_mode", getattr(declared, "monotone_mode", "postfit"))

    @property
    def constraint_rules(self) -> ConstraintRules:
        """The rule objects declared on the basis; empty for a basis that takes none.

        ``rules_of`` reads these, so an ordered term answers with real rule
        kinds rather than the describe-string ``constraint_kind`` forwards.
        """
        return getattr(self._declared_basis, "constraint_rules", ConstraintRules(()))

    @property
    def _declared_basis(self) -> Any:
        """The basis as the caller declared it, or the inner spline for a spec without one."""
        inner = self._basis_spline
        declared = getattr(self, "_spline_obj", None)
        return inner if declared is None else declared

    def _refuse_level_rules(self) -> None:
        """Refuse a cap or step limit beside special levels, which no level row can hold.

        A special level is fitted as a free effect off the curve, so rows at the
        level values do not bound its printed relativity.
        """
        held = next((rule for rule in rules_of(self) if rule.kind in _LEVEL_RULE_KINDS), None)
        if held is None or not self.has_specials:
            return
        raise ValueError(
            f"{held.describe()} binds the relativities of the ordered levels, but the special "
            f"level(s) {list(self._special_display)!r} are free effects off the curve that it "
            "cannot hold; list those levels in order= or values= to put them on the curve, "
            "or drop the rule"
        )

    def _place_basis_rules(self) -> None:
        """Split the declared rules between the inner spline and this term, once per build.

        The shape rules go onto the inner spline, each stretch end moved to the
        value of the level it names (``_stretch_end_value``); a stretch is
        inclusive at the levels that end it. The cap and step limit stay here
        (``_level_rules``) and become rows at the levels. A declaration with
        neither stretches nor level rules leaves the inner spline as built, so
        a single full-range rule builds the rows it always has.
        """
        from superglm.features._spline_config import _normalize_constraint, fit_time_select_message

        self._refuse_level_rules()
        level, shape = rules_of(self)._split(_LEVEL_RULE_KINDS)
        self._level_rules = level
        if not level and all(rule.stretch == (None, None) for rule in shape):
            return
        inner = cast(Any, self._basis_spline)  # ty: the constraint fields are set per instance
        if level and inner.select:
            raise NotImplementedError(fit_time_select_message(rules_of(self)))
        self._refuse_stretches_in_levels(shape)
        rules, kind, mode = _normalize_constraint(shape._resolve(self._stretch_end_value))
        inner.constraint_rules = rules
        inner.constraint_kind = inner.monotone = kind
        inner.constraint_mode = inner.monotone_mode = mode
        inner._axis_levels = tuple(
            (float(self._level_to_value[level]), level) for level in self._smooth_levels
        )

    def _refuse_stretches_in_levels(self, shape: ConstraintRules) -> None:
        """Refuse empty, reversed or overlapping stretches in level labels, before resolving.

        Resolving would refuse each of them too, but in axis values. A stretch
        is empty when its ends name one level, or when it lies below the first
        level or above the last. Overlap is checked within the direction,
        curvature and flat groups, as ``ConstraintRules.validate`` checks it;
        stretches that only touch at a level do not overlap.
        """
        levels = sorted(self._smooth_levels, key=self._level_to_value.__getitem__)
        name = {float(self._level_to_value[level]): level for level in levels}
        first, last = (
            float(self._level_to_value[levels[0]]),
            float(self._level_to_value[levels[-1]]),
        )
        spans = []
        for rule in shape:
            lo = -np.inf if rule.lo is None else self._stretch_end_value(rule.lo)
            hi = np.inf if rule.hi is None else self._stretch_end_value(rule.hi)
            if lo == hi:
                raise ValueError(
                    f"{rule.describe()} starts and ends at level {name[lo]!r}, which leaves no "
                    "stretch; end it at a later level"
                )
            if rule.lo is not None and rule.hi is not None and not lo < hi:
                raise ValueError(
                    f"{rule.describe()} needs its first level before its second in level "
                    "order; swap the ends or use a single below/above"
                )
            if hi <= first:
                raise ValueError(
                    f"{rule.describe()} ends at the first level, {levels[0]!r}, so the stretch "
                    "holds no step between levels; end it at a later level, or drop .below to "
                    "apply the rule on every level"
                )
            if lo >= last:
                raise ValueError(
                    f"{rule.describe()} starts at the last level, {levels[-1]!r}, so the "
                    "stretch holds no step between levels; start it at an earlier level, or "
                    "drop .above to apply the rule on every level"
                )
            spans.append((rule, lo, hi))
        for group in _DIRECTION_KINDS, _CURVATURE_KINDS, ("flat",):
            members = [span for span in spans if span[0].kind in group]
            for i, (a, a_lo, a_hi) in enumerate(members):
                for b, b_lo, b_hi in members[i + 1 :]:
                    start, stop = max(a_lo, b_lo), min(a_hi, b_hi)
                    if start < stop:
                        first = name.get(start, levels[0])
                        last = name.get(stop, levels[-1])
                        raise ValueError(
                            f"{a.describe()} and {b.describe()} both apply from level "
                            f"{first!r} to level {last!r}; give them stretches that do not "
                            "overlap"
                        )

    def _stretch_end_value(self, end: float | str) -> float:
        """The axis value of the level a stretch end names.

        Ends are level labels, numeric ones included, matched as data values
        are (``_declared_matcher``), then through the grouping as a Spline knot
        is. A group label names its group. A level the grouping merges with
        others is refused, naming the group label to use, and so is a special
        level, which has no place on the curve.
        """
        texts = [str(level) for level in self._declared_smooth_levels]
        specials = [str(level) for level in self._special_display]
        if str(_declared_matcher(list(self._special_display))(end)) in specials:
            raise ValueError(
                f"stretch end {end!r} is a special level, a free effect off the curve; end "
                f"the stretch at one of the levels {texts!r}"
            )
        matched = str(_declared_matcher(list(self._declared_smooth_levels))(end))
        grouping = self._grouping
        if matched in texts and grouping is not None:
            group = str(grouping.original_to_group.get(matched, matched))
            members = [str(m) for m in grouping.group_to_originals.get(group, [matched])]
            if len(members) > 1:
                raise ValueError(
                    f"stretch end {end!r} names level {matched!r}, which the grouping merges "
                    f"into group {group!r} with {members!r}; end the stretch at the group "
                    f"label {group!r}"
                )
            straddle = self._straddling_group(texts.index(matched))
            if straddle is not None:
                label, held = straddle
                raise ValueError(
                    f"stretch end {end!r} names level {matched!r}, and group {label!r} holds "
                    f"levels {held!r} on both sides of it, so no stretch can end there; end "
                    "the stretch at a level outside that group, or group its levels on one side"
                )
        if matched in texts:
            position = self._grouped_break_position(texts.index(matched), parameter="stretch end")
            return float(self._level_to_value[self._smooth_levels[position]])
        # A grouped term's curve levels are its group labels, and each names a level too.
        curve = [str(level) for level in self._smooth_levels]
        group = str(_declared_matcher(list(self._smooth_levels))(end))
        if group in curve:
            return float(self._level_to_value[self._smooth_levels[curve.index(group)]])
        known = texts if grouping is None else curve
        raise ValueError(
            f"stretch end {end!r} is not one of the levels {known!r}; end the stretch at "
            "one of them"
        )

    def _level_rows(self) -> LinearConstraintSet:
        """Rows ``A @ c >= b`` holding the cap and step limit at the levels.

        ``B(l) @ c`` is the curve the term predicts at level ``l`` (the inner raw
        basis under its extrapolation policy, since a level absent from training
        can sit outside the inner range). A cap is ``log lo <= (B(l) - B(base)) @ c
        <= log hi`` at every other level, and a step limit bounds ``(B(l_{j+1}) -
        B(l_j)) @ c`` by ``log(1 - down)`` and ``log(1 + up)``. Rows are scaled by
        their largest entry, and zero rows are dropped. Under
        ``extrapolation="error"`` an out-of-range level is read clipped: it cannot
        be predicted.
        """
        inner = cast(Any, self._basis_spline)  # ty: the knots are set per instance at the build
        levels = sorted(self._smooth_levels, key=self._level_to_value.__getitem__)
        values = np.array([self._level_to_value[level] for level in levels], dtype=np.float64)
        if inner.extrapolation == "error":
            values = np.clip(values, inner._lo, inner._hi)
        B = inner._basis_matrix(values)
        B = B.toarray() if sp.issparse(B) else np.asarray(B, dtype=np.float64)
        blocks: list[NDArray] = []
        bounds: list[NDArray] = []
        for rule in self._level_rules:
            if rule.kind == "cap":
                base = levels.index(self._base_level)
                D = np.delete(B - B[base], base, axis=0)
                lo, hi = cast(tuple[float, float], rule.bounds)
            else:
                D = np.diff(B, axis=0)
                up, down = cast(tuple[float, float], rule.bounds)
                lo, hi = 1.0 - down, 1.0 + up
            blocks += [D, -D]
            bounds += [np.full(len(D), np.log(lo)), np.full(len(D), -np.log(hi))]
        A = np.vstack(blocks)
        b = np.concatenate(bounds)
        scale = np.max(np.abs(A), axis=1)
        keep = scale > 0.0
        return LinearConstraintSet(A=A[keep] / scale[keep, None], b=b[keep] / scale[keep])

    def shape_axis(self, x: NDArray) -> tuple[NDArray, NDArray[np.bool_]]:
        """The inner basis' numeric axis, and the rows that sit on it.

        A shape constraint is stated over the inner spline's level-score axis,
        so engines must resolve it through the wrapper as ``transform`` does.
        ``specials=`` rows have no coordinate and are absent from the first
        return value; the mask lets a caller drop the same rows from per-row
        weights.
        """
        labels = self._resolved_labels(x)
        if not self.has_specials:
            return self._map_to_numeric(labels), np.ones(len(labels), dtype=bool)
        ordered_mask = np.asarray(~self._special_mask(labels).any(axis=1), dtype=bool)
        return self._map_to_numeric(labels[ordered_mask]), ordered_mask

    def _build_monotone_constraints_raw(self) -> LinearConstraintSet:
        """Forward the inner spline's raw monotone geometry to the builder.

        The builder asks for raw geometry off the feature spec, which here is
        the wrapper; the constraint is already in the inner spline's
        coefficient space. Because the wrapper always has this method, the
        builder's own backstop cannot fire, so the refusal for a basis
        without the geometry is raised here. A cap or step limit adds its rows
        at the levels (``_level_rows``) below the inner spline's shape rows.
        """
        from superglm.features._spline_build import _uses_fit_time_shape_constraints

        inner = self._basis_spline
        raw_builder = getattr(inner, "_build_monotone_constraints_raw", None)
        if raw_builder is None:
            raise RuntimeError(
                f"cannot build QP constraints for OrderedCategorical(basis="
                f"{type(inner).__name__}(...)): its raw constraint geometry is "
                f"unavailable. Fit-time shape constraints need a basis that "
                f"states them in coefficient space; use kind='cr', kind='bs' or kind='ps'."
            )
        if not self._level_rules:
            return raw_builder()
        if _uses_fit_time_shape_constraints(inner):
            shape = raw_builder()
        else:
            width = cast(Any, inner)._n_basis
            shape = LinearConstraintSet(A=np.zeros((0, width)), b=np.zeros(0))
        level = self._level_rows()
        return LinearConstraintSet(
            A=np.vstack([shape.A, level.A]), b=np.concatenate([shape.b, level.b])
        )

    def __repr__(self) -> str:
        n = self._n_levels
        if self._spline is not None:
            return f"OrderedCategorical(basis={self._spline!r}, {n} levels)"
        # Reachable only by a pre-0.24 step-mode pickle; repr must not raise,
        # since it is exactly what someone debugging such a pickle prints.
        return f"OrderedCategorical(basis={self.basis!r}, {n} levels)"

    @property
    def has_specials(self) -> bool:
        """True when one or more levels are fitted as free effects."""
        return bool(self._specials)

    @property
    def basis_kind(self) -> str:
        """Kind of the inner basis: ``"spline"``, ``"piecewise"`` or ``"polynomial"``."""
        from superglm.features.spline import _SplineBase

        inner = self._basis_spline
        if isinstance(inner, _SplineBase):
            return "spline"
        if isinstance(inner, Piecewise):
            return "piecewise"
        return "polynomial"

    def _install_position_axis(self) -> None:
        """Map levels to positions 0..L-1 for a Piecewise/Polynomial inner basis.

        Runs after grouping resolution: grouped levels take consecutive
        positions, and each original level sits at its group's position.
        """
        self._level_to_value = {
            level: float(position) for position, level in enumerate(self._smooth_levels)
        }
        if self._grouping is not None:
            position_of_label = {
                str(level): float(position) for position, level in enumerate(self._smooth_levels)
            }
            self._original_level_to_value = {
                str(original): position_of_label[
                    str(self._grouping.original_to_group.get(str(original), original))
                ]
                for original in self._declared_smooth_levels
            }

    def _resolve_declared_position(self, entry: Any, *, parameter: str) -> int:
        """Resolve one stated break/knot to a position in the DECLARED levels.

        Names are the primary vocabulary; integer positions (including
        integer-valued floats) are the escape hatch. Anything else has no
        band to point at and refuses.
        """
        declared_texts = [str(level) for level in self._declared_smooth_levels]
        if isinstance(entry, str):
            if entry not in declared_texts:
                raise ValueError(
                    f"OrderedCategorical {parameter} entry {entry!r} does not name a "
                    f"declared smooth level. Declared levels (in order): "
                    f"{declared_texts!r}."
                )
            return declared_texts.index(entry)
        if isinstance(entry, bool):
            raise ValueError(
                f"OrderedCategorical {parameter} entry {entry!r} is not a level name "
                "or an integer position."
            )
        if isinstance(entry, int | np.integer):
            position = int(entry)
        elif isinstance(entry, float | np.floating) and float(entry).is_integer():
            position = int(entry)
        else:
            raise ValueError(
                f"OrderedCategorical {parameter} entry {entry!r} lies between bands: "
                "on a level axis a break/knot is a band, so state a level name or "
                "an integer position."
            )
        if not 0 <= position < len(declared_texts):
            raise ValueError(
                f"OrderedCategorical {parameter} position {position} is outside the "
                f"declared levels 0..{len(declared_texts) - 1} "
                f"({declared_texts!r})."
            )
        return position

    def _grouped_break_position(self, declared_position: int, *, parameter: str) -> int:
        """Map a declared-level position through the grouping, guarding breaks.

        A grouping that absorbs a stated break level or straddles it refuses,
        naming both; grouping within a segment is allowed, and a renamed
        singleton group follows the rename.
        """
        level = self._declared_smooth_levels[declared_position]
        if self._grouping is None:
            return declared_position
        grouping = self._grouping
        label = str(grouping.original_to_group.get(str(level), str(level)))
        members = [str(m) for m in grouping.group_to_originals.get(label, [str(level)])]
        if len(members) > 1:
            raise ValueError(
                f"OrderedCategorical grouping absorbs the stated {parameter} at level "
                f"{str(level)!r}: group {label!r} merges {members!r}. A break/knot is a "
                "stated kink -- regrouping it is a spec change, not an edit. Ungroup "
                "those levels, or restate the term without this break."
            )
        straddle = self._straddling_group(declared_position)
        if straddle is not None:
            other_label, other_members = straddle
            raise ValueError(
                f"OrderedCategorical grouping straddles the stated {parameter} at "
                f"level {str(level)!r}: group {other_label!r} spans "
                f"{other_members!r}, which sit on both sides of "
                "it. Group within one segment, or restate the term without this "
                "break."
            )
        smooth_texts = [str(lev) for lev in self._smooth_levels]
        return smooth_texts.index(label)

    def _straddling_group(self, declared_position: int) -> tuple[str, list[str]] | None:
        """The group whose levels sit on both sides of a declared level, if one does."""
        grouping = self._grouping
        if grouping is None:
            return None
        position_of = {str(lev): index for index, lev in enumerate(self._declared_smooth_levels)}
        for other_label, other_members in grouping.group_to_originals.items():
            member_positions = [position_of[str(m)] for m in other_members if str(m) in position_of]
            if len(member_positions) > 1 and min(member_positions) < declared_position < max(
                member_positions
            ):
                return str(other_label), [str(m) for m in other_members]
        return None

    def _resolve_inner_piecewise(self, piecewise: Piecewise) -> None:
        """Resolve a Piecewise inner basis onto the level-position axis."""
        n = self._n_levels
        piecewise._on_level_axis = True
        piecewise.lower = 0.0
        piecewise.upper = float(n - 1)
        breaks = piecewise.breaks
        if isinstance(breaks, int | np.integer):
            # Int-mode places breaks from the data; there is nothing stated to
            # resolve or guard.
            return
        stated = list(breaks)
        resolved: list[int] = []
        for entry in stated:
            declared_position = self._resolve_declared_position(entry, parameter="Piecewise break")
            resolved.append(
                self._grouped_break_position(declared_position, parameter="Piecewise break")
            )
        for entry, position in zip(stated, resolved):
            if position <= 0 or position >= n - 1:
                edge = "first" if position <= 0 else "last"
                raise ValueError(
                    f"OrderedCategorical Piecewise break {entry!r} resolves to the "
                    f"{edge} level, which is already a boundary knot; state breaks "
                    "strictly between the first and last levels."
                )
        for i in range(len(resolved) - 1):
            if resolved[i] >= resolved[i + 1]:
                raise ValueError(
                    f"OrderedCategorical Piecewise breaks must be stated in strictly "
                    f"ascending level order: {stated[i]!r} (position {resolved[i]}) is "
                    f"not below {stated[i + 1]!r} (position {resolved[i + 1]})."
                )
        piecewise.breaks = [float(position) for position in resolved]
        if piecewise.degrees is not None:
            knot_positions = [0, *resolved, n - 1]
            for segment, degree in enumerate(piecewise.degrees):
                span = knot_positions[segment + 1] - knot_positions[segment]
                if degree >= 2 and span < degree:
                    left = self._smooth_levels[knot_positions[segment]]
                    right = self._smooth_levels[knot_positions[segment + 1]]
                    raise ValueError(
                        f"OrderedCategorical Piecewise segment from {str(left)!r} to "
                        f"{str(right)!r} spans {span + 1} band(s) but states degree "
                        f"{degree}, which needs at least {degree + 1}. Lower the "
                        "degree or move the break."
                    )

    def _resolve_inner_polynomial(self, polynomial: Polynomial) -> None:
        """Validate a Polynomial inner basis against the level count."""
        n = self._n_levels
        if polynomial.degree > n - 1:
            raise ValueError(
                f"OrderedCategorical(basis={polynomial!r}) needs max(powers) <= "
                f"n_levels - 1: {n} smooth level(s) sit at {n} distinct positions, "
                f"which identify orthogonal components only up to degree {n - 1}."
            )

    def _resolve_spline_named_knots(self, spline: _SplineBase) -> None:
        """Resolve ``Spline(knots=[...])`` band names to level values.

        Names resolve to the named level's value on the smooth's axis; numeric
        entries stay axis values. The collapse guard applies to named knots
        only, since a numeric knot states a coordinate, not a band.
        """
        named = getattr(spline, "_named_knots", None)
        if named is None:
            return
        values: list[float] = []
        for entry in named:
            if isinstance(entry, str):
                declared_position = self._resolve_declared_position(entry, parameter="Spline knot")
                grouped_position = self._grouped_break_position(
                    declared_position, parameter="Spline knot"
                )
                values.append(float(self._level_to_value[self._smooth_levels[grouped_position]]))
            else:
                values.append(float(entry))
        for i in range(len(values) - 1):
            if values[i] >= values[i + 1]:
                raise ValueError(
                    f"OrderedCategorical Spline knots must be stated in strictly "
                    f"ascending level order: {named[i]!r} (value {values[i]:.10g}) is "
                    f"not below {named[i + 1]!r} (value {values[i + 1]:.10g})."
                )
        spline._explicit_knots = np.asarray(values, dtype=np.float64)
        spline._named_knots = None
        spline.n_knots = len(values)
        # Re-run the strictly-inside boundary check the numeric spelling got at
        # construction, so both spellings refuse identically. A knot at the data
        # boundary is redundant, not wrong, and is accepted as numerically.
        from superglm.features._spline_config import _coerce_boundary

        spline._explicit_boundary = _coerce_boundary(
            spline._explicit_boundary, spline._explicit_knots
        )

    def _resolve_spline_named_ranges(self, spline: _SplineBase) -> None:
        """Resolve ``PolynomialRange`` band-name edges to level values.

        A range edge names a band exactly as a Spline knot does, so it
        resolves, and is guarded against a grouping that absorbs or straddles
        it, the same way; numeric edges stay axis values.
        """
        spline._polynomial_ranges = tuple(
            replace(r, lo=self._range_edge_value(r.lo), hi=self._range_edge_value(r.hi))
            for r in spline._polynomial_ranges
        )

    def _range_edge_value(self, edge: float | str) -> float | str:
        if not isinstance(edge, str):
            return edge
        parameter = "PolynomialRange edge"
        declared_position = self._resolve_declared_position(edge, parameter=parameter)
        grouped_position = self._grouped_break_position(declared_position, parameter=parameter)
        return float(self._level_to_value[self._smooth_levels[grouped_position]])

    def _init_spline(self) -> None:
        """Create the internal basis: a deep copy of ``basis`` that we own,
        resolved against the declared levels (band names to positions or
        values, level-axis flags, ``n_knots`` clamped to the level count)."""
        self._spline = copy.deepcopy(self._spline_obj)
        if isinstance(self._spline, Piecewise):
            self._resolve_inner_piecewise(self._spline)
            return
        if isinstance(self._spline, Polynomial):
            self._resolve_inner_polynomial(self._spline)
            return
        self._resolve_spline_named_knots(self._spline)
        self._resolve_spline_named_ranges(self._spline)
        if self._spline.n_knots > self._n_levels - 1:
            effective = self._n_levels - 1
            requested = self._spline.n_knots
            kind = _spline_kind_name(self._spline)
            if self._basis_was_explicit:
                remedy = f"Pass basis=Spline(kind={kind!r}, n_knots={effective}) instead."
            else:
                # `basis=` was omitted: say where the number came from and give
                # the declaration that silences it.
                remedy = (
                    f"No basis= was given, so this is the default "
                    f"Spline(kind={kind!r}, n_knots={requested}); pass "
                    f"basis=Spline(kind={kind!r}, n_knots={effective}) to declare it."
                )
            warnings.warn(
                f"{_CLAMP_WARNING_PREFIX}: n_knots={requested} clamped to "
                f"{effective} (n_levels - 1). {remedy}",
                UserWarning,
                stacklevel=3,
            )
            self._spline.n_knots = effective

    def _map_to_numeric(self, x: NDArray) -> NDArray:
        """Map categorical values to their numeric representations (vectorized).

        Expects x to already be mapped through grouping if applicable
        (callers build() and transform() handle that).
        """
        return pd.Series(x).map(self._level_to_value).values.astype(np.float64)

    def _choose_base(self, x: NDArray, sample_weight: NDArray | None) -> None:
        """Choose the base level for relativities.

        Specials are excluded: the base anchors every relativity and must lie
        on the smooth. A ``base=`` a rebuild marked as a band
        (``_base_is_level``) names that band even when it reads "first".
        """
        if self._base_level and self._base_level in self._smooth_levels:
            return

        policy = None if self._base_is_level else self.base
        if policy == "most_exposed" and sample_weight is not None:
            exp_by_level = {
                lev: float(sample_weight[x == lev].sum()) for lev in self._smooth_levels
            }
            self._base_level = max(exp_by_level, key=exp_by_level.get)
        elif policy == "most_exposed" and sample_weight is None:
            self._base_level = self._smooth_levels[0]
        elif policy == "first":
            self._base_level = self._smooth_levels[0]
        else:
            # `base=` is the user's spelling and need not match the declaration
            # (`base="1"` against `order=[1, 2, 3]`), so canonicalise it.
            resolved = _declared_matcher(list(self._smooth_levels))(self.base)
            if resolved in self._smooth_levels:
                self._base_level = resolved
            else:
                raise ValueError(f"Base '{self.base}' not found in levels: {self._smooth_levels}")

        self._non_base = [lev for lev in self._smooth_levels if lev != self._base_level]

    # ── Build ──────────────────────────────────────────────────────

    def _canonical(self, x: NDArray) -> NDArray:
        """Map raw column values onto the declared levels they denote.

        The single point where the data's spelling of a level meets the
        declaration's. Everything downstream -- validation, the numeric map, the
        special mask, the grouping, the base -- then works in one namespace.
        """
        declared: list[Any] = list(self._ordered_levels)
        if self._grouping is not None:
            declared = list(self._grouping.all_original_levels) + declared
        match = _declared_matcher(declared)
        return np.array([match(value) for value in x], dtype=object)

    def build(
        self,
        x: NDArray,
        sample_weight: NDArray[np.floating] | None = None,
    ) -> GroupInfo | list[GroupInfo]:
        """Build design columns from ordered categorical data.

        With ``specials=``, returns two GroupInfos: the penalized spline block,
        then the unpenalized special-indicator block, an order ``_split_beta``
        and ``transform`` rely on. When every special is pinned there is no
        second block.
        """
        return self._build_with_geometry(
            x,
            reporting_weight=sample_weight,
            geometry_weight=sample_weight,
        )

    def _build_with_geometry(
        self,
        x: NDArray,
        *,
        reporting_weight: NDArray[np.floating] | None,
        geometry_weight: NDArray[np.floating] | None,
    ) -> GroupInfo | list[GroupInfo]:
        """Build with explicit outer reporting mass and inner geometry mass."""
        x = self._canonical(np.asarray(x).ravel())

        if self._grouping is not None:
            x = _grouping_labels(x)
            self._refuse_unknown_training_levels(x)
            x = pd.Series(x).map(self._grouping.original_to_group).values
        else:
            self._refuse_unknown_training_levels(x)

        return self._build_spline(x, reporting_weight, geometry_weight)

    def _refuse_unknown_training_levels(self, x: NDArray) -> None:
        """Refuse, as a fit, training levels outside the declaration or its grouping.

        A level the fit does not admit is either not declared at all, or
        declared but missing from a grouping built from other data; the
        sentence says which. Predict-time levels keep their own sentence.
        """
        _validate_missing_only(x)
        unknown = set(pd.unique(np.asarray(x).ravel()).tolist()) - self._known_levels
        if not unknown:
            return
        declared = [*self._declared_smooth_levels, *self._special_display]
        texts = {str(level) for level in declared}
        undeclared = [
            level for level in unknown if self._grouping is None or str(level) not in texts
        ]
        if undeclared:
            raise ValueError(
                f"Training data contains levels this OrderedCategorical does not declare: "
                f"{sorted(undeclared, key=str)}. Declared: {declared}. Add them to order= or "
                f"values=, or to specials= for a level off the ordered axis, or leave those "
                f"rows out."
            )
        raise ValueError(
            f"Training data contains levels the grouping does not cover: "
            f"{sorted(unknown, key=str)}. Covered: "
            f"{sorted(self._grouping.all_original_levels, key=str)}. Build the grouping from "
            f"the full column, or leave those rows out."
        )

    def _build_inner_info(
        self,
        numeric: NDArray[np.float64],
        sample_weight: NDArray | None,
    ) -> GroupInfo | list[GroupInfo]:
        """Build the inner basis on the mapped numeric axis.

        ``sample_weight`` is the learned-state stream the caller chose, and
        every inner basis consumes it exactly.
        """
        from dataclasses import replace

        from superglm.features.spline import OneValueError, _SplineBase

        inner = self._basis_spline
        if isinstance(inner, _SplineBase):
            self._place_basis_rules()
            try:
                info = inner.build(numeric, sample_weight=sample_weight)
            except OneValueError:
                # The spline's message names a level score and its kind=, which
                # an ordered term has neither of; it takes its kind in basis=.
                raise OneValueError(
                    "every row of this ordered term is at one level, so its natural "
                    "spline basis has no range to place its knots on. Drop the term, or "
                    "pass basis=Spline(kind='ps'), which fits a term with one observed level."
                ) from None
            if not self._level_rules:
                return info
            # The level rows join the inner spline's, composed through its
            # identifiability projection as the spline composes its own; a cap
            # with no shape rule makes this the term's only QP geometry.
            info = cast(GroupInfo, info)
            constraints = self._build_monotone_constraints_raw()
            if info.projection is not None:
                constraints = constraints.compose(info.projection)
            return replace(
                info,
                constraints=constraints,
                monotone_engine="qp",
                raw_to_solver_map=info.projection,
            )
        info = inner.build(numeric, sample_weight=sample_weight)
        if isinstance(info, GroupInfo):
            # Both parametric bases are an unpenalized main block, like the
            # specials block. Polynomial is too, unlike the numeric-axis term:
            # its Wald and per-power z rows are invalid under lambda1 shrinkage.
            # Row compression stays off; it is verified for the numeric-axis term.
            info = replace(info, penalized=False, supports_row_compression=False)
        return info

    def _build_spline(
        self,
        x: NDArray,
        reporting_weight: NDArray | None,
        geometry_weight: NDArray | None,
    ) -> GroupInfo | list[GroupInfo]:
        """Map levels to the numeric axis and delegate to the inner basis."""
        self._choose_base(x, reporting_weight)
        if not self.has_specials:
            return self._build_inner_info(self._map_to_numeric(x), geometry_weight)

        special_mask = self._special_mask(x)
        # The declared mask splits the design: a pinned special has no
        # coordinate either, so its rows carry zero in both blocks.
        ordered_mask = np.asarray(~special_mask.any(axis=1), dtype=bool)
        # Effective observation, as Categorical's zero-count pin: rows when
        # unweighted, total weight otherwise; zero-weight rows leave X'WX blind.
        if reporting_weight is None:
            effective = special_mask.sum(axis=0).astype(np.float64)
        else:
            effective = np.asarray(reporting_weight, dtype=np.float64).ravel() @ special_mask
        self._active_specials = [j for j in range(len(self._specials)) if effective[j] > 0.0]
        self._pinned_specials = [
            self._special_display[j] for j in range(len(self._specials)) if not effective[j] > 0.0
        ]
        if self._pinned_specials:
            warnings.warn(
                f"OrderedCategorical special level(s) "
                f"{sorted(self._pinned_specials, key=str)} have no effective training rows "
                "and are pinned to zero contribution for this fit: no indicator column is "
                "emitted for them. Each remains a known level, and its rows predict with no "
                "contribution from this term. Supply rows carrying weight, or drop the "
                "level from specials=.",
                UserWarning,
                stacklevel=2,
            )

        # The identifiability constraint is a column sum over present rows, so
        # the spline is built only on the rows its block is nonzero on.
        ordered_numeric = self._map_to_numeric(x[ordered_mask])
        ordered_geometry_weight = (
            None
            if geometry_weight is None
            else np.asarray(geometry_weight, dtype=np.float64).ravel()[ordered_mask]
        )
        spline_info = self._build_inner_info(ordered_numeric, ordered_geometry_weight)
        # The spline bases reachable here return exactly one GroupInfo, which
        # the two-block contract assumes.
        if not isinstance(spline_info, GroupInfo):
            raise TypeError(
                f"OrderedCategorical specials require a single-group spline basis; "
                f"{type(self._basis_spline).__name__} produced {len(spline_info)} groups."
            )
        spline_info = self._expand_rows(spline_info, ordered_mask)

        if not self._active_specials:
            # Every special pinned: no free block, and the term has the shape it
            # has without specials (a zero-column group would leak downstream).
            return spline_info
        # A row carries at most one special, so this block is one-hot and is
        # carried as cat_codes: a sparse block would disqualify the design from
        # the structured kernels and the packed centering path.
        active = special_mask[:, self._active_specials]
        special_codes = np.where(active.any(axis=1), active.argmax(axis=1), -1).astype(
            np.intp, copy=False
        )
        special_info = GroupInfo(
            columns=None,
            n_cols=len(self._active_specials),
            cat_codes=special_codes,
            penalty_matrix=None,
            reparametrize=False,
            penalized=False,
            subgroup_name="special",
            projection=None,
        )
        return [spline_info, special_info]

    def _special_mask(self, x: NDArray) -> NDArray[np.bool_]:
        """(n, n_specials) boolean membership matrix, column j == self._specials[j].

        Compared through a string view, since ``_specials`` is string-coerced,
        and against the raw label for non-str declarations (a float column
        renders ``9.0`` as ``"9.0"``). pandas yields ``False`` when the dtype
        cannot hold the label. Grouped labels are spelled as reported.
        """
        raw = pd.Series(np.asarray(x).ravel())
        labels = raw.astype(str).to_numpy()
        columns = []
        for lev, raw_lev, shown in zip(self._specials, self._special_raw, self._special_display):
            hit = (labels == lev) | (labels == str(shown))
            if not isinstance(raw_lev, str):
                hit = hit | np.asarray(raw == raw_lev, dtype=bool)
            columns.append(hit)
        return np.column_stack(columns)

    @property
    def _active_special_cols(self) -> list[int]:
        """Indices into ``_specials`` of the indicator columns this fit emits.

        Per-build state: an older pickle or an unbuilt spec has every declared
        special active.
        """
        active = getattr(self, "_active_specials", None)
        if active is None:
            return list(range(len(self._specials)))
        return list(active)

    @property
    def _n_special_cols(self) -> int:
        """Width of the free block AS FITTED -- pinned specials own no column."""
        return len(self._active_special_cols)

    def _split_beta(
        self, beta: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Split a full-width feature coefficient vector into its two blocks.

        Block order is the build() contract: spline first, specials second. The
        width check is an equality, because the spline block alone is longer
        than ``n_special`` and would otherwise split silently.
        """
        beta = np.asarray(beta, dtype=np.float64).ravel()
        if not self.has_specials:
            return beta, np.empty(0, dtype=np.float64)
        n_special = self._n_special_cols
        expected = self._spline_n_cols() + n_special
        if len(beta) != expected:
            raise ValueError(
                f"OrderedCategorical received {len(beta)} coefficients but its blocks "
                f"are {expected} wide ({expected - n_special} spline + {n_special} "
                "special); a caller passed only the spline block or the blocks are "
                "out of order."
            )
        return beta[: len(beta) - n_special], beta[len(beta) - n_special :]

    def _spline_n_cols(self) -> int:
        """Fitted width of the spline block, for zero-filling special rows."""
        probe = np.array([self._level_to_value[self._smooth_levels[0]]], dtype=np.float64)
        return int(np.asarray(self._basis_spline.transform(probe)).shape[1])

    @staticmethod
    def _expand_rows(info: GroupInfo, ordered_mask: NDArray[np.bool_]) -> GroupInfo:
        """Re-embed an ordered-row basis into full-length rows, zero elsewhere."""
        import dataclasses

        compact = sp.csr_matrix(info.columns, dtype=np.float64, copy=True)
        compact.sum_duplicates()
        compact.eliminate_zeros()
        compact.sort_indices()
        # Insert empty rows through repeated CSR pointers. Keeping the data and
        # column indices in order maps each compact row to its own ordered row,
        # without allocating a Python list for every row as LIL scatter does.
        indptr = np.zeros(len(ordered_mask) + 1, dtype=compact.indptr.dtype)
        indptr[1:][ordered_mask] = np.diff(compact.indptr)
        np.cumsum(indptr, out=indptr)
        expanded = sp.csr_matrix(
            (compact.data, compact.indices, indptr),
            shape=(len(ordered_mask), compact.shape[1]),
        )
        return dataclasses.replace(info, columns=expanded)

    # ── Transform ──────────────────────────────────────────────────

    def _resolved_labels(self, x: NDArray) -> NDArray:
        """Canonicalize, validate and group a raw column into fitted labels.

        The one preamble every scoring-side entry point needs before it can ask
        anything of a level: put the data's spelling into the declaration's
        namespace, refuse levels the fit never saw, then collapse grouped levels
        onto the label their group was fitted under.
        """
        x = self._canonical(np.asarray(x).ravel())
        if self._grouping is None:
            _validate_categorical_levels(x, self._known_levels)
            return x
        x = _grouping_labels(x)
        _validate_categorical_levels(x, self._known_levels | set(self._grouping.grouped_levels))
        return np.array(
            [self._grouping.original_to_group.get(v, v) for v in x],
            dtype=object,
        )

    def transform(self, x: NDArray) -> NDArray:
        """Build design matrix for new data using learned parameters."""
        x = self._resolved_labels(x)
        if not self.has_specials:
            return self._basis_spline.transform(self._map_to_numeric(x))
        special_mask = self._special_mask(x)
        # Held out of the smooth on the declared mask, given a column only on the
        # active one: a pinned special's row is zero across the whole term.
        ordered_mask = ~special_mask.any(axis=1)
        spline_cols = np.zeros((len(x), self._spline_n_cols()), dtype=np.float64)
        if ordered_mask.any():
            spline_cols[ordered_mask] = self._basis_spline.transform(
                self._map_to_numeric(x[ordered_mask])
            )
        indicators = special_mask[:, self._active_special_cols].astype(np.float64)
        return np.column_stack([spline_cols, indicators])

    def score(self, x: NDArray, beta: NDArray[np.floating]) -> NDArray[np.floating]:
        """Score the fitted ordered-categorical contribution directly on new data."""
        x = self._resolved_labels(x)
        spline_beta, special_beta = self._split_beta(beta)
        if not self.has_specials:
            return self._basis_spline.score(self._map_to_numeric(x), spline_beta)
        special_mask = self._special_mask(x)
        ordered_mask = ~special_mask.any(axis=1)
        # A pinned special contributes through neither term of this sum: it has no
        # column in the mask below and no coordinate on the smooth's axis, so its
        # rows keep the 0.0 they start at.
        out = special_mask[:, self._active_special_cols].astype(np.float64) @ special_beta
        if ordered_mask.any():
            out[ordered_mask] = self._basis_spline.score(
                self._map_to_numeric(x[ordered_mask]), spline_beta
            )
        return out

    # ── Reconstruct ────────────────────────────────────────────────

    def _base_log_effect(self, beta: NDArray[np.floating]) -> float:
        """Return the fitted term effect at the reporting reference level."""
        spline_beta, _ = self._split_beta(beta)
        base_value = np.array([self._level_to_value[self._base_level]], dtype=np.float64)
        return float(self._basis_spline.score(base_value, spline_beta)[0])

    def reconstruct(self, beta: NDArray[np.floating]) -> dict[str, Any]:
        """Convert fitted coefficients to interpretable output."""
        return self._reconstruct_spline(beta)

    def _reconstruct_spline(self, beta: NDArray) -> dict[str, Any]:
        """Spline mode: delegate to internal spline, add per-level annotations.

        Shifts the curve so that the base level has log_relativity=0 (relativity=1),
        giving proper categorical-style relativities. Specials are reported on the
        same scale — beta_special minus the curve at the base — so the rating table
        can reconstruct predictions from one level table.
        """
        spline_beta, special_beta = self._split_beta(beta)
        raw = self._basis_spline.reconstruct(spline_beta)

        # Per-level values on the fitted curve
        level_values = np.array([self._level_to_value[lev] for lev in self._smooth_levels])
        level_log_rels = np.asarray(
            self._basis_spline.score(level_values, spline_beta), dtype=np.float64
        )

        # Shift so base level = 0 (relativity = 1)
        base_shift = self._base_log_effect(beta)
        level_log_rels = level_log_rels - base_shift
        raw["log_relativity"] = raw["log_relativity"] - base_shift
        raw["relativity"] = np.exp(raw["log_relativity"])

        # Report specials under their domain labels; read `_ordered_levels`
        # rather than re-deriving it, since the two lists are joined on equality.
        all_levels = list(self._ordered_levels)
        # A pinned special reports the zero contribution the model predicts for
        # it, -f(base) on the base-anchored scale, so every declared level stays
        # in the rating table.
        special_effects = np.zeros(len(self._specials), dtype=np.float64)
        special_effects[self._active_special_cols] = np.asarray(special_beta, dtype=np.float64)
        all_log_rels = np.concatenate([level_log_rels, special_effects - base_shift])

        raw["base_level"] = self._base_level
        raw["levels"] = all_levels
        raw["special_levels"] = list(self._special_display)
        raw["pinned_specials"] = list(getattr(self, "_pinned_specials", ()))
        # Keyed on the smooth levels only — a special never receives a coordinate
        # on the spline's axis.
        raw["level_values"] = dict(zip(self._smooth_levels, level_values.tolist()))
        raw["level_log_relativities"] = dict(zip(all_levels, all_log_rels.tolist()))
        raw["level_relativities"] = dict(zip(all_levels, np.exp(all_log_rels).tolist()))
        return raw

    # ── Reparametrisation ──────────────────────────────────────────

    def set_reparametrisation(self, R_inv: NDArray) -> None:
        self._basis_spline.set_reparametrisation(R_inv)


def resolve_interaction_parent(spec: Any, x: NDArray) -> tuple[Any, NDArray]:
    """Resolve one interaction parent (spec, column) for assembly.

    Identity for every spec — including ``None``, which FactorSmooth group
    columns carry — except OrderedCategorical, which contributes its inner
    Spline on the mapped numeric scores, applying the same grouping, level
    validation, and score mapping its own ``build``/``transform`` apply.
    A term carrying ``specials=`` cannot parent an interaction: a special is
    a free level with no position on the spline axis, so there is no single
    marginal smooth to cross with.
    """
    if not isinstance(spec, OrderedCategorical):
        return spec, x
    if spec.has_specials:
        raise NotImplementedError(
            f"OrderedCategorical with specials={spec._specials!r} cannot parent an "
            "interaction: a special is a free level with no position on the spline "
            "axis, so the term has no single marginal smooth to cross with; drop "
            "specials= to interact the smoothed ordinal parent, or use a Categorical "
            "feature for unsmoothed level effects."
        )
    if spec.basis_kind != "spline":
        # Interactions cross a penalized marginal smooth; a Piecewise/Polynomial
        # basis is an unpenalized parametric block. Backstop for the refusal
        # in dm_builder.
        raise NotImplementedError(
            f"OrderedCategorical(basis={type(spec._basis_spline).__name__}(...)) "
            "cannot parent an interaction: the interaction machinery crosses a "
            "penalized marginal smooth, and this inner basis is an unpenalized "
            "parametric block. Use basis=Spline(...) for an interactable ordinal "
            "parent, or a Categorical feature for level-by-level structure."
        )
    # Canonicalising is safe here because `FactorSmooth`, whose grouping parent
    # is read as labels, returns from `resolve_interaction_parent_of` first.
    return spec._basis_spline, spec._map_to_numeric(spec._resolved_labels(x))


def resolve_interaction_parent_of(ispec: Any, spec: Any, x: NDArray) -> tuple[Any, NDArray]:
    """``resolve_interaction_parent`` for a parent OF a given interaction.

    Identical to it except for ``FactorSmooth``, whose second parent is a
    grouping column read as labels: mapping it to level scores would re-key
    the term's levels, so a ``FactorSmooth`` takes both columns as the frame
    carries them.
    """
    from superglm.features.factor_smooth import FactorSmooth

    if isinstance(ispec, FactorSmooth):
        return spec, x
    return resolve_interaction_parent(spec, x)
