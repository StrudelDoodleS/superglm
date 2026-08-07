"""OrderedCategorical feature: ordered categories with a spline basis.

Actuarial pricing data frequently contains continuous variables that have been
pre-binned into ordered categories (e.g. age bands "18-25", "26-35", ...).
This feature type respects the ordering with two modes:

- **spline**: map categories to numeric values, build a spline on those values
- **step**: one-hot encode with a first-difference penalty (D1'D1) so adjacent
  categories are soft-fused (deprecated)
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.typing import NDArray

from superglm.features.categorical import _grouping_labels, _validate_categorical_levels
from superglm.types import GroupInfo, LinearConstraintSet

if TYPE_CHECKING:  # Spline imports this module at runtime; keep the cycle type-only.
    from superglm.features.spline import _SplineBase


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

    An ``OrderedCategorical`` has TWO sources for a level's identity -- the
    ``order=``/``values=`` declaration and the data column -- and they need not
    spell it the same way. Declared ``9`` against a float column is ``"9"`` on one
    side and ``"9.0"`` on the other, and those never compare equal. Every site
    that then asks "is this level known / special / the base" answers no, in a
    different way each time: a crash, a silent no-op, a guard that fails open.

    The declaration is canonical. This maps the DATA onto it, once, at the edge,
    so no site downstream has to know two spellings exist.

    Numeric matching applies only when the raw value is not a string. That is the
    line between ``9.0`` (a number the column happens to store as a float, which
    denotes declared ``9``) and ``"001"`` (a string whose leading zeros are part
    of its identity, and which must NOT be matched by declared ``1``). Zero-padded
    identifiers are common enough in pricing data that collapsing them would be a
    worse bug than the one this fixes.
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

    def _spellings(raw: Any) -> list[str]:
        """The renderings a numeric value can plausibly appear under."""
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return []
        out = [repr(value), str(value)]
        if value.is_integer():
            out.append(str(int(value)))
        return out

    def match(raw: Any) -> Any:
        text = str(raw)
        if text in by_text:
            return by_text[text]
        if isinstance(raw, str) and not numeric_strings:
            # A raw STRING is its own identity. "001" must never be claimed by a
            # declared 1: leading zeros are meaningful in a policy or vehicle
            # code. Only the grouping path opts in, because a LevelGrouping's
            # labels are stringified DATA rather than user-authored strings.
            return raw
        # Only a rendering of the SAME number counts. "9" and 9.0 render each
        # other ("9" -> {"9.0", "9"}), so they match; "001" renders {"1.0", "1"},
        # which is not "001", so a declared 1 can never claim it. That asymmetry
        # is the whole safety property -- leading zeros are identity in a policy
        # or vehicle code, and collapsing them would be a worse bug than the one
        # this fixes.
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return raw
        if value not in by_value:
            return raw
        # Only now is ambiguity a problem: `order=["1", "1.0"]` are distinct
        # labels and both are reachable by their exact spellings. It is a raw
        # value that matches NEITHER exactly, yet equals both numerically, that
        # cannot be resolved -- so refuse then, rather than refusing the
        # declaration outright and rejecting models that are perfectly fittable.
        if value in ambiguous:
            raise ValueError(
                f"Level {raw!r} is ambiguous: the declared levels spell the value {value} "
                "more than one way and this value matches neither exactly. Declare the "
                "levels consistently, or supply the level under one of its exact spellings."
            )
        matched = by_value[value]
        # float() is lossy past 2**53, so two distinct integers can land on one
        # value. When both sides are real numbers, confirm the match in their own
        # domain -- Python compares int to float exactly, so this costs one
        # comparison and makes the rule exact rather than approximate. A
        # string-mediated match (a grouping label) has only the float to go on.
        if not isinstance(raw, str) and not isinstance(matched, str) and raw != matched:
            return raw
        return matched

    return match


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

    Merging a special into an ordered group would smooth it after all, while
    ``_specials`` still reports it free — an inconsistent spec with no error.
    Merging two specials is refused for the same reason the editor refuses it
    (``_require_no_special_members``): the group label would have to replace
    both members in ``specials=``.  Renaming a single special under a new group
    label is refused for the same reason: the renamed level joins the smooth
    with no numeric position while ``specials=`` still names the original.
    Only a group that leaves its level untouched (label identical to its one
    member) is exempt.
    """
    if not special_set or grouping is None:
        return
    # A grouping must also COVER every special. One that simply omits it passes
    # construction (``_known_levels`` re-adds the special), and then build()'s
    # ``map(original_to_group)`` yields NaN for those rows: the special mask
    # misses them and they reach ``_map_to_numeric`` as NaN, dying inside scipy
    # with "Array must not contain infs or nans" -- a message that says nothing
    # about the grouping that caused it.
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


class OrderedCategorical:
    """Ordered categorical feature with spline or step basis.

    Designed for continuous variables that arrive pre-binned into ordered
    categories (e.g. age bands, mileage bands).  Maps category labels to
    numeric values and fits a smooth function through them, borrowing
    strength between adjacent levels.

    The canonical API passes a :func:`Spline` specification as ``basis``::

        OrderedCategorical(
            order=["low", "medium", "high"],
            basis=Spline(kind="ps", k=6),
        )

    Two modes are currently available:

    - **spline** (default): maps levels to numeric values (midpoints or
      linspace), builds a B-spline through them.  The spline smooths across
      levels and the penalty controls wiggliness.  With ``fit_reml()``,
      REML selects the smoothing parameter automatically — the effective
      degrees of freedom will typically be much less than the number of
      levels.

    - **step** (deprecated): one-hot encodes with a first-difference penalty
      (D1'D1) so adjacent categories are soft-fused. Use ``Spline(...)`` for
      smoothing or :class:`Categorical` for independent level effects.

    Parameters
    ----------
    values : dict[str, float] or None
        Explicit mapping from category labels to numeric values (e.g.
        midpoints: ``{"18-25": 21.5, "26-35": 30.5, ...}``).
        Mutually exclusive with ``order``.
    order : list[str] or None
        Ordered list of category labels.  Numeric values are generated as
        ``linspace(0, 1, len(order))``.  Mutually exclusive with ``values``.
    basis : Spline object, {"spline", "step"}, or None
        Pass a ``Spline(...)`` object for full control over kind, basis size,
        constraints, selection, and penalty::

            OrderedCategorical(order=[...], basis=Spline(kind="cr", k=6))

        Omitting ``basis`` retains the historical default P-spline. The
        string values ``"spline"`` and ``"step"`` are deprecated; step
        smoothing will be removed in a future release.
    kind : str or None
        Deprecated spline shortcut. Configure ``kind`` on ``basis=Spline(...)``.
    base : str
        Reporting reference level. ``"most_exposed"`` (default), ``"first"``,
        or a specific level name. In spline mode this changes only the reported
        relativities and reference-adjusted intercept, not the fitted smooth.
    n_knots : int or None
        Deprecated spline shortcut. Auto-clamped to ``n_levels - 1``.
    degree : int or None
        Deprecated spline shortcut for B-spline degree.
    select : bool or None
        Deprecated spline shortcut for double-penalty shrinkage.
    penalty : str or None
        Deprecated spline shortcut for penalty type.
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

    def __init__(
        self,
        values: dict[str, float] | None = None,
        order: list[str] | None = None,
        basis: Any | None = None,
        kind: str | None = None,
        base: str = "most_exposed",
        n_knots: int | None = None,
        degree: int | None = None,
        select: bool | None = None,
        penalty: str | None = None,
        grouping: Any = None,
        specials: list[str] | None = None,
    ):
        from superglm.features.spline import _SplineBase

        if values is not None and order is not None:
            raise ValueError("Specify exactly one of 'values' or 'order', not both.")
        if values is None and order is None:
            raise ValueError("Must specify either 'values' or 'order'.")

        self._specials: list[str] = []
        # The declarations as given, positionally aligned with `_specials`.
        # `_special_mask` needs them: the string view alone cannot match a
        # non-str label against a numeric column (9.0 renders as "9.0").
        self._special_raw: list[Any] = []
        if specials is not None:
            # Level labels live in `str` space throughout this file (grouping
            # coerces too), so coerce here or a non-str special silently fails
            # to pop from order=/values= and gets smoothed as well as claimed.
            for raw_lev in specials:
                lev = str(raw_lev)
                if lev in self._specials:
                    raise ValueError(f"Duplicate special level {lev!r} in 'specials'.")
                # The string forms of 9 and 9.0 differ ("9" vs "9.0"), so the
                # check above lets both through -- but `_special_mask` matches a
                # non-str special on RAW equality too, and 9 == 9.0, so both
                # indicator columns claim exactly the same rows. That is two
                # collinear unpenalized coefficients, and the pair is
                # unidentifiable however the solve resolves it.
                clash = next((prev for prev in self._special_raw if prev == raw_lev), None)
                if clash is not None:
                    raise ValueError(
                        f"Special level {raw_lev!r} in 'specials' duplicates {clash!r}: they "
                        "are spelled differently but compare equal, so both would claim the "
                        "same rows and their free coefficients would be unidentifiable."
                    )
                self._specials.append(lev)
                self._special_raw.append(raw_lev)
        # How each special is REPORTED: the label its domain spells it with, so it
        # sits in the same namespace as the smooth levels. Defaults to the raw
        # declaration and is refined below if the domain names it differently.
        self._special_display: list[Any] = list(self._special_raw)
        special_set = set(self._specials)
        # Ungrouped OrderedCategorical validates raw column labels, so a special
        # declared as `9` must be admissible in both the raw and the string form
        # its mask matches on; otherwise the level is rejected as unseen.
        known_special_labels: set[Any] = set(special_set)
        if specials is not None:
            known_special_labels |= set(specials)

        if special_set:
            # Match on BOTH views, because `_special_mask` does. A str-only test
            # keeps 9.0 in the smooth when the special was declared as 9 ("9.0" !=
            # "9"), while the mask still claims those rows numerically -- so the
            # level ends up both smoothed and free: a phantom spline position no
            # row occupies, and the same level reported twice.
            raw_specials = list(self._special_raw)

            def _is_special(label: Any) -> bool:
                if str(label) in special_set:
                    return True
                return any(
                    not isinstance(raw, str) and not isinstance(label, str) and label == raw
                    for raw in raw_specials
                )

            # Report a special under the label its DOMAIN uses, not the coerced
            # string. Smooth levels are reported with their raw `order=`/`values=`
            # labels, and rating tables, editor weights and exposure bars aggregate
            # row weights by the column's own labels before looking them up by the
            # reported level -- so a special reported as "9" beside neighbours
            # reported as 1.0, 2.0 comes back with zero weight and zero exposure.
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

        basis_was_explicit = basis is not None
        shortcut_values = {
            "kind": kind,
            "n_knots": n_knots,
            "degree": degree,
            "select": select,
            "penalty": penalty,
        }
        used_shortcuts = [name for name, value in shortcut_values.items() if value is not None]

        # Every shortcut takes a scalar; `basis` is the one parameter that takes
        # a Spline. Catch the swap here or it travels on unchecked: `kind` reaches
        # the private spline factory, which rejects the object against its list of
        # string kinds and names neither the parameter at fault nor the one to
        # use, while `degree`, `select` and `penalty` are never read as anything
        # but scalars and so configure the smooth with silent nonsense.
        misplaced = [
            name for name, value in shortcut_values.items() if isinstance(value, _SplineBase)
        ]
        if misplaced:
            names = " and ".join(f"`{name}=`" for name in misplaced)
            noun = "a scalar" if len(misplaced) == 1 else "scalars"
            kinds = ", ".join(sorted({type(shortcut_values[name]).__name__ for name in misplaced}))
            # Name the object instead of interpolating its repr: the repr
            # renders only a subset of the configuration (spline.py drops
            # `constraint=` and `penalty=`), so presenting it as a
            # copy-pasteable replacement would silently discard exactly the
            # arguments the caller was trying to pass.
            raise ValueError(
                f"OrderedCategorical received a Spline object in {names}, which "
                f"take{'s' if len(misplaced) == 1 else ''} {noun}. `basis=` is the "
                f"parameter that takes the smooth: move the {kinds} you built -- "
                f"with every argument you configured on it -- to "
                f"OrderedCategorical(..., basis=...)."
            )

        resolved_basis = "spline" if basis is None else basis
        resolved_kind = "ps" if kind is None else kind
        resolved_n_knots = 5 if n_knots is None else n_knots
        resolved_degree = 3 if degree is None else degree
        resolved_select = False if select is None else select
        resolved_penalty = "ssp" if penalty is None else penalty

        # Accept a Spline object as basis.
        if isinstance(resolved_basis, _SplineBase):
            self._spline_obj = resolved_basis
            self.basis = "spline"
        elif resolved_basis in ("spline", "step"):
            self._spline_obj = None
            self.basis = resolved_basis
        else:
            raise ValueError(
                f"basis must be 'spline', 'step', or a Spline object, got {resolved_basis!r}"
            )

        shortcut_list = ", ".join(f"`{name}`" for name in used_shortcuts)
        shortcut_noun = "shortcut" if len(used_shortcuts) == 1 else "shortcuts"
        shortcut_verb = "is" if len(used_shortcuts) == 1 else "are"
        if self.basis == "step":
            warnings.warn(
                "OrderedCategorical step smoothing (`basis='step'`) is deprecated and "
                "will be removed in a future release. Use `basis=Spline(...)` for "
                "smoothing or `Categorical(...)` for independent level effects.",
                FutureWarning,
                stacklevel=2,
            )
        elif self._spline_obj is not None and used_shortcuts:
            warnings.warn(
                f"OrderedCategorical spline {shortcut_noun} ({shortcut_list}) "
                f"{shortcut_verb} ignored "
                "because basis is a Spline object; configure the Spline object directly.",
                FutureWarning,
                stacklevel=2,
            )
        legacy_spline_string = basis_was_explicit and resolved_basis == "spline"
        if (
            self.basis == "spline"
            and self._spline_obj is None
            and (legacy_spline_string or used_shortcuts)
        ):
            if legacy_spline_string and used_shortcuts:
                deprecated_api = (
                    f"`basis='spline'` and OrderedCategorical spline {shortcut_noun} "
                    f"({shortcut_list}) are"
                )
            elif legacy_spline_string:
                deprecated_api = "`basis='spline'` is"
            else:
                deprecated_api = (
                    f"OrderedCategorical spline {shortcut_noun} ({shortcut_list}) {shortcut_verb}"
                )
            warnings.warn(
                f"{deprecated_api} deprecated; configure the smooth with "
                "`basis=Spline(...)` instead.",
                FutureWarning,
                stacklevel=2,
            )

        if self.basis == "step" and resolved_select:
            raise ValueError("select=True is not supported with basis='step'.")
        if self.basis == "step" and special_set:
            raise ValueError(
                "specials= is not supported with basis='step', which is deprecated. "
                "Use basis=Spline(...) for a smoothed ordinal term with free levels."
            )

        self.kind = resolved_kind
        self.base = base
        self.select = resolved_select
        self.penalty = resolved_penalty
        self.degree = resolved_degree
        self.n_knots = resolved_n_knots
        self._smooth_levels: list[str] = []
        # Levels as REPORTED, in block order: the smooth levels under their
        # `order=`/`values=` labels, then the specials under `_special_display`
        # (the domain's own label, not the string-coerced `_specials`). One
        # namespace, because `reconstruct()["levels"]` is spelled the same way
        # and every downstream row name, weight lookup and membership test joins
        # the two lists on equality.
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

        # Level labels must be distinct AS STRINGS. The grouping layer is
        # string-keyed by construction and every report joins on str(), so
        # `order=[1, "1", 2]` cannot be represented: one of the two silently wins
        # every lookup and the other can never be fitted or scored. Refuse rather
        # than pick.
        _texts = [str(level) for level in self._ordered_levels]
        _dupes = sorted({t for t in _texts if _texts.count(t) > 1})
        if _dupes:
            raise ValueError(
                f"OrderedCategorical level label(s) {_dupes!r} appear more than once once "
                "rendered as text. Levels are matched and reported by their string form, so "
                "two levels cannot share one; declare them with distinct labels."
            )

        # Grouping: validate and store. A grouping is built FROM DATA, so its
        # labels are in the column's spelling while everything here is in the
        # declaration's. Re-spell it once, so the whole spec lives in one
        # namespace and no downstream site has to reconcile them again -- the
        # editor previously carried its own copy of this, and the direct
        # `grouping=` path carried none.
        grouping = _regroup_to_declared(grouping, self._ordered_levels + list(self._specials))
        self._grouping = grouping
        # Before the numeric-position check below: a grouping that merges or
        # renames a special is wrong for a specific, explainable reason, and
        # that reason is more useful than the generic 'no numeric position'
        # symptom it would otherwise produce first.
        _require_no_grouped_specials(grouping, special_set)
        self._original_level_to_value: dict[str, float] | None = None
        if grouping is not None:
            # Preserve original level→value mapping for plot expansion
            # Keyed the same way as the regrouped levels. `_expand_grouped_term`
            # indexes this map with the grouping's (string) level names for the
            # smooth-curve expansion, so leaving it on the declared objects
            # raises KeyError out of the public term_inference path -- which the
            # direct-grouping regression missed by only exercising predict().
            _match = _declared_matcher(
                self._ordered_levels + list(self._specials), numeric_strings=True
            )
            orig_ltv = {str(_match(k)): v for k, v in self._level_to_value.items()}
            self._original_level_to_value = orig_ltv
            # Build level_to_value for grouped levels.
            # When values= was used, orig_ltv keys are original level names
            # so we average them per group. When order= was used, orig_ltv
            # keys are already the grouped level names — use them directly
            # if the group name matches, otherwise average originals.
            # A LevelGrouping is string-keyed by construction, while `values=`/
            # `order=` keys are the declared objects (ints, floats). Join on text
            # so `order=[1, 2, 3]` meets a grouping spelling them "1", "2", "3";
            # a plain `in` test silently found nothing and left the map empty,
            # which surfaced much later as "Array must not contain infs or nans".
            by_text = {str(k): v for k, v in orig_ltv.items()}
            grouped_ltv = {}
            for glev in grouping.grouped_levels:
                if str(glev) in by_text:
                    grouped_ltv[glev] = by_text[str(glev)]
                else:
                    originals = grouping.group_to_originals[glev]
                    vals = [by_text[str(o)] for o in originals if str(o) in by_text]
                    if vals:
                        grouped_ltv[glev] = float(np.mean(vals))
            # Specials are exempt by construction: a free level never receives a
            # coordinate on the spline's axis, so having no numeric position is
            # correct for them and only for them.
            missing = [
                g
                for g in grouping.grouped_levels
                if g not in grouped_ltv and str(g) not in special_set
            ]
            if missing:
                raise ValueError(
                    f"OrderedCategorical grouping produced level(s) {missing!r} with no "
                    f"numeric position; known levels are {sorted(by_text)}. The grouping "
                    "and the declaration disagree about how levels are named."
                )
            self._level_to_value = grouped_ltv
            self._smooth_levels = [lev for lev in grouping.grouped_levels if lev not in special_set]
            # _known_levels includes all *original* levels (for predict-time validation)
            self._known_levels = set(grouping.all_original_levels) | known_special_labels
        else:
            self._known_levels = set(self._smooth_levels) | known_special_labels
        self._ordered_levels = list(self._smooth_levels) + list(self._special_display)
        self._n_levels = len(self._smooth_levels)

        _require_two_smooth_levels(self._smooth_levels, special_set)
        if str(base) in special_set:
            raise ValueError(
                f"OrderedCategorical reporting base {base!r} is a special level. The base "
                "anchors every reported relativity and must lie on the smooth; choose one "
                f"of {self._smooth_levels!r}."
            )

        # Step mode state
        self._base_level: str = ""
        self._non_base: list[str] = []
        self._R_inv: NDArray | None = None

        # Spline mode: create internal spline (deferred until we know n_levels)
        self._spline: _SplineBase | None = None
        if self.basis == "spline":
            self._init_spline()
            if self._spline_obj is not None:
                # The object API is authoritative; ignored legacy shortcuts must
                # not leak contradictory metadata into summaries or editor clones.
                self.kind = _spline_kind_name(self._spline)
                self.select = self._basis_spline.select
                self.penalty = self._basis_spline.penalty
                self.degree = self._basis_spline.degree
                self.n_knots = self._basis_spline.n_knots

    @property
    def _basis_spline(self) -> _SplineBase:
        """The inner spline, narrowed to non-None.

        ``_spline`` is only populated in spline mode, so every path that reaches
        the basis has already branched on ``self.basis == "spline"``. Routing those
        reads through here states that invariant once instead of leaving a dozen
        implicit ``None`` dereferences, and turns a violation into a named error
        rather than ``AttributeError: 'NoneType' object has no attribute 'score'``.
        """
        spline = self._spline
        if spline is None:
            raise AttributeError(
                f"OrderedCategorical(basis={self.basis!r}) has no inner spline; "
                "this attribute is only available in spline mode."
            )
        return spline

    def _build_monotone_constraints_raw(self) -> LinearConstraintSet:
        """Forward the inner spline's raw monotone geometry to the builder.

        ``build`` delegates to the inner spline, so the ``GroupInfo`` that comes
        back is stamped ``monotone_engine="qp"`` by the SPLINE. The builder then
        asks for the matching raw geometry off the FEATURE spec -- the same
        object for a plain ``Spline``, this wrapper for an ``OrderedCategorical``
        -- and without this forward it finds nothing and reports the geometry as
        unavailable. Only the QP bases (``cr``, ``bs``) were affected: ``ps``
        routes through SCOP and never takes that branch.

        The constraint is stated in the inner spline's own coefficient space,
        which is the space the returned design lives in, so it needs no
        remapping. Level scores ascend with level order by construction, so
        "increasing" means increasing across the declared levels.
        """
        return self._basis_spline._build_monotone_constraints_raw()

    def __repr__(self) -> str:
        n = self._n_levels
        if self._spline is not None:
            return f"OrderedCategorical(basis={self._spline!r}, {n} levels)"
        return f"OrderedCategorical(basis={self.basis!r}, {n} levels, n_knots={self.n_knots})"

    @property
    def has_specials(self) -> bool:
        """True when one or more levels are fitted as free effects."""
        return bool(self._specials)

    def _init_spline(self) -> None:
        """Create the internal Spline object for spline mode."""
        import copy

        if self._spline_obj is not None:
            # User passed a Spline object — deep-copy so we own it,
            # then clamp n_knots if needed.
            self._spline = copy.deepcopy(self._spline_obj)
            if self._spline.n_knots > self._n_levels - 1:
                effective = self._n_levels - 1
                warnings.warn(
                    f"OrderedCategorical: Spline n_knots={self._spline.n_knots} "
                    f"clamped to {effective} (n_levels - 1 = {self._n_levels - 1})",
                    UserWarning,
                    stacklevel=3,
                )
                self._spline.n_knots = effective
            return

        from superglm.features.spline import Spline

        effective_n_knots = min(self.n_knots, self._n_levels - 1)
        if effective_n_knots < self.n_knots:
            warnings.warn(
                f"OrderedCategorical: n_knots={self.n_knots} clamped to "
                f"{effective_n_knots} (n_levels - 1 = {self._n_levels - 1})",
                UserWarning,
                stacklevel=3,
            )
        self._spline = Spline(
            kind=self.kind,
            n_knots=effective_n_knots,
            degree=self.degree,
            penalty=self.penalty,
            select=self.select,
        )

    def _map_to_numeric(self, x: NDArray) -> NDArray:
        """Map categorical values to their numeric representations (vectorized).

        Expects x to already be mapped through grouping if applicable
        (callers build() and transform() handle that).
        """
        return pd.Series(x).map(self._level_to_value).values.astype(np.float64)

    def _choose_base(self, x: NDArray, sample_weight: NDArray | None) -> None:
        """Choose the base level for relativities.

        Specials are excluded: the base anchors every reported relativity and
        must lie on the smooth. On a real book a MISSING band is often the most
        exposed level, so ``most_exposed`` would otherwise select it.
        """
        if self._base_level and self._base_level in self._smooth_levels:
            return

        if self.base == "most_exposed" and sample_weight is not None:
            exp_by_level = {
                lev: float(sample_weight[x == lev].sum()) for lev in self._smooth_levels
            }
            self._base_level = max(exp_by_level, key=exp_by_level.get)
        elif self.base == "most_exposed" and sample_weight is None:
            self._base_level = self._smooth_levels[0]
        elif self.base == "first":
            self._base_level = self._smooth_levels[0]
        else:
            # `base=` is the user's own spelling of a level and need not match the
            # declaration's: `base="1"` against `order=[1, 2, 3]` is the same seam
            # as everywhere else. Left unreconciled it does not raise loudly in
            # every path -- the collapse path silently re-bases to another group,
            # which changes every reported relativity.
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

        With ``specials=``, returns two GroupInfos in a fixed order: the
        penalized spline block first, the unpenalized special-indicator block
        second. Downstream metadata readers select by ``subgroup_type``, but
        the order is part of the contract — ``_split_beta`` and ``transform``
        both assume it.
        """
        x = self._canonical(np.asarray(x).ravel())

        if self._grouping is not None:
            x = _grouping_labels(x)
            _validate_categorical_levels(x, self._known_levels)
            x = pd.Series(x).map(self._grouping.original_to_group).values
        else:
            _validate_categorical_levels(x, self._known_levels)

        if self.basis == "spline":
            return self._build_spline(x, sample_weight)
        else:
            return self._build_step(x, sample_weight)

    def _build_spline(
        self, x: NDArray, sample_weight: NDArray | None
    ) -> GroupInfo | list[GroupInfo]:
        """Spline mode: map to numeric, delegate to internal Spline."""
        self._choose_base(x, sample_weight)
        if not self.has_specials:
            return self._basis_spline.build(self._map_to_numeric(x))

        special_mask = self._special_mask(x)
        ordered_mask = np.asarray(~special_mask.any(axis=1), dtype=bool)
        missing = [lev for j, lev in enumerate(self._specials) if not special_mask[:, j].any()]
        if missing:
            raise ValueError(
                f"Special level(s) {missing!r} were never observed in the training data. "
                "A special with no rows has an all-zero indicator column and an "
                "unidentifiable coefficient; remove it from specials= or supply data "
                "containing it."
            )
        # Presence is not enough: it is X'WX that has to see the indicator. A
        # special observed only on zero-weight rows contributes nothing to it, so
        # its unpenalized coefficient is exactly as unidentifiable as the absent
        # case above -- the design just does not look empty.
        if sample_weight is not None:
            w = np.asarray(sample_weight, dtype=np.float64).ravel()
            unweighted = [
                lev
                for j, lev in enumerate(self._specials)
                if not float(w[special_mask[:, j]].sum()) > 0.0
            ]
            if unweighted:
                raise ValueError(
                    f"Special level(s) {unweighted!r} carry no weight in the training data. "
                    "Their rows are present but all have zero weight, so the indicator "
                    "contributes nothing to the fit and the coefficient is unidentifiable; "
                    "remove them from specials= or supply weighted rows."
                )

        # The identifiability constraint is a column sum over the rows present, so
        # the spline must be built on exactly the rows its block is nonzero on.
        # Building over all rows would break 1'(B@Z) = 0 once the special rows are
        # zeroed, and would let a fabricated coordinate reach knot placement.
        ordered_numeric = self._map_to_numeric(x[ordered_mask])
        spline_info = self._basis_spline.build(ordered_numeric)
        # build() is declared as returning one GroupInfo or a list of them; the
        # spline bases reachable from here return exactly one, and _expand_rows and
        # the two-block contract both assume it. Say so rather than indexing on faith.
        if not isinstance(spline_info, GroupInfo):
            raise TypeError(
                f"OrderedCategorical specials require a single-group spline basis; "
                f"{type(self._basis_spline).__name__} produced {len(spline_info)} groups."
            )
        spline_info = self._expand_rows(spline_info, ordered_mask)

        indicators = sp.csr_matrix(special_mask.astype(np.float64))
        special_info = GroupInfo(
            columns=indicators,
            n_cols=len(self._specials),
            penalty_matrix=None,
            reparametrize=False,
            penalized=False,
            subgroup_name="special",
            projection=None,
        )
        return [spline_info, special_info]

    def _special_mask(self, x: NDArray) -> NDArray[np.bool_]:
        """(n, n_specials) boolean membership matrix, column j == self._specials[j].

        ``self._specials`` is string-coerced at construction (so a special
        listed as ``9`` still pops out of ``order=``/``values=``), so the
        column is compared through a string view — otherwise construction and
        build disagree and the special's rows fall through to the smooth.

        The string view alone is not enough: a float column renders ``9.0`` as
        ``"9.0"``, which never equals the declared ``"9"``, so a special
        declared as a non-str is matched against its raw label as well. That
        comparison runs through pandas, which yields element-wise ``False``
        when the column's dtype cannot hold the label rather than raising.
        """
        raw = pd.Series(np.asarray(x).ravel())
        labels = raw.astype(str).to_numpy()
        columns = []
        for lev, raw_lev in zip(self._specials, self._special_raw):
            hit = labels == lev
            if not isinstance(raw_lev, str):
                hit = hit | np.asarray(raw == raw_lev, dtype=bool)
            columns.append(hit)
        return np.column_stack(columns)

    @property
    def _n_special_cols(self) -> int:
        return len(self._specials)

    def _split_beta(
        self, beta: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Split a full-width feature coefficient vector into its two blocks.

        Callers throughout inference concatenate every GroupSlice of a feature
        and hand the result here; the block order is the documented build()
        contract — spline first, specials second.

        The width check is an equality, not a lower bound: the realistic bad
        input is the spline block alone, which is *longer* than ``n_special``
        and would otherwise be split silently, reinterpreting its last
        coefficients as free level effects. ``_spline_n_cols()`` tracks the
        inner spline's current width in both the pre-reparametrisation and the
        post-fit state, so equality is well-defined at every call site.
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

        n = len(ordered_mask)
        compact = info.columns
        expanded = sp.lil_matrix((n, compact.shape[1]), dtype=np.float64)
        # Row i of the compact basis must land on the i-th ordered row: every
        # other coefficient is fitted against these rows, so a permuted scatter
        # would fit each row against another row's basis.
        expanded[np.flatnonzero(ordered_mask)] = compact
        return dataclasses.replace(info, columns=expanded.tocsr())

    def _build_step(self, x: NDArray, sample_weight: NDArray | None) -> GroupInfo:
        """Step mode: one-hot with first-difference penalty."""
        self._choose_base(x, sample_weight)
        n = len(x)
        K = self._n_levels
        n_cols = len(self._non_base)  # K - 1

        # One-hot encode (excluding base) — sparse CSR
        rows = []
        cols = []
        for j, lev in enumerate(self._non_base):
            mask = np.where(x == lev)[0]
            rows.append(mask)
            cols.append(np.full(len(mask), j))
        rows_arr = np.concatenate(rows)
        cols_arr = np.concatenate(cols)
        data = np.ones(len(rows_arr), dtype=np.float64)
        columns = sp.csr_matrix((data, (rows_arr, cols_arr)), shape=(n, n_cols))

        # K=2 edge case: D1 is empty, fall back to unpenalized
        if n_cols <= 1:
            return GroupInfo(columns=columns, n_cols=n_cols)

        # First-difference penalty on the FULL K-level ordering, then project
        # to the (K-1)-dimensional non-base space via base-removal matrix Z.
        # This ensures the penalty respects the original adjacency even when
        # the base level is in the middle of the ordering.
        #
        # The projected penalty Z'D1'D1Z is full rank (K-1) — intentionally.
        # In the treatment-contrast parameterisation (base=0), every direction
        # is penalized including the absolute level of non-base categories
        # relative to base.  This is correct: the constraint beta_base=0
        # breaks the constant null space that a naive (K-2)-rank D1 would have.
        base_idx = self._ordered_levels.index(self._base_level)
        D1_full = np.diff(np.eye(K), n=1, axis=0)  # (K-1, K)
        # Z: (K, K-1) inserts a zero row at base_idx position
        Z = np.zeros((K, n_cols))
        j = 0
        for i in range(K):
            if i != base_idx:
                Z[i, j] = 1.0
                j += 1
        omega = Z.T @ D1_full.T @ D1_full @ Z  # (K-1, K-1)

        return GroupInfo(
            columns=columns,
            n_cols=n_cols,
            penalty_matrix=omega,
            reparametrize=True,
            penalized=True,
        )

    # ── Transform ──────────────────────────────────────────────────

    def transform(self, x: NDArray) -> NDArray:
        """Build design matrix for new data using learned parameters."""
        x = self._canonical(np.asarray(x).ravel())
        if self._grouping is not None:
            x = _grouping_labels(x)
            valid_levels = self._known_levels | set(self._grouping.grouped_levels)
            _validate_categorical_levels(x, valid_levels)
            x = np.array(
                [self._grouping.original_to_group.get(v, v) for v in x],
                dtype=object,
            )
        else:
            _validate_categorical_levels(x, self._known_levels)

        if self.basis == "spline":
            if not self.has_specials:
                return self._basis_spline.transform(self._map_to_numeric(x))
            special_mask = self._special_mask(x)
            ordered_mask = ~special_mask.any(axis=1)
            spline_cols = np.zeros((len(x), self._spline_n_cols()), dtype=np.float64)
            if ordered_mask.any():
                spline_cols[ordered_mask] = self._basis_spline.transform(
                    self._map_to_numeric(x[ordered_mask])
                )
            return np.column_stack([spline_cols, special_mask.astype(np.float64)])
        else:
            # Step mode: one-hot then apply R_inv
            onehot = np.column_stack([(x == lev).astype(np.float64) for lev in self._non_base])
            if self._R_inv is not None:
                return onehot @ self._R_inv
            return onehot

    def score(self, x: NDArray, beta: NDArray[np.floating]) -> NDArray[np.floating]:
        """Score the fitted ordered-categorical contribution directly on new data."""
        x = self._canonical(np.asarray(x).ravel())
        if self._grouping is not None:
            x = _grouping_labels(x)
            valid_levels = self._known_levels | set(self._grouping.grouped_levels)
            _validate_categorical_levels(x, valid_levels)
            x = np.array(
                [self._grouping.original_to_group.get(v, v) for v in x],
                dtype=object,
            )
        else:
            _validate_categorical_levels(x, self._known_levels)

        if self.basis == "spline":
            spline_beta, special_beta = self._split_beta(beta)
            if not self.has_specials:
                return self._basis_spline.score(self._map_to_numeric(x), spline_beta)
            special_mask = self._special_mask(x)
            ordered_mask = ~special_mask.any(axis=1)
            out = special_mask.astype(np.float64) @ special_beta
            if ordered_mask.any():
                out[ordered_mask] = self._basis_spline.score(
                    self._map_to_numeric(x[ordered_mask]), spline_beta
                )
            return out

        beta_orig = self._R_inv @ beta if self._R_inv is not None else beta
        level_scores = {self._base_level: 0.0}
        for i, lev in enumerate(self._non_base):
            level_scores[lev] = float(beta_orig[i])
        return np.array([level_scores[lev] for lev in x], dtype=np.float64)

    # ── Reconstruct ────────────────────────────────────────────────

    def _base_log_effect(self, beta: NDArray[np.floating]) -> float:
        """Return the fitted term effect at the reporting reference level."""
        if self.basis != "spline":
            return 0.0
        spline_beta, _ = self._split_beta(beta)
        base_value = np.array([self._level_to_value[self._base_level]], dtype=np.float64)
        return float(self._basis_spline.score(base_value, spline_beta)[0])

    def reconstruct(self, beta: NDArray[np.floating]) -> dict[str, Any]:
        """Convert fitted coefficients to interpretable output."""
        if self.basis == "spline":
            return self._reconstruct_spline(beta)
        else:
            return self._reconstruct_step(beta)

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

        # Report specials under their domain labels so every entry in `levels` is
        # in the same namespace as the smooth levels and as the raw column. Read
        # `_ordered_levels` rather than re-deriving the concatenation: the two
        # lists are joined on equality downstream (canonical row names come from
        # one, coefficient row names from the other), and re-deriving is how they
        # came to be spelled differently in the first place.
        all_levels = list(self._ordered_levels)
        all_log_rels = np.concatenate(
            [level_log_rels, np.asarray(special_beta, dtype=np.float64) - base_shift]
        )

        raw["base_level"] = self._base_level
        raw["levels"] = all_levels
        raw["special_levels"] = list(self._special_display)
        # Keyed on the smooth levels only — a special never receives a coordinate
        # on the spline's axis.
        raw["level_values"] = dict(zip(self._smooth_levels, level_values.tolist()))
        raw["level_log_relativities"] = dict(zip(all_levels, all_log_rels.tolist()))
        raw["level_relativities"] = dict(zip(all_levels, np.exp(all_log_rels).tolist()))
        return raw

    def _reconstruct_step(self, beta: NDArray) -> dict[str, Any]:
        """Step mode: same format as Categorical."""
        # Undo reparametrization
        if self._R_inv is not None:
            beta_orig = self._R_inv @ beta
        else:
            beta_orig = beta

        relativities = {self._base_level: 1.0}
        log_rels = {self._base_level: 0.0}
        for i, lev in enumerate(self._non_base):
            log_rels[lev] = float(beta_orig[i])
            relativities[lev] = float(np.exp(beta_orig[i]))
        return {
            "base_level": self._base_level,
            "levels": self._ordered_levels,
            "log_relativities": log_rels,
            "relativities": relativities,
        }

    # ── Reparametrisation ──────────────────────────────────────────

    def set_reparametrisation(self, R_inv: NDArray) -> None:
        if self.basis == "spline":
            self._basis_spline.set_reparametrisation(R_inv)
        else:
            self._R_inv = R_inv


def resolve_interaction_parent(spec: Any, x: NDArray) -> tuple[Any, NDArray]:
    """Resolve one interaction parent (spec, column) for assembly.

    Identity for every spec — including ``None``, which FactorSmooth group
    columns carry — except spline-mode OrderedCategorical, which
    contributes its inner Spline on the mapped numeric scores, applying
    the same grouping, level validation, and score mapping its own
    ``build``/``transform`` apply.  Step-mode OC cannot parent an
    interaction: the deprecated one-hot geometry has no marginal smooth.
    Neither can a term carrying ``specials=``: a special is a free level with
    no position on the spline axis, so there is no single marginal smooth to
    cross with.
    """
    if not isinstance(spec, OrderedCategorical):
        return spec, x
    if spec.basis != "spline" or spec._spline is None:
        raise NotImplementedError(
            "OrderedCategorical with basis='step' is deprecated and cannot parent "
            "an interaction; use basis=Spline(...) for a smoothed ordinal parent "
            "or a Categorical feature for unsmoothed level effects."
        )
    if spec.has_specials:
        raise NotImplementedError(
            f"OrderedCategorical with specials={spec._specials!r} cannot parent an "
            "interaction: a special is a free level with no position on the spline "
            "axis, so the term has no single marginal smooth to cross with; drop "
            "specials= to interact the smoothed ordinal parent, or use a Categorical "
            "feature for unsmoothed level effects."
        )
    x = np.asarray(x).ravel()
    if spec._grouping is not None:
        x = _grouping_labels(x)
        valid = spec._known_levels | set(spec._grouping.grouped_levels)
        _validate_categorical_levels(x, valid)
        x = np.array([spec._grouping.original_to_group.get(v, v) for v in x], dtype=object)
    else:
        _validate_categorical_levels(x, spec._known_levels)
    return spec._spline, spec._map_to_numeric(x)


def resolve_interaction_parent_of(ispec: Any, spec: Any, x: NDArray) -> tuple[Any, NDArray]:
    """``resolve_interaction_parent`` for a parent OF a given interaction.

    Identical to it for every interaction that reads both parents as
    *marginal* columns.  ``FactorSmooth`` does not: its second parent is a
    GROUPING column, read as labels and factorized into the term's own level
    set.  Mapping a spline-mode ``OrderedCategorical`` main on that column to
    its level scores would silently re-key that identity — ``_levels`` becomes
    ``[0.0, 0.25, ...]`` instead of ``['18-25', ...]`` — and every by-label
    lookup the term's inference exposes would fail on the fitted labels.  Its
    variable parent is numeric and carries no OC form either, so a
    ``FactorSmooth`` takes both columns as the frame carries them.
    """
    from superglm.features.factor_smooth import FactorSmooth

    if isinstance(ispec, FactorSmooth):
        return spec, x
    return resolve_interaction_parent(spec, x)
