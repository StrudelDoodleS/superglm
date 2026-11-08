"""Categorical feature: one-hot dummies as a single group.

Group lasso can zero out the entire factor (all levels shrink to base
simultaneously), which is the correct variable selection behavior.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from superglm.features.constraint import ConstraintRules, ConstraintSpec
from superglm.types import GroupInfo, LinearConstraintSet

# The named unseen policies; any other `unseen=` text names a group.
_UNSEEN_POLICIES = ("error", "base")
# The named base policies; any other `base=` names a level.
_BASE_POLICIES = ("most_exposed", "first")


def _validate_observed_categorical_levels(
    observed: set,
    known_levels: set,
    *,
    context: str = "",
) -> None:
    """Raise for observed levels outside a fitted categorical domain."""
    _refuse_unseen_categorical_levels(observed - known_levels, known_levels, context=context)


def _refuse_unseen_categorical_levels(unseen: set, known_levels: set, *, context: str = "") -> None:
    """Name rejected labels without equating them with known labels again."""
    if unseen:
        # Sort by str: a domain may mix types (numeric bands beside a labelled
        # special), and a bare sorted() would raise TypeError from the error path itself.
        msg = (
            f"Encountered unseen categorical levels at predict time: "
            f"{sorted(unseen, key=str)}. All levels must be among those seen "
            f"during fit: {sorted(known_levels, key=str)}."
        )
        if context:
            msg = f"[{context}] {msg}"
        raise ValueError(msg)


def _validate_missing_only(x: NDArray, *, context: str = "") -> None:
    """Raise for NaN/None in a categorical column, saying nothing about levels.

    Split out so `unseen="base"` can keep rejecting broken columns while it
    stops rejecting novel levels: the two checks are independent questions and
    only one of them is governed by the policy.
    """
    # Missing values are checked before the level scan: a NaN beside strings is a
    # broken column, not an unseen level.  The scan stays exact and per-element:
    # `pd.isna` is not the same question (pandas decides nullness by `v != v`, so a
    # self-equal float subclass would pass; and it is true for pd.NA, pd.NaT and
    # non-float NaNs, which are unseen LEVELS the report must name).
    import pandas as pd

    # Whether a column holds a missing value is a property of its distinct values, so the exact
    # scan reads those: a hash pass of the column keeps one of every equal value (all NaNs of
    # exact float type are one), and a column of a million rows has a few dozen levels.
    try:
        distinct = pd.unique(np.asarray(x).ravel())
    except TypeError:  # an unhashable label: scan every element, as it would have been
        distinct = x
    if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in distinct):
        msg = "Categorical column contains missing values (NaN or None)."
        if context:
            msg = f"[{context}] {msg}"
        raise ValueError(msg)


def _validate_categorical_levels(x: NDArray, known_levels: set, *, context: str = "") -> None:
    """Raise ValueError if x contains levels not seen during fit.

    Parameters
    ----------
    x : array-like
        Categorical values to validate.
    known_levels : set
        Levels seen during build() / fit().
    context : str, optional
        Feature name for error message context.
    """
    import pandas as pd

    _validate_missing_only(x, context=context)

    # Hash, don't sort: membership is all that is needed, `np.unique` costs an
    # O(n log n) interpreter-bound sort, and an object column of mixed types has no
    # order but hashes fine.
    observed = set(pd.unique(np.asarray(x).ravel()).tolist())
    _validate_observed_categorical_levels(
        observed,
        known_levels,
        context=context,
    )


def _grouping_labels(x: NDArray) -> NDArray:
    """Return string labels used by LevelGrouping, preserving missing-value checks."""
    import pandas as pd

    x = np.asarray(x).ravel()
    if np.asarray(pd.isna(x)).any():
        raise ValueError("Categorical column contains missing values (NaN or None).")
    return pd.Series(x).astype(str).to_numpy()


def _refuse_uncovered_training_levels(observed: set, grouping) -> None:
    """Raise, as a fit, for training labels that ``grouping`` does not cover."""
    uncovered = observed - set(grouping.all_original_levels)
    if uncovered:
        raise ValueError(
            f"Training data contains levels the grouping does not cover: "
            f"{sorted(uncovered, key=str)}. Covered: "
            f"{sorted(grouping.all_original_levels, key=str)}. Build the grouping from the "
            f"full column, or pass the data to Structure.apply(model, X=data), which places "
            f"them where the structure sends new levels."
        )


def _resolve_categorical_labels(
    x: NDArray,
    grouping,
    *,
    known_levels: set | None = None,
    context: str = "",
    at_fit: bool = False,
) -> NDArray:
    """Resolve one categorical parent's raw labels into its fitted level domain.

    A grouped categorical's public input is always the ORIGINAL labels (group
    labels are not an alternative domain).  Validate the raw domain, apply the
    mapping once, then (when fitted levels are supplied) certify that the collapsed
    level was present at fit.  ``at_fit`` reports a label the grouping does not
    cover as training data rather than as unseen at fit.
    """
    x = np.asarray(x).ravel()
    if grouping is None:
        if known_levels is not None:
            _validate_categorical_levels(x, known_levels, context=context)
        return x

    import pandas as pd

    labels = _grouping_labels(x)
    observed = set(pd.unique(labels).tolist())
    if at_fit:
        _refuse_uncovered_training_levels(observed, grouping)
    _validate_observed_categorical_levels(
        observed,
        set(grouping.all_original_levels),
        context=context,
    )
    mapping = grouping.original_to_group
    resolved = pd.Series(labels, copy=False).map(mapping).to_numpy()
    if known_levels is not None:
        _validate_observed_categorical_levels(
            set(pd.unique(resolved).tolist()),
            known_levels,
            context=context,
        )
    return resolved


def _codes_against(values: NDArray, levels: list) -> NDArray:
    """Index *values* into *levels*, coding anything absent as -1.

    `pd.Categorical(values, categories=levels)` states the same thing, but
    pandas has deprecated -- and will eventually reject -- values outside the
    declared categories, and values outside the universe are exactly what this
    call exists to find. A unique-Index lookup is the supported spelling; every
    call site holds a universe that is unique by construction.

    Levels that are all ``str`` are looked up in a dict, which answers as the Index does. Any
    other label (bool, number, date, None, NaN) is the Index's to compare, since it decides by
    dtype what matches what (a bool does not match 1) and a dict does not.
    """
    import pandas as pd

    values = np.asarray(values)
    position = {level: code for code, level in enumerate(levels)}
    plain = len(position) == len(levels) and all(type(level) is str for level in levels)
    if plain and values.ndim <= 1:
        try:
            if values.size <= _DICT_LOOKUP_ROWS:
                # A quote is a few rows: a dict read per value is faster than building an Index.
                return np.fromiter(
                    (position.get(value, -1) for value in values.ravel()),
                    dtype=np.intp,
                    count=values.size,
                ).reshape(values.shape)
            # A column holds few distinct labels: hash its rows once (factorize), look the distinct
            # ones up in the universe, and read each row's code back. A missing value factorizes to
            # -1, absent.
            row_codes, distinct = pd.factorize(values)
            distinct_codes = np.fromiter(
                (position.get(value, -1) for value in distinct), dtype=np.intp, count=len(distinct)
            )
            return np.append(distinct_codes, -1)[row_codes]
        except TypeError:  # an unhashable label: the Index answers, or raises as it always did
            pass
    return np.asarray(pd.Index(levels).get_indexer(values), dtype=np.intp)


#: Rows up to which a ``str`` level is looked up in a dict instead of a pandas Index built for the call.
_DICT_LOOKUP_ROWS = 64

#: Opening of the warning for rows whose level was unseen at fit.
UNSEEN_ROUTING_WARNING = "Routing rows with categorical levels unseen at fit"

#: Whether this context gives the unseen-level warning. A caller that scores a frame in row blocks
#: sets it for the blocks, because each would count only its own rows, and gives the warning once
#: from a whole-frame call. A context variable belongs to its thread and is inherited by the
#: helpers a pooled section starts, so no process-wide warning filter is touched.
_UNSEEN_ROUTING_QUIET: ContextVar[bool] = ContextVar("superglm_unseen_routing_quiet", default=False)


@contextmanager
def unseen_routing_quiet() -> Iterator[None]:
    """Within this block the calling context does not warn of levels unseen at fit."""
    token = _UNSEEN_ROUTING_QUIET.set(True)
    try:
        yield
    finally:
        _UNSEEN_ROUTING_QUIET.reset(token)


def _warn_unseen_routed(x: NDArray, codes: NDArray, *, group: Any = None) -> None:
    """Warn once per call for rows with a negative code: routed to base, or to ``group``."""
    novel_rows = codes < 0
    if not novel_rows.any() or _UNSEEN_ROUTING_QUIET.get():
        return
    novel = sorted(set(np.asarray(x)[novel_rows].tolist()), key=str)
    if group is None:
        destination = "the base level (unseen='base')"
        effect = "They contribute nothing to the linear predictor."
    else:
        destination = f"the group {group!r} (unseen={group!r})"
        effect = "They take that group's effect."
    warnings.warn(
        f"{UNSEEN_ROUTING_WARNING} to {destination}: "
        f"{novel} over {int(novel_rows.sum())} row(s). {effect}",
        UserWarning,
        stacklevel=3,
    )


def _route_unseen_to_group(x: NDArray, grouping, universe: list, group: Any) -> NDArray:
    """Resolve raw labels through ``grouping``, sending each level unseen at fit to ``group``.

    A raw label the grouping does not know, or whose group has no place in the
    fitted ``universe``, is unseen: its rows take ``group``'s label and effect, with
    one warning naming those levels and their row count.  This is scikit-learn's
    ``OneHotEncoder`` rule with ``handle_unknown="warn"`` (``"infrequent_if_exist"``
    plus the warning); here the user names the group and the fit has checked it
    exists.  ``group=None`` is ``unseen="base"`` under a grouping: unseen rows keep
    a label no fitted level matches, code as ``-1`` and take the base.
    """
    import pandas as pd

    raw = np.asarray(x).ravel()
    resolved = (
        pd.Series(_grouping_labels(raw), copy=False)
        .map(grouping.original_to_group)
        .to_numpy(dtype=object)
    )
    # A label the grouping lacks maps to NaN, which no fitted level matches.
    codes = _codes_against(resolved, universe)
    if (codes < 0).any():
        _warn_unseen_routed(raw, codes, group=group)
        if group is not None:
            resolved[codes < 0] = group
    return resolved


def _categorical_rules(constraint: ConstraintSpec | ConstraintRules | None) -> ConstraintRules:
    """The rules a ``Categorical`` stores for ``constraint``: a cap, or none."""
    if constraint is None:
        return ConstraintRules(())
    rules = ConstraintRules((constraint,)) if isinstance(constraint, ConstraintSpec) else constraint
    if not isinstance(rules, ConstraintRules):
        raise TypeError(
            f"Categorical constraint must be None or a cap such as Constraint.fit.cap(0.7, 1.5), "
            f"got {constraint!r}"
        )
    shape = next((rule for rule in rules if rule.kind != "cap"), None)
    if shape is not None:
        raise ValueError(
            f"Categorical takes only a cap, but got {shape.describe()}, a shape rule that "
            f"needs ordered levels; use OrderedCategorical with this rule, or drop it"
        )
    rules.validate()
    return rules


class Categorical:
    """One-hot encoded categorical feature.

    Parameters
    ----------
    base : str
        How to choose the reference level.

        - ``'most_exposed'`` - level with highest total sample_weight (default,
          best for insurance)
        - ``'first'`` - first level in the level universe (alphabetical when
          inferred, as declared when ``levels=`` or a categorical dtype bounds it)

        Or pass a specific level name as a string.
    grouping : LevelGrouping, optional
        Collapse original levels into groups before fitting.
    levels : list | tuple | Series | ndarray | CategoricalDtype, optional
        The level universe to bind to.  With a ``grouping`` this declares the RAW,
        pre-collapse universe, which the grouping must cover exactly.  Levels with
        no training rows are pinned to base rather than dropped; training rows
        outside the universe are an error.
    unseen : str
        Predict-time policy for levels outside the universe.  ``'error'`` (default)
        raises; ``'base'`` routes those rows, raw labels a ``grouping`` does not
        know included, to the base level with one warning per call.  Any other value
        names a group of ``grouping`` whose effect those levels then take, again
        with one warning per call; the fit checks that the group exists and has a
        place in the fitted universe.
    constraint : ConstraintSpec | ConstraintRules, optional
        A relativity cap, ``Constraint.fit.cap(lo, hi)``: every level's printed
        relativity against the base lies in ``[lo, hi]`` (needs a log link).  A
        shape rule needs ordered levels; use ``OrderedCategorical`` for one.
    """

    # Class-level defaults for the level-universe state.  Unpickling bypasses
    # `__init__`, so a spec pickled before this state existed needs these names to
    # exist; a class attribute does that without a `__setstate__` to keep in step.
    # The values are the pre-feature behaviour: no declared universe, an inferred
    # source, nothing pinned, `unseen="error"`.  IMMUTABLE defaults only, since a
    # class attribute is shared by every instance that never assigns its own.
    unseen: str = "error"
    _declared_levels: list | None = None
    _level_source: str = "inferred"
    _pinned_levels: tuple | list = ()
    _base_fallback: tuple | None = None
    _pinned_base: Any | None = None
    # Whether `base=` names a level although it reads "first" or "most_exposed".
    # Only a rebuild sets it, for a reference a fit resolved or a user chose.
    _base_is_level: bool = False
    # Levels with exposure but only boundary responses at fit (the build's
    # separation scan); ``predict`` names them when it meets them.
    _separated_levels_at_fit: tuple = ()
    # The fit-time cap, if any; a spec pickled before caps existed has none.
    _constraint_rules: ConstraintRules = ConstraintRules(())

    def __init__(
        self,
        base: Literal["most_exposed", "first"] | str = "most_exposed",
        grouping=None,
        *,
        levels=None,
        unseen: Literal["error", "base"] | str = "error",
        constraint: ConstraintSpec | ConstraintRules | None = None,
    ):
        from superglm.features._level_source import resolve_level_source

        self._constraint_rules = _categorical_rules(constraint)

        # A group label can only be checked against the grouping at fit
        # (`_require_unseen_group`): an editor rebuild may hand over a grouping
        # that no longer holds the group, and that is a refit's refusal.
        if not isinstance(unseen, str):
            raise ValueError(
                f"unseen must be 'error', 'base' or the label of a group of grouping=, "
                f"got {unseen!r}"
            )
        self.base = base
        self.unseen = unseen
        self._grouping = grouping
        self._declared_levels: list | None = (
            None if levels is None else resolve_level_source(levels, context="Categorical")
        )
        self._level_source: str = "declared" if levels is not None else "inferred"
        if self._declared_levels is not None and grouping is not None:
            # LevelGrouping keys are stringified raw labels (`_grouping_labels`),
            # so the coverage test has to be asked in that same domain.
            uncovered = [
                lev for lev in self._declared_levels if str(lev) not in grouping.original_to_group
            ]
            if uncovered:
                raise ValueError(
                    f"levels= contains labels not covered by the grouping: "
                    f"{sorted(uncovered, key=str)}. Build the grouping from the "
                    f"full column so every declared level maps somewhere."
                )
            # And the other way: a label the grouping maps but levels= leaves
            # out would be fitted and scored through its group, though the
            # declaration says it is not a level.
            declared = {str(lev) for lev in self._declared_levels}
            undeclared = [raw for raw in grouping.original_to_group if raw not in declared]
            if undeclared:
                raise ValueError(
                    f"grouping= covers labels that levels= leaves out: "
                    f"{sorted(undeclared, key=str)}. Add them to levels=, or build the "
                    f"grouping from the declared levels only."
                )
        self._levels: list = []
        self._base_level: str = ""
        self._non_base: list = []
        self._pinned_levels: list = []
        self._base_fallback: tuple | None = None
        self._pinned_base: Any | None = None
        self._base_is_level: bool = False

    @property
    def constraint_rules(self) -> ConstraintRules:
        """The fit-time rules this term carries: a cap, or none."""
        return self._constraint_rules

    @property
    def constraint_kind(self) -> str | None:
        """The rules' description, or None without a cap."""
        return self._constraint_rules.describe() if self._constraint_rules else None

    @property
    def constraint_mode(self) -> str:
        """``"fit"`` with a cap; ``"postfit"``, the unconstrained default, without."""
        return "fit" if self._constraint_rules else "postfit"

    def _build_monotone_constraints_raw(self) -> LinearConstraintSet:
        """Rows ``A @ beta >= b`` holding the cap on every emitted level.

        Each emitted column's coefficient is that level's log relativity
        against the base, as printed (``reconstruct``), so a cap is the two
        rows ``log lo <= beta_j <= log hi`` per column. The base and pinned
        levels print at 1, which every cap contains.
        """
        n = len(self._non_base)
        cap = next(rule for rule in self._constraint_rules if rule.kind == "cap")
        lo, hi = cast(tuple[float, float], cap.bounds)
        eye = np.eye(n, dtype=np.float64)
        return LinearConstraintSet(
            A=np.vstack([eye, -eye]),
            b=np.concatenate([np.full(n, np.log(lo)), np.full(n, -np.log(hi))]),
        )

    def __repr__(self) -> str:
        n = len(self._levels)
        if n:
            return f"Categorical(base={self.base!r}, {n} levels, ref={self._base_level!r})"
        return f"Categorical(base={self.base!r})"

    def adopt_dtype_categories(self, categories: list) -> None:
        """Adopt a dtype-declared universe unless one is already declared."""
        if self._declared_levels is None:
            from superglm.features._level_source import resolve_level_source

            self._declared_levels = resolve_level_source(
                list(categories), context="Categorical dtype categories"
            )
            self._level_source = "dtype"

    def apply_level_binding(self, binding) -> None:
        """Adopt a full-frame binding: universe if unset, base pin if unpinned."""
        if self._declared_levels is None and binding.levels is not None:
            self._declared_levels = list(binding.levels)
            self._level_source = "full-frame"
        if binding.base is not None and self.base == "most_exposed" and not self._base_is_level:
            self._pinned_base = binding.base

    def resolve_binding(self, values: NDArray, sample_weight=None):
        """Compute this spec's full-frame binding without mutating the spec."""
        import copy

        from superglm.types import LevelBinding

        # Build on a throwaway copy so grouping, ordering, exposure and the
        # NaN checks stay single-sourced in `build` -- a second inference path
        # here is a second thing to keep in step.
        probe = copy.deepcopy(self)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            probe.build(values, sample_weight=sample_weight)
        if self._grouping is None:
            return LevelBinding(levels=tuple(probe._levels), base=probe._base_level)

        # Under a grouping the binding must carry the RAW, pre-collapse universe:
        # `apply_level_binding` stores it as `_declared_levels` and `_working_universe`
        # maps each one through `grouping.original_to_group`.  The raw labels are read in
        # the grouping path's str-cast domain; `build` above has already rejected missing
        # values and unmappable raws.  The base stays the probe's own fitted level (a
        # GROUP).  Ordered by (group, raw) so the image comes out in sorted GROUP order,
        # as the unbound path's `pd.factorize(sort=True)` gives; sorting raws alone would
        # silently permute the design columns and move `base="first"`.
        to_group = self._grouping.original_to_group
        raws = sorted(set(_grouping_labels(values).tolist()), key=lambda raw: (to_group[raw], raw))
        return LevelBinding(levels=tuple(raws), base=probe._base_level)

    def _working_universe(self) -> list | None:
        """The bound universe in build coordinates, or None when unbound.

        `levels=` declares RAW labels; under a grouping the design speaks group
        labels, so the declared raws are mapped through and de-duplicated in
        first-occurrence order.
        """
        if self._declared_levels is None:
            return None
        if self._grouping is None:
            return list(self._declared_levels)
        return list(
            dict.fromkeys(
                self._grouping.original_to_group[str(lev)] for lev in self._declared_levels
            )
        )

    def build(
        self,
        x: NDArray,
        sample_weight: NDArray[np.floating] | None = None,
    ) -> GroupInfo:
        """Build sparse one-hot design columns, choosing the base level from *x*."""
        return self._build_with_codes(x, sample_weight, stacklevel=3)[0]

    def _build_with_codes(
        self,
        x: NDArray,
        sample_weight: NDArray[np.floating] | None = None,
        *,
        stacklevel: int = 2,
    ) -> tuple[GroupInfo, NDArray]:
        """``build``, and every row's code against the fitted level universe.

        The codes index ``_levels`` (base and pinned levels included) and are never
        negative: a row outside the universe is refused above.  A caller that scans the
        same rows (the separation scan) reads them instead of coding the column again.
        ``stacklevel`` is the frame, counted from this method, that the build's warnings
        are attributed to.
        """
        import pandas as pd

        x = _resolve_categorical_labels(x, self._grouping, at_fit=True)

        universe = self._working_universe()
        if universe is None:
            # Single-pass O(n) factorize — avoids O(n log n) sort + O(n * levels) loop
            codes, uniques = pd.factorize(x, sort=True)

            # pd.factorize encodes NaN/None as -1 in codes. Reject them here
            # so they don't silently corrupt the design matrix.
            if (codes == -1).any():
                raise ValueError("Categorical column contains missing values (NaN or None).")

            self._levels = uniques.tolist()
        else:
            codes = _codes_against(x, universe)
            if (codes == -1).any():
                # A -1 under a bound universe is either a broken column or data
                # the declaration does not admit; those are different bugs and
                # get different messages.
                outside = np.asarray(x)[codes == -1]
                _validate_missing_only(outside)
                raise ValueError(
                    f"Training data contains levels outside the declared level "
                    f"universe: {sorted(set(outside.tolist()), key=str)}. Declared: "
                    f"{sorted(universe, key=str)}. Widen levels= or fix the column."
                )
            self._levels = list(universe)

        if len(self._levels) < 2:
            raise ValueError(f"Categorical needs >= 2 levels, got {len(self._levels)}")
        self._require_unseen_group()

        # Effective observation: rows when unweighted, total weight otherwise.
        # A level carrying only zero-weight rows contributes nothing to the fit
        # and is treated exactly like a level with no rows at all.
        if sample_weight is not None:
            # O(n) bincount instead of O(n * levels) dict comprehension
            effective = np.bincount(codes, weights=sample_weight, minlength=len(self._levels))
        else:
            effective = np.bincount(codes, minlength=len(self._levels)).astype(np.float64)
        observed = effective > 0.0

        self._base_level = self._resolve_base(
            effective, observed, sample_weight, stacklevel=stacklevel + 1
        )
        self._pinned_levels = [
            lev for i, lev in enumerate(self._levels) if not observed[i] and lev != self._base_level
        ]
        if self._pinned_levels:
            warnings.warn(
                f"Categorical levels with no effective training rows in this fit are "
                f"pinned to base (zero contribution): "
                f"{sorted(self._pinned_levels, key=str)}. They remain known levels "
                f"and predict as the base level.",
                UserWarning,
                stacklevel=stacklevel,
            )

        # Remap codes: drop the base level and every pinned level, giving 0-based codes
        # for the levels with a design column.  Base-level rows are absorbed into the
        # intercept and encoded -1, the sink bin the categorical kernel reads as zero.
        emitted = [
            i for i, lev in enumerate(self._levels) if lev != self._base_level and observed[i]
        ]
        self._non_base = [self._levels[i] for i in emitted]
        if not self._non_base:
            raise ValueError(
                f"Categorical has no estimable non-base level in this fit: every "
                f"level except the base {self._base_level!r} has no effective "
                f"training rows (universe {sorted(self._levels, key=str)}). The "
                f"term cannot be identified from this data; widen the training "
                f"slice or drop the term."
            )
        n_levels = len(self._non_base)
        remap = np.full(len(self._levels), -1, dtype=np.intp)
        remap[emitted] = np.arange(n_levels, dtype=np.intp)
        cat_codes = remap[codes]

        if not self._constraint_rules:
            return GroupInfo(columns=None, n_cols=n_levels, cat_codes=cat_codes), codes
        info = GroupInfo(
            columns=None,
            n_cols=n_levels,
            cat_codes=cat_codes,
            constraints=self._build_monotone_constraints_raw(),
            monotone_engine="qp",
        )
        return info, codes

    def _require_unseen_group(self) -> None:
        """Refuse an ``unseen`` group this fit cannot route new levels to."""
        if self.unseen in _UNSEEN_POLICIES:
            return
        if self._grouping is None:
            raise ValueError(
                f"unseen={self.unseen!r} names a group, but this Categorical has no "
                f"grouping=. Group its levels with grouping=, or use unseen='error' or "
                f"unseen='base'."
            )
        if self.unseen not in self._grouping.grouped_levels:
            raise ValueError(
                f"unseen={self.unseen!r} is not a group of this Categorical's grouping, "
                f"whose groups are {list(self._grouping.grouped_levels)}. Name one of them, "
                f"or use unseen='error' or unseen='base'."
            )
        if self.unseen not in self._levels:
            raise ValueError(
                f"unseen={self.unseen!r} names a group with no level in this fit's universe "
                f"{self._levels}, so new levels would have no effect to take. Declare its "
                f"levels with levels=, or route new levels to another group."
            )

    def _resolve_base(
        self,
        effective: NDArray[np.floating],
        observed: NDArray[np.bool_],
        sample_weight: NDArray[np.floating] | None,
        stacklevel: int = 3,
    ) -> Any:
        """Pick the reference level, falling back when the request is empty.

        A binding's pin and the level a prior build resolved name a level, even
        one named "first" or "most_exposed"; only ``base=`` names a policy, and
        not when a rebuild marked it as a level (``_base_is_level``).
        """
        self._base_fallback = None
        policy = None
        if self._pinned_base is not None:
            requested = self._pinned_base
        elif self._base_level and self._base_level in self._levels:
            # Reuse from a prior build on this spec: refitting one spec object
            # must not silently move the coefficient identity.
            requested = self._base_level
        else:
            requested = self.base
            if not self._base_is_level and requested in _BASE_POLICIES:
                policy = requested

        if policy == "most_exposed":
            base_level = self._most_exposed(effective, observed, sample_weight)
        elif policy == "first":
            # Universe order, so a declared universe means first-DECLARED; the
            # inferred universe is sorted, so it still means alphabetical there.
            base_level = self._levels[0]
        elif requested in self._levels:
            base_level = requested
        else:
            raise ValueError(f"Base '{requested}' not found in levels: {self._levels}")

        if not observed[self._levels.index(base_level)]:
            fallback = self._most_exposed(effective, observed, sample_weight)
            if fallback != base_level:
                warnings.warn(
                    f"Categorical base level '{base_level}' has no effective training "
                    f"rows in this fit; falling back to '{fallback}'. Coefficient "
                    f"identity changes; predictions do not.",
                    UserWarning,
                    stacklevel=stacklevel,
                )
                self._base_fallback = (base_level, fallback)
                base_level = fallback
        return base_level

    def _most_exposed(
        self,
        effective: NDArray[np.floating],
        observed: NDArray[np.bool_],
        sample_weight: NDArray[np.floating] | None,
    ) -> Any:
        """Most-exposed observed level, demoting to first-observed unweighted."""
        if sample_weight is not None:
            return self._levels[int(np.argmax(effective))]
        return next(
            (lev for i, lev in enumerate(self._levels) if observed[i]),
            self._levels[0],
        )

    def _resolve_predict_labels(self, x: NDArray) -> NDArray:
        """Resolve predict-time labels under the term's unseen policy."""
        if self.unseen not in _UNSEEN_POLICIES:
            return _route_unseen_to_group(x, self._grouping, self._levels, self.unseen)
        if self.unseen == "base" and self._grouping is not None:
            # A raw label the grouping does not know is unseen too.
            return _route_unseen_to_group(x, self._grouping, self._levels, None)
        x = _resolve_categorical_labels(
            x,
            self._grouping,
            known_levels=set(self._levels) if self.unseen == "error" else None,
        )
        if self.unseen != "error":
            # The policy governs unknown LEVELS only. A NaN is still a broken
            # column and still says so.
            _validate_missing_only(x)
        return x

    def _predicted_levels(self, x: NDArray) -> set:
        """The fitted levels the rows of raw ``x`` are scored at.

        Resolved as ``score`` resolves them (grouping, then the unseen
        policy), so a collapsed level is met by its members' raw values.
        ``score`` has already refused or reported unseen labels, so nothing is
        said again here.
        """
        import pandas as pd

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            resolved = self._resolve_predict_labels(x)
        return set(pd.unique(pd.Series(resolved, dtype=object)))

    def _predict_codes(self, x: NDArray) -> NDArray:
        """Codes against the fitted universe; -1 for levels the policy admits."""
        codes = _codes_against(x, self._levels)
        if self.unseen == "error" and np.any(codes < 0):
            _refuse_unseen_categorical_levels(set(x[codes < 0].tolist()), set(self._levels))
        if self.unseen == "base" and self._grouping is None:
            # A group policy, or base under a grouping, has already routed and
            # reported its rows by their raw labels.
            _warn_unseen_routed(x, codes)
        return codes

    def transform(self, x: NDArray) -> NDArray:
        """One-hot encode using levels learned during build()."""
        x = self._resolve_predict_labels(x)
        codes = self._predict_codes(x)
        return np.column_stack(
            [(codes == self._levels.index(lev)).astype(np.float64) for lev in self._non_base]
        )

    def score(self, x: NDArray, beta: NDArray[np.floating]) -> NDArray[np.floating]:
        """Score the fitted categorical contribution directly on new data."""
        x = self._resolve_predict_labels(x)
        codes = self._predict_codes(x)

        # One slot past the universe holds zero, and every negative code is sent
        # there: a bare `level_effects[codes]` would wrap -1 onto the LAST
        # level's effect, which is a wrong prediction rather than a base one.
        level_effects = np.zeros(len(self._levels) + 1, dtype=np.float64)
        for i, lev in enumerate(self._non_base):
            level_effects[self._levels.index(lev)] = float(beta[i])
        return level_effects[np.where(codes >= 0, codes, len(self._levels))]

    def reconstruct(self, beta: NDArray[np.floating]) -> dict[str, Any]:
        """Coefficients -> relativity table."""
        relativities = {self._base_level: 1.0}
        log_rels = {self._base_level: 0.0}
        for i, lev in enumerate(self._non_base):
            log_rels[lev] = float(beta[i])
            relativities[lev] = float(np.exp(beta[i]))
        # A pinned level has no coefficient because it had no data, not because
        # it is unknown: it reports at base, like the base itself.
        for lev in self._pinned_levels:
            log_rels[lev] = 0.0
            relativities[lev] = 1.0
        return {
            "base_level": self._base_level,
            "levels": self._levels,
            "log_relativities": log_rels,
            "relativities": relativities,
            "pinned_levels": self._pinned_levels,
            "level_source": self._level_source,
            "base_fallback": self._base_fallback,
        }
