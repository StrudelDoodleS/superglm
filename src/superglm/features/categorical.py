"""Categorical feature: one-hot dummies as a single group.

Group lasso can zero out the entire factor (all levels shrink to base
simultaneously), which is the correct variable selection behavior.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm.types import GroupInfo


def _validate_observed_categorical_levels(
    observed: set,
    known_levels: set,
    *,
    context: str = "",
) -> None:
    """Raise for observed levels outside a fitted categorical domain."""
    unseen = observed - known_levels
    if unseen:
        # Sort by str: a domain may legitimately mix types (numeric bands beside a
        # labelled special, `order=[1..6], specials=["MISSING"]`), and a bare
        # sorted() raises TypeError comparing int to str -- turning a clean report
        # of a genuinely unseen level into a crash from the error path itself.
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
    # Missing values are checked before the level scan: a NaN beside strings is
    # a broken column, not an unseen level, and it should say so.
    #
    # This scan stays exact and per-element on purpose. `pd.isna` is a C-level
    # test and reads as the obvious pre-filter, but it is not a superset of the
    # question asked here: pandas decides nullness by asking `v != v`, so a
    # float subclass that compares equal to itself is a NaN this check owns and
    # `pd.isna` calls clean. Short-circuiting on it therefore turns a rejected
    # column into an accepted one whenever that value is also a fitted level --
    # a validation surface quietly widening, which is not a trade worth 0.08s
    # per search. The narrow test below is the boundary, so the narrow test is
    # what runs.
    #
    # `pd.isna` is wrong as the answer too, for the opposite reason: it is true
    # for a wider family this check does NOT own (pd.NA, pd.NaT, a NaN that is
    # not a Python float). Those are unseen LEVELS and the report has to name
    # them rather than blame the column.
    if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in x):
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

    # Hash, don't sort. Membership is all this function needs, and `np.unique`
    # buys it with an O(n log n) sort that on an object column runs every
    # comparison through the interpreter. Hashing also removes the reason the
    # old sorting path needed a fallback: an object column holding more than one
    # type has no order (`["MISSING", 1]` raised TypeError comparing str to int)
    # but it hashes like any other.
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


def _resolve_categorical_labels(
    x: NDArray,
    grouping,
    *,
    known_levels: set | None = None,
    context: str = "",
) -> NDArray:
    """Resolve one categorical parent's raw labels into its fitted level domain.

    A grouped categorical's public input contract is always the ORIGINAL
    labels.  Group labels are not accepted as an alternative input domain:
    they may themselves be original labels with a different mapping.  Validate
    the raw domain first, apply the mapping exactly once, then (when fitted
    levels are supplied) certify that the collapsed level was present at fit.
    """
    x = np.asarray(x).ravel()
    if grouping is None:
        if known_levels is not None:
            _validate_categorical_levels(x, known_levels, context=context)
        return x

    import pandas as pd

    labels = _grouping_labels(x)
    _validate_observed_categorical_levels(
        set(pd.unique(labels).tolist()),
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
    """
    import pandas as pd

    return np.asarray(pd.Index(levels).get_indexer(np.asarray(values)), dtype=np.intp)


def _warn_unseen_routed(x: NDArray, codes: NDArray) -> None:
    """Warn once per call for rows routed to base by `unseen="base"`."""
    novel_rows = codes < 0
    if not novel_rows.any():
        return
    novel = sorted(set(np.asarray(x)[novel_rows].tolist()), key=str)
    warnings.warn(
        f"Routing rows with categorical levels unseen at fit to the base level "
        f"(unseen='base'): {novel} over {int(novel_rows.sum())} row(s). They "
        f"contribute nothing to the linear predictor.",
        UserWarning,
        stacklevel=3,
    )


class Categorical:
    """One-hot encoded categorical feature.

    Parameters
    ----------
    base : str
        How to choose the reference level.
        'most_exposed' - level with highest total sample_weight (default, best for insurance)
        'first'        - first level in the level universe (alphabetical when inferred,
                         as declared when ``levels=`` or a categorical dtype bounds it)
        Or pass a specific level name as a string.
    grouping : LevelGrouping, optional
        Collapse original levels into groups before fitting.
    levels : list | tuple | Series | ndarray | CategoricalDtype, optional
        The level universe to bind to (spec 2026-08-11, §3.1).  With a
        ``grouping`` this declares the RAW, pre-collapse universe.  Levels with
        no training rows are pinned to base rather than dropped; training rows
        outside the universe are an error.
    unseen : {'error', 'base'}
        Predict-time policy for levels outside the universe.  'error' (default)
        is the historical behavior; 'base' routes those rows to the base level
        with one warning per call.
    """

    def __init__(
        self,
        base: str = "most_exposed",
        grouping=None,
        *,
        levels=None,
        unseen: str = "error",
    ):
        from superglm.features._level_source import resolve_level_source

        if unseen not in ("error", "base"):
            raise ValueError(f"unseen must be 'error' or 'base', got {unseen!r}")
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
        self._levels: list = []
        self._base_level: str = ""
        self._non_base: list = []
        self._pinned_levels: list = []
        self._base_fallback: tuple | None = None
        self._pinned_base: Any | None = None

    def __repr__(self) -> str:
        n = len(self._levels)
        if n:
            return f"Categorical(base={self.base!r}, {n} levels, ref={self._base_level!r})"
        return f"Categorical(base={self.base!r})"

    def adopt_dtype_categories(self, categories: list) -> None:
        """Adopt a dtype-declared universe unless one is already declared."""
        if self._declared_levels is None:
            from superglm.features._level_source import resolve_level_source

            self._declared_levels = resolve_level_source(list(categories), context="Categorical")
            self._level_source = "dtype"

    def apply_level_binding(self, binding) -> None:
        """Adopt a full-frame binding: universe if unset, base pin if unpinned."""
        if self._declared_levels is None and binding.levels is not None:
            self._declared_levels = list(binding.levels)
            self._level_source = "full-frame"
        if binding.base is not None and self.base == "most_exposed":
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
        return LevelBinding(levels=tuple(probe._levels), base=probe._base_level)

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
        import pandas as pd

        x = _resolve_categorical_labels(x, self._grouping)

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

        # Effective observation: rows when unweighted, total weight otherwise.
        # A level carrying only zero-weight rows contributes nothing to the fit
        # and is treated exactly like a level with no rows at all.
        if sample_weight is not None:
            # O(n) bincount instead of O(n * levels) dict comprehension
            effective = np.bincount(codes, weights=sample_weight, minlength=len(self._levels))
        else:
            effective = np.bincount(codes, minlength=len(self._levels)).astype(np.float64)
        observed = effective > 0.0

        self._base_level = self._resolve_base(effective, observed, sample_weight)
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
                stacklevel=2,
            )

        # Remap codes: drop the base level and every pinned level, producing
        # 0-based codes for the levels that actually get a design column.
        # Base-level observations are excluded from the design matrix entirely
        # (absorbed into the intercept), so we encode them as -1 -- the sink bin
        # the categorical kernel already reads as zero contribution.
        emitted = [
            i for i, lev in enumerate(self._levels) if lev != self._base_level and observed[i]
        ]
        self._non_base = [self._levels[i] for i in emitted]
        n_levels = len(self._non_base)
        remap = np.full(len(self._levels), -1, dtype=np.intp)
        remap[emitted] = np.arange(n_levels, dtype=np.intp)
        cat_codes = remap[codes]

        return GroupInfo(columns=None, n_cols=n_levels, cat_codes=cat_codes)

    def _resolve_base(
        self,
        effective: NDArray[np.floating],
        observed: NDArray[np.bool_],
        sample_weight: NDArray[np.floating] | None,
    ) -> Any:
        """Pick the reference level, falling back when the request is empty."""
        self._base_fallback = None
        if self._pinned_base is not None:
            requested = self._pinned_base
        elif self._base_level and self._base_level in self._levels:
            # Reuse from a prior build on this spec: refitting one spec object
            # must not silently move the coefficient identity.
            requested = self._base_level
        else:
            requested = self.base

        if requested == "most_exposed":
            base_level = self._most_exposed(effective, observed, sample_weight)
        elif requested == "first":
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
                    stacklevel=3,
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

    def _predict_codes(self, x: NDArray) -> NDArray:
        """Codes against the fitted universe; -1 for levels the policy admits."""
        codes = _codes_against(x, self._levels)
        if self.unseen != "error":
            _warn_unseen_routed(x, codes)
        return codes

    def transform(self, x: NDArray) -> NDArray:
        """One-hot encode using levels learned during build()."""
        x = self._resolve_predict_labels(x)
        if self.unseen != "error":
            # Equality masks already give a novel level an all-zero row, which
            # IS base routing; the codes are computed only to report it.
            self._predict_codes(x)
        return np.column_stack([(x == lev).astype(np.float64) for lev in self._non_base])

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
