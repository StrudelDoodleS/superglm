"""Random-effect feature specification."""

from __future__ import annotations

import math
import warnings
from collections.abc import Hashable
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from superglm.types import GroupInfo, LambdaPolicy


class RandomEffect:
    """All-level categorical effect with a REML-estimated variance component.

    ``levels=`` binds the level universe (spec 2026-08-11, §3.1) from an
    explicit sequence, a data column, or a categorical dtype.  A declared level
    with no training rows is not pinned the way an unpenalized dummy is: it
    keeps its own coefficient and shrinks to the population value through the
    variance component, exactly as a thinly observed level does.

    ``nested_in=`` names another ``RandomEffect`` feature of the same model
    that this one is nested in: every level of this effect belongs to exactly
    one of its levels (a region within a country, a vehicle model within a
    make).  Nesting is also detected from the level codes without it; the
    declaration is checked on the training rows, and a row that breaks it is
    an error naming the row and both of the parent's levels.  When this effect
    is the structured solver's leaf (the random effect with the most levels,
    with no ``FactorSmooth`` in the model), the declared parent is always in
    its elimination chain; otherwise the declaration is only validated.

    ``unseen=`` sets what a level the fit never saw is predicted with:
    ``"population"`` (the default) gives it zero, the population value;
    ``"weighted"`` gives it the mean of the fitted level effects, each
    weighted by its training weight (``sample_weight``, or rows), so a new
    level is priced as the average exposure is rather than the average
    level; ``"error"`` refuses it.  ``random_effects()`` reports that value
    and its standard error (``RandomEffectResult``).

    Notes
    -----
    When a REML-estimated ``RandomEffect`` is fitted beside an unpenalised
    ``Categorical`` whose levels include some with exposure but no positive
    response (under a log link with a zero-mass family such as Tweedie or
    Poisson), those levels separate -- their coefficients have no finite MLE
    -- and the marginal likelihood becomes nearly flat in this term's
    variance.  The fitted variance component is then poorly determined, and
    for the estimated-scale Tweedie criterion it is additionally biased
    upward relative to exact-likelihood REML.  ``fit_reml`` warns on that
    configuration; treat the published ``tau_squared`` with care there.
    """

    requires_reml = True

    def __init__(
        self,
        *,
        levels=None,
        unseen: Literal["population", "weighted", "error"] = "population",
        missing: Literal["error"] = "error",
        lambda_policy: LambdaPolicy | None = None,
        nested_in: Hashable | None = None,
    ):
        from superglm.features._level_source import resolve_level_source

        if nested_in is not None and not isinstance(nested_in, Hashable):
            raise TypeError("nested_in must be the name of another RandomEffect feature or None")
        if unseen not in ("population", "weighted", "error"):
            raise ValueError(f"unseen must be 'population', 'weighted' or 'error', got {unseen!r}")
        if missing != "error":
            raise ValueError(f"missing must be 'error', got {missing!r}")
        if lambda_policy is not None and not isinstance(lambda_policy, LambdaPolicy):
            raise TypeError("lambda_policy must be a LambdaPolicy or None")

        self.unseen = unseen
        self.missing = missing
        self._lambda_policy = lambda_policy
        self.nested_in = nested_in
        self._declared_levels: list | None = (
            None if levels is None else resolve_level_source(levels, context="RandomEffect")
        )
        self._level_source: str = "declared" if levels is not None else "inferred"
        self._levels: list[Any] = []
        self._level_to_code: dict[Any, int] = {}
        # Each level's training weight, read by ``unseen="weighted"``.
        self._level_weight: NDArray[np.floating] = np.zeros(0, dtype=np.float64)

    def adopt_dtype_categories(self, categories: list) -> None:
        """Adopt a dtype-declared universe unless one is already declared."""
        if self._declared_levels is None:
            from superglm.features._level_source import resolve_level_source

            self._declared_levels = resolve_level_source(list(categories), context="RandomEffect")
            self._level_source = "dtype"

    def apply_level_binding(self, binding) -> None:
        """Adopt a full-frame universe when nothing more specific declared one.

        Only the levels are read: a penalized term has no base level, so its
        bindings carry ``base=None`` and there is nothing to pin.
        """
        if self._declared_levels is None and binding.levels is not None:
            self._declared_levels = list(binding.levels)
            self._level_source = "full-frame"

    def resolve_binding(self, values: NDArray, sample_weight=None):
        """Compute this spec's full-frame binding without mutating the spec."""
        import copy

        from superglm.types import LevelBinding

        # Build on a throwaway copy so the universe and its NaN checks stay
        # single-sourced in `build`.
        probe = copy.deepcopy(self)
        probe.build(values, sample_weight=sample_weight)
        return LevelBinding(levels=tuple(probe._levels), base=None)

    def _declared_codes(self, values: NDArray) -> NDArray[np.intp]:
        """Code *values* against the bound universe, rejecting anything outside it."""
        codes = pd.Index(self._levels).get_indexer(values).astype(np.intp, copy=False)
        if np.any(codes < 0):
            # A -1 under a bound universe is either a broken column or data the
            # declaration does not admit; those are different bugs.
            outside = values[codes < 0]
            if np.any(pd.isna(outside)):
                raise ValueError("RandomEffect column contains missing values (NaN or None).")
            raise ValueError(
                f"Training data contains levels outside the declared level universe: "
                f"{sorted(set(outside.tolist()), key=str)}. Declared: "
                f"{sorted(self._levels, key=str)}. Widen levels= or fix the column."
            )
        return codes

    def build(
        self,
        x: NDArray,
        sample_weight: NDArray[np.floating] | None = None,
    ) -> GroupInfo:
        """Factorize all fitted levels without dropping a reference category.

        Each level's training weight (*sample_weight*, or one per row) is kept
        for ``unseen="weighted"``.
        """
        values = np.asarray(x).ravel()
        if self._declared_levels is not None:
            self._levels = list(self._declared_levels)
            codes = self._declared_codes(values)
        else:
            codes, uniques = pd.factorize(values, sort=True)
            if np.any(codes < 0):
                raise ValueError("RandomEffect column contains missing values (NaN or None).")
            self._levels = uniques.tolist()
        self._level_to_code = {level: code for code, level in enumerate(self._levels)}
        self._level_weight = np.bincount(
            codes,
            weights=None if sample_weight is None else np.asarray(sample_weight).ravel(),
            minlength=len(self._levels),
        ).astype(np.float64)
        return GroupInfo(
            columns=None,
            n_cols=len(self._levels),
            penalized=True,
            cat_codes=codes.astype(np.intp, copy=False),
            lambda_policies=(
                None if self._lambda_policy is None else {"_default": self._lambda_policy}
            ),
            structured_kind="random_effect",
            random_effect_nested_in=getattr(self, "nested_in", None),
        )

    def validate_prediction_values(self, x: NDArray) -> None:
        """Reject missing values without applying the unseen-level policy."""
        values = np.asarray(x).ravel()
        if np.any(pd.isna(values)):
            raise ValueError("RandomEffect column contains missing values (NaN or None).")

    def _prediction_codes(self, x: NDArray) -> NDArray[np.intp]:
        values = np.asarray(x).ravel()
        self.validate_prediction_values(values)
        codes = pd.Index(self._levels).get_indexer(values).astype(np.intp, copy=False)
        unseen_mask = codes < 0
        if self.unseen == "error" and np.any(unseen_mask):
            unseen = pd.unique(values[unseen_mask]).tolist()
            raise ValueError(f"Encountered unseen RandomEffect levels: {unseen}.")
        return codes

    def unseen_contrast(self) -> NDArray[np.floating]:
        """The weights over the fitted levels that an unseen level's effect takes.

        Zero under ``unseen="population"`` (and ``"error"``, which refuses an
        unseen level); each level's share of the training weight under
        ``unseen="weighted"``, zero if the levels carry no weight.
        """
        contrast = np.zeros(len(self._levels), dtype=np.float64)
        weight = getattr(self, "_level_weight", contrast)
        total = math.fsum(weight.tolist())
        if self.unseen == "weighted" and total > 0.0:
            contrast = weight / total
        return contrast

    def score(
        self,
        x: NDArray,
        beta: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Select fitted level effects without materializing one-hot columns."""
        codes = self._prediction_codes(x)
        beta = np.asarray(beta, dtype=np.float64)
        effects = np.zeros(len(codes), dtype=np.float64)
        known = codes >= 0
        effects[known] = beta[codes[known]]
        if self.unseen == "weighted" and not np.all(known):
            effects[~known] = self.unseen_contrast() @ beta
        return effects

    def transform(self, x: NDArray) -> NDArray[np.floating]:
        """Materialize a small all-level one-hot reference matrix (shares on an unseen row)."""
        codes = self._prediction_codes(x)
        transformed = np.zeros((len(codes), len(self._levels)), dtype=np.float64)
        known = codes >= 0
        transformed[np.flatnonzero(known), codes[known]] = 1.0
        if self.unseen == "weighted" and not np.all(known):
            transformed[~known] = self.unseen_contrast()
        return transformed

    def reconstruct(self, beta: NDArray[np.floating]) -> dict[str, Any]:
        """Return one fitted effect for every represented level."""
        effects = {
            level: float(value)
            for level, value in zip(self._levels, np.asarray(beta).ravel(), strict=True)
        }
        return {
            "levels": self._levels.copy(),
            "effects": effects,
            "log_relativities": effects.copy(),
            "relativities": {level: float(np.exp(value)) for level, value in effects.items()},
        }


# Near-nesting disclosure (one-engine design §3.13): an undeclared pair of
# random effects is named when all but a few rows nest.  "A few" is at most
# this share of the rows (and at least one row); a crossed pair breaks nesting
# on most of its rows, so the warning is not a routing input and its bound is
# a disclosure choice, not a numerical one.
_NEAR_NESTING_SHARE = 0.01
_NEAR_NESTING_ROWS_SHOWN = 3


def _factorized(values) -> tuple[NDArray[np.intp], NDArray]:
    codes, uniques = pd.factorize(np.asarray(values).ravel())
    return codes.astype(np.intp, copy=False), np.asarray(uniques, dtype=object)


def _first_rows(codes: NDArray[np.intp], n_levels: int) -> NDArray[np.intp]:
    """The first row of every level (``-1`` for a level with no row)."""
    first = np.full(n_levels, -1, dtype=np.intp)
    rows = np.arange(len(codes), dtype=np.intp)
    first[codes[::-1]] = rows[::-1]
    return first


def random_effect_specs(specs) -> dict:
    """The ``RandomEffect`` features of a feature mapping, in its order."""
    return {name: spec for name, spec in specs.items() if isinstance(spec, RandomEffect)}


def _first_break(child: tuple, parent: tuple) -> tuple[int, int, int] | None:
    """The first row whose child level already met another parent level, if any.

    ``child`` and ``parent`` are ``(codes, labels)`` on the same rows.
    Returns ``(row, earlier row of the same child level, rows breaking)``.
    """
    child_codes, child_labels = child
    first = _first_rows(child_codes, len(child_labels))[child_codes]
    broken = np.flatnonzero(parent[0][first] != parent[0])
    if not broken.size:
        return None
    return int(broken[0]), int(first[broken[0]]), int(broken.size)


class _TrainingLevels:
    """Codes of every fitted random-effect level on the training rows, read once each.

    A ``RandomEffect`` feature is factorized from its column; a ``Hierarchy``
    is parsed once for all its depths, including the top parts its
    ``nested_in`` level covers, which it reads but does not fit.
    """

    def __init__(self, specs, column):
        from superglm.features.hierarchy import hierarchy_specs, random_effect_level_names

        self._specs = specs
        self._column = column
        self._hierarchies = hierarchy_specs(specs)
        self.names = random_effect_level_names(specs)
        self._depth_of: dict[Hashable, tuple[Hashable, int]] = {
            spec.level_name(depth): (name, depth)
            for name, spec in self._hierarchies.items()
            for depth in spec.fitted_depths
        }
        self._parsed: dict = {}

    def __contains__(self, name) -> bool:
        return name in self.names

    def _hierarchy(self, name) -> list:
        if name not in self._parsed:
            from superglm.features.hierarchy import feature_column

            spec = self._hierarchies[name]
            self._parsed[name] = spec.training_codes(feature_column(self._column, name, spec))
        return self._parsed[name]

    def codes(self, name) -> tuple[NDArray[np.intp], Any] | None:
        """``(codes, labels)`` of one fitted level; ``None`` when its column has gaps."""
        if name in self._depth_of:
            feature, depth = self._depth_of[name]
            return self._hierarchy(feature)[depth]
        codes, labels = _factorized(self._column(name))
        return None if np.any(codes < 0) else (codes, labels)

    def depth(self, feature, depth: int) -> tuple[NDArray[np.intp], list]:
        """One depth of a hierarchy, fitted or covered by its ``nested_in`` level."""
        return self._hierarchy(feature)[depth]


def _check_same_grouping(name: Hashable, spec, levels: _TrainingLevels) -> None:
    """A hierarchy's top parts must group the rows exactly as its ``nested_in`` level does."""
    parent = spec.nested_in
    prefix = levels.depth(name, spec._shared_depth - 1)
    reference = levels.codes(parent)
    if reference is None:
        return
    covered = ", ".join(spec.names[: spec._shared_depth])
    for finer, coarser, finer_name, coarser_name in (
        (prefix, reference, f"{name!r} parts ({covered})", repr(parent)),
        (reference, prefix, repr(parent), f"{name!r} parts ({covered})"),
    ):
        found = _first_break(finer, coarser)
        if found is None:
            continue
        row, head, count = found
        raise ValueError(
            f"Hierarchy {name!r} is declared nested_in={parent!r}, so its first "
            f"{spec._shared_depth} part(s) ({covered}) must group the training rows exactly "
            f"as {parent!r} does, but {finer_name} value {finer[1][finer[0][row]]!r} meets "
            f"{coarser_name} value {coarser[1][coarser[0][head]]!r} (row {head}) and "
            f"{coarser[1][coarser[0][row]]!r} (row {row}); rows are counted from 0 in the "
            f"training data, and {count} row(s) disagree. List first the columns that "
            f"{parent!r} reads, cleaned the same way (normalize=), or remove nested_in."
        )


def _check_duplicated_levels(levels: _TrainingLevels) -> None:
    """Refuse a hierarchy level that groups the rows exactly as another term's level does.

    Two such levels are one random effect fitted twice: their columns are the
    same up to order, so the data cannot split the variance between them.
    Across two terms that is a declaration to fix (share the level through
    ``nested_in``), so it is refused.  Two levels of one hierarchy coincide
    only because these rows hold one group below each group above; the fit
    still determines their combined effect (REML splits the variance evenly,
    by symmetry), so that is disclosed in a warning rather than refused.  Two
    plain ``RandomEffect`` features are left as they were.
    """
    hierarchy_levels = set(levels._depth_of)
    if not hierarchy_levels:
        return
    coded = {name: levels.codes(name) for name in levels.names}
    coded = {name: value for name, value in coded.items() if value is not None}
    names = list(coded)
    for position, first_name in enumerate(names):
        for second_name in names[position + 1 :]:
            if first_name not in hierarchy_levels and second_name not in hierarchy_levels:
                continue
            first, second = coded[first_name], coded[second_name]
            if np.bincount(first[0]).nonzero()[0].size != np.bincount(second[0]).nonzero()[0].size:
                continue
            if _first_break(first, second) is not None:
                continue
            same_term = (
                first_name in hierarchy_levels
                and second_name in hierarchy_levels
                and levels._depth_of[first_name][0] == levels._depth_of[second_name][0]
            )
            if same_term:
                warnings.warn(
                    f"Hierarchy levels {first_name!r} and {second_name!r} group the training "
                    "rows identically: each group of the coarser level holds a single group of "
                    "the finer one in these rows, so the data cannot split the variance "
                    "between them. The fit determines their combined effect; each level's "
                    "reported variance is an arbitrary share of it. Drop one of the two "
                    "levels, or fit on rows that tell them apart.",
                    UserWarning,
                    stacklevel=4,
                )
                continue
            raise ValueError(
                f"Random-effect levels {first_name!r} and {second_name!r} group the training "
                "rows identically, so they are one effect fitted twice and the data cannot "
                f"split its variance between them; fit it once: give the term that repeats it "
                f"nested_in={first_name!r}, and its levels below continue that one instead of "
                "creating it again."
            )


def validate_declared_nesting(specs, column) -> None:
    """Check every declared random-effect nesting on the training rows.

    ``specs`` maps feature names to specs and ``column(name)`` returns an
    input column's training values.  A ``RandomEffect(nested_in=)`` parent
    must be another random-effect level of the model -- a ``RandomEffect``
    feature or a ``Hierarchy`` level ``"<feature>:<level>"`` -- and every level
    of the child must meet one parent level on the training rows; a row that
    breaks it raises a ``ValueError`` naming the row (its position in the
    training data) and both parent levels.  A ``Hierarchy(nested_in=)``'s top
    parts must group the rows exactly as its parent level does, and a
    hierarchy level that groups them exactly as another fitted level does is
    refused.  Missing values are left to the specs' own check.
    """
    from superglm.features.hierarchy import bind_hierarchies, hierarchy_specs

    effects = random_effect_specs(specs)
    hierarchies = hierarchy_specs(specs)
    if hierarchies:
        bind_hierarchies(specs)
    levels = _TrainingLevels(specs, column)
    for child, spec in effects.items():
        parent = getattr(spec, "nested_in", None)
        if parent is None:
            continue
        if parent == child or parent not in levels:
            raise ValueError(
                f"RandomEffect {child!r} is declared nested_in={parent!r}, which is not "
                "another RandomEffect feature or Hierarchy level of this model; its "
                f"random-effect levels are {sorted(levels.names, key=str)!r}."
            )
        child_coded = levels.codes(child)
        parent_coded = levels.codes(parent)
        if child_coded is None or parent_coded is None:
            continue
        found = _first_break(child_coded, parent_coded)
        if found is not None:
            row, head, count = found
            child_codes, child_levels = child_coded
            parent_codes, parent_levels = parent_coded
            raise ValueError(
                f"RandomEffect {child!r} is declared nested_in={parent!r}, but its level "
                f"{child_levels[child_codes[row]]!r} occurs under {parent!r} level "
                f"{parent_levels[parent_codes[head]]!r} (row {head}) and under "
                f"{parent_levels[parent_codes[row]]!r} (row {row}); rows are counted from 0 "
                f"in the training data, and {count} row(s) break the declaration. Fix "
                f"those rows, or remove nested_in to fit {child!r} and {parent!r} as crossed "
                "effects."
            )
    for name, spec in hierarchies.items():
        if spec.nested_in is not None:
            _check_same_grouping(name, spec, levels)
    _check_duplicated_levels(levels)


def near_nesting_notes(specs, column) -> list[str]:
    """Name each undeclared pair of random effects that nests on all but a few rows.

    For every ordered pair (child, parent) of ``RandomEffect`` features with
    at least as many child levels, the rows off their child level's most
    frequent parent level break nesting.  A pair broken by at least one row
    and at most ``_NEAR_NESTING_SHARE`` of the rows is named, with a few of
    those rows: the fit treats such a pair as crossed.  The notes never
    change a fit's route; ``validate_declared_nesting`` covers the declared
    pairs.
    """
    effects = random_effect_specs(specs)
    if len(effects) < 2:
        return []
    coded = {name: _factorized(column(name)) for name in effects}
    notes = []
    for child, (child_codes, child_levels) in coded.items():
        for parent, (parent_codes, parent_levels) in coded.items():
            if (
                parent == child
                or getattr(effects[child], "nested_in", None) == parent
                or len(child_levels) < len(parent_levels)
                or np.any(child_codes < 0)
                or np.any(parent_codes < 0)
            ):
                continue
            first = _first_rows(child_codes, len(child_levels))[child_codes]
            if np.array_equal(parent_codes[first], parent_codes):
                continue  # nested: the solver detects it from the codes
            pair = child_codes.astype(np.int64) * len(parent_levels) + parent_codes
            cells, inverse, counts = np.unique(pair, return_inverse=True, return_counts=True)
            cell_child = cells // len(parent_levels)
            # each child level's most frequent parent cell (ties: the first)
            order = np.lexsort((-counts, cell_child))
            leading = np.zeros(len(cells), dtype=bool)
            leading[order[np.r_[True, cell_child[order][1:] != cell_child[order][:-1]]]] = True
            broken = np.flatnonzero(~leading[inverse])
            limit = max(1, int(_NEAR_NESTING_SHARE * len(child_codes)))
            if not 0 < broken.size <= limit:
                continue
            majority = {
                int(cell // len(parent_levels)): int(cell % len(parent_levels))
                for cell in cells[leading]
            }
            examples = "; ".join(
                f"row {int(row)}: {child!r} level {child_levels[child_codes[row]]!r} under "
                f"{parent!r} level {parent_levels[parent_codes[row]]!r}, where most of its rows "
                f"are under {parent_levels[majority[int(child_codes[row])]]!r}"
                for row in broken[:_NEAR_NESTING_ROWS_SHOWN]
            )
            notes.append(
                f"RandomEffect {child!r} is nested in {parent!r} except for {broken.size} "
                f"row(s) ({examples}; rows are counted from 0 in the training data), so the "
                f"fit treats the two as crossed. If each {child!r} level belongs to one "
                f"{parent!r} level, fix those rows and declare "
                f"RandomEffect(nested_in={parent!r})."
            )
    return notes
