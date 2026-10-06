"""Nested random effects declared as one hierarchy of levels.

``Hierarchy`` is the SuperGLM spelling of lme4's ``(1 | make/model/year)``,
which Bates, Maechler, Bolker and Walker (*Fitting Linear Mixed-Effects Models
Using lme4*, J. Stat. Softw. 67(1), 2015, Table 2) define as
``(1 | make) + (1 | make:model) + (1 | make:model:year)``: one random
intercept per level of the hierarchy, each with its own variance component,
where every level below the top is the interaction of the parts above it.
mgcv writes the same interaction factor as ``s(make, model, bs="re")``
(``smooth.construct.re.smooth.spec``).  In insurance pricing it is the
hierarchical credibility model of Jewell (*The Use of Collateral Data in
Credibility Theory: A Hierarchical Model*, Giornale dell'Istituto Italiano
degli Attuari 38, 1975) carried into a GLM as Ohlsson does for car-model
classification (*Combining Generalized Linear Models and Credibility Models in
Practice*, Scand. Actuar. J. 2008(4)); Campo and Antonio (*Insurance Pricing
with Hierarchically Structured Data*, arXiv:2206.15244) set out the GLMM form
fitted here.

Each level's key is the tuple of the parts above and at it, built from the
parts themselves.  Two different paths therefore never share a key, even when
a separator occurs inside a name (make ``"A-B"`` with model ``"C"`` against
make ``"A"`` with model ``"B-C"``); a string re-join would merge them.
"""

from __future__ import annotations

import unicodedata
import warnings
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from superglm.types import GroupInfo, LambdaPolicy

# How many offending values or merged spellings a message names.
_SHOWN = 5

ColumnReader = Callable[[Hashable], NDArray]


def normalize_label(value: Any) -> Any:
    """The comparison form of one text label; other values are returned unchanged.

    NFKC folds compatibility characters (full-width letters, ligatures,
    non-breaking spaces), ``str.casefold`` folds case, and the second NFKC
    restores the normal form casefolding can break (the Unicode Standard,
    section 3.13, *Default Caseless Matching*).  ``str.split()`` then strips
    the ends and collapses every internal whitespace run to one space.
    """
    if not isinstance(value, str):
        return value
    folded = unicodedata.normalize("NFKC", unicodedata.normalize("NFKC", value).casefold())
    return " ".join(folded.split())


def _sorted_values(values) -> list:
    """Distinct values in a deterministic order, even when their types do not compare."""
    distinct = set(values)
    try:
        return sorted(distinct)
    except TypeError:
        return sorted(distinct, key=lambda value: (type(value).__name__, str(value)))


def _first_rows(codes: NDArray[np.intp], n_codes: int) -> NDArray[np.intp]:
    first = np.full(n_codes, -1, dtype=np.intp)
    rows = np.arange(len(codes), dtype=np.intp)
    first[codes[::-1]] = rows[::-1]
    return first


def _level_names(values, argument: str) -> tuple[str, ...]:
    names = tuple(values)
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError(f"Hierarchy {argument}= must hold non-empty strings, got {names!r}")
    if any(":" in name for name in names):
        raise ValueError(
            f"Hierarchy level names may not contain ':', which separates a term from its "
            f"level in fitted names; got {names!r}. Pass levels= to name them."
        )
    if len(set(names)) != len(names):
        raise ValueError(f"Hierarchy level names must be distinct, got {names!r}")
    return names


class HierarchyLevel:
    """One fitted level of a :class:`Hierarchy`, read as the random effect it is.

    ``random_effects()`` reports a level through the same two attributes it
    reads from a ``RandomEffect``: ``_levels`` (here, paths) and
    ``_prediction_codes`` (each row's path at this level, ``-1`` if unseen).
    """

    def __init__(self, hierarchy: Hierarchy, depth: int):
        self._hierarchy = hierarchy
        self._depth = depth

    @property
    def _levels(self) -> list[tuple]:
        return self._hierarchy._paths[self._depth]

    def _prediction_codes(self, x) -> NDArray[np.intp]:
        return self._hierarchy._prediction_codes(x)[self._depth]


class Hierarchy:
    """Nested random intercepts for a hierarchy of levels, such as make, model, year.

    The hierarchy fits one REML variance component per level, exactly as
    ``RandomEffect`` does for one factor.  Level ``k`` groups the rows by
    their first ``k`` parts -- a model is identified by make and model, a
    model year by make, model and year -- so a model name reused under two
    makes is two models, and the levels nest by construction.  A thin model
    shrinks toward its make, and a thin make toward the population.

    Declare the parts in one of two ways:

    - ``Hierarchy(levels=("make", "model", "year"), sep="-")`` reads the
      feature's own column, whose values join the parts with ``sep``
      (``"Ford-Focus-2019"``).  Every value must split into exactly
      ``len(levels)`` parts; a value that does not is an error naming it,
      because a separator inside a name cannot be split unambiguously.
    - ``Hierarchy(columns=("make", "model", "year"))`` reads one column per
      level, top level first, and needs no separator.  ``levels=`` may rename
      the levels; it defaults to the column names.

    Each fitted level is a term of its own named ``"<feature>:<level>"``
    (``"vehicle:make"``, ``"vehicle:model"``, ``"vehicle:year"``):
    ``random_effects("vehicle:model")`` reports it, and another
    ``RandomEffect`` or ``Hierarchy`` can declare ``nested_in=`` it.  Its
    level labels are paths, ``("ford", "focus")``.

    ``nested_in=`` names a level another term already fits -- a hierarchy
    level ``"vehicle:make"`` or a ``RandomEffect`` feature.  This hierarchy's
    first parts, as many as that level has, must group the training rows
    exactly as it does; they are that level, so they are not fitted again,
    and this hierarchy fits only its deeper levels.  That is how a crossed
    effect shares a parent: with ``vehicle`` as above,
    ``Hierarchy(columns=("make", "engine"), nested_in="vehicle:make")`` adds
    lme4's ``(1 | make:engine)`` without fitting make a second time.  Two
    levels that group the rows identically are the same effect fitted twice,
    whose variance the data cannot split between them, so a fit refuses any
    hierarchy level that duplicates another level.

    ``normalize=True`` compares text parts after Unicode NFKC folding,
    casefolding, trimming and collapsing internal whitespace, so ``"FORD "``
    and ``"Ford"`` are one make.  The same folding applies at fit and at
    prediction, the labels are reported in folded form, and every merge of
    distinct spellings is reported: in a warning at fit and under
    ``"merged_labels"`` in ``reconstruct_feature()``.  Values that are not
    text (a numeric year) are used as they are.

    A row whose path was not seen in training takes the effect of the deepest
    level it shares with the training rows: a new model of a known make gets
    that make's effect, and a new make gets the population value
    (``unseen="population"``, as lme4's ``allow.new.levels`` and mgcv's
    ``bs="re"`` predict a new level).  ``unseen="error"`` refuses any row whose
    full path is new.  Missing parts are an error.

    Notes
    -----
    Each level is penalised as a ``RandomEffect`` is, so the fit needs
    ``fit_reml()``.  The relativities of the term are the combined effect of
    every fitted level on each full path; ``random_effects()`` reports each
    level's own variance, credibility and posterior standard errors.
    """

    requires_reml = True

    def __init__(
        self,
        *,
        levels: Sequence[str] | None = None,
        sep: str | None = None,
        columns: Sequence[str] | None = None,
        nested_in: Hashable | None = None,
        normalize: bool = False,
        unseen: Literal["population", "error"] = "population",
        lambda_policy: LambdaPolicy | None = None,
    ):
        if (sep is None) == (columns is None):
            raise ValueError(
                "Hierarchy takes exactly one of sep= (one column holding the joined parts) "
                "or columns= (one column per level)."
            )
        if sep is not None:
            if not isinstance(sep, str) or not sep:
                raise ValueError(f"sep must be a non-empty string, got {sep!r}")
            if levels is None:
                raise ValueError(
                    "Hierarchy(sep=...) needs levels=, the name of each part from the top "
                    "level down, e.g. levels=('make', 'model', 'year')."
                )
            names = _level_names(levels, "levels")
            source = None
        else:
            if isinstance(columns, str):
                raise TypeError("columns= takes a sequence of column names, not one string")
            source = tuple(columns or ())
            if any(not isinstance(column, str) or not column.strip() for column in source):
                raise ValueError(f"Hierarchy columns= must hold column names, got {source!r}")
            if len(set(source)) != len(source):
                raise ValueError(f"Hierarchy columns must be distinct, got {source!r}")
            names = _level_names(source if levels is None else levels, "levels")
            if len(names) != len(source):
                raise ValueError(
                    f"levels= names {len(names)} level(s) but columns= lists {len(source)}"
                )
        if len(names) < 2:
            raise ValueError(
                "A Hierarchy needs at least two levels; one grouping column is a RandomEffect()."
            )
        if nested_in is not None and not isinstance(nested_in, str):
            raise TypeError(
                "nested_in must name a level another term fits, such as 'vehicle:make', or be None"
            )
        if not isinstance(normalize, bool):
            raise TypeError(f"normalize must be True or False, got {normalize!r}")
        if unseen not in ("population", "error"):
            raise ValueError(f"unseen must be 'population' or 'error', got {unseen!r}")
        if lambda_policy is not None and not isinstance(lambda_policy, LambdaPolicy):
            raise TypeError("lambda_policy must be a LambdaPolicy or None")

        self.levels = names
        self.sep = sep
        self.columns = source
        self.nested_in = nested_in
        self.normalize = normalize
        self.unseen = unseen
        self._lambda_policy = lambda_policy
        # Bound by ``bind_hierarchies`` from the model's feature mapping.
        self._name: Hashable | None = None
        self._shared_depth = 0
        # Fitted state: each fitted depth's paths and their positions.
        self._paths: dict[int, list[tuple]] = {}
        self._path_codes: dict[int, dict[tuple, int]] = {}
        self._merged: dict[str, dict[Any, tuple]] = {}

    def __repr__(self) -> str:
        source = f"sep={self.sep!r}" if self.columns is None else f"columns={self.columns!r}"
        return f"Hierarchy(levels={self.levels!r}, {source})"

    # ── Reading the parts ────────────────────────────────────────

    def source_columns(self, name: Hashable) -> tuple[Hashable, ...]:
        """The input columns this feature reads when it is named *name*."""
        return (name,) if self.columns is None else self.columns

    def read_column(self, column: ColumnReader, name: Hashable) -> NDArray:
        """The feature's values: its joined column, or an ``(n, levels)`` array of parts."""
        if self.columns is None:
            return np.asarray(column(name)).ravel()
        arrays = [np.asarray(column(source)).ravel() for source in self.columns]
        parts = np.empty((len(arrays[0]), len(arrays)), dtype=object)
        for position, values in enumerate(arrays):
            parts[:, position] = values
        return parts

    def _raw_paths(self, x) -> tuple[NDArray[np.intp], NDArray]:
        """Each row's distinct full path, and those paths' parts as given."""
        values = np.asarray(x, dtype=object)
        if self.columns is None:
            values = values.ravel()
            codes, uniques = pd.factorize(values)
            if np.any(codes < 0):
                raise ValueError("Hierarchy column contains missing values (NaN or None).")
            return codes.astype(np.intp, copy=False), self._split(uniques.tolist())
        if values.ndim != 2 or values.shape[1] != len(self.columns):
            raise ValueError(
                f"Hierarchy expects one value per column {self.columns!r} on every row"
            )
        combined = np.zeros(values.shape[0], dtype=np.int64)
        part_codes = []
        part_values = []
        for position, column in enumerate(self.columns):
            codes, uniques = pd.factorize(values[:, position])
            if np.any(codes < 0):
                raise ValueError(f"Hierarchy column {column!r} contains missing values.")
            part_codes.append(codes)
            part_values.append(uniques)
            # Re-factorized at every step, so the pairing stays below n**2.
            combined = pd.factorize(combined * len(uniques) + codes)[0].astype(np.int64)
        path_codes, _ = pd.factorize(combined)
        first = _first_rows(path_codes, int(path_codes.max()) + 1 if len(path_codes) else 0)
        raw = np.empty((len(first), len(self.columns)), dtype=object)
        for position, (codes, uniques) in enumerate(zip(part_codes, part_values, strict=True)):
            raw[:, position] = np.asarray(uniques, dtype=object)[codes[first]]
        return path_codes.astype(np.intp, copy=False), raw

    def _split(self, joined: list) -> NDArray:
        """Split each distinct joined value into exactly one part per level."""
        sep = self.sep
        assert sep is not None
        not_text = [value for value in joined if not isinstance(value, str)]
        if not_text:
            shown = ", ".join(f"{value!r} ({type(value).__name__})" for value in not_text[:_SHOWN])
            raise ValueError(
                f"Hierarchy with sep={sep!r} splits text, but the column holds {shown}. "
                "Use Hierarchy(columns=(...)) for parts held in separate columns."
            )
        pieces = [value.split(sep) for value in joined]
        wrong = [
            (value, len(parts))
            for value, parts in zip(joined, pieces, strict=True)
            if len(parts) != len(self.levels)
        ]
        if wrong:
            shown = ", ".join(f"{value!r} ({count} parts)" for value, count in wrong[:_SHOWN])
            raise ValueError(
                f"Hierarchy with sep={sep!r} needs every value to split into exactly "
                f"{len(self.levels)} parts ({', '.join(self.levels)}), but {len(wrong)} of "
                f"{len(joined)} distinct values do not: {shown}. A separator inside a name "
                "makes the split ambiguous, and it is never guessed: store the parts in "
                "separate columns and use Hierarchy(columns=(...)), or join them with a "
                "separator that no name contains."
            )
        raw = np.empty((len(pieces), len(self.levels)), dtype=object)
        for row, parts in enumerate(pieces):
            raw[row, :] = parts
        return raw

    def _folded(self, raw: NDArray) -> NDArray:
        if not self.normalize:
            return raw
        folded = np.empty_like(raw)
        for position in range(raw.shape[1]):
            folded[:, position] = [normalize_label(value) for value in raw[:, position]]
        return folded

    def _paths_of(self, x) -> tuple[NDArray[np.intp], NDArray, NDArray]:
        """Row path codes, the distinct paths as given, and their compared form."""
        codes, raw = self._raw_paths(x)
        return codes, raw, self._folded(raw)

    @staticmethod
    def _depth_codes(parts: NDArray) -> list[tuple[NDArray[np.intp], list[tuple]]]:
        """For each depth, every distinct path's code and the depth's sorted paths."""
        out = []
        combined = np.zeros(parts.shape[0], dtype=np.int64)
        for position in range(parts.shape[1]):
            ordered = _sorted_values(parts[:, position].tolist())
            rank = {value: index for index, value in enumerate(ordered)}
            ranks = np.fromiter(
                (rank[value] for value in parts[:, position]), dtype=np.int64, count=parts.shape[0]
            )
            # Lexicographic: the depth above is already sorted, so its code
            # leads and this part's rank breaks ties within it.
            keys, combined = np.unique(combined * len(ordered) + ranks, return_inverse=True)
            combined = combined.reshape(-1).astype(np.int64)
            first = _first_rows(combined.astype(np.intp), len(keys))
            paths = [tuple(parts[row, : position + 1]) for row in first.tolist()]
            out.append((combined.astype(np.intp), paths))
        return out

    # ── Fitted names ─────────────────────────────────────────────

    @property
    def fitted_depths(self) -> range:
        """The depths this hierarchy fits: all but those its ``nested_in`` level covers."""
        return range(self._shared_depth, len(self.levels))

    def level_name(self, depth: int) -> str:
        """The fitted term name of one level, ``"<feature>:<level>"``."""
        return f"{self._name}:{self.levels[depth]}"

    def level_view(self, level: str) -> HierarchyLevel:
        """The fitted level named *level* (its bare name, such as ``"model"``)."""
        depth = self.levels.index(level)
        if depth not in self._paths:
            raise KeyError(f"Hierarchy level {level!r} is not fitted by this term")
        return HierarchyLevel(self, depth)

    # ── Build, score, report ─────────────────────────────────────

    def training_codes(self, x) -> list[tuple[NDArray[np.intp], list[tuple]]]:
        """Each row's code and the paths at every depth, top first, without fitting."""
        codes, _, folded = self._paths_of(x)
        return [(depth_codes[codes], paths) for depth_codes, paths in self._depth_codes(folded)]

    def build(self, x, sample_weight: NDArray | None = None) -> list[GroupInfo]:
        """One all-level random-effect block per fitted depth, top first."""
        del sample_weight
        codes, raw, folded = self._paths_of(x)
        self._merged = self._merges(raw, folded) if self.normalize else {}
        if self._merged:
            warnings.warn(self._merge_message(), UserWarning, stacklevel=2)
        depths = self._depth_codes(folded)
        self._paths = {}
        self._path_codes = {}
        infos = []
        policies = None if self._lambda_policy is None else {"_default": self._lambda_policy}
        for depth in self.fitted_depths:
            depth_codes, paths = depths[depth]
            self._paths[depth] = paths
            self._path_codes[depth] = {path: index for index, path in enumerate(paths)}
            if depth > self._shared_depth and self._name is not None:
                parent = self.level_name(depth - 1)
            else:
                parent = self.nested_in if depth == self._shared_depth else None
            infos.append(
                GroupInfo(
                    columns=None,
                    n_cols=len(paths),
                    penalized=True,
                    cat_codes=depth_codes[codes],
                    lambda_policies=policies,
                    subgroup_name=self.levels[depth],
                    structured_kind="random_effect",
                    random_effect_nested_in=parent,
                )
            )
        return infos

    def _merges(self, raw: NDArray, folded: NDArray) -> dict[str, dict[Any, tuple]]:
        merged: dict[str, dict[Any, tuple]] = {}
        for position, level in enumerate(self.levels):
            spellings: dict[Any, set] = {}
            for given, compared in zip(raw[:, position], folded[:, position], strict=True):
                spellings.setdefault(compared, set()).add(given)
            found = {
                compared: tuple(sorted(given, key=str))
                for compared, given in spellings.items()
                if len(given) > 1
            }
            if found:
                merged[level] = dict(sorted(found.items(), key=lambda item: str(item[0])))
        return merged

    def _merge_message(self) -> str:
        entries = [
            f"{level} {compared!r} <- {', '.join(repr(value) for value in given)}"
            for level, found in self._merged.items()
            for compared, given in found.items()
        ]
        shown = "; ".join(entries[:_SHOWN])
        more = f"; and {len(entries) - _SHOWN} more" if len(entries) > _SHOWN else ""
        return (
            f"Hierarchy normalize=True merged {len(entries)} group(s) of differently spelled "
            f"labels into one label each: {shown}{more}. reconstruct_feature() lists them all "
            "under 'merged_labels'."
        )

    def validate_prediction_values(self, x) -> None:
        """Reject missing or unsplittable values without applying the unseen policy."""
        self._raw_paths(x)

    def _prediction_codes(self, x) -> dict[int, NDArray[np.intp]]:
        """Each row's code at every fitted depth; ``-1`` where the path was not fitted."""
        codes, _, folded = self._paths_of(x)
        out = {}
        for depth in self.fitted_depths:
            known = self._path_codes[depth]
            lookup = np.fromiter(
                (known.get(tuple(path[: depth + 1]), -1) for path in folded),
                dtype=np.intp,
                count=len(folded),
            )
            out[depth] = lookup[codes]
        leaf = out[len(self.levels) - 1]
        if self.unseen == "error" and np.any(leaf < 0):
            unseen = [tuple(path) for path in folded[np.unique(codes[leaf < 0])][:_SHOWN]]
            raise ValueError(f"Encountered unseen Hierarchy paths: {unseen}.")
        return out

    def _offsets(self) -> dict[int, int]:
        offsets, start = {}, 0
        for depth in self.fitted_depths:
            offsets[depth] = start
            start += len(self._paths[depth])
        return offsets

    def score(self, x, beta: NDArray[np.floating]) -> NDArray[np.floating]:
        """Add each row's fitted effect at every depth it shares with training."""
        beta = np.asarray(beta, dtype=np.float64).ravel()
        offsets = self._offsets()
        effects = None
        for depth, codes in self._prediction_codes(x).items():
            known = codes >= 0
            block = np.zeros(len(codes), dtype=np.float64)
            block[known] = beta[offsets[depth] + codes[known]]
            effects = block if effects is None else effects + block
        assert effects is not None
        return effects

    def transform(self, x) -> NDArray[np.floating]:
        """Materialize the all-level indicator columns of every fitted depth."""
        offsets = self._offsets()
        width = sum(len(paths) for paths in self._paths.values())
        codes_by_depth = self._prediction_codes(x)
        n = len(next(iter(codes_by_depth.values())))
        transformed = np.zeros((n, width), dtype=np.float64)
        for depth, codes in codes_by_depth.items():
            known = np.flatnonzero(codes >= 0)
            transformed[known, offsets[depth] + codes[known]] = 1.0
        return transformed

    def leaf_coefficients(self) -> NDArray[np.intp]:
        """``(n_paths, n_fitted_depths)`` positions in the term's coefficients of each path's levels."""
        offsets = self._offsets()
        leaf_depth = len(self.levels) - 1
        columns = [
            np.fromiter(
                (
                    offsets[depth] + self._path_codes[depth][path[: depth + 1]]
                    for path in self._paths[leaf_depth]
                ),
                dtype=np.intp,
                count=len(self._paths[leaf_depth]),
            )
            for depth in self.fitted_depths
        ]
        return np.column_stack(columns)

    def reconstruct(self, beta: NDArray[np.floating]) -> dict[str, Any]:
        """Combined effect of every fitted level on each full path, and each level's own."""
        beta = np.asarray(beta, dtype=np.float64).ravel()
        leaves = self._paths[len(self.levels) - 1]
        total = beta[self.leaf_coefficients()].sum(axis=1)
        effects = {path: float(value) for path, value in zip(leaves, total, strict=True)}
        offsets = self._offsets()
        by_level = {}
        for depth in self.fitted_depths:
            paths = self._paths[depth]
            block = beta[offsets[depth] : offsets[depth] + len(paths)]
            by_level[self.levels[depth]] = {
                path: float(value) for path, value in zip(paths, block, strict=True)
            }
        return {
            "levels": list(leaves),
            "effects": effects,
            "log_relativities": effects.copy(),
            "relativities": {path: float(np.exp(value)) for path, value in effects.items()},
            "level_effects": by_level,
            "merged_labels": {level: dict(found) for level, found in self._merged.items()},
        }


# ── Model-level helpers ──────────────────────────────────────────


def feature_source_columns(name: Any, spec: Any) -> tuple[Any, ...]:
    """The input columns a main-effect feature reads: its own, or a hierarchy's parts."""
    if isinstance(spec, Hierarchy):
        return spec.source_columns(name)
    return (name,)


def feature_column(column: ColumnReader, name: Hashable, spec: Any) -> NDArray:
    """A main-effect feature's values, read through *column* (``frame.column_array``)."""
    if isinstance(spec, Hierarchy):
        return spec.read_column(column, name)
    return column(name)


def hierarchy_specs(specs: Mapping[Any, Any]) -> dict[Any, Hierarchy]:
    """The ``Hierarchy`` features of a feature mapping, in its order."""
    return {name: spec for name, spec in specs.items() if isinstance(spec, Hierarchy)}


def bind_hierarchies(specs: Mapping[Any, Any]) -> None:
    """Give every ``Hierarchy`` its feature name and resolve its ``nested_in`` level.

    A hierarchy's levels are named ``"<feature>:<level>"`` only once it knows
    its feature name, and ``nested_in`` decides how many of its top parts are
    another term's level: as many as that level has (one for a
    ``RandomEffect`` feature, ``k`` for a hierarchy's ``k``-th level).
    """
    from superglm.features.random_effect import RandomEffect

    hierarchies = hierarchy_specs(specs)
    random_effects = {name for name, spec in specs.items() if isinstance(spec, RandomEffect)}
    # Every level a hierarchy declares, fitted or not: (owning feature, depth).
    declared = {
        f"{name}:{level}": (name, position + 1)
        for name, spec in hierarchies.items()
        for position, level in enumerate(spec.levels)
    }
    for name, spec in hierarchies.items():
        spec._name = name
        spec._shared_depth = 0
        parent = spec.nested_in
        if parent in declared:
            owner, depth = declared[parent]
            if owner == name:
                raise ValueError(
                    f"Hierarchy {name!r} cannot be nested in its own level {parent!r}."
                )
        elif parent in random_effects:
            depth = 1
        else:
            continue  # no reference, or one reported below with the fitted levels
        if depth >= len(spec.levels):
            raise ValueError(
                f"Hierarchy {name!r} is declared nested_in={parent!r}, which already covers "
                f"its first {depth} part(s), so it has no deeper level to fit; list at least "
                f"{depth + 1} columns, the first {depth} being those of {parent!r}."
            )
        spec._shared_depth = depth
    fitted = set(random_effect_level_names(specs))
    for name, spec in hierarchies.items():
        if spec.nested_in is not None and spec.nested_in not in fitted:
            raise ValueError(
                f"Hierarchy {name!r} is declared nested_in={spec.nested_in!r}, which is not a "
                f"random-effect level this model fits; the fitted levels are "
                f"{sorted(fitted, key=str)!r}."
            )


def random_effect_level_names(specs: Mapping[Any, Any]) -> list[Hashable]:
    """Every random-effect level the model fits: RandomEffect features and hierarchy levels."""
    from superglm.features.random_effect import RandomEffect

    names: list[Hashable] = []
    for name, spec in specs.items():
        if isinstance(spec, RandomEffect):
            names.append(name)
        elif isinstance(spec, Hierarchy):
            names.extend(spec.level_name(depth) for depth in spec.fitted_depths)
    return names
