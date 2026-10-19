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

import math
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
# The key of a part whose folded form no training label holds: equal only to
# itself, so a path holding it matches no fitted path.
_UNLABELLED = object()

ColumnReader = Callable[[Hashable], NDArray]


def normalize_label(value: Any) -> Any:
    """The comparison form of one text label; other values are returned unchanged.

    Two labels compare equal exactly when they are a *compatibility caseless
    match* (the Unicode Standard, section 3.13, definition D146):
    ``NFKD(toCasefold(NFKD(toCasefold(NFKD(toCasefold(NFD(X)))))))``, with
    ``str.casefold`` as ``toCasefold``.  That folds case and compatibility
    characters (full-width letters, ligatures, no-break spaces).  NFC then
    recomposes the result for display, which keeps those equalities because
    every input to it is already decomposed.  ``str.split()`` finally strips
    the ends and collapses every internal whitespace run to one space.
    """
    if not isinstance(value, str):
        return value
    folded = unicodedata.normalize("NFD", value)
    for _ in range(3):
        folded = unicodedata.normalize("NFKD", folded.casefold())
    return " ".join(unicodedata.normalize("NFC", folded).split())


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
            f"level in fitted names; got {names!r}. Pass names= to name them."
        )
    if len(set(names)) != len(names):
        raise ValueError(f"Hierarchy level names must be distinct, got {names!r}")
    return names


class HierarchyLevel:
    """One fitted level of a :class:`Hierarchy`, read as the random effect it is.

    ``random_effects()`` reports a level through the attributes it reads from
    a ``RandomEffect``: ``_levels`` (here, paths), ``_prediction_codes`` (each
    row's path at this level, ``-1`` if unseen), ``unseen`` and
    ``unseen_contrast()``.
    """

    def __init__(self, hierarchy: Hierarchy, depth: int):
        self._hierarchy = hierarchy
        self._depth = depth

    @property
    def _levels(self) -> list[tuple]:
        return self._hierarchy._paths[self._depth]

    @property
    def unseen(self) -> str:
        return self._hierarchy.unseen

    def _prediction_codes(self, x) -> NDArray[np.intp]:
        return self._hierarchy._prediction_codes(x)[self._depth]

    def unseen_contrast(self) -> NDArray[np.floating]:
        return self._hierarchy.unseen_contrast(self._depth)


class Hierarchy:
    """Nested random intercepts for a hierarchy of levels, such as make, model, year.

    The hierarchy fits one REML variance component per level, exactly as
    ``RandomEffect`` does for one factor.  Level ``k`` groups the rows by
    their first ``k`` parts -- a model is identified by make and model, a
    model year by make, model and year -- so a model name reused under two
    makes is two models, and the levels nest by construction.  A thin model
    shrinks toward its make, and a thin make toward the population.

    Declare the parts in one of two ways:

    - ``Hierarchy(names=("make", "model", "year"), sep="-")`` reads the
      feature's own column, whose values join the parts with ``sep``
      (``"Ford-Focus-2019"``).  Every value must split into exactly
      ``len(names)`` parts; a value that does not is an error naming it,
      because a separator inside a name cannot be split unambiguously.
    - ``Hierarchy(columns=("make", "model", "year"))`` reads one column per
      level, top level first, and needs no separator.  ``names=`` may rename
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
    prediction.  A merged label is reported in the spelling most training
    rows use (the first seen on a tie), so ``"Ford"`` stays ``"Ford"``, and
    every merge of distinct spellings is reported: in a warning at fit and
    under ``"merged_labels"`` in ``reconstruct_feature()``.  Values that are
    not text (a numeric year) are used as they are.

    A row whose path was not seen in training keeps the effect of every level
    it shares with the training rows: a new model of a known make gets that
    make's effect.  Each level it does not share gives

    - ``unseen="population"`` (the default): zero, the population value, as
      lme4's ``allow.new.levels`` and mgcv's ``bs="re"`` predict a new level;
    - ``unseen="weighted"``: the mean of that level's fitted effects under
      the deepest part the path shares with training, each weighted by its
      training weight (``sample_weight``, or rows), so a new model of a known
      make is priced as that make's average exposure is, and a new make as
      the book's;
    - ``unseen="error"``: a refusal of any row whose full path is new.

    ``random_effects()`` reports, per level, what a path new from the top
    takes there and its standard error (``RandomEffectResult``).  Missing
    parts are an error.

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
        names: Sequence[str] | None = None,
        sep: str | None = None,
        columns: Sequence[str] | None = None,
        nested_in: Hashable | None = None,
        normalize: bool = False,
        unseen: Literal["population", "weighted", "error"] = "population",
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
            if names is None:
                raise ValueError(
                    "Hierarchy(sep=...) needs names=, the name of each part from the top "
                    "level down, e.g. names=('make', 'model', 'year')."
                )
            level_names = _level_names(names, "names")
            source = None
        else:
            if isinstance(columns, str):
                raise TypeError("columns= takes a sequence of column names, not one string")
            source = tuple(columns or ())
            if any(not isinstance(column, str) or not column.strip() for column in source):
                raise ValueError(f"Hierarchy columns= must hold column names, got {source!r}")
            if len(set(source)) != len(source):
                raise ValueError(f"Hierarchy columns must be distinct, got {source!r}")
            level_names = _level_names(source if names is None else names, "names")
            if len(level_names) != len(source):
                raise ValueError(
                    f"names= lists {len(level_names)} level(s) but columns= lists {len(source)}"
                )
        if len(level_names) < 2:
            raise ValueError(
                "A Hierarchy needs at least two levels; one grouping column is a RandomEffect()."
            )
        if nested_in is not None and not isinstance(nested_in, str):
            raise TypeError(
                "nested_in must name a level another term fits, such as 'vehicle:make', or be None"
            )
        if not isinstance(normalize, bool):
            raise TypeError(f"normalize must be True or False, got {normalize!r}")
        if unseen not in ("population", "weighted", "error"):
            raise ValueError(f"unseen must be 'population', 'weighted' or 'error', got {unseen!r}")
        if lambda_policy is not None and not isinstance(lambda_policy, LambdaPolicy):
            raise TypeError("lambda_policy must be a LambdaPolicy or None")

        self.names = level_names
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
        # Folded form -> label per level (normalize=True), and each fitted
        # depth's training weight per path (unseen="weighted").
        self._labels: list[dict] | None = None
        self._weights: dict[int, NDArray[np.floating]] = {}

    def __repr__(self) -> str:
        source = f"sep={self.sep!r}" if self.columns is None else f"columns={self.columns!r}"
        return f"Hierarchy(names={self.names!r}, {source})"

    # ── Reading the parts ────────────────────────────────────────

    def source_columns(self, name: Hashable) -> tuple[Hashable, ...]:
        """The input columns this feature reads when it is named *name*."""
        return (name,) if self.columns is None else self.columns

    def read_column(self, column: ColumnReader, name: Hashable) -> NDArray:
        """The feature's values: its joined column, or an ``(n, len(names))`` array of parts."""
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
            if len(parts) != len(self.names)
        ]
        if wrong:
            shown = ", ".join(f"{value!r} ({count} parts)" for value, count in wrong[:_SHOWN])
            raise ValueError(
                f"Hierarchy with sep={sep!r} needs every value to split into exactly "
                f"{len(self.names)} parts ({', '.join(self.names)}), but {len(wrong)} of "
                f"{len(joined)} distinct values do not: {shown}. A separator inside a name "
                "makes the split ambiguous, and it is never guessed: store the parts in "
                "separate columns and use Hierarchy(columns=(...)), or join them with a "
                "separator that no name contains."
            )
        raw = np.empty((len(pieces), len(self.names)), dtype=object)
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

    @staticmethod
    def _spelling_labels(codes: NDArray[np.intp], raw: NDArray, folded: NDArray) -> list[dict]:
        """Per level, each folded form's label: its spelling on the most rows, first seen on a tie."""
        rows = np.bincount(codes, minlength=raw.shape[0]).tolist()
        labels = []
        for position in range(raw.shape[1]):
            tallies: dict[Any, dict[Any, int]] = {}
            for given, compared, count in zip(
                raw[:, position], folded[:, position], rows, strict=True
            ):
                tally = tallies.setdefault(compared, {})
                tally[given] = tally.get(given, 0) + count
            # ``max`` keeps the first of equal counts, and the distinct paths
            # are in the order the rows first show them.
            labels.append(
                {compared: max(tally, key=tally.__getitem__) for compared, tally in tallies.items()}
            )
        return labels

    def _paths_of(
        self, x, labels: list[dict] | None = None
    ) -> tuple[NDArray[np.intp], NDArray, NDArray]:
        """Row path codes, the distinct paths as given, and their keys.

        Without ``normalize`` a part is its own key.  With it, parts compare in
        folded form and are keyed by their label, the spelling most rows use,
        so reports read as the data does.  *labels* are the fit's at
        prediction; ``None`` reads them from these rows.  A folded form the
        labels do not hold keys as ``_UNLABELLED``, which matches no path.
        """
        codes, raw = self._raw_paths(x)
        return codes, raw, self._keys(codes, raw, labels)

    def _keys(self, codes: NDArray[np.intp], raw: NDArray, labels: list[dict] | None) -> NDArray:
        if not self.normalize:
            return raw
        folded = self._folded(raw)
        if labels is None:
            labels = self._spelling_labels(codes, raw, folded)
        keys = np.empty_like(folded)
        for position, mapping in enumerate(labels):
            keys[:, position] = [mapping.get(value, _UNLABELLED) for value in folded[:, position]]
        return keys

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
        return range(self._shared_depth, len(self.names))

    def level_name(self, depth: int) -> str:
        """The fitted term name of one level, ``"<feature>:<level>"``."""
        return f"{self._name}:{self.names[depth]}"

    def _level_view(self, level: str) -> HierarchyLevel:
        """The fitted level named *level* (its bare name, such as ``"model"``)."""
        depth = self.names.index(level)
        if depth not in self._paths:
            raise KeyError(f"Hierarchy level {level!r} is not fitted by this term")
        return HierarchyLevel(self, depth)

    # ── Build, score, report ─────────────────────────────────────

    def training_codes(self, x) -> list[tuple[NDArray[np.intp], list[tuple]]]:
        """Each row's code and the paths at every depth, top first, without fitting."""
        codes, _, keys = self._paths_of(x)
        return [(depth_codes[codes], paths) for depth_codes, paths in self._depth_codes(keys)]

    def build(self, x, sample_weight: NDArray | None = None) -> list[GroupInfo]:
        """One all-level random-effect block per fitted depth, top first.

        Each fitted path's training weight (*sample_weight*, or one per row) is
        kept for ``unseen="weighted"``.
        """
        codes, raw = self._raw_paths(x)
        self._labels = (
            self._spelling_labels(codes, raw, self._folded(raw)) if self.normalize else None
        )
        keys = self._keys(codes, raw, self._labels)
        self._merged = self._merges(raw, keys) if self.normalize else {}
        if self._merged:
            warnings.warn(self._merge_message(), UserWarning, stacklevel=2)
        row_weight = (
            np.ones(len(codes), dtype=np.float64)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float64).ravel()
        )
        depths = self._depth_codes(keys)
        self._paths = {}
        self._path_codes = {}
        self._weights = {}
        infos = []
        policies = None if self._lambda_policy is None else {"_default": self._lambda_policy}
        for depth in self.fitted_depths:
            depth_codes, paths = depths[depth]
            self._paths[depth] = paths
            self._path_codes[depth] = {path: index for index, path in enumerate(paths)}
            self._weights[depth] = np.bincount(
                depth_codes[codes], weights=row_weight, minlength=len(paths)
            )
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
                    subgroup_name=self.names[depth],
                    structured_kind="random_effect",
                    random_effect_nested_in=parent,
                )
            )
        return infos

    def _merges(self, raw: NDArray, keys: NDArray) -> dict[str, dict[Any, tuple]]:
        merged: dict[str, dict[Any, tuple]] = {}
        for position, level in enumerate(self.names):
            spellings: dict[Any, set] = {}
            for given, label in zip(raw[:, position], keys[:, position], strict=True):
                spellings.setdefault(label, set()).add(given)
            found = {
                label: tuple(sorted(given, key=str))
                for label, given in spellings.items()
                if len(given) > 1
            }
            if found:
                merged[level] = dict(sorted(found.items(), key=lambda item: str(item[0])))
        return merged

    def _merge_message(self) -> str:
        entries = [
            f"{level} {label!r} <- {', '.join(repr(value) for value in given)}"
            for level, found in self._merged.items()
            for label, given in found.items()
        ]
        shown = "; ".join(entries[:_SHOWN])
        more = f"; and {len(entries) - _SHOWN} more" if len(entries) > _SHOWN else ""
        return (
            f"Hierarchy normalize=True merged {len(entries)} group(s) of differently spelled "
            f"labels into one label each, reported as the spelling most rows use: "
            f"{shown}{more}. reconstruct_feature() lists them all under 'merged_labels'."
        )

    def validate_prediction_values(self, x) -> None:
        """Reject missing or unsplittable values without applying the unseen policy."""
        self._raw_paths(x)

    def _coded(self, x) -> tuple[NDArray[np.intp], NDArray, dict[int, NDArray[np.intp]]]:
        """Row path codes, the distinct paths' keys, and each one's code at every fitted depth."""
        codes, raw, keys = self._paths_of(x, self._labels)
        lookups = {}
        for depth in self.fitted_depths:
            known = self._path_codes[depth]
            lookups[depth] = np.fromiter(
                (known.get(tuple(path[: depth + 1]), -1) for path in keys),
                dtype=np.intp,
                count=len(keys),
            )
        leaf = lookups[len(self.names) - 1]
        if self.unseen == "error" and np.any(leaf < 0):
            unseen = [tuple(path) for path in raw[leaf < 0][:_SHOWN]]
            raise ValueError(f"Encountered unseen Hierarchy paths: {unseen}.")
        return codes, keys, lookups

    def _prediction_codes(self, x) -> dict[int, NDArray[np.intp]]:
        """Each row's code at every fitted depth; ``-1`` where the path was not fitted."""
        codes, _, lookups = self._coded(x)
        return {depth: lookup[codes] for depth, lookup in lookups.items()}

    def _shares(self, depth: int) -> dict[tuple, tuple[NDArray[np.intp], NDArray[np.floating]]]:
        """Under every training prefix above *depth*, its paths and their shares of the weight.

        The empty prefix holds all of the depth's paths.  A prefix whose paths
        carry no weight gets zero shares, the population value.
        """
        weights = self._weights[depth]
        members: dict[tuple, list[int]] = {}
        for index, path in enumerate(self._paths[depth]):
            for length in range(depth + 1):
                members.setdefault(path[:length], []).append(index)
        shares = {}
        for prefix, indices in members.items():
            positions = np.asarray(indices, dtype=np.intp)
            total = math.fsum(weights[positions].tolist())
            shares[prefix] = (
                positions,
                weights[positions] / total if total > 0.0 else np.zeros(len(positions)),
            )
        return shares

    def _unseen_contrasts(
        self, depth: int, keys: NDArray, lookup: NDArray[np.intp]
    ) -> list[tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.floating]]]:
        """``unseen="weighted"``: the distinct paths new at *depth*, by the prefix they share.

        Each entry is ``(distinct paths, member paths, shares)``: the new paths
        take the shares-weighted mean of the depth's effects under the
        longest prefix they share with the training paths, all of them when
        they share none.
        """
        new = np.flatnonzero(lookup < 0)
        if self.unseen != "weighted" or not new.size:
            return []
        shares = self._shares(depth)
        grouped: dict[tuple, list[int]] = {}
        for index in new.tolist():
            path = tuple(keys[index, : depth + 1])
            length = depth
            while path[:length] not in shares:
                length -= 1
            grouped.setdefault(path[:length], []).append(index)
        return [
            (np.asarray(indices, dtype=np.intp), *shares[prefix])
            for prefix, indices in grouped.items()
        ]

    def _offsets(self) -> dict[int, int]:
        offsets, start = {}, 0
        for depth in self.fitted_depths:
            offsets[depth] = start
            start += len(self._paths[depth])
        return offsets

    def level_scores(self, x, beta: NDArray[np.floating]) -> dict[int, NDArray[np.floating]]:
        """Each fitted depth's effect on every row, top first, with the unseen rule applied.

        A depth the row's path shares with training gives its fitted effect.
        A new one gives zero (``unseen="population"``) or, under
        ``unseen="weighted"``, the training-weighted mean of the depth's
        effects under the longest prefix the path shares with training.
        """
        beta = np.asarray(beta, dtype=np.float64).ravel()
        offsets = self._offsets()
        codes, keys, lookups = self._coded(x)
        out = {}
        for depth, lookup in lookups.items():
            block = beta[offsets[depth] : offsets[depth] + len(self._paths[depth])]
            values = np.zeros(len(lookup), dtype=np.float64)
            known = lookup >= 0
            values[known] = block[lookup[known]]
            for paths, members, shares in self._unseen_contrasts(depth, keys, lookup):
                values[paths] = shares @ block[members]
            out[depth] = values[codes]
        return out

    def score(self, x, beta: NDArray[np.floating]) -> NDArray[np.floating]:
        """Add each row's effect at every fitted depth (``level_scores``)."""
        effects = None
        for block in self.level_scores(x, beta).values():
            effects = block if effects is None else effects + block
        assert effects is not None
        return effects

    def transform(self, x) -> NDArray[np.floating]:
        """Materialize the all-level columns of every fitted depth (shares where a path is new)."""
        offsets = self._offsets()
        width = sum(len(paths) for paths in self._paths.values())
        codes, keys, lookups = self._coded(x)
        rows = np.zeros((len(keys), width), dtype=np.float64)
        for depth, lookup in lookups.items():
            known = np.flatnonzero(lookup >= 0)
            rows[known, offsets[depth] + lookup[known]] = 1.0
            for paths, members, shares in self._unseen_contrasts(depth, keys, lookup):
                rows[np.ix_(paths, offsets[depth] + members)] = shares
        return rows[codes]

    def unseen_contrast(self, depth: int) -> NDArray[np.floating]:
        """The weights over a depth's paths that a path new from the top level takes there.

        Zero under ``unseen="population"`` (and ``"error"``, which refuses
        such a path); the paths' shares of the training weight under
        ``unseen="weighted"``.
        """
        contrast = np.zeros(len(self._paths[depth]), dtype=np.float64)
        if self.unseen == "weighted":
            members, shares = self._shares(depth)[()]
            contrast[members] = shares
        return contrast

    def leaf_coefficients(self) -> NDArray[np.intp]:
        """``(n_paths, n_fitted_depths)`` positions in the term's coefficients of each path's levels."""
        offsets = self._offsets()
        leaf_depth = len(self.names) - 1
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
        leaves = self._paths[len(self.names) - 1]
        total = beta[self.leaf_coefficients()].sum(axis=1)
        effects = {path: float(value) for path, value in zip(leaves, total, strict=True)}
        offsets = self._offsets()
        by_level = {}
        for depth in self.fitted_depths:
            paths = self._paths[depth]
            block = beta[offsets[depth] : offsets[depth] + len(paths)]
            by_level[self.names[depth]] = {
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
        for position, level in enumerate(spec.names)
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
        if depth >= len(spec.names):
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
