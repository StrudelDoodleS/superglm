"""Which rows each neighbourhood removes from the fit, and which of those it scores.

A neighbourhood ``k`` removes ``drop_rows[drop_ptr[k]:drop_ptr[k + 1]]`` (ascending) and
scores the rows at positions ``score_pos[score_ptr[k]:score_ptr[k + 1]]`` inside them, so a
scored row is a removed row by construction. ``score_ptr is None`` means every removed row is
scored. The ``"rows"`` form (every row removed and scored on its own) stores no arrays.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import scipy.sparse
from numpy.typing import NDArray

from superglm._group_matrix._group_matrix_core import ONE_HOT_CODE_TYPES
from superglm._warn import warn_caller

FEW_NEIGHBOURHOODS = 10
"""Fewer neighbourhoods than this fit with a warning that the criterion is noisy."""

Form = Literal["rows", "groups", "cv", "neighbours"]


@dataclass(frozen=True)
class Neighbourhoods:
    """Removed and scored rows per neighbourhood, in CSR layout; see the module docstring."""

    n_rows: int
    form: Form
    drop_ptr: NDArray[np.intp] | None = None
    drop_rows: NDArray[np.intp] | None = None
    score_ptr: NDArray[np.intp] | None = None
    score_pos: NDArray[np.intp] | None = None
    labels: NDArray | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        for array in (self.drop_ptr, self.drop_rows, self.score_ptr, self.score_pos, self.labels):
            if array is not None:
                array.setflags(write=False)

    @property
    def m(self) -> int:
        """The number of neighbourhoods."""
        return self.n_rows if self.drop_ptr is None else len(self.drop_ptr) - 1

    def sizes(self) -> NDArray[np.intp]:
        """Removed rows per neighbourhood."""
        if self.drop_ptr is None:
            return np.ones(self.n_rows, dtype=np.intp)
        return np.diff(self.drop_ptr)

    def drop(self, k: int) -> NDArray[np.intp]:
        """The rows neighbourhood ``k`` removes, ascending."""
        if self.drop_rows is None or self.drop_ptr is None:
            return np.array([k], dtype=np.intp)
        return self.drop_rows[self.drop_ptr[k] : self.drop_ptr[k + 1]]

    def scored_positions(self, k: int) -> NDArray[np.intp] | None:
        """Positions of the scored rows inside ``drop(k)``; None when every removed row is scored."""
        if self.score_ptr is None or self.score_pos is None:
            return None
        return self.score_pos[self.score_ptr[k] : self.score_ptr[k + 1]]

    def owners(self) -> NDArray[np.intp]:
        """The neighbourhood of each entry of ``drop_rows``."""
        return np.repeat(np.arange(self.m, dtype=np.intp), self.sizes())

    def scored_rows(self) -> NDArray[np.intp]:
        """Every scored row number, neighbourhood by neighbourhood."""
        if self.drop_rows is None or self.drop_ptr is None:
            return np.arange(self.n_rows, dtype=np.intp)
        if self.score_ptr is None or self.score_pos is None:
            return self.drop_rows
        starts = np.repeat(self.drop_ptr[:-1], np.diff(self.score_ptr))
        return self.drop_rows[starts + self.score_pos]

    @property
    def rows_scored(self) -> int:
        """Rows scored at least once."""
        return int(np.count_nonzero(np.bincount(self.scored_rows(), minlength=self.n_rows)))

    @property
    def rows_scored_more_than_once(self) -> int:
        """Rows scored by more than one neighbourhood."""
        return int(np.count_nonzero(np.bincount(self.scored_rows(), minlength=self.n_rows) > 1))

    def describe(self, k: int) -> str:
        """Neighbourhood ``k`` in words, for messages."""
        if self.form == "groups" and self.labels is not None:
            return f"group {self.labels[k : k + 1].tolist()[0]!r}"
        if self.form == "cv":
            return f"split {k} of {self.source}"
        if self.form == "neighbours":
            return f"the neighbourhood of row {k}"
        return f"row {k}"


@dataclass(frozen=True)
class LevelConflict:
    """An unpenalised factor level whose weighted rows all lie in one neighbourhood.

    ``level`` is the code in the group matrix; the reference level has code ``n_levels``.
    ``neighbourhood`` is the first neighbourhood that removes the whole level.
    """

    group_index: int
    term: str
    level: int
    reference: bool
    neighbourhood: int

    def message(self, nb: Neighbourhoods, level_name: object) -> str:
        """The refusal: which level of which term, which neighbourhood, what to do."""
        if self.reference:
            consequence = f"the other levels of {self.term!r} add up to the intercept"
        else:
            consequence = "nothing in the fit estimates that level"
        return (
            f"Every weighted row of level {level_name!r} of {self.term!r} lies in "
            f"{nb.describe(self.neighbourhood)}, so with it held out {consequence}; merge the "
            "level with another or choose neighbourhoods that divide its rows."
        )


# ── constructors ──────────────────────────────────────────────────────────────


def single_rows(n_rows: int) -> Neighbourhoods:
    """Every row removed and scored on its own; no arrays are stored."""
    return Neighbourhoods(n_rows=n_rows, form="rows")


def from_groups(groups: Any, n_rows: int) -> Neighbourhoods:
    """Each distinct label removed and scored in turn, labels in sorted order."""
    values = _group_labels(groups, n_rows)
    labels, codes = np.unique(values, return_inverse=True)
    if len(labels) < 2:
        raise ValueError(
            f"`groups` holds one label only ({labels.tolist()[0]!r}), so holding it out would leave no rows "
            "to fit; pass at least two groups."
        )
    order = np.argsort(codes, kind="stable").astype(np.intp)
    return Neighbourhoods(
        n_rows=n_rows,
        form="groups",
        drop_ptr=_pointer(np.bincount(codes, minlength=len(labels))),
        drop_rows=order,
        labels=labels,
    )


def from_splitter(cv: Any, frame: Any, y: Any, groups: Any = None) -> Neighbourhoods:
    """One neighbourhood per ``(train, test)`` of ``cv.split``, called once.

    Removed rows are the complement of ``train``; scored rows are ``test``.
    """
    if not callable(getattr(cv, "split", None)):
        raise TypeError(
            f"`cv` must be a splitter with a `split` method, such as KFold(5) or GroupKFold(5); "
            f"got {type(cv).__name__}."
        )
    n_rows = len(frame)
    if groups is not None:
        _group_labels(groups, n_rows)
    source = type(cv).__name__
    splits = [
        (
            _split_rows(train, n_rows, s, "training", source),
            _split_rows(test, n_rows, s, "test", source),
        )
        for s, (train, test) in enumerate(cv.split(frame, y, groups))
    ]
    if len(splits) < 2:
        raise ValueError(
            f"`cv` ({source}) gave {len(splits)} splits and fit_ncv needs at least two to hold "
            "out; use a splitter that gives two or more."
        )
    removed, scored = [], []
    for s, (train, test) in enumerate(splits):
        kept = np.zeros(n_rows, dtype=bool)
        kept[train] = True
        overlap = int(np.count_nonzero(kept[test]))
        if overlap:
            raise ValueError(
                f"Split {s} of `cv` ({source}) scores {overlap} rows it also trains on; its test "
                "rows must be left out of its training rows."
            )
        if test.size == 0:
            raise ValueError(
                f"Split {s} of `cv` ({source}) has no test rows; every split must score at least one row."
            )
        rows = np.flatnonzero(~kept).astype(np.intp)
        removed.append(rows)
        scored.append(np.searchsorted(rows, test).astype(np.intp))
    drop_ptr = _pointer(np.array([len(r) for r in removed]))
    every_removed_scored = all(len(q) == len(r) for q, r in zip(scored, removed, strict=True))
    return Neighbourhoods(
        n_rows=n_rows,
        form="cv",
        drop_ptr=drop_ptr,
        drop_rows=np.concatenate(removed),
        score_ptr=None if every_removed_scored else _pointer(np.array([len(q) for q in scored])),
        score_pos=None if every_removed_scored else np.concatenate(scored),
        labels=np.arange(len(splits), dtype=np.intp),
        source=source,
    )


def from_neighbours(neighbours: Any, n_rows: int) -> Neighbourhoods:
    """Row ``i`` removed with ``neighbours[i]`` and scored; ``i`` is added when left out.

    ``neighbours`` is a sequence of ``n_rows`` index arrays, or an ``n_rows x n_rows`` sparse
    matrix whose stored entries in row ``i`` (whatever their value) are row ``i``'s neighbours.
    """
    if scipy.sparse.issparse(neighbours):
        owner, rows = _sparse_neighbours(neighbours, n_rows)
    else:
        owner, rows = _listed_neighbours(neighbours, n_rows)
    every = np.arange(n_rows, dtype=np.intp)
    owner = np.concatenate([owner, every])
    rows = np.concatenate([rows, every])
    order = np.lexsort((rows, owner))
    owner, rows = owner[order], rows[order]
    first = np.ones(len(rows), dtype=bool)
    first[1:] = (owner[1:] != owner[:-1]) | (rows[1:] != rows[:-1])
    owner, rows = owner[first], rows[first]
    drop_ptr = _pointer(np.bincount(owner, minlength=n_rows))
    return Neighbourhoods(
        n_rows=n_rows,
        form="neighbours",
        drop_ptr=drop_ptr,
        drop_rows=rows,
        score_ptr=np.arange(n_rows + 1, dtype=np.intp),
        score_pos=(np.flatnonzero(owner == rows) - drop_ptr[:-1]).astype(np.intp),
    )


def build_neighbourhoods(
    n_rows: int,
    active: NDArray[np.bool_],
    *,
    frame: Any = None,
    y: Any = None,
    groups: Any = None,
    cv: Any = None,
    neighbours: Any = None,
) -> Neighbourhoods:
    """The neighbourhoods ``fit_ncv``'s arguments ask for, validated against the weighted rows.

    ``active`` marks the rows of positive weight. Warns when there are fewer than
    ``FEW_NEIGHBOURHOODS`` neighbourhoods.
    """
    if cv is not None and neighbours is not None:
        raise ValueError(
            "fit_ncv takes `cv` or `neighbours`, not both; pass `cv` for folds or `neighbours` "
            "for the rows removed with each row."
        )
    if groups is not None and neighbours is not None:
        raise ValueError(
            "`groups` cannot be combined with `neighbours`, which already says which rows are "
            "removed with each row; drop `groups`, or pass `groups` alone to hold out each group in turn."
        )
    if cv is not None:
        nb = from_splitter(cv, frame, y, groups)
    elif neighbours is not None:
        nb = from_neighbours(neighbours, n_rows)
    elif groups is not None:
        nb = from_groups(groups, n_rows)
    else:
        nb = single_rows(n_rows)
    check_leaves_weighted_rows(nb, active)
    warn_few_neighbourhoods(nb)
    return nb


# ── checks ────────────────────────────────────────────────────────────────────


def check_leaves_weighted_rows(nb: Neighbourhoods, active: NDArray[np.bool_]) -> None:
    """Refuse a neighbourhood that removes every row of positive weight."""
    total = int(np.count_nonzero(active))
    if nb.drop_rows is None:
        removed = active.astype(np.intp)
    else:
        removed = np.bincount(nb.owners()[active[nb.drop_rows]], minlength=nb.m)
    whole = np.flatnonzero(removed == total)
    if whole.size:
        where = nb.describe(int(whole[0]))
        raise ValueError(
            f"{where[0].upper()}{where[1:]} removes every row with positive weight, "
            "leaving nothing to fit; choose neighbourhoods that leave weighted rows in the fit."
        )


def warn_few_neighbourhoods(nb: Neighbourhoods) -> None:
    """Warn when there are fewer than ``FEW_NEIGHBOURHOODS`` neighbourhoods."""
    if nb.m < FEW_NEIGHBOURHOODS:
        warn_caller(
            f"fit_ncv has {nb.m} neighbourhoods; with fewer than {FEW_NEIGHBOURHOODS} the "
            "held-out criterion is noisy (a rule of thumb, not a measured threshold), so the "
            "chosen smoothing can move between splits. Use more groups or folds where you can."
        )


def unidentified_levels(
    nb: Neighbourhoods,
    dm: Any,
    groups: Sequence[Any],
    active: NDArray[np.bool_],
    penalised: Collection[int],
) -> list[LevelConflict]:
    """Unpenalised one-hot levels that some neighbourhood removes whole.

    A neighbourhood conflicts with a level when its count of positive-weight rows at that
    level equals the level's count of positive-weight rows; integer counts only. The
    reference level (the sink code ``n_levels``) counts as a level. ``groups[g].name`` names
    the term of ``dm.group_matrices[g]``; ``penalised`` holds the indices of penalised groups.
    One conflict per level, at its first neighbourhood.
    """
    found: list[LevelConflict] = []
    for g, gm in enumerate(dm.group_matrices):
        if type(gm) in ONE_HOT_CODE_TYPES and g not in penalised:
            found.extend(_block_conflicts(nb, gm, active, g, str(groups[g].name)))
    return found


# ── helpers ───────────────────────────────────────────────────────────────────


def _block_conflicts(
    nb: Neighbourhoods, gm: Any, active: NDArray[np.bool_], g: int, term: str
) -> list[LevelConflict]:
    codes = np.asarray(gm.codes, dtype=np.intp)
    n_codes = int(gm.n_levels) + 1
    total = np.bincount(codes[active], minlength=n_codes)
    if nb.drop_rows is None:
        rows = np.flatnonzero(active & (total[codes] == 1))
        level, owner = codes[rows], rows
    else:
        kept = active[nb.drop_rows]
        key = nb.owners()[kept].astype(np.int64) * n_codes + codes[nb.drop_rows[kept]]
        pairs, counts = np.unique(key, return_counts=True)
        whole = pairs[counts == total[pairs % n_codes]]
        level, first = np.unique(whole % n_codes, return_index=True)
        owner = whole[first] // n_codes
    order = np.argsort(level, kind="stable")
    return [
        LevelConflict(g, term, int(lv), int(lv) == int(gm.n_levels), int(k))
        for lv, k in zip(level[order], owner[order], strict=True)
    ]


def _pointer(counts: NDArray) -> NDArray[np.intp]:
    ptr = np.zeros(len(counts) + 1, dtype=np.intp)
    np.cumsum(counts, out=ptr[1:])
    return ptr


def _group_labels(groups: Any, n_rows: int) -> NDArray:
    values = np.asarray(groups)
    if values.shape != (n_rows,):
        raise ValueError(
            f"`groups` must hold one label per row: it has shape {values.shape} for {n_rows} rows."
        )
    missing = np.flatnonzero(pd.isna(values))
    if missing.size:
        raise ValueError(
            f"`groups` has {missing.size} missing labels, the first at row {int(missing[0])}; "
            "give every row a group."
        )
    return values


def _split_rows(indices: Any, n_rows: int, s: int, which: str, source: str) -> NDArray[np.intp]:
    values = np.asarray(indices).reshape(-1)
    if values.size and values.dtype.kind not in "iu":
        raise ValueError(
            f"Split {s} of `cv` ({source}) gives {which} rows of type {values.dtype}; a splitter "
            "must give integer row positions."
        )
    outside = np.flatnonzero((values < 0) | (values >= n_rows))
    if outside.size:
        raise ValueError(
            f"Split {s} of `cv` ({source}) gives the {which} row {int(values[outside[0]])}, which "
            f"is not a row position from 0 to {n_rows - 1}; the splitter must index the rows of X."
        )
    return np.unique(values).astype(np.intp)


def _listed_neighbours(neighbours: Any, n_rows: int) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    if len(neighbours) != n_rows:
        raise ValueError(
            f"`neighbours` has {len(neighbours)} entries for {n_rows} rows; give one entry per "
            "row of X, the rows removed with that row."
        )
    parts = [np.asarray(entry).reshape(-1) for entry in neighbours]
    typed = [i for i, part in enumerate(parts) if part.size and part.dtype.kind not in "iu"]
    if typed:
        raise ValueError(
            f"`neighbours[{typed[0]}]` holds values of type {parts[typed[0]].dtype}; give integer "
            "row positions."
        )
    lengths = np.array([part.size for part in parts], dtype=np.intp)
    rows = np.concatenate([np.zeros(0, dtype=np.intp), *parts]).astype(np.int64)
    owner = np.repeat(np.arange(n_rows, dtype=np.intp), lengths)
    outside = np.flatnonzero((rows < 0) | (rows >= n_rows))
    if outside.size:
        at = int(outside[0])
        raise ValueError(
            f"`neighbours[{int(owner[at])}]` holds {int(rows[at])}, which is not a row position "
            f"from 0 to {n_rows - 1}; use positions of rows in X."
        )
    return owner, rows.astype(np.intp)


def _sparse_neighbours(neighbours: Any, n_rows: int) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    if neighbours.shape != (n_rows, n_rows):
        shape = " x ".join(str(d) for d in neighbours.shape)
        raise ValueError(
            f"`neighbours` is a {shape} matrix for {n_rows} rows; pass an {n_rows} x {n_rows} "
            "matrix whose row i marks the rows removed with row i."
        )
    csr = scipy.sparse.csr_array(neighbours)
    owner = np.repeat(np.arange(n_rows, dtype=np.intp), np.diff(csr.indptr))
    return owner, csr.indices.astype(np.intp)
