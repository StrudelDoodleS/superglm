"""Construction-time certificates for disjoint categorical row supports."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _level_types():
    from ._group_matrix_core import SplineCategoricalGroupMatrix
    from ._group_matrix_discretized import (
        DiscretizedSplineCategoricalGroupMatrix,
        SupportCompressedSplineCategoricalGroupMatrix,
    )

    return (
        SplineCategoricalGroupMatrix,
        DiscretizedSplineCategoricalGroupMatrix,
        SupportCompressedSplineCategoricalGroupMatrix,
    )


@dataclass(frozen=True)
class _LevelSupport:
    partition: object
    level: int
    rows: np.ndarray
    backing: bytes
    n: int

    def matches(self, group) -> bool:
        rows = group.row_idx
        return (
            type(group) in _level_types()
            and type(rows) is np.ndarray
            and rows is self.rows
            and rows.base is self.backing
            and rows.dtype == np.dtype(np.intp)
            and rows.shape == (len(self.backing) // np.dtype(np.intp).itemsize,)
            and rows.strides == (np.dtype(np.intp).itemsize,)
            and not rows.flags.writeable
            and group.shape[0] == group.n_rows == self.n
        )


def certify_level_partitions(groups, n: int) -> None:
    """Certify actual supports once, at the scalar design builder boundary.

    Labels only propose partitions. Equal labels must have equal rows; different
    labels must have disjoint rows, tested against one transient n-byte bitmap.
    On refusal the entire candidate keeps its ordinary intersection path. On
    acceptance equal supports share immutable bytes-backed indices, whose
    identity, backing, shape and dtype can subsequently be checked in O(1).
    Nothing depends on weights, basis, penalty, transforms or float precision.
    Subsets, row reorders and pickle restores do not inherit the certificate.
    """
    candidates = {}
    for group in groups:
        if type(group) not in _level_types():
            continue
        parent, level = group.spline_cat_feature, group.spline_cat_level
        if parent is not None and level is not None:
            candidates.setdefault(parent, {}).setdefault(level, []).append(group)
    for levels in candidates.values():
        if len(levels) < 2:
            continue
        seen = np.zeros(n, dtype=bool)
        members = []
        for same_level in levels.values():
            rows = same_level[0].row_idx
            if (
                rows.dtype != np.dtype(np.intp)
                or rows.ndim != 1
                or (rows.size and (rows[0] < 0 or rows[-1] >= n))
                or np.any(rows[1:] <= rows[:-1])
                or any(group.shape[0] != n for group in same_level)
                or any(not np.array_equal(rows, group.row_idx) for group in same_level[1:])
                or np.any(seen[rows])
            ):
                break
            seen[rows] = True
            members.append((same_level, rows))
        else:
            partition = object()
            for level, (same_level, rows) in enumerate(members):
                backing = rows.tobytes()
                owned = np.frombuffer(backing, dtype=np.intp)
                support = _LevelSupport(partition, level, owned, backing, n)
                for group in same_level:
                    group.row_idx = owned
                    group._level_support = support


def carry_level_support(source, target) -> None:
    """Carry only at a builder's basis-only rebuild using source.row_idx verbatim.

    The target constructor just copied those exact rows. Replacing that copy
    with the certified immutable storage saves another retained row array.
    """
    support = getattr(source, "_level_support", None)
    if type(support) is _LevelSupport and support.matches(source):
        target.row_idx = source.row_idx
        target._level_support = support


def certified_disjoint(left, right) -> bool:
    """A zero RAW cross block; centered mean/intercept corrections still apply."""
    a = getattr(left, "_level_support", None)
    b = getattr(right, "_level_support", None)
    return (
        type(a) is _LevelSupport
        and type(b) is _LevelSupport
        and a.partition is b.partition
        and a.level != b.level
        and a.matches(left)
        and b.matches(right)
    )
