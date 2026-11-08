"""Builder-owned immutable code correspondence, independent of weighted products."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class _Codes:
    values: np.ndarray
    backing: bytes

    @classmethod
    def own(cls, values):
        backing = np.asarray(values, dtype=np.intp).tobytes()
        return cls(np.frombuffer(backing, dtype=np.intp), backing)

    def matches(self, values) -> bool:
        return (
            type(values) is np.ndarray
            and values is self.values
            and values.base is self.backing
            and values.dtype == np.dtype(np.intp)
            and values.shape == (len(self.backing) // np.dtype(np.intp).itemsize,)
            and values.strides == (np.dtype(np.intp).itemsize,)
            and not values.flags.writeable
        )


@dataclass(frozen=True)
class _LevelCodes:
    full: _Codes
    rows: _Codes
    local: _Codes
    n_bins: int

    def matches(self, group) -> bool:
        return (
            self.full.matches(self.full.values)
            and self.rows.matches(group.row_idx)
            and self.local.matches(group.bin_idx_level)
            and group.n_bins == self.n_bins
            and group.shape[0] == group.n_rows == len(self.full.values)
        )


def certify_binned_codes(groups, n: int) -> None:
    """Certify exact row-code equality once per scalar raw design.

    Canonical full-row operands reroute byte-identical histograms through the
    existing per-weighted-assembly cache. A level's correspondence is checked
    against its ACTUAL local codes, including any separation clipping. Only
    equal codes with equal bin dimensions qualify; feature labels play no role.
    The immutable operands survive basis-only rebuilds. A subset, row reorder,
    new design, replaced operand or pickle restore needs a fresh certificate.
    """
    from ._group_matrix_discretized import (
        DiscretizedSplineCategoricalGroupMatrix,
        DiscretizedSSPGroupMatrix,
        DiscretizedTensorGroupMatrix,
        SupportCompressedSplineCategoricalGroupMatrix,
    )

    if not any(type(group) is DiscretizedTensorGroupMatrix for group in groups):
        return
    pool: list[tuple[int, _Codes]] = []

    def canonical(values, bins):
        if (
            type(values) is not np.ndarray
            or values.dtype != np.dtype(np.intp)
            or values.shape != (n,)
            or np.any(values < 0)
            or np.any(values >= bins)
        ):
            return None
        for size, codes in pool:
            if size == bins and np.array_equal(values, codes.values):
                return codes
        codes = _Codes.own(values)
        pool.append((bins, codes))
        return codes

    for group in groups:
        if type(group) is DiscretizedTensorGroupMatrix:
            first = canonical(group.idx1, group.n_bins1)
            second = canonical(group.idx2, group.n_bins2)
            if first is not None and second is not None:
                group.idx1, group.idx2 = first.values, second.values
                group._margin_codes = (first, second)
        elif type(group) is DiscretizedSSPGroupMatrix:
            codes = canonical(group.bin_idx, group.n_bins)
            if codes is not None:
                group.bin_idx = codes.values

    level_types = (
        DiscretizedSplineCategoricalGroupMatrix,
        SupportCompressedSplineCategoricalGroupMatrix,
    )
    for group in groups:
        if type(group) not in level_types or group.shape[0] != n:
            continue
        rows = group.row_idx
        if rows.ndim != 1 or np.any(rows < 0) or np.any(rows >= n):
            continue
        for bins, full in pool:
            if bins != group.n_bins or not np.array_equal(full.values[rows], group.bin_idx_level):
                continue
            # A partition certificate may already own these exact immutable rows.
            if type(rows.base) is bytes and rows.dtype == np.dtype(np.intp):
                owned_rows = _Codes(rows, rows.base)
            else:
                owned_rows = _Codes.own(rows)
            local = _Codes.own(group.bin_idx_level)
            group.row_idx, group.bin_idx_level = owned_rows.values, local.values
            group._bin_code_source = _LevelCodes(full, owned_rows, local, bins)
            break


def carry_binned_codes(source, target) -> None:
    """Builder-only handoff after constructing target with source's raw operands."""
    level = getattr(source, "_bin_code_source", None)
    if type(level) is _LevelCodes and level.matches(source):
        target.row_idx = source.row_idx
        target.bin_idx_level = source.bin_idx_level
        target._bin_code_source = level
    margins = getattr(source, "_margin_codes", None)
    if margins is not None and all(
        codes.matches(values) for codes, values in zip(margins, (source.idx1, source.idx2))
    ):
        target._margin_codes = margins


def shared_level_margin(tensor, level) -> int | None:
    """Verified shared margin, in O(1) without re-reading training rows."""
    from ._group_matrix_discretized import (
        DiscretizedSplineCategoricalGroupMatrix,
        DiscretizedTensorGroupMatrix,
        SupportCompressedSplineCategoricalGroupMatrix,
    )

    if type(tensor) is not DiscretizedTensorGroupMatrix or type(level) not in (
        DiscretizedSplineCategoricalGroupMatrix,
        SupportCompressedSplineCategoricalGroupMatrix,
    ):
        return None
    source = getattr(level, "_bin_code_source", None)
    margins = tensor._margin_codes
    if type(source) is not _LevelCodes or not source.matches(level) or margins is None:
        return None
    if not all(codes.matches(values) for codes, values in zip(margins, (tensor.idx1, tensor.idx2))):
        return None
    if tensor.shape[0] != level.shape[0]:
        return None
    for index, (codes, bins) in enumerate(zip(margins, (tensor.n_bins1, tensor.n_bins2)), 1):
        if codes is source.full and bins == level.n_bins:
            return index
    return None
