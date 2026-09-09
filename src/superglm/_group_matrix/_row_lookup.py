"""Immutable row-order lookup storage with constant-time range certification."""

from __future__ import annotations

import weakref

import numpy as np


class _RowLookupCertificate:
    """Certify ordering, without retaining replaced arrays or numeric design data."""

    __slots__ = ("_rows", "_order", "_rows_base", "_order_base", "_size", "__weakref__")

    def __init__(self, rows, order):
        self._rows_base = rows.base
        self._order_base = None if order is None else order.base
        self._size = rows.size
        self_ref = weakref.ref(self)

        def released(_):
            certificate = self_ref()
            if certificate is not None:
                # Bytes cannot be weak-referenced. Drop these last possible
                # owners as soon as either certified array is released.
                certificate._rows_base = None
                certificate._order_base = None

        self._rows = weakref.ref(rows, released)
        self._order = None if order is None else weakref.ref(order, released)

    def matches(self, rows, order):
        def same(array, reference, base):
            return (
                type(array) is np.ndarray
                and reference() is array
                and type(base) is bytes
                and array.base is base
                and array.dtype == np.dtype(np.intp)
                and array.shape == (self._size,)
                and array.strides == (np.dtype(np.intp).itemsize,)
                and not array.flags.writeable
            )

        matches = same(rows, self._rows, self._rows_base) and (
            order is None if self._order is None else same(order, self._order, self._order_base)
        )
        if not matches:
            # __setstate__ can replace storage while preserving ndarray
            # identity. Refusal must release its abandoned backing bytes too.
            self._rows_base = None
            self._order_base = None
        return matches


def build_row_lookup(row_idx, *, with_order):
    """Build the existing lazy lookup, certifying only exact immutable arrays.

    Custom arrays retain the generic sorting semantics and receive no range
    certificate. Uniqueness is tested once, never during repeated chunk work.
    """
    order = np.argsort(row_idx) if with_order else None
    rows = row_idx[order] if with_order else np.sort(row_idx)
    rows.flags.writeable = False
    if order is not None:
        order.flags.writeable = False
    ordinary = (
        type(row_idx) is np.ndarray
        and row_idx.ndim == 1
        and row_idx.dtype == np.dtype(np.intp)
        and type(rows) is np.ndarray
        and (order is None or type(order) is np.ndarray)
    )
    if not ordinary:
        return rows, order, None
    rows = np.frombuffer(rows.tobytes(), dtype=np.intp)
    if order is not None:
        order = np.frombuffer(order.tobytes(), dtype=np.intp)
    certificate = _RowLookupCertificate(rows, order) if np.all(rows[1:] > rows[:-1]) else None
    return rows, order, certificate


def certified_row_bounds(group, start, stop, *, with_order):
    """Return two indices into the cached ordering, or refuse range dispatch."""
    if group._sorted_rows is None:
        rows, order, certificate = build_row_lookup(group.row_idx, with_order=with_order)
        group._sorted_rows = rows
        if with_order:
            group._row_order = order
        group._row_lookup_certificate = certificate
    certificate = group._row_lookup_certificate
    order = group._row_order if with_order else None
    if type(certificate) is not _RowLookupCertificate or not certificate.matches(
        group._sorted_rows, order
    ):
        return None
    return np.searchsorted(group._sorted_rows, (start, stop))
