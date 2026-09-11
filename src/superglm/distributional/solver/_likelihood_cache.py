"""Bounded, fit-owned reuse of audited immutable prepared likelihood children.

The retained budget includes child payload and authority metadata. One fresh
legacy ``take`` result, and its constant-size admission metadata, are caller
work outside that budget. No numerical tables or observation copies are made
to validate a hit. This cache never substitutes for endpoint certification.
"""

from __future__ import annotations

import sys
import weakref
from dataclasses import fields

import numpy as np

from superglm.distributional.families.gamma import GammaLikelihoodPlan, GammaLS
from superglm.distributional.families.gaussian import GaussianLikelihoodPlan, GaussianLS
from superglm.distributional.family import (
    ObservationContract,
    _likelihood_reuse_contract,
    _prepared_field_modes,
)
from superglm.distributional.weights import (
    ResolvedLikelihoodWeights,
    WeightContract,
    WeightProvenance,
)

_DATACLASSES = (
    GaussianLS,
    GammaLS,
    GaussianLikelihoodPlan,
    GammaLikelihoodPlan,
    ResolvedLikelihoodWeights,
    WeightProvenance,
    WeightContract,
    ObservationContract,
)
_SCALARS = (type(None), bool, int, float, str)
_MAX_ENTRIES = 256


class _UncertifiedError(Exception):
    pass


def _array_metadata(array):
    return (
        array.dtype.str,
        array.shape,
        array.strides,
        array.__array_interface__["data"][0],
        array.flags.writeable,
    )


class _ArrayAuthority:
    """Bind exact arrays and their complete canonical immutable backing chain."""

    __slots__ = ("references", "metadata", "backing", "__weakref__")

    def __init__(self, array, source_ids):
        arrays = []
        value = array
        while type(value) is np.ndarray and len(arrays) < 2:
            if (
                value.ndim != 1
                or value.dtype not in (np.dtype(np.float64), np.dtype(np.intp))
                or not value.flags.c_contiguous
                or value.flags.writeable
            ):
                raise _UncertifiedError
            arrays.append(value)
            source_ids.add(id(value))
            value = value.base
        if type(value) is not bytes or array.nbytes != len(value):
            raise _UncertifiedError
        source_ids.add(id(value))
        self.backing = value
        self.metadata = tuple(_array_metadata(item) for item in arrays)
        self_ref = weakref.ref(self)

        def released(_):
            authority = self_ref()
            if authority is not None:
                authority.backing = None

        self.references = tuple(weakref.ref(item, released) for item in arrays)

    def matches(self, array):
        value = array
        for reference, metadata in zip(self.references, self.metadata, strict=True):
            if (
                type(value) is not np.ndarray
                or reference() is not value
                or _array_metadata(value) != metadata
            ):
                self.backing = None
                return False
            value = value.base
        matched = type(value) is bytes and value is self.backing
        if not matched:
            # Same-object __setstate__ can abandon a source-sized backing.
            self.backing = None
        return matched


def _capture(value, source_ids):
    """Snapshot only the fixed builtin plan schema, retaining arrays weakly."""
    kind = type(value)
    # The outer (family, plan) pair is temporary; excluding its recycled id
    # later could undercount an unrelated cache entry.
    if kind is not tuple:
        source_ids.add(id(value))
    if kind is np.ndarray:
        return _ArrayAuthority(value, source_ids)
    if kind in _SCALARS:
        if kind is str and len(value) > 1024:
            raise _UncertifiedError
        return (kind, value)
    if kind is tuple:
        if len(value) > 16:
            raise _UncertifiedError
        return (tuple, tuple(_capture(item, source_ids) for item in value))
    if kind in _DATACLASSES:
        if set(vars(value)) != {field.name for field in fields(kind)}:
            raise _UncertifiedError
        return (
            kind,
            weakref.ref(value),
            tuple(
                (field.name, _capture(getattr(value, field.name), source_ids))
                for field in fields(kind)
            ),
        )
    raise _UncertifiedError


def _matches(value, snapshot):
    if type(snapshot) is _ArrayAuthority:
        return snapshot.matches(value)
    kind = snapshot[0]
    if type(value) is not kind:
        return False
    if kind in _SCALARS:
        return value == snapshot[1]
    if kind is tuple:
        return len(value) == len(snapshot[1]) and all(
            _matches(item, saved) for item, saved in zip(value, snapshot[1], strict=True)
        )
    return (
        snapshot[1]() is value
        and set(vars(value)) == {name for name, _ in snapshot[2]}
        and all(_matches(getattr(value, name), saved) for name, saved in snapshot[2])
    )


def _owned_bytes(value, excluded, seen=None):
    """Count unique owned buffers and Python metadata, excluding borrowed root state."""
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen or identity in excluded or isinstance(value, type):
        return 0
    seen.add(identity)
    size = sys.getsizeof(value)
    kind = type(value)
    if kind is np.ndarray:
        return size + _owned_bytes(value.base, excluded, seen)
    if kind is _ArrayAuthority:
        return size + sum(
            _owned_bytes(getattr(value, name), excluded, seen)
            for name in ("references", "metadata", "backing")
        )
    if kind is weakref.ReferenceType:
        callback = value.__callback__
        if callback is not None and id(callback) not in seen:
            seen.add(id(callback))
            size += sys.getsizeof(callback) + sys.getsizeof(callback.__closure__)
            for cell in callback.__closure__ or ():
                size += sys.getsizeof(cell) + _owned_bytes(cell.cell_contents, excluded, seen)
        return size
    if kind is tuple:
        return size + sum(_owned_bytes(item, excluded, seen) for item in value)
    if kind in _DATACLASSES:
        return (
            size
            + sys.getsizeof(vars(value))
            + sum(
                _owned_bytes(getattr(value, field.name), excluded, seen) for field in fields(kind)
            )
        )
    return size


def _include_buffer_ownership(snapshot, excluded):
    # A child shares dropped_input_positions with the root. Count that buffer
    # too: otherwise replacing the root field could leave an unbudgeted old
    # observation-sized buffer owned only by cached children.
    if type(snapshot) is _ArrayAuthority:
        excluded.discard(id(snapshot.backing))
        for reference in snapshot.references:
            excluded.discard(id(reference()))
    elif type(snapshot) is tuple:
        for item in snapshot:
            _include_buffer_ownership(item, excluded)


def _is_builtin_gaussian_gamma(family: object) -> bool:
    """Exact adapter types admitted by the bounded likelihood optimizations."""
    return type(family) in (GaussianLS, GammaLS)


def _eligible(family, plan):
    contract = _likelihood_reuse_contract(family)
    return (
        _is_builtin_gaussian_gamma(family)
        and contract is not None
        and contract.deterministic_chunk_replay
        and type(plan) is contract.plan_type
        and _prepared_field_modes(plan, contract) is not None
        and type(plan.weights) is ResolvedLikelihoodWeights
        and type(plan.weights.provenance) is WeightProvenance
        and type(plan.weights.provenance.contract) is WeightContract
        and type(plan.observation) is ObservationContract
    )


def _is_range(indices, start, stop, n):
    if (
        type(indices) is not np.ndarray
        or indices.ndim != 1
        or indices.dtype.kind not in "iu"
        or type(start) is not int
        or type(stop) is not int
        or not 0 <= start < stop <= n
        or indices.size != stop - start
    ):
        return False
    # Bounded comparisons avoid trusting a caller's range labels or allocating
    # another entire selection vector. No signed-integer subtraction overflow.
    for offset in range(0, indices.size, 8192):
        block = indices[offset : offset + 8192]
        expected = np.arange(start + offset, start + offset + block.size, dtype=np.intp)
        if not np.array_equal(block, expected):
            return False
    return True


class _LikelihoodCache:
    def __init__(self, family, byte_budget):
        self._family = family
        self.byte_budget = byte_budget
        self._entries = {}
        self._root = None
        self._source_ids = set()
        self._closed = False
        self.retained_bytes = 0
        self.hits = 0
        self.misses = 0
        self.max_fresh_child_bytes = 0

    @property
    def entry_count(self):
        return len(self._entries)

    def clear(self):
        self._entries.clear()
        self._root = None
        self._source_ids.clear()
        self._closed = False
        self.retained_bytes = 0

    def _prepare(self, plan):
        source_ids = set()
        try:
            if not _eligible(self._family, plan):
                return False
            snapshot = _capture((self._family, plan), source_ids)
        except (AttributeError, _UncertifiedError):
            return False
        # The identity set and base cache/dictionary headers are counted too.
        size = (
            _owned_bytes(snapshot, source_ids)
            + sys.getsizeof(source_ids)
            + sum(sys.getsizeof(identity) for identity in source_ids)
            + 512
        )
        if size > self.byte_budget:
            return False
        self._root = snapshot
        _include_buffer_ownership(snapshot, source_ids)
        self._source_ids = source_ids
        self.retained_bytes = size
        return True

    def take(self, plan, indices, *, start, stop):
        """Return an unchanged legacy child, or call legacy take on any refusal."""
        try:
            unchanged = self._root is not None and _matches((self._family, plan), self._root)
        except AttributeError:
            unchanged = False
        if not unchanged:
            self.clear()
            if not self._prepare(plan):
                self.misses += 1
                return plan.take(indices)
        if not _is_range(indices, start, stop, len(plan.weights.values)):
            self.misses += 1
            return plan.take(indices)
        key = (start, stop)
        entry = self._entries.get(key)
        if entry is not None:
            child, snapshot, _ = entry
            try:
                unchanged = _matches(child, snapshot)
            except AttributeError:
                unchanged = False
            if unchanged:
                self.hits += 1
                return child
            # A child is private, but instrumentation can still expose it.
            # Revoke the whole generation after a forged frozen child change.
            self.clear()
            self._prepare(plan)
        self.misses += 1
        child = plan.take(indices)
        if self._closed or self._root is None:
            return child
        try:
            snapshot = _capture(child, set())
        except (AttributeError, _UncertifiedError):
            return child
        entry = (child, snapshot, key)
        # 128 bytes per entry conservatively covers dictionary growth. Shared
        # scalar provenance is borrowed; even shared dropped-row buffers count.
        size = _owned_bytes(entry, self._source_ids) + 128
        self.max_fresh_child_bytes = max(self.max_fresh_child_bytes, size)
        if self.entry_count >= _MAX_ENTRIES or self.retained_bytes + size > self.byte_budget:
            # Preserve the admitted prefix across sequential passes; an LRU
            # smaller than a pass would evict everything before its next hit.
            self._closed = True
            return child
        self._entries[key] = entry
        self.retained_bytes += size
        return child


def build_likelihood_cache(family, plan, *, byte_budget=64 << 20):
    """Create a private bounded cache only for audited canonical builtin plans."""
    if type(byte_budget) is not int or byte_budget <= 0:
        return None
    cache = _LikelihoodCache(family, byte_budget)
    return cache if cache._prepare(plan) else None
