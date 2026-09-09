"""Exact live-array certificates with bounded parallel hashing."""

from __future__ import annotations

import hashlib
import threading
import tracemalloc
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numba
import numpy as np
import pytest

import superglm.distributional.solver.solver as solver
from superglm.distributional.solver import _reuse_digest
from superglm.group_matrix import DenseGroupMatrix, DesignMatrix

from .test_distributional_chunk_reuse import _problem


def _canonical_certificate(events):
    """Independent, materialized-byte oracle for the versioned tree format."""
    version = b"superglm.chunk-reuse.v2\0"
    metadata = bytearray(version + b"metadata\0")
    leaves = bytearray(version + b"leaves\0")
    for kind, value in events:
        if kind == "field":
            payload = repr((type(value).__module__, type(value).__qualname__, value)).encode()
            metadata.extend(b"F" + len(payload).to_bytes(8, "big") + payload)
        else:
            metadata.extend(b"A")
            header = repr((value.dtype.str, value.shape)).encode()
            raw = value.tobytes(order="C")
            leaf = version + b"array\0" + len(header).to_bytes(8, "big") + header + raw
            leaves.extend(hashlib.sha256(leaf).digest())
    root = version + b"root\0" + hashlib.sha256(metadata).digest() + hashlib.sha256(leaves).digest()
    return hashlib.sha256(root).hexdigest()


def _certificate(events, workers, monkeypatch):
    monkeypatch.setattr(_reuse_digest, "get_num_threads", lambda: workers)
    with _reuse_digest._ReuseDigest() as digest:
        for kind, value in events:
            getattr(digest, kind)(value)
        return digest.hexdigest()


_BASE = np.arange(120, dtype=np.float64).reshape(10, 12)
_ARRAYS = [
    pytest.param(_BASE, id="c"),
    pytest.param(np.asfortranarray(_BASE), id="fortran"),
    pytest.param(_BASE[::-1, ::-1], id="reversed"),
    pytest.param(_BASE[::3, 1::2], id="strided"),
    pytest.param(_BASE[:, :1, None], id="singleton-strided"),
    pytest.param(_BASE[:1, None, :], id="singleton-contiguous"),
    pytest.param(np.empty((0, 2)), id="empty-first-axis"),
    pytest.param(np.empty((2, 0)), id="empty-last-axis"),
    pytest.param(np.array(1.25), id="scalar"),
    pytest.param(_BASE.astype(">f8"), id="non-native-contiguous"),
    pytest.param(_BASE.astype(">f8")[::-1, ::3], id="non-native-strided"),
    pytest.param(np.array([0.0, -0.0]), id="signed-zero"),
    pytest.param(np.array([True, False]), id="boolean"),
    pytest.param(
        np.array([0x7FF8000000000001, 0x7FF8000000000002], dtype=np.uint64).view(np.float64),
        id="nan-payload",
    ),
]


@pytest.mark.parametrize("values", _ARRAYS)
@pytest.mark.parametrize("workers", [1, 2, 16])
def test_digest_matches_canonical_c_order_bytes(values, workers, monkeypatch):
    events = [("field", ("group", slice(1, 4), {"lower": -0.0})), ("array", values)]
    assert _certificate(events, workers, monkeypatch) == _canonical_certificate(events)


@pytest.mark.parametrize(
    "before,after",
    [
        ([("field", 1)], [("field", "1")]),
        ([("field", [1, 2])], [("field", (1, 2))]),
        ([("field", "a"), ("field", "bc")], [("field", "ab"), ("field", "c")]),
        (
            [("field", "x"), ("array", np.ones(2))],
            [("array", np.ones(2)), ("field", "x")],
        ),
        ([("array", np.array([0.0]))], [("array", np.array([-0.0]))]),
        ([("array", np.arange(4))], [("array", np.arange(4).reshape(2, 2))]),
        ([("array", np.ones(4, dtype="i4"))], [("array", np.ones(4, dtype="f4"))]),
    ],
)
def test_certificate_commits_types_framing_shape_and_exact_bits(before, after, monkeypatch):
    assert _certificate(before, 2, monkeypatch) != _certificate(after, 2, monkeypatch)


@pytest.mark.parametrize("workers", [2, 16])
def test_reverse_leaf_completion_preserves_source_order(workers, monkeypatch):
    values = [np.full(8192, index, dtype=np.int64) for index in range(workers)]
    finished = [threading.Event() for _ in values]
    completion_order = []
    real_leaf = _reuse_digest._array_digest

    def reverse_completion(value):
        index = int(value[0])
        result = real_leaf(value)
        if index < workers - 1:
            assert finished[index + 1].wait(10)
        completion_order.append(index)
        finished[index].set()
        return result

    monkeypatch.setattr(_reuse_digest, "_array_digest", reverse_completion)
    events = [("array", value) for value in values]
    assert _certificate(events, workers, monkeypatch) == _canonical_certificate(events)
    assert completion_order == list(reversed(range(workers)))


@pytest.mark.parametrize("workers", [1, 2, 16])
def test_every_array_occurrence_is_read_fresh_on_every_call(workers, monkeypatch):
    base = np.arange(48, dtype=np.float64)
    view = base[::2]
    arrays = [base, view, base, view, base]
    events = [("array", value) for value in arrays]
    seen = []
    real_leaf = _reuse_digest._array_digest

    def observed_leaf(value):
        seen.append(id(value))
        return real_leaf(value)

    monkeypatch.setattr(_reuse_digest, "_array_digest", observed_leaf)
    first = _certificate(events, workers, monkeypatch)
    assert first == _canonical_certificate(events)
    assert Counter(seen) == Counter(map(id, arrays))
    seen.clear()
    base[0] = -0.0
    second = _certificate(events, workers, monkeypatch)
    assert second == _canonical_certificate(events)
    assert second != first
    assert Counter(seen) == Counter(map(id, arrays))


@pytest.mark.parametrize("workers", [2, 16])
def test_outstanding_leaf_results_are_bounded_by_twice_the_worker_budget(workers, monkeypatch):
    outstanding = 0
    high_water = 0

    class TrackedResult:
        def __init__(self, future):
            self.future = future

        def result(self):
            nonlocal outstanding
            try:
                return self.future.result()
            finally:
                outstanding -= 1

    class TrackedExecutor(ThreadPoolExecutor):
        def submit(self, fn, /, *args, **kwargs):
            nonlocal outstanding, high_water
            outstanding += 1
            high_water = max(high_water, outstanding)
            return TrackedResult(super().submit(fn, *args, **kwargs))

    monkeypatch.setattr(_reuse_digest, "ThreadPoolExecutor", TrackedExecutor)
    events = [("array", np.array([index])) for index in range(5 * workers)]
    assert _certificate(events, workers, monkeypatch) == _canonical_certificate(events)
    assert outstanding == 0
    assert high_water <= 2 * workers


def test_contiguous_payloads_share_storage_and_noncontiguous_scratch_is_bounded(monkeypatch):
    contiguous = np.arange(1 << 20, dtype=np.float64)
    noncontiguous = contiguous.reshape(-1, 4)[::-1, ::2]
    sources = [contiguous, noncontiguous]
    events = [("array", value) for value in sources * 4]
    expected = _canonical_certificate(events)
    real_sha256 = hashlib.sha256
    zero_copy = []
    scratch = []

    class ObservedHash:
        def __init__(self, data=b""):
            self._hash = real_sha256(data)

        def update(self, data):
            if isinstance(data, memoryview):
                owner = data.obj
                if np.shares_memory(owner, contiguous):
                    zero_copy.append(owner)
                else:
                    # Inspect ownership, not merely hash-update byte counts:
                    # the iterator owns each copied block, never a full array.
                    assert isinstance(owner.base, np.nditer)
                    assert not owner.flags.owndata
                    assert owner.nbytes <= 64 * 1024
                    scratch.append((threading.get_ident(), owner.nbytes))
            self._hash.update(data)

        def digest(self):
            return self._hash.digest()

        def hexdigest(self):
            return self._hash.hexdigest()

    monkeypatch.setattr(hashlib, "sha256", ObservedHash)
    tracemalloc.start()
    try:
        result = _certificate(events, 2, monkeypatch)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result == expected
    assert len(zero_copy) == 4
    assert all(value is contiguous for value in zero_copy)
    assert scratch
    assert len({worker for worker, _ in scratch}) <= 2
    # Two <=64 KiB iterator buffers plus bounded thread/future/metadata overhead.
    # Even one materialized noncontiguous leaf would require four MiB.
    assert peak < 2 * 64 * 1024 + 512 * 1024


@pytest.mark.parametrize("dtype", [np.longdouble, np.complex128, "S65537"])
def test_wide_public_buffer_replacements_keep_bounded_scratch(dtype, monkeypatch):
    itemsize = np.dtype(dtype).itemsize
    source = np.zeros((max(4, 262144 // itemsize), 2), dtype=dtype)
    source.view(np.uint8).reshape(-1)[:] = 7
    values = source[:, :1]
    expected = _canonical_certificate([("array", values)])
    real_sha256 = hashlib.sha256
    copied = []

    class ObservedHash:
        def __init__(self, data=b""):
            self._hash = real_sha256(data)

        def update(self, data):
            if isinstance(data, memoryview) and not np.shares_memory(data.obj, source):
                copied.append(data.obj.nbytes)
                assert data.obj.nbytes <= 64 * 1024
            self._hash.update(data)

        def digest(self):
            return self._hash.digest()

        def hexdigest(self):
            return self._hash.hexdigest()

    monkeypatch.setattr(hashlib, "sha256", ObservedHash)
    assert _certificate([("array", values)], 2, monkeypatch) == expected
    if itemsize > 64 * 1024:
        assert not copied
    else:
        assert copied


def test_schema_refusal_before_arrays_does_not_resolve_or_start_workers(monkeypatch):
    problem = _problem()
    context = solver._validated_context(
        *problem[:5], coefficient_curvature="observed", chunk_size=17, coefficient_face=None
    )
    context = replace(context, family=object())

    def unexpected_worker_budget():
        raise AssertionError("an early schema refusal initialized hashing workers")

    monkeypatch.setattr(_reuse_digest, "get_num_threads", unexpected_worker_budget)
    assert solver._chunk_reuse_data_certificate(context) is None


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("reverse", [False, True])
def test_reference_containing_arrays_retain_iterator_refusal(workers, reverse, monkeypatch):
    values = np.array([None, object()], dtype=object)
    if reverse:
        values = values[::-1]
    with pytest.raises(TypeError, match="references"):
        _certificate([("array", values)], workers, monkeypatch)


@pytest.mark.parametrize("refusal", [False, True])
@pytest.mark.parametrize("worker_failure", [False, True])
def test_certificate_joins_workers_on_success_refusal_and_worker_failure(
    refusal, worker_failure, monkeypatch
):
    problem = _problem(n=24_000)
    context = solver._validated_context(
        *problem[:5], coefficient_curvature="observed", chunk_size=17, coefficient_face=None
    )
    if refusal:

        class CustomDenseGroup(DenseGroupMatrix):
            pass

        state = context.layout.predictors[0]
        custom = CustomDenseGroup(state.design.group_matrices[0].M)
        context = replace(
            context,
            layout=replace(
                context.layout,
                predictors=(
                    replace(state, design=DesignMatrix([custom], len(context.response), 1)),
                    context.layout.predictors[1],
                ),
            ),
        )
    pools = []

    class TrackedExecutor(ThreadPoolExecutor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            pools.append(self)

    sentinel = RuntimeError("leaf failed")
    real_leaf = _reuse_digest._array_digest

    def maybe_fail(value):
        if worker_failure and value is context.response:
            raise sentinel
        return real_leaf(value)

    monkeypatch.setattr(_reuse_digest, "get_num_threads", lambda: 2)
    monkeypatch.setattr(_reuse_digest, "ThreadPoolExecutor", TrackedExecutor)
    monkeypatch.setattr(_reuse_digest, "_array_digest", maybe_fail)
    if worker_failure:
        with pytest.raises(RuntimeError) as error:
            solver._chunk_reuse_data_certificate(context)
        assert error.value is sentinel
    else:
        result = solver._chunk_reuse_data_certificate(context)
        assert (result is None) == refusal
    assert pools and pools[0]._threads
    assert all(not thread.is_alive() for pool in pools for thread in pool._threads)


def test_worker_failure_cancels_queued_leaves_and_joins_running_leaves(monkeypatch):
    second_started = threading.Event()
    release_running = threading.Event()
    shutdown_started = threading.Event()
    queued_cancelled = threading.Event()
    caller_finished = threading.Event()
    futures = []
    pools = []
    errors = []
    sentinel = ValueError("broken leaf")

    class TrackedExecutor(ThreadPoolExecutor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            pools.append(self)

        def submit(self, fn, /, *args, **kwargs):
            future = super().submit(fn, *args, **kwargs)
            future.add_done_callback(
                lambda result: queued_cancelled.set() if result.cancelled() else None
            )
            futures.append(future)
            return future

        def shutdown(self, *args, **kwargs):
            shutdown_started.set()
            return super().shutdown(*args, **kwargs)

    real_leaf = _reuse_digest._array_digest

    def failing_leaf(value):
        if value[0] == 0:
            assert second_started.wait(10)
            raise sentinel
        second_started.set()
        assert release_running.wait(10)
        return real_leaf(value)

    monkeypatch.setattr(_reuse_digest, "get_num_threads", lambda: 2)
    monkeypatch.setattr(_reuse_digest, "ThreadPoolExecutor", TrackedExecutor)
    monkeypatch.setattr(_reuse_digest, "_array_digest", failing_leaf)

    def call():
        try:
            with _reuse_digest._ReuseDigest() as digest:
                for index in range(10):
                    digest.array(np.array([index]))
                digest.hexdigest()
        except BaseException as error:
            errors.append(error)
        finally:
            caller_finished.set()

    caller = threading.Thread(target=call)
    caller.start()
    try:
        assert shutdown_started.wait(10)
        assert queued_cancelled.wait(10)
        assert not caller_finished.is_set()
    finally:
        release_running.set()
        caller.join(10)
    assert caller_finished.is_set()
    assert errors == [sentinel]
    assert any(future.cancelled() for future in futures)
    assert all(not thread.is_alive() for pool in pools for thread in pool._threads)


def test_certificate_array_reads_use_the_current_numba_worker_budget(monkeypatch):
    """The old single hash streams every payload on the caller thread."""
    problem = _problem(n=24_000)
    context = solver._validated_context(
        *problem[:5], coefficient_curvature="observed", chunk_size=17, coefficient_face=None
    )
    real_sha256 = hashlib.sha256
    caller = threading.get_ident()
    readers = set()

    class ObservedHash:
        def __init__(self, data=b""):
            self._hash = real_sha256(data)

        def update(self, data):
            if isinstance(data, memoryview) and data.nbytes >= 64 * 1024:
                readers.add(threading.get_ident())
            self._hash.update(data)

        def digest(self):
            return self._hash.digest()

        def hexdigest(self):
            return self._hash.hexdigest()

    monkeypatch.setattr(hashlib, "sha256", ObservedHash)
    previous = numba.get_num_threads()
    numba.set_num_threads(2)
    try:
        assert solver._chunk_reuse_data_certificate(context) is not None
    finally:
        numba.set_num_threads(previous)
    assert readers
    assert caller not in readers
    assert len(readers) <= 2
