"""A weighted Gram's blocks on a pool of threads (Li & Wood 2020, sec. 4).

``X'WX`` of a grouped design is a grid of blocks ``X_i'WX_j``, and every
block is formed by its own pass over the rows, writing nothing another block
reads.  Li & Wood (*Stat. Comput.* 30:19, 2020, sec. 4) parallelise the cross
product by "computing different blocks in different threads", processing the
blocks "in order of decreasing computational cost".  This module does that:

- **Tasks.**  One task per diagonal block and per cross block, each with an
  estimated cost and working set (``block_cost``, ``block_bytes``; an
  ordering and sizing heuristic, which no result depends on).
- **Order.**  Largest first (LPT; Graham, *SIAM J. Appl. Math.* 17:416,
  1969), ties in the serial order; tasks below ``_TINY_COST`` are batched so
  one queue round trip carries at least that much work.
- **Split.**  A tensor pair larger than the makespan bound ``total /
  workers`` runs its row stage in parts (``_cell_hist_raw_kron_parts``):
  runs of grid cells, whose rows sum into disjoint rows of the histogram, so
  the split block is bitwise the whole one.  The parts hold one more
  histogram, so blocks split, largest first, only while the budget also
  covers every split block's extra histogram (``split_bytes``) at once.
  Other blocks always run whole.
- **Workers.**  ``_parallel.pool_workers``: ``n_jobs`` capped by the task
  count and by ``max_memory`` over the largest task's working set.  One
  parallelism level: BLAS runs on one thread for the whole assembly
  (``pooled_blas_threads``), and the block kernels are ``nogil`` and never
  ``prange``.
- **Determinism.**  A task's arithmetic does not depend on which worker runs
  it or what ran before it: shared entries are formed once and are the
  serial values (``_SharedEntries``), scratch is per worker, and each block
  writes its own disjoint part of the Gram as soon as it is formed, so no
  worker holds a finished block.  The Gram is therefore bitwise identical
  at every worker count, the serial one included.

A small assembly (estimated cost below ``_MIN_POOLED_COST``) runs serially,
in the serial order, without starting a pool.  ``block_queue_config`` is the
hidden override the tests use to force a pool, a worker count or a split.
"""

from __future__ import annotations

import contextvars
import math
import threading
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from ._group_matrix_algebra import (
    _MAX_CROSS_EXPANSION_BYTES,
    _BlockWeightCache,
    _profile_count,
    _runtime_group_matrix_types,
    _SharedEntries,
)

# Work units are estimated multiply-adds and row visits, about a nanosecond
# each in the block kernels.  A queue round trip costs tens of microseconds,
# and starting a pool about a tenth of a millisecond a worker.
_TINY_COST = 1 << 16
_MIN_POOLED_COST = 1 << 22
# Transient row-length arrays a block route may hold at once: weight and
# index gathers, a weighted column (three float64-sized vectors).
_ROW_BYTES = 24


@dataclass(frozen=True)
class _QueueOverride:
    workers: int | None = None
    min_cost: float | None = None
    split: int | None = None


_override: ContextVar[_QueueOverride] = ContextVar(
    "superglm_block_queue_override", default=_QueueOverride()
)


@contextmanager
def block_queue_config(
    *, workers: int | None = None, min_cost: float | None = None, split: int | None = None
) -> Iterator[None]:
    """Hidden override for tests: an exact worker count, the pool threshold, a forced split.

    ``workers`` bypasses ``n_jobs`` and ``max_memory`` (capped only by the
    task count); ``min_cost=0`` pools every assembly of two or more tasks;
    ``split`` runs every splittable tensor pair's row stage in that many
    parts, serially too, whatever the memory budget.
    """
    token = _override.set(_QueueOverride(workers=workers, min_cost=min_cost, split=split))
    try:
        yield
    finally:
        _override.reset(token)


@dataclass
class BlockTask:
    """One Gram block: ``run(cache, profile)`` forms it, ``place(value)`` writes it into the Gram."""

    index: int
    cost: float
    nbytes: int
    run: Callable[[_BlockWeightCache, dict[str, Any] | None], Any]
    place: Callable[[Any], None]
    # Bytes the parts of a split add (one more histogram); 0 for a block that never splits.
    split_bytes: int = 0
    parts: int = field(default=1, compare=False)


def _side(gm, n: int) -> tuple[float, int, int]:
    """``(row work, bins, width)`` of one block side, for the cost model only.

    Row work is the per-row multiply-adds the side adds to a pass (a tensor's
    raw band, a dense or sparse group's columns; a binned side only indexes),
    and bins the cells a histogram over the side has.
    """
    categorical, scop, _spline_cat, ssp, tensor, *_rest = _runtime_group_matrix_types()
    width = int(gm.shape[1])
    if isinstance(gm, tensor):
        band = gm.raw_channels
        nonzeros = (
            band.values1.shape[1] * band.values2.shape[1]
            if band is not None
            else gm.B_unique.shape[1]
        )
        return math.sqrt(nonzeros), int(gm.n_bins1) * int(gm.n_bins2), width
    if isinstance(gm, ssp | scop):
        return 0.0, int(gm.n_bins), width
    if isinstance(gm, categorical):
        return 0.0, int(gm.n_levels), width
    return float(width), n, width


def block_cost(gm_i, gm_j, n: int) -> float:
    """Estimated work of a cross block: one row pass, then a bin-space contraction."""
    r_i, m_i, p_i = _side(gm_i, n)
    r_j, m_j, p_j = _side(gm_j, n)
    return n * (1.0 + r_i) * (1.0 + r_j) + float(min(m_i, m_j)) * p_i * p_j


def diagonal_cost(gm, n: int) -> float:
    """Estimated work of a diagonal block."""
    r, m, p = _side(gm, n)
    return n * (1.0 + r) + float(m) * p * p


def _aggregate_bytes(gm_i, gm_j, n: int) -> int:
    """The largest aggregate a cross block's route may hold; the routes cap it."""
    _r_i, m_i, _p_i = _side(gm_i, n)
    _r_j, m_j, _p_j = _side(gm_j, n)
    return min(_MAX_CROSS_EXPANSION_BYTES, 8 * m_i * m_j)


def block_bytes(gm_i, gm_j, n: int) -> int:
    """Working set of a cross block: row temporaries, one aggregate, the result.

    The routes bound their aggregates by ``_MAX_CROSS_EXPANSION_BYTES``.
    """
    return _ROW_BYTES * n + _aggregate_bytes(gm_i, gm_j, n) + 8 * gm_i.shape[1] * gm_j.shape[1]


def split_bytes(gm_i, gm_j, n: int) -> int:
    """What splitting a cross block adds, its parts' histograms; 0 if its row stage cannot split."""
    return _aggregate_bytes(gm_i, gm_j, n) if splittable_pair(gm_i, gm_j) else 0


def diagonal_bytes(gm, n: int) -> int:
    """Working set of a diagonal block."""
    _r, m, p = _side(gm, n)
    return _ROW_BYTES * n + min(_MAX_CROSS_EXPANSION_BYTES, 8 * m * max(p, 1)) + 8 * p * p


def splittable_pair(gm_i, gm_j) -> bool:
    """Whether a cross block may take the raw-band tensor route, whose row stage splits."""
    tensor = _runtime_group_matrix_types()[4]
    return (
        isinstance(gm_i, tensor)
        and isinstance(gm_j, tensor)
        and gm_i.tensor_id != gm_j.tensor_id
        and gm_i.raw_channels is not None
        and gm_j.raw_channels is not None
    )


class _Parts:
    """A split block's parts, claimed one at a time by its worker and any idle helper."""

    __slots__ = ("_part", "_count", "_next", "_done", "_results", "_error", "_lock", "_finished")

    def __init__(self, count: int, part: Callable[[int], Any]) -> None:
        self._part = part
        self._count = count
        self._next = 0
        self._done = 0
        self._results: list[Any] = [None] * count
        self._error: tuple[int, BaseException] | None = None
        self._lock = threading.Lock()
        self._finished = threading.Event()

    def help(self) -> None:
        """Run unclaimed parts until none is left; never waits."""
        while True:
            with self._lock:
                if self._next >= self._count:
                    return
                index = self._next
                self._next += 1
            value, error = None, None
            try:
                value = self._part(index)
            except BaseException as exc:  # re-raised by the block's own worker
                error = exc
            with self._lock:
                self._results[index] = value
                if error is not None and (self._error is None or index < self._error[0]):
                    self._error = (index, error)
                self._done += 1
                if self._done == self._count:
                    self._finished.set()

    def wait(self) -> list[Any]:
        """The parts' results in order, once every claimed part has finished."""
        self._finished.wait()
        if self._error is not None:
            raise self._error[1]
        return self._results


class _WorkQueue:
    """Units of tasks in LPT order, with split parts ahead of them; the first failure stops it."""

    def __init__(self, units: list[list[BlockTask]]) -> None:
        self._units: deque = deque(units)
        self._lock = threading.Lock()
        self._failure: tuple[int, BaseException] | None = None
        self._stopped = False

    def _next(self):
        with self._lock:
            if self._stopped or not self._units:
                return None
            return self._units.popleft()

    def stop(self) -> None:
        with self._lock:
            self._stopped = True

    def run_parts(self, count: int, part: Callable[[int], Any]) -> list[Any]:
        """Run ``count`` parts on this worker and on any worker that falls idle.

        The calling worker claims parts itself and waits only for parts
        another worker has already started, so it never waits on queued work.
        """
        parts = _Parts(count, part)
        with self._lock:
            self._units.extendleft([parts] * (count - 1))
        parts.help()
        return parts.wait()

    def work(self, cache: _BlockWeightCache) -> None:
        """One worker: run units until the queue is empty or a task has failed.

        Each block is placed as soon as it is formed: blocks write disjoint
        parts of the Gram, so no worker holds a finished block.
        """
        while (unit := self._next()) is not None:
            if isinstance(unit, _Parts):
                unit.help()
                continue
            for task in unit:
                cache._parts = task.parts
                try:
                    task.place(task.run(cache, cache._profile))
                except BaseException as exc:
                    with self._lock:
                        if self._failure is None or task.index < self._failure[0]:
                            self._failure = (task.index, exc)
                        self._stopped = True
                    return
                finally:
                    cache._parts = 1

    def raise_failure(self) -> None:
        if self._failure is not None:
            raise self._failure[1]


def _units(tasks: list[BlockTask]) -> list[list[BlockTask]]:
    """Tasks largest first (ties in serial order), tiny ones batched together."""
    units: list[list[BlockTask]] = []
    batch: list[BlockTask] = []
    batch_cost = 0.0
    for task in sorted(tasks, key=lambda task: (-task.cost, task.index)):
        if task.cost >= _TINY_COST:
            units.append([task])
            continue
        batch.append(task)
        batch_cost += task.cost
        if batch_cost >= _TINY_COST:
            units.append(batch)
            batch, batch_cost = [], 0.0
    if batch:
        units.append(batch)
    return units


def run_block_tasks(
    tasks: list[BlockTask],
    cache: _BlockWeightCache,
    profile: dict[str, Any] | None,
) -> None:
    """Run every task and place its block: serially in task order, or on a pool.

    ``tasks`` are in the serial order.  The serial run uses ``cache`` itself
    and is the established assembly; a pooled run gives each worker a view
    of ``cache`` (shared entries, own scratch and profile), and each worker
    places its blocks as it forms them.
    """
    from superglm._blas_threads import pooled_blas_threads
    from superglm._parallel import pool_workers, resolve_max_memory

    override = _override.get()
    units = _units(tasks)
    total = sum(task.cost for task in tasks)
    min_cost = _MIN_POOLED_COST if override.min_cost is None else override.min_cost
    if cache._batch is not None or len(units) < 2:
        workers = 1
    elif override.workers is not None:
        workers = max(1, min(int(override.workers), len(units)))
    elif total < min_cost:
        workers = 1
    else:
        workers = pool_workers(len(units), max(task.nbytes for task in tasks))

    # Every split block's parts can run at once, each block holding one more
    # histogram, so the splits share what the workers' largest tasks leave of
    # the budget, the largest blocks first.
    largest = max((task.nbytes for task in tasks), default=0)
    headroom = resolve_max_memory() - workers * largest if workers > 1 else 0
    for task in sorted(tasks, key=lambda task: (-task.cost, task.index)):
        if not task.split_bytes:
            continue
        if override.split is not None:
            task.parts = max(1, int(override.split))
        elif workers > 1 and task.cost * workers > total and task.split_bytes <= headroom:
            task.parts = min(workers, math.ceil(task.cost * workers / total))
            headroom -= task.split_bytes

    with pooled_blas_threads():
        if workers == 1:
            for task in tasks:
                cache._parts = task.parts
                try:
                    task.place(task.run(cache, profile))
                finally:
                    cache._parts = 1
            return
        shared = _SharedEntries()
        queue = _WorkQueue(units)
        profiles = [None if profile is None else {} for _ in range(workers)]
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="superglm-gram") as pool:
            # Each worker runs in its own copy of the caller's context, which
            # carries np.errstate and the held-warning scope as serial has them.
            futures = [
                pool.submit(
                    contextvars.copy_context().run,
                    queue.work,
                    cache.worker_view(shared, profiles[k], queue),
                )
                for k in range(workers)
            ]
            try:
                for future in futures:
                    future.result()
            except BaseException:
                queue.stop()
                raise
    queue.raise_failure()
    if profile is not None:
        for worker_profile in profiles:
            assert worker_profile is not None
            for key, value in worker_profile.items():
                profile[key] = profile.get(key, 0) + value
        _profile_count(profile, "block_pool_calls")
        _profile_count(profile, "block_pool_tasks", len(tasks))
        _profile_count(profile, "block_pool_units", len(units))
        _profile_count(profile, "block_pool_workers", workers)
        _profile_count(
            profile, "block_pool_split_parts", sum(t.parts for t in tasks if t.parts > 1)
        )


__all__ = [
    "BlockTask",
    "block_bytes",
    "block_cost",
    "block_queue_config",
    "diagonal_bytes",
    "diagonal_cost",
    "run_block_tasks",
    "split_bytes",
    "splittable_pair",
]
