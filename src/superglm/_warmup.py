"""``superglm.warmup``: compile the kernels in worker processes, then load them here.

Numba compiles under one lock per process, so only processes can share the
work. The stages below are grouped into units; ``warmup`` runs each unit in a
fresh interpreter, at most ``n_jobs`` at a time and longest first, and each
saves its kernels to Numba's on-disk cache. This process then runs every stage
in the original order, which loads those kernels from the cache and compiles
whatever a worker did not leave there. One unit per process keeps the set of
compiled signatures the same on every run: within a process, Numba serves a
call from an already loaded overload whose types it can convert to.

Numba's cache index is a read-modify-write of one file per function, and the
data file an entry points to does not record the signature it was compiled
for. Two processes saving different signatures of one kernel at once can both
claim the same data file and leave the index pointing at the other's code.
Numba also publishes the index before it replaces the data file, so a load in
between, or a worker stopped in between, leaves an entry that names an obsolete
kernel. While workers run, every save and load here and in the workers takes
the lock of the cache directory it touches, a file there that any other
warmup writing the same cache takes as well, and a save writes its data file
before the index (``_locked_cache``).
One ``warmup`` runs at a time in a process.

One worker, or an interpreter that cannot start workers, runs the stages here
alone, as ``warmup`` always did.
"""

from __future__ import annotations

import errno
import importlib
import itertools
import os
import pickle
import subprocess
import sys
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

# The stages in the order this process runs them: (unit, module, function, args).
_STAGES: tuple[tuple[str, str, str, tuple], ...] = (
    ("tweedie", "superglm.distributional.kernels.tweedie", "_warmup_tweedie", ()),
    ("tweedie", "superglm._tweedie_series", "warmup", ()),
    (
        "group_matrix",
        "superglm._group_matrix._group_matrix_kernels",
        "_warmup_group_matrix_kernels",
        (),
    ),
    ("small_a", "superglm.reml._compensated", "_warmup_compensated", ()),
    ("small_a", "superglm.solvers._exact_sums", "_warmup_exact_sums", ()),
    (
        "small_a",
        "superglm.distributional.solver._ordinary_packing",
        "_warmup_ordinary_packing",
        (),
    ),
    ("small_a", "superglm.distributional.solver.derivatives", "_warmup_chain_rule", ()),
    ("joint_tables", "superglm.distributional.solver._joint_tables", "_warmup_joint_tables", ()),
    ("small_b", "superglm._group_matrix._group_matrix_support", "_warmup_support_scans", ()),
    ("small_b", "superglm.features.interaction", "_warmup_row_kron", ()),
    ("small_b", "superglm.features._basis_kernels", "_warmup_basis_kernels", ()),
    ("small_b", "superglm._group_matrix._lookup_tables", "_warmup_lookup_tables", ()),
    ("small_a", "superglm.reml._leverage_rows", "_warmup_leverage_rows", ()),
    ("small_b", "superglm.distributional.kernels.gamma", "_warmup_gamma", ()),
    *(
        (
            f"declared_{part}",
            "superglm.distributional.kernels._declared_bounds",
            "_warmup_declared_bounds",
            (part,),
        )
        for part in ("gamma", "generalized_gamma", "others")
    ),
    ("declared_rows", "superglm.distributional.kernels._declared", "_warmup_rows", ()),
    (
        "negative_binomial",
        "superglm.distributional.kernels.negative_binomial",
        "_warmup_negative_binomial",
        (),
    ),
    ("negative_binomial", "superglm._nb2_deviance", "_warmup_nb2_deviance", ()),
    (
        "negative_binomial",
        "superglm.distributional.kernels._negative_binomial_crps",
        "_warmup_negative_binomial_crps",
        (),
    ),
    *(
        (f"scalar_{family}", "superglm", "_warmup_scalar_observations", (family,))
        for family in ("negative_binomial", "tweedie")
    ),
    ("dirichlet", "superglm", "_warmup_dirichlet", ()),
    (
        "generalized_gamma",
        "superglm.distributional.kernels.generalized_gamma",
        "_warmup_generalized_gamma",
        (),
    ),
    ("small_a", "superglm.solvers._structured.leaf_kernels", "_warmup_leaf_kernels", ()),
    (
        "factor_smooth",
        "superglm.solvers._structured.block_leaves",
        "_warmup_factor_smooth",
        (False,),
    ),
    ("balance_tree", "superglm.solvers._structured.balance_tree", "_warmup_balance_tree", ()),
)

# Seconds each unit took to compile alone on an empty cache, eight at a time
# (AMD EPYC 9645, Python 3.13, Numba 0.68, 2026-11-14). They set only the order workers start
# units in, longest first, which keeps the longest unit from starting last.
_UNIT_SECONDS = {
    "scalar_negative_binomial": 26.3,
    "dirichlet": 23.7,
    "declared_generalized_gamma": 22.8,
    "scalar_tweedie": 20.9,
    "declared_gamma": 14.9,
    "balance_tree": 14.6,
    "factor_smooth": 11.5,
    "group_matrix": 10.3,
    "tweedie": 6.8,
    "joint_tables": 6.8,
    "declared_others": 6.1,
    "small_a": 5.9,
    "negative_binomial": 5.7,
    "declared_rows": 5.2,
    "small_b": 4.4,
    "generalized_gamma": 4.3,
}

# Peak memory of one worker, for the cap ``max_memory`` sets on their number.
_WORKER_BYTES = 600 << 20


def _units() -> dict[str, list[tuple[str, str, tuple]]]:
    units: dict[str, list[tuple[str, str, tuple]]] = {}
    for unit, module, function, args in _STAGES:
        units.setdefault(unit, []).append((module, function, args))
    return units


def _run(stages) -> None:
    for module, function, args in stages:
        getattr(importlib.import_module(module), function)(*args)


# The lock every warmup's workers and their parent take on one cache directory.
_LOCK_NAME = "superglm-cache.lock"


@contextmanager
def _file_lock(path: str | None) -> Iterator[bool]:
    """Hold an exclusive lock on the file at ``path`` (created if missing) across processes.

    Yields whether the lock is held. Without the file (no directory yet) or on
    a file system that cannot lock, it is not, and the caller leaves the cache
    alone rather than read or write it unlocked.
    """
    try:
        file = open(path, "a+b") if path is not None else None
    except OSError:
        file = None
    if file is None:
        yield False
        return
    with file:
        if sys.platform == "win32":
            import msvcrt

            file.seek(0)
            while True:
                try:
                    msvcrt.locking(file.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError as error:  # LK_LOCK gives up after ten one-second tries
                    if error.errno not in (errno.EDEADLOCK, errno.EACCES):
                        yield False
                        return
            try:
                yield True
            finally:
                file.seek(0)
                msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            try:
                fcntl.flock(file.fileno(), fcntl.LOCK_EX)
            except OSError:
                yield False
                return
            try:
                yield True
            finally:
                fcntl.flock(file.fileno(), fcntl.LOCK_UN)


def _lock_path(cache_file) -> str | None:
    """The lock of the directory ``cache_file`` lives in, shared by every warmup that writes it."""
    path = getattr(cache_file, "_index_path", None)
    return None if path is None else os.path.join(os.path.dirname(path), _LOCK_NAME)


# The private ``IndexDataCacheFile`` methods ``_locked_cache`` composes a save from.
_CACHE_FILE_METHODS = (
    "load",
    "_load_index",
    "_save_index",
    "_save_data",
    "_data_name",
    "_data_path",
)


def _published_names(cache_file) -> set[str] | None:
    """The data files the index on disk names, whatever source or Numba wrote it; ``None`` if unread.

    Read as Numba writes it: a pickled version, then a pickled ``(stamp, overloads)``.
    """
    try:
        with open(cache_file._index_path, "rb") as file:
            pickle.load(file)
            _, overloads = pickle.loads(file.read())
        return set(overloads.values())
    except FileNotFoundError:
        return set()
    except Exception:
        return None


def _free_data_name(cache_file, overloads: dict) -> str:
    """A data file no index names: neither this source's nor the one on disk (a stale one)."""
    published = _published_names(cache_file)
    taken = set(overloads.values()) | (published or set())
    for number in itertools.count(1):
        name = cache_file._data_name(number)
        unknown = published is None and os.path.exists(cache_file._data_path(name))
        if name not in taken and not unknown:
            return name
    raise AssertionError("unreachable")


@contextmanager
def _locked_cache() -> Iterator[None]:
    """Inside the block, this process's Numba cache saves and loads take their directory's lock.

    The lock is a file in the cache directory, so two warmups writing one
    cache (cluster nodes sharing ``NUMBA_CACHE_DIR``) exclude each other too.

    A save is Numba's ``IndexDataCacheFile.save`` with its last two writes
    swapped, the data file first and then the index that names it, and a new
    entry's data file is one that no index on disk names, a stale one
    included. A process stopped between the two writes leaves every index as
    it was, naming only the code it was written with. Where the lock cannot
    be taken, a save is skipped and a load misses, so the kernel compiles
    again in this process rather than through an unlocked cache.
    """
    from numba.core import caching

    cache_file = caching.IndexDataCacheFile
    save, load = cache_file.save, cache_file.load

    # locked_save is adapted from Numba's IndexDataCacheFile.save, under Numba's licence:
    #
    #   Copyright (c) 2012, Anaconda, Inc.
    #   All rights reserved.
    #
    #   Redistribution and use in source and binary forms, with or without
    #   modification, are permitted provided that the following conditions are
    #   met:
    #
    #   Redistributions of source code must retain the above copyright notice,
    #   this list of conditions and the following disclaimer.
    #
    #   Redistributions in binary form must reproduce the above copyright
    #   notice, this list of conditions and the following disclaimer in the
    #   documentation and/or other materials provided with the distribution.
    #   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    #   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    #   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    #   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    #   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    #   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    #   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    #   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    #   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    #   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    #   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    def locked_save(self, key, data):
        with _file_lock(_lock_path(self)) as held:
            if not held:  # unsaved: this process compiles the kernel again instead
                return
            overloads = self._load_index()
            name = overloads.get(key) or _free_data_name(self, overloads)
            self._save_data(name, data)
            if overloads.get(key) != name:
                overloads[key] = name
                self._save_index(overloads)

    def locked_load(self, key):
        with _file_lock(_lock_path(self)) as held:
            return load(self, key) if held else None

    cache_file.save, cache_file.load = locked_save, locked_load
    try:
        yield
    finally:
        cache_file.save, cache_file.load = save, load


def _run_unit(unit: str) -> None:
    """A worker's whole job: run one unit's stages with its cache locked."""
    with _locked_cache():
        _run(_units()[unit])


def _worker_count(units: int) -> int:
    from superglm._parallel import resolve_max_memory, resolve_n_jobs

    return max(1, min(resolve_n_jobs(), units, resolve_max_memory() // _WORKER_BYTES))


def _can_start_workers() -> bool:
    import numba
    from numba.core import caching

    return (
        not getattr(numba.config, "DISABLE_JIT", False)
        and bool(sys.executable)
        and not getattr(sys, "frozen", False)
        and all(
            callable(getattr(caching.IndexDataCacheFile, name, None))
            for name in _CACHE_FILE_METHODS
        )
    )


def _compile_in_workers(units: list[str], workers: int) -> None:
    """Run each unit in its own interpreter, ``workers`` at a time, longest first."""
    import numba

    env = dict(os.environ)
    cache_dir = getattr(numba.config, "CACHE_DIR", "")
    if cache_dir:  # set in code or by superglm._numba_cache, not only in env
        env["NUMBA_CACHE_DIR"] = str(cache_dir)
    order = sorted(units, key=lambda unit: -_UNIT_SECONDS.get(unit, 0.0))
    stop, guard, live = threading.Event(), threading.Lock(), set()

    def start(unit: str) -> None:
        # -P and the parent's sys.path: the worker imports this superglm,
        # never a module that happens to sit in the working directory. The
        # code goes in on stdin, which has no command line's length limit,
        # as ASCII (``ascii``), which every locale's encoding can carry.
        code = (
            f"import sys; sys.path[:] = {ascii(sys.path)}\n"
            f"from superglm._warmup import _run_unit; _run_unit({ascii(unit)})\n"
        )
        with guard:
            if stop.is_set():
                return
            try:
                process = subprocess.Popen(
                    [sys.executable, "-P", "-"],
                    env=env,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
            except OSError:  # this process compiles the unit afterwards instead
                return
            live.add(process)
        try:
            process.communicate(code)
        finally:
            with guard:
                live.discard(process)

    pool = ThreadPoolExecutor(workers)
    try:
        # This process's own saves, a fit on another thread say, take the lock too.
        with _locked_cache():
            list(pool.map(start, order))
    except BaseException:  # an interrupt: start no more workers and stop those running
        stop.set()
        with guard:
            for process in live:
                process.kill()
        raise
    finally:
        pool.shutdown(wait=True)


_one_warmup = threading.Lock()


def warmup() -> None:
    """Compile every stage's kernels, in worker processes where it can, then load them here."""
    from superglm._numba_compile import forget_compiles

    with _one_warmup:
        units = _units()
        workers = _worker_count(len(units))
        if workers > 1 and _can_start_workers():
            _compile_in_workers(list(units), workers)
        _run((module, function, args) for _, module, function, args in _STAGES)
        forget_compiles()
