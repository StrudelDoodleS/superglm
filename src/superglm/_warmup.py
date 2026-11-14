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
one lock, and a save writes its data file before the index (``_locked_cache``).
One ``warmup`` runs at a time in a process.

One worker, or an interpreter that cannot start workers, runs the stages here
alone, as ``warmup`` always did.
"""

from __future__ import annotations

import importlib
import itertools
import os
import subprocess
import sys
import tempfile
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


@contextmanager
def _file_lock(path: str) -> Iterator[None]:
    """Hold an exclusive lock on the file at ``path`` (created if missing) across processes."""
    with open(path, "a+b") as file:
        if sys.platform == "win32":
            import msvcrt

            file.seek(0)
            while True:
                try:
                    msvcrt.locking(file.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError:  # LK_LOCK gives up after ten one-second tries
                    continue
            try:
                yield
            finally:
                file.seek(0)
                msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(file.fileno(), fcntl.LOCK_UN)


# The private ``IndexDataCacheFile`` methods ``_locked_cache`` composes a save from.
_CACHE_FILE_METHODS = ("load", "_load_index", "_save_index", "_save_data", "_data_name")


@contextmanager
def _locked_cache(lock_path: str) -> Iterator[None]:
    """Inside the block, this process's Numba cache saves and loads take the lock at ``lock_path``.

    A save is Numba's ``IndexDataCacheFile.save`` with its last two writes
    swapped: the data file first, then the index that names it. A process
    stopped between the two leaves the index as it was, which does not name
    the new file (a stale index loads as empty).
    """
    from numba.core import caching

    cache_file = caching.IndexDataCacheFile
    save, load = cache_file.save, cache_file.load

    def locked_save(self, key, data):
        with _file_lock(lock_path):
            overloads = self._load_index()
            name = overloads.get(key)
            if name is None:
                taken = set(overloads.values())
                name = next(
                    self._data_name(i)
                    for i in itertools.count(1)
                    if self._data_name(i) not in taken
                )
            self._save_data(name, data)
            if overloads.get(key) != name:
                overloads[key] = name
                self._save_index(overloads)

    def locked_load(self, key):
        with _file_lock(lock_path):
            return load(self, key)

    cache_file.save, cache_file.load = locked_save, locked_load
    try:
        yield
    finally:
        cache_file.save, cache_file.load = save, load


def _run_unit(lock_path: str, unit: str) -> None:
    """A worker's whole job: run one unit's stages with its cache locked."""
    with _locked_cache(lock_path):
        _run(_units()[unit])


def _worker_count(units: int) -> int:
    from superglm._parallel import resolve_max_memory, resolve_n_jobs

    return max(1, min(resolve_n_jobs(), units, resolve_max_memory() // _WORKER_BYTES))


def _can_start_workers() -> bool:
    import numba
    from numba.core import caching

    return (
        not numba.config.DISABLE_JIT
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
    if numba.config.CACHE_DIR:  # set in code or by superglm._numba_cache, not only in env
        env["NUMBA_CACHE_DIR"] = str(numba.config.CACHE_DIR)
    order = sorted(units, key=lambda unit: -_UNIT_SECONDS.get(unit, 0.0))
    stop, guard, live = threading.Event(), threading.Lock(), set()
    try:
        scratch_directory = tempfile.TemporaryDirectory(prefix="superglm-warmup-")
    except OSError:  # no place for the lock: this process compiles every unit instead
        return
    with scratch_directory as scratch:
        lock = os.path.join(scratch, "cache.lock")

        def start(unit: str) -> None:
            # -P and the parent's sys.path: the worker imports this superglm,
            # never a module that happens to sit in the working directory. The
            # code goes in on stdin, which has no command line's length limit,
            # as ASCII (``ascii``), which every locale's encoding can carry.
            code = (
                f"import sys; sys.path[:] = {ascii(sys.path)}\n"
                f"from superglm._warmup import _run_unit; _run_unit({ascii(lock)}, {ascii(unit)})\n"
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
            with _locked_cache(lock):
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
