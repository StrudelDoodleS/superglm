"""Worker counts for superglm's pooled kernels: one parallelism level, capped by memory.

A pooled kernel splits its work into tasks whose partition does not depend on
the worker count, runs them on Python threads, and pins BLAS to one thread
for the duration (``_blas_threads.pooled_blas_threads``), so the pool is the
only parallelism level and the result is the same for any number of workers.
This module only decides how many workers a kernel may use:

``n_jobs``
    The most workers any pooled kernel starts.  ``"auto"`` (the default) is
    the number of physical cores (``joblib.cpu_count(only_physical_cores=
    True)``, which also honours the process's CPU affinity and cgroup quota).
``max_memory``
    Bytes the concurrently running tasks may hold.  ``"auto"`` is a quarter
    of the memory the process may use: physical memory, or its cgroup's
    limit where that is tighter (a container or a systemd scope).  A kernel states its per-task working set, and the
    worker count is ``min(n_jobs, tasks, max_memory // task_bytes)``, at
    least one: CPU time floats, peak memory is the constraint.

Both are ``SuperGLM`` parameters: a fit runs inside :func:`estimator_scope`,
which applies the estimator's values, and ``"auto"`` there defers to the
process default.  The process default comes from ``SUPERGLM_N_JOBS`` and
``SUPERGLM_MAX_MEMORY`` (an integer, ``auto``, or for memory a number with a
``K``/``M``/``G`` binary suffix), and :func:`parallel_config` overrides both
for one context, which is how tests pin the worker count.

The pooled kernels are the data-rank factor's TSQR leaves
(``solvers.rank``) and the Gram's blocks (``_group_matrix._block_queue``).
"""

from __future__ import annotations

import os
import sys
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath

import numpy as np

_N_JOBS_ENV = "SUPERGLM_N_JOBS"
_MAX_MEMORY_ENV = "SUPERGLM_MAX_MEMORY"
# Share of physical memory the pooled tasks may hold by default, and the
# budget when the platform does not report its memory.
_MEMORY_FRACTION = 0.25
_FALLBACK_MAX_MEMORY = 2 << 30
_SUFFIXES = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}
# Where a Linux process reads its cgroup path and its controllers' limits.
_PROC_CGROUP = "/proc/self/cgroup"
_CGROUP_ROOT = "/sys/fs/cgroup"


@dataclass(frozen=True)
class _Override:
    n_jobs: int | None
    max_memory: int | None


_override: ContextVar[_Override | None] = ContextVar("superglm_parallel_override", default=None)


@contextmanager
def parallel_config(
    *, n_jobs: int | str | None = None, max_memory: int | str | None = None
) -> Iterator[None]:
    """Override ``n_jobs`` and ``max_memory`` for the current context.

    ``None`` keeps the enclosing value.  The override follows the calling
    thread's context, so concurrent fits in other threads keep their own.
    """
    outer = _override.get()
    resolved = _Override(
        n_jobs=_parse_n_jobs(n_jobs) if n_jobs is not None else (outer and outer.n_jobs),
        max_memory=(
            _parse_memory(max_memory) if max_memory is not None else (outer and outer.max_memory)
        ),
    )
    token = _override.set(resolved)
    try:
        yield
    finally:
        _override.reset(token)


def _parse_n_jobs(value: int | str) -> int:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("", "auto"):
            return physical_cores()
        value = int(text)
    if isinstance(value, bool) or int(value) < 1:
        raise ValueError(f"n_jobs must be a positive integer or 'auto', got {value!r}")
    return int(value)


def _parse_memory(value: int | str) -> int:
    if isinstance(value, str):
        text = value.strip().upper().removesuffix("B").removesuffix("I")
        if text in ("", "AUTO"):
            return default_max_memory()
        scale = _SUFFIXES.get(text[-1:], 1)
        number = text[:-1] if text[-1:] in _SUFFIXES else text
        value = int(float(number) * scale)
    if isinstance(value, bool) or int(value) < 1:
        raise ValueError(f"max_memory must be a positive byte count or 'auto', got {value!r}")
    return int(value)


@lru_cache(maxsize=1)
def physical_cores() -> int:
    """Physical cores available to this process, at least one."""
    try:
        import joblib

        with warnings.catch_warnings():
            # joblib warns when it falls back to logical cores; that fallback
            # is still a usable worker count.
            warnings.simplefilter("ignore")
            count = int(joblib.cpu_count(only_physical_cores=True))
    except Exception:  # pragma: no cover - joblib is a dependency
        count = os.cpu_count() or 1
    return max(count, 1)


def _physical_memory() -> int | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and size > 0:
            return int(pages) * int(size)
    except (AttributeError, OSError, ValueError):
        pass
    if sys.platform == "win32":  # pragma: no cover - platform specific
        try:
            import ctypes

            class _MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatus()
            status.dwLength = ctypes.sizeof(_MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys)
        except Exception:
            return None
    return None


def _cgroup_limit(path: Path) -> int | None:
    try:
        text = path.read_text().strip()
    except OSError:
        return None
    try:
        limit = int(text)
    except ValueError:  # cgroup v2 writes "max" for no limit
        return None
    return limit if limit > 0 else None


def _cgroup_memory_limit() -> int | None:
    """The tightest memory limit on this process's cgroup and its ancestors, else ``None``.

    cgroup v2 ``memory.max`` and v1 ``memory.limit_in_bytes``, read along the
    path ``/proc/self/cgroup`` gives up to the mount root, which is where a
    container sees its own limit.  A v1 "unlimited" value is a page-counter
    maximum far above physical memory, so taking the minimum ignores it.
    """
    try:
        lines = Path(_PROC_CGROUP).read_text().splitlines()
    except OSError:
        return None
    limits = []
    for line in lines:
        hierarchy, _, rest = line.partition(":")
        controllers, _, path = rest.partition(":")
        if hierarchy == "0" and controllers == "":
            base, name = Path(_CGROUP_ROOT), "memory.max"
        elif "memory" in controllers.split(","):
            base, name = Path(_CGROUP_ROOT) / "memory", "memory.limit_in_bytes"
        else:
            continue
        relative = PurePosixPath(path or "/")
        for directory in (relative, *relative.parents):
            limit = _cgroup_limit(base / str(directory).lstrip("/") / name)
            if limit is not None:
                limits.append(limit)
    return min(limits) if limits else None


@lru_cache(maxsize=1)
def default_max_memory() -> int:
    """A quarter of the memory this process may use, or 2 GiB where the platform does not say.

    The tighter of physical memory and the process's cgroup limit, so a
    memory-limited container or scope sizes the pool by its own limit, not
    the host's RAM.
    """
    sizes = [size for size in (_physical_memory(), _cgroup_memory_limit()) if size is not None]
    if not sizes:
        return _FALLBACK_MAX_MEMORY
    return max(int(min(sizes) * _MEMORY_FRACTION), 1)


def _from_env(name: str, parse) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return parse(raw)
    except ValueError:
        warnings.warn(
            f"{name}={raw!r} is not a positive value or 'auto'; using the automatic default",
            stacklevel=4,
        )
        return None


def resolve_n_jobs() -> int:
    """The context's ``n_jobs``, else ``SUPERGLM_N_JOBS``, else the physical cores."""
    override = _override.get()
    if override is not None and override.n_jobs is not None:
        return override.n_jobs
    from_env = _from_env(_N_JOBS_ENV, _parse_n_jobs)
    return physical_cores() if from_env is None else from_env


def resolve_max_memory() -> int:
    """The context's ``max_memory``, else ``SUPERGLM_MAX_MEMORY``, else a quarter of RAM."""
    override = _override.get()
    if override is not None and override.max_memory is not None:
        return override.max_memory
    from_env = _from_env(_MAX_MEMORY_ENV, _parse_memory)
    return default_max_memory() if from_env is None else from_env


def validate_n_jobs(value: int | str) -> int | str:
    """An estimator's ``n_jobs`` as given, once checked: ``"auto"`` or a positive integer."""
    if isinstance(value, str) and value.strip().lower() == "auto":
        return "auto"
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or int(value) < 1:
        raise ValueError(f"n_jobs must be a positive integer or 'auto', got {value!r}")
    return int(value)


def validate_max_memory(value: int | str) -> int | str:
    """An estimator's ``max_memory`` as given, once checked.

    ``"auto"``, a positive byte count, or a string such as ``"4G"`` or
    ``"512M"`` (binary suffixes ``K``, ``M``, ``G``, ``T``).
    """
    if isinstance(value, str) and value.strip().lower() == "auto":
        return "auto"
    if isinstance(value, bool) or not isinstance(value, int | np.integer | str):
        raise ValueError(f"max_memory must be a positive byte count or 'auto', got {value!r}")
    try:
        _parse_memory(value if isinstance(value, str) else int(value))
    except ValueError as exc:
        raise ValueError(
            f"max_memory must be a positive byte count, a size such as '4G', or 'auto', "
            f"got {value!r}"
        ) from exc
    return value if isinstance(value, str) else int(value)


@contextmanager
def estimator_scope(n_jobs: int | str = "auto", max_memory: int | str = "auto") -> Iterator[None]:
    """Apply an estimator's ``n_jobs`` and ``max_memory`` for one fit; ``"auto"`` keeps the default."""
    with parallel_config(
        n_jobs=None if n_jobs == "auto" else n_jobs,
        max_memory=None if max_memory == "auto" else max_memory,
    ):
        yield


def pool_workers(n_tasks: int, task_bytes: int) -> int:
    """Workers for ``n_tasks`` tasks of ``task_bytes`` each: ``n_jobs`` capped by tasks and memory.

    At least one, so a task larger than the whole budget still runs (alone).
    """
    by_memory = resolve_max_memory() // max(int(task_bytes), 1)
    return max(1, min(resolve_n_jobs(), int(n_tasks), int(by_memory)))
