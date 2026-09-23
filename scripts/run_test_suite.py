"""Run the test suite the way CI does, locally or on a CI shard.

Stage 1 runs every selected test not marked ``threads`` under pytest-xdist,
with one worker per logical CPU and each worker pinned to one BLAS, OpenMP
and numba thread. Unpinned workers each start a full thread pool and
oversubscribe the machine. Stage 2 runs the ``threads`` tests serially with
the default thread pools, whatever the calling shell pinned, because they
assert on thread counts or need a pool with more than one thread.

    uv run python scripts/run_test_suite.py                  # the full suite
    uv run python scripts/run_test_suite.py -m "not slow"    # the quick pass
    uv run python scripts/run_test_suite.py --splits 4 --group 1   # a CI shard

Every other argument goes to both stages; no ``--`` separator is needed, so
the same command line works under PowerShell. Select tests with ``-m`` or
``-k``: the runner always runs ``tests/``, so file paths are not supported.
A junit report from stage 2 is written beside stage 1's (``-threads``
suffix), and coverage from stage 2 is appended to stage 1's. Both stages
need the ``dev`` extra's plugins loaded: stage 1 runs under pytest-xdist,
and stage 2 always passes ``--cov-append``, which pytest-cov defines.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

PINNED_POOLS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
# The solver's own BLAS cap: a caller's value would widen a pinned worker's
# pool inside every fit, or change what the threads tests see.
SOLVER_BLAS_OVERRIDE = "SUPERGLM_BLAS_THREADS"
DEFAULT_MARKERS = "not browser and not docs"
JUNIT_OPTIONS = ("--junitxml", "--junit-xml")
NO_TESTS_COLLECTED = 5
ROOT = Path(__file__).resolve().parents[1]


def _beside(path: str) -> str:
    root, ext = os.path.splitext(path)
    return f"{root}-threads{ext or '.xml'}"


def _threads_arguments(passthrough: list[str]) -> list[str]:
    """Stage 2's arguments: its junit report beside stage 1's, coverage appended."""
    arguments, rename_next = [], False
    for arg in passthrough:
        if rename_next:
            arg, rename_next = _beside(arg), False
        elif arg in JUNIT_OPTIONS:
            rename_next = True
        elif arg.startswith(tuple(f"{option}=" for option in JUNIT_OPTIONS)):
            option, _, path = arg.partition("=")
            arg = f"{option}={_beside(path)}"
        arguments.append(arg)
    # Coverage may be enabled here, in PYTEST_ADDOPTS or in the pytest config;
    # appending keeps stage 1's data in every case and does nothing without it.
    return ["--cov-append", *arguments]


def stage_commands(markers: str, passthrough: list[str]) -> list[tuple[list[str], bool]]:
    """The two pytest invocations, each with whether its thread pools are pinned."""
    base = [sys.executable, "-m", "pytest", "tests/", "-q", "--maxfail=0"]
    parallel = [*base, "-m", f"({markers}) and not threads", "-n", "logical", "--dist", "worksteal"]
    threaded = [*base, "-m", f"({markers}) and threads"]
    return [
        ([*parallel, *passthrough], True),
        # Last, so a -n from the arguments or PYTEST_ADDOPTS cannot distribute it.
        ([*threaded, *_threads_arguments(passthrough), "-n", "0"], False),
    ]


def stage_environment(pinned: bool, base: Mapping[str, str]) -> dict[str, str]:
    """The caller's environment with every pool pinned to one thread, or none pinned."""
    inherited = (*PINNED_POOLS, SOLVER_BLAS_OVERRIDE)
    environment = {key: value for key, value in base.items() if key not in inherited}
    if pinned:
        environment.update(dict.fromkeys(PINNED_POOLS, "1"))
    return environment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Only -m: pytest's own --markers lists the registered markers.
    parser.add_argument("-m", dest="markers", default=DEFAULT_MARKERS, help="pytest -m expression")
    # Drop "--" before parsing: a -m after it would reach pytest and override both stages.
    argv = [arg for arg in (sys.argv[1:] if argv is None else argv) if arg != "--"]
    args, passthrough = parser.parse_known_args(argv)
    codes = [
        subprocess.call(command, cwd=ROOT, env=stage_environment(pinned, os.environ))
        for command, pinned in stage_commands(args.markers, passthrough)
    ]
    # Either stage may be empty (a shard without ``threads`` tests, a threads-only
    # selection); both empty means the selection matched nothing.
    if all(code == NO_TESTS_COLLECTED for code in codes):
        return NO_TESTS_COLLECTED
    failed = [code for code in codes if code not in (0, NO_TESTS_COLLECTED)]
    if not failed:
        return 0
    return failed[0] if failed[0] > 0 else 1  # a signal (negative code) is a failure too


if __name__ == "__main__":
    sys.exit(main())
