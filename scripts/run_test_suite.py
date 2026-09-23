"""Run the test suite the way CI does, locally or on a CI shard.

Stage 1 runs every selected test not marked ``threads`` under pytest-xdist,
with each worker pinned to one BLAS, OpenMP and numba thread. Unpinned
workers each start a full thread pool and oversubscribe the machine.
Stage 2 runs the ``threads`` tests serially with the default thread pools,
because they assert on thread counts or need a pool with more than one
thread, which a pinned worker cannot provide.

    uv run python scripts/run_test_suite.py                  # the full suite
    uv run python scripts/run_test_suite.py -m "not slow"    # the quick pass
    uv run python scripts/run_test_suite.py --splits 4 --group 1   # a CI shard

Every other argument goes to both stages; no ``--`` separator is needed, so
the same command line works under PowerShell. A ``--junitxml`` report from
stage 2 is written beside stage 1's (``-threads`` suffix), and ``--cov``
data from stage 2 is appended to stage 1's.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

PINNED_POOLS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
DEFAULT_MARKERS = "not browser and not docs"
NO_TESTS_COLLECTED = 5


def stage_commands(markers: str, passthrough: list[str]) -> list[tuple[list[str], dict[str, str]]]:
    """The two pytest invocations and the environment each adds."""
    base = [sys.executable, "-m", "pytest", "tests/", "-q", "--maxfail=0"]
    parallel = [*base, "-m", f"({markers}) and not threads", "-n", "auto", "--dist", "worksteal"]
    threaded = [*base, "-m", f"({markers}) and threads"]
    stage_two = []
    for arg in passthrough:
        if arg.startswith("--junitxml="):
            root, ext = os.path.splitext(arg.removeprefix("--junitxml="))
            arg = f"--junitxml={root}-threads{ext or '.xml'}"
        stage_two.append(arg)
    if any(arg.startswith("--cov") for arg in passthrough):
        stage_two.append("--cov-append")
    return [
        ([*parallel, *passthrough], dict.fromkeys(PINNED_POOLS, "1")),
        ([*threaded, *stage_two], {}),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-m", "--markers", default=DEFAULT_MARKERS, help="pytest -m expression")
    args, passthrough = parser.parse_known_args(argv)
    passthrough = [arg for arg in passthrough if arg != "--"]
    status = 0
    for command, pinned in stage_commands(args.markers, passthrough):
        code = subprocess.call(command, env={**os.environ, **pinned})
        # A shard may hold no ``threads`` test; an empty selection is not a failure.
        status = max(status, 0 if code == NO_TESTS_COLLECTED else code)
    return status


if __name__ == "__main__":
    sys.exit(main())
