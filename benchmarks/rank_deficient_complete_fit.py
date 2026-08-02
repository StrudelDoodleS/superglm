"""Complete-fit comparison for the rank-deficient decomposition change.

AGENTS.md requires performance-sensitive work to compare complete-fit timing,
memory, numerical outputs and actual backend dispatch against the relevant
baseline.  The alias-representative change is performance-sensitive -- it is the
reason a 41-level `cat_cat` refit went from minutes to seconds -- so it needs
all four in one place rather than an eigendecomposition count and an isolated
timing.

Run it on each side of the change and diff the JSON::

    uv run python benchmarks/rank_deficient_complete_fit.py --out branch.json
    # check out the baseline's src/, then
    uv run python benchmarks/rank_deficient_complete_fit.py --out baseline.json

The fit is the deficient one the change exists for: a 41-level `cat_cat` pair
on 6,000 training rows, 1,680 parameters, 54 of them unidentifiable.

Three measurement caveats the payload carries explicitly rather than leaving to
a reader, because each was got wrong once:

* wall clock is min-of-N, since a shared machine's median is not reproducible;
* `ru_maxrss` is a high-water mark for the WHOLE process, so it compares across
  runs only when both took the same number of fits -- `peak_rss_measures_fits`
  records that, and its unit differs by platform, so `ru_maxrss_unit` records
  that too;
* BLAS thread counts are changed inside the solver and restored on the way out,
  so the dispatch reading is sampled from a background thread WHILE a fit runs
  rather than around it.

`tests/test_rank_deficient_complete_fit.py` asserts those invariants of the
payload, which is what nothing was doing while five defects went by.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import threading
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info

from superglm import SuperGLM
from superglm.features import Categorical

# `ru_maxrss` is bytes on macOS and kibibytes everywhere else.  Dividing by
# 1024 unconditionally is right on Linux and 1024x wrong on macOS, while still
# labelled MiB.
_RSS_UNIT = "bytes" if sys.platform == "darwin" else "kib"
_RSS_DIVISOR = 1024.0**2 if _RSS_UNIT == "bytes" else 1024.0


def _design(levels: int, rows: int, seed: int):
    rng = np.random.default_rng(seed)
    left = rng.integers(0, levels, rows)
    right = rng.integers(0, levels, rows)
    y = (
        rng.normal(scale=0.5, size=levels)[left]
        + rng.normal(scale=0.5, size=levels)[right]
        + rng.normal(scale=1.0, size=rows)
    )
    frame = pd.DataFrame({"g": [f"G{i}" for i in left], "h": [f"H{i}" for i in right]})
    train = np.random.default_rng(seed).permutation(rows)[: rows // 2]
    return frame.iloc[train], y[train]


def _pool_snapshot() -> list[dict[str, object]]:
    return sorted(
        (
            {
                "user_api": pool.get("user_api"),
                "internal_api": pool.get("internal_api"),
                "prefix": pool.get("prefix"),
                "version": pool.get("version"),
                "threading_layer": pool.get("threading_layer"),
                "num_threads": pool.get("num_threads"),
            }
            for pool in threadpool_info()
        ),
        key=lambda pool: (str(pool["prefix"]), str(pool["version"])),
    )


class _DispatchSampler:
    """Read the loaded BLAS pools WHILE a fit runs, not after it.

    `np.show_config()` answers "what was this wheel built against", which is the
    same string on a machine dispatching elsewhere.  Reading
    `threadpool_info()` fixes that, but reading it after the fit still reports
    the ambient process: superglm sets solver thread counts on entry and
    restores them on exit, so the number that serviced the decomposition is
    only visible from inside.  A background sampler is the least invasive way
    to see it -- the alternative is monkeypatching production code from a
    benchmark.
    """

    def __init__(self, interval: float = 0.02) -> None:
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[list[dict[str, object]]] = []

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append(_pool_snapshot())
            self._stop.wait(self._interval)

    def __enter__(self) -> _DispatchSampler:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def observed(self) -> list[dict[str, object]]:
        """Every distinct pool configuration seen while the fit was running."""
        seen: list[dict[str, object]] = []
        for snapshot in self.samples:
            for pool in snapshot:
                if pool not in seen:
                    seen.append(pool)
        return sorted(seen, key=lambda pool: (str(pool["prefix"]), str(pool["version"])))


def measure(levels: int, rows: int, repeats: int, seed: int) -> dict[str, object]:
    """One complete-fit measurement, as the payload the artifact records."""
    frame, response = _design(levels, rows, seed)
    walls: list[float] = []
    model = None
    sampler = _DispatchSampler()
    for _ in range(repeats):
        model = SuperGLM(
            family="gaussian",
            features={"g": Categorical(), "h": Categorical()},
            interactions=[("g", "h")],
        )
        started = time.perf_counter()
        with sampler:
            model.fit(frame, response)
        walls.append(time.perf_counter() - started)

    if model is None:  # pragma: no cover - guarded at the flag
        raise SystemExit("no fit was run")
    result = model.result
    info = result.rank_info
    beta = np.asarray(result.beta, dtype=float)
    zeros = np.flatnonzero(beta == 0.0)
    return {
        "configuration": {
            "levels": levels,
            "rows": rows,
            "train_rows": int(len(frame)),
            "parameters": int(model._dm.p),
            "repeats": repeats,
            "seed": seed,
        },
        "timing_seconds": {
            "min": round(min(walls), 4),
            "median": round(float(np.median(walls)), 4),
            "all": [round(wall, 4) for wall in walls],
        },
        "memory": {
            "peak_rss_measures_fits": repeats,
            "ru_maxrss_unit": _RSS_UNIT,
            "peak_rss_mib": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / _RSS_DIVISOR, 1
            ),
        },
        "numerical_outputs": {
            "effective_df": round(float(result.effective_df), 9),
            "deviance": round(float(result.deviance), 9),
            "beta_l2": round(float(np.linalg.norm(beta)), 9),
            "n_zero": int(zeros.size),
            "zero_index_sum": int(zeros.sum()),
            "n_iter": int(result.n_iter),
        },
        # "backend dispatch" is two things: which decomposition route each
        # retained system took -- the branch this work replaced -- and which
        # BLAS actually serviced it.
        "backend_dispatch": {
            "data_method": info.data.method,
            "augmented_method": info.augmented.method,
            "coefficient_method": info.coefficient.method,
            "data_rank": int(info.data.rank),
            "augmented_rank": int(info.augmented.rank),
            "data_representative": info.data.pivots is not None,
            "augmented_representative": info.augmented.pivots is not None,
            "blas": {
                "numpy": np.__version__,
                "sampled_during_fit": bool(sampler.samples),
                "samples": len(sampler.samples),
                "pools_during_fit": sampler.observed(),
            },
            "python": platform.python_version(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=int, default=41)
    parser.add_argument("--rows", type=int, default=12_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=31337)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    for name, value, minimum in (
        ("--levels", args.levels, 2),
        ("--rows", args.rows, 2),
        ("--repeats", args.repeats, 1),
    ):
        if value < minimum:
            raise SystemExit(f"{name} must be >= {minimum}, got {value}")

    payload = measure(args.levels, args.rows, args.repeats, args.seed)
    text = json.dumps(payload, indent=1, sort_keys=True)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
