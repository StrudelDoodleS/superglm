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
    same string on a machine dispatching elsewhere.  Reading `threadpool_info()`
    fixes that, but reading it after the fit reports the ambient process:
    superglm caps BLAS to one thread on fit entry and releases the cap again for
    a wide design, so what was configured during the fit is only visible from
    inside.

    Three properties this has to have, each of which it once lacked:

    * REUSABLE.  `--repeats` defaults to 3, and a sampler whose stop flag is
      never cleared samples the first fit and silently records nothing for the
      rest.  `__enter__` clears it.
    * BOUNDED.  Retaining a snapshot per tick makes the sampler's own footprint
      proportional to the RUNTIME of the side being measured, which lands in the
      peak RSS this benchmark reports and biases it toward the faster side.
      Configurations are folded in as they arrive, so the store is
      O(distinct configurations) -- three or four -- rather than O(duration).
    * DWELL-COUNTING.  A configuration seen once in twelve thousand samples and
      one seen throughout are completely different claims, and discarding the
      count cannot tell them apart.  Each distinct configuration carries the
      number of samples that saw it.
    """

    def __init__(self, interval: float = 0.02) -> None:
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # configuration -> samples that saw it; folded in as they arrive so the
        # store cannot grow with the duration of the fit
        self._dwell: dict[tuple, int] = {}
        self.samples = 0

    def _run(self) -> None:
        while not self._stop.is_set():
            for pool in _pool_snapshot():
                key = tuple(sorted(pool.items(), key=lambda item: item[0]))
                self._dwell[key] = self._dwell.get(key, 0) + 1
            self.samples += 1
            self._stop.wait(self._interval)

    def __enter__(self) -> _DispatchSampler:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def observed(self) -> list[dict[str, object]]:
        """Distinct pool configurations seen during the fits, with their dwell.

        `samples_seen_in` is what distinguishes a configuration that held for
        the whole fit from one caught during a brief setup window.
        """
        rows = []
        for key, count in self._dwell.items():
            row = dict(key)
            row["samples_seen_in"] = count
            row["fraction_of_samples"] = round(count / max(self.samples, 1), 4)
            rows.append(row)
        return sorted(rows, key=lambda pool: (str(pool["prefix"]), str(pool["version"])))


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
                "sampled_during_fit": sampler.samples > 0,
                "samples": sampler.samples,
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
