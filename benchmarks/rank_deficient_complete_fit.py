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

Two measurement caveats the reader has to carry.  Wall clock is min-of-N,
because a shared machine's median is not reproducible.  Peak RSS is
`ru_maxrss`, a high-water mark for the WHOLE process, so it compares across
runs only when both took the same number of fits -- the artifact records
`peak_rss_measures_fits` so that can be checked rather than assumed, and the
published comparison uses `--repeats 1` on both sides for exactly that reason.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info

from superglm import SuperGLM
from superglm.features import Categorical


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


def _blas_backend() -> dict[str, object]:
    """Which BLAS is LOADED IN THIS PROCESS, not which one numpy was built against.

    `np.show_config()` reports build dependencies, so it answers "what was this
    wheel compiled with" and would report the same string on a machine where a
    different library is actually dispatched to.  AGENTS.md asks for actual
    backend dispatch, so this reads the loaded shared objects instead --
    including the threading layer and thread count, which are the part that
    moves between runs on one machine.
    """
    loaded = [
        {
            "user_api": pool.get("user_api"),
            "internal_api": pool.get("internal_api"),
            "prefix": pool.get("prefix"),
            "version": pool.get("version"),
            "threading_layer": pool.get("threading_layer"),
            "num_threads": pool.get("num_threads"),
        }
        for pool in threadpool_info()
    ]
    return {"numpy": np.__version__, "loaded": sorted(loaded, key=lambda p: str(p["prefix"]))}


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

    frame, response = _design(args.levels, args.rows, args.seed)
    walls: list[float] = []
    model = None
    for _ in range(args.repeats):
        model = SuperGLM(
            family="gaussian",
            features={"g": Categorical(), "h": Categorical()},
            interactions=[("g", "h")],
        )
        started = time.perf_counter()
        model.fit(frame, response)
        walls.append(time.perf_counter() - started)

    assert model is not None
    result = model.result
    info = result.rank_info
    beta = np.asarray(result.beta, dtype=float)
    zeros = np.flatnonzero(beta == 0.0)
    payload = {
        "configuration": {
            "levels": args.levels,
            "rows": args.rows,
            "train_rows": int(len(frame)),
            "parameters": int(model._dm.p),
            "repeats": args.repeats,
        },
        "timing_seconds": {
            "min": round(min(walls), 4),
            "median": round(float(np.median(walls)), 4),
            "all": [round(w, 4) for w in walls],
        },
        "memory": {
            # ru_maxrss is a high-water mark for the WHOLE process, so it only
            # compares across runs when both took the same number of fits.  At
            # --repeats 1 it is the mark for exactly one complete fit.
            "peak_rss_measures_fits": args.repeats,
            "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
        },
        "numerical_outputs": {
            "effective_df": round(float(result.effective_df), 9),
            "deviance": round(float(result.deviance), 9),
            "beta_l2": round(float(np.linalg.norm(beta)), 9),
            "n_zero": int(zeros.size),
            "zero_index_sum": int(zeros.sum()),
            "n_iter": int(result.n_iter),
        },
        # "backend dispatch" for this change is which decomposition route each
        # retained system actually took -- that is the branch the work replaced.
        "backend_dispatch": {
            "data_method": info.data.method,
            "augmented_method": info.augmented.method,
            "coefficient_method": info.coefficient.method,
            "data_rank": int(info.data.rank),
            "augmented_rank": int(info.augmented.rank),
            "data_representative": info.data.pivots is not None,
            "augmented_representative": info.augmented.pivots is not None,
            "blas": _blas_backend(),
            "python": platform.python_version(),
        },
    }
    text = json.dumps(payload, indent=1, sort_keys=True)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    sys.exit(main())
