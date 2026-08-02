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
on 6,000 training rows, 1,680 parameters, 54 of them unidentifiable.  Wall clock
is min-of-N because a shared machine's median is not reproducible; peak RSS is
`ru_maxrss`, which is a high-water mark for the whole process and so is only
meaningful as the first fit in a fresh interpreter.
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
    """Which BLAS actually runs, not which one is installed."""
    try:
        config = np.show_config(mode="dicts")
    except TypeError:  # older numpy
        return {"numpy": np.__version__, "detail": "unavailable"}
    build = config.get("Build Dependencies", {}).get("blas", {})
    return {
        "numpy": np.__version__,
        "name": build.get("name"),
        "version": build.get("version"),
        "detection": build.get("detection method"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=int, default=41)
    parser.add_argument("--rows", type=int, default=12_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=31337)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

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
            "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
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
