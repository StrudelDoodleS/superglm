from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from superglm import Constraint, PSpline, SuperGLM

RESULTS_DIR = Path("benchmarks/results")
CSV_PATH = RESULTS_DIR / "benchmark_scop_exact_support.csv"


@dataclass
class Row:
    n: int
    repeated: bool
    runtime_s: float


def dataset(n: int, repeated: bool):
    if repeated:
        x = np.repeat(np.linspace(0.0, 1.0, n // 10), 10)
    else:
        x = np.linspace(0.0, 1.0, n)
    y = 0.4 + 0.7 * x + 1.2 * x**2
    return pd.DataFrame({"x": x}), y


def main():
    rows = []
    for repeated in (True, False):
        X, y = dataset(10_000, repeated)
        model = SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            features={"x": PSpline(n_knots=10, constraint=Constraint.fit.convex)},
        )
        t0 = time.perf_counter()
        model.fit(X, y)
        rows.append(Row(n=len(X), repeated=repeated, runtime_s=time.perf_counter() - t0))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(asdict(r) for r in rows).to_csv(CSV_PATH, index=False)
    print(pd.DataFrame(asdict(r) for r in rows).to_string(index=False))
    print(f"Wrote {CSV_PATH}")


if __name__ == "__main__":
    main()
