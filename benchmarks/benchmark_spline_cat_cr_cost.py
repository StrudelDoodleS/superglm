"""Complete-fit cost of the ``spline_cat`` cr interaction basis.

An interaction marginal on a ``Spline(kind="cr")`` parent resolves to
``CardinalCRSpline``.  Its basis functions are globally supported, so the
shared ``spline_cat`` basis goes from four nonzeros per row to K -- and it is
stored as CSR, which costs 12 bytes per stored entry against 8 for the same
matrix dense.  ``ps`` is unaffected and is carried here as the control.

Each case runs in its OWN process so ``ru_maxrss`` is a clean per-case peak
rather than a running high-water mark.

Usage::

    uv run python benchmarks/benchmark_spline_cat_cr_cost.py
    uv run python benchmarks/benchmark_spline_cat_cr_cost.py --n 100000 --k 20

Measured with this file, n=100,000, k=20, 4 levels, Poisson/log, median of 5.
"before" is a2611cc, the commit this interaction routing landed on; "after" is
that commit plus the routing.  The two differ only in
``features/interaction.py``:

| case        | ref    |   wall | peak RSS |   entries | basis MB |      deviance |
|-------------|--------|-------:|---------:|----------:|---------:|--------------:|
| exact ps    | before | 2.45 s |  388 MiB |   699,556 |     9.09 | 114330.085957 |
| exact ps    | after  | 2.50 s |  388 MiB |   699,556 |     9.09 | 114330.085957 |
| exact cr    | before | 2.52 s |  388 MiB |   699,544 |     9.09 | 114329.828614 |
| exact cr    | after  | 4.72 s |  472 MiB | 3,497,704 |    42.67 | 114330.094237 |
| discrete cr | before | 0.80 s |  366 MiB |     5,632 |     0.05 | 114329.178061 |
| discrete cr | after  | 0.54 s |  407 MiB |     5,120 |     0.04 | 114329.391279 |

``p`` is 79 on every row, and the dispatched group-matrix classes are identical
before and after -- the block dimension and the backend do not move, only what
is stored in them.  So the exact cr path costs 1.87x the wall time, 5.0x the
stored entries and 4.7x the basis bytes of the projected-B-spline marginal it
replaced, at +84 MiB peak RSS, for a fit that agrees to 2.3e-6 relative.
``ps`` is unmoved and the discrete path is if anything faster, because it
stores one basis row per bin rather than per observation.

Under cProfile the extra time is the cross-block gram, at unchanged call
counts (220 ``_cross_gram_by_columns``, ~7.2k sparse matvecs): 1.43 s before,
2.16 s after.  Two things are on the table if that is judged too expensive.
The stored form is one: at 12 bytes per entry (float64 + int32 index) against
8 dense, the 42.7 MB above would be 28.0 MB as a dense block, and the gram
would run on BLAS rather than a scalar CSR loop.  Row-support compression is
the other: ``SplineCategoricalGroupMatrix`` has no equivalent of the
``detect_row_support`` gate that ``SparseSSPGroupMatrix`` gets, so a repeated
rating variable claws none of this back on the exact path.
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

from superglm import SuperGLM
from superglm.features.categorical import Categorical
from superglm.features.spline import Spline

CASES = ("exact-ps", "exact-cr", "discrete-ps", "discrete-cr")


def _frame(n: int, levels: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    x = rng.gamma(2.0, 1.5, n)
    level = rng.integers(0, levels, n).astype(str)
    df = pd.DataFrame({"x": x, "f": pd.Categorical(level)})
    eta = 0.3 + 0.2 * np.log1p(x) + 0.1 * (level == "1")
    return df, rng.poisson(np.exp(eta)).astype(np.float64)


def _basis_cost(model) -> tuple[int, float]:
    """(stored entries, MB) held by the spline_cat blocks.

    The exact path stores a shared (n, K) CSR referenced by every level plus a
    per-level row subset of it; the discrete path stores one dense
    (n_bins, K) support block per level.  The shared block is counted once,
    keyed on its data buffer address, since each level holds its own
    csr_matrix wrapper around the same arrays.
    """
    entries = 0
    total = 0
    seen: set[int] = set()
    for gm in model._dm.group_matrices:
        if "SplineCategorical" not in type(gm).__name__:
            continue
        for block in (getattr(gm, "B", None), getattr(gm, "B_level", None)):
            if block is None or not sp.issparse(block):
                continue
            address = block.data.__array_interface__["data"][0]
            if address in seen:
                continue
            seen.add(address)
            entries += block.nnz
            total += block.data.nbytes + block.indices.nbytes + block.indptr.nbytes
        support = getattr(gm, "B_unique", None)
        if support is not None:
            address = support.__array_interface__["data"][0]
            if address not in seen:
                seen.add(address)
                entries += support.size
                total += support.nbytes
    return entries, total / 1e6


def run_case(case: str, n: int, k: int, levels: int, repeats: int) -> dict:
    discrete = case.startswith("discrete")
    kind = case.split("-")[1]
    df, y = _frame(n, levels)

    walls = []
    for _ in range(repeats):
        model = SuperGLM(
            family="poisson",
            features={"x": Spline(kind=kind, k=k), "f": Categorical()},
            interactions=[("x", "f")],
            discrete=discrete,
        )
        start = time.perf_counter()
        model.fit_reml(df, y)
        walls.append(time.perf_counter() - start)

    nnz, megabytes = _basis_cost(model)
    return {
        "case": case,
        "wall_median_s": float(np.median(walls)),
        "wall_min_s": float(np.min(walls)),
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "basis_nnz": nnz,
        "basis_mb": megabytes,
        "p": int(model._dm.p),
        "deviance": float(model._result.deviance),
        "backends": sorted({type(gm).__name__ for gm in model._dm.group_matrices}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--case", choices=CASES, default=None)
    args = parser.parse_args()

    if args.case is not None:
        print(json.dumps(run_case(args.case, args.n, args.k, args.levels, args.repeats)))
        return

    rows = []
    for case in CASES:
        child = subprocess.run(
            [
                *(sys.executable, __file__),
                *("--case", case),
                *("--n", str(args.n)),
                *("--k", str(args.k)),
                *("--levels", str(args.levels)),
                *("--repeats", str(args.repeats)),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        rows.append(json.loads(child.stdout.strip().splitlines()[-1]))

    print(f"n={args.n} k={args.k} levels={args.levels} repeats={args.repeats}")
    header = f"{'case':<12} {'wall med':>9} {'wall min':>9} {'peak RSS':>9} "
    header += f"{'basis nnz':>10} {'basis MB':>9} {'p':>4} {'deviance':>15}"
    print(header)
    for row in rows:
        line = f"{row['case']:<12} {row['wall_median_s']:>8.2f}s {row['wall_min_s']:>8.2f}s "
        line += f"{row['peak_rss_mib']:>7.0f}Mi {row['basis_nnz']:>10,} {row['basis_mb']:>9.1f} "
        line += f"{row['p']:>4} {row['deviance']:>15.6f}"
        print(line)
    for row in rows:
        print(f"{row['case']:<12} backends: {', '.join(row['backends'])}")


if __name__ == "__main__":
    main()
