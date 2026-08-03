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

Measured with this file, n=100,000, k=20, 4 levels, Poisson/log, 5 repeats.
"before" is a2611cc, the commit this interaction routing landed on; "after" is
that commit plus the routing.  The two differ only in
``features/interaction.py``:

| case        | ref    | wall med | wall min | peak RSS |   entries | basis MB |      deviance |
|-------------|--------|---------:|---------:|---------:|----------:|---------:|--------------:|
| exact ps    | before |   2.46 s |   2.40 s |  391 MiB |   699,556 |     9.09 | 114330.085957 |
| exact ps    | after  |   2.44 s |   2.41 s |  391 MiB |   699,556 |     9.09 | 114330.085957 |
| exact cr    | before |   2.46 s |   2.36 s |  391 MiB |   699,544 |     9.09 | 114329.828614 |
| exact cr    | after  |   4.43 s |   4.37 s |  473 MiB | 3,497,704 |    42.67 | 114330.094237 |
| discrete cr | before |   0.77 s |   0.77 s |  365 MiB |     5,632 |     0.05 | 114329.178061 |
| discrete cr | after  |   0.52 s |   0.51 s |  407 MiB |     5,120 |     0.04 | 114329.391279 |

Taken serially on an idle box (``/proc/loadavg`` under 2 on 16 cores).  Quote
the MIN column for ratios, not the median.  Contention only ever adds time, so
the min over repeats is the best available estimate of uncontended cost, and
this harness runs all repeats of one case before moving to the next -- a spike
lands inside a single case's block and the between-case ratio absorbs all of
it.  An earlier run of this table on a box at ``loadavg`` 62 is what that
warning is drawn from, and it is worth recording which column survived: the min
basis moved 1.84x -> 1.85x, while the median basis moved 1.87x -> 1.80x.  The
non-timing columns are deterministic given the seed and did not move at all.

``p`` is 79 on every row, and the dispatched group-matrix classes are identical
before and after -- the block dimension and the backend do not move, only what
is stored in them.  So the exact cr path costs 1.85x the wall time (min basis;
1.80x on medians), 5.0x the stored entries and 4.7x the basis bytes of the
projected-B-spline marginal it replaced, at +82 MiB peak RSS, for a fit that
agrees to 2.3e-6 relative.  ``ps`` is unmoved and the discrete path is if
anything faster, because it stores one basis row per bin rather than per
observation.

Under cProfile the extra time is the cross-block gram, at unchanged call
counts (220 ``_cross_gram_by_columns``, ~7.2k sparse matvecs): 1.43 s before,
2.16 s after.  Those two are single runs taken during the contended session and
have NOT been re-measured here, so read them as an attribution -- the extra time
is in the gram -- and not as a wall figure; the table above is the wall
measurement.  Two things were on the table if that is judged too expensive.
The stored form was one: at 12 bytes per entry (float64 + int32 index) against
8 dense, the 42.7 MB above would be 28.0 MB as a dense block, and the gram
would run on BLAS rather than a scalar CSR loop.  Row-support compression was
the other: ``SplineCategoricalGroupMatrix`` had no equivalent of the
``detect_row_support`` gate that ``SparseSSPGroupMatrix`` gets, so a repeated
rating variable clawed none of this back on the exact path.

### Row-support compression (issue #197)

The ``-repeated`` cases exist because the four above cannot see that lever at
all: their covariate is drawn from a continuous distribution, so no basis row
repeats and there is nothing to deduplicate.  They quantise the same gamma
margin to 72 distinct values -- a rating factor recorded in whole years -- and
change nothing else.

Both arms below were measured back to back in one window on a box with nothing
else on it, ``/proc/loadavg`` 0.79 to 1.67 at the start of each table and 0.93
to 1.61 at the end.  ``before`` is 4e6a09b, ``after`` is the compression
commit; the harness is this file in both arms.  Two independent pairs, min over
5 repeats:

| case              | before min  | after min   | before entries | after entries | before RSS  | after RSS |
|-------------------|------------:|------------:|---------------:|--------------:|------------:|----------:|
| exact ps          | 2.39/2.36 s | 2.43/2.37 s |        699,556 |       699,556 | 404/409 MiB |   404 MiB |
| exact cr          | 4.33/4.34 s | 4.32/4.31 s |      3,497,704 |     3,497,704 | 493/491 MiB |   491 MiB |
| discrete ps       | 0.68/0.70 s | 0.71/0.69 s |          5,120 |         5,120 |     383 MiB |   383 MiB |
| discrete cr       | 0.53/0.52 s | 0.54/0.51 s |          5,120 |         5,120 |     423 MiB |   423 MiB |
| exact ps repeated | 1.07/1.04 s | 0.91/0.89 s |        699,468 |         1,440 | 393/399 MiB |   381 MiB |
| exact cr repeated | 3.20/3.16 s | 1.13/1.09 s |      3,408,686 |         1,440 |     487 MiB |   423 MiB |

The dispatched backends move only for the two repeated cases.

**This is a numerical change, not a pure storage change.**  Compression
aggregates the level weights onto the shared support rows before the gram and
before ``compute_projected_R_inv``, which reorders a sum, so the fit moves at
rounding scale rather than reproducing bit-for-bit: coefficients differ by
3.5e-9 absolute on a weighted 6,000-row fit whose largest coefficient is below
1 in magnitude, with deviance agreeing to 7.8e-14 relative and effective df to
2.7e-11.  The deviance column above is identical to all twelve printed digits
in both arms, but that is the insensitivity of a minimised quantity to a
perturbation of its argument and must not be read as bit-identity.  The
release-gate test pins coefficients at rtol/atol 1e-7, which is looser than the
measured 3.5e-9 and tighter than anything a user could observe.

So on a repeated covariate the exact cr path costs **2.83x and 2.90x less wall
time on the two pairs, both min basis** (3.20 -> 1.13 s and 3.16 -> 1.09 s;
2.60x and 2.88x on the medians), stores **2367x fewer entries** (3,408,686 ->
1,440) and **64 MiB less peak RSS**; ps gains 1.18x and 1.17x on mins.  Quote
the min basis and say so, as the table above this section does.  The four
original cases sit in a 0.68-0.73, 2.36-2.43, 4.31-4.34 and 0.51-0.54 s band
with the before and after readings interleaved inside it, so nothing moves
outside run-to-run spread -- which is the point: the gate declines the
continuous covariate rather than paying for it.

### The stored form is still open, and the first measurement of it was wrong

Compression here is deduplication.  Storing the SAME rows densely instead of as
CSR is a separate lever, and it is NOT taken by this commit -- but the reason
first recorded for not taking it does not survive re-measurement, so it is
written down properly rather than left as a decided question.

Measured first at ``loadavg`` 16.87, routing a level block through a dense
per-level support made ``_cross_gram_by_columns`` 3.8x SLOWER, which alongside
its 2.11 s cumtime (against the gram's 1.42 s) read as a clear net loss.
Re-measured at ``loadavg`` 1.33, n=100,000, k=20, 4 levels, min over 5-7:

| operation                     | CSR      | dense per-level | dense/CSR |
|-------------------------------|---------:|----------------:|----------:|
| cr: 4 level grams             | 28.28 ms |         6.52 ms |     0.23x |
| cr: ``_cross_gram_by_columns``| 22.88 ms |        17.58 ms |     0.77x |
| ps: 4 level grams             |  1.89 ms |         7.26 ms |     3.84x |
| ps: ``_cross_gram_by_columns``|  6.72 ms |         6.72 ms |     1.00x |

The cross-gram SIGN INVERTS: 3.8x slower contended, 1.3x faster idle.  BLAS is
what contention starves, and the dense side is the BLAS side, so a loaded box
penalises exactly the arm under test.  Uncontended, dense wins both operations
for ``cr`` and loses the gram for ``ps`` -- which is the shape the existing cost
model already predicts (3.14x for cr, 0.15x for ps), so the calibrated gate
would route it correctly if it were allowed to see the case at all.

What blocks it is the strict ``n_support >= n_rows`` refusal in
``_passes_support_gates``: with a continuous covariate nothing repeats, so
dense storage is refused before the model is consulted.  Whether to relax that
needs a FIT-level number, which needs per-level support plumbing this commit
does not have (it shares one support across levels, which is right for
deduplication and 4x too much gram work when there is nothing to deduplicate).
Folding the ratios above into the profile's own weights PROJECTS ~1.5x on the
exact-cr fit; that is a projection from two measured ratios and measured
profile weights, not a measurement, and it is exactly the number a follow-up
should go and take properly.
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

CASES = (
    "exact-ps",
    "exact-cr",
    "discrete-ps",
    "discrete-cr",
    "exact-ps-repeated",
    "exact-cr-repeated",
)

# Distinct values of the spline covariate in the "-repeated" cases.  A rating
# factor recorded in whole years is the shape row-support compression exists
# for, and the four original cases cannot show it: their covariate is drawn
# from a continuous distribution, so every basis row is distinct.
REPEATED_DISTINCT = 72


def _frame(n: int, levels: int, seed: int = 7, distinct: int | None = None):
    rng = np.random.default_rng(seed)
    if distinct is None:
        x = rng.gamma(2.0, 1.5, n)
    else:
        # Same marginal, quantised to `distinct` values.
        pool = np.quantile(rng.gamma(2.0, 1.5, n), np.linspace(0.0, 1.0, distinct))
        x = pool[rng.integers(0, distinct, n)]
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
    repeated = case.endswith("-repeated")
    df, y = _frame(n, levels, distinct=REPEATED_DISTINCT if repeated else None)

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
