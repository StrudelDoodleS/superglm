"""Two readings that turn a screening rank into a fitting decision.

`z` says whether a pair carries signal.  It does not say whether refitting that
pair will help, and at wide factors the two answers routinely disagree: a 41x41
`cat_cat` pair carrying a genuine 6-sigma effect in 5 of its 1681 cells scores
z = 17.75 -- unambiguous by any conventional reading -- and costs +22% holdout
MSE when refitted as a fixed interaction.  A screen that only reports z hands
the caller a number that is correct and still points the wrong way.

Two derived numbers close that gap.  Neither is new machinery; both are
arithmetic on quantities the screen already produces.

1. WORTH THRESHOLD.  Mallows' Cp says a term earns its place when its score
   beats twice the df it spends, `T > 2*edf0`.  The evaluation guide already
   notes `gain - 2*edf` as a scoring variant; what is added here is the same
   rule on PSST's own z scale, plus a measurement of where the crossing lands.
   Since `z = (T/phi - edf0) / sqrt(2*edf0)`,

       T > 2*edf0   <=>   z > sqrt(edf0 / 2)

   The bar GROWS with the block's df -- z > 4.95 at 8x8, z > 28.3 at 41x41 --
   which is exactly what a constant cutoff cannot express.

2. CONCENTRATION.  Every chi^2-family score reads only the TOTAL: PSST's `T`,
   FAST's RSS gain, Information Value, mutual information, deviance change.
   None reads the shape, so none can separate 5 live cells from 1681 faintly
   live ones carrying the same total.  The participation ratio of the per-cell
   contributions does:

       P = (sum_c t_c)^2 / sum_c t_c^2,    t_c = n_c * mean_c^2 / phi

   For k independent chi^2_1 contributions E[t] = 1 and E[t^2] = 3, so the null
   sits at P = k/3.  Reporting `P / (k/3)` gives ~1 for noise AND for genuinely
   diffuse truths, and near zero when a handful of cells carry everything.

Run:  uv run python benchmarks/screening_worth_gate.py [--reps 3]

Four tables come out, because the threshold alone answers only half of it:

  1. the gate ladder -- table width x effect size, the screen's z, and the
     holdout change from actually refitting.  The question is whether the sign
     of the holdout change flips where `z / sqrt(edf0/2)` crosses 1;
  2. concentration at matched z -- spiky and diffuse truths tuned so their z
     values coincide, which is the only honest way to ask whether `P` carries
     information `z` does not;
  3. the sparse payoff -- if `P` says the signal is concentrated, does fitting
     only those cells pay?  Cells are ranked on TRAINING residuals only;
     ranking them on the full sample is the target leakage that makes
     supervised binning look better than it is;
  4. the same pair through three model classes -- mains, a fixed interaction,
     and the cell partially pooled.  The gate says what NOT to do with a wide
     pair; this says whether the pair is worth having at all in some other
     class, and it is where the guide's +22.5% and -3.2% come from.

These are measurements of a simulated Gaussian book, not calibrated quantiles
and not p-values.  `2*edf` is a Gaussian-family argument; the constant has not
been checked for Poisson with exposure.  Expect ~45 minutes at the defaults --
most of it in the wide fixed refits, which is itself part of the finding.  Their
wall-clock moves several-fold under CPU contention; the holdout columns do not,
so do not read the timings as a benchmark of the fitting paths.
"""

from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from superglm import SuperGLM
from superglm.features import Categorical
from superglm.features.random_effect import RandomEffect

# widths, and effect sizes chosen so z straddles sqrt(edf0/2) at each width
LADDER: dict[int, tuple[float, ...]] = {
    8: (0.12, 0.23, 0.40, 0.70),
    16: (0.50, 1.00, 1.60, 2.50),
    25: (1.20, 2.50, 4.00, 6.00),
    32: (2.00, 4.00, 6.50, 9.00),
}
# spiky/diffuse magnitudes paired so each row's z matches its partner's
MATCHED: tuple[tuple[str, str, float], ...] = (
    ("noise", "none", 0.0),
    ("spiky 5 cells @ 4.0", "spike", 4.0),
    ("diffuse sd=0.20", "diffuse", 0.20),
    ("spiky 5 cells @ 6.0", "spike", 6.0),
    ("diffuse sd=0.30", "diffuse", 0.30),
    ("spiky 5 cells @ 8.0", "spike", 8.0),
    ("diffuse sd=0.41", "diffuse", 0.41),
)
TOP_M: tuple[int, ...] = (5, 10, 25, 50)
HOT_CELLS = 5
# the three model classes table 4 compares.  `pooled` is load-bearing: it is the
# only arm that shows the pair is worth HAVING once it stops being 1681 free
# cells, and the guide quotes its holdout gain.
SHRINKAGE_ARMS: tuple[str, ...] = ("mains", "fixed", "pooled")


def worth_threshold(edf0: float) -> float:
    """z a pair must clear for a plain fixed refit to pay for its own df.

    `T > 2*edf0` (Mallows' Cp) expressed on the z scale the screen reports.
    """
    return float(np.sqrt(edf0 / 2.0))


def cell_contributions(
    resid: NDArray, joint: NDArray, n_cells: int, phi: float
) -> tuple[NDArray, int]:
    """Per-cell score contributions ``t_c = n_c * mean_c^2 / phi``.

    Returns the contributions and the number of OCCUPIED cells -- empty cells
    contribute nothing and must not count toward the null, which is why the
    caller gets the occupancy back rather than assuming ``n_cells``.
    """
    count = np.bincount(joint, minlength=n_cells).astype(float)
    total = np.bincount(joint, weights=resid, minlength=n_cells)
    occupied = count > 0
    t = np.zeros(n_cells)
    t[occupied] = total[occupied] ** 2 / count[occupied] / phi
    return t, int(occupied.sum())


def participation_ratio(t: NDArray) -> float:
    """Effective number of contributing cells, ``(sum t)^2 / sum t^2``.

    Equals k when k cells contribute equally and 1 when a single cell carries
    everything, so it reads the shape of a score the total cannot see.
    """
    denom = float(np.sum(np.asarray(t, dtype=float) ** 2))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(t) ** 2 / denom)


def concentration(t: NDArray, n_occupied: int) -> float:
    """``participation_ratio`` against its null of ``k/3``.

    ~1 means the score is spread as noise spreads it -- which a genuinely
    diffuse truth also does.  Near 0 means a few cells carry the pair.
    """
    if n_occupied <= 0:
        return float("nan")
    return participation_ratio(t) / (n_occupied / 3.0)


@dataclass(frozen=True)
class PairData:
    frame: pd.DataFrame
    y: NDArray
    joint: NDArray
    n_levels: int


def _make(kind: str, magnitude: float, n_levels: int, n: int, seed: int) -> PairData:
    rng = np.random.default_rng(seed)
    left = rng.integers(0, n_levels, n)
    right = rng.integers(0, n_levels, n)
    eta = rng.normal(scale=0.5, size=n_levels)[left] + rng.normal(scale=0.5, size=n_levels)[right]
    cell = np.zeros((n_levels, n_levels))
    if kind == "spike":
        cell.flat[rng.choice(n_levels * n_levels, size=HOT_CELLS, replace=False)] = magnitude
    elif kind == "diffuse":
        cell[:] = rng.normal(scale=magnitude, size=(n_levels, n_levels))
    y = eta + cell[left, right] + rng.normal(scale=1.0, size=n)
    frame = pd.DataFrame({"g": [f"G{i}" for i in left], "h": [f"H{i}" for i in right]})
    return PairData(frame=frame, y=y, joint=left * n_levels + right, n_levels=n_levels)


def _mains(frame: pd.DataFrame, y: NDArray) -> SuperGLM:
    model = SuperGLM(family="gaussian", features={"g": Categorical(), "h": Categorical()})
    model.fit(frame[["g", "h"]], y)
    return model


def _split(n: int, seed: int) -> tuple[NDArray, NDArray]:
    order = np.random.default_rng(seed).permutation(n)
    return order[: n // 2], order[n // 2 :]


def _run_gate_ladder(reps: int, n: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for n_levels, effects in LADDER.items():
        for effect in effects:
            zs, deltas = [], []
            for rep in range(reps):
                data = _make("spike", effect, n_levels, n, 7000 + rep)
                train, test = _split(n, 7000 + rep)
                dtr, ytr = data.frame.iloc[train], data.y[train]
                dte, yte = data.frame.iloc[test], data.y[test]

                mains = _mains(dtr, ytr)
                zs.append(float(mains.screen_interactions(dtr, ytr).iloc[0]["z"]))
                full = SuperGLM(
                    family="gaussian",
                    features={"g": Categorical(), "h": Categorical()},
                    interactions=[("g", "h")],
                )
                full.fit(dtr, ytr)
                base = float(np.mean((mains.predict(dte) - yte) ** 2))
                with_pair = float(np.mean((full.predict(dte) - yte) ** 2))
                deltas.append((with_pair / base - 1.0) * 100.0)

            edf0 = float((n_levels - 1) ** 2)
            z = float(np.mean(zs))
            delta = float(np.mean(deltas))
            rows.append(
                {
                    "n_levels": n_levels,
                    "edf0": edf0,
                    "effect": effect,
                    "z": z,
                    "threshold": worth_threshold(edf0),
                    "delta_pct": delta,
                    "agrees": (z > worth_threshold(edf0)) == (delta < 0.0),
                }
            )
    return rows


def _run_concentration(reps: int, n: int, n_levels: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label, kind, magnitude in MATCHED:
        zs, cons, occs = [], [], []
        for rep in range(reps):
            data = _make(kind, magnitude, n_levels, n, 31337 + rep)
            mains = _mains(data.frame, data.y)
            zs.append(float(mains.screen_interactions(data.frame, data.y).iloc[0]["z"]))
            resid = data.y - mains.predict(data.frame)
            phi = float(np.var(resid, ddof=1))
            t, occupied = cell_contributions(resid, data.joint, n_levels * n_levels, phi)
            cons.append(concentration(t, occupied))
            occs.append(occupied)
        rows.append(
            {
                "label": label,
                "z": float(np.mean(zs)),
                "occupied": float(np.mean(occs)),
                "concentration": float(np.mean(cons)),
            }
        )
    return rows


def _run_sparse_payoff(reps: int, n: int, n_levels: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for kind, magnitude in (("spike", 6.0), ("diffuse", 0.30)):
        acc: dict[int, list[float]] = {m: [] for m in TOP_M}
        for rep in range(reps):
            data = _make(kind, magnitude, n_levels, n, 31337 + rep)
            train, test = _split(n, 31337 + rep)
            dtr, ytr, jtr = data.frame.iloc[train], data.y[train], data.joint[train]
            dte, yte, jte = data.frame.iloc[test], data.y[test], data.joint[test]

            mains = _mains(dtr, ytr)
            base = float(np.mean((mains.predict(dte) - yte) ** 2))
            resid = ytr - mains.predict(dtr)
            # ranked on TRAINING residuals only -- ranking on the full sample
            # would leak the test rows into the cell choice
            t, _ = cell_contributions(resid, jtr, n_levels * n_levels, 1.0)

            for m in TOP_M:
                hot = set(np.argsort(t)[-m:].tolist())
                a, b = dtr.copy(), dte.copy()
                a["hot"] = np.where([j in hot for j in jtr], jtr.astype(str), "other")
                b["hot"] = np.where([j in hot for j in jte], jte.astype(str), "other")
                model = SuperGLM(
                    family="gaussian",
                    features={
                        "g": Categorical(),
                        "h": Categorical(),
                        "hot": Categorical(),
                    },
                )
                model.fit(a, ytr)
                got = float(np.mean((model.predict(b) - yte) ** 2))
                acc[m].append((got / base - 1.0) * 100.0)
        for m, vals in acc.items():
            rows.append({"kind": kind, "top_m": m, "delta_pct": float(np.mean(vals))})
    return rows


def _shrinkage_spec(arm: str) -> tuple[dict[str, object], list[str]]:
    """Model spec for one arm of table 4.

    `fit_reml` finds no REML-eligible group in a pure-`Categorical` model and
    falls back to `fit()`, so `mains` and `fixed` are plain fits and only
    `pooled` estimates a variance component.  That asymmetry IS the comparison:
    the fixed arm spends its df whatever the data says, the pooled arm spends
    what REML thinks the data supports.
    """
    if arm == "mains":
        return {"features": {"g": Categorical(), "h": Categorical()}}, ["g", "h"]
    if arm == "fixed":
        return (
            {
                "features": {"g": Categorical(), "h": Categorical()},
                "interactions": [("g", "h")],
            },
            ["g", "h"],
        )
    if arm == "pooled":
        return (
            {
                "features": {
                    "g": Categorical(),
                    "h": Categorical(),
                    "gh": RandomEffect(),
                }
            },
            ["g", "h", "gh"],
        )
    raise ValueError(f"unknown shrinkage arm: {arm!r}")


def _run_shrinkage(reps: int, n: int, n_levels: int) -> list[dict[str, object]]:
    """mains vs a fixed cat_cat interaction vs the same cell partially pooled.

    The gate says a wide pair should not be refit as a fixed interaction. This
    asks the follow-on question the gate does not answer: is the pair worth
    having at all, in some other model class?
    """
    acc: dict[str, list[tuple[float, float, float, float, float]]] = {
        arm: [] for arm in SHRINKAGE_ARMS
    }
    for rep in range(reps):
        data = _make("spike", 6.0, n_levels, n, 4242 + rep)
        frame = data.frame.copy()
        frame["gh"] = data.joint.astype(str)
        train, test = _split(n, 4242 + rep)
        ytr, yte = data.y[train], data.y[test]

        for arm in SHRINKAGE_ARMS:
            kwargs, cols = _shrinkage_spec(arm)
            dtr, dte = frame[cols].iloc[train], frame[cols].iloc[test]
            started = time.perf_counter()
            model = SuperGLM(family="gaussian", **kwargs)
            model.fit_reml(dtr, ytr)
            seconds = time.perf_counter() - started
            result = model._result
            acc[arm].append(
                (
                    seconds,
                    float(np.asarray(result.beta).size),
                    float(result.effective_df),
                    float(np.mean((model.predict(dtr) - ytr) ** 2)),
                    float(np.mean((model.predict(dte) - yte) ** 2)),
                )
            )
            print(f"  rep {rep} {arm} done", flush=True)

    rows: list[dict[str, object]] = []
    for arm in SHRINKAGE_ARMS:
        values = np.array(acc[arm])
        rows.append(
            {
                "model": arm,
                "seconds": float(values[:, 0].mean()),
                "params": float(values[:, 1].mean()),
                "edf": float(values[:, 2].mean()),
                "train": float(values[:, 3].mean()),
                "holdout": float(values[:, 4].mean()),
            }
        )
    return rows


def _print_gate_ladder(rows: list[dict[str, object]]) -> None:
    print("\n1. Does z > sqrt(edf0/2) predict the sign of the holdout change?\n")
    print(
        f"{'levels':>7} {'edf0':>6} {'effect':>7} {'z':>8} {'thresh':>7} "
        f"{'z/thresh':>9} {'gate':>8} {'holdout':>9}"
    )
    for row in rows:
        gate = "INCLUDE" if row["z"] > row["threshold"] else "exclude"
        flag = "" if row["agrees"] else "   <- disagrees"
        print(
            f"{row['n_levels']:>7} {row['edf0']:>6.0f} {row['effect']:>7.2f} "
            f"{row['z']:>8.2f} {row['threshold']:>7.2f} "
            f"{row['z'] / row['threshold']:>9.2f} {gate:>8} "
            f"{row['delta_pct']:>+8.1f}%{flag}"
        )
    agree = sum(1 for r in rows if r["agrees"])
    print(f"\n  sqrt(edf0/2) agrees with the holdout sign in {agree}/{len(rows)}")
    for fixed in (2.0, 3.0, 5.0):
        hit = sum(1 for r in rows if (r["z"] > fixed) == (r["delta_pct"] < 0.0))
        print(f"    a fixed z > {fixed}: {hit}/{len(rows)}")


def _print_concentration(rows: list[dict[str, object]], n_levels: int) -> None:
    print(
        f"\n2. Concentration at matched z ({n_levels}x{n_levels}). Rows are paired so "
        "spiky and\n   diffuse truths carry the SAME z -- any separation is "
        "information z lacks.\n"
    )
    print(f"{'case':>22} {'z':>8} {'occupied':>9} {'P/(k/3)':>9}")
    for row in rows:
        print(
            f"{row['label']:>22} {row['z']:>8.2f} {row['occupied']:>9.0f} "
            f"{row['concentration']:>9.3f}"
        )
    print("\n  ~1.0 = spread as noise spreads it;  << 1.0 = a few cells carry it")


def _print_sparse_payoff(rows: list[dict[str, object]]) -> None:
    print(
        "\n3. If P says concentrated, does fitting only those cells pay?\n"
        "   Cells ranked on training residuals only.\n"
    )
    print(f"{'truth':>10} {'top-m cells':>12} {'holdout vs mains':>18}")
    for row in rows:
        print(f"{row['kind']:>10} {row['top_m']:>12} {row['delta_pct']:>+17.1f}%")


def _print_shrinkage(rows: list[dict[str, object]], n_levels: int) -> None:
    print(
        f"\n4. Same pair, three model classes ({n_levels}x{n_levels}, 5 cells @ 6.0).\n"
        "   The detection is not in doubt; the question is what to DO with it.\n"
    )
    print(
        f"{'model':>8} {'fit':>8} {'params':>8} {'edf':>9} {'train':>8} {'HOLDOUT':>9} {'vs mains':>10}"
    )
    base = next((r["holdout"] for r in rows if r["model"] == "mains"), None)
    for row in rows:
        delta = "" if base is None else f"{(row['holdout'] / base - 1) * 100:+9.1f}%"
        print(
            f"{row['model']:>8} {row['seconds']:>7.1f}s {row['params']:>8.0f} "
            f"{row['edf']:>9.1f} {row['train']:>8.4f} {row['holdout']:>9.4f} {delta:>10}"
        )
    print(
        "\n  Wall-clock here is indicative only -- it moves several-fold under CPU\n"
        "  contention.  The holdout column does not."
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--n", type=int, default=12_000)
    parser.add_argument("--wide-levels", type=int, default=41)
    return parser


def main() -> None:
    warnings.filterwarnings("ignore")
    args = _build_parser().parse_args()
    started = time.perf_counter()

    _print_gate_ladder(_run_gate_ladder(args.reps, args.n))
    _print_concentration(_run_concentration(args.reps, args.n, args.wide_levels), args.wide_levels)
    _print_sparse_payoff(_run_sparse_payoff(args.reps, args.n, args.wide_levels))
    _print_shrinkage(_run_shrinkage(args.reps, args.n, args.wide_levels), args.wide_levels)
    print(f"\ntotal {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
