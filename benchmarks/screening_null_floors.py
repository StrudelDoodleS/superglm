"""Measure per-kind null z floors for the screening docs.

Run:  uv run python benchmarks/screening_null_floors.py [--seeds 40]

Prints max |z| per screening kind over a battery of null datasets spanning
families (Poisson, Bernoulli-like binomial, gamma, dispersed Gaussian),
correlated parents, exposure spread, and rare-level factors.  The maxima go
into docs/guide/screening.md verbatim; they are documentation of measured
noise floors, not calibrated quantiles.  Nothing here is a p-value: `z`
ranks pairs, and the numbers below say how large a rank score pure noise
produced, so a reader knows what to discount.

Five tables come out, because the headline per-kind maximum on its own
would hide what the docs have to say about it:

  1. per-kind: the floor itself, with the spread and the approx share that
     puts it in context;
  2. by probe df, for the kinds whose df is exact and unpenalized: a 1-df
     probe (`numeric_numeric`, or a `numeric_cat` on a 2-level factor)
     standardizes a chi^2_1, whose right tail is far heavier than a 200-df
     one's, so its floor sits higher at the same sample count;
  3. ordered-categorical vs plain spline margins: an OC margin grids on its
     ~5 mapped level scores rather than a continuous support, and the
     question the docs must answer is whether that narrow grid moves the
     floor (measured: it does not);
  4. by family, so a floor that is really one family's dispersion artifact
     is not quoted as a property of the kind;
  5. the individual rows behind the maxima, because a maximum over a few
     hundred draws IS one draw and should be readable as such.

The battery deliberately carries a second `Numeric` and a 2-level factor so
that `numeric_numeric` and the low-df end of `numeric_cat`/`cat_cat` are
measured rather than assumed.
"""

import argparse
import warnings
from collections import Counter

import numpy as np
import pandas as pd

from superglm import Categorical, SuperGLM
from superglm.features.numeric import Numeric
from superglm.features.ordered_categorical import OrderedCategorical
from superglm.features.spline import Spline

BANDS = ["18-25", "26-35", "36-45", "46-55", "56+"]
FAMILIES = ("poisson", "gamma", "binomial", "gaussian")
# margins whose parent is a spline-mode OrderedCategorical: their pairs carry
# the spline kinds but grid on ~5 mapped scores, so they get their own row.
OC_MARGINS = frozenset({"band"})
# kinds whose block carries no penalty: one rung, and edf0 is the block's
# achieved rank rather than a ladder target, so grouping by it is meaningful.
EXACT_DF_KINDS = ("numeric_numeric", "numeric_cat", "cat_cat")


def _frame(n, rng):
    region_p = np.array([0.55, 0.25, 0.15, 0.05])  # includes a rare level
    age = rng.uniform(18.0, 80.0, n)
    df = pd.DataFrame(
        {
            "age": age,
            # correlated continuous parent
            "power": np.clip(1.5 * age + rng.normal(0.0, 25.0, n), 20.0, 220.0),
            "region": rng.choice(list("ABCD"), n, p=region_p),
            "brand": rng.choice(["B1", "B2", "B3"], n),
            # a 2-level factor puts a 1-df numeric_cat and a 2-df cat_cat in
            # the battery -- the low-df end is the tail worth measuring
            "fuel": rng.choice(["diesel", "petrol"], n),
            "bm": rng.uniform(0.5, 2.0, n),
            # a second Numeric, so numeric_numeric (a 1-df probe) is measured
            "dens": rng.uniform(0.0, 1.0, n),
            "band": rng.choice(BANDS, n),
        }
    )
    exposure = rng.uniform(0.05, 1.0, n)
    return df, exposure


def _features():
    return {
        "age": Spline(kind="ps", n_knots=6),
        "power": Spline(kind="ps", n_knots=6),
        "region": Categorical(),
        "brand": Categorical(),
        "fuel": Categorical(),
        "bm": Numeric(),
        "dens": Numeric(),
        "band": OrderedCategorical(order=BANDS, basis=Spline(kind="ps", n_knots=4)),
    }


def _null_response(df, exposure, family, rng):
    eta = -1.5 + 0.004 * df["age"] + 0.1 * (df["region"] == "B")
    if family == "poisson":
        return rng.poisson(exposure * np.exp(eta)).astype(np.float64), exposure
    if family == "gamma":
        mu = np.exp(eta)
        return rng.gamma(2.0, mu / 2.0), None
    if family == "binomial":
        p = 1.0 / (1.0 + np.exp(-eta))
        return rng.binomial(1, p).astype(np.float64), None
    mu = eta
    return rng.normal(mu, 10.0), None


def _summarize(z):
    """max / max|.| / mean / p90 of a z column, on the finite entries."""
    z = np.asarray(z, dtype=np.float64)
    z = z[np.isfinite(z)]
    if not len(z):
        return None
    return {
        "rows": len(z),
        "max": float(z.max()),
        "absmax": float(np.abs(z).max()),
        "mean": float(z.mean()),
        "p90": float(np.percentile(z, 90.0)),
    }


def _collect(seeds, n):
    """Screen the battery and return (rows frame, diagnostics)."""
    frames = []
    warned = Counter()
    fits, failures = 0, []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        df, exposure = _frame(n, rng)
        for family in FAMILIES:
            y, w = _null_response(df, exposure, family, rng)
            model = SuperGLM(family=family, features=_features())
            fits += 1
            # A degenerate margin makes the statistic 0/sqrt(0) -- a NaN row
            # by design, but numpy says so once per occurrence.  Tally them
            # instead of letting a 160-fit battery spray the console; the
            # count is reported below and a nonzero one is a finding.
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    model.fit_reml(df, y, sample_weight=w)
                    table = model.screen_interactions(df, y, sample_weight=w)
                except Exception as err:  # a failed null fit is data, not a crash
                    failures.append(f"seed={seed} family={family}: {type(err).__name__}: {err}")
                    table = None
            for item in caught:
                warned[(item.category.__name__, str(item.message).split("\n")[0][:90])] += 1
            if table is None or not len(table):
                continue
            frames.append(table.assign(seed=seed, family=family))
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(rows):
        rows["oc"] = [
            a in OC_MARGINS or b in OC_MARGINS for a, b in zip(rows["feature_a"], rows["feature_b"])
        ]
    return rows, {"fits": fits, "failures": failures, "warned": warned}


def _print_per_kind(rows):
    print("\nmax null z per kind over the battery:")
    print(
        f"  {'kind':16s} {'rows':>5s} {'max z':>7s} {'max|z|':>7s} "
        f"{'mean z':>7s} {'p90 z':>7s} {'probe df':>12s} {'approx':>7s}"
    )
    for kind, group in rows.groupby("kind"):
        stats = _summarize(group["z"])
        if stats is None:
            continue
        finite = group[np.isfinite(group["z"])]
        lo, hi = finite["edf0"].min(), finite["edf0"].max()
        approx = 100.0 * float(finite["approx"].mean())
        print(
            f"  {kind:16s} {stats['rows']:5d} {stats['max']:7.2f} {stats['absmax']:7.2f} "
            f"{stats['mean']:7.2f} {stats['p90']:7.2f} {lo:5.1f}-{hi:5.1f} {approx:6.0f}%"
        )


def _print_by_df(rows):
    print("\nnull tail by exact probe df (unpenalized kinds; low df = heavier tail):")
    print(f"  {'kind':16s} {'df':>5s} {'rows':>5s} {'max z':>7s} {'max|z|':>7s} {'p90 z':>7s}")
    sub = rows[rows["kind"].isin(EXACT_DF_KINDS) & np.isfinite(rows["z"])]
    for kind in EXACT_DF_KINDS:
        part = sub[sub["kind"] == kind]
        for df_value, group in part.groupby(part["edf0"].round(1)):
            stats = _summarize(group["z"])
            print(
                f"  {kind:16s} {df_value:5.1f} {stats['rows']:5d} {stats['max']:7.2f} "
                f"{stats['absmax']:7.2f} {stats['p90']:7.2f}"
            )


def _print_oc_split(rows):
    print("\nordered-categorical margins vs plain spline margins (spline kinds only):")
    print(
        f"  {'kind':12s} {'margins':10s} {'rows':>5s} {'max z':>7s} {'max|z|':>7s} "
        f"{'mean z':>7s} {'p90 z':>7s} {'cells':>9s}"
    )
    sub = rows[rows["kind"].isin(("ti", "spline_cat"))]
    for kind in ("ti", "spline_cat"):
        for oc, label in ((False, "plain"), (True, "oc")):
            group = sub[(sub["kind"] == kind) & (sub["oc"] == oc)]
            stats = _summarize(group["z"])
            if stats is None:
                continue
            cells = int(group[np.isfinite(group["z"])]["n_cells"].median())
            print(
                f"  {kind:12s} {label:10s} {stats['rows']:5d} {stats['max']:7.2f} "
                f"{stats['absmax']:7.2f} {stats['mean']:7.2f} {stats['p90']:7.2f} {cells:9d}"
            )


def _print_by_family(rows):
    """Whether one family drives the floor, or all four agree on it.

    A floor that is really one family's artifact would be a finding about
    that family's dispersion scaling rather than about the kind, so the
    docs table needs this to be readable before it quotes a maximum.
    """
    print("\nmax null z by family (is the floor one family's artifact?):")
    header = "".join(f"{fam:>9s}" for fam in FAMILIES)
    print(f"  {'kind':16s}{header}")
    for kind, group in rows.groupby("kind"):
        cells = []
        for family in FAMILIES:
            stats = _summarize(group[group["family"] == family]["z"])
            cells.append(f"{stats['max']:9.2f}" if stats else f"{'-':>9s}")
        print(f"  {kind:16s}{''.join(cells)}")


def _print_top_rows(rows, count=6):
    """The individual rows behind the maxima.

    A per-kind maximum over a few hundred draws is one draw; printing the
    draws keeps a reader from reading a lone outlier as a stable floor.
    """
    print(f"\nthe {count} largest single rows in the battery:")
    top = rows[np.isfinite(rows["z"])].nlargest(count, "z")
    for row in top.itertuples():
        pair = f"{row.feature_a} x {row.feature_b}"
        print(
            f"  z={row.z:5.2f}  {row.kind:16s} {pair:22s} "
            f"seed={row.seed:<3d} family={row.family:9s} edf0={row.edf0:.1f}"
        )


def _print_diagnostics(rows, diag, n, max_cells=5_000_000):
    print("\ndiagnostics:")
    print(f"  fits attempted            {diag['fits']}")
    print(f"  fits that failed          {len(diag['failures'])}")
    for line in diag["failures"][:10]:
        print(f"    {line}")
    print(f"  rows screened             {len(rows)}")
    nonfinite = int((~np.isfinite(rows["z"])).sum()) if len(rows) else 0
    print(f"  non-finite z rows         {nonfinite}   (refusals or degenerate margins)")
    if nonfinite:
        bad = rows[~np.isfinite(rows["z"])]
        for (kind, a, b), count in Counter(
            zip(bad["kind"], bad["feature_a"], bad["feature_b"])
        ).most_common(10):
            print(f"    {kind} {a} x {b}: {count}")
    # The numeric_cat budget gate is (L + 2)^2 <= max_cells on the FACTOR's
    # level count alone.  Eyeballing it against the battery is the point:
    # the widest factor here is 4 levels, so no pair is anywhere near it.
    factors = rows[rows["kind"] == "numeric_cat"]["n_cells"] if len(rows) else []
    widest = int(max(factors)) if len(factors) else 0
    admits = int(np.sqrt(max_cells)) - 2
    print(
        f"  widest numeric_cat factor {widest} levels; the gate (L+2)^2 <= "
        f"max_cells={max_cells} admits L <= {admits}"
    )
    warned = diag["warned"]
    print(f"  warnings raised           {sum(warned.values())} in {len(warned)} distinct forms")
    for (category, message), count in warned.most_common(8):
        print(f"    {count:5d}x {category}: {message}")
    seeds = diag["fits"] // len(FAMILIES)
    print(f"\n  (n={n} rows per dataset, {len(FAMILIES)} families x {seeds} seeds)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=40)
    parser.add_argument("--n", type=int, default=8000)
    args = parser.parse_args()

    rows, diag = _collect(args.seeds, args.n)
    if not len(rows):
        print("no rows screened")
        return
    _print_per_kind(rows)
    _print_by_df(rows)
    _print_oc_split(rows)
    _print_by_family(rows)
    _print_top_rows(rows)
    _print_diagnostics(rows, diag, args.n)


if __name__ == "__main__":
    main()
