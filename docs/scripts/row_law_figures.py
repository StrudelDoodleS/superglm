"""Build the threshold-selection and row-law figures for the distributional how-to.

Uses the public French motor severity book ``freMTPL2sev`` (OpenML 41215, CC0)
joined to ``freMTPL2freq`` (OpenML 41214, CC0) on ``IDpol`` for ``DrivAge``.
Fetch both parquet files first from the repository root::

    uv run python scripts/fetch_fremtpl.py --dest data/
    uv run python docs/scripts/row_law_figures.py

Pass ``--data-dir`` if the parquet files live elsewhere. Figures are written to
``docs/images/row-law/`` and every number quoted in the guide is printed here.
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from superglm import (
    ExperimentalWarning,
    GeneralizedParetoLSS,
    LogNormalLS,
    SplitSplice,
    SuperGLM,
    SuperLSS,
    mean_excess,
    plot_mean_excess,
    plot_threshold_stability,
    s,
    threshold_stability,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data"
IMAGES = ROOT / "docs" / "images" / "row-law"
CHOSEN_THRESHOLD = 20_000.0
STABILITY_THRESHOLDS = (
    2_000.0,
    3_000.0,
    4_000.0,
    5_000.0,
    6_000.0,
    7_500.0,
    10_000.0,
    12_500.0,
    15_000.0,
    17_500.0,
    20_000.0,
    25_000.0,
    30_000.0,
    35_000.0,
    40_000.0,
    50_000.0,
)
ROW_AGE = 45.0


def load_claims(data_dir: Path) -> tuple[pd.DataFrame, int, int]:
    """Load freMTPL2sev inner-joined to freMTPL2freq on IDpol."""
    sev = pd.read_parquet(data_dir / "freMTPL2sev.parquet")
    freq = pd.read_parquet(data_dir / "freMTPL2freq.parquet")
    claims = sev.merge(freq, on="IDpol", how="inner", validate="many_to_one")
    claims["DrivAge"] = claims["DrivAge"].clip(18, 90)
    return claims, len(sev), len(freq)


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA,
        help="directory containing freMTPL2sev.parquet and freMTPL2freq.parquet",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=IMAGES,
        help="directory where PNG figures are written",
    )
    args = parser.parse_args(argv)

    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=ExperimentalWarning)

    claims, n_sev, n_freq = load_claims(args.data_dir)
    y = claims["ClaimAmount"].to_numpy(dtype=np.float64)
    X = claims[["DrivAge"]]

    print("=== Data ===")
    print(f"freMTPL2sev rows: {n_sev:,}")
    print(f"freMTPL2freq policies: {n_freq:,}")
    print(f"joined claims: {len(claims):,}")
    print(
        f"amounts: min={y.min():.2f}, median={np.median(y):.2f}, "
        f"mean={y.mean():.2f}, max={y.max():.2f}"
    )

    # 1. Empirical mean excess plot with chosen threshold marked
    me = mean_excess(y)
    me_fig_payload = me.figure(threshold=CHOSEN_THRESHOLD)
    line = me_fig_payload.line
    assert line is not None
    neg_lower = me.table.loc[me.table["lower"] < 0.0]
    first_neg = neg_lower.iloc[0]
    last_me = me.table.iloc[-1]
    shaded_row = me.table.loc[me.table["threshold"] == me_fig_payload.shaded_from].iloc[0]

    print("\n=== Mean excess ===")
    print(f"candidates: {len(me.table)} (min_above={me.min_above})")
    print(
        f"range: {me.table['threshold'].iloc[0]:.2f} (n_above={int(me.table['n_above'].iloc[0])}) "
        f"to {last_me['threshold']:.2f} (n_above={int(last_me['n_above'])})"
    )
    print(
        f"shaded (< {me_fig_payload.few_claims} claims above): "
        f"from {shaded_row['threshold']:.2f} (n_above={int(shaded_row['n_above'])})"
    )
    print(
        f"lower band < 0: {len(neg_lower)} candidates, first at "
        f"{first_neg['threshold']:.2f} (n_above={int(first_neg['n_above'])}, "
        f"lower={first_neg['lower']:.2f}); last at {last_me['threshold']:.2f} "
        f"(mean_excess={last_me['mean_excess']:.2f}, lower={last_me['lower']:.2f}, "
        f"upper={last_me['upper']:.2f})"
    )
    print(
        f"tail line at {line.threshold:.0f}: n_above={line.n_above}, "
        f"scale={line.scale:.2f}, shape={line.shape:.4f}, "
        f"converged={line.converged}, wall={line.wall}, "
        f"y=({line.y[0]:.2f}, {line.y[1]:.2f})"
    )
    save_figure(plot_mean_excess(me, threshold=CHOSEN_THRESHOLD), args.out_dir / "mean_excess.png")

    # 2. Threshold stability plot (classical: one scale, one shape)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        stability = threshold_stability(None, y, thresholds=STABILITY_THRESHOLDS)
    print("\n=== Threshold stability ===")
    for item in caught:
        print(f"WARNING {item.category.__name__}: {item.message}")
    cols = [
        "threshold",
        "n_above",
        "shape",
        "shape_lower",
        "shape_upper",
        "scale",
        "modified_scale",
        "modified_scale_lower",
        "modified_scale_upper",
        "converged",
        "wall",
    ]
    print(stability.table[cols].to_string(index=False))
    save_figure(plot_threshold_stability(stability), args.out_dir / "threshold_stability.png")

    # 3. SplitSplice at chosen threshold u = 20,000
    body_family = LogNormalLS()
    tail_family = GeneralizedParetoLSS()
    splice = SplitSplice.fit(
        X,
        y,
        threshold=CHOSEN_THRESHOLD,
        body=SuperLSS(
            body_family,
            body_family.mean(s("DrivAge", kind="cr", k=8)),
            body_family.scale(),
        ),
        exceedance=SuperGLM(family="binomial", terms=[s("DrivAge", kind="cr", k=8)]),
        tail=SuperLSS(
            tail_family,
            tail_family.scale(s("DrivAge", kind="cr", k=8)),
            tail_family.shape(),
        ),
        reml=True,
    )

    above = y > CHOSEN_THRESHOLD
    tail_pit = splice.tail.residuals(X.loc[above], y[above] - CHOSEN_THRESHOLD, kind="pit")
    tail_q = splice.tail.residuals(X.loc[above], y[above] - CHOSEN_THRESHOLD, kind="quantile")
    ks = stats.kstest(tail_pit, "uniform")

    print(f"\n=== SplitSplice at u={CHOSEN_THRESHOLD:.0f} ===")
    print(
        f"body: converged={splice.body.result_.converged}, "
        f"certified={splice.body.smoothing_certified_}, "
        f"reason={splice.body.smoothing_convergence_reason_}"
    )
    print(
        f"exceedance: converged={splice.exceedance.result.converged}, "
        f"edf={splice.exceedance.result.effective_df:.2f}"
    )
    print(
        f"tail: converged={splice.tail.result_.converged}, "
        f"certified={splice.tail.smoothing_certified_}, "
        f"reason={splice.tail.smoothing_convergence_reason_}, "
        f"wall={splice.tail.result_.wall}"
    )
    print(
        f"tail diagnostics (n={int(above.sum())}): "
        f"PIT KS={ks.statistic:.4f} (p={ks.pvalue:.4f}), "
        f"qres mean={tail_q.mean():.4f}, sd={tail_q.std(ddof=1):.4f}"
    )

    # 4. One row's fitted density and mean excess (DrivAge = 45)
    row = pd.DataFrame({"DrivAge": [ROW_AGE]})
    grid = np.linspace(100.0, 60_000.0, 300)

    density_curve = splice.distribution_curve(row, kind="density", y=grid)
    piece = np.asarray(density_curve.piece)
    join = int(np.flatnonzero((piece[:-1] == "body") & (piece[1:] == "tail"))[0])
    left_density = float(density_curve.values[join])
    right_density = float(density_curve.values[join + 1])

    me_points = np.array([0.0, CHOSEN_THRESHOLD, 40_000.0, 60_000.0])
    row_me_vals = [float(splice.predict_mean_excess(row, v)[0]) for v in me_points]
    tail_shape = density_curve.parameters["tail_shape"]
    tail_slope = tail_shape / (1.0 - tail_shape)

    print(f"\n=== Fitted row (DrivAge={ROW_AGE:g}) ===")
    for name, value in density_curve.parameters.items():
        print(f"  {name}: {value:.6f}")
    print(f"  mean: {density_curve.mean:.2f}")
    print(f"  pdf at threshold (left body):  {left_density:.6e}")
    print(f"  pdf at threshold (right tail): {right_density:.6e}")
    print(f"  tail/body density ratio at u:  {right_density / left_density:.3f}")
    print(f"  tail mean-excess slope xi/(1-xi): {tail_slope:.4f}")
    for v, val in zip(me_points, row_me_vals, strict=True):
        print(f"  mean_excess(v={v:.0f}): {val:.2f}")

    # Book mean excess under the covariate model vs empirical
    w = np.ones(len(claims), dtype=np.float64)
    print("\n=== Book mean excess (model mixture vs empirical) ===")
    for v in (10_000.0, 20_000.0, 30_000.0, 50_000.0):
        sv = splice.predict_survival(X, v)
        book_me = float(np.sum(w * splice.predict_mean_excess(X, v) * sv) / np.sum(w * sv))
        emp_me = float(np.mean(y[y > v] - v))
        print(f"  v={v:7.0f}: model_book={book_me:10.2f}, empirical={emp_me:10.2f}")

    density_fig = splice.plot_distribution(row, kind="density", y=grid)
    density_fig.axes[0].set_yscale("log")
    save_figure(density_fig, args.out_dir / "row_density.png")

    me_fig = splice.plot_distribution(row, kind="mean_excess", y=grid)
    save_figure(me_fig, args.out_dir / "row_mean_excess.png")

    print("\nWrote:", sorted(p.name for p in args.out_dir.glob("*.png")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
