"""Draw actual against held-out expected services for the health portfolio's larger policies.

One ``fit_ncv(groups=policy)`` of the Poisson health model to every fitted
row; ``ncv_heldout_groups()`` then gives, for each policy, the services the
fit would have expected with that policy left out, at no extra fit.  The
figure shows the policies with 100 or more insured-year rows.  Also printed:
every one of those policies' rows of the table, and the table by year and by
product with the ordinary fitted actual over expected beside it.

Run from the repository root, after the fetch (about 30 s on one core):
    uv run python scripts/fetch_spanish_books.py --dest data/
    uv run python docs/scripts/ncv_health_heldout_policies.py
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spanish_books as sb

from superglm import SuperGLM

OUT = sb.IMAGES / "health_heldout_policies.png"
#: A policy with at least this many insured-year rows is drawn.
MIN_ROWS = 100
#: Actual over held-out expected further than this from 1 is marked.
FAR = 0.25
#: Policies named on the figure, with the label's offset in points and alignment.
LABELLED = {4728: (8, 0, "left"), 28288: (8, 0, "left"), 32614: (8, -4, "left"), 7303: (-8, 0, "right")}


def main() -> None:
    book = sb.load_health()
    terms = sb.health_terms()
    model = SuperGLM(
        family="poisson",
        features=terms["features"],
        interactions=terms["interactions"],
        discrete=False,
    )
    offset = np.log(book.exposure)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit_ncv(book.X, book.y, offset=offset, groups=book.keys["policy"])
    for w in caught:
        print(f"WARNING {type(w.message).__name__}: {w.message}")

    table = model.ncv_heldout_groups()
    big = table[table["rows"] >= MIN_ROWS].sort_values("rows", ascending=False)
    ratio = big["actual_over_expected"].to_numpy()
    far = np.abs(ratio - 1.0) > FAR
    sb.number("health.policies.total", len(table))
    sb.number("health.policies.with_100_rows", len(big))
    sb.number("health.policies.with_100_rows.rows", int(big["rows"].sum()))
    sb.number("health.policies.with_100_rows.more_than_25pct_from_expected", int(far.sum()))
    for policy, row in big.iterrows():
        sb.number(
            f"health.policies.row.{policy}",
            f"rows={int(row['rows'])} actual={row['actual']:.0f} "
            f"heldout_expected={row['heldout_expected']:.1f} actual_over_expected={row['actual_over_expected']:.2f}",
        )

    fitted = model.predict(book.X, offset=offset)
    for name, labels in (("year", book.keys["year"]), ("product", book.X["product"].to_numpy())):
        held = model.ncv_heldout_groups(by=labels)
        in_sample = (
            pd.DataFrame({"g": labels, "actual": book.y, "fitted": fitted}).groupby("g").sum()
        )
        for level, row in held.iterrows():
            sb.number(
                f"health.by_{name}.{level}",
                f"rows={int(row['rows'])} heldout_actual_over_expected={row['actual_over_expected']:.3f} "
                f"fitted_actual_over_expected={in_sample.loc[level, 'actual'] / in_sample.loc[level, 'fitted']:.3f}",
            )

    plt.style.use(str(sb.STYLE))
    expected = big["heldout_expected"].to_numpy()
    actual = big["actual"].to_numpy()
    shown = np.maximum(actual, 200.0)
    lim = np.array([150.0, 60_000.0])
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    ax.fill_between(
        lim,
        lim * (1 - FAR),
        lim * (1 + FAR),
        color=sb.BAND,
        zorder=0,
        label="within 25% of held-out expected",
    )
    ax.plot(lim, lim, color=sb.MUTED, linewidth=1.0, linestyle="--", zorder=1)
    ax.scatter(
        expected[~far],
        shown[~far],
        s=18 + big["rows"].to_numpy()[~far] / 12,
        facecolor="white",
        edgecolor=sb.INK,
        linewidth=1.2,
        zorder=3,
    )
    ax.scatter(
        expected[far],
        shown[far],
        s=18 + big["rows"].to_numpy()[far] / 12,
        facecolor=sb.RED,
        edgecolor=sb.INK,
        linewidth=1.0,
        zorder=4,
        label=f"more than {FAR:.0%} from held-out expected",
    )
    for policy, (dx, dy, ha) in LABELLED.items():
        r = big.loc[policy, "actual_over_expected"]
        ax.annotate(
            f"policy {policy}: " + ("no services" if r == 0 else f"A/E {r:.2f}"),
            (big.loc[policy, "heldout_expected"], max(big.loc[policy, "actual"], 200.0)),
            textcoords="offset points",
            xytext=(dx, dy),
            ha=ha,
            va="center",
            fontsize=8.6,
            color=sb.RED,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("Held-out expected services (the fit without that policy)")
    ax.set_ylabel("Actual services, 2017 to 2019")
    ax.set_title(f"The {len(big)} policies with {MIN_ROWS} or more insured-year rows")
    ax.legend(loc="upper left")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160)
    plt.close(fig)
    print(f"WROTE {OUT.relative_to(sb.ROOT)}")


if __name__ == "__main__":
    main()
