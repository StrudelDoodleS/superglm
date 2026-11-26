"""Draw the health portfolio's age curves under fit_reml and fit_ncv(groups=policy).

Expected medical services per insured-year at each age 0 to 90, women and men,
for a reference member (every factor at its most common level, years with the
insurer at the median), from two fits of the same Poisson model to every
fitted row: one whose smoothing REML chose, one whose smoothing was chosen by
holding out each policy in turn.  Below each panel, fit_reml's curve over
fit_ncv's.  Prints every number the guide quotes from it, one per line.

Run from the repository root, after the fetch (about 40 s on one core):
    uv run python scripts/fetch_spanish_books.py --dest data/
    uv run python docs/scripts/ncv_health_age_curves.py
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

OUT = sb.IMAGES / "health_age_curves.png"
AGES = np.arange(0, 91, dtype=float)


def fit(book: sb.Book, method: str) -> SuperGLM:
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
        if method == "reml":
            model.fit_reml(book.X, book.y, offset=offset)
        else:
            model.fit_ncv(book.X, book.y, offset=offset, groups=book.keys["policy"])
    for w in caught:
        print(f"WARNING {method} {type(w.message).__name__}: {w.message}")
    return model


def reference_curves(model: SuperGLM, X: pd.DataFrame) -> dict[str, np.ndarray]:
    reference = {
        c: X[c].mode().iloc[0] for c in X.columns if c not in ("age", "gender", "seniority")
    }
    reference["seniority"] = float(np.median(X["seniority"]))
    curves = {}
    for gender in ("F", "M"):
        grid = pd.DataFrame(
            {**{c: [v] * len(AGES) for c, v in reference.items()}, "age": AGES, "gender": gender}
        )
        curves[gender] = model.predict(grid[list(X.columns)], offset=np.zeros(len(AGES)))
    return curves


def main() -> None:
    plt.style.use(str(sb.STYLE))
    book = sb.load_health()
    sb.number("health.rows", len(book.y))
    sb.number("health.policies", len(np.unique(book.keys["policy"])))
    sb.number("health.services", int(book.y.sum()))
    sb.number("health.insured_years", round(float(book.exposure.sum()), 1))
    for reason, count in book.dropped.items():
        sb.number(f"health.dropped.{reason}", count)

    models = {method: fit(book, method) for method in ("reml", "ncv")}
    curves = {method: reference_curves(model, book.X) for method, model in models.items()}
    edf = {
        method: model.term_importance(book.X).set_index("term")["edf"]
        for method, model in models.items()
    }
    for method, model in models.items():
        sb.number(f"health.full.{method}.edf_total", round(float(model.result.effective_df), 1))
        for term in ("age", "age:gender[M]", "seniority"):
            sb.number(f"health.full.{method}.edf.{term}", round(float(edf[method][term]), 1))
    search = models["ncv"].ncv_diagnostics()
    sb.number("health.full.ncv.converged", search["converged"])
    sb.number("health.full.ncv.evaluations", search["n_evaluations"])

    fig, axes = plt.subplots(
        2, 2, figsize=(10.4, 6.2), sharex=True, gridspec_kw={"height_ratios": [3, 1.25]}
    )
    panels = (("F", "Women", "age", "EDF"), ("M", "Men", "age:gender[M]", "EDF of the difference"))
    for col, (gender, name, term, edf_label) in enumerate(panels):
        reml, ncv = curves["reml"][gender], curves["ncv"][gender]
        turns_reml, turns_ncv = sb.turning_points(reml), sb.turning_points(ncv)
        ratio = 100.0 * (reml / ncv - 1.0)
        top = axes[0, col]
        top.plot(
            AGES,
            reml,
            color=sb.RED,
            linewidth=1.5,
            label=f"fit_reml\n{turns_reml} turns, {edf_label} {edf['reml'][term]:.1f}",
        )
        top.plot(
            AGES,
            ncv,
            color=sb.INK,
            linewidth=2.2,
            label=f"fit_ncv, policies held out\n{turns_ncv} turns, {edf_label} {edf['ncv'][term]:.1f}",
        )
        top.set_title(f"{name}: medical services per insured-year")
        top.set_ylabel("Expected services per year")
        top.legend(loc="upper left", fontsize=8.5)
        low = axes[1, col]
        low.axhline(0.0, color=sb.MUTED, linewidth=0.8)
        low.plot(AGES, ratio, color=sb.RED, linewidth=1.2)
        low.set_ylim(-15, 15)
        low.set_ylabel("fit_reml over\nfit_ncv, %")
        low.set_xlabel("Age")
        sb.number(f"health.age.{gender}.reml_turning_points", turns_reml)
        sb.number(f"health.age.{gender}.ncv_turning_points", turns_ncv)
        sb.number(f"health.age.{gender}.reml_over_ncv_min_pct", round(float(ratio.min()), 1))
        sb.number(f"health.age.{gender}.reml_over_ncv_max_pct", round(float(ratio.max()), 1))
        sb.number(f"health.age.{gender}.reml_over_ncv_min_at_age", int(AGES[np.argmin(ratio)]))
        sb.number(f"health.age.{gender}.reml_over_ncv_max_at_age", int(AGES[np.argmax(ratio)]))
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160)
    plt.close(fig)
    print(f"WROTE {OUT.relative_to(sb.ROOT)}")


if __name__ == "__main__":
    main()
