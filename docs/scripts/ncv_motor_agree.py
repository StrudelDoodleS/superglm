"""Show the Spanish motor book, where fit_reml and fit_ncv choose almost the same smoothing.

Two fits of the same Poisson model of liability claim frequency to every
fitted row, one by ``fit_reml`` and one by ``fit_ncv(groups=insured)``, which
holds out each insured's years together; the figure draws their driver-age and
vehicle-age relativities.  Then five folds of whole insureds
(``GroupKFold(5, shuffle=True)``, seed in ``spanish_books.SEED``) through
``cross_validate``, each method choosing its smoothing on four fifths and
predicting the fifth, scored by held-out Poisson deviance per row; the script
prints fit_ncv's score against fit_reml's, fold by fold.

Run from the repository root, after the fetch (about 4 min on one core):
    uv run python scripts/fetch_spanish_books.py --dest data/
    uv run python docs/scripts/ncv_motor_agree.py

Each method's results are kept under ``data/ncv_examples/``, so a later run
only draws; ``--methods`` fits one method, ``--refit`` fits again.
"""

from __future__ import annotations

import argparse
import warnings
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spanish_books as sb
from sklearn.model_selection import GroupKFold

from superglm import SuperGLM, cross_validate

OUT = sb.IMAGES / "motor_curves_agree.png"
METHODS = {"reml": "fit_reml", "ncv": "fit_ncv, insureds held out"}
CURVES = {"driver_age": "Driver age", "vehicle_age": "Vehicle age"}


def model() -> SuperGLM:
    return SuperGLM(family="poisson", features=sb.motor_terms()["features"], discrete=False)


def report_warnings(key: str, caught) -> None:
    for message, count in sorted(
        Counter(f"{type(w.message).__name__}: {w.message}" for w in caught).items()
    ):
        print(f"WARNING {key} x{count} {message}")


def weighted_quantiles(values: np.ndarray, weights: np.ndarray, q: tuple) -> np.ndarray:
    """Values below which the given shares of the weight lie."""
    order = np.argsort(values, kind="stable")
    share = np.cumsum(weights[order]) / weights.sum()
    return values[order][np.searchsorted(share, q)]


def run(method: str, book: sb.Book) -> dict:
    """The whole-book fit's curves and degrees of freedom, and the five folds' scores."""
    insured = book.keys["insured"]
    offset = np.log(book.exposure)
    full = model()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if method == "reml":
            full.fit_reml(book.X, book.y, offset=offset)
        else:
            full.fit_ncv(book.X, book.y, offset=offset, groups=insured)
        folds = cross_validate(
            model(),
            book.X,
            book.y,
            cv=GroupKFold(5, shuffle=True, random_state=sb.SEED),
            groups=insured,
            offset=offset,
            fit_mode=f"fit_{method}",
            scoring=sb.poisson_score,
        )
    report_warnings(method, caught)
    edf = full.term_importance(book.X).set_index("term")["edf"]
    relativities = full.relativities()
    curves = {}
    for term in CURVES:
        table = pd.DataFrame(relativities[term])
        curves[term] = {
            "x": table["x"].tolist(),
            "log_relativity": table["log_relativity"].tolist(),
        }
    scores = folds.fold_scores
    return {
        "edf_total": float(full.result.effective_df),
        "edf": {term: float(edf[term]) for term in CURVES},
        "curves": curves,
        "poisson_deviance": scores["poisson_deviance"].tolist(),
        "fold_edf": scores["effective_df"].tolist(),
        "converged": scores["converged"].astype(bool).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(METHODS))
    parser.add_argument(
        "--refit", action="store_true", help="fit again even when results are stored"
    )
    args = parser.parse_args()

    book = sb.load_motor()
    for method in args.methods:
        sb.stored(f"motor_methods.{method}", args.refit, lambda method=method: run(method, book))
    missing = [m for m in METHODS if not (sb.RESULTS / f"motor_methods.{m}.json").exists()]
    if missing:
        print(f"Not drawn: no stored results for {', '.join(missing)}")
        return
    results = {m: sb.stored(f"motor_methods.{m}", False, None) for m in METHODS}

    sb.number("motor.rows", len(book.y))
    sb.number("motor.insureds", len(np.unique(book.keys["insured"])))
    sb.number("motor.claims", int(book.y.sum()))
    sb.number("motor.exposure_years", round(float(book.exposure.sum()), 1))
    for reason, count in book.dropped.items():
        sb.number(f"motor.dropped.{reason}", count)
    for method, record in results.items():
        sb.number(f"motor.full.{method}.edf_total", f"{record['edf_total']:.1f}")
        for term in CURVES:
            sb.number(f"motor.full.{method}.edf.{term}", f"{record['edf'][term]:.1f}")
        sb.number(
            f"motor.outer.{method}.deviance_per_row", f"{np.mean(record['poisson_deviance']):.6f}"
        )
        sb.number(f"motor.outer.{method}.edf", f"{np.mean(record['fold_edf']):.1f}")
        sb.number(
            f"motor.outer.{method}.edf_per_fold", " ".join(f"{v:.1f}" for v in record["fold_edf"])
        )
        sb.number(f"motor.outer.{method}.converged_folds", sum(record["converged"]))
    base = np.array(results["reml"]["poisson_deviance"])
    diff = np.array(results["ncv"]["poisson_deviance"]) - base
    sb.number("motor.outer.ncv_minus_reml", f"{diff.mean():+.6f}")
    sb.number("motor.outer.ncv_minus_reml_sd", f"{diff.std(ddof=1):.6f}")
    sb.number("motor.outer.ncv_minus_reml_pct", f"{100 * diff.mean() / base.mean():+.4f}")
    sb.number(
        "motor.outer.ncv_minus_reml_per_fold_pct", " ".join(f"{100 * v:+.4f}" for v in diff / base)
    )
    sb.number("motor.outer.ncv_folds_better", f"{int((diff < 0).sum())} of {len(diff)}")

    plt.style.use(str(sb.STYLE))
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9), sharey=True)
    for ax, (term, title) in zip(axes, CURVES.items(), strict=True):
        drawn = {}
        for method, colour, width in (("reml", sb.RED, 2.8), ("ncv", sb.INK, 1.3)):
            curve = results[method]["curves"][term]
            x = np.array(curve["x"])
            f = np.array(curve["log_relativity"])
            drawn[method] = np.exp(f - np.interp(np.median(x), x, f))
            label = f"{METHODS[method]} (EDF {results[method]['edf'][term]:.1f})"
            ax.plot(x, drawn[method], color=colour, linewidth=width, label=label)
        gap = 100 * np.abs(drawn["reml"] / drawn["ncv"] - 1)
        sb.number(f"motor.curve.{term}.largest_gap_pct", f"{gap.max():.1f}")
        sb.number(f"motor.curve.{term}.largest_gap_at", f"{x[np.argmax(gap)]:.1f}")
        lo, hi = weighted_quantiles(book.X[term].to_numpy(), book.exposure, (0.05, 0.95))
        inner = (x >= lo) & (x <= hi)
        sb.number(f"motor.curve.{term}.central_90pct_exposure_range", f"{lo:.0f} to {hi:.0f}")
        sb.number(f"motor.curve.{term}.largest_gap_pct_in_that_range", f"{gap[inner].max():.1f}")
        ax.set_title(f"{title}: liability claim frequency")
        ax.set_xlabel(title)
        ax.legend(loc="upper center")
    axes[0].set_ylabel("Relativity (1 at the middle of the range)")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160)
    plt.close(fig)
    print(f"WROTE {OUT.relative_to(sb.ROOT)}")


if __name__ == "__main__":
    main()
