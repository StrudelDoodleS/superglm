"""Score seven ways of choosing the health model's smoothing on policies they never saw.

Five folds of whole policies (``GroupKFold(5, shuffle=True)``, seed in
``spanish_books.SEED``): each method chooses the smoothing on four fifths of
the policies and predicts the fifth.  Every method is scored the same way, by
the held-out Poisson deviance per row, which ``spanish_books.poisson_score``
computes from the fold's predictions; ``cross_validate``'s own ``"deviance"``
would score each fold in its fitted family's deviance, which differs from
method to method.  The figure shows each method's score less Poisson
``fit_reml``'s on the same fold, with each method's mean degrees of freedom.

The methods: ``fit_reml`` under five variance laws (Poisson; Tweedie with
power 1.01, 1.2 and 1.5, each with its scale estimated; negative binomial
with its dispersion estimated), and ``fit_ncv`` holding out single rows or
whole policies.  Six run through ``cross_validate`` on the same folds,
passed as fixed folds so that ``fit_ncv`` holds out single rows unless the
policies are given as ``groups``.  The negative binomial runs fold
by fold, because its dispersion is chosen by ``estimate_theta``, which
``cross_validate`` does not call.

Run from the repository root, after the fetch (about 5 min on one core):
    uv run python scripts/fetch_spanish_books.py --dest data/
    uv run python docs/scripts/ncv_health_seven_methods.py

Each method's fold results are kept under ``data/ncv_examples/``, so a later
run only draws; ``--methods`` fits a subset, ``--refit`` fits again.
"""

from __future__ import annotations

import argparse
import warnings
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import spanish_books as sb
from sklearn.model_selection import GroupKFold

from superglm import SuperGLM, cross_validate, families

OUT = sb.IMAGES / "health_seven_methods.png"

#: Method key -> (label, fit mode, family, whether fit_ncv holds out policies).
METHODS = {
    "reml_poisson": ("fit_reml, Poisson", "fit_reml", "poisson", False),
    "reml_tweedie_1.01": ("fit_reml, Tweedie 1.01", "fit_reml", families.tweedie(p=1.01), False),
    "reml_tweedie_1.2": ("fit_reml, Tweedie 1.2", "fit_reml", families.tweedie(p=1.2), False),
    "reml_tweedie_1.5": ("fit_reml, Tweedie 1.5", "fit_reml", families.tweedie(p=1.5), False),
    "reml_negative_binomial": ("fit_reml, negative binomial", "estimate_theta", None, False),
    "ncv_rows": ("fit_ncv, single rows held out", "fit_ncv", "poisson", False),
    "ncv_policies": ("fit_ncv, policies held out", "fit_ncv", "poisson", True),
}


class FixedFolds:
    """A splitter that yields the same ``(train, test)`` pairs whatever groups it is given."""

    def __init__(self, folds: list):
        self.folds = folds

    def split(self, X, y=None, groups=None):
        yield from self.folds

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self.folds)


def model(family) -> SuperGLM:
    terms = sb.health_terms()
    return SuperGLM(
        family=family,
        features=terms["features"],
        interactions=terms["interactions"],
        discrete=False,
    )


def report_warnings(key: str, caught) -> None:
    for message, count in sorted(
        Counter(f"{type(w.message).__name__}: {w.message}" for w in caught).items()
    ):
        print(f"WARNING {key} x{count} {message}")


def run_cross_validate(key: str, book: sb.Book, folds: list) -> dict:
    _, fit_mode, family, by_policy = METHODS[key]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = cross_validate(
            model(family),
            book.X,
            book.y,
            cv=FixedFolds(folds),
            groups=book.keys["policy"] if by_policy else None,
            offset=np.log(book.exposure),
            fit_mode=fit_mode,
            scoring=sb.poisson_score,
        )
    report_warnings(key, caught)
    scores = result.fold_scores
    return {
        "poisson_deviance": scores["poisson_deviance"].tolist(),
        "edf": scores["effective_df"].tolist(),
        "scale": scores["scale"].tolist(),
        "converged": scores["converged"].astype(bool).tolist(),
    }


def run_negative_binomial(book: sb.Book, folds: list) -> dict:
    out = {"poisson_deviance": [], "edf": [], "theta": [], "converged": []}
    offset = np.log(book.exposure)
    for train, test in folds:
        fold_model = model(families.nb2(theta=1.0))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fold_model.estimate_theta(
                book.X.iloc[train], book.y[train], offset=offset[train], fit_mode="reml"
            )
        report_warnings("reml_negative_binomial", caught)
        mu = fold_model.predict(book.X.iloc[test], offset=offset[test])
        out["poisson_deviance"].append(float(sb.poisson_deviance(book.y[test], mu).mean()))
        out["edf"].append(float(fold_model.result.effective_df))
        out["theta"].append(float(fold_model.theta_))
        out["converged"].append(bool(fold_model.reml_diagnostics()["converged"]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(METHODS))
    parser.add_argument(
        "--refit", action="store_true", help="fit again even when results are stored"
    )
    args = parser.parse_args()

    book = sb.load_health()
    folds = list(
        GroupKFold(5, shuffle=True, random_state=sb.SEED).split(book.X, groups=book.keys["policy"])
    )
    for key in args.methods:
        if key == "reml_negative_binomial":
            sb.stored(
                f"health_methods.{key}", args.refit, lambda: run_negative_binomial(book, folds)
            )
        else:
            sb.stored(
                f"health_methods.{key}",
                args.refit,
                lambda key=key: run_cross_validate(key, book, folds),
            )
    missing = [key for key in METHODS if not (sb.RESULTS / f"health_methods.{key}.json").exists()]
    if missing:
        print(f"Not drawn: no stored results for {', '.join(missing)}")
        return
    results = {key: sb.stored(f"health_methods.{key}", False, None) for key in METHODS}

    base = np.array(results["reml_poisson"]["poisson_deviance"])
    sb.number("health.outer.folds", len(folds))
    sb.number("health.outer.reml_poisson.deviance_per_row", f"{base.mean():.6f}")
    sb.number("health.outer.reml_poisson.per_fold", " ".join(f"{v:.6f}" for v in base))
    for key, record in results.items():
        edf = float(np.mean(record["edf"]))
        sb.number(f"health.outer.{key}.edf", f"{edf:.1f}")
        sb.number(f"health.outer.{key}.edf_per_fold", " ".join(f"{v:.1f}" for v in record["edf"]))
        sb.number(f"health.outer.{key}.converged_folds", sum(record["converged"]))
        if key == "reml_poisson":
            continue
        diff = np.array(record["poisson_deviance"]) - base
        sb.number(
            f"health.outer.{key}.deviance_per_row", f"{np.mean(record['poisson_deviance']):.6f}"
        )
        sb.number(f"health.outer.{key}.minus_reml_poisson", f"{diff.mean():+.6f}")
        sb.number(f"health.outer.{key}.minus_reml_poisson_sd", f"{diff.std(ddof=1):.6f}")
        sb.number(
            f"health.outer.{key}.minus_reml_poisson_pct", f"{100 * diff.mean() / base.mean():+.3f}"
        )
        sb.number(
            f"health.outer.{key}.minus_reml_poisson_per_fold", " ".join(f"{v:+.6f}" for v in diff)
        )
        sb.number(f"health.outer.{key}.folds_better", f"{int((diff < 0).sum())} of {len(diff)}")
    for key in ("reml_tweedie_1.01", "reml_tweedie_1.2", "reml_tweedie_1.5"):
        sb.number(f"health.outer.{key}.scale_mean", f"{np.mean(results[key]['scale']):.1f}")
    sb.number(
        "health.outer.reml_negative_binomial.theta_per_fold",
        " ".join(f"{v:.3f}" for v in results["reml_negative_binomial"]["theta"]),
    )
    gap = np.array(results["ncv_policies"]["poisson_deviance"]) - np.array(
        results["ncv_rows"]["poisson_deviance"]
    )
    sb.number("health.outer.policies_minus_rows", f"{gap.mean():+.6f}")
    sb.number("health.outer.policies_minus_rows_sd", f"{gap.std(ddof=1):.6f}")
    sb.number(
        "health.outer.policies_minus_rows_folds_better", f"{int((gap < 0).sum())} of {len(gap)}"
    )

    plt.style.use(str(sb.STYLE))
    shown = [key for key in METHODS if key != "reml_poisson"]
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    for row, key in enumerate(shown):
        diff = np.array(results[key]["poisson_deviance"]) - base
        colour = sb.INK if key.startswith("ncv") else sb.MUTED
        ax.scatter(diff, np.full(len(diff), row), s=40, color=colour, zorder=3)
        ax.scatter(
            [diff.mean()], [row], s=140, marker="D", facecolor=sb.YELLOW, edgecolor=sb.INK, zorder=4
        )
        ax.annotate(
            f"EDF {np.mean(results[key]['edf']):.0f}",
            xy=(1.0, row),
            xycoords=("axes fraction", "data"),
            xytext=(-4, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            color=sb.MUTED,
        )
    ax.axhline(3.5, color=sb.MUTED, linewidth=0.6, linestyle=":")
    ax.axvline(0.0, color=sb.RED, linewidth=1.4)
    ax.annotate(
        f"fit_reml, Poisson (EDF {np.mean(results['reml_poisson']['edf']):.0f})",
        xy=(0.0, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(4, -12),
        textcoords="offset points",
        color=sb.RED,
    )
    ax.set_yticks(range(len(shown)), [METHODS[key][0] for key in shown])
    ax.set_ylim(len(shown) - 0.4, -1.0)
    ax.set_xlim(-0.046, 0.075)
    ax.set_xlabel("Held-out Poisson deviance per row less Poisson fit_reml's (left is better)")
    ax.set_title("Unseen policies, five folds: each fold and the mean")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160)
    plt.close(fig)
    print(f"WROTE {OUT.relative_to(sb.ROOT)}")


if __name__ == "__main__":
    main()
