"""Show what fit_ncv costs against fit_reml on the public French motor book.

One Poisson frequency model of the corrected freMTPL2 book (claims recounted
from the severity file, ``examples/fremtpl2_data.py``), fitted to a fixed 80%
of the policies three ways: ``fit_reml``; ``fit_ncv`` holding out five folds
of policies (``GroupKFold(5)``); and ``fit_ncv`` holding out each policy on
its own.  The figure draws each fit's wall time.

Fetch the book first, from the repository root:
    uv run python scripts/fetch_fremtpl.py --dest data/

One fit per process, so each is timed alone and its peak memory is its own:
    uv run python docs/scripts/ncv_cost_motor.py reml     # or folds, policy
prints one tagged line per number and adds the run to
``data/ncv_examples/cost_motor.json``.  Time each case in alternation, two
rounds, on a quiet machine at default threads; then draw from the stored runs:
    uv run python docs/scripts/ncv_cost_motor.py --figure
Delete the stored file to start the timings again.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import resource
import time
import warnings
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import spanish_books as sb
from sklearn.model_selection import GroupKFold

from superglm import Categorical, Numeric, Spline, SuperGLM

OUT = sb.IMAGES / "cost_motor.png"
STORE = sb.RESULTS / "cost_motor.json"
CASES = {
    "reml": "fit_reml",
    "folds": "fit_ncv, five folds of policies",
    "policy": "fit_ncv, each policy held out",
}
#: Seed of the 80% training split.
SPLIT_SEED = 20261012


def training_rows():
    """The training policies' rating factors, claim frequency, exposure and policy ids."""
    spec = importlib.util.spec_from_file_location(
        "fremtpl2_data", sb.ROOT / "examples" / "fremtpl2_data.py"
    )
    data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data)
    freq = data.load_corrected_freq(sb.DATA)
    freq["Exposure"] = freq["Exposure"].clip(lower=0.01)
    freq["LogDensity"] = np.log1p(freq["Density"].to_numpy(dtype=np.float64))
    perm = np.random.default_rng(SPLIT_SEED).permutation(len(freq))
    is_train = np.zeros(len(freq), dtype=bool)
    is_train[perm[: int(round(0.8 * len(freq)))]] = True
    train = freq[is_train].reset_index(drop=True)
    exposure = train["Exposure"].to_numpy(dtype=np.float64)
    y = train["ClaimNb"].to_numpy(dtype=np.float64) / exposure
    return train, y, exposure, train["IDpol"].to_numpy()


def model() -> SuperGLM:
    return SuperGLM(
        family="poisson",
        features={
            "DrivAge": Spline(n_knots=8),
            "VehAge": Spline(n_knots=12),
            "BonusMalus": Spline(n_knots=12, knot_strategy="quantile_tempered", knot_alpha=0.2),
            "LogDensity": Numeric(),
            "VehBrand": Categorical(),
            "Region": Categorical(),
        },
        weight_semantics="frequency",
        discrete=False,
    )


def run(case: str) -> dict:
    """One complete fit of ``case``: wall time, peak memory and the work it did."""
    X, y, exposure, groups = training_rows()
    fitted = model()
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t0 = time.perf_counter()
        if case == "reml":
            fitted.fit_reml(X, y, sample_weight=exposure)
        elif case == "folds":
            fitted.fit_ncv(X, y, sample_weight=exposure, cv=GroupKFold(5), groups=groups)
        else:
            fitted.fit_ncv(X, y, sample_weight=exposure, groups=groups)
        wall = time.perf_counter() - t0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record = {
        "rows": len(y),
        "groups": int(len(np.unique(groups))) if case == "policy" else 5 if case == "folds" else 0,
        "coefficients": int(np.asarray(fitted.result.beta).size) + 1,  # with the intercept
        "wall_s": wall,
        "peak_rss_mib": peak / 1024,
        "peak_rss_before_fit_mib": before / 1024,
        "edf": float(fitted.result.effective_df),
        "warnings": sorted(f"{type(w.message).__name__}: {w.message}" for w in caught),
    }
    if case == "reml":
        d = fitted.reml_diagnostics()
        record.update(steps=d.get("n_reml_iter"), evaluations=None, converged=d.get("converged"))
    else:
        d = fitted.ncv_diagnostics()
        record.update(
            steps=d["n_iter"], evaluations=d["n_evaluations"], converged=bool(d["converged"])
        )
    return record


def report(case: str, record: dict) -> None:
    for message, count in sorted(Counter(record["warnings"]).items()):
        print(f"WARNING {case} x{count} {message}")
    sb.number(f"cost.{case}.rows", record["rows"])
    sb.number(f"cost.{case}.coefficients", record["coefficients"])
    if record["groups"]:
        sb.number(f"cost.{case}.held_out_groups", record["groups"])
    sb.number(f"cost.{case}.wall_s", f"{record['wall_s']:.2f}")
    sb.number(f"cost.{case}.peak_rss_mib", f"{record['peak_rss_mib']:.0f}")
    sb.number(f"cost.{case}.peak_rss_before_fit_mib", f"{record['peak_rss_before_fit_mib']:.0f}")
    sb.number(f"cost.{case}.edf", f"{record['edf']:.2f}")
    sb.number(f"cost.{case}.steps", record["steps"])
    if record["evaluations"] is not None:
        sb.number(f"cost.{case}.evaluations", record["evaluations"])
    sb.number(f"cost.{case}.converged", record["converged"])


def load_store() -> dict:
    if STORE.exists():
        return json.loads(STORE.read_text(encoding="utf-8"))
    return {case: [] for case in CASES}


def figure() -> None:
    runs = load_store()
    missing = [case for case in CASES if not runs.get(case)]
    if missing:
        print(f"Not drawn: no stored runs for {', '.join(missing)}")
        return
    walls = {case: np.array([r["wall_s"] for r in runs[case]]) for case in CASES}
    means = {case: float(w.mean()) for case, w in walls.items()}
    for case in CASES:
        sb.number(f"cost.{case}.wall_s_runs", " ".join(f"{w:.2f}" for w in walls[case]))
        sb.number(f"cost.{case}.wall_s_mean", f"{means[case]:.2f}")
        sb.number(f"cost.{case}.ratio_to_reml", f"{means[case] / means['reml']:.1f}")

    plt.style.use(str(sb.STYLE))
    fig, ax = plt.subplots(figsize=(8.0, 2.9))
    order = list(CASES)[::-1]
    colours = {"reml": sb.RED, "folds": sb.INK, "policy": sb.INK}
    for i, case in enumerate(order):
        ax.barh(i, means[case], height=0.5, color=colours[case], alpha=0.85, zorder=2)
        ax.plot(
            walls[case],
            np.full(len(walls[case]), i),
            linestyle="none",
            marker="o",
            markersize=8,
            markerfacecolor="white",
            markeredgecolor=sb.MUTED,
            markeredgewidth=1.5,
            zorder=3,
        )
        ratio = "" if case == "reml" else f" ({means[case] / means['reml']:.1f}× fit_reml)"
        ax.text(
            max(means[case], walls[case].max()) + 0.6,
            i,
            f"{means[case]:.1f} s{ratio}",
            va="center",
            fontsize=9,
            color=sb.INK,
        )
    ax.set_yticks(range(len(order)), [CASES[c] for c in order])
    ax.set_xlim(0, 1.35 * max(w.max() for w in walls.values()))
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlabel("Wall time of one complete fit (s); bar = mean, circles = each run")
    ax.set_title("Time to fit the French motor book")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160)
    plt.close(fig)
    print(f"WROTE {OUT.relative_to(sb.ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("case", nargs="?", choices=list(CASES), help="the one fit to time")
    parser.add_argument("--figure", action="store_true", help="draw from the stored runs")
    args = parser.parse_args()
    if args.case is None and not args.figure:
        parser.error("give a case to fit, or --figure")
    if args.case:
        record = run(args.case)
        report(args.case, record)
        runs = load_store()
        runs.setdefault(args.case, []).append(record)
        STORE.parent.mkdir(parents=True, exist_ok=True)
        STORE.write_text(json.dumps(runs, indent=1), encoding="utf-8")
    if args.figure:
        figure()


if __name__ == "__main__":
    main()
