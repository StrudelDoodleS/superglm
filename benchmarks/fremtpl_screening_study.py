"""freMTPL2 study behind docs/explanation/what-screening-does.md.

PSST (``screen_interactions``) against InterpretML's FAST
(``measure_interactions``) on one fitted Poisson mains model, with every
candidate pair then refit for its in- and out-of-sample deviance gain; and the
mains model against an EBM mains model, with the Bonus-Malus shape each fits.

``interpret-core`` is not a dependency of superglm.  Install it into a
separate environment beside superglm, for example::

    uv venv study-venv
    VIRTUAL_ENV=study-venv uv pip install interpret-core==0.7.8 -e .

Each run writes one JSON file into ``--out``; ``report`` reads them back and
prints the page's tables::

    python benchmarks/fremtpl_screening_study.py pairs --book corrected --spec A --out OUT
    python benchmarks/fremtpl_screening_study.py pairs --book corrected --spec B --out OUT
    python benchmarks/fremtpl_screening_study.py baseline --book corrected --out OUT
    python benchmarks/fremtpl_screening_study.py report --out OUT

``--book raw`` reads ``freMTPL2freq`` as published; ``--book corrected``
recounts ``ClaimNb`` from ``freMTPL2sev`` (Wüthrich & Merz 2023, Section 13.1;
``examples/fremtpl2_data.py``).  ``--scale full`` uses the whole book instead
of the 200,000-row sample, and ``--split secondary`` a second, independent
70/30 split.  Pin every thread pool to 1 for comparable runs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import time
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from superglm import Categorical, Numeric, RandomEffect, Spline, SuperGLM

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["DrivAge", "VehAge", "BonusMalus", "VehBrand", "Region"]
NUMERIC = ["DrivAge", "VehAge", "BonusMalus"]
FACTORS = ["VehBrand", "Region"]
SAMPLE_ROWS = 200_000
MAX_REML_ITER = 100
BM_GRID = np.arange(50.0, 231.0)


# ── data ────────────────────────────────────────────────────────────────


def _fremtpl2_data():
    path = ROOT / "examples" / "fremtpl2_data.py"
    spec = importlib.util.spec_from_file_location("_fremtpl2_data", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_book(book: str, data_dir: str | None) -> pd.DataFrame:
    data = _fremtpl2_data()
    if book == "corrected":
        return data.load_corrected_freq(data_dir)
    return pd.read_parquet(data.find_data_dir(data_dir) / data.FREQ_FILE)


def in_train(idpol: pd.Series, split: str) -> np.ndarray:
    """70/30 on a hash of ``IDpol``: the first 8 hex digits, mod 100, below 70."""
    digest = hashlib.md5 if split == "primary" else hashlib.sha1
    ids = idpol.astype(np.int64).to_numpy()
    return np.array([int(digest(str(v).encode()).hexdigest()[:8], 16) % 100 < 70 for v in ids])


def prepare(book: str, scale: str, split: str, data_dir: str | None) -> dict:
    df = load_book(book, data_dir)
    full_rows, full_claims = len(df), int(df["ClaimNb"].sum())
    if scale == "sample":
        df = df.sample(SAMPLE_ROWS, random_state=0)
    df = df.reset_index(drop=True)
    df["Exposure"] = df["Exposure"].clip(lower=0.01)
    for col in NUMERIC:
        df[col] = df[col].astype(np.float64)
    for col in FACTORS:
        df[col] = df[col].astype(str)
    mask = in_train(df["IDpol"], split)
    out = {
        "counts": {
            "book": book,
            "full_rows": full_rows,
            "full_claims": full_claims,
            "rows": len(df),
            "claims": int(df["ClaimNb"].sum()),
            "train_rows": int(mask.sum()),
            "holdout_rows": int((~mask).sum()),
            "train_claims": int(df.loc[mask, "ClaimNb"].sum()),
            "holdout_claims": int(df.loc[~mask, "ClaimNb"].sum()),
            "bm_share_at_50": float(np.mean(df["BonusMalus"] == 50.0)),
        }
    }
    for name, m in (("train", mask), ("hold", ~mask)):
        part = df.loc[m].reset_index(drop=True)
        w = part["Exposure"].to_numpy(np.float64)
        out[name] = (part[FEATURES], part["ClaimNb"].to_numpy(np.float64) / w, w)
    return out


def poisson_deviance(y, mu, w) -> float:
    safe = np.where(y > 0, y, 1.0)
    ylogy = np.where(y > 0, y * np.log(safe / mu), 0.0)
    return float(2.0 * np.sum(w * (ylogy - (y - mu))))


# ── superglm ────────────────────────────────────────────────────────────


def features(spec: str, **override) -> dict:
    feats = {
        "DrivAge": Spline(kind="ps", n_knots=8),
        "VehAge": Spline(kind="ps", n_knots=6),
        "BonusMalus": Numeric() if spec == "A" else Spline(kind="ps", n_knots=8),
        "VehBrand": Categorical(),
        "Region": Categorical(),
    }
    feats.update(override)
    return feats


def fit(feats, train, interactions=None) -> tuple[SuperGLM, dict]:
    X, y, w = train
    model = SuperGLM(
        family="poisson",
        features=feats,
        interactions=interactions,
        weight_semantics="frequency",
    )
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit_reml(X, y, sample_weight=w, max_reml_iter=MAX_REML_ITER)
    seconds = time.perf_counter() - t0
    diag = model.reml_diagnostics()
    info = {
        "seconds": seconds,
        "converged": bool(diag.get("converged", False)),
        "reml_iter": diag.get("n_reml_iter"),
        "warnings": sorted({type(w.message).__name__ for w in caught}),
        "edf": float(model._result.effective_df),
        "fit_deviance": float(model._result.deviance),
    }
    return model, info


def deviances(model, train, hold) -> dict:
    return {
        "train": poisson_deviance(train[1], model.predict(train[0]), train[2]),
        "hold": poisson_deviance(hold[1], model.predict(hold[0]), hold[2]),
    }


def term_edf(model, a, b) -> float:
    """The interaction term's edf: the sum over its groups (one per level for a factor)."""
    edf = model._group_edf
    return float(sum(v for k, v in edf.items() if k == f"{a}:{b}" or k.startswith(f"{a}:{b}[")))


def spline_line(model, name) -> dict:
    """The summary's ``rank``, ``edf`` and Wald ``chi2(ref df)`` for one term."""
    text = str(model.summary())
    head = re.search(rf"{name}\s+\[spline, (\d+) params, chi2\(([\d.]+)\)=([\d.]+)", text)
    tail = re.search(r"rank=(\d+), edf=([\d.]+)", text[head.end() :]) if head else None
    if not head or not tail:
        return {}
    return {
        "params": int(head.group(1)),
        "ref_df": float(head.group(2)),
        "chi2": float(head.group(3)),
        "rank": int(tail.group(1)),
        "edf": float(tail.group(2)),
    }


# ── PSST ────────────────────────────────────────────────────────────────


def psst(model, train) -> dict:
    X, y, w = train
    t0 = time.perf_counter()
    table = model.screen_interactions(X, y, sample_weight=w)
    seconds = time.perf_counter() - t0
    rows = {
        f"{r.feature_a}:{r.feature_b}": {
            "kind": r.kind,
            "z": float(r.z),
            "statistic": float(r.statistic),
            "edf0": float(r.edf0),
        }
        for r in table.itertuples()
    }
    # The earlier denominator, z = (T/phi - edf0) / sqrt(2 edf0), maximised
    # over the same four-rung ladder one rung at a time.
    for rung in (2.0, 4.0, 8.0, 16.0):
        one = model.screen_interactions(X, y, sample_weight=w, edf0=rung)
        for r in one.itertuples():
            row = rows[f"{r.feature_a}:{r.feature_b}"]
            old = (float(r.statistic) - float(r.edf0)) / np.sqrt(2.0 * float(r.edf0))
            row["z_old"] = max(row.get("z_old", -np.inf), old)
    return {
        "rows": rows,
        "phi": float(table.attrs.get("phi", np.nan)),
        "deferred": {str(k): str(v) for k, v in table.attrs.get("deferred_features", {}).items()},
        "seconds": seconds,
    }


# ── FAST ────────────────────────────────────────────────────────────────


@contextmanager
def fast_flags(extra: int):
    """OR a native flag into every FAST call (not reachable through the public API)."""
    from interpret.utils._native import InteractionDetector

    original = InteractionDetector.calc_interaction_strength

    def patched(self, feature_idxs, flags, *args, **kwargs):
        return original(self, feature_idxs, flags | extra, *args, **kwargs)

    InteractionDetector.calc_interaction_strength = patched
    try:
        yield
    finally:
        InteractionDetector.calc_interaction_strength = original


def fast(X, y, w, pairs, init_score, flags=0, **kwargs) -> tuple[dict, float]:
    from interpret.utils import measure_interactions

    idx = {f: i for i, f in enumerate(X.columns)}
    types = ["continuous" if c in NUMERIC else "nominal" for c in X.columns]
    t0 = time.perf_counter()
    with fast_flags(flags):
        ranked = measure_interactions(
            X,
            y,
            interactions=[(idx[a], idx[b]) for a, b in pairs],
            init_score=init_score,
            sample_weight=w,
            feature_types=types,
            objective="poisson_deviance",
            **kwargs,
        )
    seconds = time.perf_counter() - t0
    names = list(X.columns)
    keys = {frozenset(p): f"{p[0]}:{p[1]}" for p in pairs}
    out = {keys[frozenset((names[i], names[j]))]: float(s) for (i, j), s in ranked}
    return out, seconds


def ebm_mains(train, seed=42):
    from interpret.glassbox import ExplainableBoostingRegressor

    X, y, w = train
    ebm = ExplainableBoostingRegressor(
        feature_types=["continuous" if c in NUMERIC else "nominal" for c in X.columns],
        interactions=0,
        outer_bags=8,
        objective="poisson_deviance",
        n_jobs=1,
        random_state=seed,
    )
    t0 = time.perf_counter()
    ebm.fit(X, y, sample_weight=w)
    return ebm, time.perf_counter() - t0


def fast_suite(model, train, pairs, ebm) -> dict:
    from interpret.utils._native import Native

    X, y, w = train
    eta = np.log(model.predict(X))
    out = {
        "deviance_at_eta": poisson_deviance(y, np.exp(eta), w),
        "mean_exposure": float(np.mean(w)),
    }
    runs = {
        "default": dict(init_score=eta),
        "purify": dict(init_score=eta, flags=Native.CalcInteractionFlags_Purify),
        "ebm": dict(init_score=np.log(ebm.predict(X))),
        "none": dict(init_score=None),
        "full": dict(init_score=eta, flags=Native.CalcInteractionFlags_Full),
        "bins256": dict(init_score=eta, max_interaction_bins=256),
        "bins1024": dict(init_score=eta, max_interaction_bins=1024),
    }
    for name, kw in runs.items():
        out[name], out[f"{name}_seconds"] = fast(X, y, w, pairs, **kw)
    uniform = X.copy()
    for c in NUMERIC:
        uniform[c] = stats.rankdata(X[c]) / len(X)
    out["uniform"], _ = fast(uniform, y, w, pairs, init_score=eta)
    # The same model as counts with a log(exposure) offset and unit weights.
    out["counts_offset"], _ = fast(X, y * w, None, pairs, init_score=eta + np.log(w))
    return out


# ── mechanism ───────────────────────────────────────────────────────────


def _codes(x: pd.Series, order_by=None, weight=None):
    if x.name in FACTORS:
        levels = pd.Index(sorted(x.unique()))
        code = levels.get_indexer(x)
        if order_by is not None:
            means = np.bincount(code, weight * order_by) / np.bincount(code, weight)
            rank = np.empty(len(levels), dtype=np.int64)
            rank[np.argsort(means, kind="stable")] = np.arange(len(levels))
            code = rank[code]
        return code, len(levels)
    values = np.unique(x.to_numpy())
    return np.searchsorted(values, x.to_numpy()), len(values)


def quadrant_r2(shift, w, xa, xb) -> float:
    """Largest weighted R^2 of one 2x2 step, one cut per axis, on ``shift``.

    Factor levels are first ordered by their weighted mean shift, so a cut
    on a factor is its best binary split of levels.
    """
    ca, na = _codes(xa, shift, w)
    cb, nb = _codes(xb, shift, w)
    W = np.zeros((na, nb))
    S = np.zeros((na, nb))
    np.add.at(W, (ca, cb), w)
    np.add.at(S, (ca, cb), w * shift)
    tw, ts = W.sum(), S.sum()
    sst = float(np.sum(w * shift**2) - ts**2 / tw)
    cw, cs = W.cumsum(0).cumsum(1)[:-1, :-1], S.cumsum(0).cumsum(1)[:-1, :-1]
    rw, rs = W.cumsum(0).sum(1)[:-1, None], S.cumsum(0).sum(1)[:-1, None]
    kw, ks = W.cumsum(1).sum(0)[None, :-1], S.cumsum(1).sum(0)[None, :-1]
    quads = [
        (cw, cs),
        (rw - cw, rs - cs),
        (kw - cw, ks - cs),
        (tw - rw - kw + cw, ts - rs - ks + cs),
    ]
    between = -(ts**2) / tw
    for qw, qs in quads:
        between = between + np.where(qw > 0, qs**2 / np.where(qw > 0, qw, 1.0), 0.0)
    return float(np.max(between)) / sst


def slope_r2(shift, w, num: pd.Series, fac: pd.Series) -> float:
    """Weighted R^2 of ``shift`` on a per-level intercept and a per-level slope."""
    code, n = _codes(fac)
    x = num.to_numpy(np.float64)
    x = x - np.average(x, weights=w)
    rows = np.arange(len(x))
    D = np.zeros((len(x), 2 * n))
    D[rows, code] = 1.0
    D[rows, n + code] = x
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(D * sw[:, None], shift * sw, rcond=None)
    resid = shift - D @ beta
    centred = shift - np.average(shift, weights=w)
    return 1.0 - float(np.sum(w * resid**2) / np.sum(w * centred**2))


def mechanism(mains, refit, train, a, b) -> dict:
    X, y, w = train
    shift = np.log(refit.predict(X)) - np.log(mains.predict(X))
    out = {}
    for name, wt in (("exposure", w), ("mu", refit.predict(X) * w)):
        out[f"shift_ss_{name}"] = float(np.sum(wt * (shift - np.average(shift, weights=wt)) ** 2))
    out["quadrant_r2"] = quadrant_r2(shift, w, X[a], X[b])
    if (a in FACTORS) != (b in FACTORS):
        num, fac = (a, b) if b in FACTORS else (b, a)
        out["slope_r2"] = slope_r2(shift, w, X[num], X[fac])
    return out


# ── runs ────────────────────────────────────────────────────────────────


def run_pairs(args) -> dict:
    data = prepare(args.book, args.scale, args.split, args.data_dir)
    train, hold = data["train"], data["hold"]
    mains, info = fit(features(args.spec), train)
    info.update(deviances(mains, train, hold))
    if args.spec == "B":
        info["bonusmalus"] = spline_line(mains, "BonusMalus")
    out = {"args": vars(args), "counts": data["counts"], "mains": info}
    all_pairs = [(a, b) for i, a in enumerate(FEATURES) for b in FEATURES[i + 1 :]]
    if args.pair:
        # One confirmatory refit alone, for the full book; report merges it.
        out["refits"] = {}
        a, b = args.pair.split(":")
        refit_one(out, features(args.spec), train, hold, mains, a, b)
        return out
    out["psst"] = psst(mains, train)
    if not args.no_fast:
        ebm, out["ebm_seconds"] = ebm_mains(train)
        out["fast"] = fast_suite(mains, train, all_pairs, ebm)
    out["refits"] = {}
    screened = {frozenset(k.split(":")) for k in out["psst"]["rows"]}
    for a, b in [] if args.no_refit else all_pairs:
        if frozenset((a, b)) not in screened:
            continue
        refit_one(out, features(args.spec), train, hold, mains, a, b)
    return out


def refit_one(out, feats, train, hold, mains, a, b) -> None:
    info = out["mains"]
    refit, row = fit(feats, train, interactions=[(a, b)])
    row.update(deviances(refit, train, hold))
    row["gain_in"] = info["train"] - row["train"]
    row["gain_out"] = info["hold"] - row["hold"]
    row["pair_edf"] = term_edf(refit, a, b)
    row["delta_edf"] = row["edf"] - info["edf"]
    row.update(mechanism(mains, refit, train, a, b))
    out["refits"][f"{a}:{b}"] = row
    print(
        f"{a}:{b} in {row['gain_in']:.1f} out {row['gain_out']:.1f} "
        f"edf {row['pair_edf']} {row['seconds']:.1f}s conv {row['converged']}",
        flush=True,
    )


def bm_curve(predict, probe) -> list[float]:
    X = probe.loc[probe.index.repeat(len(BM_GRID))].reset_index(drop=True)
    X["BonusMalus"] = BM_GRID
    return [float(v) for v in predict(X)]


def run_baseline(args) -> dict:
    data = prepare(args.book, args.scale, args.split, args.data_dir)
    train, hold = data["train"], data["hold"]
    X = train[0]
    probe = X.iloc[[0]].copy()
    out = {
        "args": vars(args),
        "counts": data["counts"],
        "bm_grid": [float(v) for v in BM_GRID],
        "bm_train_exposure": [float(train[2][X["BonusMalus"] == v].sum()) for v in BM_GRID],
        "fits": {},
    }
    variants = {
        "specB": features("B"),
        "specA": features("A"),
        "k20": features(
            "B",
            DrivAge=Spline(kind="ps", n_knots=20),
            VehAge=Spline(kind="ps", n_knots=20),
            BonusMalus=Spline(kind="ps", n_knots=20),
        ),
        "credibility": features("B", VehBrand=RandomEffect(), Region=RandomEffect()),
        "bm_quantile_rows": features(
            "B", BonusMalus=Spline(kind="ps", n_knots=8, knot_strategy="quantile_rows")
        ),
        "bm_k12_tempered": features(
            "B", BonusMalus=Spline(kind="ps", n_knots=12, knot_strategy="quantile_tempered")
        ),
    }
    preds = {}
    for name, feats in variants.items():
        model, row = fit(feats, train)
        row.update(deviances(model, train, hold))
        if name == "specB":
            row["bonusmalus"] = spline_line(model, "BonusMalus")
        if name != "specA":
            row["bm_mu"] = bm_curve(model.predict, probe)
        preds[name] = model.predict(X)
        out["fits"][name] = row
        print(name, f"{row['train']:.1f} {row['hold']:.1f} {row['seconds']:.1f}s", flush=True)
    out["quantile_rows_bit_identical"] = bool(
        np.array_equal(preds["specB"], preds["bm_quantile_rows"])
    )
    ebm, seconds = ebm_mains(train)
    out["fits"]["ebm"] = {
        "train": poisson_deviance(train[1], ebm.predict(X), train[2]),
        "hold": poisson_deviance(hold[1], ebm.predict(hold[0]), hold[2]),
        "seconds": seconds,
        "bm_mu": bm_curve(ebm.predict, probe),
    }
    print("ebm", out["fits"]["ebm"]["train"], out["fits"]["ebm"]["hold"], flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("task", choices=["pairs", "baseline", "report"])
    parser.add_argument("--book", choices=["raw", "corrected"], default="corrected")
    parser.add_argument("--scale", choices=["sample", "full"], default="sample")
    parser.add_argument("--split", choices=["primary", "secondary"], default="primary")
    parser.add_argument("--spec", choices=["A", "B"], default="B")
    parser.add_argument("--no-fast", action="store_true")
    parser.add_argument("--no-refit", action="store_true")
    parser.add_argument("--pair", default=None, help="refit only this pair, as A:B")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.task == "report":
        report(args.out)
        return
    result = (run_pairs if args.task == "pairs" else run_baseline)(args)
    result["args"] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    tag = f"{args.task}-{args.book}-{args.scale}-{args.split}"
    if args.task == "pairs":
        tag += f"-{args.spec}"
        if args.pair:
            tag += "--" + args.pair.replace(":", "_")
    (args.out / f"{tag}.json").write_text(json.dumps(result, indent=1) + "\n")


# ── report ──────────────────────────────────────────────────────────────

RANKERS = {
    "PSST z": ("psst", "z"),
    "PSST z, earlier denominator": ("psst", "z_old"),
    "FAST, default": ("fast", "default"),
    "FAST, Purify flag": ("fast", "purify"),
    "FAST, on its own EBM baseline": ("fast", "ebm"),
    "FAST, no baseline at all": ("fast", "none"),
    "FAST, full-tensor gain": ("fast", "full"),
}


def scores(run: dict, ranker: str, pairs: list[str]) -> np.ndarray:
    source, key = RANKERS[ranker]
    if source == "psst":
        return np.array([run["psst"]["rows"][p][key] for p in pairs])
    return np.array([run["fast"][key][p] for p in pairs])


def rank_of(values: dict, pair: str) -> int:
    """1-based rank of ``pair`` when ``values`` is sorted descending."""
    order = sorted(values, key=lambda k: -values[k])
    return order.index(pair) + 1


def gold(run: dict, kind: str) -> dict:
    refits = run["refits"]
    if kind == "out":
        return {p: r["gain_out"] for p, r in refits.items()}
    if kind == "in":
        return {p: r["gain_in"] for p, r in refits.items()}
    return {p: r["gain_in"] - 2.0 * r["pair_edf"] for p, r in refits.items()}


def correlations(run: dict, kind: str) -> dict:
    target = gold(run, kind)
    pairs = sorted(target)
    g = np.array([target[p] for p in pairs])
    out = {}
    for name in RANKERS:
        s = scores(run, name, pairs)
        out[name] = {
            "spearman": float(stats.spearmanr(s, g).statistic),
            "kendall": float(stats.kendalltau(s, g).statistic),
        }
    return out


def summarise(run: dict) -> dict:
    refits = run["refits"]
    pairs = sorted(refits)
    psst_z = {p: run["psst"]["rows"][p]["z"] for p in pairs}
    out = {
        "counts": run["counts"],
        "mains": {k: run["mains"][k] for k in ("train", "hold", "edf", "converged", "seconds")},
        "n_pairs": len(pairs),
        "correlations": {k: correlations(run, k) for k in ("out", "in", "in_minus_2edf")},
        "pairs": {},
        "fast_orders": {
            k: sorted(run["fast"][k], key=lambda p: -run["fast"][k][p])
            for k in ("default", "purify", "ebm", "none", "full", "bins256", "bins1024", "uniform")
        },
        "psst_seconds": run["psst"]["seconds"],
        "fast_default_seconds": run["fast"]["default_seconds"],
        "refit_seconds": {p: refits[p]["seconds"] for p in pairs},
        "nonconverged": [p for p in pairs if not refits[p]["converged"]]
        + ([] if run["mains"]["converged"] else ["mains"]),
        "counts_offset_ratio": [
            run["fast"]["counts_offset"][p] / run["fast"]["default"][p]
            for p in run["fast"]["default"]
        ],
        "mean_exposure": run["fast"]["mean_exposure"],
        "deviance_at_eta_minus_fit": run["fast"]["deviance_at_eta"] - run["mains"]["fit_deviance"],
    }
    if "bonusmalus" in run["mains"]:
        out["bonusmalus"] = run["mains"]["bonusmalus"]
    for p in pairs:
        r = refits[p]
        out["pairs"][p] = {
            "kind": run["psst"]["rows"][p]["kind"],
            "z": psst_z[p],
            "z_old": run["psst"]["rows"][p]["z_old"],
            "psst_rank": rank_of(psst_z, p),
            "fast_rank_all": rank_of(run["fast"]["default"], p),
            "fast_rank": rank_of({q: run["fast"]["default"][q] for q in pairs}, p),
            "fast_rank_purify": rank_of({q: run["fast"]["purify"][q] for q in pairs}, p),
            "gain_in": r["gain_in"],
            "gain_out": r["gain_out"],
            "rank_in": rank_of(gold(run, "in"), p),
            "rank_out": rank_of(gold(run, "out"), p),
            "pair_edf": r["pair_edf"],
            "delta_edf": r["delta_edf"],
            "shift_ss_exposure": r["shift_ss_exposure"],
            "shift_ss_mu": r["shift_ss_mu"],
            "quadrant_r2": r["quadrant_r2"],
            "slope_r2": r.get("slope_r2"),
        }
    vab = "VehAge:BonusMalus"
    out["fast_rank_vehage_bonusmalus"] = {
        k: rank_of(run["fast"][k], vab)
        for k in ("default", "purify", "ebm", "none", "full", "bins256", "bins1024", "uniform")
    }
    return out


def summarise_baseline(run: dict) -> dict:
    fits = run["fits"]
    grid = np.array(run["bm_grid"])
    expo = np.array(run["bm_train_exposure"])
    out = {"counts": run["counts"], "fits": {}, "shape": {}}
    out["quantile_rows_bit_identical"] = run["quantile_rows_bit_identical"]
    gap = fits["specB"]["hold"] - fits["ebm"]["hold"]
    for name, f in fits.items():
        out["fits"][name] = {
            "train": f["train"],
            "hold": f["hold"],
            "seconds": f["seconds"],
            "gap_closed": fits["specB"]["hold"] - f["hold"],
            "of_gap": gap,
        }
        if "bm_mu" not in f:
            continue
        # The fitted BonusMalus contribution on the log scale, against BM=50:
        # log mu(BM) - log mu(50) with every other feature held fixed.
        logrel = np.log(np.array(f["bm_mu"]))
        logrel -= logrel[0]
        steps = np.diff(logrel)
        at = {int(g): i for i, g in enumerate(grid)}
        dense = grid[1:] <= 150
        out["shape"][name] = {
            "dips": int(np.sum(steps < 0)),
            "violation": float(-steps[steps < 0].sum()),
            "violation_to_150": float(-steps[(steps < 0) & dense].sum()),
            "dips_to_150": int(np.sum((steps < 0) & dense)),
            "at70": float(logrel[at[70]]),
            "at90": float(logrel[at[90]]),
            "at230": float(logrel[at[230]]),
            "max": float(logrel.max()),
            "exposure_above_150": float(expo[grid > 150].sum() / expo.sum()),
        }
    for name in ("specB",):
        if "bonusmalus" in fits[name]:
            out["bonusmalus"] = fits[name]["bonusmalus"]
    out["specA_cost"] = {
        "train": fits["specA"]["train"] - fits["specB"]["train"],
        "hold": fits["specA"]["hold"] - fits["specB"]["hold"],
    }
    return out


def report(out: Path) -> None:
    runs = {}
    for path in sorted(out.glob("*.json")):
        if path.name != "summary.json" and "--" not in path.stem:
            runs[path.stem] = json.loads(path.read_text())
    for path in sorted(out.glob("*--*.json")):
        base = path.stem.split("--")[0]
        if base in runs:
            runs[base]["refits"].update(json.loads(path.read_text())["refits"])
    summary = {}
    for key, run in runs.items():
        summary[key] = (summarise if key.startswith("pairs") else summarise_baseline)(run)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    for key, s in summary.items():
        print(f"== {key}")
        print(json.dumps(s["counts"]))
        if "correlations" not in s:
            for name, f in s["fits"].items():
                print(
                    f"  {name:18s} train {f['train']:10.1f} hold {f['hold']:10.1f} "
                    f"closed {f['gap_closed']:7.1f} of {f['of_gap']:.1f}"
                )
            for name, sh in s["shape"].items():
                print(f"  {name:18s} " + " ".join(f"{k}={v:.3f}" for k, v in sh.items()))
            continue
        for kind, table in s["correlations"].items():
            print(f"  gold={kind}")
            for name, c in table.items():
                print(f"    {name:32s} {c['spearman']:+.3f}  tau {c['kendall']:+.3f}")
        for p, r in sorted(s["pairs"].items(), key=lambda kv: -kv[1]["gain_out"]):
            print(
                f"  {p:22s} "
                + " ".join(
                    f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()
                )
            )


if __name__ == "__main__":
    main()
