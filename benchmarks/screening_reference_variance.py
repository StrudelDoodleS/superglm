"""Compare complete fits, screens and candidate refits across two revisions.

Run this same file with each revision's installed package and save separate
JSON receipts. Timing runs should be separate processes with the same BLAS and
Numba thread limits. ``--refit`` measures every candidate on independent data;
the planted pairs are known before either ranking is computed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import resource
import subprocess
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

import superglm
import superglm.model.screening_ops as ops
from superglm import SuperGLM
from superglm.features import Categorical, Numeric, Spline


def _sample(case, n, seed, strength):
    rng = np.random.default_rng(seed)
    if case == "variance_budget":
        x = np.linspace(0, 1, 5094)[np.arange(n) % 5094]
        group = np.arange(n) % 200
        rng.shuffle(group)
        frame = pd.DataFrame({"x": x, "g": group.astype(str)})
        features = {"g": Categorical(), "x": Spline(kind="ps", n_knots=8)}
        return (
            frame,
            np.sin(3 * x) + rng.normal(size=n),
            features,
            set(),
            {"candidates": [("x", "g")], "edf0": (2.0, 4.0, 8.0, 16.0)},
        )
    if case in ("structured", "structured_wide"):
        wide = case == "structured_wide"
        x = rng.integers(0, 101, n) / 100
        group = rng.integers(0, 34 if wide else 160, n)
        frame = pd.DataFrame({"x": x, "g": [f"L{j}" for j in group]})
        mean = np.sin(3 * x) + 0.1 * np.sin(group)
        mean += strength * np.sin(5 * x) * np.cos(group)
        features = {
            "x": Spline(kind="ps", n_knots=42) if wide else Spline(kind="ps", k=8),
            "g": Categorical(),
        }
        return (
            frame,
            mean + rng.normal(size=n),
            features,
            {("x", "g")},
            {"max_cells": 1_000_000 if wide else 50_000},
        )
    x, z, v, w = rng.uniform(-1.0, 1.0, (4, n))
    g = rng.integers(0, 3, n)
    h = rng.integers(0, 5, n)
    # Fixed support permits exact screens, with no changing binning geometry.
    x = np.round(x, 2)
    z = np.round(z, 2)
    frame = pd.DataFrame({"x": x, "z": z, "v": v, "w": w, "g": g.astype(str), "h": h.astype(str)})
    mean = 0.3 * np.sin(2 * x) + 0.2 * z + 0.1 * v + 0.1 * g
    mean += strength * (np.sin(3 * x) * np.sin(3 * z) + v * (g - 1) + (g - 1) * (h - 2) / 2)
    features = {
        "x": Spline(kind="ps", k=8),
        "z": Spline(kind="ps", k=8),
        "v": Numeric(),
        "w": Numeric(),
        "g": Categorical(),
        "h": Categorical(),
    }
    return frame, mean + rng.normal(size=n), features, {("x", "z"), ("v", "g"), ("g", "h")}, {}


@contextmanager
def _dispatch_counts():
    counts = Counter()
    originals = {
        name: getattr(ops, name)
        for name in ("penalized_score_statistic_ladder", "structured_ladder")
    }

    def wrap(name, function):
        def record(*args, **kwargs):
            counts[name] += 1
            result = function(*args, **kwargs)
            if result is None:
                counts[name + "_refused"] += 1
            return result

        return record

    try:
        for name, function in originals.items():
            setattr(ops, name, wrap(name, function))
        yield counts
    finally:
        for name, function in originals.items():
            setattr(ops, name, function)


def _fit(frame, y, features, interactions=()):
    model = SuperGLM(
        family="gaussian", features=copy.deepcopy(features), interactions=list(interactions)
    ).fit_reml(frame, y)
    diagnostics = model.reml_diagnostics()
    if not model.result.converged or (diagnostics["enabled"] and not diagnostics.get("converged")):
        raise RuntimeError(f"Benchmark fit did not converge: {diagnostics}")
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=("mixed", "structured", "structured_wide", "variance_budget"),
        default="mixed",
    )
    parser.add_argument("--rows", type=int, default=40_000)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--strength", type=float, default=0.2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--refit", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame, y, features, planted, screen_kwargs = _sample(
        args.case, args.rows, args.seed, args.strength
    )
    holdout, y_holdout, _, _, _ = _sample(args.case, args.rows, args.seed + 1, args.strength)
    # Warm compilation on a separate model before the repeated complete fits.
    warm = _fit(frame, y, features)
    warm.screen_interactions(frame, y, **screen_kwargs)
    fits, screens, predictions, dispatches = [], [], [], []
    load_before = os.getloadavg()
    for _ in range(args.repeats):
        start = time.perf_counter()
        model = _fit(frame, y, features)
        fit_seconds = time.perf_counter() - start
        with _dispatch_counts() as dispatch:
            start = time.perf_counter()
            table = model.screen_interactions(frame, y, **screen_kwargs)
            screen_seconds = time.perf_counter() - start
        fits.append(fit_seconds)
        screens.append(screen_seconds)
        predictions.append(np.asarray(model.predict(holdout)))
        dispatches.append(dict(dispatch))
    baseline_loss = float(np.sum((y_holdout - predictions[-1]) ** 2))
    candidates = table.to_dict(orient="records")
    for row in candidates:
        # Refused pairs remain in the receipt as null scores, rather than
        # making a before/after regression measurement impossible to save.
        for key, value in row.items():
            if isinstance(value, float) and not np.isfinite(value):
                row[key] = None
        pair = (row["feature_a"], row["feature_b"])
        row["planted"] = pair in planted or pair[::-1] in planted
        if args.refit:
            start = time.perf_counter()
            alternative = _fit(frame, y, features, interactions=(pair,))
            row["refit_seconds"] = time.perf_counter() - start
            row["holdout_gain"] = baseline_loss - float(
                np.sum((y_holdout - alternative.predict(holdout)) ** 2)
            )
            row["training_gain"] = float(model._result.deviance - alternative._result.deviance)
    source_root = Path(superglm.__file__).resolve().parents[2]
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
    ).strip()
    source_hash = hashlib.sha256()
    for path in sorted((source_root / "src/superglm").rglob("*.py")):
        source_hash.update(path.relative_to(source_root).as_posix().encode())
        source_hash.update(path.read_bytes())
    receipt = {
        "case": args.case,
        "rows": args.rows,
        "holdout_rows": args.rows,
        "seed": args.seed,
        "strength": args.strength,
        "revision": revision,
        "source_sha256": source_hash.hexdigest(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "threads": {
            key: os.environ.get(key)
            for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS")
        },
        "fit_seconds": fits,
        "screen_seconds": screens,
        "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "dispatch": dispatches,
        "load_before": load_before,
        "load_after": os.getloadavg(),
        "prediction_sum": float(np.sum(predictions[-1])),
        "prediction_sum_squares": float(np.sum(predictions[-1] ** 2)),
        "repeat_prediction_max_difference": float(
            max(np.max(np.abs(p - predictions[-1])) for p in predictions)
        ),
        "holdout_mains_loss": baseline_loss,
        "mains_edf": float(model._result.effective_df),
        "mains_deviance": float(model._result.deviance),
        "phi": table.attrs["phi"],
        "candidates": candidates,
    }
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(
        f"{args.case}: {len(table)} pairs; fit {np.median(fits):.3f}s; screen {np.median(screens):.3f}s; "
        f"peak RSS {receipt['process_peak_rss_mib']:.1f} MiB; {dispatches[-1]}",
        flush=True,
    )


if __name__ == "__main__":
    main()
