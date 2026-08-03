"""Tensor-interaction cost benchmark: exact vs discrete fit_reml on freMTPL2.

Tracks the baseline this branch is trying to move. Compares

    s(DrivAge) + s(VehAge) + s(BonusMalus) + s(VehPower) + Area

against the same model plus ``ti(DrivAge, BonusMalus)``, on both the exact and
the ``discrete=True`` REML paths, recording wall time and the model's own
``_reml_profile`` phase timings.

Baseline at f082e9b, n=100_000 (see docs/audit/2026-07-28/measured-tensor-cost.md):
one tensor term costs 7.2x on exact and 3.8x on discrete.

Usage::

    uv run python benchmarks/benchmark_tensor_cost.py --n 100000
    uv run python benchmarks/benchmark_tensor_cost.py --n 60000 --profile
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import time
from pathlib import Path

import numpy as np
import pandas as pd

from superglm import SuperGLM
from superglm.features.categorical import Categorical
from superglm.features.spline import Spline

ROOT = Path(__file__).resolve().parents[1]


def _read_frame() -> pd.DataFrame:
    """Read freMTPL2freq, preferring parquet but degrading to CSV.

    The parquet engine is an optional dependency, so a parquet file being
    present does not mean it is readable.
    """
    candidates = [
        base / "data" / name
        for base in (ROOT, ROOT.parent.parent)
        for name in ("freMTPL2freq.parquet", "freMTPL2freq.csv")
    ]
    found = [path for path in candidates if path.exists()]
    if not found:
        raise FileNotFoundError("freMTPL2freq data not found under data/")
    errors = []
    for path in found:
        try:
            return pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
        except ImportError as exc:
            errors.append(f"{path.name}: {exc}")
    raise RuntimeError("no readable freMTPL2freq source; " + "; ".join(errors))


OUT_DIR = ROOT / "benchmarks" / "results"
OUT_JSON = OUT_DIR / "tensor_cost.json"

SPLINE_COLS = ("DrivAge", "VehAge", "BonusMalus", "VehPower")
TENSOR_PAIR = ("DrivAge", "BonusMalus")


def load(n_rows: int | None, seed: int = 0):
    df = _read_frame()
    df["ClaimNb"] = np.asarray(df["ClaimNb"], dtype=float).clip(0, 4)
    df["Exposure"] = np.asarray(df["Exposure"], dtype=float).clip(1e-3, 1.0)
    if n_rows is not None and n_rows < len(df):
        idx = np.random.default_rng(seed).choice(len(df), size=n_rows, replace=False)
        df = df.iloc[idx].reset_index(drop=True)
    X = df[[*SPLINE_COLS, "Area"]].copy()
    for col in SPLINE_COLS:
        X[col] = X[col].astype(float)
    X["Area"] = X["Area"].astype(str)
    y = (df["ClaimNb"] / df["Exposure"]).to_numpy()
    weights = df["Exposure"].to_numpy()
    return X, y, weights


def build(with_tensor: bool, discrete: bool, k: int = 10) -> SuperGLM:
    features: dict[str, object] = {col: Spline(kind="ps", k=k) for col in SPLINE_COLS}
    features["Area"] = Categorical(base="first")
    model = SuperGLM(family="poisson", selection_penalty=None, discrete=discrete, features=features)
    if with_tensor:
        model._add_interaction(*TENSOR_PAIR)
    return model


def _peak_rss_mb() -> float | None:
    """Whole-run high-water RSS in MB (recorded as artifact history, not gated:
    the value is dominated by the full-frame load and not per-case
    attributable without subprocess isolation). None where unavailable."""
    try:
        import resource
    except ImportError:  # non-Unix platforms have no resource module
        return None
    import sys

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is kilobytes on Linux but bytes on macOS.
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return round(peak / divisor, 1)


def run_case(
    tag: str, with_tensor: bool, discrete: bool, n_rows: int | None, profile: bool, reps: int = 1
) -> dict:
    X, y, weights = load(n_rows)
    # With reps >= 2 the first fit is a warmup (import/cache effects measured
    # at ~1.5-3x steady state) and is excluded from the median, matching the
    # flagship harness.
    warmup_reps = 1 if reps >= 2 else 0
    rep_seconds: list[float] = []
    warmup_seconds: list[float] = []
    ok, error = True, ""
    model = None
    for rep in range(warmup_reps + reps):
        model = build(with_tensor, discrete)
        profiler = cProfile.Profile() if profile and rep == 0 else None
        start = time.perf_counter()
        if profiler is not None:
            profiler.enable()
        try:
            model.fit_reml(X, y, sample_weight=weights)
        except Exception as exc:  # noqa: BLE001 - benchmark records failures
            ok, error = False, f"{type(exc).__name__}: {exc}"[:300]
        elapsed = time.perf_counter() - start
        if profiler is not None:
            profiler.disable()
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            profiler.dump_stats(str(OUT_DIR / f"{tag}.prof"))
            stream = io.StringIO()
            pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(45)
            (OUT_DIR / f"{tag}.pstats.txt").write_text(stream.getvalue())
        if not ok:
            break
        if rep < warmup_reps:
            warmup_seconds.append(elapsed)
        else:
            rep_seconds.append(elapsed)

    record = {
        "tag": tag,
        "tensor": with_tensor,
        "discrete": discrete,
        "n": int(len(y)),
        "seconds": round(float(np.median(rep_seconds)), 3) if rep_seconds else None,
        "reps": reps,
        "rep_seconds": [round(s, 3) for s in rep_seconds],
        "warmup_seconds": [round(s, 3) for s in warmup_seconds],
        "ok": ok,
        "error": error,
    }
    if ok:
        record["p"] = int(model._dm.p)
        record["deviance"] = float(model.result.deviance)
        record["effective_df"] = float(model.result.effective_df)
        record["direct_backend"] = getattr(model.result, "direct_backend", None)
        phases = getattr(model, "_reml_profile", None) or {}
        for key in sorted(phases):
            value = phases.get(key)
            if key.endswith("_s") and value and float(value) > 0.02:
                record[f"prof.{key}"] = round(float(value), 3)
    return record


CASES = (
    ("tensor_cost_base_exact", False, False),
    ("tensor_cost_base_discrete", False, True),
    ("tensor_cost_ti_exact", True, False),
    ("tensor_cost_ti_discrete", True, True),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100_000, help="rows (0 = full data)")
    parser.add_argument("--profile", action="store_true", help="dump cProfile artifacts")
    parser.add_argument(
        "--reps",
        type=int,
        default=1,
        help="steady-state fits per case; the median is recorded, and with "
        "reps >= 2 one extra warmup fit runs first and is excluded",
    )
    parser.add_argument("--json-out", default=str(OUT_JSON), help="result JSON path")
    args = parser.parse_args()
    if args.reps < 1:
        parser.error("--reps must be >= 1")
    n_rows = None if args.n == 0 else args.n

    rows = []
    for tag, with_tensor, discrete in CASES:
        record = run_case(tag, with_tensor, discrete, n_rows, args.profile, reps=args.reps)
        rows.append(record)
        print(json.dumps(record), flush=True)

    by_tag = {row["tag"]: row for row in rows if row["ok"]}
    summary: dict[str, float] = {}
    for path in ("exact", "discrete"):
        base = by_tag.get(f"tensor_cost_base_{path}")
        tensor = by_tag.get(f"tensor_cost_ti_{path}")
        if base and tensor and base["seconds"] > 0:
            summary[f"tensor_multiplier_{path}"] = round(tensor["seconds"] / base["seconds"], 2)
    for kind in ("base", "ti"):
        exact = by_tag.get(f"tensor_cost_{kind}_exact")
        discrete = by_tag.get(f"tensor_cost_{kind}_discrete")
        if exact and discrete and discrete["seconds"] > 0:
            summary[f"exact_over_discrete_{kind}"] = round(
                exact["seconds"] / discrete["seconds"], 2
            )

    summary["peak_rss_mb"] = _peak_rss_mb()

    out_json = Path(args.json_out)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"cases": rows, "summary": summary}, indent=2))
    print(json.dumps({"summary": summary}, indent=2))
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
