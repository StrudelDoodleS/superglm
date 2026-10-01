"""Speed sentinel: a typical pricing model, fitted once with ``fit_reml``.

Most pricing models share one shape: spline and categorical main effects
plus a handful of pairwise interactions, and seldom a random effect.  This
script fits that shape on pg17 (CASdatasets pricing game 2017, Poisson
claim frequency, log link, log exposure offset) so REML speed work is
measured where it matters.

    main effects     the ten P-splines and seven categoricals of step A in
                     benchmark_hierarchical_credibility.py
    cat x cat        pol_coverage x pol_usage, drv_sex1 x drv_drv2
    spline x cat     drv_age1 x drv_sex1, vh_age x vh_type
    spline x spline  drv_age1 x drv_age_lic1, vh_age x vh_value

Interaction types are resolved from the parents (CategoricalInteraction,
SplineCategorical and TensorInteraction) and recorded in the JSON.  Data
preparation, the seeded holdout and the JSON layout reuse
benchmark_hierarchical_credibility.py; each run writes one JSON to
``benchmarks/results/typical_pricing/``.

Pin every thread pool before timing::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
    NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1 SUPERGLM_BLAS_THREADS=1 \\
    uv run python benchmarks/benchmark_typical_pricing.py --discrete
"""

from __future__ import annotations

import argparse
import gc
import os
import platform
import signal
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import superglm
from superglm import Categorical, Spline, SuperGLM

try:
    from benchmarks import benchmark_hierarchical_credibility as hc
    from benchmarks._harness import dump_json
except ModuleNotFoundError:
    import benchmark_hierarchical_credibility as hc
    from _harness import dump_json

RESULTS_ROOT = hc.ROOT / "benchmarks" / "results" / "typical_pricing"
DATASET = hc.PG17
# Every interaction reads step A columns, so step A's loader and level
# trimming serve this model unchanged.
STEP = "A"
INTERACTIONS = (
    ("pol_coverage", "pol_usage"),
    ("drv_sex1", "drv_drv2"),
    ("drv_age1", "drv_sex1"),
    ("vh_age", "vh_type"),
    ("drv_age1", "drv_age_lic1"),
    ("vh_age", "vh_value"),
)


def build_model(discrete: bool) -> tuple[SuperGLM, dict[str, str]]:
    """Step A's main effects plus the pricing interactions, and a plain description."""
    spline = f"Spline(kind='ps', k={hc.SPLINE_K})"
    features = {column: Spline(kind="ps", k=hc.SPLINE_K) for column in DATASET.splines}
    features |= {column: Categorical(unseen="base") for column in DATASET.categoricals}
    terms = {column: spline for column in DATASET.splines}
    terms |= {column: "Categorical(unseen='base')" for column in DATASET.categoricals}
    model = SuperGLM(
        family=DATASET.family,
        features=features,
        interactions=list(INTERACTIONS),
        discrete=discrete,
    )
    return model, terms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--discrete", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_ROOT)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if not DATASET.path.exists():
        parser.error(f"{DATASET.path} is missing; run {DATASET.fetch_script} --dest data/")
    return args


def main() -> None:
    args = parse_args()
    thread_env = {name: os.environ.get(name) for name in hc.THREAD_ENV_VARS}
    threads_pinned = all(value == "1" for value in thread_env.values())
    if not threads_pinned:
        print(f"warning: thread pools are not all pinned to 1: {thread_env}", file=sys.stderr)
    mode = "discrete" if args.discrete else "exact"
    case_id = f"pg17_pricing_{mode}_seed{args.seed}"

    load_start = time.perf_counter()
    frame, row_counts = hc.read_rows(DATASET, STEP, None, args.seed)
    train, holdout, accounting = hc.prepare(frame, DATASET, STEP)
    del frame
    data_load_s = time.perf_counter() - load_start

    model, terms = build_model(args.discrete)
    X = train[DATASET.model_columns(STEP)]
    y = train[DATASET.response].to_numpy(np.float64)
    offset = hc.log_exposure(train, DATASET)
    warmup_start = time.perf_counter()
    superglm.warmup()
    warmup_s = time.perf_counter() - warmup_start
    gc.collect()

    signal.signal(signal.SIGTERM, hc.stop_fit)
    fit, memory = hc.timed_fit(model, X, y, offset)
    result = (
        hc.fitted_report(model, DATASET, STEP, train, holdout) if fit["status"] == "ok" else None
    )
    interaction_types = None
    if result is not None:
        interactions = model.training_telemetry()["features"]["interactions"]
        interaction_types = {name: spec["class"] for name, spec in interactions.items()}
    payload = {
        "schema_version": 1,
        "case_id": case_id,
        "dataset": DATASET.name,
        "seed": args.seed,
        "discrete": args.discrete,
        "rows": row_counts | accounting,
        "model": {
            "family": DATASET.family,
            "response": DATASET.response,
            "offset": f"log({DATASET.exposure})",
            "terms": terms,
            "interactions": interaction_types,
            "fit_reml_options": hc.FIT_REML_OPTIONS,
        },
        "fit": fit,
        "memory": memory,
        "result": result,
        "environment": {
            "thread_env": thread_env,
            "threads_pinned_to_one": threads_pinned,
            "cpu_count": os.cpu_count(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "superglm": superglm.__version__,
            "data_path": str(DATASET.path.relative_to(hc.ROOT)),
            "data_load_s": data_load_s,
            "warmup_s": warmup_s,
        },
    }
    output = args.out_dir / f"{case_id}.json"
    dump_json(output, payload)
    backend = None if result is None else result["backend"]["dispatched"]
    print(
        f"{case_id}: {fit['status']} wall={fit['wall_s']:.2f}s backend={backend} "
        f"peak_rss={memory['peak_rss_mib_after_fit']:.0f}MiB -> {output}"
    )


if __name__ == "__main__":
    main()
