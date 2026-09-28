"""Benchmark nested random effects (credibility) over a vehicle hierarchy.

Each invocation fits ONE case, so the process peak RSS belongs to that fit,
and writes one JSON to ``benchmarks/results/hierarchical_credibility/``.
Steps are cumulative: each adds one random-effect level beside the same
ordinary rating terms, fitted with ``fit_reml``.

pg17 (CASdatasets pricing game 2017, ``scripts/fetch_cas_pg17.py``): Poisson
claim frequency, log link, exposure 1.0 per policy-year.

    A    driver, policy and vehicle-attribute smooths and categoricals
    B    A + RandomEffect(vh_make)
    C    B + RandomEffect(make_model)
    E    C + RandomEffect(pol_insee_code), crossed with the vehicle hierarchy
    REF  A + Categorical(vh_make) + Categorical(make_model) as fixed effects

dvsa (DVSA MOT results 2024, Open Government Licence v3,
``scripts/fetch_dvsa_mot.py``): binomial initial failure, logit link.

    A    smooths of vehicle age, mileage and engine capacity + fuel type
    B    A + RandomEffect(make)
    C    B + RandomEffect(make_model)
    D    C + RandomEffect(variant)
    E    D + RandomEffect(postcode_area), crossed with the vehicle hierarchy

The structured direct backend eliminates only the largest RandomEffect; every
other RandomEffect joins the dense block, so a second large level is the
predicted cost cliff.  The JSON reads the dispatched backend from the fit's
own telemetry (``training_telemetry()`` and ``reml_diagnostics()``).

Data preparation, recorded in every JSON:

- Rows missing a smooth input are dropped and counted per column.  pg17 also
  reads a zero vh_cyl, vh_din, vh_speed, vh_value or vh_weight as missing,
  since none of them is a real measurement.
- The holdout is a deterministic 20% of groups (id_client for pg17,
  vehicle_id for dvsa) from a seeded hash of the group key, so all rows of a
  group fall on one side.  ``--rows`` subsamples dvsa by whole vehicles from
  the same hash while streaming the parquet, so the full file is never held;
  subsamples at one seed are nested.
- Smooth inputs are winsorised at the training 0.1% and 99.9% quantiles.
  Discrete fits bin a column into equal-width bins over its range, so DVSA's
  recorded mileages up to 999,999 and capacities up to 99,999 would otherwise
  leave almost every row in a handful of bins.
- Level columns keep only the levels seen in training: a categorical dtype
  would otherwise hand every category of the full file to the model as its
  level universe.  Holdout levels unseen in training go to the population
  mean for random effects and to the base level for categoricals, REF's
  fixed make and model effects included.

A run stopped by SIGTERM, for example by ``timeout -k 60 3600``, still
writes its JSON, recording the fit as a ``TimeoutError`` with its wall time
and peak RSS so far.

Pin every thread pool before timing::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
    NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1 SUPERGLM_BLAS_THREADS=1 \\
    uv run python benchmarks/benchmark_hierarchical_credibility.py \\
        --dataset dvsa --step D --rows 200000 --discrete
"""

from __future__ import annotations

import argparse
import gc
import os
import platform
import resource
import signal
import sys
import time
import traceback
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

import superglm
from superglm import Categorical, RandomEffect, Spline, SuperGLM, lorenz_curve

try:
    from benchmarks._harness import SystemSampler, dump_json, summarize_system_samples
except ModuleNotFoundError:
    from _harness import SystemSampler, dump_json, summarize_system_samples

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_ROOT = ROOT / "benchmarks" / "results" / "hierarchical_credibility"

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "SUPERGLM_BLAS_THREADS",
)
HOLDOUT_SHARE = 0.2
WINSOR_QUANTILES = (0.001, 0.999)
SPLINE_K = 10
SAMPLE_INTERVAL_S = 0.2
MIB = 1024.0**2
# The post-fit public-runtime parity diagnostic is not part of fitting.
FIT_REML_OPTIONS = {"runtime_validation": "skip"}
STRUCTURED_PROFILE_KEYS = (
    "structured_dominant_group",
    "structured_auto_selected",
    "structured_auto_cost_ratio",
    "structured_used_dense_fallback",
    "structured_fallback_reason",
    "structured_schur_condition",
)


@dataclass(frozen=True)
class Dataset:
    """One benchmark dataset: its file, family, rating terms and cumulative steps."""

    name: str
    path: Path
    fetch_script: str
    family: str
    response: str
    exposure: str | None
    split_key: str
    splines: tuple[str, ...]
    zero_is_missing: tuple[str, ...]
    categoricals: tuple[str, ...]
    random_effects: dict[str, tuple[str, ...]]
    fixed_level_effects: dict[str, tuple[str, ...]]
    report_gini: bool

    def level_columns(self, step: str) -> list[str]:
        return [
            *self.categoricals,
            *self.random_effects[step],
            *self.fixed_level_effects.get(step, ()),
        ]

    def model_columns(self, step: str) -> list[str]:
        return [*self.splines, *self.level_columns(step)]

    def read_columns(self, step: str) -> list[str]:
        exposure = () if self.exposure is None else (self.exposure,)
        return [*self.model_columns(step), self.response, *exposure]


PG17 = Dataset(
    name="pg17",
    path=DATA_DIR / "cas_pg17.parquet",
    fetch_script="scripts/fetch_cas_pg17.py",
    family="poisson",
    response="claim_count",
    exposure="exposure",
    split_key="id_client",
    splines=(
        "drv_age1",
        "drv_age_lic1",
        "pol_duration",
        "pol_bonus",
        "vh_age",
        "vh_din",
        "vh_value",
        "vh_weight",
        "vh_speed",
        "vh_cyl",
    ),
    zero_is_missing=("vh_cyl", "vh_din", "vh_speed", "vh_value", "vh_weight"),
    categoricals=(
        "pol_coverage",
        "pol_usage",
        "pol_pay_freq",
        "drv_sex1",
        "drv_drv2",
        "vh_fuel",
        "vh_type",
    ),
    random_effects={
        "A": (),
        "B": ("vh_make",),
        "C": ("vh_make", "make_model"),
        "E": ("vh_make", "make_model", "pol_insee_code"),
        "REF": (),
    },
    fixed_level_effects={"REF": ("vh_make", "make_model")},
    report_gini=True,
)

DVSA = Dataset(
    name="dvsa",
    path=DATA_DIR / "dvsa_mot_2024.parquet",
    fetch_script="scripts/fetch_dvsa_mot.py",
    family="binomial",
    response="fail",
    exposure=None,
    split_key="vehicle_id",
    splines=("vehicle_age_years", "test_mileage", "cylinder_capacity"),
    zero_is_missing=(),
    categoricals=("fuel_type",),
    random_effects={
        "A": (),
        "B": ("make",),
        "C": ("make", "make_model"),
        "D": ("make", "make_model", "variant"),
        "E": ("make", "make_model", "variant", "postcode_area"),
    },
    fixed_level_effects={},
    report_gini=False,
)

DATASETS = {dataset.name: dataset for dataset in (PG17, DVSA)}


def group_uniform(keys: np.ndarray, seed: int) -> np.ndarray:
    """Seeded U[0, 1) draw per group key; every row of a group gets the same draw.

    pandas' public ``hash_array`` maps any key to uint64.  XOR-ing the seed in
    and hashing again (its integer path is the splitmix64 finaliser) gives an
    independent stream per seed.
    """
    hashed = pd.util.hash_array(keys) ^ np.uint64(seed)
    mixed = pd.util.hash_array(hashed)
    return (mixed >> np.uint64(11)).astype(np.float64) * 2.0**-53


def read_rows(
    dataset: Dataset, step: str, rows: int | None, seed: int
) -> tuple[pd.DataFrame, dict]:
    """Read row group by row group, keeping whole groups whose draw falls under the row share.

    One row group at a time keeps Arrow's allocation bounded; ``iter_batches``
    was measured to accumulate about 700 MiB over the DVSA file while holding
    nothing.  The returned ``draw`` column is rescaled to U[0, 1) within the
    kept rows, so the holdout split reuses it.
    """
    parquet = pq.ParquetFile(dataset.path)
    available = parquet.metadata.num_rows
    share = 1.0 if rows is None else min(1.0, rows / available)
    columns = dataset.read_columns(step)
    kept, draws = [], []
    for index in range(parquet.num_row_groups):
        table = parquet.read_row_group(index, columns=[*columns, dataset.split_key])
        draw = group_uniform(table.column(dataset.split_key).to_numpy(), seed)
        selected = draw < share
        kept.append(table.select(columns).filter(pa.array(selected)))
        draws.append(draw[selected])
    frame = pa.concat_tables(kept).to_pandas()
    frame["draw"] = np.concatenate(draws) / share
    return frame, {"available": available, "requested": rows, "selected": len(frame)}


def level_counts(train_column: pd.Series, holdout_column: pd.Series) -> dict[str, int]:
    """Training level count and the holdout levels and rows unseen in training."""
    unseen = holdout_column[~holdout_column.isin(train_column.cat.categories)]
    return {
        "train_levels": len(train_column.cat.categories),
        "holdout_unseen_levels": int(unseen.nunique()),
        "holdout_unseen_rows": len(unseen),
    }


def prepare(
    frame: pd.DataFrame, dataset: Dataset, step: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Drop rows missing a smooth input, split by group, winsorise and trim levels."""
    splines = list(dataset.splines)
    zero_columns = list(dataset.zero_is_missing)
    frame[splines] = frame[splines].astype(np.float64)
    missing = frame[splines].isna()
    missing[zero_columns] |= frame[zero_columns].eq(0.0)
    has_missing = missing.any(axis=1).to_numpy()
    frame = frame.loc[~has_missing]

    in_holdout = frame["draw"].to_numpy() >= 1.0 - HOLDOUT_SHARE
    train = frame.loc[~in_holdout].drop(columns="draw").reset_index(drop=True)
    holdout = frame.loc[in_holdout].drop(columns="draw").reset_index(drop=True)

    lower = train[splines].quantile(WINSOR_QUANTILES[0])
    upper = train[splines].quantile(WINSOR_QUANTILES[1])
    clipped = train[splines].lt(lower) | train[splines].gt(upper)
    train[splines] = train[splines].clip(lower, upper, axis=1)
    holdout[splines] = holdout[splines].clip(lower, upper, axis=1)

    levels = dataset.level_columns(step)
    train[levels] = train[levels].apply(
        lambda column: column.astype("category").cat.remove_unused_categories()
    )
    accounting = {
        "missing_by_column": missing.sum().to_dict(),
        "dropped_missing": int(has_missing.sum()),
        "train": len(train),
        "holdout": len(holdout),
        "winsor_quantiles": list(WINSOR_QUANTILES),
        "winsor_bounds": {
            column: [float(lower[column]), float(upper[column])] for column in splines
        },
        "winsor_clipped_train_rows": clipped.sum().to_dict(),
        "levels": {column: level_counts(train[column], holdout[column]) for column in levels},
    }
    return train, holdout, accounting


def build_model(dataset: Dataset, step: str, discrete: bool) -> tuple[SuperGLM, dict[str, str]]:
    """Return the unfitted model for one step and a plain description of its terms."""
    terms = {column: f"Spline(kind='ps', k={SPLINE_K})" for column in dataset.splines}
    features: dict[str, Any] = {column: Spline(kind="ps", k=SPLINE_K) for column in dataset.splines}
    fixed = (*dataset.categoricals, *dataset.fixed_level_effects.get(step, ()))
    for column in fixed:
        features[column] = Categorical(unseen="base")
        terms[column] = "Categorical(unseen='base')"
    for column in dataset.random_effects[step]:
        features[column] = RandomEffect(unseen="population")
        terms[column] = "RandomEffect(unseen='population')"
    model = SuperGLM(family=dataset.family, features=features, discrete=discrete)
    return model, terms


def log_exposure(frame: pd.DataFrame, dataset: Dataset) -> np.ndarray | None:
    if dataset.exposure is None:
        return None
    return np.log(frame[dataset.exposure].to_numpy(np.float64))


def stop_fit(signum: int, frame: object) -> None:
    """Raise on a termination request, so the fit's except clause records the stopped fit.

    Later deliveries are ignored: GNU ``timeout`` signals the process group
    and ``uv run`` forwards the same signal, so it arrives twice.  Launchers
    should add a kill grace, as in ``timeout -k 60``.
    """
    signal.signal(signum, signal.SIG_IGN)
    raise TimeoutError(f"fit stopped by {signal.Signals(signum).name}")


def peak_rss_mib() -> float:
    # ru_maxrss is KiB on Linux and bytes on macOS.
    scale = MIB if sys.platform == "darwin" else 1024.0
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale


def warning_records(caught: list[warnings.WarningMessage]) -> list[dict[str, Any]]:
    """Distinct warnings with how often each was raised."""
    counts = Counter((item.category.__name__, str(item.message)) for item in caught)
    return [
        {"category": category, "message": message, "count": count}
        for (category, message), count in counts.items()
    ]


def timed_fit(
    model: SuperGLM, X: pd.DataFrame, y: np.ndarray, offset: np.ndarray | None
) -> tuple[dict, dict]:
    """Fit once, with only the ``fit_reml`` call inside the timers and the sampler."""
    sampler = SystemSampler(interval_s=SAMPLE_INTERVAL_S)
    rss_before = psutil.Process().memory_info().rss / MIB
    peak_before = peak_rss_mib()
    load_start = os.getloadavg()
    error = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sampler.start()
        wall_start, cpu_start = time.perf_counter(), time.process_time()
        try:
            model.fit_reml(X, y, offset=offset, **FIT_REML_OPTIONS)
        except Exception as exc:  # a failed fit is this case's result, not a crash
            error = exc
        wall_s = time.perf_counter() - wall_start
        cpu_s = time.process_time() - cpu_start
        sampler.stop()
    peak_after = peak_rss_mib()
    fit = {
        "status": "ok" if error is None else "error",
        "error": None
        if error is None
        else {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(traceback.format_exception(error)),
        },
        "wall_s": wall_s,
        "process_cpu_s": cpu_s,
        "load_avg_start": list(load_start),
        "load_avg_end": list(os.getloadavg()),
        "warnings": warning_records(caught),
    }
    sampled = summarize_system_samples(sampler.samples)
    memory = {
        "rss_mib_before_fit": rss_before,
        "peak_rss_mib_before_fit": peak_before,
        "peak_rss_mib_after_fit": peak_after,
        "fit_raised_process_peak": peak_after > peak_before,
        "sampler_rss_peak_mib": sampled.get("rss_peak_bytes", 0) / MIB,
        "sampler_interval_s": SAMPLE_INTERVAL_S,
        "sampler_error": sampler.error,
        "sampler_summary": sampled,
    }
    return fit, memory


def random_effect_report(model: SuperGLM, column: str) -> dict[str, Any]:
    effect = model.random_effects(column)
    return {
        "lambda": effect.lambda_value,
        "tau_squared": effect.tau_squared,
        "standard_deviation": effect.standard_deviation,
        "effective_df": effect.effective_df,
        "levels": len(effect.table),
        "levels_without_information": effect.diagnostics["n_levels_without_information"],
        "collapsed": effect.collapsed,
        "at_lower_boundary": effect.at_lower_boundary,
        "at_upper_boundary": effect.at_upper_boundary,
    }


def holdout_report(
    model: SuperGLM, dataset: Dataset, step: str, holdout: pd.DataFrame
) -> dict[str, Any]:
    """Holdout mean unit deviance, and the Lorenz Gini where the dataset asks for it."""
    y = holdout[dataset.response].to_numpy(np.float64)
    mu = model.predict(holdout[dataset.model_columns(step)], offset=log_exposure(holdout, dataset))
    report: dict[str, Any] = {
        "rows": len(holdout),
        "mean_deviance": float(np.mean(model.distribution_.deviance_unit(y, mu))),
        "response_mean": float(y.mean()),
        "prediction_mean": float(mu.mean()),
    }
    if dataset.report_gini:
        import matplotlib.pyplot as plt

        exposure = holdout[dataset.exposure].to_numpy(np.float64)
        curve = lorenz_curve(y / exposure, mu / exposure, exposure=exposure)
        plt.close(curve.figure)
        report |= {"gini": curve.gini_model, "gini_ratio": curve.gini_ratio}
    return report


def fitted_report(
    model: SuperGLM, dataset: Dataset, step: str, train: pd.DataFrame, holdout: pd.DataFrame
) -> dict[str, Any]:
    """Backend, REML, coefficient, variance-component and holdout results of a fit."""
    telemetry = model.training_telemetry()
    fit_meta = telemetry["fit"]["fit_meta"]
    reml = telemetry["reml"]
    profile = reml["profile"]
    groups = pd.DataFrame(telemetry["features"]["groups"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        random_effects = {
            column: random_effect_report(model, column) for column in dataset.random_effects[step]
        }
        holdout_result = holdout_report(model, dataset, step, holdout)
    return {
        "backend": {
            "source": "training_telemetry()['fit']['fit_meta'] and reml_diagnostics()['profile']",
            "direct_solve_requested": telemetry["model"]["direct_solve"],
            "dispatched": fit_meta["direct_backend"],
            "fallback_reason": fit_meta["direct_fallback_reason"],
            **{key: profile.get(key) for key in STRUCTURED_PROFILE_KEYS},
        },
        "reml": {
            "n_reml_iter": reml["n_reml_iter"],
            "converged": reml["converged"],
            "termination_reason": reml["termination_reason"],
            "objective": reml["objective"],
            "requested_max_reml_iter": profile["requested_max_reml_iter"],
            "reml_tol": profile["reml_tol_resolved"],
            "pirls_solves_total": profile["irls_calls"],
            "pirls_iterations_total": profile["irls_iters"],
            "final_pirls_iterations": telemetry["fit"]["n_iter"],
            "final_pirls_converged": telemetry["fit"]["converged"],
            "lambdas": reml["lambdas"],
        },
        "coefficients": {
            "total": len(model.result.beta),
            "intercept": 1,
            "by_term": groups.groupby("feature_name")["size"].sum().to_dict(),
        },
        "random_effects": random_effects,
        "edf": telemetry["edf"]["by_feature"],
        "edf_total": telemetry["edf"]["total"],
        "train_mean_deviance": telemetry["fit"]["deviance"] / len(train),
        "holdout": holdout_result,
        "post_fit_warnings": warning_records(caught),
        "phase_profile": profile,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--step", choices=("A", "B", "C", "D", "E", "REF"), required=True)
    parser.add_argument(
        "--rows", type=int, default=None, help="dvsa only: subsample about this many rows"
    )
    parser.add_argument("--discrete", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_ROOT)
    args = parser.parse_args()
    dataset = DATASETS[args.dataset]
    if args.step not in dataset.random_effects:
        parser.error(f"step {args.step} is not defined for {args.dataset}")
    if args.rows is not None and args.dataset != "dvsa":
        parser.error("--rows applies to dvsa only")
    if args.rows is not None and args.rows < 1:
        parser.error("--rows must be positive")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if not dataset.path.exists():
        parser.error(f"{dataset.path} is missing; run {dataset.fetch_script} --dest data/")
    return args


def main() -> None:
    args = parse_args()
    dataset = DATASETS[args.dataset]
    thread_env = {name: os.environ.get(name) for name in THREAD_ENV_VARS}
    threads_pinned = all(value == "1" for value in thread_env.values())
    if not threads_pinned:
        print(f"warning: thread pools are not all pinned to 1: {thread_env}", file=sys.stderr)
    rows_label = "all" if args.rows is None else f"n{args.rows}"
    mode = "discrete" if args.discrete else "exact"
    case_id = f"{dataset.name}_{args.step}_{rows_label}_{mode}_seed{args.seed}"

    load_start = time.perf_counter()
    frame, row_counts = read_rows(dataset, args.step, args.rows, args.seed)
    train, holdout, accounting = prepare(frame, dataset, args.step)
    del frame
    data_load_s = time.perf_counter() - load_start

    model, terms = build_model(dataset, args.step, args.discrete)
    X = train[dataset.model_columns(args.step)]
    y = train[dataset.response].to_numpy(np.float64)
    offset = log_exposure(train, dataset)
    warmup_start = time.perf_counter()
    superglm.warmup()
    warmup_s = time.perf_counter() - warmup_start
    gc.collect()

    signal.signal(signal.SIGTERM, stop_fit)
    fit, memory = timed_fit(model, X, y, offset)
    result = (
        fitted_report(model, dataset, args.step, train, holdout) if fit["status"] == "ok" else None
    )
    payload = {
        "schema_version": 1,
        "case_id": case_id,
        "dataset": dataset.name,
        "step": args.step,
        "seed": args.seed,
        "discrete": args.discrete,
        "rows": row_counts | accounting,
        "model": {
            "family": dataset.family,
            "response": dataset.response,
            "offset": None if dataset.exposure is None else f"log({dataset.exposure})",
            "terms": terms,
            "fit_reml_options": FIT_REML_OPTIONS,
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
            "pyarrow": pa.__version__,
            "superglm": superglm.__version__,
            "data_path": str(dataset.path.relative_to(ROOT)),
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
