"""Deep profiling harness for SuperBooster interaction benchmark cases."""

from __future__ import annotations

import argparse
import cProfile
import importlib.util
import logging
import time
import tracemalloc
import warnings
from collections.abc import Callable
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from superglm.profiling.harness import (
    SystemSampler,
    dump_json,
    summarize_system_samples,
    write_system_samples_csv,
    write_tracemalloc_report,
)
from superglm.profiling.harness import (
    write_cprofile_stats as write_pstats_summary,
)

ROOT = Path(__file__).resolve().parents[1]
BENCH_PATH = ROOT / "benchmarks" / "superbooster_interaction_challenger.py"
RESULTS_ROOT = ROOT / "benchmarks" / "results" / "profile_runs"


def load_benchmark_module():
    """Import the benchmark module by file path."""
    spec = importlib.util.spec_from_file_location("superbooster_interaction_challenger", BENCH_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load benchmark module from {BENCH_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def phase_record(
    phases: list[dict[str, Any]],
    name: str,
    fn: Callable[..., Any],
    *args,
    meta: Callable[[Any], dict[str, Any]] | None = None,
    **kwargs,
):
    """Time one phase and append a structured record."""
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    record = {"phase": name, "elapsed_s": elapsed}
    if meta is not None:
        record["meta"] = meta(out)
    phases.append(record)
    return out


def fit_hybrid_bundle(bench, train: dict[str, Any], valid: dict[str, Any], test: dict[str, Any]):
    """Fit the backbone + hybrid booster prerequisites once."""
    phases: list[dict[str, Any]] = []
    main_model = phase_record(
        phases,
        "fit_main_backbone",
        bench.fit_superglm,
        train["X"],
        train["y_count"],
        train["offset"],
        meta=lambda m: {
            "effective_df": float(m.result.effective_df),
            "reml_profile": dict(getattr(m, "_reml_profile", {})),
        },
    )
    area_levels = sorted(pd.Series(train["X"]["Area"]).astype(str).unique().tolist())
    Xb_train = phase_record(
        phases, "build_booster_train", bench.build_booster_frame, train["X"], area_levels
    )
    Xb_valid = phase_record(
        phases, "build_booster_valid", bench.build_booster_frame, valid["X"], area_levels
    )
    Xb_test = phase_record(
        phases, "build_booster_test", bench.build_booster_frame, test["X"], area_levels
    )
    eta_train = phase_record(
        phases, "predict_eta_train", main_model._predict_eta_exact, train["X"], train["offset"]
    )
    eta_valid = phase_record(
        phases, "predict_eta_valid", main_model._predict_eta_exact, valid["X"], valid["offset"]
    )
    eta_test = phase_record(
        phases, "predict_eta_test", main_model._predict_eta_exact, test["X"], test["offset"]
    )
    hybrid_booster = phase_record(
        phases,
        "fit_hybrid_booster",
        bench.fit_xgb,
        Xb_train,
        train["y_count"],
        eta_train,
        Xb_valid,
        valid["y_count"],
        eta_valid,
        meta=lambda booster: {
            "best_iteration": int(booster.best_iteration + 1),
            "best_score": float(booster.best_score),
        },
    )
    interaction_ranking = phase_record(
        phases,
        "rank_parent_interactions",
        bench.rank_parent_interactions,
        hybrid_booster,
        Xb_valid,
        sample_rows=int(getattr(bench, "INTERACTION_SAMPLE_ROWS", 30_000)),
    )
    return {
        "phases": phases,
        "main_model": main_model,
        "hybrid_booster": hybrid_booster,
        "interaction_ranking": interaction_ranking,
        "Xb_train": Xb_train,
        "Xb_valid": Xb_valid,
        "Xb_test": Xb_test,
        "eta_train": eta_train,
        "eta_valid": eta_valid,
        "eta_test": eta_test,
        "area_levels": area_levels,
    }


def execute_case(case: str, *, top_k: int, interaction_sample_rows: int) -> dict[str, Any]:
    """Run one profiled benchmark case and return structured metadata."""
    bench = load_benchmark_module()
    if interaction_sample_rows != getattr(
        bench, "INTERACTION_SAMPLE_ROWS", interaction_sample_rows
    ):
        bench.INTERACTION_SAMPLE_ROWS = interaction_sample_rows

    phases: list[dict[str, Any]] = []
    X, y_count, exposure, offset = phase_record(phases, "load_freq", bench.load_freq)
    split = phase_record(
        phases, "split_data", bench.split_data, X, y_count, exposure, offset, seed=42
    )
    train = split["train"]
    valid = split["valid"]
    test = split["test"]

    result: dict[str, Any] = {
        "case": case,
        "split": {
            "n_train": int(len(train["X"])),
            "n_valid": int(len(valid["X"])),
            "n_test": int(len(test["X"])),
        },
    }

    if case == "main":
        model = phase_record(
            phases,
            "fit_main_backbone",
            bench.fit_superglm,
            train["X"],
            train["y_count"],
            train["offset"],
            meta=lambda m: {
                "effective_df": float(m.result.effective_df),
                "reml_profile": dict(getattr(m, "_reml_profile", {})),
            },
        )
        mu = phase_record(phases, "predict_main_test", model.predict, test["X"], test["offset"])
        result["metrics"] = bench.evaluate_count_model(
            "superglm_main", mu, test["y_count"], test["exposure"]
        )

    elif case == "seed":
        model = phase_record(
            phases,
            "fit_seed_interaction",
            bench.fit_superglm,
            train["X"],
            train["y_count"],
            train["offset"],
            interactions=[("DrivAge", "VehAge")],
            max_reml_iter=15,
            meta=lambda m: {
                "effective_df": float(m.result.effective_df),
                "reml_profile": dict(getattr(m, "_reml_profile", {})),
            },
        )
        mu = phase_record(phases, "predict_seed_test", model.predict, test["X"], test["offset"])
        result["metrics"] = bench.evaluate_count_model(
            "superglm_seed_interaction",
            mu,
            test["y_count"],
            test["exposure"],
        )

    elif case in {"hybrid", "top1", "top3"}:
        bundle = fit_hybrid_bundle(bench, train, valid, test)
        phases.extend(bundle["phases"])
        if case == "hybrid":
            eta_hybrid = phase_record(
                phases,
                "predict_hybrid_margin_test",
                bundle["hybrid_booster"].predict,
                xgb.DMatrix(
                    bundle["Xb_test"].to_numpy(dtype=np.float32),
                    base_margin=bundle["eta_test"].astype(np.float32),
                ),
                output_margin=True,
            )
            result["metrics"] = bench.evaluate_count_model(
                "hybrid",
                np.exp(eta_hybrid),
                test["y_count"],
                test["exposure"],
            )
            result["interaction_ranking"] = bundle["interaction_ranking"]
        else:
            k = 1 if case == "top1" else top_k
            chosen = [(row["left"], row["right"]) for row in bundle["interaction_ranking"][:k]]
            challenger = phase_record(
                phases,
                f"fit_top{k}_challenger",
                bench.fit_superglm,
                train["X"],
                train["y_count"],
                train["offset"],
                interactions=chosen,
                max_reml_iter=15,
                meta=lambda m: {
                    "effective_df": float(m.result.effective_df),
                    "reml_profile": dict(getattr(m, "_reml_profile", {})),
                },
            )
            mu = phase_record(
                phases,
                f"predict_top{k}_test",
                challenger.predict,
                test["X"],
                test["offset"],
            )
            result["metrics"] = bench.evaluate_count_model(
                f"superglm_top{k}_gbm_interactions",
                mu,
                test["y_count"],
                test["exposure"],
            )
            result["interaction_ranking"] = bundle["interaction_ranking"]
            result["chosen_pairs"] = [f"{a}:{b}" for a, b in chosen]

    elif case == "pure_xgb":
        area_levels = sorted(pd.Series(train["X"]["Area"]).astype(str).unique().tolist())
        Xb_train = phase_record(
            phases, "build_booster_train", bench.build_booster_frame, train["X"], area_levels
        )
        Xb_valid = phase_record(
            phases, "build_booster_valid", bench.build_booster_frame, valid["X"], area_levels
        )
        Xb_test = phase_record(
            phases, "build_booster_test", bench.build_booster_frame, test["X"], area_levels
        )
        pure_booster = phase_record(
            phases,
            "fit_pure_xgb",
            bench.fit_xgb,
            Xb_train,
            train["y_count"],
            train["offset"],
            Xb_valid,
            valid["y_count"],
            valid["offset"],
            meta=lambda booster: {
                "best_iteration": int(booster.best_iteration + 1),
                "best_score": float(booster.best_score),
            },
        )
        eta = phase_record(
            phases,
            "predict_pure_xgb_margin_test",
            pure_booster.predict,
            xgb.DMatrix(
                Xb_test.to_numpy(dtype=np.float32),
                base_margin=test["offset"].astype(np.float32),
            ),
            output_margin=True,
        )
        result["metrics"] = bench.evaluate_count_model(
            "xgboost_pure",
            np.exp(eta),
            test["y_count"],
            test["exposure"],
        )

    else:
        raise ValueError(f"Unknown case {case!r}")

    result["phases"] = phases
    return result


def build_line_profiler():
    """Construct an optional line-profiler over likely hot Python functions."""
    from line_profiler import LineProfiler

    from superglm.model import reml_execute
    from superglm.solvers import irls_direct

    bench = load_benchmark_module()
    profiler = LineProfiler()
    for fn in (
        bench.fit_superglm,
        bench.fit_xgb,
        bench.rank_parent_interactions,
        irls_direct.fit_irls_direct,
        irls_direct._safe_decompose_H,
        reml_execute.optimize_reml_best,
    ):
        profiler.add_function(fn)
    return profiler


def configure_logger(log_path: Path) -> logging.Handler:
    """Attach a file handler to the root logger for this run."""
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    if root.level > logging.INFO:
        root.setLevel(logging.INFO)
    return handler


def parse_args() -> argparse.Namespace:
    """CLI arguments for the profiling harness."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=("main", "seed", "hybrid", "top1", "top3", "pure_xgb"),
        default="top3",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--interaction-sample-rows", type=int, default=30_000)
    parser.add_argument("--sample-interval-ms", type=int, default=500)
    parser.add_argument("--with-line-profiler", action="store_true")
    parser.add_argument("--with-memray", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (RESULTS_ROOT / f"{timestamp}_{args.case}")
    output_dir.mkdir(parents=True, exist_ok=True)

    log_handler = configure_logger(output_dir / "run.log")
    sampler = SystemSampler(interval_s=args.sample_interval_ms / 1000.0)

    line_profiler = None
    line_profile_path = output_dir / "line_profile.txt"
    if args.with_line_profiler:
        line_profiler = build_line_profiler()

    memray_tracker = nullcontext()
    memray_path = output_dir / "memray.bin"
    memray_enabled = False
    if args.with_memray:
        import memray

        memray_tracker = memray.Tracker(str(memray_path))
        memray_enabled = True

    cprof = cProfile.Profile()
    tracemalloc.start(25)
    start_snapshot = tracemalloc.take_snapshot()
    warning_records: list[dict[str, Any]] = []
    start_wall = time.perf_counter()
    sampler.start()

    result: dict[str, Any] | None = None
    error: str | None = None
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            runner: Callable[..., Any] = execute_case
            if line_profiler is not None:
                runner = line_profiler(runner)
            with memray_tracker:
                cprof.enable()
                result = runner(
                    args.case,
                    top_k=args.top_k,
                    interaction_sample_rows=args.interaction_sample_rows,
                )
                cprof.disable()
            for w in caught:
                warning_records.append(
                    {
                        "message": str(w.message),
                        "category": getattr(w.category, "__name__", str(w.category)),
                        "filename": w.filename,
                        "lineno": int(w.lineno),
                    }
                )
    except Exception as exc:  # pragma: no cover - integration path
        error = repr(exc)
        raise
    finally:
        sampler.stop()
        peak_current, peak_bytes = tracemalloc.get_traced_memory()
        end_snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()
        total_wall = time.perf_counter() - start_wall

        write_system_samples_csv(sampler.samples, output_dir / "system_timeseries.csv")
        write_tracemalloc_report(
            output_dir / "tracemalloc_top.txt",
            start_snapshot=start_snapshot,
            end_snapshot=end_snapshot,
            peak_bytes=peak_bytes,
        )
        cprof.dump_stats(str(output_dir / "cprofile.pstats"))
        write_pstats_summary(cprof, output_dir / "cprofile_top.txt")
        if line_profiler is not None:
            with line_profile_path.open("w") as f:
                line_profiler.print_stats(stream=f)
        if result is not None:
            dump_json(output_dir / "result.json", result)
            dump_json(output_dir / "phases.json", {"phases": result.get("phases", [])})
        dump_json(output_dir / "warnings.json", {"warnings": warning_records})

        summary = {
            "case": args.case,
            "top_k": args.top_k,
            "interaction_sample_rows": args.interaction_sample_rows,
            "sample_interval_ms": args.sample_interval_ms,
            "wall_time_s": total_wall,
            "system_summary": summarize_system_samples(sampler.samples),
            "sampler_error": sampler.error,
            "tracemalloc_peak_bytes": peak_bytes,
            "tracemalloc_end_current_bytes": peak_current,
            "memray_enabled": memray_enabled,
            "artifacts": {
                "system_timeseries_csv": str(output_dir / "system_timeseries.csv"),
                "cprofile_pstats": str(output_dir / "cprofile.pstats"),
                "cprofile_top_txt": str(output_dir / "cprofile_top.txt"),
                "tracemalloc_top_txt": str(output_dir / "tracemalloc_top.txt"),
                "line_profile_txt": str(line_profile_path) if line_profiler is not None else None,
                "memray_bin": str(memray_path) if memray_enabled else None,
                "warnings_json": str(output_dir / "warnings.json"),
                "phases_json": str(output_dir / "phases.json") if result is not None else None,
                "result_json": str(output_dir / "result.json") if result is not None else None,
                "run_log": str(output_dir / "run.log"),
            },
            "error": error,
        }
        dump_json(output_dir / "summary.json", summary)

        root = logging.getLogger()
        root.removeHandler(log_handler)
        log_handler.close()

    print(f"Saved profiling artifacts to {output_dir}")


if __name__ == "__main__":
    main()
