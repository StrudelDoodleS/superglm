"""Standalone boosters on the frozen PSST prediction datasets."""

from __future__ import annotations

# Limit threads before native-library imports in spawned workers.
# ruff: noqa: E402
import argparse
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import resource
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
import pandas as pd
import psst_detection_study as study

BACKENDS = ("xgboost", "catboost", "lightgbm")
DEPTHS = (2, 4, 6)
PACKAGES = {"xgboost": "xgboost-cpu", "catboost": "catboost", "lightgbm": "lightgbm"}


def category_frames(train, *others):
    frames = [frame.copy() for frame in (train, *others)]
    for column in train:
        if pd.api.types.is_string_dtype(train[column].dtype) or isinstance(
            train[column].dtype, pd.CategoricalDtype
        ):
            levels = sorted(train[column].dropna().unique().tolist())
            for frame in frames:
                values = frame[column].where(frame[column].isin(levels))
                frame[column] = pd.Categorical(values, categories=levels)
    return frames


@dataclass
class Fitted:
    backend: str
    family: str
    model: object
    best_rounds: int
    trained_rounds: int
    parameters: dict

    def predict(self, frame):
        if self.backend == "xgboost":
            import xgboost as xgb

            raw = self.model.predict(
                xgb.DMatrix(frame, enable_categorical=True, nthread=1),
                output_margin=True,
                iteration_range=(0, self.best_rounds),
            )
        elif self.backend == "lightgbm":
            raw = self.model.predict(
                frame, raw_score=True, num_iteration=self.best_rounds, num_threads=1
            )
        else:
            raw = self.model.predict(
                frame, prediction_type="RawFormulaVal", ntree_end=self.best_rounds, thread_count=1
            )
        raw = np.asarray(raw, dtype=float)
        return np.exp(raw) if self.family == "poisson" else raw


def fit_candidate(backend, family, train, y, valid, vy, *, depth, seed, rounds=1000):
    if family not in ("gaussian", "poisson"):
        raise ValueError(f"Unsupported family {family}")
    poisson = family == "poisson"
    if backend == "xgboost":
        import xgboost as xgb

        parameters = {
            "objective": "count:poisson" if poisson else "reg:squarederror",
            "eval_metric": "poisson-nloglik" if poisson else "rmse",
            "tree_method": "hist",
            "device": "cpu",
            "max_depth": depth,
            "learning_rate": 0.05,
            "nthread": 1,
            "seed": seed,
        }
        history = {}
        model = xgb.train(
            parameters,
            xgb.DMatrix(train, label=y, enable_categorical=True, nthread=1),
            num_boost_round=rounds,
            evals=[(xgb.DMatrix(valid, label=vy, enable_categorical=True, nthread=1), "valid")],
            evals_result=history,
            early_stopping_rounds=40,
            verbose_eval=False,
        )
        best, trained = model.best_iteration + 1, model.num_boosted_rounds()
    elif backend == "lightgbm":
        import lightgbm as lgb

        parameters = {
            "objective": "poisson" if poisson else "regression",
            "metric": "poisson" if poisson else "l2",
            "max_depth": depth,
            "num_leaves": 2**depth,
            "learning_rate": 0.05,
            "num_threads": 1,
            "deterministic": True,
            "force_col_wise": True,
            "verbosity": -1,
            "seed": seed,
        }
        training = lgb.Dataset(train, label=y)
        history = {}
        model = lgb.train(
            parameters,
            training,
            num_boost_round=rounds,
            valid_sets=[lgb.Dataset(valid, label=vy, reference=training)],
            valid_names=["valid"],
            callbacks=[lgb.early_stopping(40, verbose=False), lgb.record_evaluation(history)],
        )
        best = model.best_iteration or model.current_iteration()
        trained = len(next(iter(history["valid"].values())))
    elif backend == "catboost":
        from catboost import CatBoostRegressor

        parameters = {
            "loss_function": "Poisson" if poisson else "RMSE",
            "eval_metric": "Poisson" if poisson else "RMSE",
            "depth": depth,
            "learning_rate": 0.05,
            "iterations": rounds,
            "thread_count": 1,
            "task_type": "CPU",
            "random_seed": seed,
            "allow_writing_files": False,
            "verbose": False,
        }
        model = CatBoostRegressor(**parameters)
        model.fit(
            train,
            y,
            cat_features=[c for c in train if isinstance(train[c].dtype, pd.CategoricalDtype)],
            eval_set=(valid, vy),
            early_stopping_rounds=40,
            use_best_model=True,
        )
        best = model.get_best_iteration() + 1
        trained = len(next(iter(model.get_evals_result()["validation"].values())))
    else:
        raise ValueError(f"Unknown backend {backend}")
    if not 1 <= best <= trained <= rounds:
        raise RuntimeError(f"Invalid iteration selection: best={best}, trained={trained}")
    return Fitted(backend, family, model, best, trained, parameters)


def select_candidate(candidates):
    usable = [name for name, value in candidates.items() if "validation_loss" in value]
    return min(usable, key=lambda name: candidates[name]["validation_loss"])


def run_trial(task):
    backend, case, strength, replicate, phase = task
    result = {
        "id": f"{phase}/{case}/{strength:g}/{replicate}",
        "backend": backend,
        "case": case,
        "design": study.design_key(case),
        "strength": strength,
        "replicate": replicate,
        "phase": phase,
        "status": "failed",
        "pid": os.getpid(),
    }
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            # Imports are included in process RSS, but excluded from fitting time.
            __import__(backend)
            family = study.CASES[case][0]
            train, y, _, _ = study.sample(case, strength, replicate, phase, 0, 4000)
            valid, vy, _, _ = study.sample(case, strength, replicate, phase, 1, 2000)
            tick = time.perf_counter()
            train, valid = category_frames(train, valid)
            result["preprocessing_seconds"] = time.perf_counter() - tick
            intercept = float(np.mean(y))
            candidates = {
                "constant": {
                    "validation_loss": study.loss(vy, np.full(len(vy), intercept), family),
                    "best_rounds": 0,
                    "trained_rounds": 0,
                }
            }
            fitted = {}
            tick = time.perf_counter()
            for depth in DEPTHS:
                name = f"depth{depth}"
                begin = time.perf_counter()
                try:
                    candidate = fit_candidate(
                        backend,
                        family,
                        train,
                        y,
                        valid,
                        vy,
                        depth=depth,
                        seed=20260913 + replicate,
                    )
                    candidates[name] = {
                        "validation_loss": study.loss(vy, candidate.predict(valid), family),
                        "best_rounds": candidate.best_rounds,
                        "trained_rounds": candidate.trained_rounds,
                        "parameters": candidate.parameters,
                    }
                    fitted[name] = candidate
                except Exception as error:
                    candidates[name] = {"error": f"{type(error).__name__}: {error}"}
                candidates[name]["seconds"] = time.perf_counter() - begin
            result["tuning_seconds"] = time.perf_counter() - tick
            result["candidates"] = candidates
            result["candidate_failures"] = sum("error" in item for item in candidates.values())
            selected = select_candidate(candidates)
            result["selected"] = selected
            # The test sample is generated only after all selection decisions.
            test, ty, mean, _ = study.sample(case, strength, replicate, phase, 2, 8000)
            _, test = category_frames(train, test)
            tick = time.perf_counter()
            prediction = (
                np.full(len(test), intercept)
                if selected == "constant"
                else fitted[selected].predict(test)
            )
            result["test_predict_seconds"] = time.perf_counter() - tick
            result["test_loss"] = study.loss(ty, prediction, family)
            result["test_risk"] = study.prediction_risk(mean, prediction, family)
            result["status"] = "ok"
        except Exception:
            result["error"] = traceback.format_exc()
        result["warnings"] = sorted({str(w.message) for w in caught})
    result["elapsed_seconds"] = time.perf_counter() - started
    result["process_peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "study"), default="pilot")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--output", type=Path, default=Path(".benchmark-artifacts/psst-detection-study/boosters")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = [
        (backend, case, strength, replicate, "pilot" if args.phase == "pilot" else "signal")
        for backend in BACKENDS
        for case in study.CASES
        for strength in ((0.08,) if args.phase == "pilot" else study.STRENGTHS)
        for replicate in range(1 if args.phase == "pilot" else 20)
    ]
    root = Path(__file__).parent
    manifest = {
        "phase": args.phase,
        "tasks": len(tasks),
        "workers": args.workers,
        "packages": {
            backend: importlib.metadata.version(package) for backend, package in PACKAGES.items()
        },
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "source_hashes": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in (
                "psst_booster_study.py",
                "psst_booster_protocol.md",
                "psst_detection_study.py",
            )
        },
        "package_source_sha256": study.source_hash(),
        "threads": {
            name: os.environ[name]
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS")
        },
        "fresh_process_per_task": True,
    }
    manifest_path = args.output / f"{args.phase}-manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError("Manifest differs; use a fresh output directory")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    output = args.output / f"{args.phase}.jsonl"
    done = (
        {
            (r["backend"], r["id"])
            for r in (json.loads(line) for line in output.read_text().splitlines())
        }
        if output.exists()
        else set()
    )
    tasks = [
        task
        for task in tasks
        if (task[0], f"{task[4]}/{task[1]}/{task[2]:g}/{task[3]}") not in done
    ]
    begin = time.perf_counter()
    failures = 0
    candidate_failures = 0
    with (
        output.open("a") as stream,
        ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=multiprocessing.get_context("spawn"),
            max_tasks_per_child=1,
        ) as pool,
    ):
        pending = [pool.submit(run_trial, task) for task in tasks]
        for count, future in enumerate(as_completed(pending), 1):
            row = future.result()
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            failures += row["status"] != "ok"
            candidate_failures += row.get("candidate_failures", 0)
            if count % 10 == 0 or count == len(tasks) or row["status"] != "ok":
                print(
                    f"{count}/{len(tasks)} finished; failures={failures}; candidate_failures={candidate_failures}; elapsed={time.perf_counter() - begin:.1f}s; {row['backend']}/{row['id']}",
                    flush=True,
                )
                if row["status"] != "ok":
                    print(row["error"], flush=True)


if __name__ == "__main__":
    main()
