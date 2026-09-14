"""Bounded real-data additive and small-tensor trials.

This benchmark owns its train-only adapter. It is research code, with no
production feature-selection or numerical-certification claim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import pickle
import platform
import resource
import subprocess
import sys
import time
import traceback
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from benchmark_housing_tensor import retained_model_storage, run_isolated, source_fingerprint

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260913
SPLITS = ("train", "valid", "test")
DEFAULT_DATASETS = ("uci_breast_cancer", "uci_credit_default", "ames_housing", "uci_bike_sharing")


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_worker_receipt(path, stage):
    """Preserve interrupted writes before the parent records the failed attempt."""
    if not path.exists():
        return {"status": "error", "error": "Worker produced no receipt"}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        incomplete = path.with_name(f"{stage}_incomplete_{digest}.bin")
        path.rename(incomplete)
        return {
            "status": "error",
            "error": f"Worker produced an incomplete JSON receipt: {error}",
            "incomplete_receipt": {
                "path": incomplete.name,
                "sha256": digest,
                "bytes": incomplete.stat().st_size,
            },
        }


def category_labels(series):
    # Prefix real labels so user strings cannot collide with missing/pool tokens.
    return series.astype(object).map(lambda value: "missing:" if pd.isna(value) else f"v:{value}")


def numeric_values(series):
    values = pd.to_numeric(series, errors="raise").to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(values).any():
        raise ValueError(f"Nonfinite numeric values in column {series.name}")
    return values


def fit_preprocessor(train, *, categorical_columns, max_categories=32):
    """Learn every adaptive preprocessing choice from actual training rows."""
    if not train.columns.is_unique or not len(train):
        raise ValueError("Training columns must be unique and rows nonempty")
    if set(categorical_columns) - set(train.columns) or max_categories < 2:
        raise ValueError("Invalid categorical columns or category budget")
    state = {"input_columns": list(train), "features": {}, "dropped": {}}
    for name in train:
        series = train[name]
        if name in categorical_columns or not pd.api.types.is_numeric_dtype(series):
            labels = category_labels(series)
            counts = labels.value_counts()
            ordered = sorted(counts.index, key=lambda label: (-int(counts[label]), label))
            minimum_count = max(2, math.ceil(0.01 * len(train)))
            keep = [label for label in ordered if counts[label] >= minimum_count]
            # Reserve a slot for the pooled training levels when capping.
            if len(keep) > max_categories or len(keep) < len(ordered):
                keep = keep[: max_categories - 1]
            pooled = len(keep) < len(ordered)
            fallback = "pooled:" if pooled else ordered[0]
            levels = sorted([*keep, *(["pooled:"] if pooled else [])])
            if len(levels) == 1:
                state["dropped"][name] = "constant after training category pooling"
                continue
            state["features"][name] = {
                "kind": "categorical",
                "keep": keep,
                "fallback": fallback,
                "levels": levels,
                "training_raw_levels": len(ordered),
                "training_missing": int(series.isna().sum()),
                "minimum_count": minimum_count,
            }
        else:
            values = numeric_values(series)
            observed = values[~np.isnan(values)]
            if not len(observed) or np.unique(observed).size == 1:
                state["dropped"][name] = "all missing or constant in training"
                continue
            median = float(np.median(observed))
            values = np.where(np.isnan(values), median, values)
            center, scale = float(np.mean(values)), float(np.std(values))
            if not math.isfinite(center) or not math.isfinite(scale) or scale <= 0:
                raise ValueError(f"Nonfinite or degenerate training scale in column {name}")
            state["features"][name] = {
                "kind": "spline" if np.unique(observed).size >= 10 else "numeric",
                "median": median,
                "center": center,
                "scale": scale,
                "training_unique": int(np.unique(observed).size),
                "training_missing": int(series.isna().sum()),
            }
    if not state["features"]:
        raise ValueError("No nonconstant training features remain")
    return state


def transform_features(frame, state):
    """Apply frozen training state without inspecting held-out distributions."""
    if list(frame) != state["input_columns"] or not frame.columns.is_unique:
        raise ValueError("Input column contract differs from training")
    result = pd.DataFrame(index=frame.index)
    for name, feature in state["features"].items():
        if feature["kind"] == "categorical":
            labels = category_labels(frame[name])
            result[name] = labels.where(labels.isin(feature["keep"]), feature["fallback"])
        else:
            values = numeric_values(frame[name])
            values = np.where(np.isnan(values), feature["median"], values)
            values = (values - feature["center"]) / feature["scale"]
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite transformed numeric column {name}")
            result[name] = values
    return result.reset_index(drop=True)


def partition_rows(frame, *, strategy, columns, target=None, years=None, seed=SEED):
    """Return a disjoint complete partition; group boundaries are never cut."""
    from sklearn.model_selection import train_test_split

    if frame.loc[:, columns].isna().any().any():
        raise ValueError("Missing split/group keys are not allowed")
    if strategy == "fixed_year_group":
        year, *groups = columns
        partitions = {
            name: np.flatnonzero(frame[year].isin(years[name]).to_numpy()) for name in SPLITS
        }
    else:
        groups = columns
        keys = pd.MultiIndex.from_frame(frame.loc[:, columns])
        codes, unique = pd.factorize(keys, sort=True)
        unit_ids = np.arange(len(unique))
        if strategy == "chronological_group":
            units = np.split(unit_ids, [int(0.6 * len(unit_ids)), int(0.8 * len(unit_ids))])
        elif strategy in {"stratified_group", "random_group"}:
            labels = None
            if strategy == "stratified_group":
                grouped = frame.assign(_split_group=codes).groupby("_split_group")[target]
                if (grouped.nunique(dropna=False) != 1).any():
                    raise ValueError("A stratified group has inconsistent target classes")
                labels = grouped.first().reindex(unit_ids).to_numpy()
            train_units, holdout = train_test_split(
                unit_ids, test_size=0.4, random_state=seed, stratify=labels
            )
            valid_units, test_units = train_test_split(
                holdout,
                test_size=0.5,
                random_state=seed + 1,
                stratify=None if labels is None else labels[holdout],
            )
            units = [train_units, valid_units, test_units]
        else:
            raise ValueError(f"Unsupported split strategy {strategy!r}")
        partitions = {
            name: np.flatnonzero(np.isin(codes, unit))
            for name, unit in zip(SPLITS, units, strict=True)
        }
    joined = np.concatenate(list(partitions.values()))
    if not np.array_equal(np.sort(joined), np.arange(len(frame))):
        raise ValueError("Split must preserve every row exactly once")
    if any(not len(rows) for rows in partitions.values()):
        raise ValueError("Every split must contain rows")
    if groups:
        owners = {}
        for name, rows in partitions.items():
            for key in frame.iloc[rows].loc[:, groups].itertuples(index=False, name=None):
                if owners.setdefault(key, name) != name:
                    raise ValueError("A group appears in more than one partition")
    return partitions


def screen_interactions(frame, response, residual, *, spline_columns, max_features=12, max_pairs=8):
    """Rank a bounded set of centered bilinear products using training only."""
    centered_y = np.asarray(response, dtype=float) - np.mean(response)
    centered_residual = np.asarray(residual, dtype=float) - np.mean(residual)

    def correlation(left, right):
        denominator = np.linalg.norm(left) * np.linalg.norm(right)
        return 0.0 if denominator == 0 else float(abs(np.dot(left, right)) / denominator)

    values = {
        name: frame[name].to_numpy(dtype=float) - frame[name].mean() for name in spline_columns
    }
    ranked = sorted(values, key=lambda name: (-correlation(values[name], centered_y), name))
    candidates = ranked[:max_features]
    scores = []
    for left, right in itertools.combinations(candidates, 2):
        product = values[left] * values[right]
        product -= product.mean()
        scores.append(
            {"pair": sorted([left, right]), "score": correlation(product, centered_residual)}
        )
    scores.sort(key=lambda record: (-record["score"], record["pair"]))
    return {
        "policy": "training marginal correlation shortlist, centered residual product correlation",
        "shortlisted_features": candidates,
        "candidate_pair_count": len(scores),
        "scores": scores,
        "pairs": [record["pair"] for record in scores[:max_pairs]],
    }


def choose_validation_arm(records):
    candidates = [
        (record["validation"]["primary_loss"], name)
        for name, record in records.items()
        if record.get("status") == "converged"
        and math.isfinite(record.get("validation", {}).get("primary_loss", math.inf))
    ]
    return min(candidates)[1] if candidates else None


def prepare_dataset(frame, entry):
    """Translate a supported manifest split and freeze the training adapter."""
    split = entry["split"]
    strategies = {
        "time": "chronological_group",
        "time_group": "fixed_year_group",
        "group": "random_group",
    }
    strategy = strategies.get(split["strategy"], split["strategy"])
    years = None
    if strategy == "fixed_year_group":
        years = {
            name: split[key]
            for name, key in zip(
                SPLITS, ("train_years", "validation_years", "test_years"), strict=True
            )
        }
    rows = partition_rows(
        frame,
        strategy=strategy,
        columns=split["columns"],
        target=entry["primary_target"],
        years=years,
        seed=split["seed"],
    )
    feature_names = entry["features"]
    if entry["primary_target"] in feature_names or set(entry.get("exclude_columns", {})) & set(
        feature_names
    ):
        raise ValueError("A target or excluded column appears in the predictors")
    state = fit_preprocessor(
        frame.iloc[rows["train"]].loc[:, feature_names],
        categorical_columns=entry.get("categorical_columns", []),
    )
    split_digest = hashlib.sha256()
    for name in SPLITS:
        split_digest.update(name.encode())
        split_digest.update(rows[name].astype("<i8").tobytes())
    return state, rows, split_digest.hexdigest()


def family_for(entry):
    if "positive_class" in entry and entry["positive_class"] is not None:
        return "binomial"
    if entry["id"] == "uci_bike_sharing":
        return "poisson"
    if entry["id"] in {"ames_housing", "uci_sgemm"}:
        return "gaussian"
    raise ValueError(f"No response-family adapter declared for {entry['id']}")


def response_values(frame, entry):
    target = frame[entry["primary_target"]]
    if family_for(entry) == "binomial":
        return (target == entry["positive_class"]).to_numpy(dtype=float)
    values = target.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite response")
    return values


def score_predictions(response, prediction, family):
    from sklearn.metrics import (
        average_precision_score,
        log_loss,
        mean_poisson_deviance,
        roc_auc_score,
    )

    if prediction.shape != response.shape or not np.isfinite(prediction).all():
        raise ValueError("Prediction shape or finite-value contract failed")
    if family == "binomial":
        if np.any((prediction < 0) | (prediction > 1)):
            raise ValueError("Predicted probabilities are outside [0, 1]")
        loss = float(log_loss(response, prediction, labels=[0, 1]))
        return {
            "primary_loss": loss,
            "log_loss": loss,
            "average_precision": float(average_precision_score(response, prediction)),
            "prevalence": float(np.mean(response)),
            "roc_auc": float(roc_auc_score(response, prediction))
            if np.unique(response).size == 2
            else None,
            "rows": len(response),
        }
    mse = float(np.mean((response - prediction) ** 2))
    deviance = float(mean_poisson_deviance(response, prediction)) if family == "poisson" else mse
    return {"primary_loss": deviance, "mse": mse, "mean_deviance": deviance, "rows": len(response)}


def build_model(state, family, pairs):
    from superglm import Categorical, Numeric, Spline, SuperGLM

    features = {}
    for name, spec in state["features"].items():
        if spec["kind"] == "spline":
            features[name] = Spline(kind="cr", k=6, knot_strategy="uniform", penalty="ssp")
        elif spec["kind"] == "categorical":
            features[name] = Categorical(levels=spec["levels"])
        else:
            features[name] = Numeric()
    model = SuperGLM(
        family=family,
        features=features,
        selection_penalty=0.0,
        discrete=True,
        n_bins=64,
    )
    for left, right in pairs:
        model._add_interaction(left, right, n_knots=(2, 2))
    return model


def coefficient_estimate(state, pairs, n_train):
    additive = sum(
        5
        if feature["kind"] == "spline"
        else len(feature["levels"]) - 1
        if feature["kind"] == "categorical"
        else 1
        for feature in state["features"].values()
    )
    width = additive + 9 * len(pairs)
    q = sum(feature["kind"] == "spline" for feature in state["features"].values()) + 2 * len(pairs)
    return {
        "coefficient_count_without_intercept": width,
        "nominal_smoothing_components": q,
        "one_dense_design_mib": 8 * n_train * (width + 1) / 1024**2,
        "one_dense_square_mib": 8 * (width + 1) ** 2 / 1024**2,
        "scope": "Representation estimate only, not an RSS upper bound or dispatch claim",
    }


def fit_worker(args, record):
    from interaction_datasets import load_dataset, read_manifest
    from threadpoolctl import threadpool_info

    entry = next(entry for entry in read_manifest(args.manifest) if entry["id"] == args.dataset)
    if entry["schema"]["rows"] > args.max_rows:
        raise ValueError(f"Declared full table exceeds admitted row budget {args.max_rows}")
    started = time.perf_counter()
    frame = load_dataset(args.dataset, root=args.data_root, manifest=args.manifest)
    record["loading_seconds"] = time.perf_counter() - started
    if len(frame) > args.max_rows:
        raise ValueError(f"Full table exceeds admitted row budget {args.max_rows}")
    started = time.perf_counter()
    state, rows, split_hash = prepare_dataset(frame, entry)
    train = transform_features(frame.iloc[rows["train"]].loc[:, entry["features"]], state)
    valid = transform_features(frame.iloc[rows["valid"]].loc[:, entry["features"]], state)
    y = response_values(frame.iloc[rows["train"]], entry)
    valid_y = response_values(frame.iloc[rows["valid"]], entry)
    record["preprocessing_seconds"] = time.perf_counter() - started
    record.update(
        rows={name: len(indices) for name, indices in rows.items()},
        full_table_rows=len(frame),
        subsampling="none; every source row belongs to one split",
        raw_predictor_count=len(entry["features"]),
        family=family_for(entry),
        split=entry["split"],
        split_sha256=split_hash,
        preprocessing=state,
        preprocessing_sha256=digest_json(state),
        data_sha256=entry["source"]["sha256"],
        manifest_entry_sha256=digest_json(entry),
        runtime={
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in (
                    "superglm",
                    "numpy",
                    "scipy",
                    "pandas",
                    "numba",
                    "scikit-learn",
                    "threadpoolctl",
                )
            },
            "threadpools": threadpool_info(),
        },
    )
    pairs = []
    if args.arm == "interactions":
        baseline = json.loads((args.case_root / "additive" / "result.json").read_text())
        if baseline["status"] != "converged":
            raise ValueError("Interaction screen requires a converged additive fit")
        for key in (
            "package_source_sha256",
            "benchmark_script_sha256",
            "data_sha256",
            "manifest_entry_sha256",
            "split_sha256",
            "preprocessing_sha256",
        ):
            if record[key] != baseline[key]:
                raise ValueError(f"Additive screening identity differs at {key}")
        pairs = baseline["screening"]["pairs"]
        if not pairs:
            raise ValueError("No spline interaction candidates satisfy the declared policy")
    record["pairs"] = pairs
    estimate = coefficient_estimate(state, pairs, len(y))
    record["resource_estimate"] = estimate
    if estimate["coefficient_count_without_intercept"] > args.max_p:
        raise ValueError(f"Representation exceeds admitted coefficient budget {args.max_p}")
    started = time.perf_counter()
    model = build_model(state, record["family"], pairs)
    record["model_setup_seconds"] = time.perf_counter() - started
    record["status"] = "fitting"
    write_json(args.output / "result.json", record)
    started = time.perf_counter()
    try:
        model.fit_reml(train, y)
    finally:
        record["fit_seconds"] = time.perf_counter() - started
        divisor = 1024**2 if sys.platform == "darwin" else 1024
        record["fit_end_peak_process_rss_mib"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor
        )
    # Sample retained model payload before telemetry, prediction or serialization.
    record["retained_model_storage"] = retained_model_storage(model)
    telemetry = model.training_telemetry()
    record.update(
        telemetry=telemetry,
        coefficient_count_without_intercept=len(model.result.beta),
        fitted_smoothing_parameter_count=len(telemetry["reml"]["lambdas"]),
        resolved_direct_backend=model.result.direct_backend,
        backend_groups=[type(group).__name__ for group in model._dm.group_matrices],
    )
    converged = bool(model.result.converged) and bool(telemetry["reml"]["converged"])
    record["status"] = "converged" if converged else "not_converged"
    if not converged:
        return
    started = time.perf_counter()
    record["validation"] = score_predictions(valid_y, model.predict(valid), record["family"])
    record["validation_prediction_seconds"] = time.perf_counter() - started
    if args.arm == "additive":
        started = time.perf_counter()
        residual = y - model.predict(train)
        record["screening_prediction_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        record["screening"] = screen_interactions(
            train,
            y,
            residual,
            spline_columns=[
                name for name, spec in state["features"].items() if spec["kind"] == "spline"
            ],
            max_pairs=args.interactions,
        )
        record["screening_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    model_path = args.output / "model.pkl"
    with model_path.open("wb") as stream:
        pickle.dump(model, stream, protocol=pickle.HIGHEST_PROTOCOL)
    with model_path.open("rb") as stream:
        record["model_pickle_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    record["serialization_seconds"] = time.perf_counter() - started
    record["model_pickle_bytes"] = model_path.stat().st_size


def evaluation_worker(args, record):
    from interaction_datasets import load_dataset, read_manifest

    choice = json.loads((args.case_root / "choice.json").read_text())
    if choice["chosen_arm"] is None:
        raise ValueError("Test evaluation requires a persisted validation choice")
    result_path = args.output / "result.json"
    if (
        hashlib.sha256(result_path.read_bytes()).hexdigest()
        != choice["fit_result_sha256"][args.arm]
    ):
        raise ValueError("Fit result changed after validation choice")
    fitted = json.loads(result_path.read_text())
    for key in ("package_source_sha256", "benchmark_script_sha256"):
        if fitted[key] != record[key]:
            raise ValueError(f"Source changed after fit at {key}")
    entry = next(entry for entry in read_manifest(args.manifest) if entry["id"] == args.dataset)
    if digest_json(entry) != fitted["manifest_entry_sha256"]:
        raise ValueError("Manifest entry changed after fit")
    frame = load_dataset(args.dataset, root=args.data_root, manifest=args.manifest)
    state, rows, split_hash = prepare_dataset(frame, entry)
    if split_hash != fitted["split_sha256"] or digest_json(state) != fitted["preprocessing_sha256"]:
        raise ValueError("Split or preprocessing changed after fit")
    model_path = args.output / "model.pkl"
    with model_path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != fitted["model_pickle_sha256"]:
            raise ValueError("Owned model artifact differs from fitted receipt")
        stream.seek(0)
        model = pickle.load(stream)
    started = time.perf_counter()
    test = transform_features(frame.iloc[rows["test"]].loc[:, entry["features"]], state)
    test_y = response_values(frame.iloc[rows["test"]], entry)
    prediction = model.predict(test)
    record["test"] = score_predictions(test_y, prediction, fitted["family"])
    record["test_prediction_seconds"] = time.perf_counter() - started
    record["validation_chosen_arm"] = choice["chosen_arm"]
    record["status"] = "evaluated"
    np.savez_compressed(
        args.output / "test_predictions.npz",
        row_index=rows["test"],
        response=test_y,
        prediction=prediction,
    )


def worker(args):
    started = time.perf_counter()
    record = {
        "dataset": args.dataset,
        "arm": args.arm,
        "stage": args.stage,
        "status": "starting",
        "started_utc": datetime.now(UTC).isoformat(),
    }
    result_path = args.output / ("result.json" if args.stage == "fit" else "evaluation.json")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            record.update(
                package_source_sha256=source_fingerprint(),
                benchmark_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                measurement_helper_sha256=hashlib.sha256(
                    Path(__file__).with_name("benchmark_housing_tensor.py").read_bytes()
                ).hexdigest(),
                git_head=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
            )
            if args.stage == "fit":
                fit_worker(args, record)
            else:
                evaluation_worker(args, record)
        except Exception as error:
            record.update(
                status="error",
                error=f"{type(error).__name__}: {error}",
                traceback=traceback.format_exc(),
            )
        finally:
            record["warnings"] = [
                {"category": item.category.__name__, "message": str(item.message)}
                for item in caught
            ]
            record["worker_seconds"] = time.perf_counter() - started
            record["finished_utc"] = datetime.now(UTC).isoformat()
            write_json(result_path, record)
    print(
        json.dumps({key: record[key] for key in ("dataset", "arm", "stage", "status")}), flush=True
    )
    return 0 if record["status"] in {"converged", "evaluated"} else 1


def launch(args, dataset, arm, stage, timeout):
    case_root = args.output / dataset
    output = case_root / arm
    output.mkdir(parents=True, exist_ok=stage == "evaluate")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--dataset",
        dataset,
        "--arm",
        arm,
        "--stage",
        stage,
        "--output",
        str(output),
        "--case-root",
        str(case_root),
        "--manifest",
        str(args.manifest),
        "--data-root",
        str(args.data_root),
        "--interactions",
        str(args.interactions),
        "--max-p",
        str(args.max_p),
        "--max-rows",
        str(args.max_rows),
    ]
    env = os.environ.copy()
    for variable in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "MKL_NUM_THREADS",
    ):
        env[variable] = "1"
    receipt = run_isolated(command, log_path=output / f"{stage}.log", timeout=timeout, env=env)
    receipt.update(command=command, timeout_seconds=timeout)
    write_json(output / f"{stage}_process.json", receipt)
    result_path = output / ("result.json" if stage == "fit" else "evaluation.json")
    record = load_worker_receipt(result_path, stage)
    if receipt["status"] == "timeout":
        record["status"] = "timeout"
        record["warnings_complete"] = False
    elif receipt["status"] != "success" and record.get("status") != "not_converged":
        record["status"] = "error"
    elif not record:
        record = {"status": "error", "error": "Worker exited without a result receipt"}
    if "finished_utc" not in record:
        record.update(warnings_complete=False, parent_finished_utc=datetime.now(UTC).isoformat())
    write_json(result_path, record)
    record["process"] = receipt
    print(
        json.dumps(
            {
                "dataset": dataset,
                "arm": arm,
                "stage": stage,
                "status": record["status"],
                "process_seconds": receipt["process_seconds"],
            }
        ),
        flush=True,
    )
    return record


def run_suite(args):
    args.output.mkdir(parents=True, exist_ok=False)
    suite = {
        "schema_version": 1,
        "started_utc": datetime.now(UTC).isoformat(),
        "fit_worker_budget_seconds": args.total_fit_budget,
        "per_fit_worker_timeout_seconds": args.fit_timeout,
        "protocol": "Full rows; training-only preprocessing/screening/REML; validation selects one of two arms; test evaluation follows persisted choice",
        "datasets": {},
    }
    spent = 0.0
    for dataset in args.datasets:
        records = {}
        case_started = time.perf_counter()
        for arm in ("additive", "interactions"):
            if spent >= args.total_fit_budget:
                records[arm] = {"status": "budget_exhausted"}
            elif arm == "interactions" and records["additive"]["status"] != "converged":
                records[arm] = {
                    "status": "dependency_failed",
                    "reason": "Additive screening fit did not converge",
                }
            else:
                records[arm] = launch(
                    args, dataset, arm, "fit", min(args.fit_timeout, args.total_fit_budget - spent)
                )
                spent += records[arm]["process"]["process_seconds"]
        case_root = args.output / dataset
        case_root.mkdir(parents=True, exist_ok=True)
        choice = {
            "chosen_arm": choose_validation_arm(records),
            "chosen_before_test_evaluation": True,
            "validation_scores": {arm: record.get("validation") for arm, record in records.items()},
            "fit_result_sha256": {
                arm: hashlib.sha256((case_root / arm / "result.json").read_bytes()).hexdigest()
                for arm, record in records.items()
                if record["status"] == "converged"
            },
        }
        write_json(case_root / "choice.json", choice)
        if choice["chosen_arm"] is not None:
            for arm, record in records.items():
                if record["status"] == "converged":
                    record["evaluation"] = launch(
                        args, dataset, arm, "evaluate", args.evaluation_timeout
                    )
        additive = records["additive"]
        candidate = records["interactions"]
        costs = {"case_end_to_end_seconds": time.perf_counter() - case_started}
        if additive.get("fit_seconds") and candidate.get("fit_seconds"):
            screen_cost = additive.get("screening_seconds", 0) + additive.get(
                "screening_prediction_seconds", 0
            )
            costs.update(
                joint_refit_over_additive_fit=candidate["fit_seconds"] / additive["fit_seconds"],
                charged_screen_and_fit_over_additive_fit=(
                    additive["fit_seconds"] + screen_cost + candidate["fit_seconds"]
                )
                / additive["fit_seconds"],
                interaction_training_pipeline_process_seconds=additive["process"]["process_seconds"]
                + candidate["process"]["process_seconds"],
            )
        suite["datasets"][dataset] = {"arms": records, "choice": choice, "costs": costs}
        suite["fit_worker_process_seconds_spent"] = spent
        write_json(args.output / "suite.json", suite)
    suite["finished_utc"] = datetime.now(UTC).isoformat()
    write_json(args.output / "suite.json", suite)
    return (
        0
        if all(
            case["choice"]["chosen_arm"] is not None
            and all(
                record["status"] == "converged"
                and record.get("evaluation", {}).get("status") == "evaluated"
                for record in case["arms"].values()
            )
            for case in suite["datasets"].values()
        )
        else 1
    )


def main(argv=None):
    from interaction_datasets import DEFAULT_ROOT, MANIFEST

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interactions", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--fit-timeout", type=float, default=75)
    parser.add_argument("--evaluation-timeout", type=float, default=30)
    parser.add_argument("--total-fit-budget", type=float, default=600)
    parser.add_argument("--max-p", type=int, default=512)
    parser.add_argument("--max-rows", type=int, default=300000)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dataset", help=argparse.SUPPRESS)
    parser.add_argument("--arm", choices=("additive", "interactions"), help=argparse.SUPPRESS)
    parser.add_argument("--stage", choices=("fit", "evaluate"), help=argparse.SUPPRESS)
    parser.add_argument("--case-root", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    for name in ("fit_timeout", "evaluation_timeout", "total_fit_budget"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if (
        args.fit_timeout > 180
        or args.max_p < 1
        or args.max_rows < 1
        or len(set(args.datasets)) != len(args.datasets)
    ):
        parser.error(
            "At most 180 seconds per worker; positive shape budgets and unique datasets required"
        )
    args.output = args.output.resolve()
    args.manifest = args.manifest.resolve()
    args.data_root = args.data_root.resolve()
    return worker(args) if args.worker else run_suite(args)


if __name__ == "__main__":
    raise SystemExit(main())
