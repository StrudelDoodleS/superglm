"""Pre-registered gap table: smooth tensor interactions against gradient boosting.

Every arm, adapter rule, cap and decision rule comes from the 2026-09-19
protocol. One fit per fresh worker process, one receipt per fit, one summary
with the closure fractions and the R1-R4 outcomes.

    uv run python benchmarks/benchmark_gap_table.py --datasets kaggle_sberbank_housing \
        --output .benchmark-artifacts/gap-table/run1
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
import traceback
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from benchmark_gbm_interactions import COMMON, CONFIGS, native_features
from benchmark_housing_tensor import run_isolated, source_fingerprint
from benchmark_real_interactions import (
    SPLITS,
    digest_json,
    load_worker_receipt,
    partition_rows,
    prepare_dataset,
    score_predictions,
    transform_features,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = (
    Path(__file__).with_name("interaction_datasets.json"),
    Path(__file__).with_name("interaction_kaggle_datasets.json"),
)
SEED = 20260919
FEATURE_CAP = 60
SPLINE_KNOTS = 10
DISCRETE_BINS = 256
MAX_REML_ITER = 30
SCREEN_RETAIN = 32
# Pre-run amendment of 2026-09-19: the screen sweeps the pairs among the leading
# features by training-only Spearman (190 pairs), not all pairs of the capped
# sixty (1,770); the full sweep projected to about an hour per dataset at the row cap.
SCREEN_TOP = 20
MISSING_RATE = 0.01
MISSING_SUFFIX = "__missing"
DAY_SECONDS = 24 * 60 * 60
R1_SIGNAL = 0.01
R2_CLOSURE = 0.5
R3_FRACTION = 0.8
# The protocol sets R3's standalone bar at the same number as R2's, not at the same rule.
R3_STANDALONE = 0.5
R4_WALL_RATIO = 5.0
THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    # Left unset, SuperGLM caps BLAS to one thread inside every fit, so the
    # boosting controls would run at four threads against a one-thread solver.
    "SUPERGLM_BLAS_THREADS",
)
PAIR_LIMIT = 16
SPLIT_STRATEGIES = {
    "time": "chronological_group",
    "time_group": "fixed_year_group",
    "group": "random_group",
}
FAMILIES = {
    "binomial": {"name": "binomial", "power": None, "gbm_loss": "log_loss"},
    "gaussian": {"name": "gaussian", "power": 0.0, "gbm_loss": "squared_error"},
    "poisson": {"name": "poisson", "power": 1.0, "gbm_loss": "poisson"},
    "gamma": {"name": "gamma", "power": 2.0, "gbm_loss": "gamma"},
    # HistGradientBoosting has no Tweedie loss, so its control trains on Poisson
    # and is scored on the Tweedie deviance. The protocol records the handicap.
    "tweedie:1.5": {"name": "tweedie", "power": 1.5, "gbm_loss": "poisson"},
}
# The core corpus predates the protocol, which names these roles for it.
CORPUS_PROTOCOL = {
    "uci_breast_cancer": {"response_family": "binomial"},
    "uci_credit_default": {"response_family": "binomial"},
    "ames_housing": {"response_family": "gaussian"},
    "uci_bike_sharing": {"response_family": "poisson"},
    "fremtpl2": {
        "response_family": "poisson",
        "offset_column": "log(Exposure)",
        "exposure_column": "Exposure",
    },
}
# AUC and average precision already come back from every binomial score.
SECONDARY_METRICS = {
    "kaggle_porto_seguro_safe_driver": "normalised_gini",
    "kaggle_liberty_mutual_property_inspection": "normalised_gini",
    "kaggle_liberty_mutual_fire_peril": "normalised_gini",
    "kaggle_allstate_claims_severity": "mean_absolute_error",
    "kaggle_sberbank_housing": "rmsle",
}
GBM_STRUCTURES = {"G0": "no_interactions", "G1": "pairwise", "G2": None, "G2all": None}
UNCAPPED_STRUCTURE = "G2all"
EVIDENCE_ORDER = {"A": 0, "B": 1, "C": 2}
RECEIPT_FIELDS = (
    "dataset",
    "arm",
    "status",
    "family",
    "pairs",
    "rows",
    "split_sha256",
    "adapter_sha256",
    "adapter_rules",
    "feature_cap_rule",
    "kept_features",
    "smooth_features",
    "valid",
    "test",
    "fit_wall_seconds",
    "fit_cpu_seconds",
    "reml_states",
    "outer_iterations",
    "converged",
    "termination_reason",
    "model_fingerprint",
    "package_source_sha256",
)
OFFSET_EXPRESSION = re.compile(r"log\((\w+)\)")


# ── Registry ──────────────────────────────────────────────────────────────


def missing_receipt_fields(record):
    """Which of the protocol's receipt fields this fit failed to record."""
    return [name for name in RECEIPT_FIELDS if name not in record]


def resolve_entry(dataset_id):
    """Find a dataset in either manifest and add the roles the protocol names for it."""
    from interaction_datasets import read_manifest

    for manifest in MANIFESTS:
        for entry in read_manifest(manifest):
            if entry["id"] == dataset_id:
                return {**CORPUS_PROTOCOL.get(dataset_id, {}), **entry}, manifest
    raise ValueError(f"No manifest declares dataset {dataset_id}")


def family_for(entry):
    declared = entry.get("response_family")
    if declared not in FAMILIES:
        raise ValueError(f"{entry['id']} declares no gap-table family (got {declared!r})")
    return FAMILIES[declared]


# ── Adapter ───────────────────────────────────────────────────────────────


def apply_sentinels(frame, entry):
    """Map documented missing sentinels to NaN so no spline ever sees them as numbers."""
    for column, values in entry.get("sentinels", {}).items():
        frame[column] = frame[column].where(~frame[column].isin(values))
    return sorted(entry.get("sentinels", {}))


def missing_companions(frame, entry):
    """Give every numeric predictor at or above the missing rate a two-level companion factor.

    A declared categorical already carries missing as its own level, so a
    companion there would be exactly collinear with the parent.
    """
    declared = set(entry.get("categorical_columns", []))
    columns = {}
    for column in entry["features"]:
        if column in declared or frame[column].isna().mean() < MISSING_RATE:
            continue
        columns[f"{column}{MISSING_SUFFIX}"] = np.where(frame[column].isna(), "missing", "present")
    return columns


def date_parts(frame, entry):
    """Derive calendar predictors from the split's time column, which is never a predictor."""
    column = entry.get("time_column")
    if column is None:
        return {}
    values = frame[column]
    if pd.api.types.is_numeric_dtype(values):
        # TransactionDT counts seconds from a reference the competition never
        # published, so a day index and a weekday are all the calendar it has.
        days = np.floor(values.to_numpy(dtype=float) / DAY_SECONDS)
        derived = {"day_index": days, "day_of_week": days % 7}
    else:
        stamps = pd.to_datetime(values, errors="raise")
        derived = {
            "year": stamps.dt.year,
            "month": stamps.dt.month,
            "day_of_week": stamps.dt.dayofweek,
        }
    return {
        f"{column}__{suffix}": np.asarray(series, dtype=float) for suffix, series in derived.items()
    }


def offset_values(frame, entry):
    """Evaluate the entry's offset expression exactly as it is written."""
    expression = entry.get("offset_column")
    if expression is None:
        return None
    match = OFFSET_EXPRESSION.fullmatch(expression)
    if match is None:
        raise ValueError(f"Unsupported offset expression {expression!r}")
    with np.errstate(divide="ignore"):
        return np.log(frame[match.group(1)].to_numpy(dtype=float))


def drop_unusable_offset_rows(frame, entry):
    """A zero floor area has no logarithm; those rows leave before the split."""
    offset = offset_values(frame, entry)
    if offset is None:
        return frame, 0
    keep = np.flatnonzero(np.isfinite(offset))
    return frame.iloc[keep].reset_index(drop=True), len(frame) - len(keep)


def subsample_groups(frame, entry, max_rows):
    """Keep whole split groups so the row cap never cuts one group into two partitions."""
    if len(frame) <= max_rows:
        return frame
    keys = pd.MultiIndex.from_frame(frame.loc[:, entry["split"]["columns"]])
    codes, unique = pd.factorize(keys, sort=True)
    order = np.random.default_rng(SEED).permutation(len(unique))
    sizes = np.bincount(codes, minlength=len(unique))[order]
    keep = order[np.cumsum(sizes) <= max_rows]
    if not len(keep):
        raise ValueError("The largest split group alone exceeds the row cap")
    return frame.iloc[np.sort(np.flatnonzero(np.isin(codes, keep)))].reset_index(drop=True)


def gbm_offset_feature(frame, entry):
    """The offset as a plain column: the smooth arms take it as an offset, the GBM cannot."""
    if not entry.get("offset_column") or entry.get("exposure_column") is not None:
        return {}
    return {f"offset__{entry['offset_column']}": offset_values(frame, entry)}


def adapt(frame, entry):
    """Apply every protocol adapter rule, before the split and identically in every arm."""
    sentinels = apply_sentinels(frame, entry)
    companions = missing_companions(frame, entry)
    dates = date_parts(frame, entry)
    offset_column = gbm_offset_feature(frame, entry)
    derived = {**companions, **dates, **offset_column}
    # One concatenation, not one insertion per column: a wide table fragments.
    frame = pd.concat([frame, pd.DataFrame(derived, index=frame.index)], axis=1)
    companion_names, date_names = list(companions), list(dates)
    offset_names = list(offset_column)
    adapted = {
        **entry,
        "features": [*entry["features"], *companion_names, *date_names, *offset_names],
        "categorical_columns": [*entry.get("categorical_columns", []), *companion_names],
        "split": {**entry["split"], "seed": entry["split"].get("seed", SEED)},
        "offset_feature": offset_names,
    }
    notes = {
        "sentinel_columns": sentinels,
        "missing_companions": companion_names,
        "missing_rate_threshold": MISSING_RATE,
        "missing_rate_measured_on": "the whole adapted table, before the split",
        "date_features": date_names,
        "offset_expression": entry.get("offset_column"),
        "offset_as_gbm_feature": offset_names,
        "exposure_column": entry.get("exposure_column"),
        "weight_column": entry.get("weight_column"),
        "response_transform": entry.get("response_transform"),
    }
    return frame, adapted, notes


# ── Response, weights and the feature cap ─────────────────────────────────


def response_values(frame, entry):
    """The response on the scale the arm fits, with the entry's declared transform applied."""
    target = frame[entry["primary_target"]]
    if entry.get("positive_class") is not None:
        return (target == entry["positive_class"]).to_numpy(dtype=float)
    values = target.to_numpy(dtype=float)
    return np.log1p(values) if entry.get("response_transform") == "log1p" else values


def weight_values(frame, entry):
    column = entry.get("weight_column")
    return None if column is None else frame[column].to_numpy(dtype=float)


def exposure_values(frame, entry):
    column = entry.get("exposure_column")
    return None if column is None else frame[column].to_numpy(dtype=float)


def split_rows(frame, entry):
    """Partition the adapted frame with the entry's declared strategy."""
    split = entry["split"]
    strategy = SPLIT_STRATEGIES.get(split["strategy"], split["strategy"])
    years = None
    if strategy == "fixed_year_group":
        keys = ("train_years", "validation_years", "test_years")
        years = dict(zip(SPLITS, (split[key] for key in keys), strict=True))
    return partition_rows(
        frame,
        strategy=strategy,
        columns=split["columns"],
        target=entry["primary_target"],
        years=years,
        seed=split["seed"],
    )


def spearman_ranking(train, response, features):
    """Order predictors by absolute Spearman with the response; factors go in by level mean."""
    ranked_response = pd.Series(response, index=train.index).rank()
    ordinal = {}
    for name in features:
        column = train[name]
        if pd.api.types.is_numeric_dtype(column):
            ordinal[name] = column
            continue
        labels = column.astype(object).where(column.notna(), "missing:")
        ordinal[name] = labels.map(ranked_response.groupby(labels).mean())
    strength = pd.DataFrame(ordinal).corrwith(ranked_response, method="spearman")
    return list(strength.abs().fillna(0.0).sort_values(ascending=False, kind="stable").index)


def declared_rank(entry):
    """The registry's own importance order: a list of column names, or nothing declared."""
    rank = entry.get("feature_rank")
    if rank is None:
        return []
    if not isinstance(rank, list):
        raise ValueError(
            f"{entry['id']} declares feature_rank as {type(rank).__name__}; "
            "the field is a list of column names or null, and the prose belongs "
            "in feature_rank_source"
        )
    return rank


def known_good_columns(entry):
    """Every column a catalogue pair names; the cap must not delete the arm it exists to test."""
    return [
        end for pair in entry.get("known_good_pairs", []) for end in (pair["left"], pair["right"])
    ]


def ordered_unique(names):
    return list(dict.fromkeys(names))


def capped_features(train, response, entry, cap):
    """Keep at most `cap` predictors: the declared columns first, then training-only Spearman."""
    features = entry["features"]
    if cap is None or len(features) <= cap:
        return features, "none; the arm keeps every non-excluded predictor"
    present = set(features)
    groups = (
        ("the offset column", [name for name in entry["offset_feature"] if name in present]),
        ("declared feature_rank", [name for name in declared_rank(entry) if name in present]),
        ("catalogue pair columns", [n for n in known_good_columns(entry) if n in present]),
    )
    forced = ordered_unique([name for _, names in groups for name in names])
    remaining = [name for name in features if name not in set(forced)]
    kept = [*forced, *spearman_ranking(train, response, remaining)][:cap]
    rule = " then ".join([*(name for name, names in groups if names), "training-only Spearman"])
    return kept, rule


# ── Scoring ───────────────────────────────────────────────────────────────


def gini_area(response, prediction, weights):
    order = np.argsort(prediction, kind="stable")
    weighted = response[order] * weights[order]
    cumulative_response = np.cumsum(weighted) / weighted.sum()
    cumulative_weight = np.cumsum(weights[order]) / weights.sum()
    return float(
        np.sum(
            cumulative_response[:-1] * cumulative_weight[1:]
            - cumulative_response[1:] * cumulative_weight[:-1]
        )
    )


def normalised_gini(response, prediction, weights):
    """Kaggle's normalised Gini; unit weights give the unweighted competition metric."""
    perfect = gini_area(response, response, weights)
    return 0.0 if perfect == 0 else gini_area(response, prediction, weights) / perfect


def family_scores(response, prediction, family, weights):
    """Mean deviance for the fitted family, weighted where the competition weighted it."""
    if weights is None and family["name"] in {"binomial", "poisson", "gaussian"}:
        return score_predictions(response, prediction, family["name"])
    if family["power"] is None:
        raise ValueError(f"The {family['name']} family has no weighted deviance in this runner")
    from sklearn.metrics import mean_tweedie_deviance

    deviance = float(
        mean_tweedie_deviance(response, prediction, power=family["power"], sample_weight=weights)
    )
    return {"primary_loss": deviance, "mean_deviance": deviance, "rows": len(response)}


def secondary_scores(entry, response, prediction, weights):
    """The competition's own headline metric, on the scale it was scored."""
    metric = SECONDARY_METRICS.get(entry["id"])
    if metric is None:
        return {}
    if metric == "normalised_gini":
        unit = np.ones_like(response) if weights is None else weights
        return {"normalised_gini": normalised_gini(response, prediction, unit)}
    if metric == "mean_absolute_error":
        return {"mean_absolute_error": float(np.mean(np.abs(response - prediction)))}
    # RMSLE is undefined below zero, so a negative money prediction floors at zero.
    raw_prediction = np.log1p(np.clip(np.expm1(prediction), 0.0, None))
    return {"rmsle": float(np.sqrt(np.mean((raw_prediction - response) ** 2)))}


def score_split(entry, family, response, prediction, weights):
    scores = family_scores(response, prediction, family, weights)
    scores.update(secondary_scores(entry, response, prediction, weights))
    return scores


# ── Models ────────────────────────────────────────────────────────────────


def superglm_family(family):
    from superglm import families

    if family["name"] == "tweedie":
        return families.tweedie(family["power"])
    return family["name"]


def usable_pairs(requested, smooth_features):
    """Keep only the pairs whose two columns both survived the cap and the adapter."""
    fitted = set(smooth_features)
    return [pair for pair in requested if pair[0] in fitted and pair[1] in fitted]


def build_superglm(state, family, pairs, smooth_features):
    from superglm import Categorical, Numeric, Spline, SuperGLM

    features = {}
    for name in smooth_features:
        spec = state["features"][name]
        if spec["kind"] == "categorical":
            features[name] = Categorical(levels=spec["levels"])
        elif spec["kind"] == "spline":
            features[name] = Spline(kind="ps", k=SPLINE_KNOTS)
        else:
            features[name] = Numeric()
    return SuperGLM(
        family=superglm_family(family),
        features=features,
        interactions=[tuple(pair) for pair in pairs],
        selection_penalty=0.0,
        discrete=True,
        n_bins=DISCRETE_BINS,
    )


def build_gbm(family, structure, capacity):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    estimator = (
        HistGradientBoostingClassifier
        if family["name"] == "binomial"
        else HistGradientBoostingRegressor
    )
    return estimator(
        loss=family["gbm_loss"],
        interaction_cst=GBM_STRUCTURES[structure],
        **COMMON,
        **CONFIGS[capacity],
    )


def known_good_pairs(entry):
    """The catalogue's verified pairs, strongest evidence first, then registry order."""
    pairs = entry.get("known_good_pairs", [])
    ordered = sorted(
        enumerate(pairs), key=lambda item: (EVIDENCE_ORDER.get(item[1]["grade"], 3), item[0])
    )
    return [[pair["left"], pair["right"]] for _, pair in ordered]


def screened_pairs(table, count):
    """Take the top pairs by z; a refused or non-finite screen is recorded, not ranked."""
    scored = np.isfinite(table["z"].to_numpy(dtype=float))
    refused = [[row.feature_a, row.feature_b] for row in table[~scored].itertuples()]
    ranked = table[scored].sort_values("z", ascending=False, kind="stable")
    return [[row.feature_a, row.feature_b] for row in ranked.head(count).itertuples()], refused


# ── One fit ───────────────────────────────────────────────────────────────


def prepare_case(dataset, args, cap):
    """Load, adapt, cap and split one dataset; every arm of a dataset sees this."""
    from interaction_datasets import load_dataset

    entry, manifest = resolve_entry(dataset)
    frame = load_dataset(entry["id"], root=args.data_root, manifest=manifest)
    frame, dropped = drop_unusable_offset_rows(frame, entry)
    frame = subsample_groups(frame, entry, args.max_rows)
    frame, entry, adapter_notes = adapt(frame, entry)
    rows = split_rows(frame, entry)
    kept, cap_rule = capped_features(
        frame.iloc[rows["train"]], response_values(frame.iloc[rows["train"]], entry), entry, cap
    )
    entry = {
        **entry,
        "features": kept,
        "categorical_columns": [c for c in entry["categorical_columns"] if c in set(kept)],
    }
    state, rows, split_hash = prepare_dataset(frame, entry)
    offset_feature = set(entry["offset_feature"])
    return {
        "entry": entry,
        "frame": frame,
        "rows": rows,
        "state": state,
        # The offset is a fixed unit elasticity, so it is never also a free
        # smooth term; the boosting controls keep it as an ordinary column.
        "smooth_features": [name for name in state["features"] if name not in offset_feature],
        "family": family_for(entry),
        "split_sha256": split_hash,
        "adapter_sha256": digest_json(state),
        "adapter_rules": adapter_notes,
        "feature_cap": cap,
        "feature_cap_rule": cap_rule,
        "kept_features": kept,
        "offset_unusable_rows": dropped,
    }


def case_partition(case, name, *, native):
    """The frozen adapter's view of one partition, plus the arrays the arm fits against."""
    entry, frame = case["entry"], case["frame"]
    raw = frame.iloc[case["rows"][name]]
    encode = native_features if native else transform_features
    design = encode(raw.loc[:, entry["features"]], case["state"])
    return {
        "design": design if native else design.loc[:, case["smooth_features"]],
        "response": response_values(raw, entry),
        "offset": offset_values(raw, entry),
        "weights": weight_values(raw, entry),
        "exposure": exposure_values(raw, entry),
    }


def timed(record, call):
    """Run one fit and record the wall and CPU seconds it alone cost."""
    started, cpu_started = time.perf_counter(), time.process_time()
    call()
    record["fit_wall_seconds"] = time.perf_counter() - started
    record["fit_cpu_seconds"] = time.process_time() - cpu_started


def superglm_fit(case, pairs, record):
    """Fit one smooth arm and record its REML trajectory."""
    train = case_partition(case, "train", native=False)
    model = build_superglm(case["state"], case["family"], pairs, case["smooth_features"])
    timed(
        record,
        lambda: model.fit_reml(
            train["design"],
            train["response"],
            sample_weight=train["weights"],
            offset=train["offset"],
            max_reml_iter=MAX_REML_ITER,
        ),
    )
    diagnostics = model.reml_diagnostics()
    record.update(
        reml_states=len(diagnostics["lambda_history"]),
        outer_iterations=int(diagnostics["n_reml_iter"]),
        converged=bool(diagnostics["converged"]),
        termination_reason=str(diagnostics["termination_reason"]),
        coefficient_count=int(len(model.result.beta)),
        smoothing_parameter_count=len(diagnostics["lambdas"]),
    )
    return model, train


def gbm_fit(case, structure, capacity, record):
    """Fit one boosting control; a fixed round budget is not a convergence certificate."""
    train = case_partition(case, "train", native=True)
    target, weights = gbm_training_target(train)
    model = build_gbm(case["family"], structure, capacity)
    timed(record, lambda: model.fit(train["design"], target, sample_weight=weights))
    record.update(
        reml_states=None,
        outer_iterations=int(model.n_iter_),
        converged=True,
        termination_reason="fixed boosting budget; no early stopping",
        coefficient_count=None,
        smoothing_parameter_count=None,
    )
    return model, train


def gbm_training_target(partition):
    """Exposure becomes a rate target with exposure weights; other weights pass through."""
    if partition["exposure"] is not None:
        return partition["response"] / partition["exposure"], partition["exposure"]
    return partition["response"], partition["weights"]


def superglm_predictions(model, partition):
    return model.predict(partition["design"], offset=partition["offset"])


def gbm_predictions(model, partition, family):
    raw = (
        model.predict_proba(partition["design"])[:, 1]
        if family["name"] == "binomial"
        else model.predict(partition["design"])
    )
    return raw if partition["exposure"] is None else raw * partition["exposure"]


def score_arm(case, model, engine, record):
    """Score validation and test on identical rows, and fingerprint the fitted model by them."""
    predictions = {}
    for name in ("valid", "test"):
        partition = case_partition(case, name, native=engine == "gbm")
        predictions[name] = (
            gbm_predictions(model, partition, case["family"])
            if engine == "gbm"
            else superglm_predictions(model, partition)
        )
        record[name] = score_split(
            case["entry"],
            case["family"],
            partition["response"],
            predictions[name],
            partition["weights"],
        )
    fingerprint = hashlib.sha256()
    for name in ("valid", "test"):
        fingerprint.update(np.ascontiguousarray(predictions[name], dtype=float).tobytes())
    record["model_fingerprint"] = fingerprint.hexdigest()


def screened_by_the_baseline(args):
    """The pairs A0's screen ranked; a blind arm has nothing to fit without them."""
    receipt = json.loads((args.case_root / "A0" / "receipt.json").read_text())
    if "screening" not in receipt:
        raise ValueError(
            f"A0 on {args.dataset} recorded no screen ({receipt.get('status')}), "
            f"so {args.arm} has no ranked pairs"
        )
    return receipt["screening"]["pairs"]


def distinct_pairs(pairs):
    """One entry per unordered pair, first come: a tensor term is symmetric in its columns."""
    distinct = {}
    for pair in pairs:
        distinct.setdefault(tuple(sorted(pair)), pair)
    return list(distinct.values())


def arm_pairs(args, case):
    """What an arm requested, what it can fit after the cap, and where the pairs came from."""
    arm, fitted = args.arm, case["smooth_features"]
    if arm == "A0":
        return [], [], {"source": "none; the additive baseline"}
    if arm == "A1":
        requested = known_good_pairs(case["entry"])
        return (
            requested,
            distinct_pairs(usable_pairs(requested, fitted))[:PAIR_LIMIT],
            {"source": "catalogue known-good pairs", "limit": PAIR_LIMIT},
        )
    screened = screened_by_the_baseline(args)
    if arm == "A3":
        requested = known_good_pairs(case["entry"]) + screened[: args.union_pairs]
        return (
            requested,
            distinct_pairs(usable_pairs(requested, fitted))[:PAIR_LIMIT],
            {
                "source": f"A1 pairs plus the top {args.union_pairs} screened pairs",
                "limit": PAIR_LIMIT,
            },
        )
    count = int(arm.split("-")[1])
    pairs = screened[:count]
    return pairs, pairs, {"source": f"top {count} screened pairs by z"}


def screen_candidates(train, case):
    """Pairs among the leading fitted features by training-only Spearman; None means every pair."""
    names = case["smooth_features"]
    if len(names) <= SCREEN_TOP:
        return None
    kinds = {name: case["state"]["features"][name]["kind"] for name in names}
    leading = spearman_ranking(train["design"], train["response"], names)[:SCREEN_TOP]
    # The screen has no refit target for a spline by a linear numeric and skips
    # that pair by itself when it is given no candidate list.
    return [
        pair
        for pair in itertools.combinations(leading, 2)
        if {kinds[pair[0]], kinds[pair[1]]} != {"spline", "numeric"}
    ]


def run_screen(model, case, train, record):
    """Rank the candidate pairs on the training rows with the library's own screen."""
    started = time.perf_counter()
    candidates = screen_candidates(train, case)
    table = model.screen_interactions(
        train["design"], train["response"], sample_weight=train["weights"], candidates=candidates
    )
    pairs, refused = screened_pairs(table, SCREEN_RETAIN)
    record["screening"] = {
        "seconds": time.perf_counter() - started,
        "candidate_count": len(table),
        "candidate_rule": (
            "every pair of fitted features"
            if candidates is None
            else f"pairs among the {SCREEN_TOP} fitted features with the strongest "
            "training-only Spearman association with the response"
        ),
        "pairs": pairs,
        "refused_or_nonfinite": refused,
        "deferred_features": table.attrs.get("deferred_features", {}),
    }


def fit_one_arm(args, case, plan, record):
    """Fit the arm this worker is for, and return the model with its training partition."""
    if plan["engine"] == "gbm":
        record.update(
            requested_pairs=[],
            pairs=[],
            pair_source={"source": "boosting control"},
            structure=plan["structure"],
            capacity=plan["capacity"],
        )
        return gbm_fit(case, plan["structure"], plan["capacity"], record)
    requested, pairs, source = arm_pairs(args, case)
    record.update(requested_pairs=requested, pairs=pairs, pair_source=source)
    return superglm_fit(case, pairs, record)


def fit_arm(args, record):
    plan = arm_plan(args.arm)
    case = prepare_case(args.dataset, args, plan["feature_cap"])
    record.update(
        family=case["entry"]["response_family"],
        rows={name: len(indices) for name, indices in case["rows"].items()},
        full_table_rows=len(case["frame"]),
        split=case["entry"]["split"],
        split_sha256=case["split_sha256"],
        adapter_sha256=case["adapter_sha256"],
        adapter_rules=case["adapter_rules"],
        feature_cap=case["feature_cap"],
        feature_cap_rule=case["feature_cap_rule"],
        kept_features=case["kept_features"],
        smooth_features=case["smooth_features"],
        offset_unusable_rows=case["offset_unusable_rows"],
        entry_sha256=digest_json(case["entry"]),
    )
    model, train = fit_one_arm(args, case, plan, record)
    record["status"] = "converged" if record["converged"] else "not_converged"
    score_arm(case, model, plan["engine"], record)
    if args.arm == "A0":
        # The screen costs more than the fit on a wide table, and the parent's
        # deadline kills this process: the scored baseline is on disk first.
        save_receipt(args, record)
        run_screen(model, case, train, record)


def arm_plan(arm):
    """Which engine, structure and feature cap an arm name stands for."""
    if not arm.startswith("G"):
        return {"engine": "superglm", "feature_cap": FEATURE_CAP}
    structure, capacity = arm.split("-")
    return {
        "engine": "gbm",
        "structure": structure,
        "capacity": capacity,
        "feature_cap": None if structure == UNCAPPED_STRUCTURE else FEATURE_CAP,
    }


def runtime_identity():
    from threadpoolctl import threadpool_info

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("superglm", "numpy", "scipy", "pandas", "scikit-learn", "threadpoolctl")
        },
        "threadpools": threadpool_info(),
        "thread_environment": {name: os.environ.get(name) for name in THREAD_VARIABLES},
    }


def save_receipt(args, record):
    """Write the receipt as it stands, so a later kill cannot erase what is already measured."""
    record["missing_receipt_fields"] = missing_receipt_fields(record)
    write_json(args.output / "receipt.json", record)


def worker(args):
    started = time.perf_counter()
    record = {
        "dataset": args.dataset,
        "arm": args.arm,
        "status": "starting",
        "started_utc": datetime.now(UTC).isoformat(),
        "protocol": "2026-09-19 gap table",
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            record["package_source_sha256"] = source_fingerprint()
            record["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            record["git_head"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip()
            record["runtime"] = runtime_identity()
            fit_arm(args, record)
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
            save_receipt(args, record)
    print(json.dumps({key: record[key] for key in ("dataset", "arm", "status")}), flush=True)
    return 0 if record["status"] in {"converged", "not_converged"} else 1


# ── Suite ─────────────────────────────────────────────────────────────────


def launch(args, dataset, arm):
    """Run one fit in a fresh process with every numerical thread pool pinned."""
    case_root = args.output / dataset
    output = case_root / arm
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--dataset",
        dataset,
        "--arm",
        arm,
        "--output",
        str(output),
        "--case-root",
        str(case_root),
        "--data-root",
        str(args.data_root),
        "--max-rows",
        str(args.max_rows),
        "--union-pairs",
        str(args.union_pairs),
    ]
    environment = {**os.environ, **dict.fromkeys(THREAD_VARIABLES, str(args.threads))}
    # A0 pays for the screen on top of its fit; the fit's own deadline is the same one.
    deadline = args.fit_timeout + (args.screen_timeout if arm == "A0" else 0.0)
    process = run_isolated(
        command, log_path=output / "worker.log", timeout=deadline, env=environment
    )
    receipt_path = output / "receipt.json"
    record = load_worker_receipt(receipt_path, arm)
    if process["status"] != "success":
        # A fit the worker already scored and wrote survives a kill during the screen.
        record["interrupted_by"] = process["status"]
        if "test" not in record:
            record["status"] = process["status"]
    record["process"] = {**process, "command": command, "timeout_seconds": deadline}
    write_json(receipt_path, record)
    print(
        json.dumps(
            {
                "dataset": dataset,
                "arm": arm,
                "status": record["status"],
                "process_seconds": process["process_seconds"],
            }
        ),
        flush=True,
    )
    return record


def arm_menu(args, entry):
    """Every arm the protocol asks of this dataset, in the order they must run."""
    arms = ["A0", *[f"A2-{count}" for count in args.pairs]]
    if entry.get("known_good_pairs"):
        arms.insert(1, "A1")
        arms.append("A3")
    arms += [
        f"{structure}-{capacity}" for structure in GBM_STRUCTURES for capacity in args.capacities
    ]
    return arms


def test_loss(record):
    loss = record.get("test", {}).get("primary_loss")
    return loss if isinstance(loss, float) and math.isfinite(loss) else None


def select_gbm_capacity(records, structure, capacities):
    """Pick one capacity per boosting structure on validation loss alone."""
    candidates = []
    for capacity in capacities:
        arm = f"{structure}-{capacity}"
        loss = records.get(arm, {}).get("valid", {}).get("primary_loss")
        if isinstance(loss, float) and math.isfinite(loss):
            candidates.append((loss, arm))
    return min(candidates)[1] if candidates else None


def closure(additive_loss, arm_loss, ceiling_loss):
    """Fraction of the additive-to-unrestricted gap an arm closes; undefined when there is none."""
    if additive_loss is None or arm_loss is None or ceiling_loss is None:
        return None
    signal = additive_loss - ceiling_loss
    return None if signal == 0 else (additive_loss - arm_loss) / signal


def dataset_summary(args, records, entry):
    """Every arm's test loss, the closure fractions and the per-dataset rule outcomes."""
    selected = {
        structure: select_gbm_capacity(records, structure, args.capacities)
        for structure in GBM_STRUCTURES
    }
    losses = {arm: test_loss(record) for arm, record in records.items()}
    for structure, arm in selected.items():
        losses[structure] = None if arm is None else losses.get(arm)
    additive_smooth = losses.get("A0")
    unrestricted_gbm, additive_gbm = losses.get("G2"), losses.get("G0")
    smooth_arms = [arm for arm in records if not arm.startswith("G")]
    fractions = {
        arm: closure(additive_smooth, losses.get(arm), unrestricted_gbm) for arm in smooth_arms
    }
    fractions["G1"] = closure(additive_smooth, losses.get("G1"), unrestricted_gbm)
    return {
        "test_loss": losses,
        "validation_loss": {
            arm: record.get("valid", {}).get("primary_loss") for arm, record in records.items()
        },
        "secondary": {arm: secondary_of(record) for arm, record in records.items()},
        "selected_gbm_capacity": selected,
        "pairs": {arm: record.get("pairs") for arm, record in records.items()},
        "status": {arm: record.get("status") for arm, record in records.items()},
        "fit_wall_seconds": {
            arm: record.get("fit_wall_seconds") for arm, record in records.items()
        },
        "representation_gap_at_zero_interactions": none_difference(additive_smooth, additive_gbm),
        "interaction_signal": none_difference(additive_gbm, unrestricted_gbm),
        "closure": fractions,
        "has_known_good_list": bool(entry.get("known_good_pairs")),
        "rules": dataset_rules(args, records, losses, fractions),
    }


def secondary_of(record):
    known = {"normalised_gini", "mean_absolute_error", "rmsle", "roc_auc", "average_precision"}
    return {key: value for key, value in record.get("test", {}).items() if key in known}


def none_difference(left, right):
    return None if left is None or right is None else left - right


def dataset_rules(args, records, losses, fractions):
    """R1 admits a dataset to the closure summary; R4 judges each smooth arm's cost."""
    additive_gbm, unrestricted_gbm = losses.get("G0"), losses.get("G2")
    signal = (
        None
        if additive_gbm in (None, 0) or unrestricted_gbm is None
        else (additive_gbm - unrestricted_gbm) / abs(additive_gbm)
    )
    selected = select_gbm_capacity(records, "G2", args.capacities)
    reference = None if selected is None else records[selected].get("fit_wall_seconds")
    cost = {}
    for arm, record in records.items():
        wall = record.get("fit_wall_seconds")
        if arm.startswith("G") or wall is None or reference in (None, 0):
            continue
        cost[arm] = {"wall_ratio": wall / reference, "cheap": wall <= R4_WALL_RATIO * reference}
    return {
        "R1": {
            "relative_interaction_signal": signal,
            "enters_closure_summary": signal is not None and signal >= R1_SIGNAL,
            "threshold": R1_SIGNAL,
        },
        "R4": {"unrestricted_gbm_wall_seconds": reference, "arms": cost, "ratio": R4_WALL_RATIO},
        "closure_reported": {arm: value for arm, value in fractions.items() if value is not None},
    }


def suite_rules(args, cases):
    """R2 grades the representation and R3 the screen, over the R1 datasets only."""
    admitted = [
        (name, case)
        for name, case in cases.items()
        if case["rules"]["R1"]["enters_closure_summary"]
    ]
    with_list = [(name, case) for name, case in admitted if case["has_known_good_list"]]
    without = [(name, case) for name, case in admitted if not case["has_known_good_list"]]
    screen_arm = f"A2-{args.union_pairs}"
    representation = {name: case["closure"].get("A1") for name, case in with_list}
    passed = [
        value for value in representation.values() if value is not None and value >= R2_CLOSURE
    ]
    selection = {}
    for name, case in with_list:
        known, blind = case["closure"].get("A1"), case["closure"].get(screen_arm)
        selection[name] = None if known is None or blind is None else blind >= R3_FRACTION * known
    for name, case in without:
        blind = case["closure"].get(screen_arm)
        selection[name] = None if blind is None else blind >= R3_STANDALONE
    decided = [value for value in selection.values() if value is not None]
    return {
        "R1_admitted_datasets": [name for name, _ in admitted],
        "R2": {
            "closure_by_dataset": representation,
            "passing": len(passed),
            "eligible": len(with_list),
            "holds": bool(with_list) and 2 * len(passed) >= len(with_list),
            "threshold": R2_CLOSURE,
            "undecided": [name for name, value in representation.items() if value is None],
        },
        "R3": {
            "screen_arm": screen_arm,
            "by_dataset": selection,
            "holds": bool(decided) and all(decided),
            "fraction_of_known_good": R3_FRACTION,
            "standalone_threshold": R3_STANDALONE,
            "undecided": [name for name, value in selection.items() if value is None],
            # A known-good arm that lost ground makes 0.8 x closure(A1) a bar
            # anything clears, so the pass on these datasets says nothing.
            "vacuous_known_good_comparison": [
                name
                for name, case in with_list
                if case["closure"].get("A1") is not None and case["closure"]["A1"] <= 0
            ],
        },
    }


def run_suite(args):
    args.output.mkdir(parents=True, exist_ok=True)
    suite = {
        "schema_version": 1,
        "protocol": "2026-09-19 gap table",
        "started_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "threads": args.threads,
        "max_rows": args.max_rows,
        "screen_counts": args.pairs,
        "gbm_capacities": args.capacities,
        "fit_timeout_seconds": args.fit_timeout,
        "screen_timeout_seconds": args.screen_timeout,
        "datasets": {},
    }
    for dataset in args.datasets:
        entry, _ = resolve_entry(dataset)
        records = {}
        for arm in arm_menu(args, entry):
            records[arm] = launch(args, dataset, arm)
        suite["datasets"][dataset] = dataset_summary(args, records, entry)
        write_json(args.output / "summary.json", suite)
    suite["suite_rules"] = suite_rules(args, suite["datasets"])
    suite["finished_utc"] = datetime.now(UTC).isoformat()
    write_json(args.output / "summary.json", suite)
    return 0


def main(argv=None):
    from interaction_datasets import DEFAULT_ROOT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--max-rows", type=int, default=300000)
    parser.add_argument("--pairs", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--fit-timeout", type=float, default=900)
    parser.add_argument("--screen-timeout", type=float, default=900)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dataset", help=argparse.SUPPRESS)
    parser.add_argument("--arm", help=argparse.SUPPRESS)
    parser.add_argument("--case-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--union-pairs", type=int, default=8, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.smoke:
        args.max_rows, args.pairs = min(args.max_rows, 50000), [4]
    args.capacities = ["leaves15"] if args.smoke else list(CONFIGS)
    if args.max_rows < 1 or args.threads < 1 or min(args.fit_timeout, args.screen_timeout) <= 0:
        parser.error("Row cap, thread count and both timeouts must be positive")
    if len(set(args.datasets)) != len(args.datasets) or any(count < 1 for count in args.pairs):
        parser.error("Datasets must be unique and every screen count positive")
    if args.worker:
        args.output.mkdir(parents=True, exist_ok=True)
    elif not args.datasets:
        parser.error("--datasets requires at least one dataset id")
    else:
        args.union_pairs = 8 if 8 in args.pairs else max(args.pairs)
    for name in ("output", "data_root"):
        setattr(args, name, getattr(args, name).resolve())
    return worker(args) if args.worker else run_suite(args)


if __name__ == "__main__":
    raise SystemExit(main())
