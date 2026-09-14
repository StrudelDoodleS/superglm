"""Frozen adapters for the September 14 broad interaction research batch.

This module verifies source data, fixes eligible rows and partitions, and learns
preprocessing from training rows. It does not fit a response model or inspect
validation/test scores. Row indices always refer to the unchanged source frame.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from benchmark_real_interactions import (
    SPLITS,
    digest_json,
    fit_preprocessor,
    numeric_values,
    partition_rows,
)
from interaction_datasets import DEFAULT_ROOT, MANIFEST, load_dataset, read_manifest, source_path

DATASETS = (
    "uci_abalone",
    "uci_concrete",
    "uci_wine_quality",
    "uci_parkinsons",
    "uci_protein_structure",
    "uci_airfoil",
    "uci_power_plant",
    "uci_appliances",
    "uci_seoul_bike",
    "uci_metro_traffic",
    "uci_sgemm",
    "kaggle_king_county_sales",
    "uci_superconductivity",
)
MAX_ROWS = 300_000
PROTOCOL_VERSION = "2026-09-14-v1"
KAGGLE_MANIFEST = Path(__file__).with_name("interaction_kaggle_datasets.json")
TIME_FORMATS = {
    "uci_appliances": ("date", "%Y-%m-%d%H:%M:%S"),
    "uci_seoul_bike": ("Date", "%d/%m/%Y"),
    "uci_metro_traffic": ("date_time", "%Y-%m-%d %H:%M:%S"),
    "kaggle_king_county_sales": ("date", "%Y%m%dT%H%M%S"),
}


def family_for(entry):
    """Use the prespecified family on the original response scale."""
    dataset = entry["id"]
    if dataset not in DATASETS:
        raise ValueError(f"Dataset {dataset!r} is not in the approved batch")
    return (
        "poisson"
        if dataset in {"uci_abalone", "uci_seoul_bike", "uci_metro_traffic"}
        else "gaussian"
    )


def response_values(frame, entry):
    """Return raw numeric responses; declared missing targets remain NaN.

    Callers fit/score only the retained positions returned by ``prepare_frame``.
    Missing responses are never imputed. Invalid support is an error, not a
    reason to remove a row.
    """
    family = family_for(entry)
    values = numeric_values(frame[entry["primary_target"]])
    rule = entry.get("target_rule", {})
    if np.isnan(values).any() and not rule.get("allow_missing", False):
        raise ValueError("Missing target values are not allowed by this entry")
    observed = values[~np.isnan(values)]
    if not len(observed):
        raise ValueError("No observed target values")
    if "minimum" in rule and (observed < rule["minimum"]).any():
        raise ValueError("Observed target is below its declared minimum")
    if family == "poisson" and (observed < 0).any():
        raise ValueError("Count target must be nonnegative")
    if (family == "poisson" or rule.get("integer", False)) and (
        observed != np.floor(observed)
    ).any():
        raise ValueError("Count target contains noninteger values")
    return values


def _timestamps(frame, dataset):
    column, fmt = TIME_FORMATS[dataset]
    raw = frame[column].reset_index(drop=True)
    if raw.isna().any() or not raw.map(lambda value: isinstance(value, str)).all():
        raise ValueError(f"Missing or non-string timestamp in {column}")
    if dataset == "uci_seoul_bike":
        # UCI's CSV uses both padded and unpadded day/month fields. Normalize
        # spelling explicitly while fixing their order; never infer the locale.
        if not raw.str.fullmatch(r"[0-9]{1,2}/[0-9]{1,2}/[0-9]{4}").all():
            raise ValueError(f"Invalid timestamp in {column}; expected day/month/year")
        fields = raw.str.split("/", expand=True)
        raw = fields[0].str.zfill(2) + "/" + fields[1].str.zfill(2) + "/" + fields[2]
    try:
        parsed = pd.to_datetime(raw, format=fmt, exact=True, errors="raise")
    except (ValueError, TypeError) as error:
        raise ValueError(f"Invalid timestamp in {column}; expected {fmt}") from error
    if not parsed.dt.strftime(fmt).equals(raw):
        raise ValueError(f"Noncanonical timestamp in {column}; expected {fmt}")
    return parsed


def _counts(rows):
    return {name: len(rows[name]) for name in SPLITS}


def _split_hash(rows):
    digest = hashlib.sha256()
    for name in SPLITS:
        digest.update(name.encode())
        digest.update(rows[name].astype("<i8").tobytes())
    return digest.hexdigest()


def _raw_partitions(frame, entry):
    dataset, split = entry["id"], entry["split"]
    audit = {"manifest_strategy": split["strategy"], "manifest_policy": split.get("policy")}
    timestamps = None
    if dataset in TIME_FORMATS:
        column, fmt = TIME_FORMATS[dataset]
        expected = [column, "id"] if dataset == "kaggle_king_county_sales" else [column]
        strategy = "chronological_grouped" if dataset == "kaggle_king_county_sales" else "time"
        if split["columns"] != expected or split["strategy"] != strategy:
            raise ValueError("Manifest timestamp/group contract differs from the frozen adapter")
        timestamps = _timestamps(frame, dataset)
        keys = pd.DataFrame({"calendar_day": timestamps.dt.normalize()})
        rows = partition_rows(keys, strategy="chronological_group", columns=["calendar_day"])
        audit.update(
            unit="calendar_day",
            timestamp_column=column,
            timestamp_format=fmt,
            timestamp_normalization=(
                "Pad day/month fields to two digits without changing day-first order."
                if dataset == "uci_seoul_bike"
                else "none"
            ),
            unique_days=int(keys["calendar_day"].nunique()),
            fraction_rule="floor(0.6 * days), floor(0.8 * days), all remaining days",
            raw_time_ranges={
                name: {
                    "first": timestamps.iloc[index].min().isoformat(),
                    "last": timestamps.iloc[index].max().isoformat(),
                    "days": int(keys.iloc[index]["calendar_day"].nunique()),
                }
                for name, index in rows.items()
            },
        )
    elif dataset == "uci_protein_structure":
        if split["strategy"] != "exploratory_random" or split["columns"]:
            raise ValueError("CASP requires its declared exploratory source-row split")
        keys = pd.DataFrame({"source_row_position": np.arange(len(frame))})
        rows = partition_rows(
            keys,
            strategy="random_group",
            columns=["source_row_position"],
            seed=split["seed"],
        )
        audit.update(
            unit="source_row_position",
            unique_groups=len(frame),
            seed=split["seed"],
            limitation="Protein identifiers are absent; protein independence is not established.",
        )
    else:
        if split["strategy"] not in {"group", "random_group"} or not split["columns"]:
            raise ValueError("Unsupported group split in the frozen batch")
        rows = partition_rows(
            frame, strategy="random_group", columns=split["columns"], seed=split["seed"]
        )
        audit.update(
            unit="manifest_group",
            group_columns=split["columns"],
            unique_groups=len(frame.loc[:, split["columns"]].drop_duplicates()),
            seed=split["seed"],
        )
    audit["raw_rows"] = _counts(rows)
    audit["raw_split_sha256"] = _split_hash(rows)
    return rows, timestamps, audit


def prepare_frame(frame, entry):
    """Freeze a fixture-testable adapter without claiming verified source bytes.

    Every source position is either retained exactly once, ineligible for a
    declared reason, or purged by the fixed temporal/group rule. Cutoffs and
    group assignment are computed before target/operating eligibility.
    """
    family = family_for(entry)
    if not 0 < len(frame) <= MAX_ROWS:
        raise ValueError(f"Source exceeds the row budget or is empty: {len(frame)}")
    features = entry["features"]
    if not features or len(features) != len(set(features)) or not frame.columns.is_unique:
        raise ValueError("Predictor and source columns must be nonempty and unique")
    protected = {entry["primary_target"], *entry.get("exclude_columns", {})}
    if protected & set(features):
        raise ValueError("A target or excluded column appears in the predictors")
    required = set(features) | {entry["primary_target"]} | set(entry["split"]["columns"])
    if required - set(frame):
        raise ValueError(f"Required columns are absent: {sorted(required - set(frame))}")
    categorical = entry.get("categorical_columns", [])
    text_predictors = set(
        frame.loc[:, features].select_dtypes(include=["object", "string", "category"])
    )
    if set(categorical) - set(features) or text_predictors - set(categorical):
        raise ValueError("Categorical predictor declarations do not match the source columns")

    raw_rows, timestamps, split_audit = _raw_partitions(frame, entry)
    values = response_values(frame, entry)
    eligible = ~np.isnan(values)
    ineligible = {}
    if not eligible.all():
        ineligible["missing_target"] = np.flatnonzero(~eligible).tolist()
    dataset = entry["id"]
    if dataset == "uci_seoul_bike":
        if entry.get("row_policy", {}).get("after_split") != "Functioning Day == Yes":
            raise ValueError("Seoul operating eligibility must be explicitly declared")
        operating = frame["Functioning Day"].reset_index(drop=True)
        if operating.isna().any() or not operating.isin(["Yes", "No"]).all():
            raise ValueError("Functioning Day must contain only Yes/No operating flags")
        closed = operating.eq("No").to_numpy()
        if (eligible & closed).any():
            ineligible["closed_hours"] = np.flatnonzero(eligible & closed).tolist()
        eligible &= ~closed

    raw_owner = np.empty(len(frame), dtype=np.int8)
    for ordinal, name in enumerate(SPLITS):
        raw_owner[raw_rows[name]] = ordinal
    purge_masks = {}
    if dataset == "uci_appliances":
        for name in ("valid", "test"):
            boundary = timestamps.iloc[raw_rows[name]].min().normalize()
            purge_masks[f"one_day_before_{name}"] = (
                (timestamps >= boundary - pd.Timedelta(days=1)) & (timestamps < boundary)
            ).to_numpy()
        split_audit["gap_policy"] = (
            "Purge the one calendar day immediately before each later partition."
        )
    elif dataset == "kaggle_king_county_sales":
        groups = frame["id"].reset_index(drop=True)
        if groups.isna().any():
            raise ValueError("Missing property group IDs are not allowed")
        latest_owner = pd.Series(raw_owner).groupby(groups, sort=False).transform("max").to_numpy()
        purge_masks["property_seen_in_later_partition"] = raw_owner < latest_owner
        split_audit.update(
            group_columns=["id"],
            unique_groups=int(groups.nunique()),
            cross_partition_groups=int(groups[raw_owner < latest_owner].nunique()),
            group_policy="Keep a property's rows only in its latest original partition; purge earlier occurrences.",
        )

    weights = np.ones(len(frame), dtype=float)
    weighting = {"policy": "one unit per retained source row"}
    if dataset == "uci_metro_traffic":
        targets = pd.Series(values).groupby(timestamps, sort=False)
        if (targets.nunique(dropna=False) != 1).any():
            raise ValueError("Metro has inconsistent target values at a repeated timestamp")
        counts = targets.transform("size").to_numpy()
        weights = 1.0 / counts
        weighting.update(
            policy="Each retained annotation receives 1 / annotation_count_at_timestamp.",
            source_unique_timestamps=int(timestamps.nunique()),
            source_duplicate_annotations=int(len(frame) - timestamps.nunique()),
        )

    purged = {}
    retained = eligible.copy()
    for reason, mask in purge_masks.items():
        positions = np.flatnonzero(retained & mask)
        if len(positions):
            purged[reason] = positions.tolist()
        retained &= ~mask
    rows = {name: index[retained[index]] for name, index in raw_rows.items()}
    if any(not len(index) for index in rows.values()):
        raise ValueError("Every split must retain eligible rows after the frozen purge policy")
    joined = np.concatenate(list(rows.values()))
    removed = [*ineligible.values(), *purged.values()]
    accounted = np.concatenate([joined, *(np.asarray(index, dtype=np.int64) for index in removed)])
    if not np.array_equal(np.sort(accounted), np.arange(len(frame))):
        raise ValueError("Row accounting must preserve or explicitly exclude every source position")
    if dataset == "kaggle_king_county_sales":
        owners = pd.DataFrame({"id": frame["id"].to_numpy()[joined], "owner": raw_owner[joined]})
        if (owners.groupby("id")["owner"].nunique() > 1).any():
            raise ValueError("Property IDs cross retained partitions")

    state = fit_preprocessor(
        frame.iloc[rows["train"]].loc[:, features], categorical_columns=categorical
    )
    split_audit["retained_rows"] = _counts(rows)
    split_audit["retained_weight"] = {
        name: float(weights[index].sum()) for name, index in rows.items()
    }
    if timestamps is not None:
        split_audit["retained_time_ranges"] = {
            name: {
                "first": timestamps.iloc[index].min().isoformat(),
                "last": timestamps.iloc[index].max().isoformat(),
            }
            for name, index in rows.items()
        }
    if dataset == "uci_metro_traffic":
        weighting["retained_effective_hours"] = int(timestamps.iloc[joined].nunique())
        weighting["retained_effective_hours_by_split"] = {
            name: int(timestamps.iloc[index].nunique()) for name, index in rows.items()
        }
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "family": family,
        "primary_target": entry["primary_target"],
        "response_scale": "original",
        "counts_as_real_source": entry.get("counts_as_real_source", False),
        "data_kind": entry.get("data_kind"),
        "max_source_rows": MAX_ROWS,
        "source_registry": None,
        "source_bytes_verified": False,
        "data_sha256": None,
        "declared_data_sha256": entry.get("source", {}).get("sha256"),
        "manifest_entry_sha256": digest_json(entry),
        "split_sha256": _split_hash(rows),
        "preprocessing_sha256": digest_json(state),
        "sample_weight_sha256": hashlib.sha256(weights.astype("<f8").tobytes()).hexdigest(),
        "eligibility": {
            "order": "Assign all source rows, apply missing-target/operating eligibility, then purge boundaries.",
            "source_rows": len(frame),
            "eligible_rows_before_purge": int(eligible.sum()),
            "retained_rows": len(joined),
            "ineligible_rows": int((~eligible).sum()),
            "purged_rows": sum(map(len, purged.values())),
            "ineligible_positions": ineligible,
            "purged_positions": purged,
            "ineligible_rows_by_split": {
                name: int((~eligible[index]).sum()) for name, index in raw_rows.items()
            },
            "purged_rows_by_split": {
                name: int((eligible[index] & ~retained[index]).sum())
                for name, index in raw_rows.items()
            },
            "target_missing_policy": "Reject unless allow_missing is declared; then exclude after partitioning without imputation.",
        },
        "split_audit": split_audit,
        "weighting": weighting,
        "feature_audit": {
            "input_columns": features,
            "categorical_columns": categorical,
            "excluded_columns": entry.get("exclude_columns", {}),
            "training_retained_columns": list(state["features"]),
            "training_dropped_columns": state["dropped"],
        },
    }
    return {
        "entry": entry,
        "frame": frame,
        "state": state,
        "rows": rows,
        "sample_weight": weights,
        "metadata": metadata,
    }


def load_prepared(dataset, data_root=DEFAULT_ROOT):
    """Load one pinned source and return entry/frame/state/rows/weights/metadata."""
    family_for({"id": dataset})
    manifest = KAGGLE_MANIFEST if dataset == "kaggle_king_county_sales" else MANIFEST
    entry = next(entry for entry in read_manifest(manifest) if entry["id"] == dataset)
    if not 0 < entry["schema"]["rows"] <= MAX_ROWS:
        raise ValueError(f"{dataset} exceeds the fixed source row budget")
    frame = load_dataset(dataset, root=data_root, manifest=manifest)
    result = prepare_frame(frame, entry)
    result["metadata"].update(
        source_registry=str(manifest.resolve()),
        source_registry_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        source_path=str(source_path(entry, data_root).resolve()),
        source_bytes_verified=True,
        data_sha256=entry["source"]["sha256"],
        adapter_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_loader_sha256=hashlib.sha256(
            Path(__file__).with_name("interaction_datasets.py").read_bytes()
        ).hexdigest(),
        preprocessing_source_sha256=hashlib.sha256(
            Path(__file__).with_name("benchmark_real_interactions.py").read_bytes()
        ).hexdigest(),
    )
    return result
