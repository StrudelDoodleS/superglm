"""Frozen eligibility, chronology, grouping and training-state boundaries."""

from __future__ import annotations

import copy

import broad_interaction_data as data
import numpy as np
import pandas as pd
import pytest


def time_fixture(dataset, *, days=20, repeats=2):
    column, fmt = data.TIME_FORMATS[dataset]
    dates = pd.date_range("2020-01-25", periods=days, freq="D").repeat(repeats)
    frame = pd.DataFrame(
        {column: dates.strftime(fmt), "x": np.arange(len(dates)), "y": np.arange(len(dates))}
    )
    entry = {
        "id": dataset,
        "primary_target": "y",
        "features": ["x"],
        "categorical_columns": [],
        "exclude_columns": {column: "time key"},
        "split": {"strategy": "time", "columns": [column], "seed": 20260913},
        "target_rule": {"minimum": 0},
    }
    if dataset == "uci_seoul_bike":
        frame["Functioning Day"] = "Yes"
        entry["exclude_columns"]["Functioning Day"] = "operating eligibility"
        entry["row_policy"] = {"after_split": "Functioning Day == Yes"}
    if dataset == "kaggle_king_county_sales":
        frame["id"] = np.arange(len(frame))
        entry["exclude_columns"]["id"] = "property ID"
        entry["split"] = {"strategy": "chronological_grouped", "columns": [column, "id"]}
    if dataset == "uci_metro_traffic":
        # Weather annotations for one timestamp must have one shared count.
        frame["y"] = np.repeat(np.arange(days), repeats)
    return frame, entry


def assert_accounted(prepared):
    audit = prepared["metadata"]["eligibility"]
    blocks = list(prepared["rows"].values())
    blocks.extend(np.asarray(rows) for rows in audit["ineligible_positions"].values())
    blocks.extend(np.asarray(rows) for rows in audit["purged_positions"].values())
    assert np.array_equal(np.sort(np.concatenate(blocks)), np.arange(len(prepared["frame"])))


def test_seoul_dayfirst_calendar_split_precedes_operating_eligibility():
    frame, entry = time_fixture("uci_seoul_bike")
    before = frame.copy(deep=True)
    frame.loc[[1, 8, 25, 33], "Functioning Day"] = "No"
    prepared = data.prepare_frame(frame, entry)
    assert np.array_equal(prepared["rows"]["train"], np.setdiff1d(np.arange(24), [1, 8]))
    assert np.array_equal(prepared["rows"]["valid"], np.setdiff1d(np.arange(24, 32), [25]))
    assert np.array_equal(prepared["rows"]["test"], np.setdiff1d(np.arange(32, 40), [33]))
    assert prepared["metadata"]["eligibility"]["ineligible_positions"] == {
        "closed_hours": [1, 8, 25, 33]
    }
    assert prepared["metadata"]["split_audit"]["raw_rows"] == {
        "train": 24,
        "valid": 8,
        "test": 8,
    }
    pd.testing.assert_frame_equal(
        frame.drop(columns="Functioning Day"), before.drop(columns="Functioning Day")
    )
    assert_accounted(prepared)


def test_chronology_regression_lexical_dayfirst_order_is_wrong():
    frame, entry = time_fixture("uci_seoul_bike")
    prepared = data.prepare_frame(frame, entry)
    # The old generic split sorts raw strings: February 01 precedes January 25.
    from benchmark_real_interactions import partition_rows

    legacy = partition_rows(frame, strategy="chronological_group", columns=["Date"])
    assert not np.array_equal(legacy["train"], prepared["rows"]["train"])
    assert prepared["rows"]["train"].max() < prepared["rows"]["valid"].min()


@pytest.mark.parametrize("dataset", ["uci_appliances", "uci_seoul_bike"])
def test_observed_uci_date_spelling_parses_without_guessing_month_day_order(dataset):
    frame, entry = time_fixture(dataset, repeats=1)
    dates = pd.date_range("2020-01-25", periods=20, freq="D")
    column, _ = data.TIME_FORMATS[dataset]
    if dataset == "uci_appliances":
        frame[column] = dates.strftime("%Y-%m-%d17:00:00")
    else:
        frame[column] = [f"{date.day}/{date.month}/{date.year}" for date in dates]
    prepared = data.prepare_frame(frame, entry)
    assert prepared["metadata"]["split_audit"]["raw_time_ranges"]["train"]["first"].startswith(
        "2020-01-25"
    )
    assert prepared["metadata"]["split_audit"]["raw_time_ranges"]["valid"]["first"].startswith(
        "2020-02-06"
    )
    assert_accounted(prepared)


@pytest.mark.parametrize("dataset", list(data.TIME_FORMATS))
def test_time_parser_rejects_malformed_or_missing_keys(dataset):
    frame, entry = time_fixture(dataset)
    column, _ = data.TIME_FORMATS[dataset]
    frame.loc[2, column] = "2020-99-99"
    with pytest.raises(ValueError, match="timestamp"):
        data.prepare_frame(frame, entry)
    frame.loc[2, column] = None
    with pytest.raises(ValueError, match="timestamp"):
        data.prepare_frame(frame, entry)


def test_appliances_preserves_whole_days_with_one_day_before_each_boundary_purged():
    frame, entry = time_fixture("uci_appliances")
    prepared = data.prepare_frame(frame, entry)
    assert np.array_equal(prepared["rows"]["train"], np.arange(22))
    assert np.array_equal(prepared["rows"]["valid"], np.arange(24, 30))
    assert np.array_equal(prepared["rows"]["test"], np.arange(32, 40))
    assert prepared["metadata"]["eligibility"]["purged_positions"] == {
        "one_day_before_valid": [22, 23],
        "one_day_before_test": [30, 31],
    }
    assert_accounted(prepared)


def test_metro_annotations_preserve_every_row_and_one_unit_of_weight_per_hour():
    frame, entry = time_fixture("uci_metro_traffic", repeats=3)
    prepared = data.prepare_frame(frame, entry)
    assert np.allclose(prepared["sample_weight"], 1 / 3)
    weight_by_hour = pd.Series(prepared["sample_weight"]).groupby(frame["date_time"]).sum()
    assert np.array_equal(weight_by_hour.to_numpy(), np.ones(20))
    assert prepared["metadata"]["weighting"]["source_unique_timestamps"] == 20
    assert prepared["metadata"]["weighting"]["retained_effective_hours"] == 20
    assert_accounted(prepared)
    frame.loc[1, "y"] = 99
    with pytest.raises(ValueError, match="inconsistent.*timestamp"):
        data.prepare_frame(frame, entry)


def test_metro_whole_day_cannot_be_cut_between_distinct_hours():
    frame, entry = time_fixture("uci_metro_traffic", repeats=2)
    odd = np.arange(1, len(frame), 2)
    frame.loc[odd, "date_time"] = (
        pd.to_datetime(frame.loc[odd, "date_time"]) + pd.Timedelta(hours=23)
    ).dt.strftime(data.TIME_FORMATS["uci_metro_traffic"][1])
    prepared = data.prepare_frame(frame, entry)
    assert np.array_equal(prepared["rows"]["train"], np.arange(24))
    assert np.array_equal(prepared["sample_weight"], np.ones(40))


def test_king_county_purges_only_earlier_partition_occurrences_of_repeated_properties():
    frame, entry = time_fixture("kaggle_king_county_sales")
    frame.loc[[0, 24, 32], "id"] = 1000
    frame.loc[[1, 25], "id"] = 1001
    frame.loc[[2, 3], "id"] = 1002
    prepared = data.prepare_frame(frame, entry)
    assert prepared["metadata"]["eligibility"]["purged_positions"] == {
        "property_seen_in_later_partition": [0, 1, 24]
    }
    assert {2, 3} <= set(prepared["rows"]["train"])
    owners = {}
    for name, rows in prepared["rows"].items():
        for group in frame.iloc[rows]["id"]:
            assert owners.setdefault(group, name) == name
    assert_accounted(prepared)


@pytest.mark.parametrize(
    "dataset, group", [("uci_parkinsons", "subject#"), ("uci_superconductivity", "material")]
)
def test_subject_and_material_groups_are_disjoint_and_preprocessing_is_train_only(dataset, group):
    frame = pd.DataFrame(
        {group: np.repeat(np.arange(20), 3), "x": np.arange(60, dtype=float), "y": np.arange(60)}
    )
    entry = {
        "id": dataset,
        "primary_target": "y",
        "features": ["x"],
        "exclude_columns": {group: "group identifier"},
        "split": {"strategy": "group", "columns": [group], "seed": 20260913},
    }
    prepared = data.prepare_frame(frame, entry)
    owners = {}
    for name, rows in prepared["rows"].items():
        for key in frame.iloc[rows][group]:
            assert owners.setdefault(key, name) == name
    changed = frame.copy()
    changed.loc[np.concatenate([prepared["rows"]["valid"], prepared["rows"]["test"]]), "x"] = 1e15
    second = data.prepare_frame(changed, entry)
    assert second["state"] == prepared["state"]
    assert (
        second["metadata"]["preprocessing_sha256"] == prepared["metadata"]["preprocessing_sha256"]
    )
    assert_accounted(prepared)


def test_casp_explicit_source_row_ids_make_a_reproducible_exploratory_partition():
    frame = pd.DataFrame({"x": np.arange(50), "y": np.arange(50)})
    entry = {
        "id": "uci_protein_structure",
        "primary_target": "y",
        "features": ["x"],
        "split": {"strategy": "exploratory_random", "columns": [], "seed": 20260913},
    }
    prepared = data.prepare_frame(frame, entry)
    assert [len(rows) for rows in prepared["rows"].values()] == [30, 10, 10]
    assert prepared["metadata"]["split_audit"]["unit"] == "source_row_position"
    assert (
        prepared["metadata"]["split_sha256"]
        == data.prepare_frame(frame, entry)["metadata"]["split_sha256"]
    )
    assert_accounted(prepared)


def test_allowed_missing_target_is_removed_after_boundaries_and_never_imputed():
    frame, entry = time_fixture("uci_seoul_bike")
    baseline = data.prepare_frame(frame, entry)
    frame.loc[0, "y"] = np.nan
    with pytest.raises(ValueError, match="Missing target"):
        data.prepare_frame(frame, entry)
    entry["target_rule"]["allow_missing"] = True
    prepared = data.prepare_frame(frame, entry)
    assert prepared["metadata"]["eligibility"]["ineligible_positions"] == {"missing_target": [0]}
    for name in ("valid", "test"):
        assert np.array_equal(prepared["rows"][name], baseline["rows"][name])
    assert np.isnan(data.response_values(frame, entry)[0])
    assert_accounted(prepared)


@pytest.mark.parametrize("value", [-1, 0.5, np.inf])
def test_poisson_support_violations_fail_instead_of_deleting_rows(value):
    frame, entry = time_fixture("uci_seoul_bike")
    frame["y"] = frame["y"].astype(float)
    frame.loc[0, "y"] = value
    with pytest.raises(ValueError, match="target|Count|finite"):
        data.prepare_frame(frame, entry)


def test_unknown_operating_flag_refused_without_silent_filtering():
    frame, entry = time_fixture("uci_seoul_bike")
    frame.loc[0, "Functioning Day"] = "Unknown"
    with pytest.raises(ValueError, match="Functioning Day"):
        data.prepare_frame(frame, entry)


@pytest.mark.parametrize("leak", ["y", "Date", "Functioning Day"])
def test_target_related_and_reserved_columns_cannot_enter_default_features(leak):
    frame, entry = time_fixture("uci_seoul_bike")
    entry["features"].append(leak)
    with pytest.raises(ValueError, match="target or excluded"):
        data.prepare_frame(frame, entry)


def test_original_target_scale_and_family_are_frozen_for_all_thirteen():
    assert len(data.DATASETS) == 13
    for dataset in data.DATASETS:
        entry = {"id": dataset, "primary_target": "y"}
        values = np.array([1.0, 10.0, 100.0])
        assert np.array_equal(data.response_values(pd.DataFrame({"y": values}), entry), values)
        assert data.family_for(entry) == (
            "poisson"
            if dataset in {"uci_abalone", "uci_seoul_bike", "uci_metro_traffic"}
            else "gaussian"
        )


def test_unknown_dataset_and_oversized_source_are_refused_before_loading(monkeypatch):
    with pytest.raises(ValueError, match="approved"):
        data.load_prepared("not_in_frozen_batch")
    entry = {"id": "uci_abalone", "schema": {"rows": data.MAX_ROWS + 1}}
    monkeypatch.setattr(data, "read_manifest", lambda path: [entry])
    monkeypatch.setattr(data, "load_dataset", lambda *args, **kwargs: pytest.fail("must not load"))
    with pytest.raises(ValueError, match="row budget"):
        data.load_prepared("uci_abalone")


def test_prepare_frame_preserves_caller_entry_and_does_not_claim_verified_source_bytes():
    frame, entry = time_fixture("uci_seoul_bike")
    entry["source"] = {"sha256": "a" * 64}
    before = copy.deepcopy(entry)
    prepared = data.prepare_frame(frame, entry)
    assert entry == before
    assert prepared["metadata"]["source_bytes_verified"] is False
    assert prepared["metadata"]["data_sha256"] is None
