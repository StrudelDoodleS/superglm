"""Arithmetic, arm construction and receipt boundaries for the gap table, on synthetic data."""

from __future__ import annotations

import argparse
import hashlib
import json

import benchmark_gap_table as gap
import numpy as np
import pandas as pd
import pytest


def synthetic_frame(rows=480, seed=7):
    generator = np.random.default_rng(seed)
    x1 = generator.uniform(-1.0, 1.0, rows)
    x2 = generator.uniform(-1.0, 1.0, rows)
    noise = generator.normal(0.0, 0.1, rows)
    return pd.DataFrame(
        {
            "id": np.arange(rows),
            "x1": x1,
            "x2": x2,
            "x3": np.where(generator.random(rows) < 0.2, -1.0, generator.uniform(0.0, 5.0, rows)),
            "grade": generator.choice(["low", "mid", "high"], rows),
            "weight": generator.uniform(0.5, 2.0, rows).round(3),
            "y": 1.0 + x1 + 0.7 * x2 + 1.5 * x1 * x2 + noise,
        }
    )


def synthetic_entry(frame, **overrides):
    return {
        "id": "synthetic_case",
        "availability": "fetchable",
        "response_family": "gaussian",
        "primary_target": "y",
        "features": ["x1", "x2", "x3", "grade"],
        "categorical_columns": ["grade"],
        "exclude_columns": {"id": "Row identifier", "weight": "Declared prior weight"},
        "sentinels": {"x3": [-1.0]},
        "split": {"strategy": "group", "columns": ["id"]},
        "known_good_pairs": [
            {"left": "x2", "right": "x3", "grade": "C"},
            {"left": "x1", "right": "x2", "grade": "A"},
            {"left": "x1", "right": "x3", "grade": "B"},
        ],
        "schema": {
            "rows": len(frame),
            "columns": list(frame.columns),
            "missing_counts": {},
        },
        "target_rule": {},
        **overrides,
    }


def registered_dataset(tmp_path, frame, entry):
    """Write the frame and a one-entry manifest the loader will accept."""
    path = tmp_path / "data" / entry["id"] / "source.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = frame.to_csv(index=False).encode()
    path.write_bytes(raw)
    entry["source"] = {
        "url": "https://example.org/synthetic.csv",
        "filename": "source.csv",
        "format": "csv",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "max_bytes": len(raw),
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "datasets": [entry]}))
    return manifest, tmp_path / "data"


def gap_args(**fields):
    return argparse.Namespace(**fields)


def test_closure_reports_the_fraction_of_the_measured_gap():
    assert gap.closure(1.0, 0.75, 0.5) == pytest.approx(0.5)
    assert gap.closure(1.0, 0.5, 0.5) == pytest.approx(1.0)
    assert gap.closure(1.0, 1.2, 0.5) == pytest.approx(-0.4)


def test_closure_is_undefined_when_boosting_finds_no_signal():
    assert gap.closure(1.0, 0.9, 1.0) is None
    assert gap.closure(None, 0.9, 0.5) is None
    assert gap.closure(1.0, None, 0.5) is None


def test_r1_admits_only_datasets_whose_ceiling_beats_the_additive_gbm():
    losses = {"G0": 1.0, "G2": 0.98}
    rules = gap.dataset_rules(
        gap_args(capacities=["leaves15"]),
        {"G2-leaves15": {"valid": {"primary_loss": 0.98}, "fit_wall_seconds": 4.0}},
        losses,
        {},
    )
    assert rules["R1"]["relative_interaction_signal"] == pytest.approx(0.02)
    assert rules["R1"]["enters_closure_summary"] is True
    assert (
        gap.dataset_rules(gap_args(capacities=["leaves15"]), {}, {"G0": 1.0, "G2": 0.995}, {})[
            "R1"
        ]["enters_closure_summary"]
        is False
    )


def test_r4_calls_a_smooth_arm_cheap_against_the_selected_unrestricted_gbm():
    records = {
        "G2-leaves15": {"valid": {"primary_loss": 0.4}, "fit_wall_seconds": 10.0},
        "A1": {"fit_wall_seconds": 45.0},
        "A3": {"fit_wall_seconds": 60.0},
    }
    rules = gap.dataset_rules(
        gap_args(capacities=["leaves15"]), records, {"G0": 1.0, "G2": 0.4}, {}
    )
    assert rules["R4"]["unrestricted_gbm_wall_seconds"] == 10.0
    assert rules["R4"]["arms"]["A1"]["cheap"] is True
    assert rules["R4"]["arms"]["A3"]["cheap"] is False
    assert "G2-leaves15" not in rules["R4"]["arms"]


def test_known_good_pairs_run_strongest_evidence_first():
    entry = synthetic_entry(synthetic_frame())
    assert gap.known_good_pairs(entry) == [["x1", "x2"], ["x1", "x3"], ["x2", "x3"]]
    assert gap.known_good_pairs(entry, limit=2) == [["x1", "x2"], ["x1", "x3"]]


def test_screened_pairs_rank_by_z_and_record_the_refusals():
    table = pd.DataFrame(
        {
            "feature_a": ["a", "c", "e"],
            "feature_b": ["b", "d", "f"],
            "z": [1.5, np.nan, 4.0],
        }
    )
    pairs, refused = gap.screened_pairs(table, 2)
    assert pairs == [["e", "f"], ["a", "b"]]
    assert refused == [["c", "d"]]


def test_arm_plan_lifts_the_feature_cap_only_for_the_all_column_ceiling():
    assert gap.arm_plan("A2-8") == {"engine": "superglm", "feature_cap": gap.FEATURE_CAP}
    assert gap.arm_plan("G1-leaves31")["feature_cap"] == gap.FEATURE_CAP
    assert gap.arm_plan("G2all-leaves15")["feature_cap"] is None
    assert gap.arm_plan("G2-leaves15")["structure"] == "G2"


def test_family_and_weights_come_from_the_entry():
    frame = synthetic_frame()
    entry = synthetic_entry(frame)
    assert gap.family_for(entry)["gbm_loss"] == "squared_error"
    assert gap.family_for({**entry, "response_family": "tweedie:1.5"}) == {
        "name": "tweedie",
        "power": 1.5,
        "gbm_loss": "poisson",
    }
    assert gap.weight_values(frame, entry) is None
    weighted = gap.weight_values(frame, {**entry, "weight_column": "weight"})
    np.testing.assert_allclose(weighted, frame["weight"].to_numpy())


def test_an_exposure_column_turns_the_gbm_target_into_a_rate():
    partition = {
        "response": np.array([0.0, 2.0, 1.0]),
        "weights": None,
        "exposure": np.array([0.5, 2.0, 1.0]),
    }
    target, weights = gap.gbm_training_target(partition)
    np.testing.assert_allclose(target, [0.0, 1.0, 1.0])
    np.testing.assert_allclose(weights, partition["exposure"])
    plain = {"response": np.array([1.0]), "weights": np.array([3.0]), "exposure": None}
    assert gap.gbm_training_target(plain)[1].tolist() == [3.0]


def test_normalised_gini_is_one_for_a_perfect_ordering_and_minus_one_reversed():
    response = np.array([0.0, 1.0, 2.0, 5.0])
    weights = np.ones_like(response)
    assert gap.normalised_gini(response, response, weights) == pytest.approx(1.0)
    assert gap.normalised_gini(response, -response, weights) == pytest.approx(-1.0)


def test_the_feature_cap_keeps_the_declared_rank_then_the_strongest_spearman():
    frame = synthetic_frame()
    entry = {**synthetic_entry(frame), "offset_feature": [], "feature_rank": ["x3"]}
    response = gap.response_values(frame, entry)
    kept, rule = gap.capped_features(frame, response, entry, 3)
    assert kept[0] == "x3"
    assert set(kept) == {"x3", "x1", "x2"}
    assert rule == "declared feature_rank then training-only Spearman"
    everything, rule = gap.capped_features(frame, response, entry, 10)
    assert everything == entry["features"]
    assert rule.startswith("none")


def test_the_feature_cap_falls_back_to_spearman_alone():
    frame = synthetic_frame()
    entry = {**synthetic_entry(frame), "offset_feature": []}
    kept, rule = gap.capped_features(frame, gap.response_values(frame, entry), entry, 2)
    assert kept == ["x1", "x2"]
    assert rule == "training-only Spearman"


def test_the_adapter_maps_sentinels_and_adds_the_companion_and_date_features():
    frame = synthetic_frame()
    frame["stamp"] = pd.date_range("2020-01-01", periods=len(frame), freq="D").astype(str)
    entry = synthetic_entry(frame, time_column="stamp")
    entry["exclude_columns"]["stamp"] = "Split key"
    adapted, notes = gap.adapt(frame, entry)
    assert frame["x3"].isna().sum() > 0
    assert notes["sentinel_columns"] == ["x3"]
    assert notes["missing_companions"] == ["x3__missing"]
    assert "x3__missing" in adapted["categorical_columns"]
    assert notes["date_features"] == ["stamp__year", "stamp__month", "stamp__day_of_week"]
    assert adapted["split"]["seed"] == gap.SEED
    assert set(frame["x3__missing"]) == {"missing", "present"}


def test_a_seconds_since_reference_time_column_yields_a_day_index_not_a_calendar():
    frame = synthetic_frame(rows=40)
    frame["moment"] = np.arange(40) * 3 * 3600
    entry = synthetic_entry(frame, time_column="moment")
    names = gap.add_date_parts(frame, entry)
    assert names == ["moment__day_index", "moment__day_of_week"]
    assert frame["moment__day_index"].max() == 4.0
    assert frame["moment__day_of_week"].tolist()[:8] == [0.0] * 8


def test_a_row_whose_offset_has_no_logarithm_leaves_before_the_split():
    frame = synthetic_frame(rows=20)
    frame["area"] = np.r_[0.0, np.full(19, 10.0)]
    entry = synthetic_entry(frame, offset_column="log(area)")
    kept, dropped = gap.drop_unusable_offset_rows(frame, entry)
    assert dropped == 1
    assert len(kept) == 19
    np.testing.assert_allclose(gap.offset_values(kept, entry), np.log(10.0))


def test_an_arm_drops_a_pair_whose_column_the_cap_removed():
    state = {"features": {"x1": {}, "x2": {}}}
    assert gap.usable_pairs([["x1", "x2"], ["x1", "x9"]], state) == [["x1", "x2"]]


def run_worker(tmp_path, manifest, data_root, arm, monkeypatch):
    monkeypatch.setattr(gap, "MANIFESTS", (manifest,))
    output = tmp_path / "run" / "synthetic_case" / arm
    code = gap.main(
        [
            "--worker",
            "--dataset",
            "synthetic_case",
            "--arm",
            arm,
            "--output",
            str(output),
            "--case-root",
            str(output.parent),
            "--data-root",
            str(data_root),
            "--union-pairs",
            "4",
        ]
    )
    return code, json.loads((output / "receipt.json").read_text())


@pytest.mark.parametrize("arm", ["A0", "G0-leaves15"])
def test_every_fit_leaves_a_complete_receipt(tmp_path, monkeypatch, arm):
    frame = synthetic_frame()
    manifest, data_root = registered_dataset(tmp_path, frame, synthetic_entry(frame))
    code, receipt = run_worker(tmp_path, manifest, data_root, arm, monkeypatch)
    assert code == 0, receipt.get("traceback")
    assert receipt["missing_receipt_fields"] == []
    assert receipt["status"] in {"converged", "not_converged"}
    assert receipt["rows"] == {"train": 288, "valid": 96, "test": 96}
    assert receipt["family"] == "gaussian"
    assert receipt["adapter_rules"]["missing_companions"] == ["x3__missing"]
    assert len(receipt["model_fingerprint"]) == 64
    assert receipt["test"]["primary_loss"] > 0
    assert receipt["pairs"] == []


def test_the_blind_arm_fits_the_pairs_the_screen_ranked(tmp_path, monkeypatch):
    frame = synthetic_frame()
    manifest, data_root = registered_dataset(tmp_path, frame, synthetic_entry(frame))
    _, additive = run_worker(tmp_path, manifest, data_root, "A0", monkeypatch)
    assert additive["screening"]["candidate_count"] > 0
    code, blind = run_worker(tmp_path, manifest, data_root, "A2-4", monkeypatch)
    assert code == 0, blind.get("traceback")
    assert blind["pairs"] == additive["screening"]["pairs"][:4]
    assert blind["pair_source"]["source"] == "top 4 screened pairs by z"
    assert blind["split_sha256"] == additive["split_sha256"]
    assert blind["adapter_sha256"] == additive["adapter_sha256"]


def test_the_known_good_arm_fits_the_catalogue_pairs(tmp_path, monkeypatch):
    frame = synthetic_frame()
    manifest, data_root = registered_dataset(tmp_path, frame, synthetic_entry(frame))
    code, receipt = run_worker(tmp_path, manifest, data_root, "A1", monkeypatch)
    assert code == 0, receipt.get("traceback")
    assert receipt["pairs"] == [["x1", "x2"], ["x1", "x3"], ["x2", "x3"]]
    assert receipt["pair_source"]["source"] == "catalogue known-good pairs"
