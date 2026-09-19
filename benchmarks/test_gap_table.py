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


def test_the_known_good_arm_drops_absent_pairs_before_it_takes_the_limit(monkeypatch):
    case = {"entry": synthetic_entry(synthetic_frame()), "smooth_features": ["x2", "x3", "grade"]}
    monkeypatch.setattr(gap, "PAIR_LIMIT", 2)
    requested, pairs, source = gap.arm_pairs(gap_args(arm="A1"), case)
    assert requested == [["x1", "x2"], ["x1", "x3"], ["x2", "x3"]]
    # Truncating first would spend both slots on the two pairs x1 cannot supply.
    assert pairs == [["x2", "x3"]]
    assert source == {"source": "catalogue known-good pairs", "limit": 2}
    everything = {"entry": case["entry"], "smooth_features": ["x1", "x2", "x3"]}
    assert gap.arm_pairs(gap_args(arm="A1"), everything)[1] == [["x1", "x2"], ["x1", "x3"]]


def test_the_union_arm_counts_a_symmetric_pair_once(tmp_path):
    case = {"entry": synthetic_entry(synthetic_frame()), "smooth_features": ["x1", "x2", "x3"]}
    (tmp_path / "A0").mkdir()
    (tmp_path / "A0" / "receipt.json").write_text(
        json.dumps({"screening": {"pairs": [["x2", "x1"], ["x2", "x3"]]}})
    )
    args = gap_args(arm="A3", case_root=tmp_path, union_pairs=2, dataset="synthetic_case")
    _, pairs, source = gap.arm_pairs(args, case)
    assert pairs == [["x1", "x2"], ["x1", "x3"], ["x2", "x3"]]
    assert source["source"] == "A1 pairs plus the top 2 screened pairs"


def test_a_blind_arm_refuses_to_run_without_the_baseline_screen(tmp_path):
    (tmp_path / "A0").mkdir()
    (tmp_path / "A0" / "receipt.json").write_text(json.dumps({"status": "timeout"}))
    args = gap_args(arm="A2-4", case_root=tmp_path, union_pairs=4, dataset="synthetic_case")
    with pytest.raises(ValueError, match="recorded no screen"):
        gap.arm_pairs(args, {"entry": {}, "smooth_features": []})


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


def test_the_screen_sweeps_only_the_pairs_of_the_leading_features():
    rng = np.random.default_rng(3)
    design = pd.DataFrame({f"x{i}": rng.normal(size=600) for i in range(25)})
    response = 3.0 * design["x24"] + 2.0 * design["x23"] + design["x22"] + rng.normal(size=600)
    train = {"design": design, "response": response.to_numpy()}
    candidates = gap.screen_candidates(train, list(design.columns))
    assert len(candidates) == gap.SCREEN_TOP * (gap.SCREEN_TOP - 1) // 2
    leading = {name for pair in candidates for name in pair}
    assert {"x24", "x23", "x22"} <= leading
    assert len(leading) == gap.SCREEN_TOP
    assert gap.screen_candidates(train, list(design.columns)[: gap.SCREEN_TOP]) is None


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
    entry = {
        **synthetic_entry(frame),
        "offset_feature": [],
        "feature_rank": ["grade"],
        "known_good_pairs": [],
    }
    response = gap.response_values(frame, entry)
    kept, rule = gap.capped_features(frame, response, entry, 3)
    assert kept[0] == "grade"
    assert set(kept) == {"grade", "x1", "x2"}
    assert rule == "declared feature_rank then training-only Spearman"
    everything, rule = gap.capped_features(frame, response, entry, 10)
    assert everything == entry["features"]
    assert rule.startswith("none")


def test_the_feature_cap_falls_back_to_spearman_alone():
    frame = synthetic_frame()
    entry = {**synthetic_entry(frame), "offset_feature": [], "known_good_pairs": []}
    kept, rule = gap.capped_features(frame, gap.response_values(frame, entry), entry, 2)
    assert kept == ["x1", "x2"]
    assert rule == "training-only Spearman"


def test_the_feature_cap_never_deletes_a_catalogue_pair_column():
    frame = synthetic_frame()
    entry = {**synthetic_entry(frame), "offset_feature": []}
    kept, rule = gap.capped_features(frame, gap.response_values(frame, entry), entry, 3)
    # x3 is the weakest predictor here, and two of the three catalogue pairs need it.
    assert set(kept) == {"x1", "x2", "x3"}
    assert rule == "catalogue pair columns then training-only Spearman"


def test_a_prose_feature_rank_is_refused_rather_than_read_letter_by_letter():
    entry = {**synthetic_entry(synthetic_frame()), "feature_rank": "training_spearman"}
    with pytest.raises(ValueError, match="feature_rank as str"):
        gap.declared_rank(entry)
    assert gap.declared_rank({**entry, "feature_rank": None}) == []


def test_the_adapter_maps_sentinels_and_adds_the_companion_and_date_features():
    frame = synthetic_frame()
    frame["stamp"] = pd.date_range("2020-01-01", periods=len(frame), freq="D").astype(str)
    entry = synthetic_entry(frame, time_column="stamp")
    entry["exclude_columns"]["stamp"] = "Split key"
    adapted_frame, adapted, notes = gap.adapt(frame, entry)
    assert adapted_frame["x3"].isna().sum() > 0
    assert notes["sentinel_columns"] == ["x3"]
    assert notes["missing_companions"] == ["x3__missing"]
    assert "x3__missing" in adapted["categorical_columns"]
    assert notes["date_features"] == ["stamp__year", "stamp__month", "stamp__day_of_week"]
    assert adapted["split"]["seed"] == gap.SEED
    assert set(adapted_frame["x3__missing"]) == {"missing", "present"}


def test_a_declared_categorical_gets_no_companion_it_would_be_collinear_with():
    frame = synthetic_frame()
    frame.loc[: len(frame) // 2, "grade"] = np.nan
    entry = synthetic_entry(frame)
    assert gap.missing_companions(frame, entry) == {}


def test_a_seconds_since_reference_time_column_yields_a_day_index_not_a_calendar():
    frame = synthetic_frame(rows=40)
    frame["moment"] = np.arange(40) * 3 * 3600
    entry = synthetic_entry(frame, time_column="moment")
    parts = gap.date_parts(frame, entry)
    assert list(parts) == ["moment__day_index", "moment__day_of_week"]
    assert parts["moment__day_index"].max() == 4.0
    assert parts["moment__day_of_week"].tolist()[:8] == [0.0] * 8


def test_a_row_whose_offset_has_no_logarithm_leaves_before_the_split():
    frame = synthetic_frame(rows=20)
    frame["area"] = np.r_[0.0, np.full(19, 10.0)]
    entry = synthetic_entry(frame, offset_column="log(area)")
    kept, dropped = gap.drop_unusable_offset_rows(frame, entry)
    assert dropped == 1
    assert len(kept) == 19
    np.testing.assert_allclose(gap.offset_values(kept, entry), np.log(10.0))


def test_an_arm_drops_a_pair_whose_column_the_cap_removed():
    assert gap.usable_pairs([["x1", "x2"], ["x1", "x9"]], ["x1", "x2"]) == [["x1", "x2"]]


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
    assert receipt["valid"]["rows"] == 96 and receipt["test"]["rows"] == 96
    assert receipt["family"] == "gaussian"
    assert receipt["adapter_rules"]["missing_companions"] == ["x3__missing"]
    assert len(receipt["model_fingerprint"]) == 64
    assert receipt["test"]["primary_loss"] > 0
    assert receipt["pairs"] == []


def test_the_baseline_receipt_is_written_and_scored_before_the_screen_runs(tmp_path, monkeypatch):
    frame = synthetic_frame()
    manifest, data_root = registered_dataset(tmp_path, frame, synthetic_entry(frame))
    output = tmp_path / "run" / "synthetic_case" / "A0"
    seen = {}

    def screen_after_the_receipt(model, case, train, record):
        seen.update(json.loads((output / "receipt.json").read_text()))
        record["screening"] = {"pairs": [], "candidate_count": 0}

    monkeypatch.setattr(gap, "run_screen", screen_after_the_receipt)
    _, receipt = run_worker(tmp_path, manifest, data_root, "A0", monkeypatch)
    # A kill during the screen must still leave a scored, complete baseline.
    assert seen["status"] == "converged"
    assert seen["missing_receipt_fields"] == []
    assert seen["test"]["primary_loss"] == receipt["test"]["primary_loss"]


def test_the_offset_is_a_boosting_column_but_never_a_free_smooth_term(tmp_path, monkeypatch):
    frame = synthetic_frame()
    frame["area"] = np.linspace(20.0, 120.0, len(frame))
    # A purely additive response, so only the offset can explain log(area).
    noise = np.random.default_rng(11).normal(0.0, 0.1, len(frame))
    frame["y"] = np.log(frame["area"]) + 0.5 * frame["x1"] + noise
    entry = synthetic_entry(frame, offset_column="log(area)")
    entry["exclude_columns"]["area"] = "Offset denominator, never a free predictor"
    manifest, data_root = registered_dataset(tmp_path, frame, entry)
    monkeypatch.setattr(gap, "MANIFESTS", (manifest,))
    case = gap.prepare_case(
        "synthetic_case", gap_args(data_root=data_root, max_rows=10**6), gap.FEATURE_CAP
    )
    name = "offset__log(area)"
    assert name in case["kept_features"] and name not in case["smooth_features"]
    assert name in gap.case_partition(case, "train", native=True)["design"].columns
    assert name not in gap.case_partition(case, "train", native=False)["design"].columns
    _, receipt = run_worker(tmp_path, manifest, data_root, "A0", monkeypatch)
    assert receipt["smooth_features"] == case["smooth_features"]
    # area is excluded and its column is no term, so an ignored offset would leave
    # the whole variance of log(area) (about 0.25) in the residual.
    assert receipt["test"]["primary_loss"] < 0.05


def test_a_declared_weight_reaches_the_smooth_fit(tmp_path, monkeypatch):
    frame = synthetic_frame()
    manifest, data_root = registered_dataset(tmp_path, frame, synthetic_entry(frame))
    _, unweighted = run_worker(tmp_path, manifest, data_root, "A0", monkeypatch)
    weighted_manifest, weighted_root = registered_dataset(
        tmp_path / "weighted", frame, synthetic_entry(frame, weight_column="weight")
    )
    _, weighted = run_worker(
        tmp_path / "weighted", weighted_manifest, weighted_root, "A0", monkeypatch
    )
    assert weighted["status"] == "converged", weighted.get("traceback")
    assert weighted["adapter_rules"]["weight_column"] == "weight"
    assert weighted["model_fingerprint"] != unweighted["model_fingerprint"]


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


def summary_case(fractions, *, has_list, admitted=True):
    return {
        "closure": fractions,
        "has_known_good_list": has_list,
        "rules": {"R1": {"enters_closure_summary": admitted}},
    }


def test_suite_rules_grade_the_admitted_datasets_and_name_the_undecided():
    cases = {
        "strong": summary_case({"A1": 0.9, "A2-8": 0.8}, has_list=True),
        "weak": summary_case({"A1": 0.2, "A2-8": 0.05}, has_list=True),
        "lost_ground": summary_case({"A1": -0.3, "A2-8": -0.1}, has_list=True),
        "unfitted": summary_case({"A1": None, "A2-8": None}, has_list=True),
        "blind_only": summary_case({"A2-8": 0.6}, has_list=False),
        "no_signal": summary_case({"A1": 0.9, "A2-8": 0.9}, has_list=True, admitted=False),
    }
    rules = gap.suite_rules(gap_args(union_pairs=8), cases)
    assert "no_signal" not in rules["R1_admitted_datasets"]
    assert rules["R2"]["passing"] == 1 and rules["R2"]["eligible"] == 4
    assert rules["R2"]["holds"] is False
    assert rules["R2"]["undecided"] == ["unfitted"]
    assert rules["R3"]["by_dataset"] == {
        "strong": True,
        "weak": False,
        "lost_ground": True,
        "unfitted": None,
        "blind_only": True,
    }
    assert rules["R3"]["holds"] is False
    assert rules["R3"]["undecided"] == ["unfitted"]
    # A negative known-good closure makes 0.8 x closure(A1) a bar nothing can fail.
    assert rules["R3"]["vacuous_known_good_comparison"] == ["lost_ground"]


def test_the_closure_is_keyed_to_the_validation_selected_boosting_ceiling():
    records = {
        "A0": {"valid": {"primary_loss": 1.0}, "test": {"primary_loss": 1.0}},
        "A1": {"valid": {"primary_loss": 0.85}, "test": {"primary_loss": 0.8}},
        "G0-leaves15": {"valid": {"primary_loss": 0.95}, "test": {"primary_loss": 0.9}},
        "G2-leaves15": {"valid": {"primary_loss": 0.70}, "test": {"primary_loss": 0.6}},
        "G2-leaves31": {"valid": {"primary_loss": 0.90}, "test": {"primary_loss": 0.2}},
    }
    args = gap_args(capacities=["leaves15", "leaves31"])
    summary = gap.dataset_summary(args, records, {"known_good_pairs": []})
    assert summary["selected_gbm_capacity"]["G2"] == "G2-leaves15"
    assert summary["test_loss"]["G2"] == 0.6
    assert summary["closure"]["A1"] == pytest.approx(0.5)
    assert summary["interaction_signal"] == pytest.approx(0.3)
    assert summary["representation_gap_at_zero_interactions"] == pytest.approx(0.1)
    assert summary["has_known_good_list"] is False


def test_the_arm_menu_runs_the_baseline_first_and_the_union_after_the_screen():
    args = gap_args(pairs=[4, 8], capacities=["leaves15"])
    assert gap.arm_menu(args, {"known_good_pairs": [{"left": "a", "right": "b"}]}) == [
        "A0",
        "A1",
        "A2-4",
        "A2-8",
        "A3",
        "G0-leaves15",
        "G1-leaves15",
        "G2-leaves15",
        "G2all-leaves15",
    ]
    assert gap.arm_menu(args, {})[:3] == ["A0", "A2-4", "A2-8"]
    assert "A3" not in gap.arm_menu(args, {})
