"""Resource admission and honest comparison contracts for the broad trial."""

import json
from types import SimpleNamespace

import benchmark_broad_interactions as broad
import numpy as np
import pandas as pd
import pytest


def test_richer_basis_admission_skips_unsupported_or_oversize_pairs():
    state = {
        "features": {
            "x": {"kind": "spline"},
            "z": {"kind": "spline"},
            "linear": {"kind": "numeric"},
            "cat": {"kind": "categorical", "levels": list(range(32))},
            "other": {"kind": "categorical", "levels": list(range(32))},
        }
    }
    pairs = [["cat", "other"], ["x", "linear"], ["x", "z"], ["linear", "cat"]]
    receipt = broad.admit_pairs(state, pairs)
    assert receipt["pairs"] == [["x", "z"], ["linear", "cat"]]
    assert receipt["skipped"][0]["reason"] == "representation_budget"
    assert receipt["skipped"][1]["reason"] == "unsupported_parent_types"
    assert receipt["final_nominal"]["P"] == 129
    assert receipt["final_nominal"]["q"] == 4


def test_weighted_scores_equal_an_unreplicated_hourly_response():
    response = np.array([2.0, 2.0, 8.0])
    prediction = np.array([1.0, 1.0, 4.0])
    weights = np.array([0.5, 0.5, 1.0])
    for family in ("gaussian", "poisson"):
        weighted = broad.score(response, prediction, family, weights)
        hourly = broad.score(response[[0, 2]], prediction[[0, 2]], family, np.ones(2))
        assert weighted["primary_loss"] == hourly["primary_loss"]
        assert weighted["weight_sum"] == 2
        assert weighted["rows"] == 3


def test_nominal_admission_dimensions_match_compiled_mixed_and_tensor_models():
    state = {
        "features": {
            "x": {"kind": "spline"},
            "z": {"kind": "spline"},
            "linear": {"kind": "numeric"},
            "cat": {"kind": "categorical", "levels": ["a", "b", "c"]},
        }
    }
    rng = np.random.default_rng(781)
    frame = pd.DataFrame(rng.normal(size=(240, 3)), columns=["x", "z", "linear"])
    frame["cat"] = np.tile(["a", "b", "c"], 80)
    pairs = [["x", "z"], ["linear", "cat"], ["x", "cat"]]
    for k in (4, 6):
        nominal = broad.nominal_size(state, k, pairs)
        model = broad.build_model(state, "gaussian", k, pairs)
        model._build_design_matrix(frame, np.ones(len(frame)), np.ones(len(frame)), None)
        assert model._dm.p == nominal["P"]
        assert nominal["q"] == 6


def test_validation_selection_uses_best_additive_and_keeps_matched_control(tmp_path):
    records = {
        "k4_s0": {"status": "converged", "validation": {"primary_loss": 2.0}, "P": 8},
        "k6_s0": {"status": "converged", "validation": {"primary_loss": 3.0}, "P": 12},
        "k6_s2": {"status": "converged", "validation": {"primary_loss": 1.0}, "P": 60},
        "k4_s1": {"status": "not_converged", "validation": {"primary_loss": 0.0}, "P": 17},
    }
    for name, record in records.items():
        (tmp_path / name).mkdir()
        (tmp_path / name / "result.json").write_text(json.dumps(record))
    choice = broad.persist_choice(tmp_path, records)
    assert choice["chosen_arm"] == "k6_s2"
    assert choice["additive_arm"] == "k4_s0"
    assert choice["matching_additive_arm"] == "k6_s0"
    assert choice["evaluation_arms"] == ["k4_s0", "k6_s0", "k6_s2"]


def test_no_test_evaluation_without_a_converged_additive_comparator(tmp_path):
    records = {
        "k4_s0": {"status": "timeout"},
        "k6_s0": {"status": "not_converged"},
        "k4_s1": {"status": "converged", "validation": {"primary_loss": 1.0}, "P": 17},
    }
    (tmp_path / "k4_s1").mkdir()
    (tmp_path / "k4_s1" / "result.json").write_text(json.dumps(records["k4_s1"]))
    choice = broad.persist_choice(tmp_path, records)
    assert choice["additive_arm"] is None
    assert choice["evaluation_arms"] == []


def test_failed_matching_resolution_does_not_authorize_an_interaction_claim(tmp_path):
    records = {
        "k4_s0": {"status": "converged", "validation": {"primary_loss": 2.0}, "P": 8},
        "k6_s0": {"status": "not_converged"},
        "k6_s2": {"status": "converged", "validation": {"primary_loss": 1.0}, "P": 60},
    }
    for name, record in records.items():
        if record["status"] == "converged":
            (tmp_path / name).mkdir()
            (tmp_path / name / "result.json").write_text(json.dumps(record))
    choice = broad.persist_choice(tmp_path, records)
    assert choice["matching_comparison_status"] == "unavailable"
    assert not choice["interaction_comparison_eligible"]


def test_evaluation_refuses_appended_unselected_arm_before_loading_data(tmp_path):
    records = {
        "k4_s0": {"status": "converged", "validation": {"primary_loss": 2.0}, "P": 8},
        "k4_s1": {"status": "converged", "validation": {"primary_loss": 3.0}, "P": 17},
    }
    for name, record in records.items():
        (tmp_path / name).mkdir()
        (tmp_path / name / "result.json").write_text(json.dumps(record))
    choice = broad.persist_choice(tmp_path, records)
    choice["evaluation_arms"].append("k4_s1")
    (tmp_path / "choice.json").write_text(json.dumps(choice))
    with pytest.raises(ValueError, match="evaluation plan"):
        broad.evaluation_worker(SimpleNamespace(case_root=tmp_path, arm="k4_s1"), {})


def test_test_workers_are_limited_by_remaining_total_and_case_budget():
    args = SimpleNamespace(total_budget=1800, case_budget=240)
    assert broad.remaining_budget(args, 1799, 239.5, "evaluate") == 0.5
    assert broad.remaining_budget(args, 1800, 1, "evaluate") == 0
    # Search reserves 20% of the global budget and 25% of each case for test workers.
    assert broad.remaining_budget(args, 1440, 1, "fit") == 0
    assert broad.remaining_budget(args, 0, 180, "propose") == 0


def test_zero_smoothing_model_uses_coefficient_convergence_without_fictitious_reml():
    state = {"features": {"x": {"kind": "numeric"}, "z": {"kind": "numeric"}}}
    x = np.tile(np.arange(4, dtype=float), 20)
    z = np.repeat(np.arange(4, dtype=float), 20)
    frame = pd.DataFrame({"x": x, "z": z})
    y = x + z + x * z + np.random.default_rng(29).normal(size=80)
    model = broad.build_model(state, "gaussian", 4, [["x", "z"]])
    model.fit_reml(frame, y, max_reml_iter=100)
    telemetry = model.training_telemetry()
    assert not telemetry["reml"]["enabled"]
    status = broad.convergence_status(telemetry, nominal_q=0)
    assert status["combined_converged"]
    assert status["reml_required"] is False
    assert broad.admit_pairs(state, [["x", "z"]])["parent_resolutions"] == [4]
    with pytest.raises(ValueError, match="Missing REML"):
        broad.convergence_status(telemetry, nominal_q=1)


def test_generated_worker_pipeline_proposes_fits_selects_and_replays(tmp_path, monkeypatch):
    import broad_interaction_data as data

    rng = np.random.default_rng(602)
    frame = pd.DataFrame(rng.uniform(-1, 1, (256, 2)), columns=["x", "z"])
    frame["y"] = 3 * frame.x + 2 * frame.z + 4 * frame.x * frame.z + rng.normal(0, 0.2, 256)
    rows = {"train": np.arange(160), "valid": np.arange(160, 208), "test": np.arange(208, 256)}
    state = broad.base.fit_preprocessor(
        frame.iloc[rows["train"]][["x", "z"]], categorical_columns=[]
    )
    entry = {
        "id": "uci_power_plant",
        "primary_target": "y",
        "features": ["x", "z"],
        "target_rule": {},
    }
    prepared = {
        "frame": frame,
        "entry": entry,
        "state": state,
        "rows": rows,
        "sample_weight": np.ones(256),
        "metadata": {"family": "gaussian", "fixture": True},
    }
    monkeypatch.setattr(data, "load_prepared", lambda *args, **kwargs: prepared)
    case_root = tmp_path / "uci_power_plant"
    case_root.mkdir()
    broad.base.write_json(tmp_path / "protocol.json", {"source": broad.source_identity()})
    records = {}
    for stage, arm in (("propose", "proposals"), ("fit", "k4_s0"), ("fit", "k4_s1")):
        output = case_root / arm
        output.mkdir()
        args = SimpleNamespace(
            dataset="uci_power_plant",
            data_root=tmp_path,
            case_root=case_root,
            output=output,
            stage=stage,
            arm=arm,
        )
        returncode = broad.worker(args)
        record = json.loads((output / "result.json").read_text())
        if arm == "k4_s1":
            # This bilinear signal lies in the spline penalty nullspace. REML
            # can reach its cap while sending redundant curvature penalties
            # to a boundary. Such a fit must remain ineligible, never be
            # relabelled as converged just to exercise the evaluation path.
            assert record["status"] in ("converged", "not_converged"), record.get("error")
            assert returncode == (0 if record["status"] == "converged" else 1)
            if record["status"] == "not_converged":
                assert "validation" not in record
                assert not (output / "model.pkl").exists()
        else:
            assert returncode == 0, (record["status"], record.get("error"))
        if stage == "fit":
            records[arm] = record
    choice = broad.persist_choice(case_root, records)
    expected = "k4_s1" if records["k4_s1"]["status"] == "converged" else "k4_s0"
    assert choice["chosen_arm"] == expected
    args.stage = "evaluate"
    args.arm = expected
    args.output = case_root / expected
    assert broad.worker(args) == 0, (args.output / "evaluation.json").read_text()
    audit = json.loads((args.output / "evaluation.json").read_text())
    archive = np.load(args.output / "test_predictions.npz")
    assert np.array_equal(archive["row_index"], rows["test"])
    assert (
        broad.score(
            archive["response"], archive["prediction"], "gaussian", archive["sample_weight"]
        )
        == audit["test"]
    )
