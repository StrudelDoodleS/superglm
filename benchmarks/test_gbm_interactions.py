"""Contracts for the bounded GBM diagnostic, using small generated fixtures."""

from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import benchmark_gbm_interactions as gbm
import benchmark_real_interactions as original
import numpy as np
import pandas as pd
import pytest


def test_gbm_preserves_existing_partition_preprocessing_and_predictor_exclusions():
    # A second splitter or independent preprocessing would break this identity.
    frame = pd.DataFrame(
        {
            "day": np.repeat(np.arange(10), 4),
            "y": np.arange(40),
            "x": np.arange(40, dtype=float),
            "category": ["b", "a", "b", "a"] * 10,
            "leak": np.arange(40),
        }
    )
    frame.loc[3, "x"] = np.nan
    entry = {
        "primary_target": "y",
        "features": ["x", "category"],
        "categorical_columns": ["category"],
        "exclude_columns": {"leak": "post-outcome"},
        "split": {"strategy": "time", "columns": ["day"], "seed": 42},
    }
    state, rows, fingerprint = gbm.prepare_dataset(frame, entry)
    previous_state, previous_rows, previous_fingerprint = original.prepare_dataset(frame, entry)
    assert fingerprint == previous_fingerprint
    assert state == previous_state
    assert [len(rows[name]) for name in original.SPLITS] == [24, 8, 8]
    for name in original.SPLITS:
        np.testing.assert_array_equal(rows[name], previous_rows[name])
        raw = frame.iloc[rows[name]].loc[:, entry["features"]]
        result = gbm.native_features(raw, state)
        previous = original.transform_features(raw, state)
        assert list(result) == ["x", "category"]
        np.testing.assert_array_equal(result["x"], previous["x"])
        assert result["category"].tolist() == previous["category"].tolist()
    entry["features"].append("leak")
    with pytest.raises(ValueError, match="excluded"):
        gbm.prepare_dataset(frame, entry)


def test_native_categories_use_only_training_levels_and_keep_final_32_level_cap():
    # Inferring category metadata from held-out rows would admit 999.
    labels = np.repeat(np.arange(40), 5)
    train = pd.DataFrame({"code": pd.Categorical(labels, categories=[*range(40), 999])})
    state = original.fit_preprocessor(train, categorical_columns=["code"])
    before = copy.deepcopy(state)
    transformed = gbm.native_features(train, state)
    heldout = gbm.native_features(pd.DataFrame({"code": [999, np.nan, 0]}), state)
    assert isinstance(transformed["code"].dtype, pd.CategoricalDtype)
    assert len(transformed["code"].cat.categories) == 32
    assert not transformed["code"].cat.ordered
    assert heldout["code"].cat.categories.equals(transformed["code"].cat.categories)
    assert heldout["code"].iloc[:2].tolist() == ["pooled:", "pooled:"]
    assert not heldout.isna().any().any()
    assert "999" not in str(state)
    assert state == before
    with pytest.raises(ValueError, match="column"):
        gbm.native_features(pd.DataFrame({"other": [0]}), state)


@pytest.mark.parametrize(
    "family,loss", [("binomial", "log_loss"), ("gaussian", "squared_error"), ("poisson", "poisson")]
)
def test_structural_classes_share_capacity_without_implicit_early_stopping(family, loss):
    for config in gbm.CONFIGS:
        params = [
            gbm.build_model(family, structure, config).get_params() for structure in gbm.STRUCTURES
        ]
        assert [p.pop("interaction_cst") for p in params] == ["no_interactions", "pairwise", None]
        assert params[0] == params[1] == params[2]
        assert params[0]["early_stopping"] is False
        assert params[0]["max_iter"] == 200
        assert params[0]["loss"] == loss
        assert params[0]["categorical_features"] == "from_dtype"


def test_actual_tree_diagnostic_detects_structural_constraint_and_native_categorical_dispatch():
    # Removing interaction_cst would permit the third input on one path.
    from threadpoolctl import threadpool_limits

    rng = np.random.default_rng(314)
    frame = pd.DataFrame(rng.normal(size=(256, 3)), columns=["a", "b", "c"])
    frame["group"] = pd.Categorical(np.tile(["odd", "even"], 128))
    target = (frame["a"] > 0) * (frame["b"] > 0) * (frame["c"] > 0) * 10.0
    observed = {}
    with threadpool_limits(limits=1):
        for structure in gbm.STRUCTURES:
            model = gbm.build_model("gaussian", structure, "leaves15")
            model.fit(frame, target)
            observed[structure] = gbm.tree_diagnostics(model)
    assert observed["additive"]["max_distinct_features_per_branch"] == 1
    assert observed["pairwise"]["max_distinct_features_per_branch"] == 2
    assert observed["unrestricted"]["max_distinct_features_per_branch"] >= 3
    assert all(result["tree_count"] == 200 for result in observed.values())
    assert all(result["native_categorical_count"] == 1 for result in observed.values())


def fit_records():
    losses = {"additive": [0.2, 0.3], "pairwise": [0.5, 0.4], "unrestricted": [0.9, 0.8]}
    return {
        f"{structure}_{config}": {
            "structure": structure,
            "config": config,
            "status": "fitted",
            "validation": {"primary_loss": losses[structure][i]},
            "test": {"primary_loss": -1000 if structure == "unrestricted" else 1000},
        }
        for structure in gbm.STRUCTURES
        for i, config in enumerate(gbm.CONFIGS)
    }


def test_selection_uses_validation_only_and_can_choose_zero_interactions():
    records = fit_records()
    choice = gbm.choose_validation_models(records)
    assert choice["chosen_arm"] == "additive_leaves15"
    assert choice["selected_by_structure"] == {
        "additive": "additive_leaves15",
        "pairwise": "pairwise_leaves31",
        "unrestricted": "unrestricted_leaves31",
    }
    for record in records.values():
        record["test"]["primary_loss"] *= -1
    assert gbm.choose_validation_models(records) == choice
    records["additive_leaves15"]["status"] = "timeout"
    records["additive_leaves31"]["validation"]["primary_loss"] = np.nan
    assert gbm.choose_validation_models(records)["chosen_arm"] == "pairwise_leaves31"


def test_test_evaluation_requires_persisted_choice_and_rejects_unselected_capacity(tmp_path):
    case_root = tmp_path / "case"
    case_root.mkdir()
    args = SimpleNamespace(case_root=case_root, arm="additive_leaves31")
    with pytest.raises(ValueError, match="persisted validation choice"):
        gbm.evaluation_worker(args, {})
    records = fit_records()
    for arm, record in records.items():
        (case_root / arm).mkdir()
        original.write_json(case_root / arm / "result.json", record)
    gbm.persist_choice(case_root, records)
    with pytest.raises(ValueError, match="selected"):
        gbm.evaluation_worker(args, {})


def test_persisted_choice_binds_every_fit_receipt_before_test(tmp_path):
    records = fit_records()
    for arm, record in records.items():
        (tmp_path / arm).mkdir()
        original.write_json(tmp_path / arm / "result.json", record)
    choice = gbm.persist_choice(tmp_path, records)
    assert choice["chosen_before_test_evaluation"] is True
    assert (
        choice["fit_result_sha256"]["additive_leaves15"]
        == hashlib.sha256((tmp_path / "additive_leaves15" / "result.json").read_bytes()).hexdigest()
    )
    records["unrestricted_leaves15"]["validation"]["primary_loss"] = 0.0
    original.write_json(
        tmp_path / "unrestricted_leaves15" / "result.json", records["unrestricted_leaves15"]
    )
    with pytest.raises(ValueError, match="changed"):
        gbm.evaluation_worker(SimpleNamespace(case_root=tmp_path, arm="additive_leaves15"), {})


@pytest.mark.parametrize(
    "flag,value",
    [("--fit-timeout", "121"), ("--total-fit-budget", "601"), ("--evaluation-timeout", "121")],
)
def test_cli_refuses_workload_budget_overruns_before_launch(tmp_path, flag, value):
    with pytest.raises(SystemExit) as error:
        gbm.main(["--output", str(tmp_path / "run"), flag, value])
    assert error.value.code == 2
    assert not (tmp_path / "run").exists()


def test_binary_metric_receipt_includes_class_counts_and_baseline_prevalence():
    scores = gbm.metrics(np.array([0.0, 0.0, 0.0, 1.0]), np.full(4, 0.5), "binomial")
    assert scores["class_counts"] == {"0": 3, "1": 1}
    assert scores["average_precision"] == scores["prevalence"] == 0.25
    assert scores["log_loss"] == pytest.approx(np.log(2))


def test_full_table_row_cap_is_checked_before_dataset_io(tmp_path):
    manifest = tmp_path / "manifest.json"
    original.write_json(
        manifest,
        {"schema_version": 1, "datasets": [{"id": "uci_bike_sharing", "schema": {"rows": 30001}}]},
    )
    with pytest.raises(ValueError, match="row budget"):
        gbm.fit_worker(SimpleNamespace(manifest=manifest, dataset="uci_bike_sharing"), {})


def test_fit_and_gated_evaluation_replay_identical_owned_model_and_adapter(tmp_path):
    from threadpoolctl import threadpool_limits

    # A real pinned toy CSV exercises the loader, fit, persistence and evaluation.
    dataset = "uci_bike_sharing"
    source = tmp_path / "data" / dataset / "toy.csv"
    source.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "day": np.repeat(np.arange(10), 8),
            "count": np.tile(np.arange(8), 10),
            "x": np.tile(np.arange(8), 10),
            "code": np.tile(["a", "b"], 40),
        }
    ).to_csv(source, index=False)
    entry = {
        "id": dataset,
        "availability": "fetchable",
        "source": {
            "filename": "toy.csv",
            "format": "csv",
            "bytes": source.stat().st_size,
            "max_bytes": 10000,
            "sha256": gbm.file_hash(source),
        },
        "schema": {"rows": 80, "columns": ["day", "count", "x", "code"], "missing_counts": {}},
        "primary_target": "count",
        "features": ["x", "code"],
        "categorical_columns": ["code"],
        "target_rule": {"minimum": 0, "integer": True},
        "split": {"strategy": "time", "columns": ["day"], "seed": original.SEED},
    }
    manifest = tmp_path / "manifest.json"
    original.write_json(manifest, {"schema_version": 1, "datasets": [entry]})
    output = tmp_path / "run" / dataset / "pairwise_leaves15"
    output.mkdir(parents=True)
    original.write_json(tmp_path / "run" / "protocol.json", {"source": gbm.source_identity()})
    args = SimpleNamespace(
        dataset=dataset,
        arm="pairwise_leaves15",
        stage="fit",
        output=output,
        case_root=output.parent,
        manifest=manifest,
        data_root=source.parents[1],
    )
    with threadpool_limits(limits=1):
        assert gbm.worker(args) == 0
        fitted = json.loads((output / "result.json").read_text())
        assert "test" not in fitted
        assert not (output / "test_predictions.npz").exists()
        assert fitted["rows"] == {"train": 48, "valid": 16, "test": 16}
        assert fitted["tree_diagnostics"]["native_categorical_count"] == 1
        gbm.persist_choice(output.parent, {args.arm: fitted})
        args.stage = "evaluate"
        assert gbm.worker(args) == 0
    evaluation = json.loads((output / "evaluation.json").read_text())
    assert evaluation["status"] == "evaluated"
    assert evaluation["test"]["rows"] == 16
    with np.load(output / "test_predictions.npz") as predictions:
        np.testing.assert_array_equal(predictions["row_index"], np.arange(64, 80))
        np.testing.assert_array_equal(predictions["response"], np.tile(np.arange(8), 2))
        assert np.isfinite(predictions["prediction"]).all()
