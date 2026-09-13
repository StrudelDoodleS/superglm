"""Leakage and row/column contracts for the real-data research adapter."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import benchmark_real_interactions as trial
import numpy as np
import pandas as pd
import pytest


def test_heldout_extremes_cannot_change_training_imputation_or_feature_choice():
    # Fitting on a concatenation would replace the median and rescue the
    # train-constant column, making both assertions fail.
    train = pd.DataFrame({"amount": [1.0, 3.0, np.nan], "constant": [7, 7, 7]})
    state = trial.fit_preprocessor(train, categorical_columns=[])
    before = copy.deepcopy(state)
    heldout = pd.DataFrame({"amount": [np.nan, 1e12], "constant": [0, 99]})
    transformed = trial.transform_features(heldout, state)
    transformed_training_median = trial.transform_features(
        pd.DataFrame({"amount": [2.0], "constant": [7]}), state
    )
    assert list(transformed) == ["amount"]
    assert transformed.iloc[0, 0] == transformed_training_median.iloc[0, 0]
    assert state == before


def test_declared_categorical_semantics_and_unobserved_dtype_levels_do_not_leak():
    train = pd.DataFrame(
        {"code": pd.Categorical([1, 1, 2, 2], categories=[1, 2, 999]), "value": [0, 1, 2, 3]}
    )
    state = trial.fit_preprocessor(train, categorical_columns=["code"])
    assert state["features"]["code"]["kind"] == "categorical"
    transformed_train = trial.transform_features(train, state)
    heldout = pd.DataFrame({"code": [999, np.nan], "value": [2, 3]})
    transformed_test = trial.transform_features(heldout, state)
    assert set(transformed_test["code"]) <= set(transformed_train["code"])
    assert "999" not in str(state["features"]["code"])


def test_high_cardinality_pooling_preserves_rows_and_caps_observed_levels():
    train = pd.DataFrame({"label": ["common"] * 20 + [f"rare{i}" for i in range(20)]})
    state = trial.fit_preprocessor(train, categorical_columns=["label"], max_categories=4)
    transformed = trial.transform_features(train, state)
    assert len(transformed) == 40
    assert 1 < transformed["label"].nunique() <= 4
    assert transformed["label"].iloc[20:].nunique() == 1


@pytest.mark.parametrize(
    "bad",
    [pd.DataFrame({"other": [1]}), pd.DataFrame({"x": [1], "unexpected": [2]})],
)
def test_transform_rejects_changed_column_contract(bad):
    state = trial.fit_preprocessor(pd.DataFrame({"x": [1, 2, 3]}), categorical_columns=[])
    with pytest.raises(ValueError, match="column"):
        trial.transform_features(bad, state)


def test_nonfinite_numeric_input_is_refused_without_becoming_missing():
    with pytest.raises(ValueError, match="finite"):
        trial.fit_preprocessor(pd.DataFrame({"x": [1.0, np.inf]}), categorical_columns=[])
    state = trial.fit_preprocessor(pd.DataFrame({"x": [1.0, 2.0]}), categorical_columns=[])
    with pytest.raises(ValueError, match="finite"):
        trial.transform_features(pd.DataFrame({"x": [-np.inf]}), state)


def test_temporal_split_preserves_whole_days_and_every_row():
    frame = pd.DataFrame({"day": np.repeat(np.arange(10), 3), "target": np.arange(30)})
    splits = trial.partition_rows(frame, strategy="chronological_group", columns=["day"])
    assert sorted(np.concatenate(list(splits.values())).tolist()) == list(range(30))
    day_sets = [set(frame.iloc[rows]["day"]) for rows in splits.values()]
    assert max(day_sets[0]) < min(day_sets[1])
    assert max(day_sets[1]) < min(day_sets[2])


def test_stratified_group_split_never_places_one_subject_in_two_partitions():
    frame = pd.DataFrame({"id": np.repeat(np.arange(30), 2), "y": np.repeat([0, 1] * 15, 2)})
    splits = trial.partition_rows(frame, strategy="stratified_group", columns=["id"], target="y")
    assert sorted(np.concatenate(list(splits.values())).tolist()) == list(range(60))
    owners = {}
    for name, rows in splits.items():
        assert set(frame.iloc[rows]["y"]) == {0, 1}
        for group in frame.iloc[rows]["id"]:
            assert owners.setdefault(group, name) == name


def test_fixed_year_split_rejects_a_property_seen_in_training_and_test():
    frame = pd.DataFrame({"year": [2006, 2009, 2010], "pid": [4, 5, 4]})
    with pytest.raises(ValueError, match="group"):
        trial.partition_rows(
            frame,
            strategy="fixed_year_group",
            columns=["year", "pid"],
            years={"train": [2006, 2007, 2008], "valid": [2009], "test": [2010]},
        )


def test_centered_product_screen_finds_the_only_residual_signal_with_a_fixed_budget():
    grid = np.linspace(-1, 1, 11)
    left, right = np.meshgrid(grid, grid)
    frame = pd.DataFrame(
        {"x": left.ravel(), "y": right.ravel(), "z": np.tile(np.arange(11) % 2, 11)}
    )
    residual = frame["x"].to_numpy() * frame["y"].to_numpy()
    screen = trial.screen_interactions(
        frame, residual, residual, spline_columns=list(frame), max_features=3, max_pairs=1
    )
    assert screen["pairs"] == [["x", "y"]]
    assert screen["candidate_pair_count"] == 3


def test_validation_choice_refuses_nonconverged_better_scoring_fit():
    records = {
        "additive": {"status": "converged", "validation": {"primary_loss": 2.0}},
        "interactions": {"status": "not_converged", "validation": {"primary_loss": 0.0}},
    }
    assert trial.choose_validation_arm(records) == "additive"
    records["additive"]["status"] = "timeout"
    assert trial.choose_validation_arm(records) is None


def test_validation_choice_has_no_test_score_input():
    records = {
        "additive": {
            "status": "converged",
            "validation": {"primary_loss": 2.0},
            "test": {"primary_loss": 0.0},
        },
        "interactions": {
            "status": "converged",
            "validation": {"primary_loss": 1.0},
            "test": {"primary_loss": 100.0},
        },
    }
    assert trial.choose_validation_arm(records) == "interactions"
    records["additive"]["test"]["primary_loss"] = 1e20
    records["interactions"]["test"]["primary_loss"] = -1e20
    assert trial.choose_validation_arm(records) == "interactions"


def test_manifest_target_and_excluded_predictors_are_refused():
    frame = pd.DataFrame({"day": np.arange(10), "y": np.arange(10), "leak": np.arange(10)})
    entry = {
        "primary_target": "y",
        "features": ["leak"],
        "exclude_columns": {"leak": "post outcome"},
        "split": {"strategy": "time", "columns": ["day"], "seed": 42},
    }
    with pytest.raises(ValueError, match="excluded"):
        trial.prepare_dataset(frame, entry)


def test_binary_scores_include_prevalence_and_refuse_invalid_probabilities():
    y = np.array([0.0, 0.0, 0.0, 1.0])
    scores = trial.score_predictions(y, np.full(4, 0.5), "binomial")
    assert scores["log_loss"] == np.log(2)
    assert scores["average_precision"] == scores["prevalence"] == 0.25
    with pytest.raises(ValueError, match="probabilities"):
        trial.score_predictions(y, np.full(4, 2.0), "binomial")


def test_test_evaluation_requires_a_persisted_validation_choice(tmp_path):
    trial.write_json(tmp_path / "choice.json", {"chosen_arm": None})
    with pytest.raises(ValueError, match="persisted validation choice"):
        trial.evaluation_worker(SimpleNamespace(case_root=tmp_path), {})


def test_small_tensor_keeps_both_main_effects_in_the_joint_model():
    x = np.linspace(-1, 1, 11)
    left, right = np.meshgrid(x, x)
    train = pd.DataFrame({"x": left.ravel(), "z": right.ravel()})
    y = train["x"].to_numpy() + train["x"].to_numpy() * train["z"].to_numpy()
    state = trial.fit_preprocessor(train, categorical_columns=[])
    transformed = trial.transform_features(train, state)
    model = trial.build_model(state, "gaussian", [["x", "z"]])
    model.fit(transformed, y)
    assert model.result.converged
    assert len(model.result.beta) == 19  # Two centered k=6 margins plus a 3-by-3 tensor.
    assert np.isfinite(model.predict(transformed)).all()
    assert {group.feature_name for group in model._groups} >= {"x", "z"}


def test_declared_row_budget_is_checked_before_attempting_to_read_a_large_table(tmp_path):
    manifest = tmp_path / "manifest.json"
    trial.write_json(
        manifest, {"schema_version": 1, "datasets": [{"id": "huge", "schema": {"rows": 1000000}}]}
    )
    with pytest.raises(ValueError, match="row budget"):
        trial.fit_worker(SimpleNamespace(manifest=manifest, dataset="huge", max_rows=100), {})
