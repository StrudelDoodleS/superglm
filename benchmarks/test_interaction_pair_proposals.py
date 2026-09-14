"""Pair identity, counting and training-boundary contracts on generated data."""

from __future__ import annotations

import copy
import inspect
import json

import benchmark_gbm_interactions as gbm
import benchmark_real_interactions as original
import interaction_pair_proposals as proposals
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble._hist_gradient_boosting.common import PREDICTOR_RECORD_DTYPE
from sklearn.ensemble._hist_gradient_boosting.predictor import TreePredictor
from threadpoolctl import threadpool_limits


@pytest.fixture(scope="module")
def fitted_template():
    frame = pd.DataFrame(
        {
            "numeric_z": np.linspace(-1, 1, 80),
            "category|with separator": pd.Categorical(np.tile(["odd", "even"], 40)),
            "numeric_a": np.tile(np.arange(8), 10),
        }
    )
    with threadpool_limits(limits=1):
        model = gbm.build_model("gaussian", "pairwise", "leaves15").fit(
            frame, frame["numeric_z"] * frame["numeric_a"]
        )
    return model, list(frame.columns)


def chain_tree(features, gains):
    """A left-descending chain with a right leaf at each internal split."""
    nodes = np.zeros(2 * len(features) + 1, dtype=PREDICTOR_RECORD_DTYPE)
    nodes["is_leaf"] = True
    nodes["gain"] = -1
    for i, (feature, gain) in enumerate(zip(features, gains, strict=True)):
        nodes[i]["is_leaf"] = False
        nodes[i]["feature_idx"] = feature
        nodes[i]["gain"] = gain
        nodes[i]["left"] = i + 1
        nodes[i]["right"] = len(features) + 1 + i
    return TreePredictor(
        nodes, np.zeros((0, 8), dtype=np.uint32), np.zeros((0, 8), dtype=np.uint32)
    )


def with_trees(template, *trees):
    model = copy.copy(template)
    model._predictors = [[tree] for tree in trees]
    return model


def test_scores_restore_original_names_after_native_categorical_reordering(fitted_template):
    model, columns = fitted_template
    # Internal order is category, numeric_z, numeric_a. Ignoring the reorder
    # would incorrectly label the pair as numeric_z-by-numeric_a.
    tree = chain_tree([0, 2, 0], [100.0, 7.0, 3.0])
    result = proposals.score_fitted_pairs(with_trees(model, tree), columns)
    assert result["internal_feature_order"] == [columns[1], columns[0], columns[2]]
    assert result["pairs"] == [(columns[1], columns[2])]
    assert result["scores"] == [{"pair": (columns[1], columns[2]), "score": 10.0, "split_nodes": 2}]
    assert json.loads(json.dumps(result, allow_nan=False))["pairs"] == [[columns[1], columns[2]]]


def test_exact_counting_clips_negative_gain_and_orders_ties_by_original_positions(fitted_template):
    model, columns = fitted_template
    first = chain_tree([0, 1, 0], [100.0, 10.0, -5.0])
    second = chain_tree([0, 2], [200.0, 10.0])
    result = proposals.score_fitted_pairs(with_trees(model, second, first), columns)
    assert result["pairs"] == [(columns[0], columns[1]), (columns[1], columns[2])]
    assert [record["score"] for record in result["scores"]] == [10.0, 10.0]
    assert [record["split_nodes"] for record in result["scores"]] == [2, 1]
    replay = proposals.score_fitted_pairs(with_trees(model, first, second), columns)
    assert replay == result


def test_one_feature_branches_do_not_propose_pairs(fitted_template):
    model, columns = fitted_template
    result = proposals.score_fitted_pairs(
        with_trees(model, chain_tree([0, 0], [100.0, 10.0])), columns
    )
    assert result["pairs"] == result["scores"] == []


def test_higher_order_path_is_refused_instead_of_decomposed_into_false_pairs(fitted_template):
    model, columns = fitted_template
    with pytest.raises(ValueError, match="two"):
        proposals.score_fitted_pairs(
            with_trees(model, chain_tree([0, 1, 2], [10.0, 5.0, 3.0])), columns
        )


@pytest.mark.parametrize("gain", [np.nan, np.inf, -np.inf])
def test_nonfinite_split_gain_is_refused(fitted_template, gain):
    model, columns = fitted_template
    with pytest.raises(ValueError, match="finite"):
        proposals.score_fitted_pairs(with_trees(model, chain_tree([0, 1], [1.0, gain])), columns)


def test_fitted_column_contract_rejects_wrong_input_order(fitted_template):
    model, columns = fitted_template
    with pytest.raises(ValueError, match="column"):
        proposals.score_fitted_pairs(model, columns[::-1])


@pytest.mark.parametrize(
    "family,loss", [("binomial", "log_loss"), ("gaussian", "squared_error"), ("poisson", "poisson")]
)
def test_discovery_uses_only_supplied_training_rows_and_fixed_pairwise_fit(family, loss):
    frame = pd.DataFrame(
        {
            "amount": np.tile(np.arange(8), 20),
            "label": pd.Categorical(np.repeat(["a", "b"], 80), categories=["a", "b", "heldout"]),
            "constant": np.ones(160),
        }
    )
    state = original.fit_preprocessor(frame, categorical_columns=["label"])
    before = copy.deepcopy(state)
    response = ((frame["amount"] > 3) & (frame["label"] == "a")).to_numpy(dtype=float)
    signature = inspect.signature(proposals.discover_pairs)
    assert list(signature.parameters) == [
        "raw_train_frame",
        "state",
        "response",
        "family",
        "sample_weight",
    ]
    with threadpool_limits(limits=1):
        receipt = proposals.discover_pairs(frame, state, response, family)
    assert receipt["training_rows"] == 160
    assert receipt["training_total_weight"] == 160
    assert receipt["sample_weight_provided"] is False
    assert receipt["input_columns"] == ["amount", "label"]
    assert receipt["native_categorical_levels"] == {"label": ["v:a", "v:b"]}
    assert receipt["pairs"] == [("amount", "label")]
    assert state == before
    assert receipt["model_parameters"]["loss"] == loss
    assert receipt["model_parameters"]["interaction_cst"] == "pairwise"
    assert receipt["model_parameters"]["max_iter"] == receipt["tree_diagnostics"]["n_iter"] == 200
    assert receipt["model_parameters"]["max_leaf_nodes"] == 15
    assert receipt["model_parameters"]["early_stopping"] is False
    assert receipt["tree_diagnostics"]["native_categorical_count"] == 1
    assert receipt["retained_model_storage"]["total_payload_bytes"] > 0
    assert receipt["fit_end_peak_process_rss_mib"] > 0
    timing = receipt["timing"]
    assert timing["fit_seconds"] > 0
    assert timing["total_seconds"] >= sum(
        timing[name] for name in ("native_preparation_seconds", "fit_seconds", "extraction_seconds")
    )
    json.dumps(receipt, allow_nan=False)


@pytest.mark.parametrize("response", [np.array([1.0]), np.array([1.0, np.nan]), np.ones((2, 1))])
def test_invalid_training_response_is_refused_before_model_fit(response):
    frame = pd.DataFrame({"x": [0.0, 1.0]})
    state = original.fit_preprocessor(frame, categorical_columns=[])
    with pytest.raises(ValueError, match="response"):
        proposals.discover_pairs(frame, state, response, "gaussian")


@pytest.mark.parametrize(
    "weights",
    [[1.0], [[1.0], [1.0]], [0.0, 0.0], [-1.0, 2.0], [np.inf, 1.0], [np.nan, 1.0], [1e308, 1e308]],
)
def test_invalid_weights_are_refused_before_model_fit(weights):
    frame = pd.DataFrame({"x": [0.0, 1.0]})
    state = original.fit_preprocessor(frame, categorical_columns=[])
    with pytest.raises(ValueError, match="weight"):
        proposals.discover_pairs(frame, state, np.ones(2), "gaussian", sample_weight=weights)


def test_discovery_forwards_the_original_weight_object_to_the_real_estimator(monkeypatch):
    frame = pd.DataFrame({"x": np.linspace(-1, 1, 80), "z": np.tile(np.arange(8), 10)})
    state = original.fit_preprocessor(frame, categorical_columns=[])
    response = frame["x"].to_numpy() * frame["z"].to_numpy()
    weights = np.tile([0.5, 0.5, 1.0, 1.0], 20)
    model = gbm.build_model("gaussian", "pairwise", "leaves15")
    original_fit = model.fit
    observed = []

    def fit(features, target, *, sample_weight=None):
        observed.append(sample_weight)
        return original_fit(features, target, sample_weight=sample_weight)

    monkeypatch.setattr(model, "fit", fit)
    monkeypatch.setattr(proposals, "build_model", lambda *args: model)
    with threadpool_limits(limits=1):
        receipt = proposals.discover_pairs(
            frame, state, response, "gaussian", sample_weight=weights
        )
    assert len(observed) == 1
    assert observed[0] is weights
    assert receipt["training_total_weight"] == 60.0
    assert receipt["sample_weight_provided"] is True
    assert receipt["tree_diagnostics"]["n_iter"] == 200
