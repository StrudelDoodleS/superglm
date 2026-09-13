"""Matched-parent and model-selection contracts for targeted real-data controls."""

import copy

import benchmark_real_interactions as base
import benchmark_targeted_interactions as target
import numpy as np
import pandas as pd


def test_mixed_candidate_adds_clock_and_weather_without_changing_parent_state():
    from superglm.features.interaction import CategoricalInteraction, SplineCategorical

    frame = pd.DataFrame(
        {
            "hr": np.tile(np.arange(4), 60),
            "workingday": np.repeat(np.arange(2), 120),
            "temp": np.random.default_rng(17).uniform(size=240),
        }
    )
    state = base.fit_preprocessor(frame, categorical_columns=["hr", "workingday"])
    original = copy.deepcopy(state)
    data = base.transform_features(frame, state)
    for k, expected in [(4, 19), (6, 27)]:
        model = target.build_targeted_model(
            state, "poisson", k, [["hr", "workingday"], ["temp", "hr"]]
        )
        model._build_design_matrix(data, np.ones(len(data)), np.ones(len(data)), None)
        assert model._dm.p == expected
        assert isinstance(model._interaction_specs["hr:workingday"], CategoricalInteraction)
        assert isinstance(model._interaction_specs["temp:hr"], SplineCategorical)
    assert state == original


def test_each_resolution_has_a_matching_additive_and_known_candidate_control():
    menu = target.variant_menu("uci_bike_sharing")
    for k in (4, 6):
        variants = [variant for variant in menu if variant["parent_k"] == k]
        assert any(not variant["pairs"] for variant in variants)
        assert any(variant["pairs"] == [["hr", "workingday"]] for variant in variants)
    ames = target.variant_menu("ames_housing")
    assert any(variant["pairs"] == [["Gr Liv Area", "Bldg Type"]] for variant in ames)


def test_validation_choice_ignores_test_and_refuses_unfinished_better_fit():
    records = {
        "unfinished": {"status": "not_converged", "validation": {"primary_loss": 0.0}},
        "complex": {
            "status": "converged",
            "validation": {"primary_loss": 1.0},
            "P": 20,
            "test": 0.0,
        },
        "simple": {
            "status": "converged",
            "validation": {"primary_loss": 1.0},
            "P": 5,
            "test": 100.0,
        },
    }
    assert target.select_variant(records) == "simple"
    records["simple"]["status"] = "not_converged"
    assert target.select_variant(records) == "complex"


def test_unresolved_validation_metric_cannot_win_selection():
    records = {
        "invalid": {"status": "converged", "validation": {"primary_loss": float("nan")}, "P": 1},
        "valid": {"status": "converged", "validation": {"primary_loss": 2.0}, "P": 5},
    }
    assert target.select_variant(records) == "valid"
