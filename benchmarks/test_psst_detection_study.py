"""Oracles for the benchmark, kept outside the production test suites."""

import numpy as np
import pytest
from psst_detection_study import (
    loss,
    prediction_risk,
    rank_ladders,
    score_frozen_selections,
    select_by_validation,
)

from superglm.screening._score_stat import ScreenedPair


def test_each_normalization_selects_its_own_rung():
    # Old z is 2 then 1.5. Corrected z is 2 then 3.
    ladders = [[ScreenedPair(6.0, 2.0, 4.0, 4.0), ScreenedPair(14.0, 8.0, 1.0, 4.0)]]
    out = rank_ladders([("a", "b")], ladders, phi=1.0)
    assert out["old"][0]["z"] == 2.0
    assert out["old"][0]["rung"] == 0
    assert out["corrected"][0]["z"] == 3.0
    assert out["corrected"][0]["rung"] == 1


def test_dispersion_is_applied_once_and_unpenalized_scores_agree():
    out = rank_ladders([("a", "b")], [[ScreenedPair(12.0, 2.0, 0.0, 4.0)]], phi=2.0)
    assert out["old"][0]["z"] == out["corrected"][0]["z"] == 2.0


def test_missing_or_invalid_candidate_cannot_silently_leave_denominator():
    with pytest.raises(ValueError):
        rank_ladders([("a", "b")], [], phi=1.0)
    with pytest.raises(ValueError):
        rank_ladders([("a", "b")], [[ScreenedPair(np.nan, 2.0, 0.0, 4.0)]], phi=1.0)


def test_validation_selection_does_not_pick_test_winner():
    results = {
        "baseline": {"validation_loss": 3.0, "test_loss": 5.0},
        "a:b": {"validation_loss": 2.0, "test_loss": 9.0},
        "c:d": {"validation_loss": 4.0, "test_loss": 1.0},
    }
    assert select_by_validation(["a:b", "c:d"], results) == "a:b"


@pytest.mark.parametrize("family", ["gaussian", "poisson"])
@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_nonfinite_predictions_are_recordable_failures(family, invalid):
    with pytest.raises(ValueError, match="finite"):
        loss(np.ones(2), np.array([1.0, invalid]), family)
    with pytest.raises(ValueError, match="finite"):
        prediction_risk(np.ones(2), np.array([1.0, invalid]), family)


def test_failed_test_prediction_does_not_change_frozen_selection():
    class Prediction:
        def __init__(self, fails=False):
            self.fails = fails

        def predict(self, frame):
            if self.fails:
                raise ValueError("test-only failure")
            return np.zeros(len(frame))

    results, _ = score_frozen_selections(
        {"old": "a:b", "corrected": "baseline"},
        {"baseline": Prediction(), "a:b": Prediction(fails=True)},
        np.zeros((2, 1)),
        np.ones(2),
        np.ones(2),
        "gaussian",
    )
    assert results["old"]["selected"] == "a:b"
    assert results["old"]["evaluation_error"] == ["ValueError: test-only failure"]
    assert "test_loss_gain" not in results["old"]
    assert results["corrected"]["test_loss_gain"] == 0.0
