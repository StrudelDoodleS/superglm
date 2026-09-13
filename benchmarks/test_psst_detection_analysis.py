"""Independent small oracles for the study's reported measurements."""

import numpy as np
import pytest
from psst_detection_analysis import cutoff, paired_summary, recovery, threshold_summary


def test_null_cutoff_uses_191st_order_statistic_for_200_runs():
    assert cutoff(np.arange(200.0)) == 190.0
    assert cutoff(np.arange(200.0)[::-1]) == 190.0


def test_strict_threshold_distinguishes_method_outcomes():
    calibration = np.column_stack([np.arange(200.0), 2 * np.arange(200.0)])
    scores = np.array([[191.0, 380.0], [189.0, 382.0], [195.0, 378.0], [-np.inf, -np.inf]])
    result = threshold_summary(calibration, scores, seed=7, repeats=100)
    assert result["old"]["count"] == 2
    assert result["corrected"]["count"] == 1
    assert result["difference"] == -0.25
    assert result["n"] == 4


def test_pairing_retains_opposite_discordances():
    result = paired_summary([0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0], seed=9)
    assert result["difference"] == 0.0
    assert result["improved"] == result["worsened"] == 1
    assert result["difference_ci"][0] < 0 < result["difference_ci"][1]


def test_calibration_is_resampled_even_when_evaluation_rows_are_identical():
    calibration = np.column_stack([np.arange(200.0), np.arange(200.0)])
    scores = np.full((100, 2), 191.0)
    result = threshold_summary(calibration, scores, seed=7, repeats=1000)
    assert result["old"]["bootstrap_ci"] == [0.0, 1.0]
    assert result["difference_ci"] == [0.0, 0.0]


def test_mismatched_methods_cannot_be_bootstrapped_as_pairs():
    with pytest.raises(ValueError):
        paired_summary([0.0], [1.0, 2.0])


def test_later_validation_failure_does_not_erase_successful_screen():
    record = {
        "status": "failed",
        "dispatch": {"dense_ladders": 435},
        "methods": {
            "old": {"target_rank": 4, "target_z": 1.0, "max_z": 4.0},
            "corrected": {"target_rank": 2, "target_z": 2.0, "max_z": 4.0},
        },
    }
    result = recovery([record], 3)
    assert result["old"]["count"] == 0
    assert result["corrected"]["count"] == 1
