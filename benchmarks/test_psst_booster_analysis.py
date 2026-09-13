"""Keep joins and shared-seed uncertainty valid for the booster comparison."""

import numpy as np
import pytest
from psst_booster_analysis import clustered_difference, join_boosters


def test_join_does_not_pair_by_completion_order_or_drop_failed_jobs():
    reference = {"a": {"models": {}}, "b": {"models": {}}}
    records = [
        {"backend": "xgboost", "id": "b", "status": "failed"},
        {"backend": "xgboost", "id": "a", "status": "ok", "test_loss": 2.0, "test_risk": 1.0},
    ]
    joined = join_boosters(reference, records, backends=("xgboost",))
    assert joined["a"]["models"]["xgboost"]["test_risk"] == 1.0
    assert joined["b"]["models"]["xgboost"]["status"] == "failed"
    with pytest.raises(ValueError, match="identit"):
        join_boosters(reference, records[:1], backends=("xgboost",))
    with pytest.raises(ValueError, match="duplicate"):
        join_boosters(reference, [*records, records[0]], backends=("xgboost",))


def test_repeating_shared_seed_cells_does_not_shrink_uncertainty():
    rows = [
        {
            "design": "gaussian_independent",
            "replicate": i,
            "models": {"psst": {"test_risk": 4.0}, "xgboost": {"test_risk": value}},
        }
        for i, value in enumerate((1.0, 2.0, 5.0, 8.0))
    ]
    once = clustered_difference(rows, "xgboost", "test_risk")
    repeated = clustered_difference([row for row in rows for _ in range(5)], "xgboost", "test_risk")
    assert once["difference"] == repeated["difference"] == 0.0
    np.testing.assert_array_equal(once["difference_ci"], repeated["difference_ci"])
    assert once["difference_ci"][0] < 0 < once["difference_ci"][1]


def test_shared_booster_seed_is_clustered_across_designs():
    rows = [
        {
            "design": design,
            "replicate": replicate,
            "models": {"psst": {"test_risk": 2.0}, "xgboost": {"test_risk": value}},
        }
        for design, values in (("independent", (1.0, 3.0)), ("correlated", (3.0, 1.0)))
        for replicate, value in enumerate(values)
    ]
    # Both design contributions cancel within each shared random-seed block.
    result = clustered_difference(rows, "xgboost", "test_risk")
    np.testing.assert_array_equal(result["difference_ci"], [0.0, 0.0])
