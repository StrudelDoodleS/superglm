"""Protect dataset pairing and failure accounting in the external comparison."""

from copy import deepcopy

import numpy as np
import pytest
from psst_fast_analysis import audit_fast_gains, join_rows, paired_values, verify_shared_fits


def record(identity, rank):
    return {
        "id": identity,
        "case": "gaussian_wave",
        "design": "gaussian_independent",
        "strength": 0.06,
        "replicate": int(identity),
        "phase": "signal",
        "dispatch": {},
        "methods": {
            method: {"target_rank": rank, "target_z": 2.0, "max_z": 4.0}
            for method in ("old", "corrected")
        },
    }


def test_join_pairs_by_identity_not_completion_order():
    joined = join_rows([record("0", 1), record("1", 8)], [record("1", 2), record("0", 9)])
    np.testing.assert_array_equal(
        paired_values(joined, "target_rank", "fast_default", limit=3),
        [[False, True], [True, False]],
    )


def test_join_rejects_missing_duplicate_or_different_datasets():
    with pytest.raises(ValueError, match="identit"):
        join_rows([record("0", 1)], [record("1", 1)])
    with pytest.raises(ValueError, match="duplicate"):
        join_rows([record("0", 1), record("0", 1)], [record("0", 1)])
    changed = record("0", 1)
    changed["strength"] = 0.08
    with pytest.raises(ValueError, match="strength"):
        join_rows([record("0", 1)], [changed])


def test_failed_fast_screen_does_not_erase_successful_psst():
    fast = record("0", 2)
    fast.pop("dispatch")
    joined = join_rows([record("0", 1)], [fast])
    np.testing.assert_array_equal(
        paired_values(joined, "target_rank", "fast_default", limit=3), [[False, True]]
    )
    np.testing.assert_array_equal(paired_values(joined, "max_z", "fast_purify"), [[-np.inf, 4.0]])


def test_shared_refit_mismatch_is_rejected():
    psst = record("0", 1)
    psst["refits"] = {"baseline": {"validation_loss": 1.0, "test_risk": 0.2}}
    fast = deepcopy(psst)
    assert verify_shared_fits([psst], [fast])["metric_comparisons"] == 2
    fast["refits"]["baseline"]["validation_loss"] = 1.01
    with pytest.raises(ValueError, match="Shared fit"):
        verify_shared_fits([psst], [fast])


def test_finite_negative_native_failure_is_rejected():
    row = record("0", 1)
    for method in row["methods"].values():
        method["all_z"] = {str(i): 0.01 for i in range(435)}
    assert audit_fast_gains([row])["scores"] == 870
    row["methods"]["old"]["all_z"]["0"] = -1e308
    with pytest.raises(ValueError, match="Invalid native FAST gain"):
        audit_fast_gains([row])
