"""Focused tests for retained caller-data mutation guards."""

import numpy as np
import pandas as pd

from superglm.model.fit_data_guard import FitDataGuard


def test_capture_hashes_frame_without_making_a_shallow_copy(monkeypatch):
    X = pd.DataFrame({"x": [1.0, 2.0], "group": ["a", "b"]})
    y = np.array([3.0, 4.0])

    def fail_copy(self, *args, **kwargs):
        del self, args, kwargs
        raise AssertionError("FitDataGuard.capture copied the retained frame")

    monkeypatch.setattr(pd.DataFrame, "copy", fail_copy)

    guard = FitDataGuard.capture(X, y)

    assert guard.matches(X, y, None, None, fit_weights=None, fit_offset=None)
