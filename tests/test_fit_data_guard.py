"""Focused tests for retained caller-data mutation guards."""

import numpy as np
import pandas as pd

from superglm import Numeric, SuperGLM
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


def test_explicit_fit_guard_ignores_unhashable_values_in_unused_columns():
    X = pd.DataFrame(
        {
            "used": np.linspace(-1.0, 1.0, 12),
            "unused": [[index] if index % 2 else {"index": index} for index in range(12)],
        }
    )
    y = 0.5 + 1.2 * X["used"].to_numpy()
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"used": Numeric()},
    ).fit(X, y)

    first = model.metrics(X, y)
    X.at[0, "unused"] = {"changed": True}
    second = model.metrics(X, y)

    assert second is first
