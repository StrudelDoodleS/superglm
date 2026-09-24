"""Editor structural steps: one stack, set reference, transform, revert."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, Spline, SuperGLM
from superglm.editor import EditorSession


@pytest.fixture
def region_model():
    rng = np.random.default_rng(20260924)
    region = rng.choice(["A", "B", "C", "D"], 600, p=[0.3, 0.3, 0.2, 0.2])
    x = rng.uniform(0.0, 10.0, 600)
    effects = {"A": 0.0, "B": 0.15, "C": 0.2, "D": -0.1}
    y = 0.4 + np.array([effects[r] for r in region]) + 0.05 * x + rng.normal(0.0, 0.05, 600)
    X = pd.DataFrame({"region": region, "x": x})
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        spline_penalty=0.1,
        features={"region": Categorical(base="first"), "x": Spline(n_knots=6)},
    )
    model.fit(X, y)
    return model, X


def test_every_structural_step_pushes_one_restorable_entry(region_model):
    model, _ = region_model
    session = EditorSession.from_model(model, terms=["region"])
    opened = session.model

    session.select_levels("region", ["B", "C"])
    collapsed = session.replace_with_collapsed_levels("region", method="fit")
    assert [s.operation for s in session.structure_history] == ["collapse_levels"]
    assert session.structure_history[-1].previous_model is opened
    assert session.structure_history[-1].label == "collapse B + C in region"

    session.select_levels("region", ["B", "C"])
    session.replace_with_ungrouped_levels("region", method="fit")
    # The ungroup removes the last group, so the pre-collapse fit is reused ...
    assert session.model is opened
    # ... but it is still a step of its own: Restore undoes it.
    assert [s.operation for s in session.structure_history] == [
        "collapse_levels",
        "ungroup_levels",
    ]
    assert session.structure_history[-1].label == "ungroup B, C in region"
    assert session.uncollapse_levels() is collapsed
    assert session.uncollapse_levels() is opened
    assert not session.can_uncollapse_levels()
