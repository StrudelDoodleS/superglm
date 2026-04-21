from pathlib import Path

import pandas as pd

from superglm import Constraint, PSpline, SuperGLM
from superglm._debug import set_debug_level


def test_debug_level_zero_emits_no_reml_trace(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SUPERGLM_DEBUG_DIR", str(tmp_path))
    set_debug_level(0)

    x = pd.DataFrame({"x": [0.0, 0.5, 1.0, 0.25, 0.75] * 20})
    y = x["x"].to_numpy() ** 2 + 0.1

    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        discrete=True,
        features={"x": PSpline(n_knots=8, constraint=Constraint.fit.convex)},
    )
    model.fit_reml(x, y, max_reml_iter=4)

    assert not list(tmp_path.glob("*.jsonl"))


def test_debug_level_two_writes_reml_trace_files(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SUPERGLM_DEBUG_DIR", str(tmp_path))
    set_debug_level(2)

    x = pd.DataFrame({"x": [0.0, 0.5, 1.0, 0.25, 0.75] * 20})
    y = x["x"].to_numpy() ** 2 + 0.1

    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        discrete=True,
        features={"x": PSpline(n_knots=8, constraint=Constraint.fit.convex)},
    )
    model.fit_reml(x, y, max_reml_iter=4)

    assert list(tmp_path.glob("*run.json"))
    assert list(tmp_path.glob("*reml.jsonl"))
    assert list(tmp_path.glob("*pirls.jsonl"))
