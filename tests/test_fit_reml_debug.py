import json
from pathlib import Path

import pandas as pd

from superglm import Constraint, PSpline, SuperGLM


def test_debug_level_zero_emits_no_reml_trace(tmp_path: Path, monkeypatch):
    from superglm._debug import set_debug_level

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
    from superglm._debug import set_debug_level

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

    run_files = list(tmp_path.glob("*run.json"))
    reml_files = list(tmp_path.glob("*reml.jsonl"))
    pirls_files = list(tmp_path.glob("*pirls.jsonl"))
    scop_files = list(tmp_path.glob("*scop.jsonl"))

    assert run_files
    assert reml_files
    assert pirls_files
    assert scop_files

    run_payload = json.loads(run_files[0].read_text(encoding="utf-8"))
    assert run_payload["debug_level"] == 2
    assert run_payload["method"] == "fit_reml"

    reml_rows = [
        json.loads(line)
        for line in reml_files[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert reml_rows
    assert reml_rows[0]["iteration"] >= 1

    pirls_rows = [
        json.loads(line)
        for line in pirls_files[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert pirls_rows
    assert pirls_rows[0]["iteration"] >= 1

    scop_rows = [
        json.loads(line)
        for line in scop_files[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert scop_rows
    assert scop_rows[0]["step_norm"] >= 0.0
