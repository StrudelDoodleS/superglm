"""Benchboard fit records: absent and unimported by default, one per fit when asked, never a fit's failure."""

from __future__ import annotations

import gzip
import json
import sys

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, Numeric, Spline, SuperGLM

RECORDER = "superglm.diagnostics._fit_record"


def _book(n=600, seed=3):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "a": rng.uniform(0, 1, n),
            "c": rng.integers(0, 4, n).astype(str),
            "d": rng.normal(size=n),
        }
    )
    w = rng.uniform(0.5, 1.5, n)
    y = rng.poisson(w * np.exp(0.4 * np.sin(3 * X.a) - 1.0)) / w
    return X, y, w


def _model():
    return SuperGLM(
        family="poisson",
        features={"a": Spline(n_knots=5), "c": Categorical(), "d": Numeric()},
        interactions=[("a", "c")],
    )


def _records(folder):
    return [json.loads(gzip.decompress(p.read_bytes())) for p in sorted(folder.glob("*.json.gz"))]


@pytest.fixture(autouse=True)
def _no_ambient_benchboard(monkeypatch, tmp_path_factory):
    """Each test chooses its own destination; the machine's benchboard directory stays out."""
    for name in ("BENCHBOARD_FIT_DIR", "BENCHBOARD_TASK_ID", "SUPERGLM_FIT_RECORD"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("BENCHBOARD_DIR", str(tmp_path_factory.mktemp("absent") / "benchboard"))


def test_unset_writes_nothing_and_never_imports_the_recorder(monkeypatch, tmp_path):
    monkeypatch.delitem(sys.modules, RECORDER, raising=False)
    X, y, w = _book()
    _model().fit_reml(X, y, sample_weight=w, max_reml_iter=2)
    assert RECORDER not in sys.modules
    assert not list(tmp_path.iterdir())


def test_set_spools_one_record_per_fit(monkeypatch, tmp_path):
    monkeypatch.setenv("BENCHBOARD_FIT_DIR", str(tmp_path))
    monkeypatch.setenv("BENCHBOARD_TASK_ID", "task-1")
    X, y, w = _book()
    _model().fit_reml(X, y, sample_weight=w, max_reml_iter=3)
    (payload,) = _records(tmp_path)
    assert not list(tmp_path.glob("started-*"))  # the marker goes once the record is written
    assert payload["protocol"] == 1 and payload["task_id"] == "task-1"
    (fit,) = payload["fits"]
    assert fit["method"] == "fit_reml" and fit["status"] == "success" and fit["wall_s"] > 0
    config, summary = fit["config"], fit["summary"]
    assert config["n_rows"] == len(y) and config["distribution"] == "poisson"
    assert config["n_features"] == 3 and config["n_pairs"] == 1
    assert config["data"]["weights"]["min"] == pytest.approx(w.min())
    assert config["data"]["offset_present"] is False
    cat = next(t for t in config["terms"] if t["name"] == "c")
    assert cat["cardinality"] == 4 and cat["min_level_count"] > 0
    assert config["design"]["nnz_per_row"] > 0
    assert "pools" in config["threads"] and "SUPERGLM_N_JOBS" in config["threads"]["env"]
    assert summary["n_columns"] > 0 and summary["n_iter"] >= 1 and summary["converged"] is not None
    assert summary["n_penalties"] >= 1 and summary["cpu_s"] >= 0 and summary["numba_compile_s"] >= 0
    assert cat["zero_response_cells"] >= 0 and "skew" in config["numeric_inputs"]["a"]
    assert config["family_params"] == {"family": "poisson", "estimate_p": False}
    assert isinstance(config["first_fit_in_process"], bool)
    assert config["under_pytest"] is True
    assert fit["phases"] and all(p["wall_s"] >= 0 for p in fit["phases"])
    json.dumps(payload, allow_nan=False)


def test_a_recorder_failure_never_fails_the_fit(monkeypatch, tmp_path):
    from superglm.diagnostics import _fit_record

    monkeypatch.setenv("BENCHBOARD_FIT_DIR", str(tmp_path))
    monkeypatch.setenv("BENCHBOARD_TASK_ID", "task-2")
    X, y, w = _book()

    def broken(*args, **kwargs):
        raise RuntimeError("recorder broke")

    for target in ("begin", "_process_facts"):
        with monkeypatch.context() as patch:
            patch.setattr(_fit_record, target, broken)
            model = _model().fit_reml(X, y, sample_weight=w, max_reml_iter=2)
        assert np.isfinite(model.predict(X)).all()
    assert not list(tmp_path.iterdir())


def test_a_failing_fit_keeps_its_own_exception_and_is_recorded(monkeypatch, tmp_path):
    monkeypatch.setenv("BENCHBOARD_FIT_DIR", str(tmp_path))
    monkeypatch.setenv("BENCHBOARD_TASK_ID", "task-3")
    X, y, w = _book()
    with pytest.raises(ValueError):
        _model().fit(X, y[:-1])
    (payload,) = _records(tmp_path)
    assert payload["fits"][0]["status"] == "failed"
    assert payload["fits"][0]["summary"]["error"].startswith("ValueError")


def test_outside_a_job_records_into_benchboards_unowned_spool(monkeypatch, tmp_path):
    base = tmp_path / "benchboard"
    base.mkdir()
    monkeypatch.setenv("BENCHBOARD_DIR", str(base))
    X, y, w = _book()
    _model().fit_reml(X, y, sample_weight=w, max_reml_iter=2)
    (payload,) = _records(base / "fits" / "unowned")
    assert payload["task_id"].startswith("unowned-")
    assert payload["fits"][0]["config"]["unowned"] is True
    monkeypatch.setenv("SUPERGLM_FIT_RECORD", "0")
    _model().fit_reml(X, y, sample_weight=w, max_reml_iter=2)
    assert len(_records(base / "fits" / "unowned")) == 1


def test_a_fit_that_never_ends_leaves_a_started_marker(monkeypatch, tmp_path):
    from superglm import families
    from superglm.diagnostics import _fit_record

    monkeypatch.setenv("BENCHBOARD_FIT_DIR", str(tmp_path))
    monkeypatch.setenv("BENCHBOARD_TASK_ID", "task-4")
    monkeypatch.setattr(_fit_record, "end", lambda *args: None)  # as if the process were killed
    X, y, w = _book()
    model = SuperGLM(family=families.nb2(theta="auto"), features={"a": Spline(n_knots=5)})
    model.fit_reml(X, np.round(y * w), max_reml_iter=2)
    (marker,) = tmp_path.glob("started-*.json")
    started = json.loads(marker.read_text())
    assert started["task_id"] == "task-4" and started["method"] == "fit_reml" and started["pid"] > 0
    assert started["config"]["family_params"]["theta"] == "auto"
