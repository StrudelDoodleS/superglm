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


_DORMANT = """
import sys
import numpy as np, pandas as pd
from superglm import Numeric, Spline, SuperGLM
rng = np.random.default_rng(0)
X = pd.DataFrame({"a": rng.uniform(0, 1, 300), "d": rng.normal(size=300)})
y = rng.poisson(np.exp(0.3 * X.a)).astype(float)
SuperGLM(family="poisson", features={"a": Spline(n_knots=5), "d": Numeric()}).fit_reml(
    X, y, max_reml_iter=2
)
print("superglm.diagnostics._fit_record" in sys.modules)
"""


def test_without_benchboard_nothing_is_written_and_the_recorder_never_loads(tmp_path):
    """A fresh interpreter, so no other test can have imported the recorder first."""
    import os
    import subprocess

    env = {k: v for k, v in os.environ.items() if not k.startswith(("BENCHBOARD_", "SUPERGLM_FIT"))}
    env["BENCHBOARD_DIR"] = str(tmp_path / "absent")  # a benchboard directory that does not exist
    run = subprocess.run(
        [sys.executable, "-c", _DORMANT],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=True,
    )
    assert run.stdout.strip().splitlines()[-1] == "False"
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
    assert config["family_params"] == {
        "family": "poisson",
        "estimate_p": False,
        "estimate_theta": False,
    }
    assert "offset" not in config["data"]
    a, pair = (next(t for t in config["terms"] if t["name"] == n) for n in ("a", "a:c"))
    # an exact spline-by-factor term counts each level's own rows, not the full basis per level:
    # each data row sits in at most one level, so the term has at most one basis row's non-zeros
    assert pair["pair"] and not a["pair"]
    assert 0 < pair["nnz_per_row"] <= max(g["basis_width"] for g in pair["groups"])
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

    import warnings

    monkeypatch.setenv("SUPERGLM_FIT_RECORD", "0")
    with warnings.catch_warnings(record=True) as plain_warnings:
        warnings.simplefilter("always")
        plain = _model().fit_reml(X, y, sample_weight=w, max_reml_iter=2).predict(X)
    monkeypatch.delenv("SUPERGLM_FIT_RECORD")
    # before the fit (begin), after it (end), and inside end's own reading of the model
    for target in ("begin", "end", "_glm_outcome", "_write_atomic"):
        with monkeypatch.context() as patch, warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            patch.setattr(_fit_record, target, broken)
            predicted = _model().fit_reml(X, y, sample_weight=w, max_reml_iter=2).predict(X)
        np.testing.assert_array_equal(predicted, plain)
        assert [str(m.message) for m in seen] == [str(m.message) for m in plain_warnings], target
    # only the _glm_outcome failure still writes a record, its outcome marked unavailable
    (payload,) = _records(tmp_path)
    assert payload["fits"][0]["summary"]["n_iter"] is None
    # end() replaced outright cannot clear its marker: that fit reads as killed, the rest do not
    assert len(list(tmp_path.glob("started-*.json"))) == 1


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


def test_random_effect_levels_have_no_phantom_base_level(monkeypatch, tmp_path):
    from superglm import RandomEffect

    monkeypatch.setenv("BENCHBOARD_FIT_DIR", str(tmp_path))
    monkeypatch.setenv("BENCHBOARD_TASK_ID", "task-5")
    X, y, w = _book()
    model = SuperGLM(family="poisson", features={"a": Spline(n_knots=5), "c": RandomEffect()})
    model.fit_reml(X, y, sample_weight=w, max_reml_iter=2)
    (payload,) = _records(tmp_path)
    term = next(t for t in payload["fits"][0]["config"]["terms"] if t["name"] == "c")
    assert term["cardinality"] == 4 and term["empty_cells"] == 0 and term["min_level_count"] > 0
    assert term["nnz_per_row"] == pytest.approx(1.0)


def test_a_non_string_feature_name_keeps_its_descriptor(monkeypatch, tmp_path):
    monkeypatch.setenv("BENCHBOARD_FIT_DIR", str(tmp_path))
    monkeypatch.setenv("BENCHBOARD_TASK_ID", "task-6")
    X, y, w = _book()
    X = pd.DataFrame({0: X["a"].to_numpy(), 1: X["d"].to_numpy()})
    model = SuperGLM(family="poisson", features={0: Spline(n_knots=5), 1: Numeric()})
    model.fit_reml(X, y, sample_weight=w, max_reml_iter=2)
    (payload,) = _records(tmp_path)
    config = payload["fits"][0]["config"]
    assert config["n_features"] == 2 and config["n_pairs"] == 0 and len(config["terms"]) == 2
    assert all(term["kind"] is not None for term in config["terms"])  # 0 is a name, not a gap


def test_an_offset_is_summarised_as_an_offset(monkeypatch, tmp_path):
    monkeypatch.setenv("BENCHBOARD_FIT_DIR", str(tmp_path))
    monkeypatch.setenv("BENCHBOARD_TASK_ID", "task-7")
    X, y, w = _book()
    offset = np.full(len(y), 800.0)  # exp() would overflow; the offset itself is recorded
    SuperGLM(family="gaussian", features={"a": Spline(n_knots=5)}).fit(X, y, offset=offset)
    (payload,) = _records(tmp_path)
    data = payload["fits"][0]["config"]["data"]
    assert data["offset_present"] is True and data["offset"]["max"] == 800.0


def test_a_superlss_fit_is_recorded_with_its_predictors(monkeypatch, tmp_path):
    from superglm import GaussianLS, SuperLSS, s

    monkeypatch.setenv("BENCHBOARD_FIT_DIR", str(tmp_path))
    monkeypatch.setenv("BENCHBOARD_TASK_ID", "task-8")
    rng = np.random.default_rng(4)
    X = pd.DataFrame({"a": rng.uniform(0, 1, 400)})
    y = np.sin(3 * X.a.to_numpy()) + rng.normal(0, 0.3, 400)
    family = GaussianLS()
    SuperLSS(family, family.location(s("a", kind="cr", k=5)), family.scale()).fit_reml(X, y)
    (payload,) = _records(tmp_path)
    fit = payload["fits"][0]
    assert fit["config"]["model"] == "SuperLSS" and fit["status"] == "success"
    assert fit["summary"]["n_columns"] > 0 and fit["summary"]["n_iter"] >= 1
    assert {t["predictor"] for t in fit["config"]["terms"]} >= {"location"}
    assert fit["phases"] and fit["phases"][0]["source"] == "exclusive"


def test_an_eager_frame_input_keeps_its_numeric_shape(monkeypatch, tmp_path):
    from superglm import GaussianLS, SuperLSS, s
    from superglm._frame import as_eager_frame

    monkeypatch.setenv("BENCHBOARD_FIT_DIR", str(tmp_path))
    monkeypatch.setenv("BENCHBOARD_TASK_ID", "task-10")
    rng = np.random.default_rng(4)
    X = pd.DataFrame({"a": rng.uniform(0, 1, 400)})
    y = np.sin(3 * X.a.to_numpy()) + rng.normal(0, 0.3, 400)
    family = GaussianLS()
    model = SuperLSS(family, family.location(s("a", kind="cr", k=5)), family.scale())
    model.fit_reml(as_eager_frame(X), y)
    (payload,) = _records(tmp_path)
    assert payload["fits"][0]["config"]["numeric_inputs"]["a"]["distinct"] == 400


def test_estimate_theta_is_recorded(monkeypatch, tmp_path):
    from superglm import families

    monkeypatch.setenv("BENCHBOARD_FIT_DIR", str(tmp_path))
    monkeypatch.setenv("BENCHBOARD_TASK_ID", "task-9")
    X, y, w = _book()
    model = SuperGLM(family=families.nb2(theta=1.0), features={"a": Spline(n_knots=5)})
    model.estimate_theta(X, np.round(y * w))
    records = [p["fits"][0] for p in _records(tmp_path)]
    (record,) = [r for r in records if r["method"] == "estimate_theta"]
    assert record["config"]["family_params"]["estimate_theta"] is True
