"""Protect the benchmark's timeout and data-identity boundaries."""

import json
import os
import sys
from types import SimpleNamespace

import benchmark_housing_tensor as benchmark
import numpy as np
import pandas as pd
import pytest
from benchmark_housing_tensor import (
    COLUMNS,
    data_fingerprint,
    prepare_data,
    run_isolated,
    source_fingerprint,
)


def test_timeout_kills_and_reaps_the_worker(tmp_path):
    result = run_isolated(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        log_path=tmp_path / "worker.log",
        timeout=0.1,
        env=os.environ.copy(),
    )
    assert result["status"] == "timeout"
    with pytest.raises(ProcessLookupError):
        os.kill(result["pid"], 0)


def test_failed_worker_is_not_reported_as_success(tmp_path):
    result = run_isolated(
        [sys.executable, "-c", "raise RuntimeError('fit failed')"],
        log_path=tmp_path / "worker.log",
        timeout=5,
        env=os.environ.copy(),
    )
    assert result["status"] == "error"
    assert result["returncode"] != 0
    assert "fit failed" in (tmp_path / "worker.log").read_text()


def test_reordered_rows_are_refused_before_the_fixed_split():
    values = np.arange(20640 * len(COLUMNS), dtype=float).reshape(20640, -1)
    frame = pd.DataFrame(values, columns=COLUMNS)
    frame["MedHouseVal"] = np.arange(20640, dtype=float)
    reference = data_fingerprint(frame)
    with pytest.raises(ValueError, match="data fingerprint"):
        prepare_data(frame.iloc[::-1], expected_fingerprint=reference)


def test_response_change_is_refused_before_the_fixed_split():
    frame = pd.DataFrame(np.ones((20640, len(COLUMNS))), columns=COLUMNS)
    frame["MedHouseVal"] = 1.0
    reference = data_fingerprint(frame)
    frame.loc[0, "MedHouseVal"] = 2.0
    with pytest.raises(ValueError, match="data fingerprint"):
        prepare_data(frame, expected_fingerprint=reference)


def test_import_from_another_checkout_is_not_attributed_to_this_source(tmp_path):
    package = tmp_path / "superglm"
    package.mkdir()
    (package / "__init__.py").write_text("__version__ = 'different'\n")
    with pytest.raises(RuntimeError, match="imported SuperGLM"):
        source_fingerprint(package)


@pytest.mark.parametrize("change", ["columns", "operation"])
def test_changed_preprocessing_is_refused(monkeypatch, change):
    frame = pd.DataFrame(np.full((20640, len(COLUMNS)), 2.0), columns=COLUMNS)
    frame["MedHouseVal"] = 1.0
    raw_identity = data_fingerprint(frame)
    splits, _ = prepare_data(frame, expected_fingerprint=raw_identity)
    feature_identity = benchmark.feature_fingerprint(splits)
    if change == "columns":
        monkeypatch.setattr(benchmark, "LOG_COLUMNS", ("AveRooms", "AveBedrms", "AveOccup"))
    else:
        monkeypatch.setattr(np, "log1p", np.sqrt)
    with pytest.raises(ValueError, match="transformed feature fingerprint"):
        prepare_data(frame, expected_fingerprint=raw_identity, expected_features=feature_identity)


def test_worker_checks_transformed_inputs_before_building_the_model(monkeypatch, tmp_path):
    frame = pd.DataFrame(np.full((20640, len(COLUMNS)), 2.0), columns=COLUMNS)
    frame["MedHouseVal"] = 1.0
    reference = json.loads(benchmark.REFERENCE.read_text())
    reference["data_fingerprint"] = data_fingerprint(frame)
    splits, reference["split_sha256"] = prepare_data(
        frame, expected_fingerprint=reference["data_fingerprint"]
    )
    reference["transformed_features_sha256"] = benchmark.feature_fingerprint(splits)
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(reference))
    monkeypatch.setattr(benchmark, "REFERENCE", reference_path)
    monkeypatch.setattr(pd, "read_parquet", lambda path: frame)
    monkeypatch.setattr(benchmark, "LOG_COLUMNS", ("AveRooms", "AveBedrms", "AveOccup"))

    def unexpected_build(case):
        raise AssertionError("The model must not be built with changed benchmark inputs")

    monkeypatch.setattr(benchmark, "build_model", unexpected_build)
    with pytest.raises(ValueError, match="transformed feature fingerprint"):
        benchmark.worker(SimpleNamespace(data="in-memory.parquet", case="rows20"))
