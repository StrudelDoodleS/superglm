"""Protect the benchmark's timeout and data-identity boundaries."""

import os
import sys

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
