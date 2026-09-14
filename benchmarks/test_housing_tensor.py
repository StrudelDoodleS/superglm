"""Protect benchmark isolation, input identity, and memory measurements."""

import array
import json
import os
import resource
import sys
import weakref
from collections import deque
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

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


def _measure_storage(root):
    measure = getattr(benchmark, "retained_model_storage", None)
    assert callable(measure), "The runner must report retained model backing storage"
    return measure(root)


def test_retained_storage_counts_full_numpy_owner_once_for_shared_views():
    owner = np.arange(24, dtype=np.float64)
    matrix = owner.reshape(4, 6)
    report = _measure_storage([matrix[1:2, 1:3], matrix.T, memoryview(owner)[2:4]])

    assert report["numpy_owned_bytes"] == 192
    assert report["numpy_owner_count"] == 1
    assert report["total_payload_bytes"] == 192
    assert report["unmeasured_buffer_count"] == 0


def test_retained_storage_counts_copies_and_byte_snapshots_without_buffer_alias_duplicates():
    owner = np.arange(8, dtype=np.float64)
    first_snapshot = owner.tobytes()
    second_snapshot = owner.tobytes()
    assert first_snapshot is not second_snapshot
    report = _measure_storage(
        [
            owner[1:3],
            owner.copy(),
            first_snapshot,
            second_snapshot,
            np.frombuffer(first_snapshot, dtype=np.float64)[2:3],
            memoryview(second_snapshot)[8:16],
        ]
    )

    assert report["numpy_owned_bytes"] == 128
    assert report["numpy_owner_count"] == 2
    assert report["bytes_payload_bytes"] == 128
    assert report["bytes_owner_count"] == 2
    assert report["total_payload_bytes"] == 256


def test_retained_storage_resolves_memoryviews_to_bytearray_and_numpy_owners():
    numpy_owner = np.arange(12, dtype=np.int32)
    byte_owner = bytearray(32)
    report = _measure_storage(
        [
            np.frombuffer(memoryview(numpy_owner)[1:3], dtype=np.int32),
            np.frombuffer(memoryview(byte_owner)[8:16], dtype=np.uint8),
            byte_owner,
        ]
    )

    assert report["numpy_owned_bytes"] == 48
    assert report["bytearray_payload_bytes"] == 32
    assert report["bytearray_owner_count"] == 1
    assert report["total_payload_bytes"] == 80


def test_retained_storage_visits_model_slots_containers_and_cycles():
    from superglm import SuperGLM

    class Parent:
        __slots__ = ("__buffer",)

        def __init__(self, value):
            self.__buffer = value

    @dataclass(slots=True, eq=False)
    class Node(Parent):
        next: object = None

    owner = np.arange(5, dtype=np.float64)
    node = Node()
    Parent.__init__(node, owner[1:2])
    model = object.__new__(SuperGLM)
    model.retained = {"nested": deque([({node}, frozenset({node}))])}
    node.next = [model, node]

    report = _measure_storage(model)
    assert report["numpy_owned_bytes"] == 40
    assert report["numpy_owner_count"] == 1
    assert report["total_payload_bytes"] == 40


def test_retained_storage_does_not_follow_functions_classes_or_modules():
    excluded = np.zeros(1000)

    def function():
        return excluded

    function.buffer = excluded

    class Class:
        buffer = excluded

    module = ModuleType("external")
    module.buffer = excluded
    report = _measure_storage(SimpleNamespace(code=[function, Class, module]))
    assert report["total_payload_bytes"] == 0


def test_retained_storage_does_not_charge_weakly_referenced_arrays():
    owner = np.zeros(1000)
    report = _measure_storage([weakref.ref(owner), weakref.proxy(owner)])
    assert report["total_payload_bytes"] == 0


def test_retained_storage_visits_object_array_entries_without_looping():
    values = np.empty(2, dtype=object)
    values[0] = np.zeros(8, dtype=np.float64)
    values[1] = values

    report = _measure_storage(values)
    assert report["numpy_owned_bytes"] == 64 + 2 * np.dtype(object).itemsize
    assert report["numpy_owner_count"] == 2


def test_retained_storage_reports_unknown_external_buffer_without_claiming_view_bytes():
    owner = array.array("d", range(8))
    view = np.frombuffer(owner, dtype=np.float64)[2:3]
    report = _measure_storage([view, memoryview(owner)])

    assert report["total_payload_bytes"] == 0
    assert report["unmeasured_buffer_count"] == 1
    assert report["unmeasured_buffer_types"] == {"array.array": 1}


@pytest.mark.parametrize("profile", [False, True])
def test_worker_samples_fit_rss_before_profile_export_or_storage_inspection(
    monkeypatch, tmp_path, profile
):
    frame = pd.DataFrame(np.ones((20640, len(COLUMNS))), columns=COLUMNS)
    frame["MedHouseVal"] = 1.0
    data_hash = data_fingerprint(frame)
    splits, split_hash = prepare_data(frame, expected_fingerprint=data_hash)
    reference_arrays = tmp_path / "reference.npz"
    np.savez(
        reference_arrays,
        valid_y=splits["valid"][1],
        test_y=splits["test"][1],
        rows20_valid=np.zeros(4128),
        rows20_test=np.zeros(4128),
    )
    reference = {
        "data_fingerprint": data_hash,
        "split_sha256": split_hash,
        "transformed_features_sha256": benchmark.feature_fingerprint(splits),
        "prediction_archive_sha256": benchmark.hashlib.sha256(
            reference_arrays.read_bytes()
        ).hexdigest(),
        "cases": {"rows20": {"valid_mse": 1.0, "test_mse": 1.0}},
    }
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(reference))
    monkeypatch.setattr(benchmark, "REFERENCE", reference_path)
    monkeypatch.setattr(benchmark, "REFERENCE_ARRAYS", reference_arrays)
    monkeypatch.setattr(pd, "read_parquet", lambda path: frame)
    events = []
    peak_mib = 120
    divisor = 1024**2 if sys.platform == "darwin" else 1024

    class Model:
        def __init__(self):
            self.result = SimpleNamespace(converged=True, beta=np.zeros(513))
            self._dm = SimpleNamespace(group_matrices=[])

        def fit_reml(self, train, y, *, sample_weight):
            events.append("fit")
            self.snapshot = np.arange(6, dtype=np.float64).tobytes()

        def training_telemetry(self):
            nonlocal peak_mib
            events.append("telemetry")
            peak_mib = 360
            self.postfit_array = np.ones(1000)
            return {"reml": {"converged": True}}

        def predict(self, features):
            events.append("predict")
            return np.zeros(len(features))

    def usage(who):
        assert who == resource.RUSAGE_SELF
        events.append("rss")
        return SimpleNamespace(ru_maxrss=peak_mib * divisor)

    measure = getattr(benchmark, "retained_model_storage", None)
    assert callable(measure), "The runner must report retained model backing storage"

    def inspect_storage(model):
        nonlocal peak_mib
        events.append("storage")
        peak_mib = 240
        return measure(model)

    class Profile(benchmark.cProfile.Profile):
        def dump_stats(self, path):
            nonlocal peak_mib
            events.append("profile_dump")
            peak_mib = 200
            return super().dump_stats(path)

    monkeypatch.setattr(benchmark, "build_model", lambda case: Model())
    monkeypatch.setattr(resource, "getrusage", usage)
    monkeypatch.setattr(benchmark, "retained_model_storage", inspect_storage)
    monkeypatch.setattr(benchmark.cProfile, "Profile", Profile)
    benchmark.worker(
        SimpleNamespace(data="synthetic.parquet", case="rows20", profile=profile, output=tmp_path)
    )

    result = json.loads((tmp_path / "result.json").read_text())
    assert result["fit_end_peak_process_rss_mib"] == 120
    assert result["peak_process_rss_mib"] == 360
    assert result["retained_model_storage"]["numpy_owned_bytes"] == 4104
    assert result["retained_model_storage"]["bytes_payload_bytes"] == 48
    assert result["retained_model_storage"]["total_payload_bytes"] == 4152
    assert events[:2] == ["fit", "rss"]
    assert events.index("storage") < events.index("telemetry") < events.index("predict")
