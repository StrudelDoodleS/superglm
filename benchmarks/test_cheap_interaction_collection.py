"""Timing summaries require explicit uninstrumented evidence on both repetitions."""

import json
import shutil
import sys

import collect_cheap_interactions as collector
import numpy as np
import pytest


@pytest.fixture
def repetitions(tmp_path):
    for repetition, seconds in enumerate((1.0, 3.0), 1):
        directory = tmp_path / f"m0-fixed-r{repetition}"
        directory.mkdir()
        (directory / "run.json").write_text(json.dumps({"profiled": False}))
        result = {
            "profiled": False,
            "mode": "fixed",
            "status": "converged",
            "interactions": 0,
            "k": 4,
            "coefficient_count_without_intercept": 4,
            "fitted_smoothing_parameter_count": 1,
            "fit_seconds": seconds,
            "fit_end_peak_process_rss_mib": 10.0,
            "retained_model_storage": {"total_payload_bytes": 32},
            "mse": {"test": 2.0},
            "telemetry": {"reml": {"n_reml_iter": 2}},
            "resolved_direct_backend": "gram",
            "package_source_sha256": "source",
            "data_sha256": "train",
            "test_data_sha256": "test",
            "runtime": {
                "python": "3.13.14",
                "platform": "Linux-test",
                "packages": {
                    name: "test-version"
                    for name in ("superglm", "numpy", "scipy", "pandas", "numba", "threadpoolctl")
                },
                "threadpools": [
                    {
                        "user_api": "blas",
                        "internal_api": "openblas",
                        "num_threads": 1,
                        "prefix": "libopenblas",
                        "filepath": "/test/libopenblas.so",
                        "version": "test",
                    }
                ],
            },
        }
        (directory / "result.json").write_text(json.dumps(result))
        np.savez_compressed(directory / "predictions.npz", prediction=np.array([1.0, 2.0]))
    return tmp_path


def test_unprofiled_repetitions_produce_the_complete_fit_median(repetitions):
    result = collector.collect(repetitions)
    assert result["summaries"]["m0-fixed"]["fit_seconds_median"] == 2.0
    assert result["summaries"]["m0-fixed"]["time_ratio_to_additive"] == 1.0


@pytest.mark.parametrize("repetition", [1, 2])
@pytest.mark.parametrize("filename", ["run.json", "result.json"])
@pytest.mark.parametrize("profiled", [True, None])
def test_timing_summary_refuses_profiled_or_unverified_samples(
    repetitions, repetition, filename, profiled
):
    path = repetitions / f"m0-fixed-r{repetition}" / filename
    record = json.loads(path.read_text())
    if profiled is None:
        record.pop("profiled")
    else:
        record["profiled"] = profiled
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="unprofiled"):
        collector.collect(repetitions)


@pytest.mark.parametrize("explicit_archive", [False, True])
def test_collection_cli_preserves_the_frozen_measurement(
    repetitions, tmp_path, monkeypatch, explicit_archive
):
    root = tmp_path / "checkout"
    archive = root / "notes/research/2026-09-13-cheap-interaction-measurements.json"
    archive.parent.mkdir(parents=True)
    archive.write_text("original frozen evidence\n")
    monkeypatch.setattr(collector, "ROOT", root)
    arguments = [collector.__file__, "--input", str(repetitions)]
    if explicit_archive:
        arguments.extend(["--output", str(archive)])
    monkeypatch.setattr(sys, "argv", arguments)
    if explicit_archive:
        with pytest.raises(SystemExit) as error:
            collector.main()
        assert error.value.code == 2
    else:
        collector.main()
        assert (root / ".benchmark-artifacts/cheap-interaction-measurements-replay.json").is_file()
    assert archive.read_text() == "original frozen evidence\n"


@pytest.mark.parametrize(
    "field,replacement",
    [
        (("runtime", "python"), "3.14.0"),
        (("runtime", "platform"), "Darwin-test"),
        (("runtime", "packages", "numpy"), "different-version"),
        (("runtime", "threadpools", 0, "num_threads"), 8),
        (("runtime", "threadpools", 0, "internal_api"), "blis"),
        (("resolved_direct_backend",), "qr"),
    ],
)
def test_collection_refuses_mixed_repetition_runtimes(repetitions, field, replacement):
    path = repetitions / "m0-fixed-r2/result.json"
    record = json.loads(path.read_text())
    owner = record
    for key in field[:-1]:
        owner = owner[key]
    owner[field[-1]] = replacement
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="[Rr]untime|backend|thread pool"):
        collector.collect(repetitions)


@pytest.mark.parametrize(
    "field",
    [
        ("runtime",),
        ("runtime", "python"),
        ("runtime", "platform"),
        ("runtime", "packages"),
        ("runtime", "threadpools"),
        ("resolved_direct_backend",),
    ],
)
def test_collection_refuses_missing_runtime_evidence_on_both_repetitions(repetitions, field):
    for path in repetitions.glob("*/result.json"):
        record = json.loads(path.read_text())
        owner = record
        for key in field[:-1]:
            owner = owner[key]
        owner.pop(field[-1])
        path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="[Rr]untime|backend"):
        collector.collect(repetitions)


@pytest.mark.parametrize("changed_runtime", [False, True])
def test_additive_comparison_requires_matching_runtime_but_allows_backend_dispatch(
    repetitions, changed_runtime
):
    for repetition in (1, 2):
        directory = repetitions / f"m1-fixed-r{repetition}"
        shutil.copytree(repetitions / f"m0-fixed-r{repetition}", directory)
        path = directory / "result.json"
        record = json.loads(path.read_text())
        record["interactions"] = 1
        record["resolved_direct_backend"] = "qr"
        if changed_runtime:
            record["runtime"]["platform"] = "Darwin-test"
        path.write_text(json.dumps(record))
    if changed_runtime:
        with pytest.raises(ValueError, match="[Rr]untime"):
            collector.collect(repetitions)
    else:
        result = collector.collect(repetitions)
        assert result["summaries"]["m0-fixed"]["backend"] == "gram"
        assert result["summaries"]["m1-fixed"]["backend"] == "qr"
