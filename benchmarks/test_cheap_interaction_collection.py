"""Timing summaries require explicit uninstrumented evidence on both repetitions."""

import json
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
