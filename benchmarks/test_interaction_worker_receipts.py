"""Interrupted workers retain their evidence and costs across older launchers."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import benchmark_gbm_interactions as gbm
import benchmark_real_interactions as real
import benchmark_targeted_interactions as targeted
import pytest


@pytest.fixture(params=[real, targeted, gbm], ids=["real", "targeted", "gbm"])
def launcher(request, tmp_path):
    args = SimpleNamespace(
        output=tmp_path / "run",
        manifest=tmp_path / "manifest.json",
        data_root=tmp_path / "data",
        interactions=2,
        max_p=100,
        max_rows=1000,
    )
    return request.param, args


@pytest.mark.parametrize("stage", ["fit", "evaluate"])
@pytest.mark.parametrize("status", ["timeout", "error"])
@pytest.mark.parametrize("raw", [b"", b'{"status":', b"\xff"])
def test_launch_recovers_interrupted_receipt_bytes_and_costs(
    launcher, monkeypatch, stage, status, raw
):
    runner, args = launcher

    def interrupted(command, **kwargs):
        output = Path(command[command.index("--output") + 1])
        path = output / ("result.json" if stage == "fit" else "evaluation.json")
        path.write_bytes(raw)
        return {"status": status, "process_seconds": 2.5, "returncode": -9}

    monkeypatch.setattr(runner, "run_isolated", interrupted)
    record = runner.launch(args, "fixture", "arm", stage, 3)
    assert record["status"] == status
    assert record["process"]["process_seconds"] == 2.5
    assert record["warnings_complete"] is False
    assert "parent_finished_utc" in record and "finished_utc" not in record
    output = args.output / "fixture" / "arm"
    preserved = record["incomplete_receipt"]
    assert preserved["sha256"] == hashlib.sha256(raw).hexdigest()
    assert preserved["bytes"] == len(raw)
    assert (output / preserved["path"]).read_bytes() == raw
    path = output / ("result.json" if stage == "fit" else "evaluation.json")
    persisted = json.loads(path.read_text())
    assert persisted["status"] == status
    assert persisted["incomplete_receipt"] == preserved
    assert persisted["parent_finished_utc"] == record["parent_finished_utc"]
    process = json.loads((output / f"{stage}_process.json").read_text())
    assert process == record["process"]


@pytest.mark.parametrize("stage", ["fit", "evaluate"])
def test_launch_preserves_complete_worker_evidence(launcher, monkeypatch, stage):
    runner, args = launcher
    status = "evaluated" if stage == "evaluate" else "fitted" if runner is gbm else "converged"
    original = {
        "status": status,
        "finished_utc": "2026-09-14T00:00:00+00:00",
        "warnings": [],
        "source_identity": "fixture",
        "validation": {"primary_loss": 0.25},
    }

    def completed(command, **kwargs):
        output = Path(command[command.index("--output") + 1])
        real.write_json(output / ("result.json" if stage == "fit" else "evaluation.json"), original)
        return {"status": "success", "process_seconds": 1.5, "returncode": 0}

    monkeypatch.setattr(runner, "run_isolated", completed)
    record = runner.launch(args, "fixture", "arm", stage, 3)
    assert {key: record[key] for key in original} == original
    assert record["process"]["process_seconds"] == 1.5
    assert "incomplete_receipt" not in record and "parent_finished_utc" not in record
