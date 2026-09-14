"""Check receipt refusal paths without starting a fit worker."""

import json
import os
import subprocess
import sys
from pathlib import Path

import benchmark_many_interactions as benchmark
import pytest


@pytest.mark.parametrize("optimization", ["-O", "-OO", "environment"])
def test_benchmark_rejects_disabled_assertions(optimization, tmp_path):
    env = os.environ.copy()
    env.pop("PYTHONOPTIMIZE", None)
    flags = [] if optimization == "environment" else [optimization]
    if optimization == "environment":
        env["PYTHONOPTIMIZE"] = "1"
    output = tmp_path / "receipt"
    result = subprocess.run(
        [sys.executable, *flags, benchmark.__file__, "--help", "--output", str(output)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "requires enabled assertions" in result.stderr
    assert not output.exists()


@pytest.fixture
def run_receipt(tmp_path, monkeypatch):
    output = tmp_path / "receipt"
    monkeypatch.setattr(sys, "argv", [benchmark.__file__, "--output", str(output)])

    def run(process_seconds):
        def isolated(*args, **kwargs):
            return {
                "status": "success",
                "pid": 123,
                "returncode": 0,
                "process_seconds": process_seconds,
            }

        monkeypatch.setattr(benchmark, "run_isolated", isolated)
        return benchmark.main()

    return run, output


@pytest.mark.parametrize("process_seconds", [float("nan"), float("inf"), float("-inf")])
def test_run_receipt_refuses_nonfinite_metrics(run_receipt, process_seconds, capsys):
    run, output = run_receipt
    with pytest.raises(ValueError, match="Out of range float values"):
        run(process_seconds)
    assert not (output / "run.json").exists()
    assert capsys.readouterr().out == ""


def test_run_receipt_preserves_finite_metrics(run_receipt, capsys):
    run, output = run_receipt
    assert run(0.125) == 0
    receipt = json.loads((output / "run.json").read_text())
    assert receipt["process_seconds"] == 0.125
    assert receipt["status"] == "success"
    assert Path(receipt["command"][1]) == Path(benchmark.__file__).resolve()
    assert json.loads(capsys.readouterr().out) == receipt
