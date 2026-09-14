"""Check receipt refusals, portable resource units and single-threaded workers."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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


@pytest.fixture
def worker_args(tmp_path):
    return SimpleNamespace(
        rows=128,
        interactions=0,
        k=4,
        interaction_k=None,
        mode="fixed",
        profile=False,
        output=tmp_path,
    )


@pytest.mark.parametrize("platform,maximum_rss", [("linux", 102400), ("darwin", 104857600)])
def test_worker_reports_peak_rss_in_mib(worker_args, monkeypatch, platform, maximum_rss):
    import threadpoolctl

    monkeypatch.setattr(threadpoolctl, "threadpool_info", lambda: [{"num_threads": 1}])
    monkeypatch.setattr(benchmark, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(
        benchmark,
        "resource",
        SimpleNamespace(
            RUSAGE_SELF=0, getrusage=lambda who: SimpleNamespace(ru_maxrss=maximum_rss)
        ),
    )
    benchmark.worker(worker_args)
    result = json.loads((worker_args.output / "result.json").read_text())
    assert result["fit_end_peak_process_rss_mib"] == 100.0


@pytest.mark.parametrize("variable", ["VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"])
def test_launcher_overrides_inherited_backend_thread_counts(tmp_path, monkeypatch, variable):
    monkeypatch.setenv(variable, "8")
    monkeypatch.setattr(sys, "argv", [benchmark.__file__, "--output", str(tmp_path / "receipt")])
    observed = {}

    def isolated(*args, env, **kwargs):
        observed.update(env)
        return {"status": "success", "process_seconds": 0.125}

    monkeypatch.setattr(benchmark, "run_isolated", isolated)
    assert benchmark.main() == 0
    assert observed[variable] == "1"


def test_worker_refuses_an_observed_multithreaded_pool(worker_args, monkeypatch):
    import threadpoolctl

    monkeypatch.setattr(threadpoolctl, "threadpool_info", lambda: [{"num_threads": 8}])
    with pytest.raises(ValueError, match="thread pool"):
        benchmark.worker(worker_args)
    assert not (worker_args.output / "result.json").exists()
