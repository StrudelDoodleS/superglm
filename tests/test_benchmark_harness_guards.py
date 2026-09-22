"""The harness must reject inputs it cannot honour, and label its own output.

This benchmark's numbers are quoted in pull request bodies as evidence a change
is neutral or faster.  An instrument used that way has to refuse what it cannot
measure, and has to say which tree it measured.
"""

from __future__ import annotations

import json
import subprocess
import sys
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest
from benchmarks import rank_deficient_complete_fit as bench

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "benchmarks" / "rank_deficient_complete_fit.py"


@pytest.mark.parametrize(
    "name",
    [
        "rank_deficient_complete_fit",
        "multi_penalty_support",
        "solver_repair_complete_fit",
        "c3_c1_complete_fit",
    ],
)
def test_imported_drivers_do_not_require_posix_modules(name):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib, os, sys; sys.modules['resource'] = None; "
            "os.process_cpu_count = os.cpu_count; "
            "[delattr(os, name) for name in ('getloadavg', 'sched_getaffinity') "
            "if hasattr(os, name)]; importlib.import_module(sys.argv[1])",
            f"benchmarks.{name}",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("platform", "raw", "unit"), [("linux", 3072, "kib"), ("darwin", 3145728, "bytes")]
)
def test_peak_rss_converts_posix_units(monkeypatch, platform, raw, unit):
    monitor = import_module("benchmarks._platform")
    monkeypatch.setattr(monitor, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setitem(
        sys.modules,
        "resource",
        SimpleNamespace(RUSAGE_SELF=0, getrusage=lambda who: SimpleNamespace(ru_maxrss=raw)),
    )
    peak = monitor.peak_rss()
    assert peak.bytes == 3145728
    assert peak.source == "resource.ru_maxrss"
    assert peak.ru_maxrss_unit == unit


def test_windows_peak_rss_uses_high_water_mark_not_current_rss(monkeypatch):
    monitor = import_module("benchmarks._platform")
    monkeypatch.setattr(monitor, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setitem(sys.modules, "resource", None)
    memory = SimpleNamespace(peak_wset=3145728, rss=1048576)
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(Process=lambda: SimpleNamespace(memory_info=lambda: memory)),
    )
    peak = monitor.peak_rss()
    assert peak.bytes == 3145728
    assert peak.source == "psutil.peak_wset"
    assert peak.ru_maxrss_unit is None


def test_missing_posix_load_and_affinity_are_unknown(monkeypatch):
    monitor = import_module("benchmarks._platform")
    monkeypatch.setattr(monitor, "os", SimpleNamespace(cpu_count=lambda: 6))
    assert monitor.load_average() is None
    assert monitor.cpu_affinity() is None
    assert monitor.available_cpu_count() == 6


def test_available_cpu_count_respects_affinity(monkeypatch):
    monitor = import_module("benchmarks._platform")
    monkeypatch.setattr(
        monitor,
        "os",
        SimpleNamespace(sched_getaffinity=lambda pid: {1, 3}, cpu_count=lambda: 6),
    )
    assert monitor.available_cpu_count() == 2


def test_c3_environment_snapshot_without_posix_load_or_affinity(monkeypatch):
    driver = import_module("benchmarks.c3_c1_complete_fit")
    monkeypatch.setattr(driver, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.delattr(driver.os, "getloadavg", raising=False)
    monkeypatch.delattr(driver.os, "sched_getaffinity", raising=False)
    snapshot = driver.environment_snapshot()
    assert snapshot["load_average"] is None
    assert snapshot["affinity"] is None
    assert snapshot["process_activity"] is None


@pytest.mark.parametrize("name", ["multi_penalty_support", "solver_repair_complete_fit"])
@pytest.mark.parametrize("threads", ["0", "-1", "1.5"])
def test_complete_fit_drivers_reject_invalid_threads(monkeypatch, tmp_path, capsys, name, threads):
    driver = import_module(f"benchmarks.{name}")
    case = "scalar" if name == "multi_penalty_support" else "qp"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            name,
            "--case",
            case,
            "--label",
            "test",
            "--out",
            str(tmp_path / "r.json"),
            "--threads",
            threads,
        ],
    )
    with pytest.raises(SystemExit) as rejected:
        driver.main()
    assert rejected.value.code == 2
    assert "--threads" in capsys.readouterr().err


def test_timing_load_guard_refuses_missing_load_average(monkeypatch, tmp_path, capsys):
    driver = import_module("benchmarks.multi_penalty_support")
    monkeypatch.delattr(import_module("os"), "getloadavg", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            driver.__file__,
            "--case",
            "scalar",
            "--label",
            "test",
            "--out",
            str(tmp_path / "r.json"),
            "--measure-time",
        ],
    )
    with pytest.raises(SystemExit) as rejected:
        driver.main()
    assert rejected.value.code == 2
    assert "load average is unavailable" in capsys.readouterr().err


def test_unmeasured_fit_records_unknown_load_without_posix_apis(monkeypatch, tmp_path):
    driver = import_module("benchmarks.multi_penalty_support")
    monitor = import_module("benchmarks._platform")
    monkeypatch.setattr(monitor, "os", SimpleNamespace(cpu_count=lambda: 6))
    frame = bench.pd.DataFrame({"x": [0.0, 1.0]})
    model = SimpleNamespace(fit=lambda *_args: None)
    monkeypatch.setattr(
        driver, "_fixture", lambda *_args, **_kwargs: (model, frame, bench.np.ones(2), {})
    )
    monkeypatch.setattr(driver, "_source_identity", lambda: {"source_digest": "test"})
    monkeypatch.setattr(driver, "_fit_outputs", lambda *_args: {"coefficients": [1.0]})
    path = tmp_path / "receipt.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            driver.__file__,
            "--case",
            "scalar",
            "--mode",
            "fixed",
            "--label",
            "test",
            "--out",
            str(path),
        ],
    )
    driver.main()
    receipt = json.loads(path.read_text())
    assert receipt["load_before"] is None and receipt["load_after"] is None
    assert receipt["available_cores"] == 6
    assert receipt["fit_seconds"] is None
    assert receipt["process_peak_rss_mib"] > 0
    assert receipt["outputs"] == {"coefficients": [1.0]}


def test_batched_compensation_dispatch_counts_entries_and_fallbacks():
    driver = import_module("benchmarks.multi_penalty_support")
    kernels = import_module("superglm.reml.multi_penalty")
    original = kernels._dot2_selected
    left = bench.np.array([[1.0, 2.0], [bench.np.nextafter(0.0, 1.0), 0.0]])
    right = bench.np.ones((2, 1))
    indices = bench.np.array([[0, 0], [1, 0]])
    with driver._kernel_dispatch() as observed:
        kernels._dot2_selected(left, right, indices)
    assert kernels._dot2_selected is original
    assert observed["calls"].get("_dot2_selected", 0) == 1
    assert observed["batched_dot2_results"] == {
        "selected_entries": 2,
        "native": 1,
        "fallback_requested": 1,
    }
    assert observed["batched_dot2_signatures"]


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=300,
    )


@pytest.mark.parametrize(
    ("flag", "value"),
    [("--seed", "-1"), ("--levels", "1"), ("--rows", "1"), ("--repeats", "0")],
)
def test_every_flag_is_validated_at_the_flag(flag: str, value: str) -> None:
    """A rejected input must name the flag, not die inside a library.

    ``--seed -1`` used to raise eight NumPy frames deep inside ``default_rng``,
    mentioning neither the flag nor the harness.
    """
    # Do not append a second --repeats: argparse keeps the last occurrence, so
    # it would silently overwrite the value under test.
    extra = () if flag == "--repeats" else ("--repeats", "1")
    result = _run(flag, value, *extra)

    assert result.returncode != 0, f"{flag} {value} was accepted"
    assert flag in result.stderr, (
        f"rejection did not name {flag}; stderr was:\n{result.stderr[-500:]}"
    )


def test_row_floor_is_coupon_collector_not_pigeonhole() -> None:
    """``rows // 2 >= levels`` is the wrong bound and blesses a smaller design.

    Levels are drawn uniformly, so realizing all ``L`` of them takes about
    ``L * ln L`` draws, not ``L``.  At 41 levels the old bound accepted 82 rows,
    where 41 training draws realize roughly 26 distinct levels -- and the run
    then reported ``levels: 41``.
    """
    result = _run("--levels", "41", "--rows", "82", "--repeats", "1")

    assert result.returncode != 0, "82 rows for 41 levels was accepted"
    assert "--rows" in result.stderr
    # The message must state the real requirement, not the pigeonhole one.
    assert "82" in result.stderr


def test_payload_identifies_the_tree_that_produced_it() -> None:
    """Two payloads from the same source must not be labellable before/after.

    ``baseline_commit`` and ``branch_src_commit`` beside the committed artifact
    are typed in by hand.  Without provenance read from the tree itself, two
    runs of identical code satisfy every invariant this suite asserts.
    """
    payload = bench.measure(levels=4, rows=400, repeats=1, seed=31337)
    provenance = payload["provenance"]

    assert provenance["superglm_version"], "no version recorded"
    assert provenance["superglm_path"], "no import path recorded"
    # git fields may be None off a checkout, but the keys must exist so a
    # consumer can tell "unknown" from "not recorded at all".
    assert "git_commit" in provenance
    assert "git_dirty" in provenance


def test_provenance_points_at_the_tree_actually_imported() -> None:
    """Read from the package, not from a flag, so it cannot be mislabelled."""
    import superglm

    provenance = bench.measure(levels=4, rows=400, repeats=1, seed=31337)["provenance"]

    assert Path(provenance["superglm_path"]) == Path(superglm.__file__).resolve().parent
