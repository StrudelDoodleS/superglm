"""Guards for the complete-fit comparison's own payload.

This benchmark is the evidence AGENTS.md requires for the alias-representative
change, and it has now had five defects found by review rather than by anything
in the repository: the BLAS column reported build metadata, the thread count was
sampled after the fit rather than during it, peak RSS covered three fits on one
side and one on the other, `ru_maxrss` was divided as KiB unconditionally, and
`--repeats 0` died on an unrelated assertion.

Every one of those is a property of the PAYLOAD rather than of a number in it,
which is why nothing caught them: the numbers all looked plausible.  So these
tests assert the invariants a reader of the artifact is entitled to rely on --
that the two sides are comparable, that each field measures what its name says,
and that the measurement happened when it claims to have happened.
"""

from __future__ import annotations

import json
import resource
import subprocess
import sys
import threading
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from benchmarks import rank_deficient_complete_fit as bench

# Small enough to run in a test, and ACTUALLY deficient. The previous fixture
# (4 levels, 200 rows) was not: measured `parameters` 15, `data_rank` 15, both
# methods `cholesky`, both representative flags False, `n_zero` 0. It was fully
# determined, so every assertion below ran against the dense path while the
# committed artifact -- and the benchmark's whole reason to exist -- takes
# `qr_svd` at `data_rank` 1627 of 1680.
#
# (41, 800) measures `parameters` 1680, `data_rank` 344, `qr_svd`, `n_zero`
# 1336, and realizes all 41 levels of both factors. It also satisfies the CLI's
# coupon-collector row floor, so the fixture and the flag agree.
TINY = {"levels": 41, "rows": 800, "repeats": 1, "seed": 31337}


def test_the_fixture_reaches_the_deficient_path_it_claims_to_measure(payload) -> None:
    """Guard the fixture itself, not just the harness.

    A benchmark whose test data never reaches the branch it was written to
    measure reports on something else entirely, and nothing else here would
    notice: every other assertion in this file passes on a full-rank design.
    """
    dispatch = payload["backend_dispatch"]
    assert dispatch["data_rank"] < payload["configuration"]["parameters"], (
        "fixture is full rank; it cannot exercise the rank-deficient path"
    )
    assert dispatch["data_method"] != "cholesky"
    assert payload["numerical_outputs"]["n_zero"] > 0


def test_the_configuration_reports_the_levels_the_design_realizes(payload) -> None:
    """`levels` is what was REQUESTED; a uniform draw may realize fewer.

    At 41 levels the old row guard blessed 82 rows, where 41 draws realize
    about 26 distinct levels -- so the payload said `levels: 41` for a design
    that was really 26x26.
    """
    realized = payload["configuration"]["levels_realized"]
    assert realized["g"] == realized["h"] == payload["configuration"]["levels"]


@pytest.fixture(scope="module")
def payload() -> dict:
    return bench.measure(**TINY)


def _pool(**changes):
    return {
        "user_api": "blas",
        "internal_api": "openblas",
        "prefix": "libscipy_openblas",
        "version": "test",
        "threading_layer": "pthreads",
        "num_threads": 1,
        **changes,
    }


def _unexpected_observer(*_args, **_kwargs):
    raise AssertionError("Timed fits must not construct or invoke observers.")


def test_pool_observations_require_explicit_events_on_the_fitting_thread(monkeypatch):
    thread = threading.get_ident()
    snapshots = []

    def snapshot():
        snapshots.append(threading.get_ident())
        return [_pool()]

    monkeypatch.setattr(bench, "_pool_snapshot", snapshot)
    monkeypatch.setattr(threading, "Thread", _unexpected_observer)
    sampler = bench._DispatchSampler()
    with sampler:
        assert sampler.samples == 0
        sampler.sample("kernel:call")
        assert snapshots == [thread]
        with monkeypatch.context() as foreign:
            foreign.setattr(bench, "get_ident", lambda: thread + 1)
            with pytest.raises(RuntimeError, match="active fitting thread"):
                sampler.sample("kernel:return")
        with pytest.raises(RuntimeError, match="already active"):
            sampler.__enter__()
    with pytest.raises(RuntimeError, match="active fitting thread"):
        sampler.sample("kernel:return")
    assert snapshots == [thread]


def test_pool_sampling_restarts_its_event_schedule_for_each_fit(monkeypatch):
    monkeypatch.setattr(bench, "_pool_snapshot", lambda: [_pool()])
    sampler = bench._DispatchSampler()
    for fit in range(1, 4):
        with sampler:
            for _ in range(8):
                sampler.sample("kernel:call")
        assert sampler.samples == 4 * fit  # events 1, 2, 4, 8 in each fit
    observation = bench._native_pool_receipt(sampler)["native_pool_observation"]
    assert observation["fit_entries"] == observation["fit_entries_with_samples"] == 3
    assert observation["event_counts"] == {"kernel:call": 24}
    assert observation["sampled_event_counts"] == {"kernel:call": 12}


def test_pool_observation_storage_and_snapshot_attempts_are_bounded(monkeypatch):
    calls = []

    def snapshot():
        calls.append(None)
        return [_pool(num_threads=len(calls))]

    monkeypatch.setattr(bench, "_pool_snapshot", snapshot)
    sampler = bench._DispatchSampler(max_samples=5, max_configurations=2, max_event_names=2)
    with sampler:
        for _ in range(1024):
            sampler.sample("kernel:call")
        sampler.sample("kernel:return")
        for index in range(100):
            sampler.sample(f"unretained:{index}")
    observation = bench._native_pool_receipt(sampler)["native_pool_observation"]
    assert sampler.samples == sampler.sample_attempts == len(calls) == 5
    assert len(sampler.observed()) == 2
    assert len(observation["event_counts"]) == 2
    assert len(observation["sampled_event_counts"]) <= 2
    assert len(sampler._session_events) <= 2
    assert observation["dropped_configuration_observations"] == 3
    assert observation["dropped_event_notifications"] == 100
    assert observation["eligible_events_skipped_at_sample_cap"] == 7


def test_pool_frequency_counts_selected_events_and_deduplicates_metadata(monkeypatch):
    monkeypatch.setattr(bench, "_pool_snapshot", lambda: [_pool(), _pool()])
    sampler = bench._DispatchSampler()
    with sampler:
        for _ in range(8):
            sampler.sample("kernel:call")
    receipt = bench._native_pool_receipt(sampler)
    assert receipt["native_pool_samples"] == 4
    assert receipt["native_pool_observation"]["events_seen"] == 8
    assert "not elapsed-time dwell" in receipt["native_pool_observation"]["semantics"]
    assert receipt["native_pools_during_fit"] == [
        {**_pool(), "samples_seen_in": 4, "fraction_of_samples": 1.0}
    ]


def test_failed_pool_enumerations_consume_the_attempt_cap(monkeypatch):
    attempts = []

    def failure():
        attempts.append(None)
        raise RuntimeError("enumeration failed")

    monkeypatch.setattr(bench, "_pool_snapshot", failure)
    sampler = bench._DispatchSampler(max_samples=3)
    with sampler:
        for _ in range(1024):
            sampler.sample("kernel:call")
    receipt = bench._native_pool_receipt(sampler)
    observation = receipt["native_pool_observation"]
    assert len(attempts) == observation["sample_attempts"] == observation["error_count"] == 3
    assert observation["eligible_events_skipped_at_sample_cap"] == 8
    assert receipt["native_pool_samples"] == 0
    assert observation["status"] == "observer_error"
    assert observation["errors"] == ["RuntimeError: enumeration failed"] * 3


@pytest.mark.parametrize("events", [0, 1])
def test_missing_pool_evidence_is_reported_explicitly(monkeypatch, events):
    monkeypatch.setattr(bench, "_pool_snapshot", lambda: [])
    sampler = bench._DispatchSampler()
    with sampler:
        for _ in range(events):
            sampler.sample("kernel:call")
    status = bench._native_pool_receipt(sampler)["native_pool_observation"]["status"]
    assert status == ("no_native_pools_observed" if events else "no_solver_events_observed")
    assert bench._native_pool_receipt(None) == {
        "native_pools_during_fit": [],
        "native_pool_samples": 0,
        "native_pool_observation": {"status": "not_observed_in_timed_fit"},
    }


@pytest.mark.parametrize("raises", [False, True])
def test_pool_profile_observes_actual_kernel_events_and_restores_on_failure(monkeypatch, raises):
    monkeypatch.setattr(bench, "_pool_snapshot", lambda: [_pool()])
    sampler = bench._DispatchSampler()

    def kernel():
        if raises:
            raise ValueError("kernel failed")
        return 3

    previous = sys.getprofile()
    try:
        with sampler, bench._native_pool_dispatch(sampler, {"kernel": kernel}):
            if raises:
                with pytest.raises(ValueError, match="kernel failed"):
                    kernel()
            else:
                assert kernel() == 3
        assert sys.getprofile() is previous
    finally:
        sys.setprofile(previous)
    assert sampler._events == {"kernel:call": 1, "kernel:return": 1}
    assert sampler.samples == 2


@pytest.mark.parametrize("repeats", [1, 3])
def test_timed_rank_fit_has_no_observer_and_keeps_first_use_inside_clock(monkeypatch, repeats):
    events = []
    clock_active = False
    initialized = False
    ticks = iter(float(index) for index in range(2 * repeats))
    frame = bench.pd.DataFrame({"g": ["a", "b"], "h": ["b", "a"]})
    response = bench.np.array([1.0, 2.0])
    decomposition = SimpleNamespace(method="test", rank=1, pivots=None)

    class Model:
        def __init__(self, **_kwargs):
            self._dm = SimpleNamespace(p=1)
            self.result = SimpleNamespace(
                beta=bench.np.array([1.0]),
                effective_df=1.0,
                deviance=2.0,
                n_iter=1,
                converged=True,
                rank_info=SimpleNamespace(
                    data=decomposition, augmented=decomposition, coefficient=decomposition
                ),
            )

        def fit(self, *_args):
            nonlocal initialized
            assert clock_active
            events.append("fit")
            if not initialized:
                events.append("first_use")
                initialized = True

        def predict(self, _frame):
            assert not clock_active
            return response

    def clock():
        nonlocal clock_active
        clock_active = not clock_active
        events.append("clock_start" if clock_active else "clock_stop")
        return next(ticks)

    monkeypatch.setattr(bench, "SuperGLM", Model)
    monkeypatch.setattr(bench, "_design", lambda *_args: (frame, response))
    monkeypatch.setattr(bench, "_provenance", lambda: {})
    monkeypatch.setattr(bench.time, "perf_counter", clock)
    for name in ("_DispatchSampler", "_native_pool_dispatch", "_pool_snapshot"):
        monkeypatch.setattr(bench, name, _unexpected_observer)
    receipt = bench.measure(2, 4, repeats, 1, measure_time=True)
    assert events == ["clock_start", "fit", "first_use", "clock_stop"] + [
        "clock_start",
        "fit",
        "clock_stop",
    ] * (repeats - 1)
    assert receipt["wall_time_status"] == "measured"
    assert receipt["timing_seconds"] == {"min": 1.0, "median": 1.0, "all": [1.0] * repeats}
    blas = receipt["backend_dispatch"]["blas"]
    assert not blas["sampled_during_fit"]
    assert blas["samples"] == 0 and blas["pools_during_fit"] == []
    assert blas["observation"]["status"] == "not_observed_in_timed_fit"


@pytest.mark.parametrize(
    ("driver_name", "case", "mode"),
    [
        ("multi_penalty_support", "scalar", "fixed"),
        ("multi_penalty_support", "scalar", "reml"),
        ("multi_penalty_support", "gamma", "fixed"),
        ("multi_penalty_support", "gamma", "reml"),
        ("solver_repair_complete_fit", "qp", "reml"),
    ],
)
def test_timed_complete_fit_drivers_have_no_observer(
    monkeypatch, tmp_path, driver_name, case, mode
):
    from importlib import import_module

    driver = import_module(f"benchmarks.{driver_name}")
    events = []
    clock_active = False
    ticks = iter([10.0, 11.0])
    frame = bench.pd.DataFrame({"x": [0.0, 1.0]})
    response = bench.np.array([1.0, 2.0])

    class Model:
        def fit(self, *_args, **_kwargs):
            assert clock_active
            events.extend(["fit", "first_use"])

        fit_reml = fit

        def diagnose(self):
            return SimpleNamespace(to_dict=lambda: {})

    def clock():
        nonlocal clock_active
        clock_active = not clock_active
        events.append("clock_start" if clock_active else "clock_stop")
        return next(ticks)

    def outputs(*_args):
        assert not clock_active
        events.append("outputs")
        return {"coefficients": [1.0]}

    def rss(*_args):
        assert not clock_active
        events.append("rss")
        return SimpleNamespace(ru_maxrss=1024.0)

    fixture = (
        (Model(), frame, response, {})
        if driver_name == "multi_penalty_support"
        else (Model(), frame, response, None, None, {})
    )
    monkeypatch.setattr(driver, "_fixture", lambda *_args, **_kwargs: fixture)
    monkeypatch.setattr(driver, "_source_identity", lambda: {"source_digest": "test"})
    monkeypatch.setattr(driver, "_fit_outputs", outputs)
    monkeypatch.setattr(driver.time, "perf_counter", clock)
    monkeypatch.setattr(driver.resource, "getrusage", rss)
    monkeypatch.setattr(driver, "threadpool_limits", lambda **_kwargs: nullcontext())
    monkeypatch.setattr(driver.os, "getloadavg", lambda: (0.0, 0.0, 0.0))
    monkeypatch.setattr(driver, "_DispatchSampler", _unexpected_observer)
    observer = "_kernel_dispatch" if driver_name == "multi_penalty_support" else "_solver_dispatch"
    monkeypatch.setattr(driver, observer, _unexpected_observer)
    monkeypatch.setattr(bench, "_pool_snapshot", _unexpected_observer)
    path = tmp_path / "receipt.json"
    argv = [
        driver.__file__,
        "--case",
        case,
        "--label",
        "test",
        "--out",
        str(path),
        "--measure-time",
    ]
    if driver_name == "multi_penalty_support":
        argv.extend(["--mode", mode])
    monkeypatch.setattr(sys, "argv", argv)
    driver.main()
    receipt = json.loads(path.read_text())
    assert events == ["clock_start", "fit", "first_use", "clock_stop", "rss", "outputs"]
    assert receipt["fit_seconds"] == 1.0  # synthetic clock; no performance measurement
    assert receipt["native_pool_samples"] == 0 and receipt["native_pools_during_fit"] == []
    assert receipt["native_pool_observation"]["status"] == "not_observed_in_timed_fit"
    assert receipt["outputs"] == {"coefficients": [1.0]}


def test_peak_memory_records_how_many_fits_it_covers(payload: dict) -> None:
    """`ru_maxrss` is a process high-water mark, so it only compares like for like.

    The published comparison once had `repeats: 1` on the baseline against
    `repeats: 3` on the branch while claiming both were one fit in a fresh
    interpreter.  The mark is only meaningful beside the count of fits it
    covers, so the payload has to carry that count rather than leave a reader
    to infer it from `configuration`.
    """
    assert payload["memory"]["peak_rss_measures_fits"] == TINY["repeats"]
    assert payload["memory"]["peak_rss_measures_fits"] == payload["configuration"]["repeats"]
    assert payload["memory"]["peak_rss_mib"] > 0.0


def test_the_memory_unit_matches_the_platform(payload: dict) -> None:
    """`ru_maxrss` is KiB on Linux and BYTES on macOS.

    Dividing by 1024 unconditionally is right on one and 1024x wrong on the
    other, while still labelled MiB.  The payload names the unit it converted
    from so the conversion can be checked rather than trusted.
    """
    unit = payload["memory"]["ru_maxrss_unit"]
    assert unit in {"bytes", "kib"}
    assert unit == ("bytes" if sys.platform == "darwin" else "kib")

    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024.0**2 if unit == "bytes" else 1024.0
    # the reported figure is this process's own mark, so it must be in range
    assert payload["memory"]["peak_rss_mib"] == pytest.approx(raw / divisor, rel=0.5)


def test_the_thread_count_was_sampled_during_a_fit(payload: dict) -> None:
    """Sampling after the loop reports the ambient value, not the fit's.

    superglm changes BLAS thread counts inside the solver and restores them on
    the way out, so a reading taken at payload-construction time describes the
    process between fits.  The payload has to say when it looked, and it has to
    have looked while a fit was running.
    """
    blas = payload["backend_dispatch"]["blas"]
    assert blas["sampled_during_fit"] is True
    assert blas["samples"] >= 1
    assert blas["pools_during_fit"], "no BLAS pool observed while the fit ran"
    observation = blas["observation"]
    assert observation["status"] == "observed_at_solver_events"
    assert observation["events_seen"] > 0 and observation["error_count"] == 0
    assert any(
        observation["event_counts"].get(f"{name}:call", 0) > 0
        for name in ("decompose_gram", "decompose_factor")
    )


def test_dispatch_comes_from_a_live_process_not_build_metadata(payload: dict) -> None:
    """The one field AGENTS.md names by hand, and the one that was wrong twice.

    `np.show_config()` answers "what was this wheel compiled against" and would
    print the same string on a machine dispatching elsewhere.  A live reading
    carries a filepath and a threading layer; build metadata carries neither.
    """
    pools = payload["backend_dispatch"]["blas"]["pools_during_fit"]
    assert any(pool["user_api"] == "blas" for pool in pools)
    for pool in pools:
        assert pool["prefix"], "a loaded pool always has a library prefix"
        assert pool["num_threads"] is not None
    # build metadata has no notion of a running thread count
    assert all("build" not in str(key).lower() for key in payload["backend_dispatch"])


def test_the_payload_says_what_it_measured_on(payload: dict) -> None:
    """Provenance a reader needs before comparing two of these files."""
    configuration = payload["configuration"]
    assert configuration["levels"] == TINY["levels"]
    assert configuration["rows"] == TINY["rows"]
    assert configuration["train_rows"] == TINY["rows"] // 2
    assert configuration["parameters"] > 0
    assert payload["backend_dispatch"]["python"]
    assert payload["wall_time_status"] == "unmeasured"
    assert payload["timing_seconds"] == {"min": None, "median": None, "all": []}


def test_the_numerical_outputs_are_the_ones_a_comparison_would_diff(payload: dict) -> None:
    """If two runs agree here they agree on the fit, not merely on its speed."""
    outputs = payload["numerical_outputs"]
    assert set(outputs) == {
        "effective_df",
        "deviance",
        "beta_l2",
        "n_zero",
        "zero_index_sum",
        "n_iter",
    }
    assert outputs["effective_df"] > 0
    assert outputs["n_iter"] >= 1


def test_two_runs_of_the_same_configuration_are_comparable(payload: dict) -> None:
    """The whole artifact is a diff of two payloads, so the diff has to be sound.

    Everything except timing and memory must be reproducible; if it is not, an
    apparent difference between baseline and branch could be noise rather than
    the change.
    """
    again = bench.measure(**TINY)
    assert again["numerical_outputs"] == payload["numerical_outputs"]
    assert again["configuration"] == payload["configuration"]
    for field, value in payload["backend_dispatch"].items():
        if field == "blas":
            continue
        assert again["backend_dispatch"][field] == value


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--repeats", "0"),
        ("--repeats", "-1"),
        ("--levels", "1"),
        ("--rows", "1"),
        # clears the >= 2 floor and still cannot build the design: half of 2
        # rows is one, against 41 levels per factor
        ("--rows", "2"),
        ("--rows", "80"),
    ],
)
def test_a_dimension_that_cannot_run_is_refused_at_the_flag(flag: str, value: str) -> None:
    """`--repeats 0` skipped the loop and died on `assert model is not None`."""
    completed = subprocess.run(
        [sys.executable, bench.__file__, flag, value],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode != 0
    assert flag in completed.stderr or flag in completed.stdout


def test_the_artifact_on_disk_still_satisfies_its_own_invariants() -> None:
    """The committed comparison is the thing the PR cites, so check the file.

    Measurements are deliberately not pinned -- they move with the machine.
    What must hold is that the two sides remain comparable, and that every
    claim the summary and the history make ABOUT the measurements re-derives
    from the measurements themselves.  A flag the record asserts about itself
    is the shape of check this file's docstring exists to warn against: the
    record may move, but it may not disagree with itself.
    """
    path = "benchmarks/results/rank_deficient_complete_fit.json"
    with open(path) as handle:
        record = json.load(handle)

    baseline, branch = record["baseline"], record["branch"]
    assert (
        baseline["memory"]["peak_rss_measures_fits"] == branch["memory"]["peak_rss_measures_fits"]
    ), "peak RSS covers a different number of fits on each side"
    assert baseline["configuration"]["repeats"] == branch["configuration"]["repeats"]
    for side in (baseline, branch):
        assert side["memory"]["ru_maxrss_unit"] in {"bytes", "kib"}
        assert side["backend_dispatch"]["blas"]["sampled_during_fit"] is True
        assert side["backend_dispatch"]["blas"]["pools_during_fit"]
        for pool in side["backend_dispatch"]["blas"]["pools_during_fit"]:
            # Preserve internal counts in this historical observation artifact.
            assert pool["samples_seen_in"] >= 1
            assert 0.0 < pool["fraction_of_samples"] <= 1.0
    # The summary's claims, re-derived from the two sides rather than read back
    # from the summary: `numerical_outputs_identical` is the flag that turns
    # the timing difference into "same answer, less work", so it is checked
    # against the values, not against its own say-so.
    assert baseline["numerical_outputs"] == branch["numerical_outputs"]
    assert record["summary"]["numerical_outputs_identical"] is True
    baseline_route = {k: v for k, v in baseline["backend_dispatch"].items() if k != "blas"}
    branch_route = {k: v for k, v in branch["backend_dispatch"].items() if k != "blas"}
    assert (baseline_route == branch_route) is record["summary"]["decomposition_route_identical"]
    assert record["summary"]["peak_rss_delta_mib"] == pytest.approx(
        round(branch["memory"]["peak_rss_mib"] - baseline["memory"]["peak_rss_mib"], 1)
    )

    # The history is the artifact's most-quoted content, so it may not disagree
    # with its own rows or with the payloads committed beside it: every row
    # carries its load context, every ratio re-derives from its own seconds,
    # and the newest row IS the two payloads above.
    history = record["history"]
    assert [row["run"] for row in history] == [1, 2, 3, 4, 5]
    for row in history:
        assert row["one_minute_loadavg"], f"run {row['run']} carries no load context"
        assert row["ratio"] == pytest.approx(
            round(row["baseline_seconds"] / row["branch_seconds"], 1)
        ), f"run {row['run']}'s ratio does not derive from its own seconds"
    assert history[-1]["baseline_seconds"] == baseline["timing_seconds"]["min"]
    assert history[-1]["branch_seconds"] == branch["timing_seconds"]["min"]
    # the published claim names no single multiplier; the rows carry those
    assert record["summary"]["speedup_claim"] == "tens of times faster"
    assert record["summary"]["speedup_note"]
