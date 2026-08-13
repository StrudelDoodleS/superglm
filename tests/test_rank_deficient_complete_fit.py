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
import time

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

# The long side of the footprint comparison. A per-tick store would hold this
# many entries, so it breaks both bounds below by a wide margin, while the wait
# stays short: 100 ticks at `interval=0.001` is 0.15 s on an idle machine.
_LONG_TICKS = 100

# How long a starved sampler is given to reach a tick target before the
# precondition is reported as unmet. It bounds only the FAILING path. The
# budget is `_LONG_TICKS` times the worst per-tick cost this could be
# reproduced at -- 175 ms, under two CPU-bound threads on a four-CPU cgroup,
# against 1.5 ms idle and the 50 ms of the hosted-runner failure this replaced
# -- which is 18 s, with the rest as margin for a slower machine.
_TICK_TIMEOUT = 120.0


def _hold_open_until(sampler, target_samples: int, timeout: float = _TICK_TIMEOUT) -> None:
    """Run `sampler` until it has recorded `target_samples` ticks, then stop it.

    The sampler ticks on a background thread, so "sleep 0.5 s and you will have
    ten times the ticks of a 0.05 s sleep" is an assumption about the OS
    scheduler, not a measurement -- and it is false under contention. A hosted
    runner recorded 25 ticks for the 0.05 s window and 10 for the 0.5 s one
    (50 ms per tick against 2 ms), inverting the ratio the tests asserted;
    locally, two CPU-bound threads on a four-CPU cgroup reproduce the inversion
    in 4 runs out of 4.

    Waiting on the tick COUNT makes the precondition something these tests
    establish rather than something they hope for. Contention then makes them
    slower, never wrong, and a sampler that has genuinely stopped ticking is
    reported as exactly that.
    """
    deadline = time.monotonic() + timeout
    with sampler:
        while sampler.samples < target_samples and time.monotonic() < deadline:
            time.sleep(0.002)
    assert sampler.samples >= target_samples, (
        f"sampler recorded {sampler.samples} of {target_samples} ticks in {timeout:.0f}s"
    )


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


def test_the_sampler_records_every_fit_not_just_the_first() -> None:
    """`--repeats` defaults to 3 and the sampler is entered once per fit.

    A stop flag set in `__exit__` and never cleared in `__enter__` makes the
    second and third fits record nothing, silently -- the payload still reports
    a sample count, just one from the first fit only.  `samples` is cumulative
    across re-entries, so a sampler that stopped recording never reaches the
    next target and is reported as stalled rather than as slow.
    """
    sampler = bench._DispatchSampler(interval=0.001)
    counts = []
    for _ in range(3):
        # RELATIVE to the count on entry, not an absolute ladder. `samples` is
        # cumulative, so absolute targets (5, 10, 15) are satisfied on ENTRY by
        # one overshooting first wait -- the poll is 2 ms against a 1 ms
        # interval, and a delayed wake-up is the same contention this rewrite
        # exists to survive. The later holds would then skip their `while` body
        # entirely, and `counts[i] > counts[i-1]` would be left to the race
        # between a freshly started sampler thread and the `_stop` flag that
        # `__exit__` sets microseconds later: `__exit__` sets the flag and then
        # JOINS (`benchmarks/rank_deficient_complete_fit.py`), so a thread that
        # loses that race records nothing at all and the assertion fires on a
        # sampler that is working. Asking each re-entry for five ticks OF ITS
        # OWN is what the docstring claims and what the assertions below need.
        _hold_open_until(sampler, sampler.samples + 5)
        counts.append(sampler.samples)
    assert counts[0] > 0
    # each re-entry must add samples, not sit at the first fit's total
    assert counts[1] > counts[0], f"second fit recorded nothing: {counts}"
    assert counts[2] > counts[1], f"third fit recorded nothing: {counts}"


def test_the_sampler_footprint_does_not_grow_with_the_fit_it_measures() -> None:
    """Its own retention lands in the peak RSS this benchmark reports.

    Keeping a snapshot per tick makes the sampler's memory proportional to the
    RUNTIME of the side being measured, so the slower side carries a larger
    term -- an asymmetry biased toward whichever side is faster, in the one
    figure the comparison publishes as a memory result.

    The long side is defined by the TICKS it took, not by how long it slept:
    ticks are what a per-tick store would retain, and a sleep only buys them on
    an idle machine.
    """
    # The ratio has to be ESTABLISHED, not hoped for: `_LONG_TICKS` fixes
    # `long >= 100` and `short >= 5`, which leaves `long > short * 3` true only
    # while `short` stays at or under 33, and `short`'s own wait bounds it
    # below, never above.
    #
    # Scaling the long target with `short.samples` would establish it, but at
    # the cost of making the long side's WORK scale with a quantity that has no
    # upper bound -- `short.samples` is bounded below by its own wait and not
    # bounded above at all. That trades a wrong assertion for a misattributed
    # timeout: the long sampler is failed by `_TICK_TIMEOUT` for a stall on the
    # short side, while both samplers are behaving. The bound is the point, not
    # any particular overshoot: at the 50 ms/tick this file documents from a
    # hosted runner, a short side of merely 800 ticks already exceeds the 120 s
    # budget.  (A short side in the thousands is reachable only by making a
    # tick artificially cheap -- forcing it that way is how the branch was
    # demonstrated -- but nothing has to be reachable for an unbounded target
    # to be the wrong shape.)
    #
    # A short side that overshot a 5-tick target by orders of magnitude is a
    # broken measurement, not a reason to triple the long side's work, so it is
    # re-taken instead. 33 is `_LONG_TICKS // 3`, the largest short count the
    # fixed long target can still dominate: 33 * 3 = 99 < 100.
    ceiling = _LONG_TICKS // 3
    for _ in range(3):
        short = bench._DispatchSampler(interval=0.001)
        _hold_open_until(short, 5)
        if short.samples <= ceiling:
            break
    assert short.samples <= ceiling, (
        f"short sampler overshot its 5-tick target to {short.samples} on every attempt "
        f"(ceiling {ceiling}); the poll stalled, so this fixture cannot establish the ratio"
    )
    long = bench._DispatchSampler(interval=0.001)
    _hold_open_until(long, _LONG_TICKS)

    # established by the two waits above, restated as the premise of the rest
    assert long.samples > short.samples * 3, "fixture did not produce a longer run"
    # the store is keyed by configuration, so a 20x longer run must not make it
    # meaningfully bigger
    assert len(long._dwell) <= len(short._dwell) + 2
    assert len(long._dwell) < 20


def test_the_sampler_reports_dwell_not_just_presence() -> None:
    """Seen once in twelve thousand samples and seen throughout are not the same claim.

    Discarding the count cannot tell a brief startup window from a
    configuration that held for the whole fit, which is exactly the
    distinction this benchmark got wrong once.
    """
    sampler = bench._DispatchSampler(interval=0.001)
    _hold_open_until(sampler, 10)
    observed = sampler.observed()
    assert observed
    for pool in observed:
        assert pool["samples_seen_in"] >= 1
        assert 0.0 < pool["fraction_of_samples"] <= 1.0
        assert pool["samples_seen_in"] <= sampler.samples


def test_the_sampler_counts_duplicate_pool_metadata_once_per_tick(monkeypatch) -> None:
    """Two loaded libraries may report one identical BLAS configuration.

    SciPy 1.18 and NumPy can each load an OpenBLAS library whose selected
    threadpool metadata is identical.  The sampler reports configuration dwell,
    not library-instance dwell, so one tick must not count that key twice and
    produce a fraction greater than one.
    """
    pool = {
        "user_api": "blas",
        "internal_api": "openblas",
        "prefix": "libscipy_openblas",
        "version": "0.3.31.dev",
        "threading_layer": "pthreads",
        "num_threads": 1,
    }
    monkeypatch.setattr(bench, "_pool_snapshot", lambda: [pool.copy(), pool.copy()])
    sampler = bench._DispatchSampler(interval=0.001)
    _hold_open_until(sampler, 5)

    observed = sampler.observed()
    assert sampler.samples > 0
    assert len(observed) == 1
    assert observed[0]["samples_seen_in"] == sampler.samples
    assert observed[0]["fraction_of_samples"] == 1.0


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
    assert len(payload["timing_seconds"]["all"]) == TINY["repeats"]
    assert payload["timing_seconds"]["min"] <= payload["timing_seconds"]["median"]


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
            # dwell is what separates a startup window from a phase that held
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
