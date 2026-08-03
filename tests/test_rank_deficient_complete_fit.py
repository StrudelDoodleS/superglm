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

# Small enough to run in a test, deficient enough to exercise the path: a
# 4-level pair on 200 rows still leaves empty joint cells.
TINY = {"levels": 4, "rows": 200, "repeats": 1, "seed": 31337}


@pytest.fixture(scope="module")
def payload() -> dict:
    return bench.measure(**TINY)


def test_the_sampler_records_every_fit_not_just_the_first() -> None:
    """`--repeats` defaults to 3 and the sampler is entered once per fit.

    A stop flag set in `__exit__` and never cleared in `__enter__` makes the
    second and third fits record nothing, silently -- the payload still reports
    a sample count, just one from the first fit only.
    """
    sampler = bench._DispatchSampler(interval=0.001)
    counts = []
    for _ in range(3):
        with sampler:
            time.sleep(0.05)
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
    """
    short = bench._DispatchSampler(interval=0.001)
    with short:
        time.sleep(0.05)
    long = bench._DispatchSampler(interval=0.001)
    with long:
        time.sleep(0.5)

    assert long.samples > short.samples * 3, "fixture did not produce a longer run"
    # the store is keyed by configuration, so a 10x longer run must not make it
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
    with sampler:
        time.sleep(0.1)
    observed = sampler.observed()
    assert observed
    for pool in observed:
        assert pool["samples_seen_in"] >= 1
        assert 0.0 < pool["fraction_of_samples"] <= 1.0
        assert pool["samples_seen_in"] <= sampler.samples


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
