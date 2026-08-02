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

import pytest
from benchmarks import rank_deficient_complete_fit as bench

# Small enough to run in a test, deficient enough to exercise the path: a
# 4-level pair on 200 rows still leaves empty joint cells.
TINY = {"levels": 4, "rows": 200, "repeats": 1, "seed": 31337}


@pytest.fixture(scope="module")
def payload() -> dict:
    return bench.measure(**TINY)


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
    [("--repeats", "0"), ("--repeats", "-1"), ("--levels", "1"), ("--rows", "1")],
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

    Values are deliberately not asserted -- they are measurements and will move
    with the machine.  What must hold is that the two sides remain comparable.
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
    assert set(baseline["numerical_outputs"]) == set(branch["numerical_outputs"])
    assert record["summary"]["numerical_outputs_identical"] is True
