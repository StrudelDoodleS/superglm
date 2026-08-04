"""The harness must reject inputs it cannot honour, and label its own output.

This benchmark's numbers are quoted in pull request bodies as evidence a change
is neutral or faster.  An instrument used that way has to refuse what it cannot
measure, and has to say which tree it measured.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from benchmarks import rank_deficient_complete_fit as bench

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "benchmarks" / "rank_deficient_complete_fit.py"


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
