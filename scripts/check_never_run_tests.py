"""Fail when a test was skipped on every CI platform: it passed in no shard and checked nothing.

The compatibility shards upload their junit reports (artifact
``pytest-<os>-<arch>-py<version>-<group>``). This reads them all and fails on any
test that no report ran, unless the script names the workflow that does run it
(``RUN_ELSEWHERE`` by node id; ``SKIPPED_FOR`` by the skip's message, for a test
in a module that workflow names, and only when every skip of it has that
message), or says that nothing runs it and why. Those are listed in the summary,
so a test no job runs is counted, never passed silently. It also fails unless
each of the ``--shards`` shards reported at least one test, so a shard whose
report is missing or empty cannot hide a skip.

    python scripts/check_never_run_tests.py junit-dir --shards 20

An expected failure (``xfail``) ran and failed as declared; it is not a skip, but
one marked ``run=False`` (``[NOTRUN]``) never ran. A module skipped at collection
on one platform ran if any of its tests ran elsewhere.
"""

from __future__ import annotations

import argparse
import functools
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from update_test_durations import node_id

_ROOT = Path(__file__).resolve().parents[1]
_NO_R = "nowhere: needs R with gamlss and jsonlite, which no CI job installs"
# Where the tests a skip message names do run, by the message's prefix: the
# workflow, which must name the test's module, and what it does.
SKIPPED_FOR = {
    "[dataset] ": (
        ".github/workflows/real-data.yml",
        "the Real data workflow, which fetches freMTPL2 and fails on this skip",
    ),
}
# Where a test does run, by node id (a module skipped at collection by its path).
RUN_ELSEWHERE = {
    "tests/docs/test_notebooks.py": "the docs workflow, which executes the notebooks",
    "tests/test_generalized_gamma_lss_vertical_slice.py::"
    "test_location_form_matches_gamlss_gg_at_a_parametric_specification": _NO_R,
    "tests/test_generalized_pareto_lss_vertical_slice.py::"
    "test_the_fit_matches_gamlss_gp_at_a_parametric_specification": _NO_R,
    "tests/test_log_normal_ls_vertical_slice.py::"
    "test_location_form_matches_gamlss_logno_at_a_parametric_specification": _NO_R,
    "tests/test_two_piece_lss_vertical_slice.py::"
    "test_real_line_family_matches_gamlss_sn2_at_a_parametric_specification": _NO_R,
}


def platform_of(report: Path) -> str:
    """``pytest-Linux-X64-py3.13-2/pytest-results.xml`` -> ``Linux-X64-py3.13``."""
    return report.parent.name.removeprefix("pytest-").rsplit("-", 1)[0]


def cases(report: Path) -> list[ET.Element]:
    """The test cases of one report; none when it is empty or does not parse."""
    try:
        return list(ET.parse(report).getroot().iter("testcase"))
    except ET.ParseError:
        return []


def outcomes(reports: list[list[ET.Element]]) -> tuple[set[str], dict[str, set[str]]]:
    """The tests some report ran, and every skip message of each test a report skipped."""
    ran: set[str] = set()
    skips: dict[str, set[str]] = {}
    for report in reports:
        for case in report:
            classname, name = case.get("classname", ""), case.get("name", "")
            test = node_id(classname, name) or (
                name.replace(".", "/") + ".py" if not classname else f"{classname}::{name}"
            )
            skip = case.find("skipped")
            message = "" if skip is None else skip.get("message", "")
            if skip is None or (
                skip.get("type") == "pytest.xfail" and not message.startswith("[NOTRUN]")
            ):
                ran.add(test)
            else:
                skips.setdefault(test, set()).add(message)
    return ran, skips


def where_it_runs(test: str, messages: set[str]) -> str | None:
    """The workflow that runs a test no report ran, else ``None``."""
    if test.split("[", 1)[0] in RUN_ELSEWHERE:
        return RUN_ELSEWHERE[test.split("[", 1)[0]]
    prefixes = {next((p for p in SKIPPED_FOR if m.startswith(p)), None) for m in messages}
    if len(prefixes) != 1 or None in prefixes:
        return None  # a skip whose message names no workflow
    workflow, where = SKIPPED_FOR[prefixes.pop()]
    return where if test.split("::", 1)[0] in modules_named_by(workflow) else None


@functools.cache
def modules_named_by(workflow: str) -> set[str]:
    """The test modules a workflow file names."""
    return set(re.findall(r"tests/[\w/]+\.py", (_ROOT / workflow).read_text(encoding="utf-8")))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("junit", type=Path, help="the downloaded pytest-* artifacts")
    parser.add_argument(
        "--shards", type=int, required=True, help="the shards that must each report a test"
    )
    args = parser.parse_args(argv)
    reports = {report: cases(report) for report in sorted(args.junit.rglob("pytest-results*.xml"))}
    reported = {report.parent.name for report, found in reports.items() if found}
    if len(reported) != args.shards:
        silent = sorted({report.parent.name for report in reports} - reported)
        print(
            f"{len(reported)} of {args.shards} shards reported a test; "
            f"empty or unreadable: {', '.join(silent) or 'none'}. A shard without results "
            "hides what it skipped: rerun the workflow, or correct --shards to the matrix."
        )
        return 1
    ran, skips = outcomes(list(reports.values()))
    ran_modules = {test.split("::", 1)[0] for test in ran}
    platforms = sorted({platform_of(report) for report in reports})
    never = {
        test: messages
        for test, messages in skips.items()
        # a module skipped at collection ran where any of its tests ran
        if test not in ran and test not in ran_modules
    }
    elsewhere = {test: where_it_runs(test, messages) for test, messages in never.items()}
    unexplained = sorted(test for test, where in elsewhere.items() if where is None)
    print(f"{len(ran | skips.keys())} tests on {len(platforms)} platforms: {', '.join(platforms)}")
    print(f"{len(never)} skipped on every platform; {len(never) - len(unexplained)} run elsewhere:")
    for where in sorted({where for where in elsewhere.values() if where is not None}):
        tests = [test for test, w in elsewhere.items() if w == where]
        print(f"  {len(tests):4d} in {where}")
        if where.startswith("nowhere"):
            print("".join(f"         {test}\n" for test in sorted(tests)), end="")
    if unexplained:
        print(f"{len(unexplained)} run on no platform and no workflow is named for them:")
        for test in unexplained:
            print(f"  {test}: {' | '.join(sorted(never[test]))}")
        print(
            "Force the event a skip waits for, delete a parametrisation that cannot apply, or "
            "name the workflow that runs it in scripts/check_never_run_tests.py."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
