"""Fail when a test was skipped on every CI platform: it passed in no shard and checked nothing.

The compatibility shards upload their junit reports (artifact
``pytest-<os>-<arch>-py<version>-<group>``). This reads them all, groups them by
platform (the artifact name without its group) and fails on any test that every
platform reporting it skipped, unless the script names the workflow that
does run it (``RUN_ELSEWHERE``, or ``SKIPPED_FOR`` by the skip's message), or
says that nothing runs it and why. Those are listed in the summary, so a test
no job runs is counted, never passed silently.

    python scripts/check_never_run_tests.py junit-dir

A strict expected failure (``xfail``) ran and failed as declared; it is not a skip.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from update_test_durations import node_id

_NO_R = "nowhere: needs R with gamlss and jsonlite, which no CI job installs"
# Where the tests a skip message names do run, by the message's prefix.
SKIPPED_FOR = {
    "[dataset] ": "the Real data workflow, which fetches freMTPL2 and fails on this skip",
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


def outcomes(reports: list[Path]) -> dict[str, dict[str, tuple[bool, str]]]:
    """``{test: {platform: (skipped, message)}}``, skipped where every report of it skipped it."""
    seen: dict[str, dict[str, tuple[bool, str]]] = {}
    for report in reports:
        platform = platform_of(report)
        for case in ET.parse(report).getroot().iter("testcase"):
            classname, name = case.get("classname", ""), case.get("name", "")
            test = node_id(classname, name) or (
                name.replace(".", "/") + ".py" if not classname else f"{classname}::{name}"
            )
            skip = case.find("skipped")
            skipped = skip is not None and skip.get("type") != "pytest.xfail"
            message = skip.get("message", "") if skipped else ""
            earlier = seen.setdefault(test, {}).get(platform, (True, message))
            seen[test][platform] = (earlier[0] and skipped, earlier[1] or message)
    return seen


def where_it_runs(test: str, message: str) -> str | None:
    """The workflow that runs a test every platform skipped, else ``None``."""
    by_reason = (where for prefix, where in SKIPPED_FOR.items() if message.startswith(prefix))
    return RUN_ELSEWHERE.get(test.split("[", 1)[0]) or next(by_reason, None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("junit", type=Path, help="the downloaded pytest-* artifacts")
    args = parser.parse_args(argv)
    reports = sorted(args.junit.rglob("pytest-results*.xml"))
    if not reports:
        parser.error(f"no pytest-results*.xml under {args.junit}")
    seen = outcomes(reports)
    platforms = sorted({platform_of(report) for report in reports})
    never = {
        test: next(message for _, message in by_platform.values())
        for test, by_platform in seen.items()
        if all(skipped for skipped, _ in by_platform.values())
    }
    elsewhere = {test: where_it_runs(test, message) for test, message in never.items()}
    unexplained = sorted(test for test, where in elsewhere.items() if where is None)
    print(f"{len(seen)} tests on {len(platforms)} platforms: {', '.join(platforms)}")
    print(f"{len(never)} skipped on every platform; {len(never) - len(unexplained)} run elsewhere:")
    for where in sorted({where for where in elsewhere.values() if where is not None}):
        tests = [test for test, w in elsewhere.items() if w == where]
        print(f"  {len(tests):4d} in {where}")
        if where.startswith("nowhere"):
            print("".join(f"         {test}\n" for test in sorted(tests)), end="")
    if unexplained:
        print(f"{len(unexplained)} run on no platform and no workflow is named for them:")
        for test in unexplained:
            print(f"  {test}: {never[test]}")
        print(
            "Force the event a skip waits for, delete a parametrisation that cannot apply, or "
            "name the workflow that runs it in scripts/check_never_run_tests.py."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
