"""Refresh ``.test_durations`` from the per-test times CI measured.

CI splits the suite into four shards with pytest-split's ``least_duration``
algorithm on ``.test_durations``. The file is only worth anything if its times are
what the runners see, so rebuild it from the ``pytest-results*.xml`` files the
shards upload (artifact ``pytest-<os>-<arch>-py<version>-<group>``, kept 7 days):

    uv run python scripts/update_test_durations.py --run 39563911618 --run 39560218439
    uv run python scripts/update_test_durations.py path/to/junit-dir-or-file ...

``--run`` downloads a CI run's artifacts with ``gh``. The Python 3.12 job of a
push run is built under coverage, which slows the pure-Python tests far more than the
numerical ones, so it is dropped from push runs rather than scaled.

One duration per test: the mean over every runtime and run that executed it. The
shared fixtures that tests build lazily (an ``lru_cache`` reference) are paid by
whichever test reaches them first on a worker, so one test's time is not a property
of the test; the mean keeps its share of that cost in the expected shard total, which
the median discards and which leave-one-run-out replay of CI shows to balance
shards better. A test CI skipped keeps its CI time (about zero), not a local one;
an entry for a test CI never ran is kept as it stands.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COVERAGE_RUNTIME = "-py3.12-"  # the push run measures this one under coverage


def node_id(classname: str, name: str) -> str | None:
    """The pytest node id of a junit test case, found by matching the dotted classname to a file."""
    parts = classname.split(".")
    for i in range(len(parts), 0, -1):
        path = "/".join(parts[:i]) + ".py"
        if (ROOT / path).is_file():
            return "::".join([path, *parts[i:], name])
    return None  # a module skipped at collection has no classname


def download(run: str, repo: str | None, into: Path) -> list[Path]:
    """The junit files of one CI run, minus the coverage-instrumented runtime of a push run."""
    scope = ["--repo", repo] if repo else []
    subprocess.run(
        ["gh", "run", "download", run, *scope, "-p", "pytest-*", "-D", str(into / run)],
        check=True,
        cwd=ROOT,
    )
    event = subprocess.run(
        ["gh", "run", "view", run, *scope, "--json", "event", "--jq", ".event"],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    ).stdout.strip()
    files = sorted((into / run).rglob("pytest-results*.xml"))
    return [f for f in files if event != "push" or COVERAGE_RUNTIME not in str(f)]


def read_times(files: list[Path]) -> tuple[dict[str, list[float]], set[str]]:
    """Every observed time of each node id, and the case names that matched no file."""
    times: dict[str, list[float]] = {}
    unmapped: set[str] = set()
    for file in files:
        seen: dict[str, float] = {}
        for case in ET.parse(file).getroot().iter("testcase"):
            nid = node_id(case.get("classname", ""), case.get("name", ""))
            if nid is None:
                unmapped.add(f"{case.get('classname', '')}::{case.get('name')}")
            else:
                seen[nid] = seen.get(nid, 0.0) + float(case.get("time", 0))
        for nid, seconds in seen.items():
            times.setdefault(nid, []).append(seconds)
    return times, unmapped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="*", type=Path, help="junit files or directories of them")
    parser.add_argument("--run", action="append", default=[], help="CI run id to download")
    parser.add_argument("--repo", help="OWNER/NAME for --run (default: the checkout's remote)")
    parser.add_argument("--output", type=Path, default=ROOT / ".test_durations")
    args = parser.parse_args(argv)

    files = [
        f
        for p in args.paths
        for f in (sorted(p.rglob("pytest-results*.xml")) if p.is_dir() else [p])
    ]
    if any(COVERAGE_RUNTIME in str(f) for f in files):
        print(
            f"warning: {COVERAGE_RUNTIME} files are timed as given; if they come from a push run, "
            "they ran under coverage and should be left out (--run drops them itself)",
            file=sys.stderr,
        )
    with tempfile.TemporaryDirectory() as tmp:
        for run in args.run:
            files += download(run, args.repo, Path(tmp))
        if not files:
            parser.error("no junit files: give paths or --run")
        times, unmapped = read_times(files)

    recorded = json.loads(args.output.read_text(encoding="utf-8")) if args.output.exists() else {}
    refreshed = {nid: round(statistics.fmean(v), 3) for nid, v in times.items()}
    kept = len(recorded.keys() - refreshed.keys())
    merged = {**recorded, **refreshed}
    args.output.write_text(json.dumps(merged, indent=4, sort_keys=True) + "\n", encoding="utf-8")
    elsewhere = sorted(name for name in unmapped if not name.startswith("::"))
    print(
        f"{len(files)} junit files, {len(refreshed)} tests timed, {kept} earlier entries kept, "
        f"{len(unmapped) - len(elsewhere)} module-level skips unmapped, {len(merged)} entries written"
    )
    if elsewhere:  # a file renamed or deleted since the run, or a checkout on another commit
        print(f"{len(elsewhere)} cases name no file in this checkout, e.g. {elsewhere[:3]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
