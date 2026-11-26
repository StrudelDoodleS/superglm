"""Regression checks for merge-gate configuration and shard metadata."""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[1]


def _jobs(workflow: str) -> dict[str, str]:
    """Split a workflow into its top-level job blocks, keyed by job id."""
    body = workflow.split("\njobs:\n", maxsplit=1)[1]
    starts = [(match.start(), match.group(1)) for match in re.finditer(r"(?m)^  ([\w-]+):$", body)]
    return {
        job_id: body[start : starts[index + 1][0] if index + 1 < len(starts) else len(body)]
        for index, (start, job_id) in enumerate(starts)
    }


def _check_run_names(workflow: str) -> dict[str, list[str]]:
    """Map each job id to the check-run name(s) that job publishes."""
    published: dict[str, list[str]] = {}
    for job_id, block in _jobs(workflow).items():
        declared = re.search(r"(?m)^    name: (.+)$", block)
        if declared is None:
            published[job_id] = [job_id]
            continue
        template = declared.group(1).strip().strip("\"'")
        if "${{ matrix.runtime.python-version }}" in template:
            published[job_id] = [
                template.replace("${{ matrix.runtime.python-version }}", version)
                .replace("${{ matrix.label }}", label)
                .replace("${{ matrix.runtime.suffix }}", suffix)
                for version, _os, _group, label, suffix in _compatibility_cases(block)
            ]
            continue
        matrix_key = re.fullmatch(r"\$\{\{ *matrix\.([\w-]+) *\}\}", template)
        if matrix_key is None:
            published[job_id] = [template]
            continue
        published[job_id] = [
            value.strip().strip('"')
            for value in re.findall(
                rf"(?m)^ +-? *{re.escape(matrix_key.group(1))}: (.+)$",
                block,
            )
        ]
    return published


def _compatibility_cases(block: str) -> list[tuple[str, str, int, str, str]]:
    """Expand this workflow's runtime/shard product and shard labels."""
    matrix = yaml.safe_load(block)["test-compatibility"]["strategy"]["matrix"]
    assert set(matrix) == {"runtime", "group", "include"}, (
        "native compatibility must run every shard on every declared runtime"
    )
    labels = matrix["include"]
    assert all(set(item) == {"group", "label"} for item in labels)
    by_group = {item["group"]: item["label"] for item in labels}
    assert len(labels) == len(by_group) == len(matrix["group"])
    assert set(by_group) == set(matrix["group"])
    return [
        (runtime["python-version"], runtime["os"], group, by_group[group], runtime["suffix"])
        for runtime in matrix["runtime"]
        for group in matrix["group"]
    ]


def _mirror_workflows(root: Path, *, dev_ci: str, compatibility: str) -> None:
    """Write a two-workflow tree that the contract tests can be pointed at."""
    workflows = root / ".github" / "workflows"
    workflows.mkdir(parents=True, exist_ok=True)
    (workflows / "dev-ci.yml").write_text(dev_ci, encoding="utf-8")
    (workflows / "ci.yml").write_text(compatibility, encoding="utf-8")


def _collect_non_browser_nodeids() -> set[str]:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "--collect-only",
            "-qq",
            "--no-cov",
            "-m",
            "not browser",
        ],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return {
        line for line in completed.stdout.splitlines() if line.startswith("tests/") and "::" in line
    }


def _duration_shortfall(collected: set[str], recorded: dict[str, float]) -> str | None:
    """What to do when ``.test_durations`` records under 99% of the suite, else None.

    pytest-split gives an unrecorded test the mean recorded time, about 50 ms,
    whatever it really takes. At 95% the 1,132 tests added by #611 and #612 went
    unrecorded, 4-12% of a CI shard's measured time.
    """
    missing = sorted(collected - recorded.keys())
    if len(collected) - len(missing) >= 0.99 * len(collected):
        return None
    return (
        f"{len(missing)} of {len(collected)} collected tests have no entry in .test_durations "
        f"(at most 1% may), for example {', '.join(missing[:3])}. Refresh it from this pull "
        "request's own CI run, whose shards upload their per-test times even when they fail: "
        "`uv run python scripts/update_test_durations.py --run <run id>`, then commit "
        ".test_durations. The run id is the number after /actions/runs/ in the link of any of "
        "the pull request's CI checks, or what `gh run list --workflow ci.yml --event "
        "pull_request --branch <your branch> --limit 1 --json databaseId -q '.[0].databaseId'` "
        "prints. Its artifacts last 7 days."
    )


def test_duration_manifest_covers_the_non_browser_suite() -> None:
    recorded = json.loads((_ROOT / ".test_durations").read_text(encoding="utf-8"))
    collected = _collect_non_browser_nodeids()

    assert collected
    shortfall = _duration_shortfall(collected, recorded)
    assert shortfall is None, shortfall


def test_a_stale_duration_manifest_says_how_to_refresh_it() -> None:
    """The failure names the unrecorded tests and the refresh command a pull request
    can run from its own CI run, rather than a bare ratio."""
    collected = {f"tests/test_x.py::test_{i}" for i in range(1000)}
    recorded = {node: 0.05 for node in sorted(collected)[:989]}

    assert _duration_shortfall(collected, {**recorded, "tests/test_x.py::test_999": 0.05}) is None
    message = _duration_shortfall(collected, recorded)
    assert message is not None
    assert message.startswith("11 of 1000 collected tests have no entry in .test_durations")
    assert sorted(collected)[989] in message
    assert "scripts/update_test_durations.py --run <run id>" in message
    assert "/actions/runs/" in message
    # gh run list's first column is the run's status; --json prints the id alone.
    assert (
        "`gh run list --workflow ci.yml --event pull_request --branch <your branch> --limit 1 "
        "--json databaseId -q '.[0].databaseId'`" in message
    )


_PYTHON_FLOOR_CHECK = "Python 3.12 · non-browser suite · version floor"


def test_required_workflow_runs_for_pull_requests_and_python_floor() -> None:
    workflow = (_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    names = _check_run_names(workflow)
    floor = [
        block for job_id, block in _jobs(workflow).items() if _PYTHON_FLOOR_CHECK in names[job_id]
    ]

    assert "pull_request:" in workflow
    pull_request = workflow.split("  pull_request:\n", maxsplit=1)[1].split("\n\n", maxsplit=1)[0]
    assert "paths:" not in pull_request and "paths-ignore:" not in pull_request
    push = workflow.split("  push:\n", maxsplit=1)[1].split("  pull_request:\n", maxsplit=1)[0]
    assert '"scripts/**"' in push, "a change to the suite runner must run master CI"
    assert "cancel-in-progress: ${{ github.event_name == 'pull_request' }}" in workflow
    assert len(floor) == 1, f"exactly one job must publish {_PYTHON_FLOOR_CHECK!r}"
    assert "if: ${{ always() }}" in floor[0]
    assert "needs: [test-compatibility, polars-floor, never-run]" in floor[0]
    matrix = _jobs(workflow)["test-compatibility"]
    assert {case for case in _compatibility_cases(matrix) if case[0] == "3.12"} == {
        ("3.12", "ubuntu-latest", group, label, "") for group, label in enumerate("ABCD", start=1)
    }
    assert "uv run --with mpmath python scripts/run_test_suite.py" in matrix
    assert "--extra dev --extra bench --extra plotting" in matrix, (
        "the compatibility test matrix must install the bench and plotting extras"
    )
    assert "continue-on-error: true" not in workflow


def test_polars_floor_job_runs_the_declared_floor_with_the_matrix_extras() -> None:
    """The floor job tests the Polars release pyproject.toml declares as the
    floor, in the environment the matrix installs, and lists its skips."""
    job = yaml.safe_load((_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))["jobs"][
        "polars-floor"
    ]
    project = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    requirements = [*project["dependencies"], *sum(project["optional-dependencies"].values(), [])]
    floors = [
        requirement.removeprefix("polars>=")
        for requirement in requirements
        if requirement.startswith("polars>=")
    ]
    assert floors == [job["env"]["POLARS_FLOOR"]], (
        "the floor job must run the declared Polars floor"
    )
    assert f"Polars {floors[0]} floor" in job["name"]
    runs = [step["run"] for step in job["steps"] if "run" in step]
    assert any(
        run.startswith("uv sync --locked ") and "--extra dev --extra bench --extra plotting" in run
        for run in runs
    ), "the Polars floor job must install the matrix's bench and plotting extras"
    uv_runs = [line for run in runs for line in run.splitlines() if line.startswith("uv run ")]
    assert len(uv_runs) == 2
    assert all('--with "polars==${POLARS_FLOOR}"' in line for line in uv_runs)
    assert "polars.__version__ == os.environ['POLARS_FLOOR']" in uv_runs[0]
    assert " pytest " in uv_runs[1] and " -rs " in uv_runs[1]


def test_dependency_floor_job_runs_the_quick_suite_at_lowest_direct_on_the_oldest_python() -> None:
    """One job installs every direct dependency at its declared floor, with the
    matrix's extras, on the oldest Python pyproject.toml admits, and runs the
    quick suite in that environment without letting uv re-sync it."""
    job = yaml.safe_load((_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))["jobs"][
        "dependency-floors"
    ]
    project = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    oldest = project["requires-python"].removeprefix(">=")
    runs = [" ".join(step["run"].split()) for step in job["steps"] if "run" in step]
    assert job["runs-on"] == "ubuntu-latest" and "strategy" not in job
    assert f"Python {oldest} " in job["name"]
    (sync,) = [run for run in runs if run.startswith("uv sync ")]
    assert sync == (
        f"uv sync --resolution lowest-direct --python {oldest} "
        "--extra dev --extra bench --extra plotting"
    )
    (suite,) = [run for run in runs if "run_test_suite.py" in run]
    assert suite.startswith("uv run --no-sync python scripts/run_test_suite.py")
    assert '-m "not slow and not browser and not docs"' in suite
    assert all(run.startswith("uv run --no-sync ") for run in runs if run.startswith("uv run"))


def test_compatibility_shards_do_not_queue_platforms_behind_each_other() -> None:
    workflow = (_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    block = _jobs(workflow)["test-compatibility"]
    strategy = yaml.safe_load(block)["test-compatibility"]["strategy"]
    cases = _compatibility_cases(block)
    assert strategy.get("max-parallel", len(cases)) >= len(cases)


def test_compatibility_shards_collect_all_failures_with_a_hang_limit() -> None:
    workflow = (_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    job = yaml.safe_load(workflow)["jobs"]["test-compatibility"]
    for runtime in job["strategy"]["matrix"]["runtime"]:
        assert 0 < _shard_hang_limit(job, runtime, push=False) <= 20
    assert job["strategy"].get("fail-fast", True) is False
    suite_commands = [
        shlex.split(step["run"])
        for step in job["steps"]
        if "run_test_suite.py" in step.get("run", "")
    ]
    assert suite_commands
    assert all("--junitxml=pytest-results.xml" in command for command in suite_commands)
    for command, _ in _suite_runner().stage_commands("not browser and not docs", []):
        assert "-x" not in command and "--exitfirst" not in command
        assert "--maxfail=0" in command
    report_step = next(step for step in job["steps"] if step.get("name") == "Upload shard results")
    assert report_step["if"] == "${{ always() }}"
    # The threads stage writes pytest-results-threads.xml beside the parallel report.
    assert report_step["with"]["path"] == "pytest-results*.xml"


def _coverage_condition(job: dict) -> tuple[str, str]:
    """The coverage step's ``if`` and the Python version whose push run it covers."""
    (coverage_if,) = {step["if"] for step in job["steps"] if "--cov=" in step.get("run", "")}
    covered_version = re.fullmatch(
        r"github\.event_name == 'push' && matrix\.runtime\.python-version == '([\d.]+)'",
        coverage_if,
    ).group(1)
    return coverage_if, covered_version


def _shard_hang_limit(job: dict, runtime: dict, *, push: bool) -> int:
    """The minutes ``ci.yml`` gives one runtime's compatibility shard on a push or a
    pull-request run: its ``timeout-minutes`` expression evaluated for that runtime."""
    coverage_if, covered_version = _coverage_condition(job)
    match = re.fullmatch(
        rf"\$\{{\{{ {re.escape(coverage_if)} && (\d+) \|\| (\d+|matrix\.runtime\.hang-limit) \}}\}}",
        str(job["timeout-minutes"]),
    )
    assert match, "the coverage shards' limit must key on the coverage step's own condition"
    if push and runtime["python-version"] == covered_version:
        return int(match.group(1))
    return runtime["hang-limit"] if match.group(2).startswith("matrix.") else int(match.group(2))


def _compatibility_job() -> dict:
    workflow = (_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    return yaml.safe_load(workflow)["jobs"]["test-compatibility"]


def test_coverage_shards_get_a_longer_limit_than_the_uninstrumented_ones() -> None:
    """Branch coverage on Python 3.12 makes a shard 1.5-2x slower, so its limit
    is longer, and only the shards that run under coverage get it."""
    job = _compatibility_job()
    runtimes = job["strategy"]["matrix"]["runtime"]
    (linux_312,) = [r for r in runtimes if r["python-version"] == "3.12"]
    plain = _shard_hang_limit(job, linux_312, push=False)
    covered = _shard_hang_limit(job, linux_312, push=True)
    # A coverage shard measured 869 s in pytest alone and was cancelled at 15 minutes.
    assert plain <= 15 < covered <= 2 * plain
    for runtime in runtimes:
        if runtime is not linux_312:
            on_push = _shard_hang_limit(job, runtime, push=True)
            assert on_push == _shard_hang_limit(job, runtime, push=False)


# The slowest shard of each runtime, in job seconds, over the 15 CI runs from
# 42640955169 to 42937426038 (2026-11-09 to 11, four of them master pushes),
# keyed by the job name's suffix and whether it ran under coverage. These are
# measured snapshots: the headroom test fails when a limit is lowered below a
# runtime's measured worst case, or when a runtime has no figure, but it cannot
# notice a shard growing slower. The coverage figure is the slowest completed
# shard (run 42690868368).
_SLOWEST_SHARD_SECONDS = {
    ("3.12", "", False): 671,
    ("3.12", "", True): 897,
    ("3.14", "", False): 634,
    ("3.13", " · Windows", False): 850,
    ("3.13", " · macOS ARM64", False): 670,
    ("3.13", " · Linux ARM64", False): 452,
}
# Lower bounds from shards cancelled at their limit, whose real time is unknown:
# run 42839237921's coverage shard B was cancelled at the old 15 minutes.
_CANCELLED_SHARD_SECONDS = {("3.12", "", True): 916}

_MEASURE_A_SHARD = (
    "Measure one from a recent CI run, `gh run view <run id> --json jobs -q '.jobs[] | "
    '"\\(.name)\\t\\((.completedAt|fromdate) - (.startedAt|fromdate))"\'`, which prints each '
    "job's seconds (`gh run list --workflow ci.yml --branch master --limit 1 --json databaseId "
    "-q '.[0].databaseId'` prints the latest master run's id), and record the "
    "runtime's slowest shard in _SLOWEST_SHARD_SECONDS in tests/test_ci_contracts.py before "
    "setting its hang-limit."
)


def _headroom_shortfalls(job: dict, slowest: dict, cancelled: dict) -> list[str]:
    """Every compatibility shard whose hang limit lacks a measured figure, or leaves
    under 20% over it, plus table entries that match no shard of the matrix."""
    _, covered_version = _coverage_condition(job)
    shards = [
        ((runtime["python-version"], runtime["suffix"], push), runtime)
        for runtime in job["strategy"]["matrix"]["runtime"]
        for push in (False, True)
        if not push or runtime["python-version"] == covered_version
    ]
    shortfalls = []
    for key, runtime in shards:
        limit = _shard_hang_limit(job, runtime, push=key[2])
        shard = f"Python {key[0]}{key[1]}{' under coverage (push)' if key[2] else ''}"
        seconds = slowest.get(key)
        if seconds is None:
            shortfalls.append(f"{shard} has no slowest-shard figure. {_MEASURE_A_SHARD}")
        elif seconds > 0.8 * 60 * limit:
            shortfalls.append(f"{shard}: slowest shard {seconds} s is over 80% of {limit} minutes")
        if cancelled.get(key, 0) >= 60 * limit:
            shortfalls.append(f"{shard}: a shard was cancelled after {cancelled[key]} s")
        if limit > 20:
            shortfalls.append(f"{shard}: {limit} minutes is a budget, not a hang limit")
    unmatched = sorted((slowest.keys() | cancelled.keys()) - {key for key, _ in shards})
    shortfalls += [f"{key} matches no shard of the matrix" for key in unmatched]
    return shortfalls


def test_every_shard_limit_leaves_headroom_over_its_slowest_measured_shard() -> None:
    """A hang limit within 20% of a runtime's slowest measured shard cancels a
    healthy run whenever the runner is slow; at 15 minutes Windows ran at 94%."""
    job = _compatibility_job()
    shortfalls = _headroom_shortfalls(job, _SLOWEST_SHARD_SECONDS, _CANCELLED_SHARD_SECONDS)
    assert not shortfalls, shortfalls


def test_a_runtime_without_a_measured_shard_says_how_to_measure_one() -> None:
    """A runtime added to the matrix, or one whose suffix changes, has no figure:
    the headroom check names it and says how to measure one, rather than passing
    it unchecked or failing on an unpack."""
    job = _compatibility_job()
    runtimes = job["strategy"]["matrix"]["runtime"]
    (windows,) = [r for r in runtimes if r["suffix"] == " · Windows"]
    renamed = {**windows, "suffix": " · Windows x64"}
    arm = {**windows, "os": "windows-11-arm", "suffix": " · Windows ARM64", "hang-limit": 10}
    others = [r for r in runtimes if r is not windows]
    job["strategy"]["matrix"]["runtime"] = [*others, renamed, arm]

    shortfalls = _headroom_shortfalls(job, _SLOWEST_SHARD_SECONDS, _CANCELLED_SHARD_SECONDS)
    assert shortfalls == [
        f"Python 3.13 · Windows x64 has no slowest-shard figure. {_MEASURE_A_SHARD}",
        f"Python 3.13 · Windows ARM64 has no slowest-shard figure. {_MEASURE_A_SHARD}",
        "('3.13', ' · Windows', False) matches no shard of the matrix",
    ]
    assert "gh run view <run id> --json jobs" in _MEASURE_A_SHARD

    lowered = {**_CANCELLED_SHARD_SECONDS, ("3.12", "", True): 1200}
    assert _headroom_shortfalls(_compatibility_job(), _SLOWEST_SHARD_SECONDS, lowered) == [
        "Python 3.12 under coverage (push): a shard was cancelled after 1200 s"
    ]


@pytest.mark.parametrize("stage", [0, 1], ids=["parallel", "threads"])
def test_suite_runner_stages_report_both_failures(tmp_path: Path, stage: int) -> None:
    """Changing either stage of the suite runner back to exit-first loses the second failure."""
    command, _ = _suite_runner().stage_commands("not browser and not docs", [])[stage]
    stop_options = [
        arg for arg in command if arg in ("-x", "--exitfirst") or arg.startswith("--maxfail=")
    ]
    fixture = tmp_path / "test_failures.py"
    fixture.write_text("def test_first(): assert False\ndef test_second(): assert False\n")
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-o", "addopts=", *stop_options, str(fixture)],
        cwd=tmp_path,
        env=os.environ | {"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert completed.returncode == 1, completed.stdout + completed.stderr
    assert "2 failed" in completed.stdout, completed.stdout + completed.stderr


@pytest.mark.parametrize("never_run_result", ["success", "failure"])
@pytest.mark.parametrize("floor_result", ["success", "failure", "cancelled", "skipped"])
@pytest.mark.parametrize("result", ["success", "failure", "cancelled", "skipped"])
def test_python_floor_aggregate_executes_the_matrix_verdict(
    result: str, floor_result: str, never_run_result: str
) -> None:
    """The required context passes only when the matrix, the Polars floor and the run check passed."""
    workflow = (_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    aggregate = _jobs(workflow)["python-floor"]
    assert "TEST_RESULT: ${{ needs.test-compatibility.result }}" in aggregate
    assert "POLARS_FLOOR_RESULT: ${{ needs.polars-floor.result }}" in aggregate
    assert "NEVER_RUN_RESULT: ${{ needs.never-run.result }}" in aggregate
    commands = re.findall(r"(?m)^        run: (.+)$", aggregate)
    assert len(commands) == 1
    command = shlex.split(commands[0])
    assert command[:2] == ["python", "-c"]
    completed = subprocess.run(
        [sys.executable, *command[1:]],
        env=os.environ
        | {
            "TEST_RESULT": result,
            "POLARS_FLOOR_RESULT": floor_result,
            "NEVER_RUN_RESULT": never_run_result,
        },
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    passed = result == floor_result == never_run_result == "success"
    assert (completed.returncode == 0) == passed, completed.stderr


def test_dev_ci_job_names_do_not_claim_merge_gate_membership() -> None:
    """No check-run name may assert that the ruleset requires it.

    The `Protect master` ruleset lives outside the repository, so nothing in the
    tree can verify which contexts it lists.  A job whose published name says
    "required" therefore tells a reviewer the opposite of what the gate does
    whenever that context has not been added to the ruleset.
    """
    workflow = (_ROOT / ".github/workflows/dev-ci.yml").read_text(encoding="utf-8")
    claiming = sorted(
        name
        for names in _check_run_names(workflow).values()
        for name in names
        if "required" in name.casefold()
    )

    assert claiming == [], (
        "dev-ci.yml check-run names must not claim merge-gate membership that this "
        f"repository cannot verify: {claiming}"
    )


def test_python_floor_contract_rejects_extras_installed_only_where_no_tests_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The extras contract must be scoped to the job that actually runs pytest.

    `--extra bench --extra plotting` also appears in `type-check`, which runs no
    tests. Dropping the extras from the compatibility matrix must fail even
    though the type-check job still installs them.
    """
    dev_ci = (_ROOT / ".github/workflows/dev-ci.yml").read_text(encoding="utf-8")
    compatibility = (_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    matrix = _jobs(compatibility)["test-compatibility"]
    mutant = compatibility.replace(matrix, matrix.replace(" --extra bench --extra plotting", ""))

    assert mutant != compatibility
    assert "--extra bench --extra plotting" not in _jobs(mutant)["test-compatibility"]
    assert "--extra bench --extra plotting" in _jobs(dev_ci)["type-check"]

    _mirror_workflows(tmp_path, dev_ci=dev_ci, compatibility=mutant)
    monkeypatch.setattr(sys.modules[__name__], "_ROOT", tmp_path)

    with pytest.raises(AssertionError, match="bench and plotting extras"):
        test_required_workflow_runs_for_pull_requests_and_python_floor()


@pytest.mark.pins_behaviour
def test_type_check_enforces_and_validates_the_accepted_backlog() -> None:
    workflow = (_ROOT / ".github/workflows/dev-ci.yml").read_text(encoding="utf-8")
    type_job = workflow.split("  type-check:", maxsplit=1)[1].split("  quality:", maxsplit=1)[0]

    assert "uv sync --locked --python 3.13" in type_job
    assert "--extra dev --extra bench --extra plotting" in type_job
    assert 'pipeline_status=("${PIPESTATUS[@]}")' in type_job
    assert "diagnostics > 903" in type_job
    assert 'grep -qx "All checks passed!"' in type_job
    assert "diagnostics?" in type_job
    assert "ty_status > 1" in type_job
    assert "ty did not emit a diagnostic count" in type_job
    assert "ty reported diagnostics with a successful exit status" in type_job
    assert "ty failed despite reporting zero diagnostics" in type_job
    assert "fix-tests:" not in type_job  # the slice is the type-check job and nothing else


def test_fix_tests_job_fails_on_a_flag_and_writes_its_summary_inside_its_limit() -> None:
    """A test a pull request adds that passes on its base fails the job, after the summary."""
    workflow = (_ROOT / ".github/workflows/dev-ci.yml").read_text(encoding="utf-8")
    job = yaml.safe_load(workflow)["jobs"]["fix-tests"]
    run = next(step["run"] for step in job["steps"] if "check_fix_tests" in step.get("run", ""))
    limit = int(re.search(r"timeout --kill-after=\d+s (\d+)m", run).group(1))

    assert "continue-on-error" not in job and "--report-only" not in run
    assert "set -o pipefail" in run and "| tee fix-tests.txt" in run
    assert run.rstrip().endswith('exit "$status"')  # the check's own status, flags included
    assert limit < job["timeout-minutes"]  # the summary is written before the job's own limit
    assert run.index("timeout") < run.index("GITHUB_STEP_SUMMARY")


def test_frontend_check_names_are_unambiguous() -> None:
    """Regression guard for a fix that shipped before this branch.

    The duplicate-`frontend` defect (issue #227 item 3) was already fixed in
    commit 67b90f8, which renamed ci.yml's `frontend` job to `frontend-browser`
    and gave it an explicit `name:`.  That commit is an ancestor of this
    branch's parent 37a1c18, so no form of this test can fail at 37a1c18; it is
    kept to stop the duplicate name coming back, not to prove a fix made here.

    The mutants it is verified against are built by
    `test_frontend_contract_rejects_a_second_job_publishing_the_frontend_check`:
    the exact inverse of 67b90f8, a second `frontend` job added beside
    `frontend-browser`, and a matrix job whose quoted `name:` resolves to
    `frontend`.
    """
    required = (_ROOT / ".github/workflows/dev-ci.yml").read_text(encoding="utf-8")
    compatibility = (_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "\n  frontend:\n" in required, (
        "dev-ci.yml must define the job that owns the required check run"
    )
    assert "\n  frontend-browser:\n" in compatibility, (
        "ci.yml's browser job must keep the disambiguated frontend-browser id"
    )
    assert "if: github.event_name == 'push'" in _jobs(compatibility)["frontend-browser"]
    assert "frontend" not in {
        name for names in _check_run_names(compatibility).values() for name in names
    }, "ci.yml must not define a job that publishes the required 'frontend' check run"


_RENAME_67B90F8 = "  frontend-browser:\n    name: frontend-browser\n"

_MATRIX_FRONTEND_JOB = """  frontend-checks:
    name: "${{ matrix.check }}"
    runs-on: ubuntu-latest
    strategy:
      matrix:
        include:
          - check: frontend

    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1

      - name: Check frontend modules
        run: npm run check:frontend

"""


def _reverse_the_disambiguating_rename(compatibility: str) -> str:
    """Undo commit 67b90f8: hand ci.yml's browser job its `frontend` id back."""
    mutant = compatibility.replace(_RENAME_67B90F8, "  frontend:\n", 1)
    assert mutant != compatibility
    return mutant


def _duplicate_the_frontend_job(compatibility: str) -> str:
    """Add a second `frontend` job while keeping `frontend-browser` intact."""
    browser = _jobs(compatibility)["frontend-browser"]
    mutant = compatibility.replace(
        browser, _reverse_the_disambiguating_rename(browser) + browser, 1
    )
    assert mutant != compatibility
    return mutant


def _publish_frontend_from_a_matrix(compatibility: str) -> str:
    """Add a job whose quoted matrix `name:` resolves to the `frontend` context."""
    browser = _jobs(compatibility)["frontend-browser"]
    mutant = compatibility.replace(browser, _MATRIX_FRONTEND_JOB + browser, 1)
    assert mutant != compatibility
    return mutant


def test_frontend_contract_rejects_a_second_job_publishing_the_frontend_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each way of restoring the duplicate `frontend` context must fail the guard.

    Both workflows fire on `pull_request`, so a job publishing `frontend` in
    either one attaches a second check run to the required `frontend` context on
    the same head SHA, and the weaker of the two is enough to satisfy the gate.
    This proves the guard above is not inert by running it against three faithful
    mutants of the tree it protects.
    """
    dev_ci = (_ROOT / ".github/workflows/dev-ci.yml").read_text(encoding="utf-8")
    compatibility = (_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "_ROOT", tmp_path)

    # Mutant 1 - the exact inverse of commit 67b90f8, the state ci.yml was in
    # when two workflows really did publish `frontend`.
    reverted = _reverse_the_disambiguating_rename(compatibility)
    assert "\n  frontend:\n" in reverted
    assert "\n  frontend-browser:\n" not in reverted
    _mirror_workflows(tmp_path, dev_ci=dev_ci, compatibility=reverted)
    with pytest.raises(AssertionError, match="disambiguated frontend-browser id"):
        test_frontend_check_names_are_unambiguous()

    # Mutant 2 - the forward-looking shape: a `frontend` job re-added beside the
    # renamed one, so only the negative assertion can catch it.
    duplicated = _duplicate_the_frontend_job(compatibility)
    assert "\n  frontend:\n" in duplicated
    assert "\n  frontend-browser:\n" in duplicated
    _mirror_workflows(tmp_path, dev_ci=dev_ci, compatibility=duplicated)
    with pytest.raises(AssertionError, match="must not define a job that publishes"):
        test_frontend_check_names_are_unambiguous()

    # Mutant 3 - the same duplicate smuggled in behind a quoted matrix
    # reference, which no job-id substring check can see.
    from_matrix = _publish_frontend_from_a_matrix(compatibility)
    assert "\n  frontend:\n" not in from_matrix
    assert "\n  frontend-browser:\n" in from_matrix
    _mirror_workflows(tmp_path, dev_ci=dev_ci, compatibility=from_matrix)
    with pytest.raises(AssertionError, match="must not define a job that publishes"):
        test_frontend_check_names_are_unambiguous()


def test_coverage_omit_targets_the_plotting_package() -> None:
    config = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    omit = config["tool"]["coverage"]["run"]["omit"]

    assert omit == ["src/superglm/plotting/*"]
    assert sorted((_ROOT / "src/superglm/plotting").glob("*.py"))


def _suite_runner():
    """scripts/run_test_suite.py, the one runner CI and local runs share."""
    path = _ROOT / "scripts" / "run_test_suite.py"
    spec = importlib.util.spec_from_file_location("run_test_suite", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "arguments",
    [
        ["tests/test_reml.py"],
        ["tests/test_reml.py::TestExactPresolve"],
        ["-k", "reml", "tests/"],
        ["test_reml.py"],
    ],
)
def test_the_suite_runner_refuses_a_test_file_it_would_run_beside_the_suite(
    arguments, capsys, monkeypatch
) -> None:
    runner, started = _suite_runner(), []
    monkeypatch.setattr(
        runner.subprocess, "call", lambda command, **_: started.append(command) or 0
    )
    with pytest.raises(SystemExit) as stopped:
        runner.main(arguments)
    assert started == []
    assert stopped.value.code == 2
    assert "always runs all of tests/" in capsys.readouterr().err


def test_the_suite_runner_passes_option_values_through() -> None:
    passthrough = [
        *["--splits", "4", "--group", "1", "-k", "reml and not slow", "--junitxml=r.xml"],
        *["--ignore", "tests/test_reml.py", "--deselect", "tests/test_reml.py::TestExactPresolve"],
        *["--junitxml", "tests/out.xml", "--ignore=tests/docs"],
    ]
    assert _suite_runner().named_tests(passthrough) == []


def test_the_suite_runner_pins_pools_in_parallel_and_frees_them_for_threads() -> None:
    runner = _suite_runner()
    passthrough = [
        "--splits",
        "4",
        "--group",
        "1",
        "--junitxml=pytest-results.xml",
        "--cov=superglm",
    ]
    (parallel, pinned), (threaded, unpinned) = runner.stage_commands(
        "not browser and not docs", passthrough
    )
    assert (pinned, unpinned) == (True, False)
    # Markers follow tests/; the first -m in each command is `python -m pytest`.
    marks = parallel.index("-m", parallel.index("tests/"))
    assert parallel[marks + 1] == "(not browser and not docs) and not threads"
    start = parallel.index("-n")
    assert parallel[start : start + 4] == ["-n", "logical", "--dist", "worksteal"]
    marks = threaded.index("-m", threaded.index("tests/"))
    assert threaded[marks + 1] == "(not browser and not docs) and threads"
    assert threaded[-2:] == ["-n", "0"], "stage 2 stays serial after any forwarded -n"
    assert "--junitxml=pytest-results.xml" in parallel and "--cov-append" not in parallel
    assert "--junitxml=pytest-results-threads.xml" in threaded and "--cov-append" in threaded
    # Coverage enabled by PYTEST_ADDOPTS or the config must not erase stage 1's data.
    assert "--cov-append" in runner.stage_commands("not docs", [])[1][0]
    # VECLIB is the only pin that reaches Accelerate BLAS on the macOS runners;
    # BLIS is the one a BLIS-backed NumPy reads.
    assert {
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    } <= set(runner.PINNED_POOLS)
    caller = {
        "PATH": "/bin",
        "OMP_NUM_THREADS": "3",
        "NUMBA_NUM_THREADS": "2",
        "SUPERGLM_BLAS_THREADS": "8",
    }
    assert runner.stage_environment(True, caller) == {
        "PATH": "/bin",
        **dict.fromkeys(runner.PINNED_POOLS, "1"),
    }
    # Stage 2 drops pins the calling shell exported, or its threads tests run pinned.
    assert runner.stage_environment(False, caller) == {"PATH": "/bin"}
    assert '"threads:' in (_ROOT / "pyproject.toml").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "junit",
    [
        ["--junitxml=r.xml"],
        ["--junit-xml=r.xml"],
        ["--junitxml", "r.xml"],
        ["--junit-xml", "r.xml"],
    ],
)
def test_the_suite_runner_writes_stage_two_junit_beside_stage_one(junit: list[str]) -> None:
    runner = _suite_runner()
    (parallel, _), (threaded, _) = runner.stage_commands("not docs", junit)
    assert "r.xml" in " ".join(parallel) and "r-threads.xml" not in " ".join(parallel)
    assert "r-threads.xml" in " ".join(threaded) and " r.xml" not in " " + " ".join(threaded)
    # pytest reads PYTEST_ADDOPTS as arguments, so a report named there moves too.
    caller = {"PYTEST_ADDOPTS": shlex.join([*junit, "-k", "not slow"])}
    assert runner.stage_environment(True, caller)["PYTEST_ADDOPTS"] == caller["PYTEST_ADDOPTS"]
    moved = shlex.split(runner.stage_environment(False, caller)["PYTEST_ADDOPTS"])
    assert moved == [
        arg.replace("r.xml", "r-threads.xml") for arg in shlex.split(caller["PYTEST_ADDOPTS"])
    ]


def test_the_pytest_config_names_no_junit_report() -> None:
    """The runner renames reports named on its command line or in PYTEST_ADDOPTS only;
    one named in the config would reach both stages, and stage 2's would replace stage 1's."""
    config = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    addopts = shlex.split(config["tool"]["pytest"]["ini_options"]["addopts"])
    assert not any(arg.startswith(_suite_runner().JUNIT_OPTIONS) for arg in addopts)


@pytest.mark.parametrize(
    ("codes", "expected"),
    [
        ((0, 0), 0),
        ((0, 5), 0),  # a shard with no threads test
        ((5, 0), 0),  # a selection of threads tests only
        ((5, 5), 5),  # the selection matched nothing: a broken -m or -k, not a pass
        ((1, 0), 1),
        ((0, 1), 1),
        ((0, -11), 1),  # a segfault in the threads stage
        ((-9, 0), 1),  # an OOM kill in the parallel stage
        ((2, 1), 2),  # the first failure's code is kept
    ],
)
def test_the_suite_runner_fails_on_any_failing_stage(monkeypatch, codes, expected) -> None:
    runner = _suite_runner()
    calls = []

    def fake_call(command, cwd, env):
        calls.append((command, cwd, env))
        return codes[len(calls) - 1]

    monkeypatch.setattr(runner.subprocess, "call", fake_call)
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    assert runner.main(["-m", "not docs", "-k", "x"]) == expected
    (first, cwd, first_env), (second, _, second_env) = calls
    assert cwd == _ROOT
    assert all(first_env[pool] == "1" for pool in runner.PINNED_POOLS)
    assert not set(runner.PINNED_POOLS) & set(second_env)
    assert first[-2:] == ["-k", "x"] and second[-4:] == ["-k", "x", "-n", "0"]


@pytest.mark.parametrize("argv", [["--", "-m", "slow"], ["-k", "x", "--", "-mslow"]])
def test_a_marker_after_the_separator_still_selects_per_stage(monkeypatch, argv) -> None:
    # pytest keeps the last -m, so one reaching it would give both stages the same tests.
    runner = _suite_runner()
    commands = []

    def fake_call(command, cwd, env):
        commands.append(command[command.index("tests/") :])
        return 0

    monkeypatch.setattr(runner.subprocess, "call", fake_call)
    assert runner.main(argv) == 0
    stages = ["(slow) and not threads", "(slow) and threads"]
    for pytest_args, expression in zip(commands, stages, strict=True):
        assert [arg for arg in pytest_args if arg.startswith("-m")] == ["-m"]
        assert pytest_args[pytest_args.index("-m") + 1] == expression


# Tests that touch thread-pool APIs but do not need the default pools.
_UNPINNED_THREAD_API_ALLOWLIST = {
    "tests/test_c3_stress_stationarity.py::test_original_stress_fixture_has_fresh_strict_newton_authority": (
        "caps its own pools"
    ),
    "tests/test_rank_deficient_complete_fit.py::test_dispatch_comes_from_a_live_process_not_build_metadata": (
        "checks a BLAS pool exists"
    ),
}
_THREAD_POOL_API = {
    "set_num_threads",
    "get_num_threads",
    "threadpool_limits",
    "threadpool_info",
    "ThreadpoolController",
}


def _names(node: ast.AST) -> set[str]:
    """The identifiers a node uses; a name in a string, such as a patch target, is not one."""
    return {
        getattr(child, "id", None) or getattr(child, "attr", None) or child.arg
        for child in ast.walk(node)
        if isinstance(child, ast.Name | ast.Attribute | ast.arg)
    }


def _marks_threads(text: str, nodes: list[ast.AST]) -> bool:
    return any("mark.threads" in ast.get_source_segment(text, node) for node in nodes)


def _pytestmark(body: list[ast.stmt]) -> list[ast.stmt]:
    return [
        node
        for node in body
        if isinstance(node, ast.Assign)
        and any(getattr(target, "id", "") == "pytestmark" for target in node.targets)
    ]


def _thread_api_tests(path: Path) -> dict[str, bool]:
    """The tests in ``path`` that touch a pool API, each mapped to whether it is marked.

    A test touches the API when its source names it, or names a module-level
    helper that does, directly or through other helpers.
    """
    text = path.read_text(encoding="utf-8")
    if not _THREAD_POOL_API & set(re.findall(r"\w+", text)):
        return {}
    tree = ast.parse(text)
    helpers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.ClassDef)
        and not node.name.startswith(("test_", "Test"))
    ]
    api = set(_THREAD_POOL_API)
    while grown := {node.name for node in helpers if node.name not in api and api & _names(node)}:
        api |= grown
    module = path.relative_to(_ROOT).as_posix()
    module_marked = _marks_threads(text, _pytestmark(tree.body))
    scopes = [(module, tree.body, module_marked)] + [
        (
            f"{module}::{node.name}",
            node.body,
            module_marked or _marks_threads(text, node.decorator_list + _pytestmark(node.body)),
        )
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name.startswith("Test")
    ]
    return {
        f"{scope}::{node.name}": marked or _marks_threads(text, node.decorator_list)
        for scope, body, marked in scopes
        for node in body
        if isinstance(node, ast.FunctionDef)
        and node.name.startswith("test_")
        and api & _names(node)
    }


def test_tests_that_read_or_set_thread_pools_run_unpinned() -> None:
    """A test that reads or sets pool sizes silently loses its meaning when pinned.

    scripts/run_test_suite.py pins every stage-1 worker to one thread, so such a
    test must carry the ``threads`` marker (stage 2, default pools) unless it
    is listed above with the reason it does not need them. The scan follows
    helpers and fixtures defined in the test's own module and walks test
    classes; it cannot see helpers imported from elsewhere, helpers or fixtures
    defined on a test class, conftest fixtures, import aliases, or tests that
    read the pinned environment variables.
    """
    touched = {}
    for path in sorted((_ROOT / "tests").rglob("test_*.py")):
        touched |= _thread_api_tests(path)
    allowed = _UNPINNED_THREAD_API_ALLOWLIST
    unmarked = [test for test, marked in touched.items() if not marked and test not in allowed]
    stale = sorted(set(allowed) - touched.keys())
    assert not unmarked, f"mark these @pytest.mark.threads or allowlist them: {unmarked}"
    assert not stale, f"these allowlist entries match no test that touches a pool API: {stale}"


def test_the_durations_refresh_drops_the_runtime_ci_measures_under_coverage() -> None:
    """scripts/update_test_durations.py leaves out the coverage shards by artifact name."""
    from scripts.update_test_durations import COVERAGE_RUNTIME

    job = yaml.safe_load((_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))["jobs"][
        "test-compatibility"
    ]
    covered = [step for step in job["steps"] if "--cov=" in step.get("run", "")]
    assert covered and all(
        f"'{COVERAGE_RUNTIME.strip('-py')}'" in step["if"]
        and "github.event_name == 'push'" in step["if"]
        for step in covered
    )
    upload = next(step for step in job["steps"] if step.get("name") == "Upload shard results")
    assert "-py${{ matrix.runtime.python-version }}-" in upload["with"]["name"]


def test_the_never_run_check_reads_every_shard_after_the_matrix() -> None:
    """The check downloads each shard's junit artifact, whatever the shards' verdict, and
    requires a report from every shard of the matrix."""
    jobs = yaml.safe_load((_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))["jobs"]
    matrix = jobs["test-compatibility"]["strategy"]["matrix"]
    job, upload = (
        jobs["never-run"],
        next(
            step
            for step in jobs["test-compatibility"]["steps"]
            if step.get("name") == "Upload shard results"
        ),
    )
    download = next(step for step in job["steps"] if "download-artifact" in step.get("uses", ""))

    assert job["needs"] == "test-compatibility" and job["if"] == "${{ always() }}"
    assert (
        upload["with"]["name"].startswith("pytest-") and download["with"]["pattern"] == "pytest-*"
    )
    assert upload["with"]["name"].endswith("-${{ matrix.group }}")  # the platform is the rest
    shards = len(matrix["group"]) * len(matrix["runtime"])
    assert any(
        step.get("run")
        == f"python3 scripts/check_never_run_tests.py {download['with']['path']} --shards {shards}"
        for step in job["steps"]
    )
    # each shard's two reports, the parallel stage's and the threads stage's, are the ones it reads
    written = {
        re.search(r"--junitxml=(\S+)", step["run"])[1]
        for step in jobs["test-compatibility"]["steps"]
        if "run_test_suite.py" in step.get("run", "")
    }
    assert [(report, _suite_runner()._beside(report)) for report in written] == [_REPORTS]


# A shard's reports: run_test_suite.py's parallel stage, then its serial threads stage.
_REPORTS = ("pytest-results.xml", "pytest-results-threads.xml")


def _junit(
    directory: Path,
    platform: str,
    cases: dict[tuple[str, str], str],
    group: int = 1,
) -> Path:
    """One shard's two reports: ``{(classname, name): outcome}`` in its parallel stage's and
    one passing test in its threads stage's; outcome ``passed``, ``xfail``, ``notrun``
    (``xfail(run=False)``) or a skip message, in the forms pytest writes."""
    shard = directory / f"pytest-{platform}-{group}"
    shard.mkdir(parents=True)
    threads = {("tests.test_ci_contracts", "test_with_the_default_pools"): "passed"}
    for report, stage in zip(_REPORTS, (cases, threads), strict=True):
        body = []
        for (classname, name), outcome in stage.items():
            inner = {
                "passed": "",
                "xfail": '<skipped type="pytest.xfail" message="declared" />',
                "notrun": '<skipped type="pytest.xfail" message="[NOTRUN] disabled until fixed" />',
            }.get(outcome, f'<skipped type="pytest.skip" message="{outcome}" />')
            body.append(
                f'<testcase classname="{classname}" name="{name}" time="0.1">{inner}</testcase>'
            )
        (shard / report).write_text(
            f'<testsuites><testsuite name="pytest">{"".join(body)}</testsuite></testsuites>',
            encoding="utf-8",
        )
    return shard


def _check_never_run(directory: Path, shards: int = 2) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(_ROOT / "scripts/check_never_run_tests.py"),
            str(directory),
            "--shards",
            str(shards),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_the_never_run_check_fails_only_on_a_test_no_platform_ran(tmp_path: Path) -> None:
    """Skipped on one platform, run on another: fine; skipped on both: named, unless run
    elsewhere, which a skip's message shows only when every skip of the test has it and
    the workflow it names runs the test's module."""
    gamlss = (
        "tests.test_two_piece_lss_vertical_slice",
        "test_real_line_family_matches_gamlss_sn2_at_a_parametric_specification",
    )
    book = "[dataset] freMTPL2freq.parquet is not readable"
    cases = {
        ("tests.test_ci_contracts", "test_one_platform"): "requires Windows",
        ("tests.test_ci_contracts", "test_declared_failure"): "xfail",
        ("tests.test_realdata_parity", "test_on_the_book"): book,
        gamlss: "requires R with mgcv and jsonlite",
        ("", "tests.docs.test_notebooks"): "collection skipped",
        ("tests.test_check_fix_tests", "test_runs_on_linux"): "passed",
    }
    on_windows = cases | {
        ("tests.test_ci_contracts", "test_one_platform"): "passed",
        ("", "tests.test_check_fix_tests"): "collection skipped",  # its tests ran on Linux
    }
    del on_windows[("tests.test_check_fix_tests", "test_runs_on_linux")]
    _junit(tmp_path / "ran", "Linux-X64-py3.12", cases)
    _junit(tmp_path / "ran", "Windows-X64-py3.13", on_windows)
    never = {
        ("tests.test_ci_contracts", "test_waits_for_an_event"): "the fit did not reach it",
        ("tests.test_ci_contracts", "test_disabled"): "notrun",
        ("tests.test_ci_contracts", "test_on_a_book_no_workflow_runs"): book,
        ("tests.test_realdata_parity", "test_skipped_for_two_reasons"): book,
        ("tests.test_check_fix_tests", "test_skipped_where_its_module_ran"): "requires a GPU",
    }
    _junit(tmp_path / "never", "Linux-X64-py3.12", cases | never)
    _junit(
        tmp_path / "never",
        "Windows-X64-py3.13",
        on_windows
        | never
        | {("tests.test_realdata_parity", "test_skipped_for_two_reasons"): "requires a GPU"},
    )

    ran, never_ran = _check_never_run(tmp_path / "ran"), _check_never_run(tmp_path / "never")
    assert ran.returncode == 0, ran.stdout + ran.stderr
    assert "1 in the Real data workflow" in ran.stdout and "1 in the docs workflow" in ran.stdout
    assert f"tests/test_two_piece_lss_vertical_slice.py::{gamlss[1]}" in ran.stdout
    assert "test_one_platform" not in ran.stdout and "test_declared_failure" not in ran.stdout
    assert "test_check_fix_tests" not in ran.stdout
    assert never_ran.returncode == 1
    assert f"{len(never)} run on no platform" in never_ran.stdout
    assert "tests/test_ci_contracts.py::test_waits_for_an_event: the fit did not reach it" in (
        never_ran.stdout
    )
    for (classname, name), _ in never.items():
        assert f"{classname.replace('.', '/')}.py::{name}: " in never_ran.stdout, name


@pytest.mark.parametrize(
    ("report", "fault"),
    [(report, fault) for report in _REPORTS for fault in ("missing", "unreadable")]
    + [(_REPORTS[0], "holds no test"), (None, "no artifact")],
)
def test_the_never_run_check_fails_unless_every_shard_sent_both_reports(
    tmp_path: Path, report, fault
) -> None:
    """Either report of a shard missing or truncated, or its parallel one empty, hides what that
    stage skipped, and the check names the shard and the file; a shard that sent nothing is
    named by the platform and the group the other shards show, and a whole runtime missing is
    counted."""
    for platform in ("Linux-X64-py3.12", "Windows-X64-py3.13"):
        for group in (1, 2):
            _junit(tmp_path, platform, {("tests.test_ci_contracts", "test_a"): "passed"}, group)
    shard = tmp_path / "pytest-Windows-X64-py3.13-2"
    complete, runtime_missing = _check_never_run(tmp_path, 4), _check_never_run(tmp_path, 6)
    if fault == "no artifact":
        for name in _REPORTS:
            (shard / name).unlink()
        shard.rmdir()
    elif fault == "missing":
        (shard / report).unlink()
    else:
        text = (
            '<testsuites><testsuite name="pytest">' if fault == "unreadable" else "<testsuites />"
        )
        (shard / report).write_text(text, encoding="utf-8")

    done = _check_never_run(tmp_path, 4)

    assert complete.returncode == 0, complete.stdout + complete.stderr
    assert runtime_missing.returncode == 1 and "2 more sent no artifact" in runtime_missing.stdout
    assert done.returncode == 1, done.stdout + done.stderr
    assert re.search(r"\b3 of 4 shards\b", done.stdout), done.stdout  # the complete shards' count
    faults = [(name, "missing") for name in _REPORTS] if report is None else [(report, fault)]
    for name, why in faults:
        assert f"pytest-Windows-X64-py3.13-2/{name} {why}" in done.stdout


def test_the_never_run_check_accepts_a_threads_stage_that_selected_nothing(tmp_path: Path) -> None:
    """A group with no ``threads`` test writes a readable threads report of zero tests, which
    the suite runner accepts (pytest's exit 5), so the check accepts it too."""
    for group in (1, 2):
        _junit(
            tmp_path, "Linux-X64-py3.12", {("tests.test_ci_contracts", "test_a"): "passed"}, group
        )
    (tmp_path / "pytest-Linux-X64-py3.12-2" / _REPORTS[1]).write_text(
        '<testsuites><testsuite name="pytest" tests="0" /></testsuites>', encoding="utf-8"
    )

    done = _check_never_run(tmp_path, 2)

    assert done.returncode == 0, done.stdout + done.stderr


# A fit's outcome: what an assertion behind ``if <outcome>:`` or a skip waits on.
_OUTCOMES = {"converged", "convergence_reason", "termination_reason"}
_ONE_SIDED = "one-sided by design: a certified result must be right, an unconverged one disclosed"
_IMPLIES = "an implication between two published fields, both outcomes correct behaviour"
_NEVER_RUN = "never runs locally (test-suite audit 2026-11-19): force the event, a follow-up"
_TRAJECTORY = "skips where the fit's path misses its event; runs elsewhere, a follow-up"
_CAPABILITY = "skips on what the machine or a package provides, not on the fit"
# Each existing instance and why it stays; a new one needs a reason here.
_OUTCOME_GUARDS = {
    "test_binomial_log_mean_space.py::test_subnormal_weights_never_certify_a_wrong_maximum": _ONE_SIDED,
    "test_binomial_log_mean_space.py::test_a_subnormal_weight_level_publishes_finite_degrees_of_freedom": _ONE_SIDED,
    "test_binomial_log_mean_space.py::test_a_light_level_is_never_certified_away_from_its_own_maximum": _ONE_SIDED,
    "test_binomial_log_mean_space.py::test_a_weight_ratio_never_certifies_a_level_away_from_its_maximum": _ONE_SIDED,
    "test_binomial_log_mean_space.py::test_a_light_base_cell_of_an_interaction_is_never_certified_away_from_its_maximum": _ONE_SIDED,
    "test_binomial_log_mean_space.py::test_weight_and_penalty_scales_never_certify_a_wrong_maximum": _ONE_SIDED,
    "test_binomial_log_mean_space.py::test_an_intercept_only_fit_at_the_smallest_weight_is_never_certified_wrong": _ONE_SIDED,
    "test_binomial_log_mean_space.py::test_the_block_coordinate_backstop_does_not_read_a_boundary_stop_as_a_drift": _TRAJECTORY,
    "test_constrained_qp.py::test_return_projection_only_receives_an_infeasible_point": "counts the converged solves of its sweep and asserts at least 60",
    "test_constrained_qp.py::test_an_exchange_whose_dual_rows_overflow_when_squared_still_certifies": _ONE_SIDED
    + "; the warm solve's outcome depends on the BLAS kernel",
    "test_declared_scalar_vertical_slice.py::test_smooth_reml_starts_backends_compaction_and_serialization": _IMPLIES,
    "test_distributional_newton_curvature_refresh.py::test_failed_curvature_refresh_keeps_bounded_bfgs_fallback": _IMPLIES,
    "test_distributional_newton_practical_handoff.py::test_a_finish_that_moved_far_and_was_rejected_does_not_relabel_where_it_stalled": _TRAJECTORY,
    "test_distributional_newton_practical_handoff.py::test_a_finish_on_a_face_that_cannot_reach_stationary_leaves_the_stop_to_the_strict_test": _IMPLIES,
    "test_distributional_perf_levers.py::test_narrow_fit_keeps_blas_capped_regardless_of_row_count": _CAPABILITY,
    "test_distributional_score_units.py::test_gaussian_fit_reaches_the_centred_mle_or_discloses_the_stop": _ONE_SIDED,
    "test_factor_smooth_sz_thin_and_influence.py::test_sz_same_x_reml_decisions_do_not_follow_the_blas_thread_count": _CAPABILITY,
    "test_generalized_gamma_lss_vertical_slice.py::test_infinite_mean_data_either_diagnose_or_stop_at_the_boundary": _NEVER_RUN,
    "test_gram_block_queue.py::test_pooled_gram_under_default_thread_pools_holds_blas_at_one_thread": _CAPABILITY,
    "test_negative_binomial_lss_efs.py::test_claims_book_shaped_fit_finishes_instead_of_refusing": _NEVER_RUN,
    "test_negative_binomial_lss_efs.py::test_size_reaching_its_poisson_limit_during_selection_still_converges": _TRAJECTORY,
    "test_plot_api.py::test_term_dropdown_resets_to_response_scale": "skips on its fixture's term count, not on a fit",
    "test_probit_link_kernels.py::test_probit_eta_kernels_keep_parity_with_the_scipy_stats_master_used": _CAPABILITY,
    "test_probit_link_kernels.py::test_probit_deriv_keeps_parity_with_scipy_stats": _CAPABILITY,
    "test_rank_deficient_complete_fit.py::test_dispatch_comes_from_a_live_process_not_build_metadata": _CAPABILITY,
    "test_tweedie_lss_vertical_slice.py::test_automatic_smoothing_tweedie_lss_fit": _IMPLIES,
}


_OUTCOME_WORDS = re.compile("|".join([*_OUTCOMES, r"pytest\.skip"]))


def _outcome_guards(source: str) -> set[str]:
    """Test functions that assert behind ``if <fit outcome>:``, assert ``not converged or ...``,
    or call ``pytest.skip`` under an ``if`` on a value the test computed."""
    if not _OUTCOME_WORDS.search(source):
        return set()  # each pattern names an outcome or pytest.skip; most modules do neither

    def names(node) -> set[str]:
        return {
            getattr(n, "attr", None) or getattr(n, "id", None) or str(getattr(n, "value", ""))
            for n in ast.walk(node)
        }

    def calls(nodes, name) -> bool:
        return any(
            isinstance(n, ast.Call) and ast.unparse(n.func) == name
            for b in nodes
            for n in ast.walk(b)
        )

    def asserts(nodes) -> bool:
        """An ``assert``, ``pytest.fail`` or an ``assert_*`` call, NumPy's (``np.testing.``) or
        bare; a method or an underscored helper that asserts is not seen."""
        return any(
            isinstance(n, ast.Assert)
            or (
                isinstance(n, ast.Call)
                and re.fullmatch(r"pytest\.fail|(np\.testing\.)?assert_\w+", ast.unparse(n.func))
                is not None
            )
            for b in nodes
            for n in ast.walk(b)
        )

    flagged = set()
    for test in ast.walk(ast.parse(source)):
        if not isinstance(test, ast.FunctionDef) or not test.name.startswith("test"):
            continue
        computed = {
            n.id for n in ast.walk(test) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
        } - {a.arg for a in test.args.args}
        for node in ast.walk(test):
            guarded = isinstance(node, ast.If) and (
                (names(node.test) & _OUTCOMES and asserts(node.body) and not asserts(node.orelse))
                or (calls(node.body, "pytest.skip") and names(node.test) & computed)
            )
            either = (
                isinstance(node, ast.Assert)
                and isinstance(node.test, ast.BoolOp)
                and isinstance(node.test.op, ast.Or)
                and isinstance(node.test.values[0], ast.UnaryOp)
                and isinstance(node.test.values[0].op, ast.Not)
                and names(node.test.values[0]) & _OUTCOMES
            )
            if guarded or either:
                flagged.add(test.name)
    return flagged


def test_no_assertion_waits_on_a_fit_outcome_without_a_reason() -> None:
    """An assertion behind ``if result.converged:`` checks nothing on a fit that does not
    converge, and a skip on the fit's path passes where the event never comes (AGENTS.md:
    a test forces its event). Each one is forced, or carries its reason here."""
    found = {
        f"{path.relative_to(_ROOT / 'tests').as_posix()}::{name}"
        for path in sorted((_ROOT / "tests").rglob("test_*.py"))
        for name in _outcome_guards(path.read_text(encoding="utf-8"))
    }
    assert sorted(found - _OUTCOME_GUARDS.keys()) == [], "force the event, or give a reason"
    assert sorted(_OUTCOME_GUARDS.keys() - found) == [], "these no longer wait on an outcome"


def test_the_outcome_lint_flags_each_pattern_and_spares_forced_events() -> None:
    flagged = _outcome_guards(
        "def test_guard(fit):\n    r = fit()\n    if r.converged:\n        assert r.x\n"
        "def test_either(fit):\n    assert not fit().converged or ok\n"
        "def test_skip(fit):\n    k = fit().n_iter\n    if k < 10:\n        pytest.skip('early')\n"
        "def test_forced(fit):\n    r = fit()\n    assert r.converged\n    assert r.x\n"
        "def test_both(fit):\n    r = fit()\n    if r.converged:\n        assert r.x\n"
        "    else:\n        assert r.warned\n"
        "def test_parameter(case):\n    if case == 'n/a':\n        pytest.skip('does not apply')\n"
        "def test_numpy(fit):\n    r = fit()\n    if r.converged:\n"
        "        np.testing.assert_allclose(r.beta, 0.0)\n"
        "def test_imported(fit):\n    r = fit()\n    if r.converged:\n"
        "        assert_array_equal(r.beta, 0.0)\n"
    )
    assert flagged == {"test_guard", "test_either", "test_skip", "test_numpy", "test_imported"}


_RERUN_WORKFLOW = _ROOT / ".github/workflows/rerun-starved-shards.yml"
_RERUN_FIXTURES = _ROOT / "tests/fixtures/ci_rerun"
# Recorded from the API: run 44552382922's first attempt, whose macOS ARM64 shard C GitHub
# cancelled without a runner, and run 44482549977's, which failed on a test.
_STARVED_RUN, _STARVED_JOB = "44552382922", 135659280345
_FAILED_RUN = "44482549977"


def test_the_starved_shard_rerun_fires_once_on_a_failed_ci_run_and_runs_no_code() -> None:
    workflow = yaml.safe_load(_RERUN_WORKFLOW.read_text(encoding="utf-8"))
    ci = yaml.safe_load((_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    # PyYAML reads the bare key ``on`` as True.
    assert workflow[True] == {"workflow_run": {"workflows": [ci["name"]], "types": ["completed"]}}
    assert workflow["permissions"] == {"contents": "read"}
    (job,) = workflow["jobs"].values()
    assert job["permissions"] == {"actions": "write"}
    assert "github.event.workflow_run.run_attempt == 1" in job["if"]
    assert "github.event.workflow_run.conclusion == 'failure'" in job["if"]
    assert all("uses" not in step for step in job["steps"])  # no checkout, no action
    # The check it unblocks stays strict: every shard must still report.
    (never_run,) = [step["run"] for step in ci["jobs"]["never-run"]["steps"] if "run" in step]
    assert "--shards 20" in never_run


_FAKE_GH = """#!{python}
import json, pathlib, sys
fixtures = pathlib.Path({fixtures!r})
with open(fixtures / "calls.jsonl", "a", encoding="utf-8") as log:
    log.write(json.dumps(sys.argv[1:]) + "\\n")
if sys.argv[1] == "api":
    name = "jobs.json" if "/jobs" in sys.argv[2] else sys.argv[2].split("/")[-2] + ".json"
    path = fixtures / name
    print(path.read_text(encoding="utf-8") if path.exists() else "[]")
"""


def _rerun_calls(tmp_path: Path, run: str, jobs: dict, annotations: dict[int, list]) -> list:
    """The ``gh`` calls the workflow's script makes against recorded responses."""
    fixtures = tmp_path / "gh"
    fixtures.mkdir()
    (fixtures / "jobs.json").write_text(json.dumps(jobs), encoding="utf-8")
    for job_id, notes in annotations.items():
        (fixtures / f"{job_id}.json").write_text(json.dumps(notes), encoding="utf-8")
    gh = fixtures / "gh"
    gh.write_text(_FAKE_GH.format(python=sys.executable, fixtures=str(fixtures)), "utf-8")
    gh.chmod(0o755)
    workflow = yaml.safe_load(_RERUN_WORKFLOW.read_text(encoding="utf-8"))
    ((step,),) = [job["steps"] for job in workflow["jobs"].values()]
    script = tmp_path / "rerun.py"
    script.write_text(step["run"], encoding="utf-8")
    env = {
        **os.environ,
        "PATH": f"{fixtures}{os.pathsep}{os.environ['PATH']}",
        "GH_REPO": "owner/repo",
        "RUN_ID": run,
    }
    subprocess.run([sys.executable, str(script)], env=env, check=True, capture_output=True)
    lines = (fixtures / "calls.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def _recorded(name: str):
    return json.loads((_RERUN_FIXTURES / name).read_text(encoding="utf-8"))


@pytest.mark.skipif(os.name == "nt", reason="the stand-in gh is a script with a #! line")
def test_the_rerun_filter_reruns_a_run_whose_shard_never_got_a_runner(tmp_path: Path) -> None:
    jobs = _recorded("jobs_starved.json")
    annotations = {_STARVED_JOB: _recorded("annotations_starved.json")}
    calls = _rerun_calls(tmp_path, _STARVED_RUN, jobs, annotations)
    assert ["run", "rerun", _STARVED_RUN, "--failed"] in calls
    assert sum(call[:2] == ["run", "rerun"] for call in calls) == 1


@pytest.mark.skipif(os.name == "nt", reason="the stand-in gh is a script with a #! line")
def test_the_rerun_filter_leaves_a_run_that_failed_on_a_test(tmp_path: Path) -> None:
    calls = _rerun_calls(tmp_path, _FAILED_RUN, _recorded("jobs_failed.json"), {})
    assert not any(call[:2] == ["run", "rerun"] for call in calls)


@pytest.mark.skipif(os.name == "nt", reason="the stand-in gh is a script with a #! line")
@pytest.mark.parametrize(
    ("started", "annotated", "reran"),
    [
        (True, True, True),  # the annotation alone identifies it
        (False, False, True),  # a cancelled job that never started a step had no runner
        (True, False, False),  # cancelled after it started: not starvation
    ],
)
def test_the_rerun_filter_reads_either_sign_of_a_starved_job(
    tmp_path: Path, started: bool, annotated: bool, reran: bool
) -> None:
    jobs = _recorded("jobs_starved.json")
    (starved,) = [job for job in jobs["jobs"] if job["id"] == _STARVED_JOB]
    if started:
        starved["steps"] = [
            {"name": "Set up job", "status": "completed", "conclusion": "cancelled"}
        ]
        starved["runner_name"] = "GitHub Actions 1"
    notes = _recorded("annotations_starved.json") if annotated else []
    calls = _rerun_calls(tmp_path, _STARVED_RUN, jobs, {_STARVED_JOB: notes})
    assert any(call[:2] == ["run", "rerun"] for call in calls) == reran
