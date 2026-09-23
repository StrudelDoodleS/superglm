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


def test_duration_manifest_covers_the_non_browser_suite() -> None:
    recorded = json.loads((_ROOT / ".test_durations").read_text(encoding="utf-8"))
    collected = _collect_non_browser_nodeids()
    covered = collected.intersection(recorded)

    assert collected
    assert len(covered) / len(collected) >= 0.95


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
    assert "needs: test-compatibility" in floor[0]
    matrix = _jobs(workflow)["test-compatibility"]
    assert {case for case in _compatibility_cases(matrix) if case[0] == "3.12"} == {
        ("3.12", "ubuntu-latest", group, label, "") for group, label in enumerate("ABCD", start=1)
    }
    assert "uv run --with mpmath python scripts/run_test_suite.py" in matrix
    assert "--extra dev --extra bench --extra plotting" in matrix, (
        "the compatibility test matrix must install the bench and plotting extras"
    )
    assert "continue-on-error: true" not in workflow


def test_compatibility_shards_do_not_queue_platforms_behind_each_other() -> None:
    workflow = (_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    block = _jobs(workflow)["test-compatibility"]
    strategy = yaml.safe_load(block)["test-compatibility"]["strategy"]
    cases = _compatibility_cases(block)
    assert strategy.get("max-parallel", len(cases)) >= len(cases)


def test_compatibility_shards_collect_all_failures_with_a_hang_limit() -> None:
    workflow = (_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    job = yaml.safe_load(workflow)["jobs"]["test-compatibility"]
    assert 0 < job.get("timeout-minutes", 0) <= 15
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


@pytest.mark.parametrize("result", ["success", "failure", "cancelled", "skipped"])
def test_python_floor_aggregate_executes_the_matrix_verdict(result: str) -> None:
    workflow = (_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    aggregate = _jobs(workflow)["python-floor"]
    assert "TEST_RESULT: ${{ needs.test-compatibility.result }}" in aggregate
    commands = re.findall(r"(?m)^        run: (.+)$", aggregate)
    assert len(commands) == 1
    command = shlex.split(commands[0])
    assert command[:2] == ["python", "-c"]
    completed = subprocess.run(
        [sys.executable, *command[1:]],
        env=os.environ | {"TEST_RESULT": result},
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert (completed.returncode == 0) == (result == "success"), completed.stderr


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
