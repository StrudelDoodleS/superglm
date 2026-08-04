"""Regression checks for merge-gate configuration and shard metadata."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

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
    workflow = (_ROOT / ".github/workflows/dev-ci.yml").read_text(encoding="utf-8")
    names = _check_run_names(workflow)
    floor = [
        block for job_id, block in _jobs(workflow).items() if _PYTHON_FLOOR_CHECK in names[job_id]
    ]

    assert "pull_request:" in workflow
    assert "\n  push:\n" not in workflow
    assert "workflow_dispatch:" not in workflow
    assert len(floor) == 1, f"exactly one job must publish {_PYTHON_FLOOR_CHECK!r}"
    assert "uv python install 3.12" in floor[0], (
        "the Python floor job must run on the declared minimum version"
    )
    assert "uv run pytest tests/" in floor[0], "the Python floor job must run the test suite"
    assert "--extra bench --extra plotting" in floor[0], (
        "the Python floor job must install the bench and plotting extras, so that the "
        "oracle and plotly tests run somewhere that executes tests"
    )
    assert "continue-on-error: true" not in workflow


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
    tests.  Dropping the extras from the Python floor job therefore has to fail
    the contract even though the bare string survives elsewhere in the file.
    """
    dev_ci = (_ROOT / ".github/workflows/dev-ci.yml").read_text(encoding="utf-8")
    compatibility = (_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    floor_job = _jobs(dev_ci)["pytest-312"]
    mutant = dev_ci.replace(floor_job, floor_job.replace(" --extra bench --extra plotting", ""))

    assert mutant != dev_ci
    assert "--extra bench --extra plotting" not in _jobs(mutant)["pytest-312"]
    assert "--extra bench --extra plotting" in mutant

    _mirror_workflows(tmp_path, dev_ci=mutant, compatibility=compatibility)
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
