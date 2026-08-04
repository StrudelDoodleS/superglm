"""Regression checks for merge-gate configuration and shard metadata."""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


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


def test_required_workflow_runs_for_pull_requests_and_python_floor() -> None:
    workflow = (_ROOT / ".github/workflows/dev-ci.yml").read_text(encoding="utf-8")

    assert "pull_request:" in workflow
    assert "\n  push:\n" not in workflow
    assert "workflow_dispatch:" not in workflow
    assert "Python 3.12 · required non-browser suite" in workflow
    assert "--extra bench --extra plotting" in workflow
    assert "continue-on-error: true" not in workflow


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
    required = (_ROOT / ".github/workflows/dev-ci.yml").read_text(encoding="utf-8")
    compatibility = (_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "\n  frontend:\n" in required
    assert "\n  frontend-browser:\n" in compatibility


def test_coverage_omit_targets_the_plotting_package() -> None:
    config = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    omit = config["tool"]["coverage"]["run"]["omit"]

    assert omit == ["src/superglm/plotting/*"]
    assert sorted((_ROOT / "src/superglm/plotting").glob("*.py"))
