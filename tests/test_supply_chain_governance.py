from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _workflow_header(workflow: str) -> str:
    return workflow.split("jobs:", maxsplit=1)[0]


def test_security_workflow_runs_on_master_prs_and_schedule():
    workflow = _read(".github/workflows/security.yml")

    assert "name: Security / Supply Chain" in workflow
    assert "pull_request:" in workflow
    assert "branches: [master]" in workflow
    assert "schedule:" in workflow
    assert 'cron: "0 6 * * 1"' in workflow
    assert "permissions:" in workflow
    assert "contents: read" in workflow


def test_security_workflow_collects_core_governance_evidence():
    workflow = _read(".github/workflows/security.yml")

    expected_markers = [
        "github/codeql-action/init",
        "github/codeql-action/analyze",
        "actions/dependency-review-action",
        "pip-audit",
        "python -m build",
        "twine check dist/*",
        "check-wheel-contents",
        "cyclonedx-py environment",
        "actions/upload-artifact",
        "cyclonedx-sbom.json",
        "package-dist",
        "retention-days: 30",
    ]

    for marker in expected_markers:
        assert marker in workflow


def test_scorecard_workflow_uploads_sarif_on_master_and_schedule():
    workflow = _read(".github/workflows/scorecard.yml")

    assert "name: OpenSSF Scorecard" in workflow
    assert "branches: [master]" in workflow
    assert "schedule:" in workflow
    assert "ossf/scorecard-action" in workflow
    assert "results_format: sarif" in workflow
    assert "github/codeql-action/upload-sarif" in workflow
    assert "security-events: write" in workflow
    assert "upload-sarif:" in workflow
    assert "needs: scorecard" in workflow
    assert "name: scorecard-results" in workflow
    assert (
        "if: github.event_name == 'branch_protection_rule' || github.ref == 'refs/heads/master'"
    ) in workflow


def test_scorecard_generation_permissions_allow_openssf_publication():
    workflow = _read(".github/workflows/scorecard.yml")
    workflow_header = _workflow_header(workflow)
    scorecard_job = workflow.split("  scorecard:", maxsplit=1)[1].split(
        "  upload-sarif:", maxsplit=1
    )[0]

    assert "id-token: write" not in workflow_header
    assert "security-events: write" not in scorecard_job
    assert "id-token: write" in scorecard_job
    assert "publish_results: true" in scorecard_job


def test_ci_workflows_define_read_only_top_level_permissions():
    ci = _read(".github/workflows/ci.yml")
    dev_ci = _read(".github/workflows/dev-ci.yml")

    for workflow in (ci, dev_ci):
        header = _workflow_header(workflow)
        assert "permissions:" in header
        assert "contents: read" in header
        assert "contents: write" not in header
        assert "id-token: write" not in header
        assert "security-events: write" not in header


def test_docs_workflow_scopes_write_permission_to_deploy_job():
    workflow = _read(".github/workflows/docs.yml")
    header = _workflow_header(workflow)
    deploy_job = workflow.split("  deploy:", maxsplit=1)[1]

    assert "permissions:" in header
    assert "contents: read" in header
    assert "contents: write" not in header

    assert "permissions:" in deploy_job
    assert "contents: write" in deploy_job


def test_dependabot_updates_python_and_github_actions():
    dependabot = _read(".github/dependabot.yml")

    assert 'package-ecosystem: "uv"' in dependabot
    assert 'package-ecosystem: "github-actions"' in dependabot
    assert 'directory: "/"' in dependabot
    assert 'interval: "weekly"' in dependabot


def test_security_policy_and_codeowners_cover_governance_surfaces():
    security_policy = _read("SECURITY.md")
    security_policy_lower = security_policy.lower()
    codeowners = _read(".github/CODEOWNERS")

    assert "reporting a vulnerability" in security_policy_lower
    assert "codeql code scanning" in security_policy_lower
    assert "dependency vulnerability scanning" in security_policy_lower
    assert "sbom generation" in security_policy_lower
    assert "package build/content checks" in security_policy_lower

    assert ".github/workflows/" in codeowners
    assert ".github/CODEOWNERS" in codeowners
    assert "SECURITY.md" in codeowners
    assert "pyproject.toml" in codeowners
    assert "src/superglm/" in codeowners
