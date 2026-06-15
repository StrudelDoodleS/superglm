from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


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
