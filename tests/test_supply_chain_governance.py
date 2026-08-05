import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_FILES = (
    ".github/workflows/ci.yml",
    ".github/workflows/dev-ci.yml",
    ".github/workflows/docs.yml",
    ".github/workflows/release.yml",
    ".github/workflows/scorecard.yml",
    ".github/workflows/security.yml",
)


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _workflow_header(workflow: str) -> str:
    return workflow.split("jobs:", maxsplit=1)[0]


def _required_uv_version() -> str:
    """Return the exact uv version pyproject requires.

    Derived rather than hard-coded: the release job installs a pinned uv, and
    ``uv build`` refuses to run when that pin disagrees with
    ``[tool.uv] required-version``. A literal here goes stale silently the
    moment the floor moves, and the first symptom is a release tag that fails
    after it has already been pushed.
    """
    pyproject = tomllib.loads(_read("pyproject.toml"))
    required = pyproject["tool"]["uv"]["required-version"]
    match = re.fullmatch(r"==(\d+\.\d+\.\d+)", required)
    assert match, f"expected an exact uv pin in pyproject, got {required!r}"
    return match.group(1)


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
        "uvx --from 'build==1.2.2.post1' python -m build",
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


def test_workflow_actions_are_pinned_to_full_commit_shas():
    uses_pattern = re.compile(r"uses:\s+[^@\s]+@([^\s#]+)")

    for path in WORKFLOW_FILES:
        workflow = _read(path)
        refs = uses_pattern.findall(workflow)
        assert refs, path
        for ref in refs:
            assert re.fullmatch(r"[0-9a-f]{40}", ref), f"{path} uses unpinned ref {ref}"


def test_security_workflow_avoids_hash_unpinned_pip_installs():
    workflow = _read(".github/workflows/security.yml")

    assert "pip install" not in workflow
    assert "python -m pip" not in workflow


def test_release_workflow_publishes_checked_artifacts_from_version_tags():
    workflow = _read(".github/workflows/release.yml")

    assert "name: Publish release" in workflow
    assert 'tags: ["v*.*.*"]' in workflow
    assert "workflow_dispatch:" not in workflow
    assert f'version: "{_required_uv_version()}"' in workflow
    assert "Verify release tag and source version" in workflow
    assert 'expected_tag = f"v{project_version}"' in workflow
    assert "source_version != project_version" in workflow
    assert 'git merge-base --is-ancestor "$GITHUB_SHA" "origin/master"' in workflow
    assert "persist-credentials: false" in workflow
    assert "enable-cache: false" in workflow

    assert "uv build --out-dir dist" in workflow
    assert "twine check dist/*" in workflow
    assert "check-wheel-contents dist/*.whl" in workflow
    assert "python scripts/verify_release_artifacts.py dist" in workflow
    assert workflow.count("name: release-distributions") == 3


def test_release_workflow_uses_trusted_publishing_and_least_privilege():
    workflow = _read(".github/workflows/release.yml")
    header = _workflow_header(workflow)
    publish_job = workflow.split("  publish:", maxsplit=1)[1].split(
        "  github-release:", maxsplit=1
    )[0]
    release_job = workflow.split("  github-release:", maxsplit=1)[1]

    assert "permissions:" in header
    assert "contents: read" in header
    assert "id-token: write" not in header
    assert "contents: write" not in header

    assert "needs: build" in publish_job
    assert "name: pypi" in publish_job
    assert "url: https://pypi.org/p/superglm" in publish_job
    assert "id-token: write" in publish_job
    assert "contents: write" not in publish_job
    assert "pypa/gh-action-pypi-publish@" in publish_job

    assert "needs: publish" in release_job
    assert "contents: write" in release_job
    assert "id-token: write" not in release_job
    assert "gh release view" in release_job
    assert "gh release create" in release_job
    assert "gh release upload" in release_job
    assert "GH_REPO: ${{ github.repository }}" in release_job
    assert "--clobber" in release_job
    assert "--verify-tag" in release_job
    assert "--generate-notes" in release_job


def test_release_workflow_uses_node24_artifact_actions():
    workflow = _read(".github/workflows/release.yml")

    assert "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a" in workflow
    assert workflow.count("actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c") == 2


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


def test_ci_browser_suites_run_in_separate_pytest_processes():
    legacy = "uv run pytest tests/test_editor_browser.py -m browser --run-browser -q"
    workspace = "uv run pytest tests/editor/ -m browser --run-browser -q"

    for path in (".github/workflows/ci.yml", ".github/workflows/dev-ci.yml"):
        workflow = _read(path)
        assert legacy in workflow, path
        assert workspace in workflow, path
        assert "pytest tests/test_editor_browser.py tests/editor/" not in workflow, path
        assert "pytest tests/editor/ tests/test_editor_browser.py" not in workflow, path


def test_master_ci_runs_complete_supported_python_matrix_efficiently():
    workflow = _read(".github/workflows/ci.yml")
    header = _workflow_header(workflow)

    compatibility_job = workflow.split("  test-compatibility:", maxsplit=1)[1].split(
        "  test-coverage:", maxsplit=1
    )[0]
    coverage_shard_job = workflow.split("  test-coverage:", maxsplit=1)[1].split(
        "  coverage:", maxsplit=1
    )[0]
    coverage_job = workflow.split("  coverage:", maxsplit=1)[1].split("  lint:", maxsplit=1)[0]

    assert '      - ".test_durations"' in header

    assert "fail-fast: false" in compatibility_job
    for version in ("3.12", "3.14"):
        assert f'"{version}"' in compatibility_job
    # 3.13 gets dedicated coverage-shard jobs below; running it here too is waste.
    assert '"3.13"' not in compatibility_job
    assert "uv run --with pyarrow pytest tests/" in compatibility_job
    assert '-m "not browser"' in compatibility_job
    assert "--splits" not in compatibility_job
    assert "--cov" not in compatibility_job
    assert "ruff check" not in compatibility_job

    assert "fail-fast: false" in coverage_shard_job
    assert "uv python install 3.13" in coverage_shard_job
    for group, label in enumerate(("A", "B", "C", "D"), start=1):
        assert f"- group: {group}" in coverage_shard_job
        assert f"label: {label}" in coverage_shard_job
    assert "uv run --with pyarrow pytest tests/" in coverage_shard_job
    assert '-m "not browser"' in coverage_shard_job
    assert "--splits 4" in coverage_shard_job
    assert "--group ${{ matrix.group }}" in coverage_shard_job
    assert "--splitting-algorithm least_duration" in coverage_shard_job
    assert "--cov=superglm" in coverage_shard_job
    assert "--cov-branch" in coverage_shard_job
    assert "--cov-report=" in coverage_shard_job
    assert "COVERAGE_FILE: .coverage.${{ matrix.group }}" in coverage_shard_job
    assert "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a" in coverage_shard_job
    assert "name: coverage-py313-${{ matrix.group }}" in coverage_shard_job
    assert "path: .coverage.${{ matrix.group }}" in coverage_shard_job
    assert "if-no-files-found: error" in coverage_shard_job
    assert "include-hidden-files: true" in coverage_shard_job

    assert "needs: test-coverage" in coverage_job
    assert "actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c" in coverage_job
    assert "pattern: coverage-py313-*" in coverage_job
    assert "merge-multiple: true" in coverage_job
    assert "uv run coverage combine coverage-data" in coverage_job
    assert "uv run coverage xml -o coverage.xml" in coverage_job
    assert "codecov/codecov-action@fb8b3582c8e4def4969c97caa2f19720cb33a72f" in coverage_job

    assert workflow.count("uv run ruff check src/ tests/") == 1
    assert workflow.count("uv run ruff format --check src/ tests/") == 1


def test_dev_ci_parallelizes_complete_python314_suite():
    workflow = _read(".github/workflows/dev-ci.yml")

    assert "  quick-check:" not in workflow
    assert "  py314-full:" not in workflow
    for job in ("quality", "docs", "frontend", "browser", "type-check", "pytest-314"):
        assert f"  {job}:" in workflow

    pytest_job = workflow.split("  pytest-314:", maxsplit=1)[1]
    assert "name: ${{ matrix.label }}" in pytest_job
    assert "fail-fast: false" in pytest_job
    assert "include:" in pytest_job
    for group in range(1, 5):
        assert f"- group: {group}" in pytest_job
    for label in (
        "Python 3.14 · non-browser regression suite · balanced A",
        "Python 3.14 · non-browser regression suite · balanced B",
        "Python 3.14 · non-browser regression suite · balanced C",
        "Python 3.14 · non-browser regression suite · balanced D",
    ):
        assert f"label: {label}" in pytest_job
    assert "uv python install 3.14" in pytest_job
    assert "uv sync --python 3.14 --extra dev" in pytest_job
    assert "uv run --with pyarrow pytest tests/" in pytest_job
    assert '-m "not browser"' in pytest_job
    assert "--splits 4" in pytest_job
    assert "--group ${{ matrix.group }}" in pytest_job
    assert "--splitting-algorithm least_duration" in pytest_job
    assert "--maxfail=1" in pytest_job


def test_dev_ci_keeps_browser_and_non_test_checks_independent():
    workflow = _read(".github/workflows/dev-ci.yml")

    quality_job = workflow.split("  quality:", maxsplit=1)[1].split("  docs:", maxsplit=1)[0]
    docs_job = workflow.split("  docs:", maxsplit=1)[1].split("  frontend:", maxsplit=1)[0]
    frontend_job = workflow.split("  frontend:", maxsplit=1)[1].split("  browser:", maxsplit=1)[0]
    browser_job = workflow.split("  browser:", maxsplit=1)[1].split("  pytest-314:", maxsplit=1)[0]

    assert "ruff check src/ tests/" in quality_job
    assert "ruff format --check src/ tests/" in quality_job
    assert "mkdocs build --strict" in docs_job
    assert "npm run check:frontend" in frontend_job
    assert "playwright install --with-deps chromium" in browser_job
    assert "pytest tests/test_editor_browser.py" in browser_job
    assert "pytest tests/editor/" in browser_job


def test_pre_push_pytest_uses_uv_dev_environment():
    config = _read(".pre-commit-config.yaml")
    pytest_hook = config.split("- id: pytest", maxsplit=1)[1]

    assert 'entry: uv run --extra dev python -m pytest tests/ -q -m "not slow"' in pytest_hook
    assert "language: system" in pytest_hook


def test_security_archive_check_requires_modular_editor_assets():
    workflow = _read(".github/workflows/security.yml")
    release_workflow = _read(".github/workflows/release.yml")
    verifier = _read("scripts/verify_release_artifacts.py")
    editor_root = ROOT / "src/superglm/editor/app"
    current_assets = {
        f"superglm/editor/app/{path.relative_to(editor_root).as_posix()}"
        for path in editor_root.rglob("*")
        if path.is_file() and (path.name == "index.html" or path.suffix in {".js", ".css"})
    }
    derivation_markers = (
        'editor_root = source_root / "src/superglm/editor/app"',
        'for path in editor_root.rglob("*")',
        'path.name == "index.html"',
        'path.suffix in {".js", ".css"}',
        "path.relative_to(editor_root).as_posix()",
    )

    representative_assets = {
        "superglm/editor/app/index.html",
        "superglm/editor/app/main.js",
        "superglm/editor/app/chart/geometry.js",
        "superglm/editor/app/styles/tokens.css",
        "superglm/editor/app/views/popover.js",
    }

    assert current_assets
    assert all(asset.startswith("superglm/editor/app/") for asset in current_assets)
    assert representative_assets <= current_assets
    for marker in derivation_markers:
        assert marker in verifier
    assert 'glob("*.whl")' in verifier
    assert 'glob("*.tar.gz")' in verifier
    assert '"superglm/editor/app/index.html"' not in verifier
    for consumer in (workflow, release_workflow):
        assert "python scripts/verify_release_artifacts.py dist" in consumer


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


def test_dependabot_groups_python_and_github_actions_updates():
    dependabot = _read(".github/dependabot.yml")
    uv_config = dependabot.split('package-ecosystem: "uv"', maxsplit=1)[1].split(
        'package-ecosystem: "github-actions"', maxsplit=1
    )[0]
    actions_config = dependabot.split('package-ecosystem: "github-actions"', maxsplit=1)[1]

    assert "groups:" in uv_config
    assert "python-dependencies:" in uv_config
    assert "patterns:" in uv_config
    assert '- "*"' in uv_config

    assert "groups:" in actions_config
    assert "github-actions:" in actions_config
    assert "patterns:" in actions_config
    assert '- "*"' in actions_config


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
    assert "scripts/verify_release_artifacts.py" in codeowners
    assert "src/superglm/" in codeowners
    for release_surface in (
        ".codex/agents/",
        ".github/PULL_REQUEST_TEMPLATE.md",
        "AGENTS.md",
        "docs/development/releases.md",
        "scripts/bump_version.py",
    ):
        assert release_surface in codeowners
