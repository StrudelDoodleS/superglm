import json
import os
import re
import subprocess
import tomllib
from datetime import datetime
from pathlib import Path

from packaging.version import Version

from tests.test_ci_contracts import _check_run_names, _compatibility_cases, _jobs

ROOT = Path(__file__).resolve().parents[1]


def _workflow_files() -> tuple[str, ...]:
    """Every workflow in the tree, rather than a list someone must remember to extend.

    The pinning check below is worth exactly its coverage, and a hardcoded tuple
    omits whatever lands next without saying so -- ``claude.yml`` was already
    missing from it, and ``real-data.yml`` would have been too.  An unchecked
    workflow reads identically to a checked one, which is the same failure this
    branch is fixing for skipped tests.
    """
    directory = ROOT / ".github" / "workflows"
    found = sorted(
        str(path.relative_to(ROOT).as_posix())
        for path in (*directory.glob("*.yml"), *directory.glob("*.yaml"))
    )
    assert found, f"no workflows found under {directory}"
    return tuple(found)


WORKFLOW_FILES = _workflow_files()


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
    """Every ``uses:`` in the tree names a 40-hex commit, and some ``uses:`` exists.

    The emptiness check is on the SCAN, not per file: ``WORKFLOW_FILES`` became
    directory-derived, and a legitimate future workflow whose steps are all
    ``run:`` would have failed a per-file ``assert refs`` on a technicality --
    it pins nothing because it uses nothing.  Keeping one global assertion is
    what stops the pattern silently matching nothing, which is the failure the
    per-file version was really guarding against.
    """
    uses_pattern = re.compile(r"uses:\s+[^@\s]+@([^\s#]+)")

    scanned = 0
    for path in WORKFLOW_FILES:
        for ref in uses_pattern.findall(_read(path)):
            scanned += 1
            assert re.fullmatch(r"[0-9a-f]{40}", ref), f"{path} uses unpinned ref {ref}"
    assert scanned, f"no `uses:` found in any of {WORKFLOW_FILES}; the pattern matched nothing"


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

    jobs = _jobs(workflow)
    compatibility_job = jobs["test-compatibility"]
    coverage_job = jobs["coverage"]

    assert '      - ".test_durations"' in header

    cases = _compatibility_cases(compatibility_job)
    assert len(cases) == 20
    assert set(cases) == {
        (version, os, group, label, suffix)
        for version, os, suffix in (
            ("3.12", "ubuntu-latest", ""),
            ("3.14", "ubuntu-latest", ""),
            ("3.13", "windows-2025", " · Windows"),
            ("3.13", "macos-15", " · macOS ARM64"),
            ("3.13", "ubuntu-24.04-arm", " · Linux ARM64"),
        )
        for group, label in enumerate("ABCD", start=1)
    }
    assert set(_check_run_names(workflow)["test-compatibility"]) == {
        f"Python {version} · non-browser regression suite · balanced {label}{suffix}"
        for version, _os, _group, label, suffix in cases
    }
    assert "runs-on: ${{ matrix.runtime.os }}" in compatibility_job
    assert "uv sync --locked --python ${{ matrix.runtime.python-version }}" in compatibility_job
    assert "--extra dev --extra bench --extra plotting" in compatibility_job
    assert "ruff check" not in compatibility_job

    run_steps = re.findall(r"(?ms)^      - name: [^\n]+\n(.*?)(?=^      - |\Z)", compatibility_job)
    runner = "uv run --with mpmath python scripts/run_test_suite.py"
    pytest_steps = [step for step in run_steps if runner in step]
    assert len(pytest_steps) == compatibility_job.count("scripts/run_test_suite.py") == 2
    regression, coverage = pytest_steps
    # Complementary conditions make coverage replace the normal invocation.
    assert (
        "if: github.event_name != 'push' || matrix.runtime.python-version != '3.12'" in regression
    )
    assert "if: github.event_name == 'push' && matrix.runtime.python-version == '3.12'" in coverage
    for step in pytest_steps:
        assert '-m "not browser and not docs"' in step
        assert "--splits 4" in step
        assert "--group ${{ matrix.group }}" in step
        assert "--splitting-algorithm least_duration" in step
    assert "--cov" not in regression
    assert "--cov=superglm" in coverage
    assert "--cov-branch" in coverage
    assert "--cov-report=" in coverage
    assert "COVERAGE_FILE: .coverage.${{ matrix.group }}" in coverage
    upload = next(
        step
        for step in run_steps
        if "actions/upload-artifact@" in step and "name: coverage-Linux-py312-" in step
    )
    assert "if: github.event_name == 'push' && matrix.runtime.python-version == '3.12'" in upload
    assert "name: coverage-Linux-py312-${{ matrix.group }}" in upload
    assert "path: .coverage.${{ matrix.group }}" in upload
    assert "if-no-files-found: error" in upload
    assert "include-hidden-files: true" in upload

    assert "if: github.event_name == 'push'" in coverage_job
    assert "needs: test-compatibility" in coverage_job
    assert "actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c" in coverage_job
    assert "pattern: coverage-Linux-py312-*" in coverage_job
    assert "uv sync --locked --python 3.12 --extra dev" in coverage_job
    assert "merge-multiple: true" in coverage_job
    assert "uv run coverage combine coverage-data" in coverage_job
    assert "uv run coverage xml -o coverage.xml" in coverage_job
    # Pinned to a full commit; which commit is Dependabot's to move.
    assert re.search(r"codecov/codecov-action@[0-9a-f]{40}\s", coverage_job)

    assert workflow.count("uv run ruff check src/ tests/") == 1
    assert workflow.count("uv run ruff format --check src/ tests/") == 1


def test_dev_ci_keeps_auxiliary_checks_without_duplicating_the_regression_suite():
    workflow = _read(".github/workflows/dev-ci.yml")

    assert "pull_request:" in workflow
    assert "\n  push:\n" not in workflow
    assert "workflow_dispatch:" not in workflow
    assert "  quick-check:" not in workflow
    assert "  py314-full:" not in workflow
    for job in ("quality", "docs", "frontend", "browser", "type-check"):
        assert f"  {job}:" in workflow

    full_suite_jobs = [
        (path, name)
        for path in (".github/workflows/ci.yml", ".github/workflows/dev-ci.yml")
        for name, block in _jobs(_read(path)).items()
        # The full suite runs as `pytest tests/` or through the shared suite runner.
        if re.search(r"pytest\s+tests/(?:\s|$)|scripts/run_test_suite\.py", block)
    ]
    assert full_suite_jobs == [(".github/workflows/ci.yml", "test-compatibility")]


def test_dev_ci_keeps_browser_and_non_test_checks_independent():
    workflow = _read(".github/workflows/dev-ci.yml")

    quality_job = workflow.split("  quality:", maxsplit=1)[1].split("  docs:", maxsplit=1)[0]
    docs_job = workflow.split("  docs:", maxsplit=1)[1].split("  frontend:", maxsplit=1)[0]
    frontend_job = workflow.split("  frontend:", maxsplit=1)[1].split("  browser:", maxsplit=1)[0]
    browser_job = _jobs(workflow)["browser"]

    assert "ruff check src/ tests/" in quality_job
    assert "ruff format --check src/ tests/" in quality_job
    assert "sphinx-build -b html -n -W --keep-going" in docs_job
    assert "npm run check:frontend" in frontend_job
    assert "playwright install --with-deps chromium" in browser_job
    assert "pytest tests/test_editor_browser.py" in browser_job
    assert "pytest tests/editor/" in browser_job


def test_pre_push_pytest_uses_uv_dev_environment():
    config = _read(".pre-commit-config.yaml")
    pytest_hook = config.split("- id: pytest", maxsplit=1)[1]

    assert (
        'entry: uv run --extra dev python scripts/run_test_suite.py -m "not slow and not docs"'
        in pytest_hook
    )
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
    assert "pages: write" in deploy_job
    assert "id-token: write" in deploy_job
    assert "contents: write" not in deploy_job


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


_COOLDOWN_DAYS = 7
# Packages admitted before the cooldown passes them: a security fix younger than the
# window, with the advisories it fixes.  Each gets a fixed cutoff just after the fixing
# release's upload in pyproject's ``exclude-newer-package``; drop it (and its entry
# here) at the first lock bump after the window has passed that release.
_COOLDOWN_EXEMPTIONS = {
    "virtualenv": "21.14.2 fixes GHSA-8rjx-v5ww-45pp and GHSA-c947-3pg5-gm8q",
}
# Packages a lock change may move below the base lock, each with its reason.
_LOCK_DOWNGRADE_EXEMPTIONS: dict[str, str] = {}
_LOCK_BASELINE = "tests/fixtures/uv_lock_baseline.json"


def _span_days(span: str) -> float:
    """Days in a relative uv ``exclude-newer``: friendly (``7 days``, ``1 week``) or ISO 8601 (``P7D``)."""
    hours = {"hour": 1, "day": 24, "week": 168}
    friendly = re.fullmatch(r"(\d+)\s*(hour|day|week)s?", span.strip())
    if friendly:
        return int(friendly[1]) * hours[friendly[2]] / 24
    iso = re.fullmatch(r"P(?:(\d+)W)?(?:(\d+)D)?(?:T(\d+)H)?", span.strip())
    assert iso and any(iso.groups()), f"not a relative exclude-newer span: {span!r}"
    weeks, days, hours_part = (int(value or 0) for value in iso.groups())
    return 7 * weeks + days + hours_part / 24


def _instant(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp)


def _upload_times(lock: dict, keep) -> list[datetime]:
    """Upload instants of every locked file of the packages ``keep`` accepts by name."""
    return [
        _instant(artifact["upload-time"])
        for package in lock["package"]
        if keep(package["name"])
        for artifact in [package.get("sdist", {}), *package.get("wheels", [])]
        if "upload-time" in artifact
    ]


def test_lock_bumps_wait_out_a_minimum_release_age():
    """Every resolution skips files younger than the cooldown, and the lock was resolved under it.

    A relative ``[tool.uv] exclude-newer`` makes every ``uv lock`` ignore files
    uploaded within the window (uv docs, "Resolution", dependency cooldowns), so a
    lock bump by anyone takes only releases public that long.  uv records the span
    in ``uv.lock``, where ``uv lock --check`` (CI) fails if the setting and the
    lock disagree, and re-dates the cutoff only when it resolves again.
    Dependabot's own ``cooldown`` keeps its version updates to releases that
    resolution accepts.  Seven days: 8 of the 10 supply-chain attacks Woodruff
    surveyed were caught within a week ("We should all be using dependency
    cooldowns", 2025).  Fails on a lock resolved without the window (#428 locked
    a virtualenv published the same morning).

    A security fix younger than the window is admitted by a listed exemption
    (``_COOLDOWN_EXEMPTIONS``): a fixed RFC 3339 cutoff in ``exclude-newer-package``
    just after the fixing release, which admits it and nothing newer, never
    ``false`` or a duration, which would lift the window for that package.
    """
    uv = tomllib.loads(_read("pyproject.toml"))["tool"]["uv"]
    span = uv.get("exclude-newer")
    assert isinstance(span, str), "pyproject sets no relative uv exclude-newer"
    assert _span_days(span) >= _COOLDOWN_DAYS
    lock = tomllib.loads(_read("uv.lock"))
    options = lock.get("options", {})
    assert "exclude-newer-span" in options, "uv.lock was not resolved under the cooldown"
    assert _span_days(options["exclude-newer-span"]) == _span_days(span)

    exempt = uv.get("exclude-newer-package", {})
    assert set(exempt) == set(_COOLDOWN_EXEMPTIONS), "every cooldown exemption is listed"
    assert options.get("exclude-newer-package", {}) == exempt
    for name, cutoff in exempt.items():
        assert isinstance(cutoff, str) and re.fullmatch(
            r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", cutoff
        ), f"{name}: an exemption is a fixed cutoff, not {cutoff!r}"
        uploads = _upload_times(lock, lambda package, name=name: package == name)
        assert uploads and max(uploads) <= _instant(cutoff), f"{name} is locked past its exemption"
        # A file the window admitted is newer than the exemption's cutoff only once
        # the window has passed the exempted release, whose cutoff then only refuses
        # later ones (a security fix among them): drop it at that lock bump.
        others = _upload_times(lock, lambda package: package not in exempt)
        assert max(others) <= _instant(cutoff), (
            f"the cooldown has passed {name}'s exempted release: drop its exemption"
        )

    dependabot = _read(".github/dependabot.yml")
    uv_config = dependabot.split('package-ecosystem: "uv"', maxsplit=1)[1].split(
        "package-ecosystem:", maxsplit=1
    )[0]
    cooldown = re.search(r"cooldown:\s*\n\s+default-days:\s*(\d+)", uv_config)
    assert cooldown, "the uv updates set no Dependabot cooldown"
    assert int(cooldown[1]) >= _span_days(span)


def _locked_versions(text: str) -> dict[str, list[str]]:
    """Each locked package's versions (one per resolution fork), oldest first."""
    versions: dict[str, list[str]] = {}
    for package in tomllib.loads(text).get("package", []):
        versions.setdefault(package["name"], []).append(package["version"])
    return {name: sorted(found, key=Version) for name, found in versions.items()}


def _git(*args: str) -> str | None:
    try:
        done = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=60, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout if done.returncode == 0 else None


def _base_locks() -> list[tuple[str, dict[str, list[str]]]]:
    """The locks this tree's ``uv.lock`` must not fall below, each with its source.

    Always the recorded baseline (master's lock when it was last refreshed), which a
    shallow clone can read.  Then the lock this tree's changes start from:
    ``SUPERGLM_LOCK_BASE`` (a git ref; the security workflow fetches a pull
    request's base branch), else the merge base with ``origin/master`` when the
    history is present.  A named base that cannot be read fails, never passes.
    """
    recorded = json.loads(_read(_LOCK_BASELINE))
    bases = [(_LOCK_BASELINE, {name: sorted(v, key=Version) for name, v in recorded.items()})]
    named = os.environ.get("SUPERGLM_LOCK_BASE")
    ref = named or (_git("merge-base", "HEAD", "origin/master") or "").strip()
    if ref:
        text = _git("show", f"{ref}:uv.lock")
        assert text is not None or not named, f"SUPERGLM_LOCK_BASE={named!r} has no uv.lock"
        if text is not None:
            bases.append((f"uv.lock at {ref}", _locked_versions(text)))
    return bases


def _moved_below(base: dict[str, list[str]], lock: dict[str, list[str]]) -> dict[str, str]:
    """Packages whose newest or oldest locked version is older than the base's."""
    return {
        name: f"{base[name]} -> {lock[name]}"
        for name in base.keys() & lock.keys()
        if Version(lock[name][-1]) < Version(base[name][-1])
        or Version(lock[name][0]) < Version(base[name][0])
    }


def test_a_lock_change_never_moves_a_package_below_the_base_lock():
    """No package goes back to an older version than the base lock without a listed reason.

    A re-resolution can move a package down as readily as up: #445's first
    cooldown re-lock took virtualenv from 21.14.2 back to 21.12.1, reopening
    GHSA-8rjx-v5ww-45pp and GHSA-c947-3pg5-gm8q, which pip-audit could not see
    (published as repository advisories that morning, in neither OSV nor
    PyPI's advisory data).  Compared with the recorded baseline and, where git
    can read it, the lock the change starts from (``_base_locks``).  A package
    may go down only when ``_LOCK_DOWNGRADE_EXEMPTIONS`` names it with a reason.
    Refresh the baseline at a lock bump with ``_record_lock_baseline()``.
    """
    lock = _locked_versions(_read("uv.lock"))
    for source, base in _base_locks():
        moved = {
            name: change
            for name, change in _moved_below(base, lock).items()
            if name not in _LOCK_DOWNGRADE_EXEMPTIONS
        }
        assert not moved, f"uv.lock moves packages below {source}: {moved}"


def _record_lock_baseline() -> None:
    """Write ``uv.lock``'s versions as the recorded baseline (a maintenance helper)."""
    versions = _locked_versions(_read("uv.lock"))
    (ROOT / _LOCK_BASELINE).write_text(json.dumps(versions, indent=1, sort_keys=True) + "\n")


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
