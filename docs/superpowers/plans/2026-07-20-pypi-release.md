# PyPI release implementation plan

**Goal:** Ship a reproducible `superglm` 0.12.0 wheel/sdist through Trusted
Publishing and create the GitHub Release only after PyPI succeeds.

## Task 1: Lock the package and workflow contracts with tests

**Files:** `tests/test_release_packaging.py`,
`tests/test_supply_chain_governance.py`

1. Add failing metadata tests for core editor dependencies, removed compatibility
   extras, public URLs, license metadata, and a universal wheel.
2. Add failing workflow tests for the tag trigger, version guard, master ancestry
   guard, artifact handoff, least-privilege OIDC publication, and release ordering.
3. Run the two focused test files and confirm the new assertions fail for the
   missing implementation.

## Task 2: Make package metadata ready for PyPI

**Files:** `pyproject.toml`, `uv.lock`, `README.md`,
`docs/getting-started/installation.md`

1. Move FastAPI and Uvicorn to runtime dependencies; remove `editor` and `all`;
   leave `ipykernel` only in the docs group.
2. Add PyPI classifiers, keywords, project links, SPDX license metadata, and
   explicit sdist contents.
3. Regenerate the lockfile with uv.
4. Update installation examples and make the README logo PyPI-safe.
5. Run the focused metadata tests.

## Task 3: Implement the release workflow

**Files:** `.github/workflows/release.yml`,
`tests/test_supply_chain_governance.py`

1. Resolve and pin official action commit SHAs.
2. Build and validate exactly one wheel and one sdist on `v*.*.*` tag pushes.
3. Verify tag/version agreement and master ancestry before building.
4. Pass the artifacts to an OIDC-only `pypi` environment job.
5. Create generated GitHub release notes and attach the same artifacts only
   after publication succeeds.
6. Run workflow contract tests and a static Actions/security audit.

## Task 4: Validate actual release artifacts

**Files:** `tests/test_release_packaging.py`

1. Build from a clean tree.
2. Run Twine and `check-wheel-contents` checks.
3. Inspect wheel/sdist file lists and confirm excluded repository content is
   absent while editor assets are present.
4. Install the wheel in an isolated environment and smoke-test `superglm` plus
   the editor imports.
5. Run Ruff, ty, lock verification, dependency verification, focused tests, and
   the repository smoke test.

## Task 5: Publish the stacked change for review

1. Review the complete diff against `codex/post156-review-fixes`.
2. Commit in small conceptual groups and push `codex/pypi-release`.
3. Open a draft PR targeting `codex/post156-review-fixes` and report the one
   remaining manual action: ensure the GitHub `pypi` environment is protected
   before pushing `v0.12.0` after the stack reaches master.
