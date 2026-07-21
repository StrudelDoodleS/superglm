# Release-bearing feature pull requests

## Goal

Keep each material SuperGLM change, its compatibility assessment, and its
pre-1.0 version bump in one reviewable pull request. A merged release-bearing
pull request becomes the exact release candidate, while tagging and PyPI
publication remain separate, explicitly authorized operations.

## Decision

SuperGLM will use release-bearing feature pull requests rather than a second,
version-only release pull request.

Every pull request continues to declare exactly one impact:

- `release:none` does not change package version fields.
- `release:patch` includes the exact next patch version.
- `release:minor` includes the exact next minor version.

The code diff remains authoritative. A declaration does not override the
release manager's assessment, and an ambiguous assessment blocks the version
change for a human decision.

PR #163 changes an established default and requires migration for callers that
want the former automatic selection penalty. It is therefore a minor release
candidate and will update SuperGLM from `0.12.0` to `0.13.0` in the same PR.

## Why this model

A dedicated release PR or change-fragment system is useful when many teams
merge concurrently and batch several changes into one release. SuperGLM is
currently better served by keeping the behavior change and version review
together. This removes a redundant post-merge review while preserving an exact
release boundary.

The cost is intentional coordination: only one release-bearing PR may advance
from a given published version at a time. A concurrent patch or minor PR must
rebase after the preceding release is published and recompute its version.
`release:none` PRs do not compete for a version.

## Pull-request contract

A release-bearing PR contains:

1. the production, test, and documentation changes;
2. one release-impact declaration and rationale;
3. explicit compatibility and migration notes;
4. the policy-derived version in `pyproject.toml`,
   `src/superglm/__init__.py`, and `uv.lock`;
5. validation evidence for the changed behavior and release artifacts.

A `release:none` PR must leave those version fields unchanged. CI and focused
policy tests protect this documented contract, while the release manager
performs the semantic assessment that cannot be inferred safely from line
counts or labels.

## Release-manager lifecycle

The project-scoped `release_manager` remains a spawned specialist. It does not
replace the current agent and has three separately authorized modes.

### Assess

The user asks the specialist to assess a specific open PR. Assessment is
read-only and is bound to:

- the latest canonical PyPI release and tag;
- the PR number;
- the exact base SHA;
- the exact head SHA.

The specialist classifies the PR diff, challenges its declared impact, checks
whether another unpublished version already occupies the base branch, and
returns the exact proposed version. Any base or head movement expires the
assessment.

### Prepare

The user explicitly approves an exact version and assessment ID. The
specialist then modifies only the assessed feature branch, using
`scripts/bump_version.py` and `uv lock` to synchronize the three version
records. It updates the release metadata, validates the branch, commits, and
pushes that same PR. It does not create a second release PR, merge the feature
PR, create a tag, or publish a package.

### Publish

After the release-bearing PR is merged and its required master checks pass,
the user must issue a new instruction naming the exact `v0.x.y` tag. The
specialist binds that tag to the reviewed PR merge commit, pushes only that
annotated tag, and monitors the existing Trusted Publishing workflow. A merge,
general approval, or prior preparation never implies publication authority.

## State and concurrency rules

- The source version on the PR base must agree with the latest published PyPI
  version before preparing a new release-bearing PR.
- If the base already contains an unpublished version bump, a second
  release-bearing PR is blocked until that candidate is published or resolved.
- If master or the PR head changes after assessment, reassess before writing.
- If another release is published first, rebase and derive a fresh version.
- The tag points to the release-bearing PR's merge commit, not an arbitrary
  later master tip.
- Later `release:none` commits may exist on master without changing the release
  candidate's tagged commit.
- Published versions and tags are never reused, moved, or overwritten.

## Repository changes

The existing release-governance surfaces will be updated, not replaced:

- `.codex/agents/release_manager.toml` will assess and prepare an open feature
  PR instead of preparing a dedicated release branch and PR.
- `AGENTS.md` will require patch/minor version bumps in the same PR and retain
  the explicit publication boundary.
- `.github/PULL_REQUEST_TEMPLATE.md` will record both impact and intended
  version.
- `docs/development/releases.md` will document the release-bearing workflow
  and concurrency rule.
- `tests/test_release_management.py` will encode the new durable policy.

The version-transition helper and the tag-triggered
`.github/workflows/release.yml` remain unchanged unless a focused test exposes
a defect.

## Validation

Implementation follows test-first policy changes. Focused tests must prove
that repository guidance, the PR template, human documentation, and custom
agent all agree on these points:

- `none` means no version bump;
- `patch` and `minor` are release-bearing;
- preparation updates the assessed feature PR rather than opening a second PR;
- assessment is base/head SHA-bound;
- publication still requires a new exact-tag instruction;
- concurrent or stale release candidates block rather than guessing.

For PR #163, the release manager must independently confirm `release:minor`
and `0.13.0`. The branch must then pass version consistency, lock, packaging,
artifact, lint, smoke, focused regression, and existing repository test gates.

## Non-goals

This change does not publish automatically on merge, introduce prereleases,
define `1.0.0`, infer release impact from commit messages, or add a general
release automation framework. It does not change SuperGLM numerical code beyond
the already reviewed behavior in PR #163.
