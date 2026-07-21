# Releases

SuperGLM uses deliberate `0.x.y` releases. A patch or minor feature pull
request carries its own version bump so the behavior, migration impact, and
release candidate are reviewed together. A `release:none` pull request leaves
the version unchanged.

The authoritative machine-operational policy is
`.codex/agents/release_manager.toml`. It defines classification, evidence,
authority, failure recovery, and publication checks. This page explains how to
invoke that policy; it does not replace it.

## Impact summary

- `release:none`: no independent packaged-runtime or public-contract impact;
  do not change the version.
- `release:patch`: a compatible and limited fix, stability, performance,
  memory, packaging, or implementation improvement; include the exact next
  patch version in that pull request.
- `release:minor`: a material capability, behavior, numerical semantics,
  compatibility, dependency-floor, or migration change; include the exact next
  minor version in that pull request.

The code diff is authoritative. Impact declarations and labels are evidence,
and ambiguous materiality blocks preparation for a human decision.

## One release-bearing pull request

Only one release-bearing pull request may advance from a published version at
a time. A concurrent patch or minor pull request must rebase after the preceding
candidate is published and recompute its version. This coordination cost keeps
each material change and its release metadata in one review.

The pull-request base version must agree with the latest PyPI release before
preparation. `release:none` work may merge without competing for a version, but
a release tag still points to the reviewed release-bearing PR's merge commit,
not an arbitrary later master tip.

## Invoke the specialist

The release specialist is a spawned Codex subagent. It does not replace the
current agent and never starts automatically.

Assessment of an exact open PR is read-only:

```text
Use the release_manager agent to assess PR #N as a release candidate.
```

After approving the exact version and base/head-bound assessment:

```text
Use the release_manager agent to prepare the approved 0.x.y on PR #N.
```

The specialist updates versions through `scripts/bump_version.py`, refreshes
`uv.lock`, validates the release candidate, and updates that existing PR. It
does not merge the feature PR or create a second release PR.

After the feature PR is reviewed, merged, and green, publication requires a
new exact instruction:

```text
Use the release_manager agent to publish v0.x.y and monitor PyPI deployment.
```

The specialist pushes the approved annotated tag at the reviewed merge commit.
The existing release workflow builds and verifies the wheel and sdist,
publishes them through PyPI Trusted Publishing, then creates the GitHub Release.
Direct uploads, reused versions, moved tags, and inferred publication authority
are forbidden.

## Failure handling

A changed PR base or head invalidates its assessment. An inconsistent base
version, an existing unpublished release candidate, ambiguous impact, partial
publication, uncertain provenance, or an existing tag blocks automation and
requires a human decision. Published corrections use a new patch version.
