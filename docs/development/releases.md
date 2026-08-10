# Releases

SuperGLM uses deliberate `0.x.y` releases. Feature and fix pull requests never
touch the version files (`pyproject.toml`'s version, `superglm.__version__`,
`uv.lock`'s own pin) and never carry version-record commits. Each pull request
declares an impact, and that declaration is advice to the next release, never
a version change.

The authoritative machine-operational policy is
`.codex/agents/release_manager.toml`. It defines classification, evidence,
authority, failure recovery, and publication checks. This page explains how to
invoke that policy; it does not replace it.

## Impact summary

- `release:none`: no independent packaged-runtime or public-contract impact.
- `release:patch`: a compatible and limited fix, stability, performance,
  memory, packaging, or implementation improvement.
- `release:minor`: a material capability, behavior, numerical semantics,
  compatibility, dependency-floor, or migration change.

The code diff is authoritative. Impact declarations and labels are evidence,
and ambiguous materiality blocks preparation for a human decision. None of the
three changes any version field: the exact version is chosen at release time.

## One release, one act

A release is one deliberate act after merging: a bump-only pull request whose
single commit's message is the consolidated changelog since the previous
release, rebase-merged under the master linear-history rule, then the tag
`vX.Y.Z` bound to that rebased bump commit, then `release.yml` publishing
through PyPI Trusted Publishing. The version file, the tag, and PyPI move
together and can never disagree. Tags remain release-only, and a merge never
authorizes a tag or publication.

The one-act shape means the release assessor always works from merged history:
the impact of everything since the last published tag is reconstructed from
the `release-tag..master-head` diff, taking the highest declared-and-verified
impact. The previous convention argued the opposite — that a catch-up release
"is the recovery, not an alternative route", precisely because reconstructing
materiality from merged history costs the reviewer the original diff. That
position is reversed here, deliberately: the reconstruction cost is accepted
in exchange for never minting phantom versions on master, and it is mitigated
by the advisory impact declaration every pull request still makes at review
time, when the diff is in front of the reviewer.

## Version history notes

`0.16.2` was never released. It was prepared on the branch that became PR #174
and merged to master, but no tag was pushed, so it existed only as an
unpublished candidate. PR #176 then declared `release:minor` and bumped master
to `0.17.0`, which subsumes it: a minor release carries everything the skipped
patch would have. The gap between `0.16.1` on PyPI and `0.17.0` on master is
therefore expected, and `0.16.2` should never be published.

This is the ordinary consequence of merging without tagging. A merged
release-bearing pull request leaves master carrying a version that is not on
PyPI, and the next release-bearing pull request either waits for it to publish
or, as here, supersedes it.

`0.19.0` was prepared as a standalone release pull request, PR #240, which is
not the shape this page describes. PR #235 added
`OrderedCategorical(specials=...)`, and PR #238 changed how a declared level's
identity is matched against the spelling in the column. Both are material, and
neither declared an impact or carried a version bump, so master reached three
publishable changes while still reading `0.18.0`.

`0.22.0` through `0.24.0` were recorded on master under the former per-PR
convention and never published. They remain as changelog commits — the version
history their messages carry is real, but no tag or PyPI release exists for
any of them. The next release consolidates everything since the last published
tag; the numbers themselves are skipped, exactly as `0.16.2` was.

## Invoke the specialist

The release specialist is a spawned Codex subagent. It does not replace the
current agent and never starts automatically.

Assessment of the unreleased changes on master is read-only:

```text
Use the release_manager agent to assess the unreleased changes as a release candidate.
```

The assessment is bound to the latest published release and the exact
`origin/master` head SHA; its authoritative diff is release-tag to master
head. After approving the exact version and that SHA-bound assessment:

```text
Use the release_manager agent to prepare the approved 0.x.y.
```

The specialist updates versions through `scripts/bump_version.py`, refreshes
`uv.lock`, writes the single bump commit whose message is the consolidated
changelog, and opens the bump-only pull request. It merges nothing and holds
no tag or publication authority.

After the bump-only pull request is reviewed, rebase-merged, and green,
publication requires a new exact instruction:

```text
Use the release_manager agent to publish v0.x.y and monitor PyPI deployment.
```

The specialist pushes the approved annotated tag at the rebased bump commit on
master. The existing release workflow builds and verifies the wheel and sdist,
publishes them through PyPI Trusted Publishing, then creates the GitHub Release.
Direct uploads, reused versions, moved tags, and inferred publication authority
are forbidden.

## Failure handling

An assessment expires as soon as `origin/master` moves. Ambiguous impact,
partial publication, uncertain provenance, or an existing tag blocks
automation and requires a human decision. Published corrections use a new
patch version.
