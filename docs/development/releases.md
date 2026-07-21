# Releases

SuperGLM uses deliberate `0.x.y` releases. Ordinary pull requests declare an
expected impact but do not change package versions. A dedicated release pull
request applies the highest impact accumulated since the latest PyPI tag.

The authoritative machine-operational policy is
`.codex/agents/release_manager.toml`. It defines classification, evidence,
authority, failure recovery, and publication checks. This page explains how to
invoke that policy; it does not replace it.

## Impact summary

- `release:none`: no independent packaged-runtime or public-contract impact.
- `release:patch`: compatible and limited fixes, stability, performance,
  memory, packaging, or implementation improvements.
- `release:minor`: material capability, behaviour, numerical semantics,
  compatibility, dependency-floor, or migration change.

The complete tagged-release-to-master diff is authoritative. The highest
impact wins, and ambiguous materiality blocks publication for a human decision.

## Invoke the specialist

The release specialist is a spawned Codex subagent. It does not replace the
current agent and never starts automatically.

Assessment is read-only:

```text
Use the release_manager agent to assess changes since the latest PyPI release.
```

After approving the exact version and SHA-bound assessment:

```text
Use the release_manager agent to prepare the approved 0.x.y release PR.
```

The specialist updates versions through `scripts/bump_version.py`, refreshes
`uv.lock`, validates the artifacts, and opens a dedicated PR. It does not merge the release PR.

After that PR is reviewed, merged, and green, publication requires a new exact
instruction:

```text
Use the release_manager agent to publish v0.x.y and monitor PyPI deployment.
```

The specialist pushes the approved annotated tag. The existing release
workflow builds and verifies the wheel and sdist, publishes them through PyPI
Trusted Publishing, then creates the GitHub Release. Direct uploads, reused
versions, moved tags, and inferred publication authority are forbidden.

## Failure handling

A changed master invalidates the assessment. Invalid version state, ambiguous
impact, partial publication, uncertain provenance, or an existing tag blocks
automation and requires a human decision. Published corrections use a new
patch version.
