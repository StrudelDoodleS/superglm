# Release-manager policy design

## Goal

Give SuperGLM a deliberate, reproducible pre-1.0 versioning process that a
project-scoped Codex specialist can assess and execute without taking control
of ordinary development sessions. The process must classify the complete code
impact since the latest PyPI release, prepare a dedicated release pull request,
and publish only after explicit approval of an exact version and commit.

The existing tag-triggered PyPI Trusted Publishing workflow remains the only
publisher. The agent never uploads distributions directly and never receives a
PyPI token.

## Scope and non-goals

This design covers versions of the form `0.x.y` only. It does not define when
SuperGLM is ready for `1.0.0`, nor does it introduce prereleases, post-releases,
nightly packages, or publication after every merge. Those require separate
policy decisions.

Normal feature, fix, documentation, and infrastructure pull requests do not
edit package-version files. They declare their expected release impact, while
a dedicated release pull request performs the single version change for all
work included since the preceding tag.

## Authoritative policy and Codex integration

The project-scoped custom agent lives at:

```text
.codex/agents/release_manager.toml
```

Its `developer_instructions` contain the complete machine-operational policy
defined here. The custom-agent file is the authoritative instruction source for
the running release specialist. Human documentation explains how to invoke the
process and links to that file rather than maintaining a second independent
classification policy.

The custom-agent file defines `name`, `description`, and
`developer_instructions`. It deliberately omits `model` and
`model_reasoning_effort`, allowing the spawned specialist to inherit the active
Codex session's selected model and reasoning configuration. It also avoids
overriding built-in agent names.

Repository `AGENTS.md` instructions map explicit release requests to the
specialist. They must also state that proactive delegation, an ordinary merge
request, and phrases such as “finish”, “ship”, or “deploy” do not grant release
authority. The current agent remains the orchestrator, spawns the specialist,
waits for its structured result, and presents that result to the user.

Supported user-facing invocations are natural-language requests such as:

```text
Use the release_manager agent to assess changes since the latest PyPI release.
Use the release_manager agent to prepare the approved 0.13.0 release PR.
Use the release_manager agent to publish v0.13.0 and monitor PyPI deployment.
```

There is no automatic invocation and no replacement of the current agent.

## Release-impact declarations

Every ordinary pull request declares exactly one expected impact in its body:

- `release:none`
- `release:patch`
- `release:minor`

The declaration includes a short rationale and any compatibility or migration
notes. Matching GitHub labels may be applied for filtering and summaries, but
labels and pull-request prose are advisory evidence. The base-to-head code diff
is authoritative, and the release specialist must challenge an unsupported
declaration rather than copy it.

`release:none` means that a change does not independently justify a new
release. It does not exclude the commit from a later source snapshot.

## Pre-1.0 version-impact policy

The highest impact among all changes since the latest valid release tag
determines the next version.

| Classification | Version result | Meaning |
| --- | --- | --- |
| `none` | no independent bump | No packaged-runtime, public-contract, compatibility, or user-visible numerical impact |
| `patch` | `0.a.b` to `0.a.(b+1)` | Compatible, limited, corrective, or implementation-quality improvement |
| `minor` | `0.a.b` to `0.(a+1).0` | Material new capability, behaviour, numerical semantics, or compatibility change |
| `needs-human-decision` | publication blocked | Evidence is incomplete, contradictory, or not safely classifiable |

Diff size and engineering effort do not determine impact. A large internal
refactor can be a patch, while a one-line public default change can be minor.
Security or defect severity determines release urgency, not compatibility
classification.

### `none`

Use `none` only when the change has no independent packaged-runtime or public
behaviour consequence, including:

- documentation-only changes;
- tests and test fixtures that do not alter shipped package data;
- CI, repository governance, and contributor-tooling changes;
- benchmark harnesses and recorded benchmark data not shipped to users;
- formatting, comments, and mechanically equivalent metadata cleanup;
- development-only dependency or lock changes.

If production Python, packaged assets, runtime dependencies, public metadata,
or supported environments change, `none` requires specific evidence and must
not be assumed.

### `patch`

Use `patch` for compatible and limited changes, including:

- backwards-compatible bug fixes;
- localized correctness fixes that restore documented or authoritative theory
  without materially changing ordinary valid workloads;
- speed or memory improvements that preserve public and numerical behaviour;
- internal refactors preserving APIs, outputs, accepted inputs, defaults,
  warnings, persistence, and supported environments;
- small additive conveniences requiring no user migration;
- packaging corrections and compatible runtime-dependency updates;
- precision or stability improvements whose differences remain within the
  library's established numerical contract.

A patch may be important or urgent. Patch does not mean trivial.

### `minor`

Use `minor` for material capability, behaviour, or compatibility changes,
including:

- a new family, link, loss, objective, solver, fitting mode, or substantial
  public workflow;
- material support for a new input or dataframe ecosystem;
- new public modules or APIs that expand the library's conceptual surface;
- removal, rename, or incompatible signature change of public behaviour;
- changed defaults, warnings relied upon by users, convergence semantics, or
  established fitted-result meaning;
- correctness changes that materially alter results for ordinary valid
  workloads, even when the previous implementation was wrong;
- incompatible result, trace, export, serialization, or persistence schemas;
- a new mandatory runtime dependency;
- raising the minimum Python or runtime-dependency floor, or dropping a
  supported environment;
- any change requiring users to inspect, migrate, or revalidate existing code
  or fitted models.

### Materiality assessment

The specialist must cite evidence against these questions:

1. Must existing users change code, configuration, stored artifacts, or
   validation expectations?
2. Does the change introduce a new concept that users intentionally select or
   learn?
3. Do ordinary valid inputs produce meaningfully different coefficients,
   predictions, objectives, convergence decisions, warnings, or schemas?
4. Does the change alter supported Python versions, dependency compatibility,
   accepted data containers, or required installations?
5. Does published documentation promise the old behaviour?

No universal numeric percentage defines materiality. The specialist uses the
library's documented contract, existing test tolerances, breadth of affected
workloads, migration burden, and domain significance. If those signals
conflict, it returns `needs-human-decision` with the competing cases instead of
silently choosing a version.

## Assessment mode

Assessment mode is strictly read-only. It performs the following sequence:

1. Fetch `origin/master`, release tags, and current GitHub metadata without
   changing the working tree.
2. Query PyPI for the currently published SuperGLM version and files.
3. Select the highest valid `v0.x.y` tag that is an ancestor of
   `origin/master`, then record its tag object, peeled commit, and version.
4. Record the exact assessed `origin/master` SHA.
5. Refuse to proceed if the tag, PyPI version, `pyproject.toml`, or
   `superglm.__version__` establish inconsistent release history.
6. Inspect the complete tagged-commit-to-head diff.
7. Separate production source, public API, runtime dependencies, packaging,
   tests, CI, benchmarks, generated data, and documentation.
8. Resolve merged pull requests and read their declared impacts, descriptions,
   migration notes, reviews, and relevant issue links.
9. Inspect public signatures and exports, accepted input types, defaults,
   warnings, result and persistence schemas, numerical semantics, supported
   environments, and runtime dependency changes directly from the diff.
10. Classify each material change independently, aggregate the maximum impact,
    and report any disputed declaration.
11. Re-fetch `origin/master` before reporting. If it moved, discard the
    assessment and restart against the new head.

The structured report contains:

```text
Assessment ID: <base-tag>..<head-sha>
Latest release: <tag and PyPI version>
Base SHA: <peeled release commit>
Assessed master SHA: <exact full SHA>
Recommendation: none | patch | minor | needs-human-decision
Recommended version: <0.x.y or none>
Release urgency: routine | recommended | urgent
Included PRs: <number, title, merge SHA, declared impact>
Per-change classification: <impact, evidence, affected contract>
Compatibility and migration notes: <explicit list>
Uncertainties: <explicit list or none>
Exact next permitted action: <assessment, preparation, or blocked>
```

Urgency is separate from version impact. The report makes no file, branch,
pull-request, label, comment, tag, release, or package-index changes.

An assessment is bound to its exact head SHA and expires immediately when
`origin/master` moves.

## Preparation mode

Preparation requires an explicit user-approved version and a non-stale
assessment. General prior approval is insufficient. The current agent must
pass the exact approved version and assessment ID to the specialist.

The specialist then:

1. Re-fetches and verifies the assessment head is unchanged.
2. Confirms the requested version equals the policy-derived version.
3. Creates an isolated `.worktrees/` release worktree and dedicated branch from
   the assessed master SHA.
4. Uses a narrow version-bump script to update `pyproject.toml` and
   `superglm.__version__` together, then refreshes lock metadata with uv.
5. Refuses unchanged versions, downgrades, malformed versions, `1.x` versions,
   and any version inconsistent with the approved impact.
6. Generates a release summary listing every included pull request, impact,
   compatibility note, and any required user action.
7. Runs version-consistency, packaging, artifact, lint, lock, and focused
   release tests.
8. Commits and pushes only the release branch and opens a dedicated release
   pull request.
9. Reports the branch, commit, pull request, validation evidence, and exact
   commit range.

The specialist never merges the release pull request. It must pass ordinary
review and required CI.

## Publication mode

Publication requires a new explicit instruction naming the exact tag, for
example `publish v0.13.0`. Approval to assess, prepare, merge, or generally
“release” does not imply permission to push a tag.

Before tagging, the specialist:

1. Fetches master, tags, the merged release pull request, its required checks,
   and PyPI state.
2. Resolves the exact approved release-PR merge commit rather than assuming the
   current tip of master.
3. Verifies both version sources and lock metadata agree with the requested
   tag.
4. Verifies the release commit is descended from the assessed SHA, belongs to
   master, and contains no unassessed production changes.
5. Verifies all required release-PR and master checks succeeded.
6. Verifies neither the Git tag nor the PyPI version already exists.
7. Creates one annotated tag at that exact commit and pushes only that tag.

The existing `.github/workflows/release.yml` remains authoritative after the
tag push. It builds and validates one universal wheel and one source
distribution, publishes them through OIDC, and creates the GitHub Release only
after PyPI succeeds.

The specialist monitors the workflow through completion and then independently
checks that:

- every release job succeeded;
- PyPI reports the exact version;
- both wheel and source distribution exist;
- their filenames and hashes match the workflow evidence;
- the GitHub Release points to the correct tag and contains the same artifacts.

## State transitions and authority

| State | Permitted next action | Required authority |
| --- | --- | --- |
| Unassessed | assess | Explicit request to use `release_manager` for assessment |
| Assessed | prepare or reassess | Explicit approval of version and assessment ID |
| Release PR open | review and CI only | No publication authority |
| Release PR merged | publish | Explicit instruction naming the exact `v0.x.y` tag |
| Tag pushed | monitor and verify | Already granted by the exact publish instruction |
| Published | report | No further mutation |
| Stale or ambiguous | reassess or ask user | No preparation or publication permitted |

The root/current agent retains orchestration throughout. The specialist cannot
broaden its authority from one state to another.

## Failure and recovery policy

- Never move, overwrite, or silently recreate a published tag.
- Never overwrite or reuse a PyPI version.
- Never upload manually to bypass Trusted Publishing or a failed workflow.
- Never weaken release validation to make publication pass.
- If master moves before the release PR is prepared, reassess.
- If unassessed production changes enter the proposed release commit, reassess.
- If validation fails before a tag is pushed, fix through the ordinary pull
  request process and reassess the resulting commit.
- If a tag-triggered workflow fails, inspect whether PyPI accepted zero, one,
  or both artifacts before proposing recovery.
- If publication is partial, provenance is uncertain, or a tag already exists,
  stop with `needs-human-decision`. Do not delete remote state automatically.
- Corrections to a published release use a new patch version.

## Repository changes

Implementation adds or updates:

- `.codex/agents/release_manager.toml` for the custom specialist and complete
  operational policy;
- `AGENTS.md` for durable invocation, impact-declaration, and authority rules;
- `.github/PULL_REQUEST_TEMPLATE.md` for one release-impact declaration and
  compatibility rationale per pull request;
- `docs/development/releases.md` for concise human invocation and release-flow
  documentation;
- `scripts/bump_version.py` for atomic validation and updates of the two source
  version fields, followed by explicit uv lock refresh in the release process;
- focused policy, configuration, version-bump, and release-workflow tests.

The implementation does not add an automatic publisher on master pushes and
does not replace the existing trusted-publishing workflow.

## Validation

Focused tests must establish that:

- the custom-agent TOML parses and defines the required fields;
- it does not pin a model or reasoning effort;
- ordinary development guidance cannot invoke it implicitly;
- assessment, preparation, and publication have distinct approval gates;
- the policy examples classify deterministically;
- ambiguous cases block rather than defaulting to a release;
- the version-bump tool accepts only the approved next `0.x.y` version and
  updates both source fields consistently;
- downgrades, unchanged versions, malformed versions, `1.x` versions, and
  impact-inconsistent versions fail without partial edits;
- the pull-request template exposes exactly one release-impact declaration;
- existing tag trigger, version agreement, master ancestry, least-privilege
  OIDC, artifact handoff, and GitHub-release ordering remain intact.

Repository lint, lock checks, smoke tests, the complete test suite, wheel and
sdist builds, archive inspection, and installed-wheel smoke tests remain final
implementation gates.
