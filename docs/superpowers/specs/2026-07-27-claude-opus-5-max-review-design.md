# Claude Opus 5 Max PR Review Design

## Status

Draft for written review. The model and effort choice was approved on 2026-07-27.

## Context

The existing `Claude PR Review` workflow is manually triggered by an `@claude`
mention. It does not currently select a model or effort level, so the latest
observed run inherited Claude Code's default and used `claude-sonnet-5`.

Claude Code v2.1.219 and later supports the explicit model name
`claude-opus-5`. Opus 5 supports `max` effort, and `max` must be selected for
each session through `--effort max` or the corresponding environment variable.
The action currently installs Claude Code v2.1.220 or later, so both launch
arguments are supported.

## Requirements

- Every invocation of the existing Claude review job must request
  `claude-opus-5`.
- Every invocation must request `max` effort.
- The workflow must continue to run only when a PR comment or review contains
  `@claude`.
- Existing reviewer-only tool restrictions, authentication, permissions, and
  comment behavior must remain unchanged.
- The change must reach `master` through a dedicated pull request because
  comment-triggered workflows execute the workflow definition from the default
  branch.

## Design

Add two startup arguments to the existing `claude_args` block in
`.github/workflows/claude.yml`:

```yaml
--model "claude-opus-5"
--effort "max"
```

The full model name is used instead of the moving `opus` alias, making the
selected model generation explicit. The effort is passed as a launch argument
because `max` is session-scoped and is not accepted as a persistent
`effortLevel` setting.

No new workflow or job will be introduced. The existing `@claude` trigger,
read-only repository access, denied mutation tools, and PR-comment permissions
will remain as they are.

## Validation and Rollout

1. Parse every workflow YAML file to catch syntax errors.
2. Confirm the branch diff changes only the design document, implementation
   plan, and the two Claude launch arguments.
3. Run the repository's workflow-policy and supply-chain tests.
4. Open a dedicated pull request and require the normal protected-branch checks.
5. After merge, trigger a fresh `@claude` review on PR #165.
6. Confirm the run succeeds and its initialization reports
   `claude-opus-5`; confirm the default-branch workflow contains
   `--effort "max"`.

If the OAuth account cannot use Opus 5 or max effort, the review job should fail
visibly. The response is to correct account access or revert the two arguments,
not to silently fall back to Sonnet.

## Accepted Trade-offs

Opus 5 at max effort will take longer and consume more usage than Sonnet 5.
That cost is intentional: review quality is the priority for every manual
Claude review.

## Non-goals

- Adding another CI task or automatic PR review.
- Changing workflow permissions or the credential-helper workaround.
- Pinning the dynamically installed Claude Code executable.
- Modifying or rebasing PR #165.
