# Claude Opus 5 Max PR Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every manually triggered Claude PR review run on `claude-opus-5` at `max` effort.

**Architecture:** Pass the exact model and session-scoped effort through the existing Claude Code Action `claude_args` block. Preserve the current trigger, permissions, authentication, and reviewer-only tool boundary, then verify the configuration locally and through a real post-merge review run.

**Tech Stack:** GitHub Actions YAML, `anthropics/claude-code-action`, Claude Code CLI, pytest, PyYAML, GitHub CLI.

---

### Task 1: Select Opus 5 at max effort

**Files:**
- Modify: `.github/workflows/claude.yml:77`

- [ ] **Step 1: Verify the required launch arguments are absent**

Run:

```bash
rtk grep -F -- '--model "claude-opus-5"' .github/workflows/claude.yml
rtk grep -F -- '--effort "max"' .github/workflows/claude.yml
```

Expected: both commands report zero matches and return a non-zero status.

- [ ] **Step 2: Add the exact model and effort arguments**

Change the existing block to:

```yaml
          claude_args: |
            --model "claude-opus-5"
            --effort "max"
            --allowedTools "mcp__github_inline_comment__create_inline_comment"
            --disallowedTools "Bash,Edit,Write,MultiEdit,NotebookEdit,Agent,Task,mcp__github_file_ops__commit_files,mcp__github_file_ops__delete_files"
            --append-system-prompt "Act only as a code reviewer. Do not modify files, create commits, or push branches. Return review feedback through PR comments only."
```

- [ ] **Step 3: Verify both arguments are present exactly once**

Run:

```bash
rtk grep -n -F -- '--model "claude-opus-5"' .github/workflows/claude.yml
rtk grep -n -F -- '--effort "max"' .github/workflows/claude.yml
```

Expected: one match for each argument in the `claude_args` block.

- [ ] **Step 4: Parse every workflow and run governance tests**

Run:

```bash
rtk proxy uv run python -c 'from pathlib import Path; import yaml; [yaml.safe_load(path.read_text(encoding="utf-8")) for path in Path(".github/workflows").glob("*.yml")]'
rtk proxy uv run pytest tests/test_supply_chain_governance.py -q
rtk git diff --check
```

Expected: YAML parsing exits successfully, all 20 governance tests pass, and
`git diff --check` emits no output.

- [ ] **Step 5: Commit the workflow change**

Run:

```bash
rtk git add .github/workflows/claude.yml
rtk git commit -m "Run Claude reviews with Opus 5 max"
```

Expected: one new commit changing only `.github/workflows/claude.yml`.

### Task 2: Publish through branch protection and verify the live model

**Files:**
- No repository files change in this task.

- [ ] **Step 1: Confirm the final branch scope**

Run:

```bash
rtk git status --short --branch
rtk git diff --check origin/master...HEAD
rtk git diff --stat origin/master...HEAD
```

Expected: a clean branch containing only the approved design document,
implementation plan, and two workflow arguments.

- [ ] **Step 2: Push and open the dedicated pull request**

Run:

```bash
rtk git push -u origin chore/claude-opus-5-max
rtk gh pr create --base master --head chore/claude-opus-5-max --title "Run Claude reviews with Opus 5 max" --body "Pin every manually triggered Claude PR review to claude-opus-5 with max effort. Preserve the existing trigger and reviewer-only permission boundary. Validation: workflow YAML parse, supply-chain governance tests, and clean rebased non-slow baseline."
```

Expected: GitHub creates a non-draft pull request targeting `master`.

- [ ] **Step 3: Require protected-branch validation and independent approval**

Run:

```bash
rtk gh pr checks --watch --interval 10
rtk gh pr view --json mergeable,mergeStateStatus,reviewDecision,statusCheckRollup
```

Expected: all required checks pass, the PR is mergeable, and a distinct
authorized reviewer supplies any approval required by branch protection.

- [ ] **Step 4: Merge through the pull request**

Run:

```bash
rtk gh pr merge --squash --delete-branch
```

Expected: GitHub records the pull request as merged into `master`; no direct
push or protection bypass is used.

- [ ] **Step 5: Trigger a post-merge Claude review on PR #165**

Run:

```bash
rtk gh pr comment 165 --body "@claude review

Review the exact current head independently. Begin by stating the active model and effort level, then perform a release-safety review. Do not modify the branch."
```

Expected: the default-branch `Claude PR Review` workflow starts from the new
workflow definition.

- [ ] **Step 6: Verify the live workflow uses Opus 5 and max effort**

Run:

```bash
review_run_id=$(rtk gh run list --workflow "Claude PR Review" --event issue_comment --limit 1 --json databaseId --jq '.[0].databaseId')
rtk gh run watch "$review_run_id" --exit-status
rtk gh run view "$review_run_id" --log | rtk grep -F '"model": "claude-opus-5"'
rtk git fetch origin master
rtk git show origin/master:.github/workflows/claude.yml | rtk grep -F -- '--effort "max"'
```

Expected: the run succeeds, its initialization identifies
`claude-opus-5`, and the executed default-branch workflow contains
`--effort "max"`.
