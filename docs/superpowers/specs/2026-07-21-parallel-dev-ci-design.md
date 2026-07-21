# Parallel Developer CI Design

## Goal

Reduce healthy developer-branch CI wall time from roughly ten minutes to approximately
three or four minutes without dropping tests, weakening failures, or changing master CI.

## Current bottleneck

The `py314-full` job runs the complete non-browser suite serially and currently spends
about ten minutes in pytest. The `quick-check` job independently repeats the non-slow
tests and serialises lint, frontend, browser, and documentation checks.

## Design

Remove `quick-check` and redistribute its responsibilities into independent jobs:

- `quality`: Ruff lint and formatting checks;
- `docs`: strict documentation build;
- `frontend`: frontend dependency and module checks;
- `browser`: legacy and workspace browser tests;
- `type-check`: the existing advisory Astral ty check;
- `pytest-3.14`: four duration-balanced matrix shards covering the complete non-browser
  suite exactly once.

The pytest shards use Python 3.14 and retain PyArrow so optional dataframe coverage is
not lost. Browser tests remain excluded from the shards because they have their own
prepared Chromium job. Every job starts concurrently, and existing workflow concurrency
cancellation remains enabled.

The master workflow remains unchanged: it continues to run the complete suite across
Python 3.10 through 3.14 after merge.

## Sharding contract

Use a maintained pytest duration record and a small development-only pytest sharding
plugin. Each matrix member receives a stable shard number and the same total shard count.
The plugin must guarantee that the union of the four selections contains every collected
non-browser test exactly once.

Unknown tests must still be assigned to a shard. Updating duration data may rebalance
jobs but must never affect test selection or test semantics.

## Failure behaviour

Each pytest shard retains `--maxfail=1`. Matrix failures remain independently visible so
the failing shard and test are obvious. A failure in quality, docs, frontend, browser,
type checking, or any pytest shard makes that check visibly fail, except that type
checking retains its existing advisory `continue-on-error` status.

## Validation

Before pushing:

1. Validate the workflow YAML and dependency lock.
2. Prove shard collection is exhaustive and disjoint.
3. Run all four shard commands locally and confirm their union matches ordinary
   `pytest --collect-only` for non-browser tests.
4. Run the independent quality, frontend, browser, and docs commands where locally
   available.
5. Inspect the GitHub Actions run and compare its wall time and test counts with the
   current ten-minute baseline.

## Non-goals

- No changes to test bodies, markers, tolerances, fixtures, or numerical algorithms.
- No sharding of master CI in this change.
- No use of in-process parallel pytest workers, avoiding contention between numerical
  and performance-sensitive tests.
