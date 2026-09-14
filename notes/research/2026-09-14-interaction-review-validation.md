# Interaction archive: PR review follow-up

Codex and Claude reviewed the initial research PR tree `aa5be1d81b68` against
the performance base `b1980e31f19d`. Their findings concern the tools used for
future reruns and verification. The frozen experiment results remain unchanged.

## Corrections

- Final worker-time totals include failed searches and cases with no evaluation.
  Three regressions fail against the original runner and pass with the correction.
- Parent-normalized timeout/error receipts are written back to disk. Partial
  worker evidence is retained; `parent_finished_utc` identifies a completion time
  recorded by the parent when the worker could not finish its own receipt.
  If termination interrupts a JSON write, its original bytes are preserved in
  a file named by stage and SHA-256, and the normalized record references that
  file. Empty and truncated receipts no longer abort the parent suite.
- The audit retains interrupted attempts and their full process costs even when
  a worker leaves no usable identity or runtime evidence. Missing evidence makes
  the corresponding aggregate verification flag false and lists the affected
  workers; every identity field that is present must still match. Parent
  completion times, incomplete warning status and interrupted-write receipts
  survive into the summary. Preserved bytes are hashed and checked. A model
  artifact left by a failed fit is indexed as untrusted and cannot be selected
  or evaluated.
- Audits and plots compare all data, source-code, registry, preprocessing and
  split evidence while excluding only the two local location fields
  `source_path` and `source_registry`. Archived metadata and its original digest
  are still checked exactly against all worker records.
- `--source-root` selects both benchmark modules and the package source used
  for saved-model replay. Two fresh-process regressions demonstrate that a
  different installed checkout cannot silently supply the package instead.
- The audit reconstructs admission from the re-prepared state and proposed
  pairs, then reconstructs the arm menu from that admission and the frozen
  protocol. Dropping an arm or changing its recorded admission cannot silently
  reduce attempt counts.
- An unavailable matching additive baseline is recorded with null comparison
  and fit-ratio values and outcome `matching_additive_unavailable`. It cannot
  support a claim of established interaction gain. No selected model or no
  converged additive comparator is an explicit refusal at the audit's scope
  boundary.
- The plotter handles odd pair counts and hides unused axes.
- The audit, plotter and many-interaction benchmark refuse Python execution
  with assertions disabled. The many-interaction writer rejects nonfinite JSON
  numbers at serialization.
- The original Lean validation receipt now states that its Lake configuration
  hash predates the integrated operator build. Proof hashes, compiler output and
  exit statuses are preserved; the later receipt records the updated Lake hash.

## Validation

Focused tests cover failed-search costs, normalized fit/evaluation receipts,
optimized-Python refusal, location-only changes with content/split mutations,
missing planned arms, unavailable baselines, a one-pair plot and nonfinite JSON.
An additional 23 cases run preparation, the parent suite and its audit with
synthetic child receipts, without fitting models. They cover absent, empty,
truncated and intermediate receipts, missing evidence, incorrect identities,
tampered preserved bytes and ineligible leftover model artifacts. The audit
before this correction raises `KeyError('source')` for the absent, empty and
truncated cases; the corrected audit records their costs with incomplete
verification evidence.
The original plotter fails the one-pair test. All eight initial optimized-Python
and parent-receipt tests fail before their corrections; the six relevant
many-interaction regressions also fail before their corrections.

The updated audit was run with the frozen source at `639f499e` checked out in a
different worktree. It replays all 26 saved scores and verifies all 552 artifact
hashes, with a new check accounting for all 100 planned arms. After allowing for
the displayed archive location and the new menu-check field, its complete
measurement object equals the committed original exactly.

The updated plotter also ran against that relocated source. All 40 arrays for
the ten archived interaction surfaces match exactly, and all three saved
models reproduce their stored test predictions exactly. No model was fitted,
selected or retuned during these checks. The generated replay copies remain
local; the archived figures and measurement receipts are preserved.

The earlier report's 95 tests and the initial PR's 130 tests describe different
test selections at those checkpoints. Additional review regressions are reported
in the PR body with their final count; they do not revise the experiment record.

## Replaying a historical source tree

Both tools accept `--source-root`, `--run-root` and `--data-root`. Source and
dependency fingerprints must still match the archived protocol. Use the
environment from the frozen source tree and point the current audit script at
that tree, rather than substituting the corrected runner for the measured one:

```bash
git fetch origin 639f499eaec04967de0b7c31090276671144b99f
git worktree add --detach .worktrees/interaction-frozen 639f499e
uv sync --project .worktrees/interaction-frozen --python 3.13 --extra dev
uv run --project .worktrees/interaction-frozen python docs/research/check_broad_interaction_measurements.py \
  --source-root .worktrees/interaction-frozen \
  --run-root /path/to/frozen-20260914 \
  --data-root /path/to/interaction-datasets \
  --output /tmp/broad-replay.json
```

The original commit is retained on the pushed `research/adaptive-interactions`
branch: its head `18f21fa24ea7` is one commit after `639f499e`. The explicit fetch
requests that exact commit so it also works when the archival branch has
already been fetched with depth one. This was checked in a fresh shallow
repository: fetching the branch again left its parent unavailable, while
fetching the full commit ID retrieved it successfully.

The raw archive and pinned source tables are required. The plotter takes the
same roots and an output directory through `--output`; keep replay output
separate from the archived figures. These options relocate inputs without
waiving any content, source-version or numerical replay checks.
