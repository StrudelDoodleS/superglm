# Repository Guidelines

## Commands and isolation

Use `rg` or `rg --files` for searches. Create isolated feature work under the
ignored `.worktrees/` directory and preserve unrelated user changes.

Install the development environment with `uv sync --python 3.13 --extra dev`.
The ordinary checks are:

- `uv run pytest tests/ -q`
- `uv run pytest tests/ -q -m "not slow"`
- `uv run ruff check src/ tests/`
- `uv run ruff format --check src/ tests/`
- `uv lock --check`
- `uv pip check`
- `uv run python run_test.py`

Three suites are anchored to the real freMTPL2 book and skip without a local
copy of it: `tests/test_realdata_parity.py`,
`tests/test_screening_guide_numbers.py` and
`tests/test_mixed_interaction_screening.py`. Fetch the data with
`uv run python scripts/fetch_fremtpl.py --dest data/`, and set
`SUPERGLM_REQUIRE_DATA=1` so a dataset skip fails instead of passing silently.
The `Real data` workflow does both, which is what stops those suites reporting
a green tick for tests that never ran.

Use `apply_patch` for repository file edits. Do not discard dirty-worktree
changes or use destructive git commands.

## Project structure and style

Package code lives under `src/superglm/`; tests under `tests/`; benchmarks and
exploratory work remain outside production paths. Target Python 3.12+ and the
existing Ruff configuration. Preserve mathematical names where they make the
numerical implementation clearer. Public APIs are exported through
`src/superglm/__init__.py`.

New solver, REML, family, input-boundary, or feature behaviour requires focused
regression tests. Performance-sensitive work must compare complete-fit timing,
memory, numerical outputs, and actual backend dispatch against the relevant
baseline.

## Numerical test policy

- Boundary tests assert mathematical or certified invariants such as rank,
  subspace, residual, reconstruction, prediction, or backward error—not the
  sign or magnitude of BLAS/LAPACK roundoff.
- Numerical tolerances derive from dimensions, dtype epsilon, norms, and
  conditioning or error bounds.
- Coefficient-forward accuracy uses well-conditioned fixtures. Near-rank or
  cancellation fixtures test certification, refusal, and stable observables.
- Performance and backend-dispatch behaviour are tested separately from
  numerical correctness.
- Adversarial regressions include a mutation check or a demonstration against
  the unfixed implementation.

## Release impact and publishing

Every pull request must declare exactly one advisory impact in its body with a
rationale:

- `release:none`
- `release:patch`
- `release:minor`

The declaration is advice to the next release, never a version change. Feature
and fix pull requests do not touch `pyproject.toml`'s version,
`superglm.__version__`, or `uv.lock`'s own version pin, and do not carry
version-record commits; reviewers flag any pull request that does. The code
diff is authoritative; declarations and labels are evidence only.

A release is one deliberate act on master after merging: a single bump commit
whose message is the consolidated changelog since the previous release, tagged
`vX.Y.Z` on that same commit, then published — the version file, the tag, and
PyPI move together and can never disagree. Tags remain release-only. A merge
never authorizes a tag or publication.

Why one act instead of per-pull-request version records: the record convention
accumulated concrete unpublished versions — 0.22.0 through 0.24.0 sit on
master untagged and unpublished, and remain as changelog commits that never
shipped. The next release bumps directly from 0.24.0 to whatever ships next.

Only an explicit user request to assess, prepare, or publish a release may
spawn the project-scoped `release_manager` specialist from
`.codex/agents/release_manager.toml`. Proactive delegation and ordinary words
such as “finish”, “merge”, “ship”, or “deploy” do not authorize publication.

Use these three separate gates:

1. “Use the release_manager agent to assess the unreleased changes as a
   release candidate.” Assessment is read-only and bound to the exact base
   and head SHAs.
2. “Use the release_manager agent to prepare the approved 0.x.y.” Preparation
   requires an exact approved version and assessment ID, writes the single
   bump commit via a bump-only pull request (the diff contains only that
   commit), rebase-merged under the linear-history rule; the release tag
   binds to the rebased bump commit on master. Preparation
   does not authorize publication.
3. “Use the release_manager agent to publish v0.x.y.” Publication requires a
   new explicit instruction naming the exact tag.

The specialist does not merge feature pull requests, upload distributions
directly, move an existing tag, or bypass `.github/workflows/release.yml` and
PyPI Trusted Publishing.
