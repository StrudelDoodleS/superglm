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

Use `apply_patch` for repository file edits. Do not discard dirty-worktree
changes or use destructive git commands.

## Project structure and style

Package code lives under `src/superglm/`; tests under `tests/`; benchmarks and
exploratory work remain outside production paths. Target Python 3.10+ and the
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

`release:none` leaves package version fields unchanged. `release:patch`
includes the exact next patch version in the same pull request, and
`release:minor` includes the exact next minor version in the same pull request.
The code diff is authoritative; declarations and labels are evidence only.

Only one release-bearing pull request may advance from a published version at
a time. Concurrent patch or minor pull requests must rebase after the preceding
candidate is published and recompute their version. A merge never authorizes a
tag or publication.

Only an explicit user request to assess, prepare, or publish a release may
spawn the project-scoped `release_manager` specialist from
`.codex/agents/release_manager.toml`. Proactive delegation and ordinary words
such as “finish”, “merge”, “ship”, or “deploy” do not authorize publication.

Use these three separate gates:

1. “Use the release_manager agent to assess PR #N as a release candidate.”
   Assessment is read-only and bound to the exact base and head SHAs.
2. “Use the release_manager agent to prepare the approved 0.x.y on PR #N.”
   Preparation requires an exact approved version and assessment ID, updates
   that same pull request, and does not authorize publication.
3. “Use the release_manager agent to publish v0.x.y.” Publication requires a
   new explicit instruction naming the exact tag.

The specialist does not merge the feature PR, create a second release PR,
upload distributions directly, move an existing tag, or bypass
`.github/workflows/release.yml` and PyPI Trusted Publishing.
