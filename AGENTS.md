# Repository Guidelines

These guidelines apply whenever an agent works in this repository. The user's
instructions for the task at hand take precedence over them, and they take
precedence over any skill or harness default. When two instructions conflict,
say which one you are following and why.

## Environment and checks

Install the development environment with `uv sync --python 3.13 --extra dev`.
The package supports Python 3.12 to 3.14 and CI runs all three; 3.13 is the
development default. The ordinary checks are:

- `uv run python scripts/run_test_suite.py`, which runs the suite as CI does, in
  parallel. Add `-m "not slow and not browser and not docs"` for the quick pass.
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

Do feature work in an isolated worktree under `.worktrees/` or
`.claude/worktrees/`; both are git-ignored. Several sessions share this
checkout, its stash stack and the main working tree, so leave uncommitted
changes you did not make alone: a reset, a checkout over a dirty file or a
stash pop can discard another session's work.

## Delegation and effort

When you spawn a sub-agent, set its model and reasoning effort explicitly and
put the effort in the task name, for example `scores_low` or
`posterior_review_high`, so the transcript shows what each run was given. Low
effort suits straightforward fixes, medium suits multi-module or semantic
work, and high or above suits difficult mathematical or numerical work. Say
what the sub-agent may run in parallel and who consolidates the results; the
spawning agent keeps working while sub-agents run and owns the final answer.
Sub-agents run the focused tests for their stage; the full suite takes many
minutes and runs once at the end.

## Code and tests

Package code lives under `src/superglm/`, tests under `tests/`, and
benchmarks and exploratory work outside production paths. Public APIs are
exported through `src/superglm/__init__.py`. Follow the existing Ruff
configuration, and keep mathematical names where they make the numerical
implementation easier to read against the method it implements.

New solver, REML, family, input-boundary or feature behaviour needs a focused
regression test. An adversarial regression includes a mutation check or a
demonstration against the unfixed implementation; a test that also passes
against the unfixed code proves nothing.

## Performance evidence

Changes to solver logic, REML, numerical kernels, or anything else that can
change time to fit need performance evidence alongside correctness. What
counts is a complete fit on representative data, compared with the baseline on
wall time, memory including any retained caches, numerical outputs, and the
backend actually dispatched; a microbenchmark alone does not establish a fit
improvement. Use the existing diagnostics, such as `SuperLSS.diagnose()`, to
attribute the time to phases, iterations, retries and refits, and keep
profiling runs separate from the timing runs you report.

Before adding a cache, look for reuse the code already misses: discarded or
recomputed results, repeated factorizations or products, and matrices
materialized where a product would do. A cache you do add needs a stated
owner, lifetime and invalidation conditions, which here means weights,
parameters, basis, penalty target and numerical precision, and it must
preserve rank decisions, error bounds and validation evidence. Record the
findings and any remaining regression with the change.

## Project direction

Read [notes/ROADMAP.md](notes/ROADMAP.md) before proposing substantial new
functionality, making a major architectural or API decision, choosing what to
work on next, or doing strategic research. Narrowly scoped fixes, tests, CI
and build work, dependency maintenance, mechanical refactors and explicitly
scoped audits do not need it. The roadmap is directional project state, not
an implementation specification: if repository evidence contradicts it, report
the discrepancy and propose an update rather than conforming the
implementation to an outdated assumption.

## Numerical policy

Production numerical code targets portable IEEE binary64 (`numpy.float64`) on
Linux, Windows and macOS. Do not depend on `longdouble`, `float96`, `float128`
or any platform-specific mantissa or exponent range: those types differ
between platforms and compilers, and the failures they cause surface only on
native Windows and macOS runners. Derive algorithms and error bounds for the
float64 operations actually performed, using stable scaling, factorizations
and narrowly justified compensated reductions where they are needed. The
replacement for an extended dtype is float64 analysis, not a custom
arbitrary-precision layer. Higher-precision arithmetic belongs in independent
test references, not in production fitting, and loosening a numerical check
needs an analysis of why the old bound was wrong.

A numerical portability fix needs native Windows and macOS regression
coverage; a Linux simulation of the platform does not establish native
compatibility. Compare complete-fit cost before accepting the change.

Tests assert what the mathematics certifies. Boundary tests check invariants
such as rank, subspace, residual, reconstruction, prediction or backward
error, never the sign or magnitude of BLAS/LAPACK round-off, which varies by
driver and machine. Tolerances derive from dimensions, dtype epsilon, norms
and conditioning or error bounds, not from what passed locally.
Coefficient-forward accuracy is tested on well-conditioned fixtures; near-rank
and cancellation fixtures test certification, refusal and the stable
observables instead. Performance and backend dispatch are tested separately
from numerical correctness.

## Release impact and publishing

Every pull request declares exactly one advisory impact in its body, with a
rationale: `release:none`, `release:patch` or `release:minor`. The declaration
is advice to the next release, never a version change. Feature and fix pull
requests do not touch `pyproject.toml`'s version, `superglm.__version__` or
`uv.lock`'s own version pin, and carry no version-record commits; reviewers
flag any pull request that does. The code diff is authoritative; declarations
and labels are evidence only.

A release is one deliberate act on master after merging: a single bump commit
whose message is the consolidated changelog since the previous release, tagged
`vX.Y.Z` on that same commit, then published, so the version file, the tag and
PyPI move together and can never disagree. Tags remain release-only. A merge
never authorizes a tag or publication. Determine the release base from the
current published release and its matching tag, never from a version-record
commit on master; [docs/development/releases.md](docs/development/releases.md)
records the versions that were recorded but never published.

Only an explicit user request to assess, prepare or publish a release may
invoke the project-scoped `release_manager` specialist defined in
`.codex/agents/release_manager.toml`. Proactive delegation and ordinary words
such as "finish", "merge", "ship" or "deploy" do not authorize publication.
The three gates are separate requests, and each does only its own step:

1. "Use the release_manager agent to assess the unreleased changes as a
   release candidate." Assessment is read-only and bound to the exact base
   and head SHAs.
2. "Use the release_manager agent to prepare the approved 0.x.y." Preparation
   requires the exact approved version and assessment ID. It writes the
   single bump commit via a bump-only pull request, rebase-merged under the
   linear-history rule so the release tag binds to the rebased commit on
   master. Preparation does not authorize publication.
3. "Use the release_manager agent to publish v0.x.y." Publication requires a
   new explicit instruction naming the exact tag.

The specialist never merges feature pull requests, uploads distributions
directly, moves an existing tag, or bypasses `.github/workflows/release.yml`
and PyPI Trusted Publishing.

## Harness notes

- Codex: sub-agents default to `gpt-6-astra`, and the task name carries the
  model as well as the effort, as in `scores_astra_low`. Edit files with
  `apply_patch`. Put worktrees under `.worktrees/`.
- Claude Code: sub-agents run on the model the session is configured with,
  and the same effort ladder applies. Put worktrees under `.claude/worktrees/`.
