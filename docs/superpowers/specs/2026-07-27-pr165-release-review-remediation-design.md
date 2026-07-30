# PR 165 Release Review Remediation Design

**Date:** 2026-07-27

**Status:** Approved

## Goal

Resolve the independently reproduced release blockers and integration defects
reported against PR #165 at commit
`0dcd99ca731f5560974b1e1766fea377fcd94f5e`, without redesigning the structured
solver, weakening numerical contracts, dropping Python 3.10, or regressing the
compact large-\(n\) execution paths.

## Scope

This pass addresses:

1. zero-penalty RE/FS Schur rank corruption;
2. NB2 automatic-theta dispatch with credibility terms;
3. the unsupported stacked `scipy.linalg.eigh` call at the declared dependency
   floor;
4. fit-time and post-fit shape constraints combined with structured terms;
5. scalar structured `S_override` geometry validation;
6. silent wide-system estimability degradation;
7. undocumented discrete-REML compatibility changes;
8. low-risk review cleanup in examples, typing, covariance selectors, tests,
   and documentation;
9. missing Python 3.10 validation on pull requests.

The pass does not add multi-dominant structured algebra, rank-deficient
constrained REML, structured rating-table export, or a new REML optimizer.

## 1. Zero-Penalty Structured Geometry

An all-level unpenalized `RandomEffect` is exactly aliased with SuperGLM's
fitted intercept. An FS component fixed at zero can likewise expose an
unpenalized direction that aliases the intercept or a co-fitted population
smooth. Floating-point cancellation can leave a small positive Schur pivot,
causing Cholesky to report full rank and corrupting the REML log determinant.

The backend policy will therefore be:

- `direct_solve="auto"` falls back to Gram when the dominant `RandomEffect`
  has lambda zero.
- `direct_solve="structured"` rejects that geometry before assembly with a
  precise ineligibility message.
- Ordinary FS falls back or rejects when any of its repeated penalty
  components is explicitly zero. SZ retains its structured path because its
  exact sum-to-zero constraint removes the corresponding population alias.
- Positive lambdas, including the optimizer's lower working bound, remain
  eligible.

The scalar and block Schur factors will also gain an absolute pivot floor based
on the pre-subtraction reference scale. A rank-deficient Schur factor is safe
only when its null directions are uncoupled from the eliminated structured
block. Coupled singular geometry will raise instead of publishing an invalid
pseudo-determinant or generalized inverse. This is a defensive factor-level
contract; normal public zero-lambda cases are handled earlier by dispatch.

Tests will cover auto fallback, forced rejection, the exact cancellation
construction, positive-lambda continuity, and Gram parity.

## 2. NB2 Automatic Theta

The NB2 theta pre-pass currently builds the design but omits compact REML
penalty components when calling the direct solver. It will build the same
penalty context used by the later REML fit and pass those components into every
theta-iteration IRLS call. This preserves structured performance rather than
forcing a dense \(K \times K\) penalty.

As a defensive low-level contract, `fit_irls_direct(..., direct_solve="auto")`
will fall back to Gram if a structured candidate lacks both `S_override` and
compact penalty components. Forced structured mode will retain a clear error.

Tests will cover public `families.nb2(theta="auto")` with RE and FS terms,
automatic structured dispatch, finite theta, and explicit low-level fallback.

## 3. Dependency-Floor-Compatible SZ Eigensolve

Only `_decompose_local_psd_batch` passes stacked `(K, k, k)` input to
`scipy.linalg.eigh`. SciPy added that batched contract in 1.16, while SuperGLM
supports Python 3.10 and declares `scipy>=1.10`.

The one stacked call will use `np.linalg.eigh` unconditionally:

- NumPy's batched contract is available at the existing `numpy>=1.24` floor.
- The function already validates finiteness before decomposition.
- Eigenvalue ordering, rank thresholds, null spaces, pseudo-inverses, and
  reconstruction remain unchanged.
- Local measurements show NumPy is materially faster for the small blocks used
  by FS/SZ.

There will be no SciPy version branch and no dependency-floor change. Tests
will include full-rank and deficient batches and compare rank, reconstruction,
and pseudo-inverse geometry with dense references.

## 4. Shape Constraints and Structured Terms

Fit-time constrained REML is not currently defined for identity, repeated, or
sum-to-zero penalty components. `fit_reml()` will reject any combination of a
fit-time shape constraint and RE/FS/SZ terms before NB profiling or solver
dispatch. The message will name both unsupported feature classes and direct
users to separate models or post-fit repair.

Post-fit repair does not require a constrained REML redesign. It will evaluate
the fitted penalty using the existing compact
`penalty_component_quadratic()` algebra, which already supports identity,
repeated, and sum-to-zero components. This keeps RE/FS/SZ compact and allows a
violated spline constraint to be repaired without accessing `R_inv` or
expanding \(Kk \times Kk\) penalties.

Tests will cover:

- early fit-time rejection for RE, FS, and SZ;
- successful, coefficient-changing post-fit repair with RE, FS, and SZ;
- compact execution guarded by sentinels against dense penalty expansion;
- transactional preservation when repair fails.

## 5. Authoritative `S_override`

The scalar structured operator accepts only a diagonal penalty inside its
dominant RE block. It will explicitly inspect the entire dominant block and
raise when any off-diagonal mass is present, matching the existing block-FS
and SZ validation contracts. Dense-small penalties and diagonal RE penalties
remain supported.

## 6. Estimability Failure Contract

`centered_operator_coefficient_estimable()` will no longer convert every
`ValueError` or iterative eigensolver failure into an unreported all-false mask.

- Programming/contract `ValueError`s propagate.
- Numerical certification failures may use the exact dense fallback when the
  existing 512-coefficient allocation bound permits it.
- Above that bound, the function raises a precise `RuntimeError` explaining
  that compact estimability certification failed, rather than silently
  returning all-NaN inference.

Normal, explicitly identified unestimable geometry continues to use the
existing conservative mask where the algorithm calls for it.

## 7. Compatibility Documentation

The PR compatibility section and credibility guide will state:

- `fit_reml()` accepts only `selection_penalty=None` or `0.0`; explicit zero in
  examples documents that REML owns smoothing while `fit()`/`fit_path()` own
  sparse selection.
- Discrete tensor REML changed its post-stall ratio cap and no longer lets a
  two-margin update reintroduce a frozen coordinate.
- Exact and discrete REML no longer install an unevaluated steepest-descent
  fallback after every line-search trial is rejected.
- Exact REML may terminate as `line_search_failed`; discrete REML retains the
  evaluated candidate and may repeat it until the iteration limit.
- The final discrete `lambda_history` snapshot is the authoritative converged
  refit and can equal the preceding evaluated candidate.
- The freMTPL2 table is a dated seeded demonstration, not a CI-pinned
  performance or data contract.

## 8. Low-Risk Cleanup

The pass will:

- remove the unused `_penalty_component_cross_trace`;
- correct `dH_extra` typing to include compact symmetric operators;
- validate boolean covariance-selector length like NumPy;
- replace private model navigation in the flagship example with public
  `diagnostics()`, `features`, and known variant contracts;
- document broad the reference implementation lambda tolerances as flat-optimum allowances.

It will not change intentional lambda-history retention semantics.

## 9. Pull-Request Compatibility CI

`.github/workflows/ci.yml` will gain a `pull_request` trigger for the same
source/test/dependency paths.

To avoid duplicating the expensive master suite:

- pull requests run one complete non-browser suite on Python 3.10;
- master pushes retain the full 3.10/3.11/3.12/3.14 compatibility matrix,
  coverage shards, lint, and browser jobs;
- all PR jobs remain read-only and use no publishing credentials.

This catches dependency-floor and syntax regressions before merge while the
existing development workflow continues to exercise Python 3.14.

## Verification

Every behavior change will be introduced test-first. Completion requires:

- focused regressions for all P1 and P2 findings;
- existing structured dense-parity, SZ, RE, FS, NB2, shape, covariance, and
  REML tests;
- Python 3.10 dependency resolution and focused SZ execution;
- Ruff check and format check on touched files;
- the complete non-browser/full local suite;
- package build and repository smoke test;
- exact-head GitHub checks, a fresh Codex review, and no unresolved review
  threads.
