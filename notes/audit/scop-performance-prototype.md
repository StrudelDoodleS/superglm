# SCOP Performance Prototype Blueprint

This branch is reserved for prototyping faster `discrete=True fit_reml()` paths
for monotone `PSpline` terms.

The goal is not to invent a different model. The goal is to keep the current
SCOP statistical story intact while reducing the structural cost of the inner
linear algebra and repeated objective evaluation.

## What This Branch Is For

Allowed:

- kernel-level speedups
- caching and reuse of assembled quantities
- exact structured linear algebra
- size-gated iterative linear solves
- controlled approximate solves with exact fallback
- benchmark and parity harnesses

Not allowed:

- changing the monotone constraint
- replacing the outer SCOP EFS update with a different statistical objective
- silently downgrading the final fit to a heuristic approximation
- mixing unrelated API cleanup into this branch

## Current Baseline

From the initial sweep on the current SCOP path:

- `1` monotone term, `n=1_000_000`: about `4.3s`
- `2` monotone terms, `n=1_000_000`: about `7.6s`
- `10` monotone terms, `n=100_000`: about `6.7s`

The current path is stable and convergent on these cases. The limit does not
yet look like a correctness failure. It looks like structural runtime growth.

## Hot Paths To Target

Primary:

- `_compute_cross_gram()` in `src/superglm/solvers/scop_newton.py`
- `_safe_joint_objective()` in `src/superglm/solvers/scop_newton.py`
- joint Hessian assembly in `scop_joint_newton_step()`
- repeated inner fits in `optimize_scop_efs_reml()`

Secondary:

- cross-block Jacobian scaling in `assemble_joint_hessian()`
- repeated late-stage outer iterations with barely moving lambdas

## Prototype Ladder

Work down this list in order.

### Phase 1: Exact But Faster

Keep the math identical.

Targets:

- Numba or equivalent acceleration for histogram/scatter kernels
- caching discretization structures used by cross-grams
- incremental objective evaluation during SCOP line search
- reuse of near-stationary block assemblies late in EFS

Acceptance:

- same final lambdas within tight tolerance
- same predictions within tight tolerance
- same monotonicity guarantees

### Phase 2: Exact Structured Linear Algebra

Keep the exact system, but solve it better.

Targets:

- explicit block structure for the joint SCOP Hessian
- block-diagonal preconditioners
- size-gated switch between dense direct solve and iterative exact solve

Acceptance:

- exact solver remains available as fallback
- no regression on small problems

### Phase 3: Controlled Approximation

Only after Phases 1 and 2.

Targets:

- inexact Newton solves in early outer iterations
- matrix-free `H @ v` products for larger SCOP systems
- weak-coupling truncation of tiny SCOP-SCOP blocks

Guardrails:

- approximations must be off by default
- exact fallback must remain available
- final cleanup iteration(s) should use the exact path

### Phase 4: Mixed-Policy Outer Optimization

Algorithmic acceleration for mixed models.

Targets:

- temporarily freeze stable SSP lambdas
- continue SCOP-focused iterations
- finish with final joint cleanup iterations

This is a heuristic and should remain clearly labeled as such.

## Benchmarks To Preserve

Every prototype should be checked against:

- `benchmarks/scop_discrete_limit.py`
- MTPL2 discrete Poisson with `1` monotone term
- MTPL2 discrete Poisson with `2` monotone terms if available
- `scasm` / `scam` parity harnesses in `scratch/` or a tracked equivalent

Minimum reporting:

- runtime
- outer iteration count
- average inner PIRLS iterations
- monotonicity check
- native centering check

## Success Criteria

The prototype is worth keeping only if it improves at least one of:

- runtime for `1-2` monotone terms at large `n`
- scaling in number of monotone terms
- scaling in basis dimension `k`

without materially hurting:

- convergence
- monotonicity
- fitted curves
- parity against the existing exact path

## Notes On Identifiability

The current SCOP implementation already uses SCAM-style centering:

- `B @ Sigma`
- drop the constant column
- center the remaining columns

So this branch is not trying to fix a known SCOP identifiability bug.

The current native means are effectively zero. Exposure-weighted means are not
zero, but that is also true for ordinary spline terms under the current native
identifiability convention. Treat that as a broader model-centering choice, not
as a SCOP-only bug.
