# Discrete REML Terminal-State Coherence

## Problem

The batched discrete `FactorSmooth` Gram prototype exposed a latent control-flow
bug in `optimize_discrete_reml_cached_w`.

For the pinned Poisson SZ fixture, the retained compact-loop moments and the
batched cell contraction agree to a maximum relative Gram error of
`2.43e-16`. Both fits then follow the same REML path through the evaluated
candidate

```text
x = 0.5488426744
x:f:sz:wiggle = 6.424373509
```

to roughly 13--15 significant digits. Despite that agreement, the batched fit
publishes lambdas `(0.5245731761, 2.363394936)` and its predictions differ by
up to 12.9%.

The divergence occurs because the POI loop:

1. evaluates the objective and gradient at `cand_lambdas`;
2. computes and accepts a line-search step that mutates `rho`;
3. checks convergence using the objective and gradient from `cand_lambdas`;
4. breaks as converged; and
5. performs its terminal full PIRLS refit at the mutated, unevaluated `rho`.

On a flat REML surface, a tiny gradient and near-singular curvature can produce
a large Newton proposal. A few ulps in an algebraically equivalent Gram can
decide whether the trial objective is strictly smaller, exposing the state
mismatch.

## External Reference

This design follows the control-flow invariant visible in the reference implementation without copying
its implementation. The documented discrete `bam` path uses performance
oriented iteration and warns that POI is less stable than nested iteration:

- <https://search.r-project.org/CRAN/refmans/the reference implementation/html/bam.html>

In the reference implementation 1.9-4's `bgam.fitd`, convergence is checked after preparing the current
working model and before requesting the next smoothing-parameter Newton step:

- <https://github.com/cran/the reference implementation/blob/master/R/bam.r#L3383-L3436>

The relevant lesson is state coherence, not a reference-specific formula: a
convergence decision must apply to the same smoothing parameters and
coefficient state that will be retained.

## Chosen Design

### Terminal-state invariant

A converged discrete REML result must publish one coherent tuple:

```text
(working model, coefficients, lambdas, objective, gradient)
```

The objective and convergence criteria must have been evaluated at those
lambdas. No subsequent smoothing-parameter step may be installed without a
new working-model evaluation.

### POI control flow

Move the compound convergence check to the current-candidate boundary, after
the current objective and projected gradient are available but before Hessian
construction and line search.

When the current candidate is converged:

- retain `rho_clipped`, which generated `cand_lambdas`;
- skip Hessian construction and line search;
- exit the POI loop; and
- run the existing full terminal PIRLS refit at those same lambdas.

When it is not converged, continue with the existing Newton, step cap,
line-search, fallback, and design-rebuild logic. This preserves POI behavior
away from the terminal iteration and removes unnecessary terminal line-search
work.

### Batched FactorSmooth moments

After the optimizer invariant is fixed, restore the algebraically exact batched
cell contraction:

1. retain the single compiled row pass that accumulates level-by-bin weights
   and weighted right-hand sides;
2. evaluate the effective support basis once as `B_unique @ natural_map`;
3. form all per-level Grams with batched matrix multiplication;
4. form `X'W` and `X'Wz` from the same cells; and
5. canonicalize Gram symmetry at the existing moment boundary.

For SZ, `natural_map` is the identity. For FS, this computes

```text
(B M)' diag(w) (B M) = M' B' diag(w) B M
```

directly in natural coordinates. The retained dense-small cross contraction is
unchanged.

## Alternatives Considered

### Restore `rho` only after detecting convergence

This is a smaller patch, but it still computes and may report an accepted
line-search step that is then silently discarded. It also wastes the most
expensive lambda-only work on the terminal iteration.

### Require one more POI iteration after every accepted step

This would ensure the accepted step is evaluated before publication, but it
adds an iteration even when the current candidate is already converged. It
also makes stopping depend on whether a numerically insignificant trial was
strictly accepted.

### Keep the compact Gram loop

This avoids exposing the bug for the pinned fixture but leaves the optimizer
state mismatch intact. Another BLAS implementation, platform, thread count, or
otherwise harmless summation-order change could trigger it later.

## Tests

Implementation follows red-green-refactor.

1. Add a regression test using the real pinned discrete Poisson SZ fit and an
   algebraically equivalent batched moment implementation. Before the fix, it
   must reproduce the terminal lambda jump.
2. Assert that a converged result's terminal lambdas equal the last candidate
   whose convergence was evaluated, and that the terminal predictions retain
   exact/discrete SZ parity.
3. Retain direct moment-oracle tests for FS and SZ across rectangular natural
   maps, signed weights, and weighted right-hand sides.
4. Run focused discrete REML, structured solver, FactorSmooth, and reference-parity
   tests.
5. Run the complete test suite and Ruff.

The regression test must fail for the state-coherence reason before production
code changes, rather than merely asserting one preferred numerical checksum.

## Performance and Acceptance Criteria

- The pinned batched SZ fit must retain the evaluated solution near
  `(0.54884, 6.42437)` instead of the unevaluated `(0.52457, 2.36339)` step.
- Batched and retained moment arrays must remain within floating-point
  summation error of the explicit row oracle.
- Existing the reference implementation SZ parity and exact-versus-discrete prediction tests must pass.
- Existing FS and RE behavior must remain unchanged.
- The million-row benchmark must show no time-to-fit regression relative to
  the current retained implementation; report median timings from repeated
  runs.
- No LSS code and no C, C++, Cython, or Rust implementation is in scope.
