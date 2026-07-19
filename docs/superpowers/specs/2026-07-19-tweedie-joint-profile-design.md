# Tweedie Exact Profile Acceleration Design

## Goal

Make exact Tweedie `p`/`phi` profiling competitive with or faster than mgcv on
ordinary data without changing ordinary `fit()` behavior, weakening the
likelihood, or allowing pathological series work to crash or hang a fit.

The primary warm target is at least a tenfold reduction from the current
roughly 0.215-second 800-row exact profile. The corresponding mgcv 1.9-4 run on
this machine takes about 0.023--0.025 seconds. SuperGLM should reach 0.022
seconds or less for the internal profile when feasible, and should pursue
public end-to-end parity without sacrificing correctness.

## Measured cause

On the shared 800-row neutral fixture, the current exact implementation performs
14 fixed-power fits and 278 complete density-series passes. Approximately 94%
of profile time is in the inner exact likelihood. mgcv performs seven exact
series passes and six joint Newton updates. Its advantage is primarily avoided
work, not the language of its series loop.

A clean-room prototype established the available gains:

- using only one dispersion seed reduces 278 passes to 158 and time to about
  0.128 seconds;
- compiling the existing series loop reduces the full profile to about 0.116
  seconds but leaves the repeated-work architecture intact;
- an analytic outer `p` score plus compiled series takes about 0.064 seconds;
- analytic dispersion curvature and Schur-corrected `p` steps take about 0.031
  seconds despite still performing 18 series passes and repeated preparation;
- four warm fixed-power fits on the relevant candidate sequence take only about
  0.006 seconds in total.

The remaining path to parity is therefore to fuse the likelihood derivatives,
remove curvature probes and repeated preparation, and reduce ordinary profiles
to roughly 4--8 exact series passes.

## Public behavior

Add `method="joint_ml"` as a real exact profiling method. Add
`method="auto"` and make it the default for both the model and standalone
entry points:

1. `auto` uses the joint exact fast path when `fit_mode="fit"`,
   `phi_method="mle"`, and the coefficient fit satisfies the fast path's
   envelope-theorem requirements.
2. Otherwise `auto` uses the existing Brent profile. Its fixed-power dispersion
   evaluations may still use the new exact Newton accelerator.
3. Explicit `method="brent"`, `grid`, `grid_refine`, and `profile_opt` retain
   their search semantics.
4. Explicit `method="joint_ml"` attempts the joint exact method and falls back
   to Brent with a recorded reason when exact work, curvature, coefficient-fit
   eligibility, or validation is unsafe. It never substitutes Pearson or
   saddlepoint likelihood silently.
5. `method="integrated"` remains unimplemented and out of scope.

For an unpenalized maximum-likelihood coefficient fit, the exact partial `p`
score is the fitted-profile score by the envelope theorem. Active coefficient
penalties can make it differ, so the fast solver treats that score only as a
proposal and certifies the winner with actual fixed-power profile objectives on
both sides. A positive-curvature parabolic certificate must place the true
profile vertex within `xatol`; otherwise the solver falls back to Brent. This
keeps the common default penalized fit fast without assuming an invalid score
identity. Monotonic constraints and REML smoothing retain the existing outer
optimizer.

The reported `method`, search trace, convergence fields, density provenance,
callbacks, lazy likelihood-ratio confidence intervals, and final atomic refit
remain supported. Every outer fast-path record contains a genuinely profiled
`phi`, rather than an intermediate joint iterate, so existing profile and CI
semantics remain valid.

## Exact sufficient-statistics kernel

Create a small private module for the compiled kernel rather than adding another
large numerical block to `profiling/tweedie.py`. It uses Numba with persistent
caching, strict IEEE behavior, and no `fastmath`.

For `1 < p < 2`, define

```
a = (2 - p) / (p - 1)
q_ij = j * log(t_i) - lgamma(j + 1) - lgamma(a * j)
S_i = log(sum_j exp(q_ij))
```

The positive-response log density is `S_i - log(y_i)` plus the canonical
`w_i / phi` term. Zero-response contributions are analytic.

For each contributing term, the kernel evaluates analytic first and second
derivatives of `q_ij` with respect to `p` and `u = log(phi)`. In the same
mode-centered upward/downward sweep it accumulates scaled mass and the five
moments needed for

```
dS/dx    = E[dq/dx]
d2S/dx2 = E[d2q/dx2] + Var(dq/dx)
d2S/dpdu = E[d2q/dpdu] + Cov(dq/dp, dq/du)
```

Analytic canonical and zero-response derivatives are then added. One call
returns the aggregate negative log-likelihood, gradients in `(p, u)`, the full
2-by-2 Hessian, exact row/term counts, and a compact failure status. Per-row
likelihood and derivative arrays are not materialized.

The kernel uses independently implemented scalar digamma and trigamma
recurrences with asymptotic tails. Tests compare them directly with SciPy over
the full argument range exercised by the series. This avoids a new runtime
dependency and keeps the compiled loop nopython-compatible.

## Series work and memory

The kernel retains the validated Dunn--Smyth mode-centered policy from the
density recovery:

- locate and check the adjacent integer mode;
- sweep upward and downward from the largest term;
- stop only after the term is at least 37 log units below the maximum;
- enforce the existing per-row and total exact-work budgets;
- reject non-finite indices, sums, derivatives, or unsafe modes through a
  status code rather than raising inside compiled code.

Performance optimizations that preserve the exact sum are required:

- prepare `y`, `log(y)`, `log(weight)`, zero/positive indices, and immutable
  scalar metadata once per profile context;
- avoid repeated NumPy masks, copies, closures, Python callbacks, and density
  result objects inside Newton iterations;
- fuse likelihood, score, curvature, diagnostics, and term counting;
- reuse caller-owned fixed-size output/work buffers when measurement shows a
  benefit;
- reuse `j`-dependent gamma/digamma/trigamma terms within one candidate
  evaluation when a bounded contiguous table is cheaper than recomputation;
- use scalar recurrences in sparse or wide-index cases to avoid mgcv's large
  contiguous-buffer failure mode;
- avoid object construction for rejected trial steps;
- retain deterministic summation order and full binary64 convergence checks.

No optimization may replace the exact objective with saddlepoint, Pearson, a
looser series cutoff, `fastmath`, or silent clipping.

## Fixed-power dispersion Newton

The fused kernel supplies the exact score and Hessian in `u = log(phi)`. For a
fixed fitted mean and power:

1. Start from the previous accepted `phi`, otherwise the stable Pearson seed.
2. Take a bounded Newton step using the exact `u` score and curvature.
3. Limit large steps, require finite improving trial objectives, and halve a
   rejected step.
4. Stop on both score and step tolerances.
5. Validate the final objective and score with the public exact density route.

Ordinary exact cases should require one to three kernel passes after the first
power because dispersion is warm-started. If curvature is non-positive, the
series branch changes, validation disagrees, or Newton cannot certify a local
minimum, call the existing globally defensive dispersion profiler and preserve
its diagnostics.

This accelerator is also available to fixed-power evaluations made by Brent,
grid, and REML outer searches. Thus fallback does not necessarily lose all of
the performance improvement.

## Safeguarded outer power solve

For an eligible ordinary ML profile, the fitted coefficient score is zero, so
the envelope theorem permits the partial exact likelihood score in `p` to drive
the profiled optimum. Each outer candidate performs one warm-started coefficient
fit followed by the exact fixed-power dispersion Newton.

The proposed `p` step uses the dispersion-profiled curvature

```
H_profile = H_pp - H_pu * inverse(H_uu) * H_up
```

which is the scalar Schur complement `H_pp - H_pu**2 / H_uu`. Coefficient
response can make this curvature imperfect even when the score is exact, so it
is a step proposal rather than an unchecked convergence certificate.

The outer solver:

- works inside strict `p_bounds`, with a bounded transform or explicit step
  clipping away from `p=1` and `p=2`;
- carries coefficient and dispersion warm starts;
- uses the analytic score for convergence;
- prefers the Schur-Newton step, then a secant step, then a bracketed safe step;
- accepts only finite candidates that improve the exact profiled objective or
  advance a valid score bracket;
- validates the final score orientation and exact objective;
- falls back to existing Brent from accumulated valid profile points when the
  derivative path cannot be certified.

The final winning record is materialized through the existing immutable result
and density-diagnostic machinery. Lazy CI evaluations continue to call the
authoritative fixed-power profile objective.

## Failure and fallback policy

The fast path is an optimization, never a new failure mode.

- A work-budget status, non-finite derivative, bad curvature, invalid Newton
  step, coefficient-fit failure, boundary/KKT ambiguity, objective mismatch, or
  score-validation failure triggers the existing bounded profile machinery.
- Pathological tiny-`phi` rows retain the diagnosed bounded saddlepoint fallback
  from the density recovery; the profiler does not hang or allocate an
  unbounded gamma table.
- Explicit `joint_ml` reports that fallback occurred and why.
- Valid finite input must not crash merely because Numba compilation or the
  fast kernel is unavailable; the Python exact path remains authoritative.

## Verification and performance gates

Implementation is test-first and must establish:

1. Scalar digamma/trigamma helpers agree with SciPy to binary64-appropriate
   tolerances over small, moderate, and large positive arguments.
2. Kernel likelihood, both scores, both Hessian diagonals, and cross derivative
   agree with high-accuracy central differences and the existing exact density
   on weighted data containing zeros.
3. Kernel results are invariant to row ordering and match neutral high-precision
   fixtures on both sides of `p=1.5`.
4. Fixed-power Newton matches the existing globally defensive MLE `phi` profile
   across routine powers and dispersions, including weighted cases.
5. Joint profiling matches independent exact references and existing Brent
   estimates across deterministic and randomized fixtures, including the neutral
   reference near `p=1.1968971`, `phi=0.8068142`.
6. Ineligible, boundary, non-finite-curvature, work-limit, and deliberately
   corrupted-validation cases fall back deterministically with honest
   diagnostics.
7. Existing exact/saddlepoint provenance, trace callbacks, CI behavior, final
   refit synchronization, caller-state isolation, and REML tests remain green.
8. Ordinary `fit()` benchmarks show no material regression because the new
   kernel is profile-only.

Performance measurements use warm, counterbalanced medians and record exact
kernel calls, coefficient fits, and total terms. On the shared 800-row fixture:

- hard target: no more than 12 exact kernel passes and at least 8x faster than
  the current 0.215-second implementation;
- parity target: no more than 8 exact kernel passes and no slower than the
  measured 0.023--0.025-second mgcv run;
- stretch target: internal profile at or below 0.015 seconds and public
  `model.estimate_p()` at or below mgcv when final-refit cost permits.

Wall-time thresholds are reported rather than placed in ordinary CI. Stable
call-count and exact-work assertions prevent algorithmic regressions in CI.

## Excluded scope

- Porting or copying mgcv C/R implementation details.
- Fourier inversion or revival of the orphaned certified-density engine.
- Replacing the exact ordinary profile with saddlepoint or Pearson estimates.
- Full implicit differentiation through REML smoothing parameters in this
  change. REML retains its correct outer search while receiving the fixed-power
  dispersion acceleration where eligible.
- Metaclasses, general optimizer frameworks, or unrelated profile refactors.
