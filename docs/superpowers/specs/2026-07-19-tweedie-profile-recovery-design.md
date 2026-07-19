# Tweedie Profile Recovery Design

## Objective

Deliver correct, practical Tweedie power and dispersion profiling without slowing ordinary model
fitting. This branch starts from `codex/fit-reml-remediation` at `3656b50` and replaces the abandoned
`codex/tweedie-correctness` effort. The original branch remains archived and is not part of this
branch's history.

The existing `2026-07-16-tweedie-profile-correctness-performance-design.md` remains the statistical
starting point. This recovery design narrows its implementation scope and makes performance a hard
merge requirement.

## Supported contract

- Support finite numeric response, mean, offset, and strictly positive EDM prior-weight arrays.
- Support Tweedie powers in the existing open interval `1 < p < 2`.
- Keep exact maximum-likelihood dispersion profiling and bounded Brent search as the defaults.
- Keep Pearson dispersion as an explicitly approximate, faster plug-in option.
- Keep likelihood-ratio confidence intervals explicit and available only when their profile
  evaluations are valid exact-likelihood evaluations.
- Preserve the existing public `estimate_p()`, result, trace, and CI interfaces unless a failing
  correctness test demonstrates that an interface is internally inconsistent.

Correctness means agreement with independent high-precision and R references over representative
supported ranges. It does not mean bit-identical certification for every finite float64 pattern.
Inputs outside a numerically supported region must fail clearly rather than silently masquerade as
exact likelihood evaluations.

## Audit targets

Only demonstrated defects receive production changes. The initial audit will test:

1. Candidate fits use the same solver configuration, offsets, smoothing parameters, and convergence
   policy as the corresponding ordinary fit.
2. Weighted density, deviance, Pearson dispersion, and log-likelihood use the EDM convention
   `Var(Y_i) = phi * mu_i**p / w_i` consistently.
3. Exact density and log-dispersion score agree with independent references for zero and positive
   observations across ordinary powers and scales.
4. Inner dispersion failures, boundary optima, or non-finite objectives cannot be published as
   successful outer power estimates.
5. Grid and Brent searches finalize from the actual winning candidate rather than mutable
   last-evaluation state.
6. Profile and CI evaluation leave the returned model internally consistent.
7. Fixed-power `fit()` and `fit_reml()` report coherent deviance, likelihood, EDF, and dispersion.

An audit target that already behaves correctly gets a characterization test, not a rewrite.

## Design

### Numerical evaluation

Ordinary arrays use vectorized NumPy/SciPy evaluation. Exact positive-density work is prepared once
for fixed `(y, p, weights)` and reused across dispersion evaluations. Fitted and null likelihoods
reuse their shared normalizing work.

The existing fast Wright-Bessel route may be retained where reference tests establish its accuracy.
Rows it cannot evaluate reliably may use a small scalar high-precision fallback. The fallback runs
only on the exceptional subset; ordinary fitting must never loop through Python or `Decimal` once
per observation. An unsupported exact row raises the existing numerical error instead of silently
becoming a saddlepoint value labelled as exact.

Unit deviance and Pearson dispersion remain simple vectorized kernels with stable formulas around
equal `y` and `mu` and near the power boundaries. Public input validation occurs once at the API
boundary, not repeatedly inside solver hot loops.

### Profile search

Each candidate power produces one immutable evaluation record containing fitted means, dispersion,
likelihood, EDF, convergence, boundary status, and density method. The outer search caches these
records, evaluates both power bounds, and may select only finite, converged candidates whose stated
likelihood method is valid.

Exact MLE dispersion uses prepared density data and an analytic score when the score passes a
finite-difference check. A bounded derivative-free exact objective remains the fallback. Warm starts
may reduce work but cannot change the selected objective or validity rules.

Pearson mode remains separate and cannot produce likelihood-ratio intervals. REML profiling remains
a plug-in likelihood over REML-selected smooths; it will not be described as joint mgcv-style REML
over power, scale, and smoothing.

### Model state

Use the remediation branch's existing fit workspace and normal model-copy mechanisms. Profile work
is prepared, evaluated, and then committed once. Failures leave the public model at its pre-call
state.

No new recursive object-graph validator, copy-protocol auditor, metaclass inspection, descriptor
inspection, or manual `__dict__` transaction framework will be introduced.

## Performance gates

Performance is part of correctness for this work and blocks merge.

- Alternate single-threaded benchmark runs between this branch and frozen `3656b50`.
- Representative fixed-power `fit()` and `fit_reml()` medians may not regress by more than 10% or
  50 milliseconds, whichever allowance is larger.
- Ordinary 10,000-row fitted/null likelihood evaluation must remain vectorized and invoke no scalar
  high-precision fallback. A structural routing assertion guards this property.
- Representative `estimate_p(phi_method="mle")` workloads may not regress by more than 25% or
  250 milliseconds against remediation, whichever allowance is larger, while satisfying the new
  exact-likelihood reference checks.
- Benchmarks report candidate-fit count and density-evaluation count so caching cannot hide extra
  work behind acceptable wall time on one machine.

These limits are regression gates, not permission to consume the full allowance.

## Test strategy

Every production change begins with a focused failing test. The compact test set will cover:

- independent density, score, deviance, dispersion, and fit references;
- weighted, unweighted, zero-heavy, offset, boundary-power, and flexible-spline profiles;
- inner failure and boundary propagation;
- winning-record and post-profile model-state consistency;
- exact-MLE versus derivative-free objective agreement;
- Pearson CI rejection;
- public `fit()`, `fit_reml()`, and `estimate_p()` performance regressions.

Reference fixtures record their generator and tolerance rationale. Tests should exercise public
behavior where possible; private routing tests are limited to proving vectorized ordinary execution
and exceptional-row fallback.

Verification consists of focused red/green cycles, Ruff and formatting for touched files, targeted
mypy, the existing non-slow/non-browser developer suite, the relevant slow Tweedie tests, the full
suite, and alternating performance measurements.

## Scope controls

Expected production scope is limited to Tweedie numerical evaluation, `profiling/tweedie.py`, the
small profile orchestration boundary, and fit-statistic reuse. Tests and reference fixtures may be
added only for the contracts above.

The following are explicit non-goals:

- hostile or arbitrary Python objects, custom metaclasses, descriptors, reducers, slots, or copy
  hooks;
- new serialization, concurrency, retention-lifecycle, editor, reporting, or plotting frameworks;
- auditing pandas or NumPy implementation internals;
- universal bit-pattern certification or bit-for-bit equality to a decimal oracle;
- generic IRLS, PIRLS, REML, matrix, or design-matrix rewrites;
- changing the sample-weight convention or claiming joint mgcv-style Tweedie REML;
- unrelated cleanup or refactoring.

Any proposed production change outside this scope requires a demonstrated failing public-behavior
test and explicit user approval before implementation.
