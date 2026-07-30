# Tweedie Profile Correctness and Performance Design

## Goal

Make `SuperGLM.estimate_p()` a statistically coherent, state-safe Tweedie
profile-likelihood implementation. Exact maximum-likelihood profiling of
dispersion is the default, bounded Brent remains the default search over `p`,
and the exact MLE path must be fast enough for routine use.

## Public contract

- Default to `phi_method="mle"` and `method="brent"`.
- Keep Pearson dispersion as an explicitly approximate plug-in method. It must
  not be described as an MLE, and likelihood-ratio confidence intervals must
  not silently be reported for it.
- Do not compute a confidence interval as an implicit side effect of
  `estimate_p()`. The existing cached `result.ci()` API remains the explicit
  way to request one.
- Continue to interpret `sample_weight` as a strictly positive EDM prior
  weight, so the observation dispersion is `phi / weight`. Reject non-finite
  or non-positive weights before fitting or profiling.
- Preserve bounded Brent over `p`. Probe endpoints and surface boundary,
  convergence, and saddlepoint diagnostics rather than treating local optimizer
  success as proof of a globally reliable profile.

## Correctness changes

1. Make profile fitting dispatch match ordinary `fit()`:
   forward `lambda2`, tolerance, iteration limit, convergence mode, active-set
   configuration, and constrained/direct-solver requirements. This restores
   the correct spline effective degrees of freedom used by Pearson dispersion.
2. Include offsets whenever fitted means are reconstructed, including the REML
   profile path.
3. Cache a complete immutable evaluation record per candidate `p`: NLL, phi,
   fitted mean, EDF, convergence information, and density diagnostics. Finalize
   results from that exact record instead of mutable "last evaluation" state.
4. Isolate REML profile and CI evaluation from the returned fitted model. A
   profile/CI call must either use an isolated model or restore every mutated
   runtime and REML attribute. The final model must have consistent public and
   internal family, distribution, fit result, smoothing parameters, and caches.
5. Correct the all-invalid Wright-Bessel branch so every invalid term receives
   its saddlepoint fallback. Reject non-finite density objectives.
6. Propagate inner-phi convergence, evaluation count, and boundary status into
   the search trace/result diagnostics. A failed inner profile cannot be
   reported as a successful outer profile.

## Exact phi optimization

The MLE path will optimize `u = log(phi)` using an analytic first derivative of
the exact Tweedie log density. For positive observations on the Wright-Bessel
branch, use the Wright-function derivative identity

`d W(a, b, t) / dt = W(a, a + b, t)`

and the analytic chain rule for `t(phi)`. Zero-mass observations and
saddlepoint terms have closed-form scale derivatives. Sum these terms to
obtain the score with respect to `u`.

Use the previous candidate's MLE as the warm start while retaining a
data-derived mean-deviance/Pearson fallback start. Solve the bounded score root
when a sign-changing bracket is available; otherwise minimize the bounded
exact NLL. Every derivative result is checked for finiteness, objective
decrease, score tolerance, and boundary hits. The derivative-free bounded
optimizer remains a safeguarded fallback, not a separate statistical method.

Fixed `(y, mu, p, weights)` quantities are prepared once per phi profile so
repeated likelihood/score calls do not recompute masks, powers, canonical
terms, and weight transformations.

## Confidence intervals and reporting

- `result.ci()` is available only for exact MLE profiling. Pearson users receive
  a clear error directing them to bootstrap/sandwich inference.
- Profile plots use neutral labels for approximate methods and `MLE` only for
  exact MLE results.
- CI evaluations are cached but not included in the original search-evaluation
  count; expose a separate total/profile diagnostic so work is not hidden.
- REML mode is documented as a plug-in likelihood over REML-selected smooths,
  not as joint mgcv-style REML over `p`, scale, and smoothing parameters.

## Test strategy

Add failing regressions before production changes for:

- PIRLS `lambda2` forwarding and spline EDF/phi agreement with ordinary fit;
- offset-aware REML objective evaluation;
- grid and Brent final phi/NLL consistency with the exact winning record;
- complete model-state consistency after REML profiling and CI;
- zero, negative, NaN, and infinite sample weights;
- all-invalid Wright-Bessel fallback;
- inner-phi failure/boundary propagation and Pearson CI rejection;
- MLE default and lazy CI behavior;
- analytic score agreement with centered finite differences across zeros,
  positive values, weights, and saddlepoint fallback;
- analytic optimizer agreement with the derivative-free reference;
- p/phi recovery on unweighted, prior-weighted, zero-heavy, and flexible-spline
  simulations;
- a benchmark asserting fewer exact density evaluations than the current nested
  bounded search without weakening the objective or parameter tolerances.

Run focused tests during each red/green cycle, then Ruff, mypy for touched
modules, all non-slow tests, the full suite, and an explicit performance
comparison before completion.

## Non-goals

- Replace bounded Brent over `p` with a gradient optimizer.
- Claim that the current REML fit mode is identical to `mgcv::tw` joint REML.
- Change the EDM prior-weight convention to replication-frequency weights.
