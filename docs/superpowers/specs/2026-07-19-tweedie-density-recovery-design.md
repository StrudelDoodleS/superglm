# Tweedie Density Recovery Design

## Goal

Restore reliable Tweedie `p` and `phi` profiling without making ordinary fits pay
for unbounded exact-density work. Valid finite fits must not crash, and ordinary
fit time must remain close to PR #156.

## Evidence

- A 1,000-row `p=1.4` fit takes 0.198 seconds on PR #158 versus 0.011 seconds
  on PR #156. Post-fit likelihood statistics consume 94% of the PR #158 time.
- A 300-row MLE dispersion profile takes 1.584 seconds versus 0.004 seconds.
  Sixteen calls to the current series evaluator account for 1.578 seconds.
- Near-perfect fits pass `phi` near `2.6e-26` into fit statistics. The general
  series mode is then far beyond its 100,000-term cap, while SciPy's scaled
  Bessel evaluation for `p=1.5` returns `NaN` at the corresponding argument.
- PR #156 is exact where SciPy's Wright-Bessel evaluation succeeds and uses a
  saddlepoint fallback elsewhere. That fallback is accurate in many large-mode
  cases, but a neutral `p=1.05` reference demonstrates that it can be materially
  biased where exact series evaluation remains necessary.

## Chosen approach

Use one adaptive implementation. Do not add Fourier inversion or a user-facing
"thorough" mode.

### Exact general-power series

Replace the left-to-right sum with the Dunn-Smyth strategy for `1 < p < 2`:

1. Locate the contributing series mode from
   `log(j_max) = (log(t) - a*log(a)) / (a + 1)`.
2. Check the adjacent integer indices and anchor at the exact largest term.
3. Find lower and upper contributing bounds around that anchor. Terms outside
   the bounds must be at least 37 log units below the maximum, matching the
   published binary64 cutoff.
4. Evaluate only the bounded ranges, normalized by the maximum term, in ragged
   vector batches. Accumulate both mass and first moment so the analytic
   log-dispersion score remains available.

The work limits are internal safety policy, not public tuning parameters:

- at most 100,000 contributing terms for one row;
- at most 1,000,000 exact term evaluations in one density pass;
- at most 262,144 term elements in one allocation batch.

Rows beyond the exact-work budget use the diagnosed asymptotic fallback. They
must never make the entire density call raise or force feasible rows away from
their exact path.

### Density routing

For each positive observation:

1. Use SciPy Wright-Bessel when its result and the resulting log density are
   finite and positive.
2. At `p=1.5`, use the scaled Bessel identity. If SciPy's `ive` leaves its
   numerical range, evaluate the large-argument scaled-Bessel expansion in log
   space, including its corresponding score expansion.
3. At other powers, use the mode-centered exact series when it fits the work
   budget.
4. Otherwise use the stable saddlepoint density and report that row through
   existing `n_saddlepoint` diagnostics.

The extreme-mode fallback is acceptable because it prevents both unbounded work
and numerical failure. Exact neutral references remain the authority for deciding
whether the boundary is conservative enough for reliable `p` and `phi`.

### Fit statistics

Ordinary IRLS fitting does not require the exact density. Its fitted/null
statistics will:

- reuse one normalizer pass;
- use exact Wright rows where cheap;
- use the centered compound-Poisson series when its estimated work fits a
  strict fit-stat budget;
- otherwise use the stable saddlepoint base measure;
- permit mixed exact and saddlepoint rows instead of raising.

The saddlepoint base measure is independent of `mu`, so fitted/null reuse remains
algebraically valid. This restores PR #156's bounded fit-stat cost without changing
the fitted coefficients or dispersion calculation.

### Pearson stability

Compute Tweedie Pearson contributions as
`square((y - mu) / power(mu, p / 2))`. This avoids underflow in `mu**p` when the
mathematical contribution is finite. The public estimator and profile seed must
share this implementation; the private `mu >= 1e-10` floor is removed rather than
retained as a second definition.

## Error behavior

- Valid finite Tweedie inputs do not fail merely because an exact evaluator is
  outside its numerical or work range.
- Truly non-finite mathematical results remain non-finite and are diagnosed.
- Profile diagnostics distinguish exact from saddlepoint rows.
- No silent clipping of `phi`, `mu`, or response values is introduced.

## Verification

Test-first regressions will prove:

1. Near-perfect ordinary fits complete for representative powers on both sides
   of `p=1.5`.
2. The `p=1.5` large-argument density and score agree with independent
   asymptotic or finite-difference references.
3. The centered series matches existing neutral high-precision density and score
   fixtures while evaluating only terms near a distant mode.
4. MLE `phi` and joint `p`/`phi` profiles retain their independent reference
   tolerances and report approximation honestly.
5. Pearson dispersion returns zero for `y == mu == 1e-300` and the finite
   expected value for a `1e-310` residual.
6. The measured fit, profile, and likelihood reproductions are compared directly
   with PR #156. Any material ordinary-fit regression blocks completion.

## Excluded scope

- Fourier inversion or the orphaned certified density engine.
- A user-facing accuracy selector without evidence that it materially changes
  reliable `p` or `phi` estimates.
- Unrelated profile-optimizer, REML, or metaclass refactoring.
