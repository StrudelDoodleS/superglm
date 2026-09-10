# Scalar working arithmetic and LSS variances

This is the final family-arithmetic addition to the convergence and rank
repair. The quick scan found representable Fisher weights and Pearson
contributions lost by intermediate products, plus four public LSS variance
methods with the same problem. It did not find a common failure across the
nine LSS fitting kernels. This change addresses those demonstrated cases.

## Scalar calculation

The generic Fisher expression is `w * (dmu/deta)**2 / V(mu)`. Multiplying by
the weight before cancelling factors can overflow or underflow even when the
answer is representable. The shared helper uses the following identities for
exact built-in family/link pairs:

| Family and link | Fisher weight |
| --- | --- |
| Gaussian, identity | `w` |
| Gamma, log | `w` |
| Poisson, log | `w * mu` |
| Poisson, square root | `4 * w`, including the limit at zero |
| Negative binomial, log | `w * min(mu, theta) / (1 + min(mu, theta)/max(mu, theta))` |
| Tweedie, log | `w * mu**(2-p)` |

For Binomial/logit, the fitted mean can be clipped independently of the
inverse-link derivative. Its calculation therefore retains that derivative
and evaluates `w * (d * (d / V))`. Alternate links and subclass overrides use
their actual derivative and variance. The alternate Binomial/log regression
demonstrates why the fitted mean cannot replace every log-link derivative.

Other family/link pairs retain the generic expression. Only entries with
unsafe intermediate products use the existing binary product/quotient
routine. Pearson calculations similarly reuse the existing scaled weighted
residual calculation, with family-specific recovery when an intermediate
variance is outside float64 range. Reporting retains its existing unfloored
variance convention; coefficient fitting retains its existing variance floor.

Coefficient fitting, REML reconstruction, shape inference, screening,
credibility reports and reported Pearson statistics use these shared
calculations. The random-effects score formula is unchanged.

## LSS calculation

Only the public variance methods for GaussianLS, GammaLS, LogNormalLS and
TweedieLSS change in this addition. Their ordinary vector calculations remain
in place. Entries with unsafe squares, powers or products use binary scaling;
the log-normal exceptional path also uses a log expression when its
exponential factor would overflow. True final overflow still returns infinity.
The fitting kernels, parameter bounds and optimizer stopping rules do not
change in this addition.

## Regression evidence and limits

The new scalar suite has 329 cases and the LSS variance suite has 43. References
use independent family identities evaluated on the represented inputs with
Decimal or high precision. Error allowances account for floating-point
roundings, gradual underflow and, where needed, transcendental conditioning.
The tests cover zero and extreme weights, custom family overrides, clipped
means, alternate links and true final overflow.

The original 318-case scalar scan failed 48 cases on the pre-addition source.
Expanded scalar tests subsequently reproduced five more failures. Four further
unfloored Pearson cases and the alternate-link compatibility regression were
also executed before their fixes. The LSS suite reproduced 17 failing cases
before its fixes. These are parameterized cases, not counts of independent
defects. The prior scan's 35 weighted likelihood-law comparisons and selected
kernel/oracle tests cover all nine LSS families; they do not certify arbitrary
inputs or every tail of every family.

## Reuse and performance method

The exact log-family pairs reuse the fitted mean as their derivative. The
negative-binomial calculation owns and reuses two temporary arrays. Existing
SCOP and fit-state caches remain in use. Working weights depend on the current
fit and do not gain a persistent cache with ambiguous invalidation.

`family_arithmetic_complete_fit.py` runs six 12,000-row complete fits with
frequency weights, spline and numeric terms. Each case has a 300-row warmup
and a fresh model for measurement. First use of any other backend remains
inside the measured fit. Each process runs the same six-case order, so RSS is
the process high-water mark through that case, not isolated per-family memory.
Five alternating source pairs compare this addition with the previously
accepted strict tensor candidate `c7ee4477`. Separate profiles record the
executed solver functions and callers. Profiling is excluded from the fit
timings. The existing 2,000-row tensor REML fixture is also compared with
three alternating source pairs at its explicit `1e-6` REML tolerance.

The [numerical audit](numerical_robustness_audit.md) retains the earlier
rank/penalty, mgcv and pyGAM comparisons and their narrower conclusions.

## Final measurements

The candidate source digest is
`9a813cd7ce25ffac15cd6041cd4c916f152ed56309ffce39c6fac62524ce4d0a`.
The baseline is
`c7ee44778b9bf8cd424b48a21dbca6ba6ffeb8c88ef982ec0f743d86668c7aaf`.
These comparisons isolate this addition, not the entire repair against master.

| Complete fit | Before, seconds | After, seconds | Change |
| --- | ---: | ---: | ---: |
| Gaussian | 0.80230 | 0.74814 | -6.75% |
| Gamma | 0.02659 | 0.02767 | +4.06% |
| Poisson | 0.02972 | 0.02789 | -6.14% |
| Binomial | 0.02427 | 0.02386 | -1.67% |
| Negative binomial | 0.02888 | 0.02914 | +0.89% |
| Tweedie | 0.09433 | 0.09608 | +1.86% |
| Scalar tensor REML | 11.71413 | 12.02343 | +2.64% |

All ordinary fits retain their iteration counts and `gram` backend, with no
fallback. Their largest prediction difference is `1.25e-14`; Gaussian and Gamma
predictions are identical. Median process high-water memory differs by less
than 0.12 MiB at each family checkpoint. These small fixtures do not establish
a general speed improvement.

All six tensor fits converge in twelve smoothing iterations with rank 255/255.
Maximum prediction difference is `4.60e-11`, and relative objective difference
is `4.13e-10`. Median peak RSS is 447.12 MiB before and 446.51 MiB after. The
2.64% timing difference is a small additional cost on this measurement; it does
not erase the earlier repair's measured cost against its older baseline.

All twelve ordinary profiles reproduce every corresponding timed result
exactly. Each executes one direct fit. Working-row call counts and backend
selection are unchanged; no exceptional binary-product repair runs on these
ordinary fixtures. Estimated-scale fits retain two Pearson consumers, the
solver's floored dispersion statistic and unfloored reporting. Routing both
through the shared helper does not add those passes: the old reporting path
calculated its expression directly.

The untimed tensor observer also reproduces all three candidate fits exactly.
It records 16 penalty summaries, 2,448 strict float64 slice products, 46 wide
products, twelve native factor proposals, 94 dense tensor Grams and 282
batched cross-Grams. Native pool samples report one thread and no observer
errors. This records executed operations; it does not intercept BLAS symbols.
The observer differs from the preserved earlier version only in the expected
module hash after an annotation-only correction.

The [receipt](family_arithmetic_receipt.json) contains source hashes, all paired
times, memory, numerical differences, profile counts and verification results.
Raw logs, XML, profiles, prediction arrays and dispatch evidence are preserved
under `.superpowers/sdd/2026-09-10-numerical-resume/artifacts/family-arithmetic-final/`.

## Final verification

Combined verification has 14,478 passes, including all 88 required real-data
cases, with 87 unique optional or inapplicable skips. The full non-browser run
initially had 14,474 passes and two failures. One dependency-table assertion
needed to include and check the new shared helper. The other required a
Tweedie fit to exhaust eight iterations, which the corrected working arithmetic
no longer does. Its replacement preserves real coefficients and KKT evidence,
pins the candidate's budget verdict, and proves by mutation that removing
certificate deferral refuses the fit. All 494 tests in the four affected files
then pass; both 16-worker controls also pass separately.

The only production changes after the full run are two verified annotation-only
corrections admitting the scalar and long-double values already accepted by
their numerical bodies. The independent reviewer accepted both test updates.
Ruff, formatting, lock/environment consistency and the public smoke fit pass.
The CI type check reports 890 diagnostics, below its existing 903-diagnostic
limit; this is backlog compliance, not a claim of a clean type check.
