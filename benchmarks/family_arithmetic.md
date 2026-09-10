# Scalar working arithmetic and LSS variances

The subsequent [PR review follow-up](pr381_review_followup.md) records the
reporting, SSP ownership and exceptional-arithmetic fixes made after this batch.

Intermediate products could overflow or underflow even when a scalar Fisher
weight, Pearson contribution or LSS variance was representable. This addition
to the convergence and rank repair fixes those cases. The scan found failures
in four public LSS variance methods; selected fitting-kernel checks passed
across all nine LSS families.

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

For Binomial/logit, clipping can change the fitted mean without changing the
inverse-link derivative. The helper uses that derivative and evaluates
`w * (d * (d / V))`. Alternate links and subclass overrides use their own
derivative and variance. The Binomial/log regression catches an incorrect
substitution of the clipped mean for the derivative.

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
These methods report moments; this addition leaves the LSS fitting kernels,
parameter bounds and stopping rules unchanged.

## Regression evidence and limits

The new scalar suite has 329 cases and the LSS variance suite has 43. References
use independent family identities evaluated on the represented inputs with
Decimal or high precision. Error allowances account for floating-point
roundings, gradual underflow and, where needed, transcendental conditioning.
The tests cover zero and extreme weights, custom family overrides, clipped
means, alternate links and true final overflow.

The scalar suite reproduces failures before each fix: 48 in the original
318-case scan, five in its first extension, four unfloored Pearson cases and
the alternate-link case. The LSS suite reproduces 17 failures before its fixes.
Several parameterized cases exercise the same defect. The earlier scan also
passed 35 weighted likelihood-law comparisons and selected kernel/oracle tests
across all nine LSS families.

The 88 real-data checks use freMTPL2. They do not measure the failure rate on
unrelated datasets. Shared arithmetic and rank corrections apply across
datasets, but selected kernel tests are not equivalent to complete real-data
fits for every LSS family. A fit can also have an unidentifiable or unbounded
optimum. In the reproduced weak-start Gamma case, the repaired solver reports
nonconvergence instead of accepting the unresolved score. Broader real-data
coverage remains a gap; the checks here cannot guarantee every failure will
be detected.

## Reuse and performance method

The exact log-family pairs reuse the fitted mean as their derivative. The
negative-binomial calculation reuses two temporary arrays. Existing SCOP and
fit-state caches remain in use. Working weights are recalculated when the fit
changes.

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

The timed candidate's source digest is
`9a813cd7ce25ffac15cd6041cd4c916f152ed56309ffce39c6fac62524ce4d0a`.
The baseline is
`c7ee44778b9bf8cd424b48a21dbca6ba6ffeb8c88ef982ec0f743d86668c7aaf`.
This comparison measures the addition against the accepted strict repair.

| Complete fit | Before, seconds | After, seconds | Change |
| --- | ---: | ---: | ---: |
| Gaussian | 0.80230 | 0.74814 | -6.75% |
| Gamma | 0.02659 | 0.02767 | +4.06% |
| Poisson | 0.02972 | 0.02789 | -6.14% |
| Binomial | 0.02427 | 0.02386 | -1.67% |
| Negative binomial | 0.02888 | 0.02914 | +0.89% |
| Tweedie | 0.09433 | 0.09608 | +1.86% |
| Scalar tensor REML | 11.71413 | 12.02343 | +2.64% |

All ordinary fits use the same iteration counts and `gram` backend, with no
fallback. Their largest prediction difference is `1.25e-14`; Gaussian and Gamma
predictions are identical. Median process high-water memory differs by less
than 0.12 MiB at each family checkpoint. These small fixtures do not establish
a general speed improvement.

All six tensor fits converge in twelve smoothing iterations with rank 255/255.
Maximum prediction difference is `4.60e-11`, and relative objective difference
is `4.13e-10`. Median peak RSS is 447.12 MiB before and 446.51 MiB after. The
2.64% timing difference is an additional cost on this measurement. The earlier
repair's comparison with its own baseline remains in the numerical audit.

All twelve ordinary profiles reproduce their timed results exactly. Each
executes one direct fit, with the same working-row call counts and backend.
No exceptional binary-product repair runs on these fixtures. Estimated-scale
fits have two Pearson consumers, the solver's floored dispersion statistic
and unfloored reporting. Routing both
through the shared helper does not add those passes: the old reporting path
calculated its expression directly.

The untimed tensor observer also reproduces all three candidate fits exactly.
It records 16 penalty summaries, 2,448 strict float64 slice products, 46 wide
products, twelve native factor proposals, 94 dense tensor Grams and 282
batched cross-Grams. Native pool samples report one thread and no observer
errors. The observer records executed operations without intercepting BLAS
symbols.
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

After the full run, two production annotations were corrected to admit the
scalar and long-double values already accepted by their numerical bodies.
The independent reviewer accepted both test updates.
Ruff, formatting, lock/environment consistency and the public smoke fit pass.
The CI type check reports 890 diagnostics against its existing limit of 903.

Python 3.12 CI then exposed a brittle replication-test control. It required
two equivalent fits to differ by more than arithmetic roundoff, although closer
agreement is valid. The replacement uses an analytic Gaussian likelihood
change between two candidates in the same certified local neighborhood.
Omitting the coefficient-movement term fails the control; the actual fit-parity
bounds are unchanged. All 55 tests in `test_distributional_efs.py` pass on
Python 3.12, and the affected test also passes on Python 3.13.

The later prose cleanup changes production comments and docstrings only.
Comparison of the executable syntax trees confirms no solver-logic changes.
The strict documentation build passes after correcting a benchmark link.
