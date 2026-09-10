# PR 381 review follow-up

The [Claude review](https://github.com/StrudelDoodleS/superglm/pull/381#issuecomment-5625838120)
and [Codex finding](https://github.com/StrudelDoodleS/superglm/pull/381#discussion_r3983515125)
led to these additional repairs.

## Correctness and ownership

`SparseSSPGroupMatrix` previously copied values for its Gram calculations but
read a different buffer for predictions and cross-products. It now owns one
float64 value buffer in `B`; the raw-array accessors forward to `B`'s current
buffers. Thirteen regressions fail on the previous implementation and cover
in-place changes, buffer replacement, ordinary and exceptional ranges,
prediction, transpose products, Grams, cross-products and row subsets.

The Pearson reporting recovery catches the binary-product helper's plain
`ValueError`, including invalid nonfinite factors. Six regressions cover
Gamma, negative binomial and Tweedie. The weighted natural-channel helper
preserves a NaN factor on a nonzero recovery row instead of producing an infinity
whose sign came from the NaN sign bit. Six further regressions cover both signs
in each factor position. Normal channels retain their original arithmetic.

The Python 3.12 and 3.14 CI failures were in an independent Gaussian profile
reference. It required remaining objective improvement below one rounded objective ULP.
The reference now polishes and certifies the analytic score residual, while
retaining the original public coefficients, Newton radius and profile error
bounds. Exact quadratic controls straddle an objective ULP; replacing the
Newton correction with zero makes both controls fail.

## Exceptional SSP work

One weight of `1e-40` in a 21,000-row, ten-column sparse group triggered
`Fraction` arithmetic for every row. The replacement exploits the fact that
each finite binary64 input is an integer times a power of two. Each operand
has a shared binary scale; all inner products and reductions use exact Python
integers, and only final outputs require rational-to-float rounding. The finite
input exponent range bounds integer widths apart from logarithmic growth with
reduction length. It retains one effective row and the requested output.

The arithmetic selector, tiny terms, CSR duplicates, signed cancellation,
bin aggregation and source validation remain intact. An explicit regression
shows why widening the admission interval is insufficient: a raw Gram can lose
a tiny positive direction before the transform cancels its large terms.

The isolated Gram takes 4.26138 seconds with the previous rational loop and
0.30740 seconds with exact integers, with bitwise-identical results. Ordinary
native arithmetic takes 0.000566 seconds on the corresponding ordinary-weight
fixture. The exceptional path is about 13.9 times faster but remains much
slower than native arithmetic. These are single-operation measurements.

Twelve new tests check realistic-size dispatch, independent rational answers,
the number of rational conversions, native dispatch, least-subnormal residuals
and inactive/nonfinite factors. The old implementation fails the three controls
that prohibit rational work inside row loops. Direct comparison against the
saved old implementation agrees on 24 successful moment calculations and 11
expected exceptions in the existing fixtures.

A complete public Gaussian spline fit with 21,000 observations and one weight
of `1e-40` confirms the effect. Three alternating pairs compare PR commit
`c55009f1` with the repaired source. Median fit time falls from 7.36657 to
0.54292 seconds; median process peak RSS changes from 385.00 to 384.71 MiB.
Every timed and profiled fit returns identical coefficients, predictions,
deviance, EDF and rank. All converge in two iterations. Separate profiles
record exactly one exceptional SSP moment calculation on each side.

Two review suggestions were not adopted. Array-identity memoization would miss
in-place structural mutation, as the existing stale-canonical-flag regression
demonstrates. Also, Gamma log-scale Fisher information tends to twice the
multiplier as CV tends to zero; collapsing CV alone does not establish the tiny
working-weight example proposed in the review.

## Complete-fit comparisons against the resumed baseline

Five alternating pairs for each case run in fresh processes with one native
thread. Other numerical jobs are paused. The fit clock includes first-use
work; disk compilation caches may be reused. Separate runs collect dispatch
and profiles. The baseline is source digest `167873e1`, which already includes
the original Gamma convergence repair.

| Complete fit | Fit time before | Fit time after | Peak process memory before | Peak process memory after |
| --- | ---: | ---: | ---: | ---: |
| Gaussian shared penalties | 0.0975 s | 0.1885 s | 336.4 MiB | 338.0 MiB |
| Gamma shared penalties, discrete | 0.4760 s | 0.6203 s | 377.9 MiB | 379.6 MiB |
| LSS tensor shared penalties, discrete | 0.7921 s | 1.4394 s | 397.0 MiB | 398.1 MiB |
| Scalar shared penalties | 0.1907 s | 0.2986 s | 376.8 MiB | 378.5 MiB |
| Scalar tensor | 10.2483 s | 11.4379 s | 406.6 MiB | 445.5 MiB |
| Inactive constrained QP | 0.0295 s | 0.0458 s | 336.7 MiB | 338.3 MiB |
| Binding constrained QP | 0.0351 s | 0.0470 s | 336.9 MiB | 338.3 MiB |
| Single SCOP | 0.0289 s | 0.0364 s | 336.2 MiB | 337.8 MiB |
| Joint SCOP, discrete | 0.2645 s | 0.3121 s | 377.7 MiB | 379.5 MiB |
| Sum-to-zero | 0.4325 s | 0.5089 s | 383.8 MiB | 388.0 MiB |

Each value is a median. The saved summary also records the five individual
times and median absolute deviations. Memory covers the whole Python process.

Both joint SCOP runs stop at the 40-iteration smoothing limit without smoothing
convergence. Every other case retains convergence, and all iteration counts
are unchanged. Each source's five repetitions have identical complete numerical
outputs. Between sources, the largest scalar prediction difference is
`7.76e-11`; the largest LSS parameter difference is `1.01e-10`. These comparisons
describe stable outputs, not coefficient-forward guarantees near rank loss.

The scalar tensor retains a 1.19-second cost, or 11.6%, and 38.9 MiB of peak
process memory over the resumed baseline. The remaining cases add between
0.0075 and 0.6473 seconds. These costs remain in the proposed repair; the
exceptional-arithmetic improvement does not remove them.

All 40 separate dispatch and profile runs reproduce every corresponding timed
output. None of these ordinary fixtures enters the exceptional SSP path.
Profiles locate the remaining work in certified penalty summaries and checked
factor construction. The QP cases each calculate two summaries, single SCOP
one, and joint SCOP two; reuse is active. The sum-to-zero fixture builds the
same 49 structured factors on both sides, with more checking per construction.
The scalar tensor calculates 16 summaries from 47 penalty-evaluation requests;
its diagonal Gram work falls from 7.67 to 0.94 diagnostic seconds. Profile
times identify work and are not additional complete-fit timing measurements.

The measured candidate source digest is
`58f5888d739400817473fa8ac175649f616a7674c033f6d23ed839286dbd274e`.
The final source differs only by a corrected comment in `_group_matrix_algebra.py`;
every parsed Python syntax tree matches the frozen measured candidate.
Its digest is `d2b7a5cc6aa5b70740aa537cb846e2f85bd5852d1a31ab63ee796d9bdb5ac19d`.

Receipts, raw profiles, source snapshots, comparison scripts and numerical
outputs are preserved under
`.superpowers/sdd/2026-09-10-numerical-resume/artifacts/pr381-followup/`.
The earlier rank-fixture and Gamma-book measurements retain their original
source attribution; this batch remeasures the ten cases listed above.

## CI and validation

One matrix now runs four duration-balanced shards on each of Python 3.12,
3.13 and 3.14, with at most four matrix jobs executing concurrently. Each
version's non-browser suite runs once. Coverage is collected during the 3.13
master run, and PR browser checks run only in Dev CI. Superseded PR runs are
cancelled. Existing required check names are preserved, and the aggregate
required check fails if any matrix job fails, is cancelled or is skipped.

The final local integration check passes 1,128 tests with four expected skips.
The corrected independent-profile file and new SSP regressions pass all 44
tests on both Python 3.12 and 3.14.
The CI contract and supply-chain checks pass all 32 tests; workflow syntax,
Ruff, formatting, lock and installed-dependency checks pass. These local checks
do not substitute for the subsequent GitHub matrix run.

The release advice is `release:minor`: the automatic LSS smoothing start changes
a public default. No version files are changed.
