# SuperLSS convergence repair

The original Gamma failures have separate causes. The shared coefficient solver
could report a stationary fit when it had stalled. The smoothing loop also
rejected harmless exact-face projection roundoff. Independently, the common
starting lambda of 0.1 ignored the units and scale of each penalty.

This work starts at `origin/master`, commit
`66141d2873afc03287ca924f4592e33482281a0e`, in the isolated
`fix/lss-convergence-repair` worktree. No likelihood, response, version or
publication changes are included.

## What failed

The training book contains one claim at vehicle age 69, with no age between 39
and 69. The training-plus-validation book adds one claim at age 84. Uniform
cubic spline knots give the model directions that can isolate these tail rows.

At an exactly fitted Gamma row, log likelihood grows approximately as
`-log(CV)`. Moving log CV down by `t` along a penalized direction contributes a
linear likelihood gain and a quadratic penalty cost. The recorded penalty
quadratic at lambda 0.1 was about `1.63e-5`. Its conditional optimum can require
tens of thousands of log-CV units. A positive penalty alone does not make that
mode numerically representable.

A high-precision spot check agreed with the production Gamma scores at the
collapsed row. The inspected joint Laplace objective had the expected likelihood,
penalty and determinant terms. Adding a global dispersion or dividing the
objective by sample count would change the statistical problem.

The stopping bugs made this harder to diagnose:

- Small objective and coefficient changes could certify convergence without
  a small score or a valid Newton-decrement bound.
- A failed terminal retry replaced the coefficients but could inherit an
  earlier successful convergence flag.
- Exact-face revalidation required bitwise-identical coefficients. A stationary
  zero-iteration refit still computes `Q @ (Q.T @ beta)`, which can change stored
  coefficients by rounding. The BOHB failure involved movement of `6.94e-18`.

## Repair

`fit_reml(initial_lambda=None)` now chooses data-scaled starting penalties. For
each penalty it profiles the unpenalized directions out of the initial Fisher
information, then uses the largest supported generalized eigenvalue. Before
caller bounds are applied, no supported penalized mode starts with more than
half a degree of freedom. This criterion is invariant to within-term basis
changes and penalty units. Literal row replication scales the starting lambda
by the replication count.

An earlier average-half-EDF candidate fixed the training split but failed on the
larger book. The modewise criterion prevents one nearly unpenalized direction
from being hidden by the average. This is a starting policy, not a guarantee of
a unique REML optimum. Explicit numeric starts, per-component values and fixed
policies keep their meaning. Families without expected information, or blocks
with unresolved numerical support, retain the bounded 0.1 fallback.

Coefficient stopping now requires retained stationarity evidence and carries
the retry's verdict with its state. Numerical decrement checks must agree
between iteration stopping and final result validation. Their bounds include
linear-solve error, not just rounding in the final dot product.

Endpoint revalidation permits a changed coefficient vector only when it is the
zero-iteration replay of the exact face projection and lies within the
dimension-derived projection error bound. Stationarity, rank, provenance,
objective and endpoint-direction checks remain in place. Genuine state-change
refusals retain their reason.

## Real-data results

All responses are positive individual claims divided by 1,000, without clipping.
The policy-disjoint split is fixed with seed 20260909. Every retained row and
all listed features stay in each fit.

| Fit | Master result | Repaired result |
| --- | --- | --- |
| All nine features in both predictors, 15,846 training claims | Failed; rank 107/146; relative score `1.95e14`; CV collapsed to `5.4e-17` | Practical convergence; rank 146/146; score `3.12e-11`; minimum CV 0.52808 |
| All nine features in both predictors, 21,124 training and validation claims | Not separately timed on master | Practical convergence; rank 146/146; score `1.10e-10`; minimum CV 0.22713 |
| BOHB selection, 21,124 claims; all nine mean features, driver age and vehicle power in scale | `endpoint_revalidation_failed` | `objective_plateau`; certified exact face; score `1.82e-16` |
| Caller-fixed lambda 0.1, all training features | Incorrect coefficient success despite unresolved score | Returns nonconverged, `line_search_failed` |

The BOHB exact face removes six vehicle-age wiggle directions, leaving rank
82 in the 88-coefficient representation. This is intentional, not accidental
rank loss. The repaired successful dense fits use observed curvature without
Fisher fallback.

Wall time is unmeasured for these original diagnostic runs, which used one
native thread under concurrent machine load. Training process peak RSS was
482.1 MiB before and 501.9 MiB after; full repaired peak RSS was 519.8 MiB.
These process high-water marks include imports and input preparation. The
scripts record the actual backend and native thread pools; controlled final
measurements use separate fresh processes on a quiet machine.

## mgcv comparison

mgcv 1.9-3 fits the same nine-feature cubic-spline function spaces successfully.
Both its default bounded Gamma scale link and the unbounded log-dispersion
parameterization converge on the training data. The unbounded version also
converges on the full book without warnings. Its full-book average negative
log-likelihood is 1.663441, compared with 1.663395 for SuperLSS.

For the full book, relative RMS differences against that mgcv fit are 0.157%
for mean and 0.276% for CV. These averages do not establish tail equivalence.
The largest individual CV difference is about 27.5%; minimum fitted CV is
0.31348 for that mgcv run and 0.22713 for SuperLSS. Neither run proves branch
uniqueness. Their stopping criteria also differ.

mgcv's default scale link has a lower dispersion bound, but the unbounded
comparison shows that this bound is not required for these successful fits.
Its data-aware initialization and convergence methods are documented in
[initial.sp](https://stat.ethz.ch/R-manual/R-devel/library/mgcv/html/initial.sp.html),
[gammals](https://stat.ethz.ch/R-manual/R-devel/library/mgcv/html/gammals.html) and
[gam.convergence](https://stat.ethz.ch/R-manual/R-devel/library/mgcv/html/gam.convergence.html).

Stable REML computation is established work. Relevant methods appear in
[Wood, 2011](https://doi.org/10.1111/j.1467-9868.2010.00749.x) and
[Wood, Pya and Säfken, 2016](https://arxiv.org/abs/1511.03864). The latter covers
general smooth models, including location-scale-shape models. The automatic
start in this repair follows the established need to account for data and
penalty scaling. Its largest-eigenvalue criterion is our chosen starting
policy, not a claim to reproduce mgcv's initializer or inherit a convergence
guarantee from those papers.

## Relation to the roadmap

The [C3 dossier](../docs/research/2026-09-superglm-feature-roadmap-dossier.md)
identifies reliable stationarity and stable penalty calculations as numerical
requirements. The [C3 follow-through](../docs/research/2026-09-pragmatic-convergence.md)
already distinguishes useful practical stopping from certified smoothing
stationarity and records a prior coefficient-stagnation problem in NB2.
This repair strengthens the shared coefficient stopping and retained-evidence
checks, fixes Gamma initialization, and admits bounded exact-face replay.

The [roadmap additions](../docs/research/2026-09-superglm-feature-roadmap-additions.md)
also propose Fisher-based preconditioning and an adaptive Newton controller.
This repair does not implement those research items. The reproduced failures
have narrower fixes, and the successful runs do not establish robustness for
every family, model or stationary branch.

## Verification and reproduction

The new real-data suite derives the three failing configurations directly from
the parquet book. It failed on unmodified master, then passed all three cases
on the repair. The Real data workflow includes it with the dataset skip guard.

```bash
SUPERGLM_REQUIRE_DATA=1 SUPERGLM_DATA_DIR=/path/to/data \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest tests/test_distributional_fremtpl_convergence.py -q
```

`benchmarks/lss_convergence_repair.py` records complete-fit receipts and
predictions. `benchmarks/lss_repair_mgcv.R` records the independent R fits.
The current receipts, predictions and `model.diagnose()` reports are under
`/tmp/superglm-lss-repair.EJW6bI/`.

To rerun the unbounded mgcv comparison on the same prepared CSV:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  Rscript benchmarks/lss_repair_mgcv.R /path/to/data.csv /path/to/results unbounded full
```

The original affected distributional suite passed 2,172 tests, including all
three guarded real-data cases. Two opt-in performance tests skipped and one
slow test was deselected. Final scoped review of coefficient stopping and
endpoint revalidation approved the repairs. A separate numerical audit has
found additional shared-decomposition, scalar stopping and multi-penalty
defects. Their repairs now have independent review and focused regression
evidence, including a complete scalar tensor fit that preserves support
through terminal coordinate reconstruction. The subsequent frozen non-browser
suite passed 13,941 tests with no failures and all 88 required real-data checks.
Two worker-limit skips passed separately with 16 workers: combined verification
has 13,943 passes and 87 optional/inapplicable skips. Controlled final
performance checks remain in progress; these results do not establish
general numerical robustness.
