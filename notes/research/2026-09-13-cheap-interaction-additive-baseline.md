# Cheap interactions relative to an additive fit

Date: 2026-09-13. Production source is unchanged from `03766e8d`.

The research objective is many useful interactions for a modest multiple of
additive-model fit time, with measured peak/retained memory and prediction
quality. Let `T0` be the complete additive fit on the same rows and parent
specifications. The user's examples motivate 2T0, 5T0 and 10T0 plotting guides;
they are exploratory budgets, not a universal tenfold guarantee.

The [budget derivation](2026-09-13-cheap-interaction-budget.md) and
[representation analysis](2026-09-13-cheap-interaction-representations.md)
describe the hypotheses. This note supplies an initial measured baseline.
The [real-data corpus](2026-09-13-interaction-dataset-corpus.md) and
[Kaggle corpus](2026-09-13-interaction-kaggle-corpus.md) broaden validation beyond
this one smooth synthetic response law. They are a prerequisite for choosing
a generally useful method; dataset availability is not benchmark performance.
The [real-data scaling design](2026-09-13-real-data-scaling-design.md) separates
row count, predictor count and compiled model size for the larger tables.

## Matched additive comparison

The unchanged
[many-interaction runner](../../benchmarks/benchmark_many_interactions.py)
was run at `M=0,8,16,28`, with two fresh single-threaded worker processes per
case and smoothing mode, reversing the count order in the second pass. All
sixteen fits converged with the `gram` backend. The data have 2,048 training
rows, eight independent uniform features, a fixed 1,024-row test set, eight
cubic-regression spline main effects (`k=6`) and up to 28 tensor interactions.
Each tensor has 25 coefficients. The response has the same main effects and
all 28 separable smooth pair signals in every case, plus Gaussian noise of
variance 0.09. Omitting terms changes the statistical model.

The table uses the ratio of two-run median fit times. Fixed smoothing is
compared with fixed-smoothing additive fitting; REML is compared with REML
additive fitting. The additive medians are 0.17718 s and 0.36633 s respectively.
Model specification and imports precede the fit clock; the complete public
fit call, including construction and finalization, is timed. Test evaluation
and artifact export follow it. The process high-water is sampled before those
post-fit operations. No owned tests, profiles, other fits or data parsing ran
concurrently with the timing workers. Other host users remain uncontrolled.

| Interactions | Coefficients excluding intercept | Fixed fit / T0 | REML fit / T0 | REML time (s) | REML test MSE | REML peak RSS (MiB) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 40 | 1.00 | 1.00 | 0.366 | 0.74072 | 383.04 |
| 8 | 240 | 1.53 | 3.33 | 1.220 | 0.58046 | 398.75 |
| 16 | 440 | 2.09 | 7.66 | 2.806 | 0.42329 | 428.99 |
| 28 | 740 | 4.14 | 11.37 | 4.164 | 0.11046 | 500.78 |

At M=28, REML adds 117.74 MiB to peak process RSS. Retained NumPy/byte owner
payload grows from 329,992 to 27,212,116 bytes (about 0.315 to 25.951 MiB).
The process-memory ratio is therefore much less dramatic than the retained
model-payload ratio; the interpreter and libraries already occupy substantial
memory. Both measurements are needed.

This demonstrates an approximately 11T0 regime for these small full tensors,
not cheap arbitrary-rich interactions. T0 is subsecond and includes fit-time
runtime/cache-loading overhead. Wider bases, larger row counts, correlated
features, different response families and different interaction structure can
change the ratio. Two runs are not enough for confidence intervals or a
scaling exponent. The test set has now been inspected for research decisions;
future confirmatory comparisons require fresh evaluation data/splits.

## Smaller ordinary tensors as a control

The runner now supports `--interaction-k`, leaving additive `k` fixed. It uses
the existing tensor `n_knots` option through the established interaction
registration path. The optional benchmark argument introduces no production
feature or solver change.

Eight additional converged fits compare all 28 interactions at marginal k=4
and k=5 (nine and sixteen coefficients per tensor). Each case again has two
fresh workers. The additive specifications, rows, seed, smoothing settings
and default stopping tolerances stay fixed.

| Coefficients per tensor | Total coefficients | Fixed fit / T0 | REML fit / T0 | REML time (s) | REML test MSE | REML peak RSS (MiB) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 9 | 292 | 2.80 | 6.79 | 2.487 | 0.10478 | 406.91 |
| 16 | 488 | 3.25 | 7.54 | 2.760 | 0.10655 | 442.23 |
| 25 | 740 | 4.14 | 11.37 | 4.164 | 0.11046 | 500.78 |

In this particular smooth signal, the smaller ordinary tensor has lower test
loss and lower fit cost. That is a useful control against which to evaluate a
new method: it does not establish that choosing small tensors always works.
The smaller models are separately built spline spaces. Their knots need not
be nested in the rich model, their normalized penalties differ, and their
REML problems differ despite all having 64 smoothing parameters. These results
are not certificates of equivalence to the rich problem and are not results
for the proposed nested product dictionary. The coarse controls ran after the
matched ladder; ratios to T0 are descriptive comparisons on this shared host,
not simultaneously bracketed measurements of every coarse case.

## What the next representation must improve

A useful adaptive method must beat or complement this simple low-resolution
control across datasets: retain a small set of products where adequate, add
resolution where it improves the joint fit, and account for selection and
validation costs. The proposed first dictionary retains a fixed rich basis
and pulls back its actual penalty, so numerical fidelity can be assessed
against a well-defined target. Global shared factors offer another route to
cheap aggregate all-pairs effects, under a low-rank completion assumption and
a different nonconvex mean class. Neither is implemented by this control.

Real trial data must include housing, credit/default, claims severity, fraud,
small classification controls and larger mixed/high-dimensional tables. On
imbalanced classification, record log loss, calibration and precision-recall
performance with prevalence; accuracy alone is insufficient. Use the appropriate
deviance/loss for count and positive-response families. Preserve chronological
and group boundaries where the data require them. Repeated targets or splits
from one table are not additional independent datasets.

The immediate same-model performance hypothesis remains reuse of unchanged
Gaussian tensor cross-products across coefficient-solver calls. This can be
tested against exact numerical replay without conflating it with changing
the interaction function class. The representation experiments and broad
real-data corpus proceed alongside it.

## Reproduction and validation

[The receipt](2026-09-13-cheap-interaction-measurements.json) contains all 26
worker records: 24 converged fits and two failed coarse-harness setups. The
first setup attempted to pass unnamed low-level tensor specifications through
a constructor requiring named declarations. It failed before fitting; the
corrected harness uses existing `_add_interaction` registration. Both failed
logs are retained and excluded from fit statistics.

[collect_cheap_interactions.py](../../benchmarks/collect_cheap_interactions.py)
rebuilds the receipt from `.benchmark-artifacts/cheap-interactions/`. It verifies
common production/input hashes and exact repeated coefficients, predictions,
non-timing telemetry and retained owner payload for all twelve paired cases.
The current collector also requires matching recorded runtimes for the
repetitions and additive comparisons, and matching resolved direct backends
within each repetition pair. See the [pre-merge validation record](
2026-09-14-interaction-review-validation.md#pre-merge-review-and-portable-timing-evidence)
for the portability corrections and replay against the original receipts.
Raw commands, source/runner hashes, losses, convergence, dispatch, memory and
artifact hashes remain available in the receipt. Different runner hashes
distinguish the original matched ladder and optional-k control. Exact local
repeatability is not a portable floating-point tolerance.

At the measured checkpoint, an independent reconstruction matched the complete
receipt exactly and verified all 100 raw artifact hashes. That checkpoint's
production fingerprint matched every measured fit. The benchmark runner and
collector passed focused Ruff and format checks.
The production source and dependencies are unchanged. No full production test
rerun is claimed for this research-only change.
