# Real-data discovery, row and coefficient scaling

Date: 2026-09-13. This is an experiment design and dimension calculation, not
a measured fit limit or an estimated scaling law.

The [public corpus](2026-09-13-interaction-dataset-corpus.md) and
[Kaggle corpus](2026-09-13-interaction-kaggle-corpus.md) now include large and
wide tables. Their manifests pin source bytes, target columns, predictor
exclusions and split policies. A verified raw table still needs an appropriate
modelling adapter before it supports a performance or accuracy claim.

The user clarified the workload: cheaply find a small, unknown number of
valuable interactions among many possible pairs, then fit the survivors
jointly. A table with 384 predictors presents 73,536 possible pairs; fitting
73,536 tensor terms is not the proposed experiment. Candidate discovery and
deciding when to stop are primary costs and modelling decisions.

## Separate the three sizes

Use `n` for eligible training observations, `d` for retained input predictors,
and `P` for compiled coefficients excluding the intercept. Distinguish the
potential pair count `U=d(d-1)/2`, candidates actually scored `C`, candidates
receiving a more expensive score/refit, and interactions retained `s`. Existing
fit receipts call the last quantity `M`. Also record smoothing parameters `q`,
penalty nullity `k0` and graph degrees. A raw table's column count is not a coefficient bound.
Category pooling, constants, spline widths and interaction widths change `P`.

The following are verified raw dimensions and the predictor lists currently
declared in the registries. They are not completed fit dimensions.

| Source | Raw rows | Declared predictors | Split constraint |
| --- | ---: | ---: | --- |
| Berkeley Earth city temperature | 8,599,212 | 5 | Shared chronological cutoffs across locations; 364,130 target values are missing. |
| NYC TLC yellow taxis, January 2024 | 2,964,624 | 8 | Later-day holdout after declared timestamp and fare eligibility rules. |
| YearPredictionMSD | 515,345 | 90 | Official 463,715-row training prefix and 51,630-row test suffix. |
| ULB credit-card fraud | 284,807 | 29 | Contiguous time partitions; report fraud counts and prevalence. |
| SGEMM | 241,600 | 14 | Hold out tiling configurations; other measured runtimes are excluded predictors. |
| BlogFeedback | 60,021 | 280 | Official historical training file, February validation and March test files. |
| CT slices | 53,500 | 384 | Keep each of the 74 patients in one partition. |
| Superconductivity | 21,263 | 81 | Keep repeated chemical formulas together. |
| Communities and Crime | 1,994 | 118 | Hold out states; exclude identifiers and specified concurrent outcomes. |

The [core registry](../../benchmarks/interaction_datasets.json) and
[Kaggle registry](../../benchmarks/interaction_kaggle_datasets.json) give exact
source URLs and schemas. Temperature and taxi records need additional
eligibility and feature derivation. Their raw row counts must not be reported
as fitted row counts. YearPrediction's artist-separated outer split does not
provide artist IDs for a certified artist-disjoint inner validation split.

## Discovery and selected-model costs

Report the complete procedure relative to the matching additive baseline:

\[
T_{\mathrm{procedure}}=T_0+T_{\mathrm{propose}}+
T_{\mathrm{score/refine}}+\sum_{h\in\mathrm{tried\ models}}T_{\mathrm{joint\ fit},h}
+T_{\mathrm{validation}}.
\]

Record preprocessing, loading and output costs as well, with the clock scope
explicit. Reusing an additive fit can save actual work, but cannot make its
cost disappear from interaction discovery. A cheap final fit is insufficient
if candidate generation or trying many model sizes dominates the procedure.

The first discovery comparison should use all retained predictors without a
hard main-effect significance gate. Marginal screening can discard both
variables of a strong pure interaction. Cheap residual-product or multiscale
summaries may rank a bounded candidate set, after which existing
nuisance-adjusted score tests and joint refits can assess survivors. Their
numerical screening error and their statistical usefulness are different
questions. No unproved guarantee of finding every useful pair is assumed.

Treat the retained count as adaptive: compare a small predeclared path of
candidate budgets or use declared joint regularization, allowing zero retained
interactions. Use training for proposals and validation for count/stopping
selection. Final test data stay untouched until that choice is fixed. A fixed
top-eight cap does not estimate the number of useful interactions. Correlated
predictors can also make the identity or count of a predictive representation
nonunique.

The [discovery memo](2026-09-13-cheap-interaction-discovery.md) specifies the
next bounded prototype and its mathematical obligations.

For a hypothetical model with `d` nondegenerate numeric main splines of
effective width five and `s` selected small tensors of width nine,

\[
P=5d+9s.
\]

One explicit float64 `P` by `P` matrix then occupies `8 P^2` bytes, before
factors, copies, basis data, trace work or runtime overhead. This is the cost
of that dense representation, not a lower bound for every solver.
For example, `d=384, s=20` gives `P=2,100` and 33.65 MiB for one such matrix.
These are conditional dimensions, not an actual fit or its RSS: categorical
predictors, constants and chosen widths change the calculation. Even when
few interactions survive, the additive model itself may be expensive.

There is also an identification condition. Under independent centered cubic
blocks with the usual second-derivative penalty, positive smoothing in both
tensor directions and no penalty on the remaining linear directions, each
main spline contributes one penalty-null direction and each tensor contributes
one linear-by-linear direction. Thus `k0=d+s` before a separately handled
intercept. The
[positive-definiteness argument](2026-09-13-many-interaction-scaling-analysis.md)
requires `k0 <= n_positive` for a strictly positive-definite coefficient
system. The condition is necessary, not sufficient; correlations can create
additional rank loss. Centering against a fitted intercept can tighten the
available observation rank further.

This condition applies to the selected model, not the universe of unbuilt
candidates. It does not require materializing candidate tensors. For reference,
the earlier counterfactual all-pairs calculation at `d=384` gave `P=663,744`,
one dense matrix of 3,282.40 GiB and `k0=73,920`. That is an illustration of an
unwanted workload, not a target or a barrier to discovering a small useful set.

## Bounded experiments

1. Establish reliable full-split additive and small-interaction controls on a
   few modest tables first. Charge train-only preprocessing, feature and pair
   selection, all fits and validation. Save convergence, actual backend,
   eligible row counts, `P`, `q`, fit-end peak RSS and retained model payload.
2. For row scaling, freeze the predictor list and interaction specification.
   Use declared nested subsets of the training partition only, for example
   10,000, 50,000, 100,000 and the full available training size. Preserve groups
   and chronological boundaries. Keep validation and final test fixed. A
   subset is a scaling experiment, not a full-book fit or a new dataset.
3. For discovery scaling, fix training rows and grow the predictor universe
   with predeclared irrelevant, correlated and signal-bearing features. Record
   candidates proposed, scored, refined and retained, discovery RSS/time and
   held-out gain. On synthetic controls, measure missed known interactions,
   including weak-main-effect and localized alternatives. On real data,
   compare independent held-out gain; an exhaustive fit is not an oracle.
4. For coefficient scaling, fix the training rows. Grow a predeclared predictor
   prefix through 8, 16, 32, 64, 128 and all available columns, then vary the
   interaction count and width separately. Column order must not depend on
   final test performance. Measure the realized `P`; do not substitute `d`.
5. Use bounded selected-interaction counts. Compare spread and hub-heavy
   graphs at matched `M` and `P` in controlled fits, since
   shared-parent assembly has a separate cost. Check projected allocations
   and nullity before starting each larger dense model.
6. Use fresh single-threaded workers, serial timed fits and hard deadlines.
   Repeat only promising completed cases for timing stability. Retain timeout,
   refusal and nonconvergence receipts as outcomes; do not relax stopping
   tolerances to turn them into successes.

The initial real-trial implementation is deliberately narrower than this
matrix. Larger-row and wider-model trials follow validated adapters and
bounded pilots. No scaling exponent follows from a few timings or from the
dimension table. A claim of cheap interactions must report held-out loss and
the full cost relative to its own additive baseline, including discovery.
