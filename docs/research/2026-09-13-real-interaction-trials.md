# Real interaction trials

The first four-table pilot does not establish an interaction accuracy gain.
Three of seven fits stopped at the existing REML iteration limit. The only
dataset with two converged arms, bike sharing, selected interactions on
validation but had worse final chronological test loss. These are full-table
trials with an intentionally simple discovery control.

The objective is to discover useful interactions cheaply, including how many
deserve a joint refit. This runner does not solve that selection problem.
Its marginal shortlist and cap of eight are a crude control. It compares the
additive model with one fixed candidate set, which contains six pairs for
bike sharing. It neither estimates the true interaction count nor searches
over candidate-set sizes. No claim depends on fitting every possible pair.

The [runner](../../benchmarks/benchmark_real_interactions.py),
[17 focused tests](../../benchmarks/test_real_interactions.py) and
[tracked measurements](2026-09-13-real-interaction-trials-measurements.json)
record the protocol. The full local receipt is
`.benchmark-artifacts/real-interactions/pilot-20260913/suite.json`.
Its hash, commands, per-arm source identities and test-prediction hashes are
in the tracked measurements. Production source was
`03766e8d5ce5b67a2ce1091ab5ae92cabf9076c3`, with package source hash
`7151ca3bdf15181144e935c937434a0924d1cce28940b3ef584f3a49a5bfcfb5`.
The measured runner hash is
`d1fd02b97005d0a71c094dec05488343f5a712c821ae7866d5227d4ec1a16b13`.

## Data and training contract

The runner reads the verified raw tables through the
[public registry](../../benchmarks/interaction_datasets.json) and
[loader](../../benchmarks/interaction_datasets.py). It uses their target,
predictor exclusions, categorical declarations, grouping and split seed.
Byte and schema verification can inspect the whole raw table. Model choices
use training and validation only. Every source row belongs to exactly one
partition, with no smoke subsampling.

| Dataset | Target and family | Train / validation / test rows | Split |
| --- | --- | ---: | --- |
| Breast cancer | Malignant diagnosis, binomial | 341 / 114 / 114 | Target-stratified ID groups, 60/20/20 |
| Credit default | Following-month default, binomial | 18,000 / 6,000 / 6,000 | Target-stratified client ID groups, 60/20/20 |
| Ames housing | SalePrice on its raw scale, Gaussian | 1,941 / 648 / 341 | 2006–2008 / 2009 / 2010, disjoint PID groups |
| Bike sharing | Hourly count, Poisson | 10,389 / 3,502 / 3,488 | Chronological whole-day groups, 60/20/20 |

Breast cancer is a historical diagnostic control. Credit has one historical
cohort. Bike weather is contemporaneous information, so this is not a
forecast evaluated with weather known in advance. The registry excludes
bike's component counts, Ames transaction metadata and the diagnostic or
client identifiers from predictors.

Every arm learns the same deterministic preprocessing rules on training rows:

- Numeric columns reject infinity, impute missing values with the training
  median, then center and scale using the imputed training values. An
  all-missing or constant training column is dropped. A constant column in
  training cannot re-enter because it varies in a later partition.
- Declared categorical semantics take precedence over numeric dtype. Other
  string or categorical columns also use categorical encoding. The adapter
  ignores unobserved levels carried in pandas categorical dtype metadata.
  Levels with fewer than `max(2, ceil(0.01 * n_train))` rows pool. At most
  32 observed levels remain, including a pool when needed. This preserves
  all 24 bike-hour levels. Unseen or missing held-out labels map to the
  training pool, or the training mode when no pool existed. Real labels
  have a separate prefix from the missing and pooling tokens.
- Remaining numeric columns with at least ten distinct observed training
  values use natural cubic splines with `k=6`, uniform knots and the existing
  SSP penalty. Other numeric columns use a standardized linear term.
  These cutoffs and category rules are modelling heuristics.

No columns happened to be dropped in these four training partitions. Actual
feature kinds and observed category counts appear in the measurement file.
The runner rejects changed input-column order or membership. Preprocessing,
splits, source bytes and the selected manifest entry have per-arm hashes.

Both arms use `selection_penalty=0`, `discrete=True`, `n_bins=64` and the
unmodified `fit_reml` defaults. In particular, `max_reml_iter=20` and the
existing numerical tolerances were not changed. Each added tensor uses
`n_knots=(2, 2)`, giving two centered three-column margins and nine
interaction coefficients. All main effects are refitted jointly with the
interactions. The additive `k=6` margins remain unchanged.

The dimensions, bin count, knot placement and screening policy do not certify
approximation error. A convergence flag here means the existing fit and
REML stopping checks passed. It is not a statistical calibration guarantee,
a global-optimum guarantee or a certified out-of-sample error bound.

## Discovery control and its cost

The additive worker predicts its own training rows to form response
residuals. It ranks spline-eligible predictors by absolute training marginal
correlation with the response, retaining at most twelve. It then scores the
at most 66 centered bilinear products by absolute correlation with the
centered additive residual. It proposes at most eight pairs for one joint
tensor refit. No test values or validation residuals enter this ranking.

The shortlist can exclude a strong interaction whose margins have weak main
effects. The bilinear score can also miss nonlinear pure interactions.
Correlated predictors can produce redundant proposals. Categorical and
mixed interactions are outside this control, even when their corresponding
main effects are present. The cap therefore says nothing about how many
useful interactions exist.

Bike's four eligible spline features were `temp`, `atemp`, `hum` and
`windspeed`. The adapter retained categorical hour, month and other calendar
main effects at the final cap of 32, but never proposed hour-by-weather or
other mixed calendar interactions. It also had no cyclic interaction
representation. This periodic and mixed-feature blind spot is separate
from the weak-main-effect blind spot.

| Dataset | All raw-feature pairs | Eligible spline pairs | Shortlisted features | Pairs scored | Pairs proposed for one refit |
| --- | ---: | ---: | ---: | ---: | ---: |
| Breast cancer | 435 | 435 | 12 | 66 | 8 |
| Credit default | 253 | 91 | 0 | 0 | 0, screening fit did not converge |
| Ames housing | 2,701 | 253 | 12 | 66 | 8 |
| Bike sharing | 66 | 6 | 4 | 6 | 6 |

Generation, shortlisting and product scoring share the `screening_seconds`
clock. They are not individually timed. Training prediction has its own
clock because discovery needs it. The interaction pipeline also pays for
the preceding additive fit. Sharing that fitted baseline between the two
reported arms does not make its cost free.

| Dataset | Training prediction for screening, s | Shortlisting and scoring, s | Joint refit / additive fit | Additive fit + prediction + screen + joint refit / additive fit |
| --- | ---: | ---: | ---: | ---: |
| Breast cancer | 0.0040 | 0.0016 | 2.17 | 3.18, failed refit |
| Ames housing | 0.0410 | 0.0023 | 1.88 | 2.89, failed refit |
| Bike sharing | 0.0249 | 0.0007 | 3.18 | 4.21 |

These ratios exclude loading, preprocessing, validation prediction,
serialization and process startup. They describe work spent on the measured
attempts. A failed refit is not an interaction model delivered within that
cost multiple. The receipt separately records each of those clocks and
whole-process time. Additive process time itself includes screening and
export, so it is not a clean baseline for a full-pipeline cost ratio.

## Observed fits and losses

`P` is the compiled coefficient count excluding the intercept. `q` is the
observed number of fitted smoothing parameters. The values below come from
the fitted model, rather than the raw column count.

| Dataset and arm | P | q | Fit, s | Fit-end peak RSS, MiB | Retained payload, MiB | REML result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Breast, additive | 150 | 30 | 1.673 | 402.19 | 1.22 | Converged, 10 iterations |
| Breast, 8 interactions | 222 | 46 | 3.637 | 419.80 | 2.63 | Iteration limit, 20 |
| Credit, additive | 104 | 14 | 2.794 | 438.15 | 4.23 | Iteration limit, 20 |
| Credit, interactions | | | | | | Refused because additive screening fit did not converge |
| Ames, additive | 305 | 23 | 3.826 | 461.16 | 8.97 | Converged, 12 iterations |
| Ames, 8 interactions | 377 | 39 | 7.205 | 482.50 | 13.85 | Iteration limit, 20 |
| Bike, additive | 69 | 4 | 0.798 | 446.98 | 1.64 | Converged, 4 iterations |
| Bike, 6 interactions | 123 | 16 | 2.533 | 459.02 | 4.08 | Converged, 11 iterations |

Every completed coefficient solve reported convergence. The three failures
were REML `max_reml_iter` terminations, and all four converged fits reported
`score_objective_tolerance`. No warnings were emitted. Stopped arms were
excluded from validation selection and final test evaluation. No failed
case was retried with a larger budget or a changed tolerance.

Every fit actually used the `gram` direct backend. The observed groups were
`DiscretizedSSPGroupMatrix`, plus `CategoricalGroupMatrix` where needed,
`DenseGroupMatrix` for Ames linear numeric terms, and
`DiscretizedTensorGroupMatrix` for added tensors. Thread-pool receipts
record one thread. Merely requesting `discrete=True` is not the dispatch
evidence.

The runner uses the housing benchmark's retained-storage helper. Retained
payload counts reachable NumPy and byte-buffer owners once, including the
full owners of views. It omits Python-object overhead and unknown extension
allocations. Fit-end RSS is the process high-water immediately after the fit,
before storage traversal, prediction or export. It includes imports and raw
data. These two memory measures describe different quantities.

Validation chooses the converged arm with lower primary loss. The runner
writes that choice and hashes the fit receipts before launching separate
test-evaluation workers. Both converged bike arms are then evaluated, while
the already chosen arm stays fixed.

| Dataset and arm | Validation primary loss | Final test primary loss | Other final test metrics |
| --- | ---: | ---: | --- |
| Breast, additive | Log loss 0.93073 | Log loss 0.46044 | Average precision 0.96280; prevalence 0.37719; ROC AUC 0.98084 |
| Ames, additive | MSE 488,646,459 | MSE 426,676,939 | MSE uses squared original price units |
| Bike, additive | Mean Poisson deviance 63.75153 | 60.58204 | MSE 20,876.62 |
| Bike, 6 interactions | Mean Poisson deviance 62.15325 | 100.82707 | MSE 21,639.10 |

Bike validation selected the six-interaction model. Its final test deviance
was worse, and its MSE was also worse. This is a direct failure of the
pilot's validation choice to carry over to the final period. The breast
additive fit had training deviance about 0.000297 but appreciable held-out
log loss. Passing the numerical stopping checks did not establish good
probability calibration.

The four cases took 54.188197 seconds from suite start to finish. Fit-worker
process time was 43.206240 seconds in total, below the 600-second budget.
Each fit had a hard 75-second whole-worker deadline, including startup and
parsing. Test workers had 30-second deadlines. There were no timeouts.
The suite correctly exited with status 1 because some arms did not converge.
There were seven actual fits: four converged and three did not reach REML
convergence. One further planned arm, credit interactions, was refused
because its screening fit did not converge. Four test-evaluation workers
completed after the corresponding validation choices.
One attempt per arm on one host is not a stable timing estimate or a
cross-dataset ranking.

## Staged large-n and large-P work, not launched

This pilot observed at most 18,000 training rows, 377 compiled coefficients
and 46 smoothing parameters. It does not establish large-n or large-P
performance. Raw predictors are not interchangeable with compiled
coefficients: Ames has 74 predictors but 305 additive coefficients, whereas
credit has 23 predictors and 104 coefficients.

The current runner admits at most 300,000 source rows and an estimated
512 coefficients by default. It checks the registry row count before
parsing and the actual row count afterward. It checks the representation
estimate before fitting, then records actual compiled P and q afterward.
It does not implement a hard RSS limit or a generic compile-only stage.
Its explicit family adapters cover these four tables and Gaussian SGEMM;
unsupported family or split policies fail rather than being guessed.

The next stages should preserve complete real tables and start with a
separate, bounded compilation receipt for every newly adapted cohort.
Actual P, q and dispatch should determine admission to complete fits.
No stage below was run in this pilot.

| Stage | Candidate real data | Prospective admission and unresolved work |
| --- | --- | --- |
| More rows at modest P | SGEMM, 241,600 rows; ULB fraud, 284,807 rows | Start with P at most 512 and q at most 64. SGEMM has an explicit group policy; ULB needs its Kaggle schema and chronological adapter integrated. Verify minority-class counts in each declared split. |
| More coefficients at modest n | Superconductivity, 21,263 rows and 81 raw predictors; communities, 1,994 rows and 118 predictors | Adapt their target and groups, compile first, and initially admit P at most 640 and q at most 128. These are admission caps, not compiled counts. |
| More than half a million rows | YearPredictionMSD, 515,345 rows and 90 predictors | Its provided train/test boundary needs a separate train/validation policy. All-continuous k=6 encoding plus eight tensors would have 522 coefficients before any constant removal. Review actual dispatch and memory before admitting a fit. |
| Around two thousand coefficients | CT slices, 53,500 rows and 384 predictors | An all-spline upper representation estimate is 1,992 coefficients with eight tensors. Compile actual feature kinds, groups and q first. Do not infer that REML is affordable from the coefficient count alone. |

One dense float64 square matrix needs roughly 2, 8 and 32 MiB at P of 512,
1,024 and 2,048 respectively. Those are single-array sizes, not process
memory bounds. For example, a full 1,000,000-by-512 float64 design alone
would be about 3.81 GiB. Discrete groups can avoid that particular array,
while retaining other row data, dense factors and smoothing workspaces.
Some derivative paths also carry q-dependent dense storage. The
[existing scaling analysis](2026-09-13-many-interaction-scaling-analysis.md)
describes these obligations; the small pilot does not measure their limits.

The much larger TLC and city-temperature tables still need task, target and
cohort adapters. Their existence is not a fit-performance result. The next
discovery experiment should test interaction-set size and include proposals
that can find weak-margin and mixed interactions, while charging candidate
generation, scoring and the final joint refit separately. That work belongs
to the discovery design, not a claim that the fixed top-eight control is
already adequate.

## Verification and identity limits

The focused tests cover training-only imputation and feature choice,
categorical-universe leakage, pooling, finite values, column contracts,
complete group/time splits, a product-signal fixture, validation selection
and the persisted choice before test evaluation. A small real SuperGLM fit
checks the joint two-margin plus tensor coefficient contract. Seventeen
tests passed, and Ruff check and formatting checks passed.

A temporary mutation that admitted nonconverged candidates into validation
selection made the corresponding test fail by choosing the wrong arm.
The row-budget test also failed before the pre-parse guard was added. No
repository source was changed for the mutation run.

Workers captured package, runner, helper, source-byte, manifest-entry, split
and preprocessing hashes directly. They did not capture the loader's source
hash separately before each fit. The archive records its after-run hash and
21:23:45 UTC modification time, before the 21:27:26–21:28:20 fit window.
The dataset agent confirmed that its last changes were formatting and a
categorical-declaration validation guard, with unchanged parsing, and that
it made no loader edits during the fits. This supports the audit without
turning an after-run observation into a claimed per-worker capture.
