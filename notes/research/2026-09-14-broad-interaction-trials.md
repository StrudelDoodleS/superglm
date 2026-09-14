# Broad interaction trials with one fixed procedure

The frozen procedure selected interactions with lower test loss on **10 of
12 real sources** and retained the additive model on two. The separate CASP
decoy control also improved. Each selected interaction group beat both the
validation-selected additive model and its matching-k additive control.
These are observed changes on one fixed split per source, not significance
claims or isolated evidence for every pair in a group.

The selected real-data fits cost 0.94–1.65 times their selected additive
fit. Searching for those models cost much more: 679.43 seconds across all
thirteen tables, followed by 85.51 seconds of evaluation worker time.
The run fits at most four pairs per model; it does not establish the cost
of hundreds of interactions or an optimal basis resolution.

The [measurement record](2026-09-14-broad-interaction-measurements.json)
contains all 100 fit outcomes, choices, data identities, proposal scores for
admitted pairs, timing, memory, dispatch and 552 raw-artifact fingerprints.
The [named reference bank](2026-09-14-known-interaction-candidates.md) supplies
the complementary published candidates, including exact Kaggle feature
constructions and their evidence limitations.
The [fitted interaction plots](2026-09-14-broad-interaction-plots.md) show
the selected Airfoil, Concrete and King County terms with training support.

## Observed test results

All losses below use the declared original-scale outcome: weighted mean
Poisson deviance for Abalone, Seoul and Metro; weighted MSE elsewhere.
Lower is better. `k4_s2` means parent resolution k=4 and the first two
admitted pairs. A/B values in the resource tables mean the selected
additive control followed by the validation-selected overall model.

| Source | Additive / selected arm | Additive test loss | Selected test loss | Reduction | Selected/additive fit time |
| --- | --- | ---: | ---: | ---: | ---: |
| Abalone | k4_s0 / k4_s1 | 0.386198 | 0.380678 | 1.43% | 1.46× |
| Concrete | k6_s0 / k6_s4 | 38.7687 | 32.3014 | 16.68% | 1.64× |
| Wine quality | k6_s0 / k6_s2 | 0.496582 | 0.490827 | 1.16% | 1.29× |
| Parkinsons | k4_s0 / k4_s0 | 155.232 | 155.232 | Additive retained | 1.00× |
| Airfoil | k6_s0 / k4_s2 | 21.7973 | 11.0627 | 49.25% | 1.31× |
| Power plant | k6_s0 / k6_s4 | 17.1077 | 15.8083 | 7.59% | 1.48× |
| Appliances | k4_s0 / k4_s2 | 19992.2 | 17144.1 | 14.25% | 1.34× |
| Seoul bike | k6_s0 / k6_s0 | 156.075 | 156.075 | Additive retained | 1.00× |
| Metro traffic | k6_s0 / k4_s1 | 1336.12 | 1333.74 | 0.18% | 1.04× |
| SGEMM | k4_s0 / k4_s4 | 67673.4 | 65135.9 | 3.75% | 0.94× |
| King County sales | k6_s0 / k6_s4 | 2.56133e10 | 2.11803e10 | 17.31% | 1.65× |
| Superconductivity | k6_s0 / k6_s4 | 231.256 | 223.330 | 3.43% | 1.33× |
| CASP decoy control, excluded from real-source count | k6_s0 / k6_s4 | 25.4002 | 23.3341 | 8.13% | 1.84× |

Airfoil's matching-k additive test MSE is 21.7822, giving a 49.21%
reduction. Metro's matching-k additive deviance is 1337.93, giving 0.31%.
The other cases use the same additive arm in both comparisons. Small
differences such as Metro's 0.18% are weak evidence without replication and
appropriate uncertainty estimates. SGEMM's slightly shorter selected fit is
a single observation, not evidence that adding interactions accelerates it.

### Exact pair groups selected by validation

| Source | Selected pairs, in training-proposal order |
| --- | --- |
| Abalone | `Shucked_weight × Shell_weight` |
| Concrete | `Cement × Age`; `Water × Age`; `Cement × Blast Furnace Slag`; `Cement × Water` |
| Wine quality | `volatile_acidity × alcohol`; `free_sulfur_dioxide × alcohol` |
| Parkinsons | None |
| Airfoil | `frequency × suction-side-displacement-thickness`; `frequency × attack-angle` |
| Power plant | `AT × V`; `V × AP`; `AT × RH`; `AT × AP` |
| Appliances | `RH_1 × RH_2`; `RH_3 × RH_8` |
| Seoul bike | None |
| Metro traffic | `temp × clouds_all` |
| SGEMM | `MDIMC × NDIMC`; `MWG × MDIMC`; `MWG × NDIMC`; `NWG × NDIMC` |
| King County sales | `sqft_living × grade`; `sqft_living × lat`; `grade × lat`; `grade × long` |
| Superconductivity | `wtd_gmean_ThermalConductivity × range_ThermalConductivity`; `wtd_gmean_Valence × wtd_entropy_Valence`; `wtd_entropy_atomic_mass × mean_ThermalConductivity`; `range_ThermalConductivity × wtd_gmean_Valence` |
| CASP decoy control | `F3 × F4`; `F4 × F5`; `F3 × F8`; `F4 × F8` |

Concrete's proposer independently admitted the water/cement variable pair
from the published hypothesis bank as its fourth candidate. This is useful
candidate recovery, but the result above is for four jointly fitted tensors.
It does not isolate that pair or test the source's ratio representation.

These results provide fixed local cases for the representation research.
They do not supply ground-truth interactions: a selected group can compensate
for restricted main effects, represent correlated proxies, or have a split-
specific benefit. In particular, k=4/6 is a limited main-effect menu. Numeric
parents remain linear; this matters for all-numeric SGEMM. Metro's contract
excludes `date_time` and adds no derived hour/day features, so its weather-only
gain should not be compared with a fully engineered traffic forecasting model.
Additive retention on Seoul or Parkinsons does not establish absent signal.

### Fit resources

Fit time is the complete `fit_reml` call, including its existing coefficient
fallback for q=0. Peak RSS is whole-process high water at fit completion;
retained payload is the separately traversed array/buffer storage. Interpreter,
imports and loaded data account for much of the RSS; payload is not a complete
process-memory estimate. Every completed model dispatches the `gram` solver.
Per-arm group-matrix classes are recorded in the measurement JSON, including
discretized tensors and mixed terms; SGEMM uses numeric interactions.

| Source | Fit seconds A/B | P A/B | q A/B | Peak RSS MiB A/B | Retained payload MiB A/B |
| --- | ---: | ---: | ---: | ---: | ---: |
| Abalone | 0.657 / 0.960 | 23 / 32 | 7 / 9 | 406.2 / 409.3 | 0.321 / 0.572 |
| Concrete | 0.647 / 1.064 | 40 / 140 | 8 / 16 | 403.4 / 411.9 | 0.176 / 1.243 |
| Wine quality | 0.926 / 1.191 | 56 / 106 | 11 / 15 | 419.1 / 431.3 | 0.759 / 1.794 |
| Parkinsons | 1.182 / 1.182 | 55 / 55 | 18 / 18 | 414.6 / 414.6 | 0.888 / 0.888 |
| Airfoil | 0.287 / 0.376 | 17 / 29 | 3 / 7 | 361.0 / 404.1 | 0.104 / 0.275 |
| Power plant | 0.524 / 0.773 | 20 / 120 | 4 / 12 | 402.9 / 410.9 | 0.485 / 2.919 |
| Appliances | 1.415 / 1.893 | 72 / 90 | 24 / 28 | 428.5 / 439.8 | 2.907 / 3.705 |
| Seoul bike | 0.606 / 0.606 | 66 / 66 | 8 / 8 | 418.7 / 418.7 | 0.850 / 0.850 |
| Metro traffic | 1.109 / 1.150 | 47 / 48 | 4 / 6 | 465.5 / 468.4 | 2.816 / 3.617 |
| SGEMM | 0.410 / 0.385 | 14 / 18 | 0 / 0 | 548.7 / 551.0 | 12.697 / 15.238 |
| King County sales | 1.436 / 2.368 | 100 / 200 | 13 / 21 | 463.3 / 477.5 | 3.007 / 6.237 |
| Superconductivity | 8.618 / 11.429 | 397 / 497 | 79 / 87 | 584.4 / 604.2 | 12.879 / 21.183 |
| CASP decoy control | 0.919 / 1.695 | 45 / 145 | 9 / 17 | 419.9 / 437.6 | 3.271 / 7.885 |

SGEMM supplies 241,600 source rows, of which 83,168 train this fixed
group-partitioned model; it is not a fit on 241,600 training rows.
Superconductivity supplies 81 raw predictors and 12,854 training rows, with
P=497 and q=87 in its selected model. These are observed workloads, not n/p
capacity bounds or scaling laws.

### Discovery and unsuccessful work are charged

Proposal time below includes the helper's native preparation, GBM fitting
and pair extraction. Fit sum includes all attempted SuperGLM fits, including
failures. Search workers include loading, setup, imports, proposal, fitting,
validation prediction, storage accounting and serialization. Evaluation
workers load the saved chosen models and produce their test predictions.

| Source | Proposal seconds | Sum of fit seconds | Search worker seconds | Evaluation worker seconds | All worker seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Abalone | 0.273 | 13.695 | 42.489 | 6.252 | 48.741 |
| Concrete | 0.158 | 15.662 | 43.541 | 5.955 | 49.496 |
| Wine quality | 0.294 | 9.103 | 37.620 | 6.103 | 43.723 |
| Parkinsons | 0.828 | 11.815 | 40.686 | 3.276 | 43.963 |
| Airfoil | 0.178 | 2.779 | 32.159 | 8.876 | 41.035 |
| Power plant | 0.363 | 5.384 | 33.953 | 6.358 | 40.311 |
| Appliances | 1.711 | 14.849 | 45.603 | 6.651 | 52.255 |
| Seoul bike | 0.493 | 13.292 | 40.374 | 3.478 | 43.851 |
| Metro traffic | 0.749 | 118.835 | 149.281 | 9.680 | 158.961 |
| SGEMM | 1.612 | 1.526 | 20.189 | 7.808 | 27.997 |
| King County sales | 0.991 | 14.570 | 44.560 | 6.805 | 51.365 |
| Superconductivity | 5.341 | 69.976 | 109.081 | 7.511 | 116.592 |
| CASP decoy control | 1.163 | 9.053 | 39.893 | 6.754 | 46.647 |

Total worker time is 764.94 seconds; suite wall time is 770.77 seconds.
The differences include orchestration and receipt-writing overhead. Most
eight-arm menus already cost roughly ten additive fit calls before discovery
and process overhead. Metro spends 118.84 fit seconds against a 1.109-second
additive fit, largely because three variants use all 100 REML iterations.
Cheap selected fits therefore do not yet meet a cheap end-to-end discovery
claim. This batch leaves the original numerical tolerances untouched.

## Failures and verification

All thirteen proposals succeed. Of 100 fits, 94 converge, four reach the
unchanged 100-iteration REML limit, and two raise numerical errors. Abalone,
Concrete and Metro retain `search_incomplete` status even though their
selected comparisons are evaluable. All 26 permitted evaluations succeed;
there are no timeout or budget-exhaustion records and no captured Python
warnings. The suite exits with status 1 because its entire search menu did
not converge.

The unfinished REML arms are Abalone `k6_s4` and Metro `k6_s1`, `k6_s2`,
`k6_s4`. Concrete `k4_s1` and `k6_s1` refuse a materially indefinite computed
data Gram during finalization. The
[numerical-failure memo](2026-09-14-broad-interaction-numerical-failures.md)
records the failure path and the missing construction-error certificate.
No failed arm produces a saved model, validation score or test evaluation.

The [audit script](check_broad_interaction_measurements.py) re-prepares all
data, verifies the frozen source, raw/data/split/preprocessing identities,
choices and hashes, and exactly replays all 26 recorded test scores from
the saved response, prediction and weight arrays. It verifies original test
row positions and labels against re-prepared data. Every reported worker
threadpool has one thread. All choices were persisted by
2026-09-14 00:08:08.342722 UTC; the first test worker started at
00:08:08.744399 UTC. An independent read-only review found no blocker in
the runner, integration, evidence claims or numerical-failure note.

Totals in the measurement record are independently summed from every process
receipt. The frozen runner assigns its optional `all_worker_process_seconds`
fields inside the evaluation loop, so other runs with no evaluable models
can omit that field. All cases in this batch have evaluations and raw totals;
the derived totals agree. The measured runner is preserved byte-for-byte.

The six related benchmark test modules pass **95 tests**. Ruff check and
format checks cover the new Python files and the audit script. No production
source, dependency or version changes, or full production-suite rerun, are
claimed for this batch.

Reproduce with the measured environment and local source corpus:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python benchmarks/benchmark_broad_interactions.py \
  --output .benchmark-artifacts/broad-interactions/new-run
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python docs/research/check_broad_interaction_measurements.py \
  --output .benchmark-artifacts/broad-interactions/replayed-measurements.json
```

The audit defaults to the archived `frozen-20260914` run and reproduces the
measurement JSON values. A new run has new timestamps, process receipts and
potential timing variation. These inspected test blocks now belong to
development evidence; later method changes require new confirmation data.

## Protocol before model fitting

The earlier corpus work established one SuperGLM interaction gain on Bike
Sharing. GBM gains on other tables were evidence for further investigation,
not additional SuperGLM successes. This batch tests a portable, bounded
proposal/refit procedure on twelve new real sources and one explicitly
separate protein-decoy control. It does not claim to know the true
interaction set of any observational dataset.

The [data protocol](2026-09-14-broad-interaction-data-protocol.md) fixes the
thirteen tables, source identities, predictors, original-scale outcomes,
group/time partitions, weights and eligibility rules. It contains 435,555
source rows and 434,820 retained rows. The CASP structural-decoy table is
excluded from counts of real observational/experimental sources. SGEMM has
241,600 source rows; superconductivity has 81 predictors. No source is
subsampled to meet an implementation budget.

This broad run addresses transferable discovery. The accompanying source
review addresses the user's distinct need for named candidate interactions
with prior evidence. A documented feature in a successful competition
pipeline is a useful reference candidate; without an isolated comparison it
is not a proven positive interaction label. Those references are not used
to change the broad run's candidate lists after seeing test results.

### Candidate generation and representation

For each dataset, a single histogram gradient boosting model is trained
using training rows and weights only. It uses pairwise interaction
constraints, 15 leaves, 200 rounds and the same fixed remaining settings
as the [GBM controls](2026-09-14-gbm-interaction-comparison.md). Gaussian and
Poisson losses match the declared task. There is no early-stopping split,
validation tuning or test input to the proposer.

For each non-leaf node whose root-to-node path uses exactly two distinct
original predictors, add its nonnegative split gain to that unordered pair.
Each such node contributes once. One-feature nodes contribute no pair score.
The extractor resolves scikit-learn's internal categorical-column reorder
through the fitted column transformer. Ties use original feature positions.

This score is a heuristic proposal. It can reflect main effects or
correlated proxies and is not a purified interaction statistic. Greedy
boosting can miss a balanced pure interaction when neither feature supplies
a useful first split. A generated balanced XOR fixture demonstrates that
failure; an AND fixture supplies the positive extraction check. Empty
proposals do not imply an additive data-generating process.

Scan ranked pairs and admit at most four under the richer k=6 resource
estimate. Skip unsupported spline/numeric combinations, oversized
representations and any later pairs beyond the count cap, recording reasons.
The current constructor supports spline/spline, spline/categorical,
categorical/categorical, numeric/categorical and numeric/numeric candidates.

At each k in {4,6}, fit the additive model and prefixes of one, two and four
admitted pairs. Prefix counts truncate to available pairs and duplicate
counts are removed. Both additive controls run first. All parents at one k
use identical specifications across its variants. Spline parents use natural
cubic splines, uniform knots and SSP penalties. Tensor marginals use the
same k as the parents; varying categorical curves inherit their parent
geometry. Other settings are `selection_penalty=0`, `discrete=True` and
`n_bins=64`.

When there are no spline parents, only one k label is fitted because the
models would otherwise be identical. SGEMM has zero smoothing components
under this feature contract. The existing `fit_reml` fallback fits its
coefficient model directly; there is no smoothing optimization to certify.

Let w_j be k-1 for a spline, L_j-1 for a categorical with L_j retained
training levels, or one for a numeric term. The nominal coefficient count
excluding the intercept is

\[
P=\sum_j w_j+\sum_{(j,l)\in S}w_jw_l.
\]

The nominal smoothing count starts at one per spline main effect. Each
tensor pair adds two; each spline/categorical pair adds L_j-1; the other
admitted types add none. Refuse P>512 or q>96. Generated mixed/tensor design
checks compare compiled dimensions with this admission calculation. These
are representation limits, not RSS upper bounds or statistical degrees of
freedom. Uniform k=4 and k=6 grids need not be nested, so this is a comparison
of two spaces, not a certified hierarchical approximation experiment.

### Selection and evaluation

Every SuperGLM arm gets `max_reml_iter=100` with default numerical tolerances
unchanged. A model is eligible only if both the coefficient fit and REML
report convergence when smoothing is required. With zero smoothing
components, coefficient convergence suffices and REML is marked not required.
Refused, failed, incomplete and timed-out models remain
in the records. No test prediction is produced from an unfinished fit.

Primary validation loss selects the model across the fixed menu, breaking
ties by smaller actual P then name. It independently selects the best
additive model across both k values. All datasets' choices and fit hashes
are saved before any final test evaluation. Evaluation reuses the owned,
hashed fitted models without refitting.

Evaluate the overall winner, the validation-selected additive baseline,
and the winner's matching-k additive model when different and converged.
The main comparison uses the best validation-selected additive baseline;
the matching-k comparison separates interaction value from a changed main
effect space. If the matching control is unfinished, record that comparison
as unavailable and exclude the case from claims of established interaction
gain. If neither additive control converges, there is no test evaluation.
The evaluator recomputes the allowed model set from frozen fit receipts.

Test loss is weighted MSE for Gaussian tasks and weighted mean Poisson
deviance for count tasks. Metro's repeated weather annotations each receive
1/count(timestamp), so one hour has total weight one. The same training
weights enter GBM and SuperGLM. All other source weights are one. Neither
small positive loss changes nor large ones are labelled statistically
significant without an uncertainty analysis respecting the group/time split.

For the report, distinguish selected interactions with lower test loss,
selected interactions with worse test loss, additive retention, and cases
without usable comparisons. Count independent real sources once. A partial
search is reported as partial even when a converged candidate can be scored.
These finite tables and one fixed split each do not establish a scaling law,
optimal basis count, causal interactions or universal predictive superiority.

### Time, memory and reproducibility

All model workers run serially in fresh processes with BLAS, OpenMP and
Numba thread counts one. Fit-worker deadlines are at most 120 seconds;
proposal deadlines at most 90 seconds; evaluation deadlines at most 60
seconds. The total worker-time budget is 1,800 seconds and the per-dataset
budget is 240 seconds, including proposal, fits and evaluations. Search is
restricted to 80% of the global budget and 75% of each dataset budget,
reserving time for evaluation. Each launch is capped by remaining budget;
termination/reaping can add small scheduling overhead.

Record complete model fit time, proposal time, all failed/search work,
validation/evaluation time and total worker process time. A selected fit's
ratio to its additive baseline excludes discovery overhead and is reported
separately from the full menu cost. Record fit-end peak process RSS before
storage traversal, retained array/buffer payload, actual P and q, dispatched
backend, warnings and convergence telemetry. These are single observations
on a shared host, not replicated speed estimates.

The new runner is
[benchmark_broad_interactions.py](../../benchmarks/benchmark_broad_interactions.py).
It freezes source identities and this menu in its machine-readable protocol
before fitting. Data, split, preprocessing and weight identities must agree
between proposal, fit and evaluation workers. No production, dependency or
version changes are part of this batch.

An integration fixture with a bilinear Gaussian signal reaches the unchanged
100-iteration REML limit while sending curvature penalties toward their
boundary. Its coefficient fit converges, but the REML projected score
does not satisfy its existing stopping rule. The failed receipt is preserved
at `.benchmark-artifacts/broad-interactions/generated-nullspace-refusal/result.json`.
The integration check verifies exclusion of unfinished fits and replays an
eligible additive model; it does not require every generated model to
converge or relax tolerances to make the test pass. This fixture is not part
of the real-data count.
