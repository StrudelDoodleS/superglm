# GBM interaction comparison

Pairwise GBM improved test loss over additive GBM on all four datasets in
this bounded comparison. Bike's Poisson deviance fell from 44.11 to 31.48;
credit's log loss improved by only 0.86%. This supports investigating
interactions missed by the earlier spline-product pilot. It does not identify
which pairs SuperGLM should fit or show that its total gap to GBM comes from
interactions.

The experiment compares additive, pairwise and unrestricted histogram
gradient boosting on the same four datasets. The existing test outcomes have
already been inspected. Every result here is an exploratory diagnostic on
those splits, even though this runner selects models before its test workers.

## Observed results

All 24 fits completed their 200 rounds and all 12 selected-capacity test
evaluations succeeded. There were no warnings, errors, timeouts or retries.
Validation selected pairwise GBM with 15 leaves for breast, credit and Ames,
and unrestricted GBM with 31 leaves for bike. These choices remained fixed
after test evaluation.

Each cell below gives validation loss followed by test loss for that class's
validation-selected capacity. Ames MSE is divided by one million for display.

| Dataset and primary loss | Additive | Pairwise | Unrestricted | Class selected on validation |
| --- | ---: | ---: | ---: | --- |
| Breast, log loss | 0.15515 / 0.10209 | 0.11530 / 0.05440 | 0.11912 / 0.03539 | Pairwise |
| Credit, log loss | 0.44177 / 0.43033 | 0.43472 / 0.42662 | 0.43620 / 0.42665 | Pairwise |
| Ames, MSE in millions | 587.064 / 602.409 | 490.640 / 457.102 | 528.656 / 520.880 | Pairwise |
| Bike, mean Poisson deviance | 43.7646 / 44.1068 | 25.3159 / 31.4800 | 23.9055 / 30.4920 | Unrestricted |

The pairwise class's test improvement over the additive class was 46.7% for
breast, 0.86% for credit, 24.1% for Ames and 28.6% for bike. These are observed
loss differences on one reused split per dataset, with no uncertainty or
population-wide superiority claim. Breast has only 114 test rows. Its lower
unrestricted test loss does not change the validation choice.

Breast's test set contains 43 malignant and 71 benign cases, with prevalence
0.37719. Test average precision was 0.99004, 0.99893 and 0.99947 for additive,
pairwise and unrestricted GBM. Credit has 1,327 defaults and 4,673 nondefaults
in test, with prevalence 0.22117. Its average precision was 0.55334, 0.55734
and 0.56319. The receipt includes validation class counts, prevalence, ROC
AUC and all six validation scores per dataset.

The matching data, manifest-entry, split and preprocessing fingerprints agree
exactly with the earlier pilot for every fit. That makes the within-GBM
controls more informative than comparing unrestricted GBM with SuperGLM
alone. For example, bike's earlier SuperGLM additive test deviance was 60.58,
while additive GBM already reaches 44.11. The full difference therefore
cannot be assigned to interactions. Ames gives the opposite warning:
pairwise GBM's test MSE of 457.102 million still exceeds the earlier
SuperGLM additive result of 426.677 million. Credit's earlier REML fit did
not converge, so it provides no selected SuperGLM test baseline here.

Bike gives the clearest practical reason to revisit the pilot's candidate
space. Pairwise GBM improves substantially while retaining the original
categorical hour and calendar features. The earlier screen only admitted
products of four numeric weather predictors. This experiment does not rank
or attribute specific pairs. Unrestricted GBM further reduces test deviance
by 3.1% relative to pairwise GBM, but the selected capacities differ and this
small menu does not establish a need for higher-order effects.

## Observed cost

The 24 fit workers used 54.124 seconds of their 600-second aggregate budget.
The full suite, including 12 test workers, took 75.029 seconds. All model
fit clocks together sum to 8.452 seconds; imports, loading, preprocessing and
process startup explain much of the remaining time. The selected models'
fit clocks appear below alongside the cost of trying all six candidates.

| Dataset | Selected additive / pairwise / unrestricted fit, s | All six fits plus tuning and selection, s | Tuning, selection and three test evaluations, s |
| --- | ---: | ---: | ---: |
| Breast | 0.121 / 0.107 / 0.137 | 10.285 | 14.793 |
| Credit | 0.393 / 0.480 / 0.528 | 16.009 | 21.470 |
| Ames | 0.217 / 0.353 / 0.681 | 14.459 | 19.574 |
| Bike | 0.235 / 0.337 / 0.522 | 13.406 | 18.474 |

These are one-run observations on a shared host, not repeated speed estimates
or a SuperGLM performance gate. The GBM workers were serial and single-threaded.
The coordinating agent did light implementation and test work during this run.

| Dataset | Selected additive / pairwise / unrestricted fit-end peak RSS, MiB | Retained owner payload, MiB | Estimator pickle, MiB |
| --- | ---: | ---: | ---: |
| Breast | 207.87 / 207.75 / 207.68 | 0.332 / 0.320 / 0.288 | 0.357 / 0.345 / 0.313 |
| Credit | 241.02 / 240.59 / 240.99 | 0.255 / 0.363 / 0.362 | 0.283 / 0.391 / 0.390 |
| Ames | 224.19 / 224.21 / 226.36 | 0.254 / 0.366 / 0.771 | 0.286 / 0.397 / 0.803 |
| Bike | 228.49 / 228.83 / 228.89 | 0.350 / 0.398 / 0.889 | 0.377 / 0.425 / 0.917 |

The actual trees in every additive and pairwise fit used at most one and two
distinct predictors per branch, respectively. The selected unrestricted
models used up to 9, 10, 19 and 10 predictors per branch. Every fit contained
200 trees, with one tree per boosting iteration. Native categorical dispatch
was observed wherever categorical columns were present. The receipt retains
actual node and leaf totals, rather than inferring them from the leaf cap.

The [tracked measurements](2026-09-14-gbm-interaction-comparison-measurements.json)
bind the raw suite at
`.benchmark-artifacts/gbm-interactions/diagnostic-20260914/suite.json`, whose
SHA256 is `62cdbd67a57322b4f80b32d3728f2c6a234ed2e0d85896a140a6da8e7d272284`.
The measured runner hash is
`8df4dcc9e9600be64052f1d010748eedd309826b87ff59eb69ff962ad1c62dc0`.
Production source stayed at package hash
`7151ca3bdf15181144e935c937434a0924d1cce28940b3ef584f3a49a5bfcfb5`.
The existing runner, loader and storage helper were unchanged throughout.

## Preregistered comparison

The [runner](../../benchmarks/benchmark_gbm_interactions.py) uses the existing
[verified loader](../../benchmarks/interaction_datasets.py) and calls
`benchmark_real_interactions.prepare_dataset` and `transform_features`
directly. Predictor exclusions, full rows, split seeds, chronological and
group boundaries, numeric imputation and scaling, dropped columns, and
training category pooling therefore match the
[earlier pilot](2026-09-13-real-interaction-trials.md). The classifier models
malignant breast diagnosis or following-month credit default. Ames uses
Gaussian squared error on the original sale-price scale. Bike uses Poisson
loss on hourly counts.

Categorical columns keep the training adapter's exact labels, fallback and
final cap of 32 levels. The GBM input uses unordered pandas categorical
columns with `categorical_features="from_dtype"`. It does not use category
codes as ordered numeric predictors. Unseen held-out labels follow the
existing training pool or mode rule. Numeric columns use trees on the
adapter's standardized values; the spline basis from the SuperGLM pilot is
not part of the GBM representation.

The installed scikit-learn 1.9.0 signature and source were inspected before
implementation. Its supported `interaction_cst` values agree with the
current [official estimator documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html).
`"no_interactions"` restricts branches to one predictor, `"pairwise"` permits
two, and `None` permits unrestricted interactions. These are constraints on
the tree ensemble's linear predictor. Applying the binomial inverse-logit
or Poisson exponential link can create nonadditivity on the response scale.
The [official ensemble guide](https://scikit-learn.org/stable/modules/ensemble.html#interaction-constraints)
describes this structural restriction. The runner also inspects the actual
fitted branches and rejects a violated support limit. A branch using two
predictors does not itself establish that their statistical interaction is
useful.

Each structural class receives the same two capacities, `max_leaf_nodes=15`
and `31`. Every fit uses `max_iter=200`, `learning_rate=0.1`,
`min_samples_leaf=20`, `max_bins=255`, `l2_regularization=0`, `max_features=1`,
the fixed seed 20260913 and `early_stopping=False`. No training rows enter an
internal early-stopping holdout. The menu contains 24 fits across breast,
credit, Ames and bike. It is a small capacity check, not an exhaustive GBM
tuning exercise. Equal hyperparameter budgets do not make the statistical
capacity of the three structural classes equal.

Validation primary loss first selects one capacity within each class, then
selects the class overall. Zero interactions can win. Exact ties prefer the
smaller capacity; class ties prefer additive, then pairwise. All four
datasets' choices and the fit-receipt hashes are persisted before any test
evaluation. Only the selected capacity from each class receives a test
worker. The fitted training models are reused without refitting. Test
results cannot change the persisted choices.

Every worker runs in a fresh process with one thread per observed numerical
thread pool. Fit workers have a 120-second deadline and share a 600-second
aggregate process-time budget. Evaluation workers have 30-second deadlines.
Timeouts, errors and budget exhaustion remain in the receipt. The cap
includes process startup, loading, preprocessing, fitting, validation and
serialization. Termination and reaping can add small scheduling overhead
after a deadline. Fitting all three classes' full menus is required for a
complete comparison.

## Measurement scope

The fit clock encloses `model.fit` only. Loading, preprocessing, model setup,
validation prediction and serialization have separate clocks. The receipt
also reports the sum of all six fit workers per dataset, each class's two
fit workers, and total tuning, selection and evaluation time. The selected
model's fit time is not the cost of finding that model. HistGradientBoosting
reports completion of the fixed 200 rounds, not an optimization convergence
certificate comparable to SuperGLM's REML stopping checks.

Fit-end peak RSS is the process high-water immediately after fitting. It
includes imports, source verification, raw data, preprocessing and fit
allocations. The shared storage helper counts reachable NumPy and byte-buffer
owner payloads once. It omits Python-object overhead and unknown extension
allocations. Pickle size describes the fitted estimator only; the shared
adapter state appears separately in JSON and is required to replay raw
inputs. These memory measures have different scopes.

Per-arm source identities bind the runner, existing adapter, loader, storage
helper, SuperGLM package source and installed sklearn implementation. Data,
manifest entry, split, preprocessing, model pickle and test-prediction hashes
bind the remaining artifacts. Models and predictions remain under the ignored
`.benchmark-artifacts/gbm-interactions/` directory. The tracked receipt contains
measurements and provenance, not raw observations or fitted model artifacts.

The [focused tests](../../benchmarks/test_gbm_interactions.py) check identical
splits and preprocessing, native categories fitted from training only,
matched menus, actual branch constraints, validation-only selection with an
additive winner, immutable selection receipts, workload caps and a complete
fit-to-evaluation replay on a pinned generated CSV. The first 13 tests failed
against the empty runner before implementation. The row-budget regression
failed before the pre-load admission check was added. These fixtures are
contract checks, not real-data evidence. An additional in-memory mutation
removed the fitted-status selection guard. The selector test then failed by
admitting a timed-out additive candidate, without changing the measured
runner file. The 15 new tests and 17 existing adapter tests pass on the
unchanged runner; both new Python files pass Ruff checking and formatting.
