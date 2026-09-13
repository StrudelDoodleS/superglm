# Standalone booster comparison

Written before the main booster outcomes, 2026-09-13. The user requested
XGBoost, CatBoost and LightGBM after the PSST/FAST experiment.

This first comparison measures prediction by standalone models. It does not
extract GBM pair rankings or evaluate their interaction-recovery rates.
Reuse the 600 preselected predictive datasets from the frozen PSST protocol:
six cases, five strengths, first 20 replicates each. Keep 4,000 training,
2,000 validation and 8,000 independent test observations and all seeds and
generating functions unchanged. Use an excluded six-case pilot first.

Each library gets three depth candidates, 2, 4 and 6. Learning rate is 0.05,
maximum boosting rounds 1,000, and validation early-stopping patience 40.
Choose depth using the same common validation loss used in the PSST study.
Each candidate predicts using its own best validation iteration. Freeze both
depth and iteration before generating or evaluating test observations.

Use XGBoost CPU 3.4.1, CatBoost 1.2.10 and LightGBM 4.7.0. Train with one
thread and each library's native categorical treatment. Category vocabularies
come from training inputs; never encode nominal labels as ordered numeric
predictors. Set Gaussian squared-error and Poisson objectives explicitly.
Convert raw Poisson predictions to means exactly once. Test both observed
loss and risk against the known generating mean, retaining their separate
Gaussian MSE and Poisson KL units.

XGBoost uses histogram trees. LightGBM sets num_leaves=2**depth as well as
max_depth, deterministic mode and column-wise histogram construction.
CatBoost uses CPU trees with its default categorical statistics. All other
regularization and sampling settings retain library defaults. Each library
uses seed 20260913+replicate. These settings do not equate model capacity or
compute budgets across the different tree algorithms.

An intercept fitted from the training response is an additional validation
choice. Fit all three candidates regardless of which wins. Record each fit,
its validation loss, best iteration, rounds trained and fitting time. A failed
candidate remains recorded and unavailable; the intercept remains available.
Test failure is an evaluation failure and never triggers reselection. Report
all failures, including datasets without paired usable test results.

Run each library/dataset job in a fresh spawned process. Record process peak
RSS, including imports and retained search models, and preprocessing plus
tuning and validation time. Record test-prediction time separately. These
are concurrent one-thread measurements, not dedicated-machine speed ratios.
The older SuperGLM worker RSS figures have a different process lifetime and
are not an equivalent per-job memory baseline. The three-depth search also
searches boosting iterations, so it is not a three-refit compute match.

Join by dataset ID to original additive, PSST-selected, FAST-default-selected
and FAST-Purify-selected SuperGLM test metrics. Require all 600 IDs for each
library. Report paired differences against PSST, per cell and per case.
For per-case averages over five strengths, resample the 20 replicate blocks
because those strengths share seeds. Never pool Gaussian MSE and Poisson KL.
Intervals are exploratory and pointwise, not simultaneous discoveries.

The six existing generators favor low-dimensional smooth additive structure
with one weak interaction. This is not a general-purpose tabular benchmark,
a heavily tuned competition, a sparse-count insurance study, or a comparison
on steps and higher-order interactions. No outcome will change this grid or
the chosen tuning search. Broader shapes and sample sizes require a separate
study. Record script, generator, protocol, source and prior-record hashes.
