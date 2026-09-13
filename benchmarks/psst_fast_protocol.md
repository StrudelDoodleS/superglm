# FAST comparison on the fixed PSST study grid

Added after the normalization experiment, before observing FAST outcomes,
in response to the user's question about external comparators.

Use InterpretML 0.7.8 `measure_interactions` with its explicit default
hyperparameters: 64 bins, four samples per leaf, minimum Hessian 0.0001,
and no L1/L2/leaf-step constraint. Include a separate Purify variant by setting
the installed native `CalcInteractionFlags_Purify` bit; the public wrapper
does not expose that option. Instrument all 435 native calls in each arm.
An excluded pilot verifies the default wrapper against the unwrapped public
function and verifies link-scale initialization against SuperGLM predictions.

Reuse every dataset, feature specification, seed, split, default fitting
setting and failure rule from `psst_detection_protocol.md`. Regenerate the
same baseline fit and supply its link-scale predictions as `init_score`.
Use `rmse` for Gaussian and `poisson_deviance` for Poisson outcomes. Specify
numeric columns as continuous and categorical columns as nominal. Screen
exactly the same 435 candidate pairs.

The existing runner's `old` and `corrected` slots hold FAST default and FAST
Purify in this output directory; `fast-metadata.json` records the mapping.
Join records to the PSST run by dataset ID, and compare each FAST variant
separately to corrected PSST. Each has its own maximum-score null cutoff
from the same 200 calibration datasets and the same 100 independent audits.
Keep the existing paired and calibration-aware uncertainty calculations.

Each shortlist gets three single-pair SuperGLM refits and validation chooses
among those refits and the baseline before evaluating test data. This
compares screening utility for SuperGLM. It does not compare full EBM or GBM
fitting pipelines. The six cases were specified for the PSST correction;
they do not cover sharp step interactions, other sample sizes, sparse-count
insurance data, or every region in which FAST/EBM can perform well. Results
do not establish a universal ranking between methods.

No benchmark outcomes will change the feature menu, tuning, seeds or run
counts. A separate random shortlist has expected recovery 3/435 when one
pair is planted; no noisy Monte Carlo random baseline is needed for that
reference probability.
