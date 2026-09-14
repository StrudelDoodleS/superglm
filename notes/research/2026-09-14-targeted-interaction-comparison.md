# Targeted mixed-interaction controls

## Protocol frozen before fits

This follow-up tests whether a small, prior-informed mixed interaction menu
can improve on the additive models in the
[four-table pilot](2026-09-13-real-interaction-trials.md).
It does not test an automatic discovery algorithm. All resulting test
scores are exploratory audits: these test blocks have already been
inspected. They cannot serve as fresh confirmation of hypotheses chosen
after the pilot.

The [external evidence review](2026-09-14-interaction-dataset-evidence.md)
motivates Bike Sharing `hr × workingday`. The additional `temp × hr` term
is a domain hypothesis. Ames `Gr Liv Area × Bldg Type` is motivated by the
Kuhn/Silge example, but this trial keeps the original raw-price Gaussian
response and untransformed area; it is not a replication of their log-price,
log-area comparison.

The fixed menu has ten complete fits:

| Table | Parent spline k | Variants |
| --- | --- | --- |
| Bike Sharing | 4 and 6 | Additive; add `hr × workingday`; add that pair and `temp × hr` |
| Ames Housing | 4 and 6 | Additive; add `Gr Liv Area × Bldg Type` |

At each resolution the additive and interaction arms use identical parent
specifications. The current `SplineCategorical` implementation inherits its
parent spline geometry, so independent interaction knot changes would not
provide the intended control. Comparing each candidate with its additive
model at the same k distinguishes interaction value from changed main-effect
flexibility. Two fixed resolutions do not establish an optimal basis count
or certify approximation error.

Every arm uses `fit_reml(max_reml_iter=100)`, with the existing numerical
tolerances unchanged. The earlier pilot's limit was 20. This is a uniform
optimization-budget change for the new experiment, and previous failures
remain in their original receipts. Each worker has a 120-second deadline;
aggregate fit-worker wall time is capped at 600 seconds. Workers run
serially in fresh processes with BLAS, OpenMP and Numba threads set to one.
No timed GBM workers overlap this run.

The existing verified adapter supplies the exact full-table splits,
training-only preprocessing, category pooling (cap 32), feature exclusions
and response families. Bike keeps Poisson deviance; Ames keeps raw-price
MSE. There is no subsampling. Natural cubic spline parents use uniform
knots and SSP penalties; the other settings remain `selection_penalty=0`,
`discrete=True`, `n_bins=64`.

Only fits with both coefficient and REML convergence may compete. Primary
validation loss selects a variant within each k, and then across both k;
ties prefer fewer fitted coefficients. Before any new test evaluation,
the runner saves all choices and hashes all converged fit receipts.
It then evaluates only the validation winner within each k and its matching
converged additive control. A better test score never changes a saved choice.

Fit time includes complete fitting, while process time also charges loading,
preprocessing, diagnostics and serialization. Fit-end peak process RSS is
sampled before retained-storage traversal or prediction. Reachable retained
array/buffer payload is recorded separately from RSS. Receipts record actual
compiled dimensions, smoothing parameters, interaction classes, backend,
warnings and convergence telemetry, plus source, data, split and preprocessing
identities. Source files are frozen in `protocol.json` before fits; test
evaluation also checks the owned serialized model hash.

Runner: [benchmark_targeted_interactions.py](../../benchmarks/benchmark_targeted_interactions.py).
Artifacts will be written to the ignored directory
`.benchmark-artifacts/targeted-interactions/diagnostic-20260914/`.
The results below report all attempted variants, including failures,
and charge the full fixed-menu search separately from the selected model's fit.

## Results

The two-pair SuperGLM model improves Bike Sharing test Poisson deviance by
46.5% relative to its matching k=6 additive model: **60.5820 to 32.3932**.
Its complete fit takes **3.308 seconds versus 0.724 seconds**, or 4.57 times
the additive fit. The chosen terms are `hr × workingday` and `temp × hr`.
Both k resolutions select this pair set using validation alone. These are
measured results for one prior-informed menu, not evidence that automatic
discovery has been solved.

All ten fits converge under the unchanged numerical tolerances. None emits
a warning or reaches a deadline. The largest observed REML iteration count
is 17, so the new limit of 100 is not reached. Both k=6 additive models
exactly reproduce their earlier pilot test predictions. Thus the new bike
gain does not arise from the larger iteration allowance.

### Bike Sharing

Full hourly table: 17,379 rows; training 10,389, validation 3,502 and test
3,488, with chronological whole-day boundaries. Losses are mean Poisson
deviance on raw counts. Smaller is better.

| Parents | Added pairs | P | q | REML iterations | Fit seconds | Validation loss | Test loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| k=4 | None | 61 | 4 | 4 | 0.699 | 64.8163 | 61.2956 |
| k=4 | Clock | 84 | 4 | 3 | 0.824 | 41.1041 | Not evaluated |
| k=4 | Clock + temperature/hour | 153 | 27 | 11 | 3.666 | 37.4019 | 33.3504 |
| k=6 | None | 69 | 4 | 4 | 0.724 | 63.7515 | 60.5820 |
| k=6 | Clock | 92 | 4 | 4 | 0.942 | 39.8953 | Not evaluated |
| k=6 | Clock + temperature/hour | 207 | 27 | 8 | 3.308 | 35.8313 | 32.3932 |

Here Clock is `hr × workingday`, and temperature/hour is `temp × hr`.
P excludes the intercept; q counts fitted smoothing parameters. The clock
term adds 23 categorical coefficients and no smoothing parameter. The
temperature/hour varying-curve term adds 23 spline groups: 69 coefficients
at k=4 or 115 at k=6, and 23 smoothing parameters. Every model actually
dispatches to the Gram backend.

The one-pair model reduces validation loss by about 37% at 1.18 times the
additive fit time for k=4, or 1.30 times for k=6. It was not separately
evaluated on test because the frozen evaluation policy admits each k's
validation winner and its additive control. Its validation improvement
alone does not establish the single term's test gain.

The k=4 two-pair model has 54 fewer coefficients than k=6 and about 3% higher
test loss. It takes longer to fit in this observation because its REML
optimization uses 11 rather than 8 iterations. Smaller basis dimension
does not by itself determine complete-fit time. This is one reason to
measure both basis resolution and optimization work, rather than treating
PSST EDF or coefficient count as a timing law.

For the selected k=6 model, fit-end peak process RSS rises from 445.90 to
477.85 MiB (7.2%). Retained measured array/buffer payload rises from
1,719,359 to 4,328,853 bytes (2.52 times). The process's large fixed runtime
footprint makes the RSS percentage much smaller than the retained-model
percentage; report both. There are no unmeasured external buffers in these
payload records.

All six bike variants together use **10.163 seconds of model fitting**, or
14.04 times the k=6 additive fit, and **28.290 seconds of fit-worker process
time**. The latter includes process startup, loading, preprocessing,
diagnostics, validation prediction and model serialization. Four test audit
workers add 11.649 seconds. The selected 4.57-times fit ratio excludes this
search overhead. No timed fit was repeated, so these clocks are individual
observations on a shared host, not a stable benchmark distribution.

A descriptive post-selection audit divides the chosen k=6 predictions by
the five calendar-month blocks present in test. It changes no model choice.

| Test month | Rows | Additive deviance | Two-pair deviance | Reduction |
| --- | ---: | ---: | ---: | ---: |
| August 2012, partial month | 600 | 56.4546 | 30.0391 | 46.8% |
| September 2012 | 720 | 73.2009 | 31.7982 | 56.6% |
| October 2012 | 708 | 61.2622 | 30.0908 | 50.9% |
| November 2012 | 718 | 56.6307 | 35.2945 | 37.7% |
| December 2012 | 742 | 54.8494 | 34.2636 | 37.5% |

The improvement appears in every block, rather than being confined to one
month. These dependent time blocks do not supply independent replications
or a confidence interval. This audit is additional descriptive evidence,
not a formal guarantee that the interaction is useful on future data.

### Ames Housing

Full table: 2,930 sales; training 1,941 rows in 2006-2008, validation 648 in
2009, and test 341 in 2010. Every row is retained. Primary loss is raw-price
MSE; the displayed losses below are divided by one million for readability.

| Parents | Added pair | P | q | REML iterations | Fit seconds | Validation MSE / 10⁶ | Test MSE / 10⁶ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| k=4 | None | 259 | 23 | 7 | 2.055 | 527.783 | 504.087 |
| k=4 | Area/building type | 271 | 27 | 8 | 2.553 | 529.471 | Not evaluated |
| k=6 | None | 305 | 23 | 12 | 3.486 | 488.646 | 426.677 |
| k=6 | Area/building type | 325 | 27 | 17 | 5.461 | 492.615 | Not evaluated |

The mixed area/building-type term loses validation at both resolutions;
the selected model is the k=6 additive baseline. This rejects this candidate
under this response transformation, parent geometry and split. It does not
show that other Ames interactions cannot help. The source-supported log-area,
log-price comparison remains a distinct, unrun experiment.

All four Ames variants together use 13.555 seconds of fitting and 25.899
seconds of fit-worker process time. Its two test audit workers add 5.849
seconds. All models use the Gram backend. The selected additive model's
fit-end peak process RSS is 460.85 MiB and retained payload is 9,410,915 bytes.

## Interpretation alongside the GBM controls

The [matched GBM study](2026-09-14-gbm-interaction-comparison.md) obtains bike
test deviance 44.1068 for its additive class, 31.4800 for pairwise and 30.4920
for unrestricted, using validation-selected capacity within each class.
The selected two-pair SuperGLM deviance is about 2.9% above pairwise GBM.
The additive GBM also improves on additive SuperGLM: the total cross-model
gap cannot all be attributed to interactions.

Ames pairwise GBM improves on its own additive class, but its test MSE
457.102 million remains above the SuperGLM additive model's 426.677 million.
Both comparisons support continued interaction research; neither gives a
universal advantage to a model family.

The earlier bike screen considered only six numeric weather pairs and never
proposed hour-by-working-day or hour-by-temperature. The present result is
evidence that candidate coverage matters: a small set of suitable mixed
terms can give a large gain even when that earlier set worsens test loss.
The next discovery experiment should admit mixed/categorical candidates,
jointly check small pair sets and coarse/richer representations, and allow
the additive model to win. Fresh outer data is still required for
confirmatory accuracy or coverage claims.

## Receipts and verification

The [measurement record](2026-09-14-targeted-interaction-measurements.json)
contains all ten variants, six evaluated prediction summaries, validation
choices, cost totals, monthly audit, source identities and hashes of 68 raw
artifacts. The source/manifest/split/preprocessing identities match the
pilot and the GBM run. All saved model hashes and frozen choices were checked;
recomputing losses from the saved prediction archives gives the exact
reported values. Both additive k=6 prediction archives exactly replay the
pilot. All observed numerical thread pools contain one thread.

The suite exits successfully. Its UTC start/finish timestamps span 71.770
seconds; the sum of monotonic fit-worker process clocks is 54.189 seconds.
All 36 focused targeted, GBM and existing-adapter tests pass, as do Ruff
checking and formatting on all four new Python files. An in-memory mutation
removing the selection convergence guard admits an unfinished lower-loss fit;
the unchanged measured runner selects the converged control. Production,
dependencies and versions are unchanged. No numerical proof,
scaling law or out-of-sample error certificate is claimed by these trials.
