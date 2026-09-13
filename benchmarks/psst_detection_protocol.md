# PSST shortlist study protocol

Written before the study outcomes, 2026-09-13. The user approved the
simulation and fixed-refit-budget comparison described in the conversation.

## Question and comparison

Does replacing `2*edf` with the Gaussian-reference variance improve recovery
of a planted interaction or the predictive value of a three-pair shortlist?
Both scores use every rung of the same public screen and the same fitted
baseline. Each normalization chooses its own winning rung. This isolates the
denominator change; it is not a comparison of every change between releases.
The corrected ranking must reproduce the public table, and a separate public
screen with the old denominator substituted must reproduce the old ranking.

## Data and measurements

- Thirty features: twelve P-spline margins with alternating widths 6, 8, 10,
  and eighteen categorical margins with alternating 2, 3, 5 levels. All 435
  pairs are eligible. Numeric support is a fixed, equally likely 41-point
  grid on [-1, 1], so this study exercises exact dense screening.
- Gaussian bilinear, Gaussian wave, Gaussian spline-by-category, Gaussian
  category-by-category, Poisson wave, and Gaussian correlated-wave cases.
  Each has one planted pair. Signal functions are population-centred and
  have population second moment one under the independent design.
- Correlated inputs copy each target margin into a separate nuisance margin
  with probability 0.7. The two target margins remain independent. Recovery
  of the named pair is descriptive here: correlated substitutes can have
  predictive value and are not labelled scientifically false interactions.
- 4,000 training, 2,000 validation and 8,000 independent test observations.
  The noise standard deviation is one for Gaussian outcomes. Poisson signal
  strength is measured on the log-mean scale, not equated with Gaussian SNR.
- Signal strengths 0.02, 0.04, 0.06, 0.08, 0.12; 100 replicates per case and
  strength. The pilot uses a separate seed namespace and is excluded.
- Primary outcome: planted-pair inclusion in the top three. Also record its
  rank, top-one and top-ten inclusion, shortlist changes, winning rungs,
  maximum scores, failures, fit/screen time, process peak RSS and dispatch.
- For independent Gaussian, independent Poisson and correlated Gaussian
  designs, run 200 separate no-interaction calibration datasets and 100
  independent no-interaction audit datasets. Each method's cutoff is the
  191st ascending maximum among 200 runs, with strict `score > cutoff`.
  Maxima include all pairs and each method's own rung selection. Report both
  any-pair rejection and the planted pair exceeding its cutoff. Audit false
  alarms without retuning thresholds. This targets a common 5% false-alarm
  rate; 100 audit runs cannot certify tightly matched achieved rates.
  These are empirical full-pipeline comparisons for these data
  generators, not universal p-values. Include uncertainty from finite
  calibration by bootstrapping paired old/new calibration rows, recomputing
  both cutoffs, then independently bootstrapping paired audit or signal rows.
- On the first 20 preselected replicates at each nonzero strength, refit the
  union of both top-three lists, one interaction at a time. Each arm selects
  by validation loss among its own three refits and the baseline. Evaluate
  that choice once on the separate test set. Shared candidates are fitted
  once and credited identically to both arms. Also report prediction risk
  against the known generating mean to distinguish test-noise variation.
- Pair the methods within datasets. Report Wilson intervals for recovery and
  false-alarm proportions, paired bootstrap intervals for differences and
  gains. Resample datasets, not test observations; report discordant counts.
  The 20 predictive replicates are exploratory. An all-zero paired bootstrap
  interval with no discordances does not establish equivalence.
  Intervals are pointwise, not simultaneous across all cases. A null
  maximum cutoff answers a different question from fixed-shortlist recovery.
- Never exclude a failed fit or refused row silently. Count and retain the
  failure. Do not replace its seed. Failed screens count as missed recovery
  and no alert in the operational result. Their calibration maximum is
  negative infinity, reflecting that the procedure issued no alert. Failed
  refits are unavailable to validation selection; baseline stays available.
  Freeze validation winners before evaluating any test predictions. A test
  evaluation failure remains an evaluation failure and never triggers a new
  selection. Nonfinite predictions and metrics are explicit failures.
  Report failure counts and usable denominators alongside operational rates.

The first execution was interrupted during its first scenario after review
identified the two prediction-failure safeguards above. Its engineering
records are retained separately and excluded. The final execution uses the
same fixed grid and seeds; no outcome-based protocol changes were made.

## Execution and evidence

1. Verify the scorer on hand-computable rungs, including a case where the
   winning rung changes. Check a real public screen and old-score replay.
2. Run an excluded pilot to verify convergence, dispatch and runtime. Any
   protocol revision must be recorded before starting the main experiment.
3. Run the fixed grid, with one BLAS/Numba thread per process. Write each
   completed dataset to resumable JSONL with package/source hashes and seeds.
4. Summarize all outcomes and produce standalone figures. Keep raw records
   under `.benchmark-artifacts/psst-detection-study/`; commit a compact receipt,
   the protocol, scripts and report. Runtime under concurrent workers is
   descriptive and is not a performance comparison between the two scores.

No production solver or screening changes are part of this experiment.
Reference: Morris, White and Crowther, 2019, *Using simulation studies to
evaluate statistical methods*, https://doi.org/10.1002/sim.8086.
