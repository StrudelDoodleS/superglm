# Structured Diagnostics Compactness Design

**Date:** 2026-07-27
**Status:** Approved in conversation; awaiting written-spec review
**PR:** #165
**Baseline:** `bf6d0b4`

## Context

The structured credibility implementation keeps dominant random-effect and
factor-smooth geometry compact during fitting, covariance access, reporting,
and prediction. A release re-review found three remaining generic-surface
problems:

1. `feature_se_from_cov()` requests a full random-effect covariance block
   before using only its diagonal. Structured factors intentionally reject
   dominant inverse blocks wider than 256 coefficients.
2. `term_importance()` and holdout `term_drop_diagnostics()` call
   `transform()`. For `RandomEffect` and `FactorSmooth`, this expands
   coefficient geometry that their existing `score()` methods avoid.
3. `refit_unpenalised()` rejects REML-only terms indirectly inside ordinary
   `fit()`, rather than defining its own variance-component contract.

All three findings reproduce on the baseline head. A 270-level structured
random effect fits and reports successfully, while its generic SE-enabled
term surfaces fail at the selected-inverse-block guard. Replacing
`RandomEffect.transform()` with a sentinel also makes both generic diagnostics
fail immediately.

## Goals

- Keep random-effect pointwise standard errors on selected covariance
  diagonals, regardless of term width.
- Keep term importance and holdout drop diagnostics on compact scoring paths
  for RE, FS, and SZ terms.
- Reuse prediction's canonical coefficient ordering, reparameterisation, and
  unknown-level behaviour.
- Preserve existing diagnostic schemas and term/group semantics.
- Reject unsupported unpenalised refits before cloning or ordinary fitting.
- Add regressions at and beyond the structured covariance materialisation
  boundary.

## Non-goals

- No structured solver, Schur algebra, SZ constraint, REML, or Tabmat redesign.
- No new multi-dominant structured topology.
- No new REML model-comparison method for `drop1()` or
  `refit_unpenalised()`.
- No change to holdout offset or validation-weight API semantics.
- No change to public diagnostic DataFrame columns or row granularity.
- No broad rewrite of covariance or prediction-plan internals.

## Design

### 1. Random-effect covariance diagonal dispatch

`feature_se_from_cov()` will identify `RandomEffect` before requesting a
principal covariance block.

For an active random effect it will call
`covariance_selected_diagonal(Cov_active, indices)`, clamp tiny negative
roundoff to zero, and take the square root. Dense covariance inputs retain the
same result through the helper's dense fallback. Compact structured covariance
inputs call `selected_diagonal()` and never invoke `selected_block()` or
`selected_inverse_block()`.

All other term types retain their current covariance-block behaviour because
splines, polynomials, and categorical reparameterisations may genuinely need
cross-covariances.

### 2. Canonical exact term-contribution scoring

Prediction already owns the correct compact scoring contract:

- the cached prediction plan contains canonical term coefficient indices;
- main features and interactions have distinct argument shapes;
- `_score_feature()` and `_score_interaction()` prefer `score()` and use
  `transform() @ beta` only as a compatibility fallback;
- RE and FS/SZ `score()` methods avoid level-expanded design matrices.

The model prediction module will expose a narrow internal helper that scores
one prediction-plan term from a term-local coefficient vector. Existing
whole-model exact prediction will call that helper after selecting the term's
coefficients. Diagnostics will use the same helper rather than duplicating
feature-type dispatch.

The helper remains internal. It does not create a new public API or expose the
prediction-plan dictionary as a supported user contract.

### 3. Compact `term_importance()`

`term_importance()` currently returns one row per `GroupSlice`, including
subgroup metadata. That behaviour will remain unchanged.

For each group:

1. Resolve the containing prediction-plan term.
2. Allocate a term-local coefficient vector, not a row-by-coefficient design.
3. Copy the current group's coefficients into their canonical positions and
   leave other subgroups at zero.
4. Score that term through the canonical exact term scorer.
5. Compute the existing weighted centred variance and output fields.

For single-group RE, FS, and SZ terms, this is simply their compact `score()`
path. For multi-group terms, zeroing non-target term-local coefficients
preserves the existing per-group interpretation without relying on a
term-width-specific transform.

Zero-norm groups retain the existing fast zero row and do not score.

### 4. Compact holdout drop diagnostics

Holdout diagnostics operate at unique feature/interaction level, not subgroup
level. They will:

1. Score each canonical term exactly once on the validation rows.
2. Retain one one-dimensional contribution vector per term.
3. Construct the raw full linear predictor from the fitted intercept plus
   those vectors.
4. Stabilise and inverse-link the full predictor for the baseline deviance.
5. For each dropped term, subtract its contribution vector from the same raw
   full predictor, then stabilise and inverse-link before computing deviance.

This removes coefficient copying, repeated full-model reconstruction,
`np.hstack()`, and expanded RE/FS/SZ transforms. Main effects and interactions
share the same plan and scorer, so coefficient ordering stays identical to
prediction.

The raw predictor must be assembled before stabilisation. Subtracting from an
already clipped/stabilised predictor can produce different answers for extreme
links and is not allowed.

### 5. `refit_unpenalised()` REML preflight

A small internal helper in `_term_model_ops.py` will return configured main
features and interactions whose specs set `requires_reml=True`.

Both `drop1()` and `refit_unpenalised()` will use it. The latter will reject
immediately after the fitted-model check with a method-specific
`NotImplementedError` explaining that an ordinary unpenalised fit cannot
preserve the variance-component contract.

The check occurs before selected-group discovery, model cloning, or fitting.
No implicit fixed-lambda or REML refit semantics will be invented for this
release.

## Complexity and performance contract

Let:

- `n` be evaluation rows;
- `T` be fitted terms;
- `K` be categorical levels;
- `k` be the factor-smooth marginal basis width.

The forbidden paths allocate `O(nK)` for RE and `O(nKk)` for FS/SZ, repeatedly
for holdout drops. At one million rows, 300 levels, and `k=10`, the FS
compatibility matrix alone is approximately 24 GB.

The new paths retain:

- one `O(n)` output vector per scored term for holdout diagnostics;
- term-local coefficient vectors;
- the FS/SZ marginal scoring workspace, bounded by `O(nk)` and independent of
  the number of levels.

No diagnostic may allocate an evaluation-row matrix whose width is the fitted
RE or FS/SZ coefficient count. The implementation should not add
parallelisation: compact vector and existing score kernels are the relevant
first-order improvement, and parallel overhead is outside this fix.

## Error handling

- Large structured RE inference must not hit a block-materialisation error.
- Any score/transform fallback shape error remains explicit and is not hidden.
- Unknown structured levels retain prediction's existing validation and
  population/conditional behaviour.
- `refit_unpenalised()` names all offending REML-only terms in its direct
  error.

## Testing

### Covariance boundary regression

Fit a 270-level random effect with `direct_solve="structured"` and verify:

- the structured backend is selected;
- `random_effects()`, `term_inference()`,
  `relativities(with_se=True)`, pointwise `plot_data()`, and pointwise `plot()`
  succeed;
- generic term standard errors agree with
  `random_effects().table["posterior_se"]`;
- selected covariance block access is not requested for the random-effect SE.

### Compact diagnostic regressions

For RE and FS/SZ models:

- obtain small-model reference diagnostic results;
- replace the structured spec's `transform()` with a sentinel failure after
  fitting;
- verify `term_importance()` and holdout drop diagnostics still succeed;
- compare results with the reference values;
- verify score calls use complete term-local coefficient vectors in canonical
  order.

For ordinary features and interactions, retain existing numerical and
DataFrame-boundary tests. Add a mixed main-effect/interaction case so the
full-minus-term predictor agrees with an independently zeroed-coefficient
reference.

### Refit preflight regression

Fit a model containing an RE or FS/SZ term and verify
`refit_unpenalised()` raises its own specific error before clone/fit work.

### Validation gates

- focused covariance, random-effect inference, factor-smooth, diagnostics, and
  refit tests;
- Ruff formatting and lint;
- MyPy for touched source;
- full pytest suite;
- exact-head package build/install smoke test;
- fresh code review on the final SHA.

## Acceptance criteria

- No structured term wider than the inverse-block guard fails merely because
  a public surface needs pointwise standard errors.
- Generic diagnostics never call RE or FS/SZ `transform()`.
- Holdout predictions remain numerically equivalent to zeroing the same
  fitted term in a small dense reference model.
- Existing diagnostic output schemas and row semantics do not change.
- REML-only unpenalised refits fail early with an intentional public contract.
- Focused and full validation pass on the exact final head.
