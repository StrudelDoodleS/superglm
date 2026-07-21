# Explicit Selection Penalties and Tweedie Profile CIs

## Problem

SuperGLM currently gives `selection_penalty=None` different meanings depending on
the entry point:

- `fit()` and the internal Tweedie/NB profile fits auto-calibrate a positive
  penalty;
- `fit_reml()` resolves it to zero;
- the sklearn wrappers resolve it to zero.

The configured model can consequently continue to report `None` while the fitted
model used a positive penalty. This is deterministic internally but is not a clear
public contract.

Separately, `estimate_p()` returns a Tweedie profile result whose confidence
interval is lazy. Calling `result.ci()` populates only the detached returned
result, while `model.summary()` reads the independently owned installed result.
There is therefore no public way to request the interval during estimation and
have it appear in the model summary.

## Selected Public Contract

### Selection penalty

`selection_penalty` accepts `None`, `"auto"`, or a finite non-negative number:

- `None` means no selection penalty;
- `0.0` means no selection penalty and is exactly equivalent to `None` at fit
  time;
- `"auto"` explicitly requests the existing `0.1 * lambda_max` calibration;
- a positive number requests that exact penalty strength;
- negative, non-finite, Boolean, or unknown string values are rejected.

The constructor default remains `None`, so an ordinary default `fit()` becomes
unpenalized. `selection_penalty` continues to expose constructor intent, while
`selection_penalty_` reports the resolved numeric value from the installed fit:
zero for `None`/`0.0`, the calibrated value for `"auto"`, or the supplied positive
value.

The same resolution helper is used by ordinary fits and internal Tweedie and NB
profile fits. Clone, cross-validation, and transactional workspace boundaries
preserve the configured intent without mutating it.

`fit_path()` remains an explicitly requested regularization-path operation. Its
lambda sequence controls the successive penalties and its installed result reports
the terminal numeric penalty through `selection_penalty_`.

### REML

`fit_reml()` permits only disabled selection (`None` or numeric zero). It rejects
`"auto"` and every positive selection penalty before expensive fit work starts.
REML continues to optimize only its supported smooth quadratic penalties. Sparse
selection must use `fit()` or `fit_path()`; REML-managed null-space shrinkage uses
the existing spline `select=True` mechanism.

Tweedie `estimate_p(fit_mode="reml")` and NB profiling obey the same restriction
because their candidate and final fits use the REML contract.

### Tweedie CI request

`SuperGLM.estimate_p()` gains:

```python
ci_alpha: float | None = None
```

- `None` preserves today's lazy behavior and adds no CI evaluations, coefficient
  fits, density passes, or material performance cost;
- a finite value strictly between zero and one explicitly computes the
  likelihood-ratio profile interval;
- requesting a CI with `phi_method="pearson"` raises the existing clear error
  because that profile does not support likelihood-ratio intervals.

The requested interval is computed on the attempt-owned profile result after the
final refit is synchronized but before publication. The installed profile copy is
then created from the populated result, so both the returned result and
`model.summary(alpha=ci_alpha)` contain independently owned cached intervals.
Their cache objects remain isolated after publication.

If requested CI computation fails, the `estimate_p()` transaction fails and the
previous public fitted revision remains unchanged. No partially updated family,
fit, profile result, or summary cache is published.

## Alternatives Considered

1. Keep `None` as automatic calibration. Rejected because its meaning already
   changes by entry point and hides a material modeling choice.
2. Add separate `selection_mode` and `selection_penalty` parameters. Rejected as
   unnecessary two-field state with invalid combinations.
3. Make numeric values the only API and remove auto-calibration. Clear, but it
   needlessly removes a useful existing capability. An explicit `"auto"` value
   retains that capability without hiding it.
4. Make `model.summary()` compute Tweedie CIs automatically. Rejected because a
   reporting call would unexpectedly perform potentially expensive profile fits.
5. Share the returned and installed CI cache. Rejected because detached-result
   mutation must not be able to poison authoritative model reporting.

## Validation

Tests will establish that:

- `None` and `0.0` both resolve to a fitted selection penalty of zero for
  ordinary fits, Tweedie profiles, and NB profiles;
- `"auto"` is the only implicit-calibration request and preserves constructor
  intent while publishing a finite numeric fitted value;
- REML rejects `"auto"` and positive values before profiling or design work;
- invalid penalty settings fail clearly;
- default categorical-only fits use direct unpenalized geometry and report
  integer-rank EDF up to floating-point roundoff;
- `ci_alpha=None` performs no CI work;
- an explicit `ci_alpha` populates both independently owned caches and makes the
  matching model summary interval available;
- a mismatched summary alpha remains `not computed`;
- returned-result mutation cannot alter the installed interval;
- CI failure rolls back the complete profile-fit transaction;
- coefficient fits, profile passes, and default-path timings do not increase.

## Scope

This change does not alter the IRLS/PIRLS algorithms, REML optimizer, Tweedie
profile likelihood, CI construction, matrix kernels, Tabmat dispatch, numerical
tolerances, or fallback ordering. It changes only public configuration resolution,
entry-point validation, explicit CI orchestration, documentation, and regression
tests.
