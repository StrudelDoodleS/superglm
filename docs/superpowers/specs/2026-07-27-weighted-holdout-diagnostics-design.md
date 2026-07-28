# Weighted Refit and Holdout Diagnostics Design

## Context

Two public refit paths currently accept training weights but do not pass them to
the reduced or unpenalised fit:

- `refit_unpenalised()` discards `sample_weight`.
- `term_drop_diagnostics(mode="refit")` calls `drop1()` without
  `sample_weight`.

Holdout drop diagnostics have a separate contract problem. They accept
validation rows through `X_val` and `y_val`, but reuse the training
`sample_weight` argument as validation weights and ignore `offset`. This can
silently compare fits under different weighting geometries, omit an insurance
exposure offset, or broadcast a training-length vector against a validation
portfolio.

This change fixes those public diagnostic contracts without changing fitting,
structured scoring, REML, RE/FS/SZ algebra, or LSS.

## Public API

`SuperGLM.term_drop_diagnostics()` gains two keyword-only arguments:

```python
def term_drop_diagnostics(
    self,
    X,
    y,
    sample_weight=None,
    offset=None,
    *,
    mode="refit",
    X_val=None,
    y_val=None,
    sample_weight_val=None,
    offset_val=None,
):
    ...
```

The same arguments flow through `model.explain_ops.term_drop_diagnostics()` and
`diagnostics.term_diagnostics.term_drop_diagnostics()`.

The existing arguments retain their meanings:

- `X`, `y`, `sample_weight`, and `offset` describe training/refit rows.
- `X_val`, `y_val`, `sample_weight_val`, and `offset_val` describe holdout
  evaluation rows.

## Refit Semantics

`refit_unpenalised()` forwards the supplied `sample_weight` and `offset` to the
new model's ordinary `fit()` call.

`term_drop_diagnostics(mode="refit")` forwards both `sample_weight` and
`offset` to `drop1()`. Its full and reduced model statistics therefore use the
same likelihood geometry.

The existing early rejection of variance-component terms in
`refit_unpenalised()` remains unchanged.

## Holdout Resolution Rules

Validation-specific arguments always win.

For backwards compatibility, training vectors may be reused only when
`X_val is X` and `y_val is y`:

- If `sample_weight_val` is omitted, use `sample_weight`.
- If `offset_val` is omitted, use `offset`.

Identity is checked before converting either frame to the internal eager-frame
representation. Equal lengths or equal values are not sufficient: the
implementation must not guess that separately supplied rows have identical
ordering.

For separate validation objects:

- If `sample_weight` is supplied and `sample_weight_val` is omitted, raise a
  `ValueError` requesting validation-specific weights. A deliberately
  unweighted validation calculation can pass an all-one
  `sample_weight_val`.
- If the fitted model used an offset, or `offset` is supplied, and
  `offset_val` is omitted, raise a `ValueError` requesting the validation
  offset. A deliberately zero-offset calculation can pass an all-zero
  `offset_val`.
- If neither training weights nor validation weights are supplied, validation
  weights default to one.
- If the fitted model did not use an offset and neither offset argument is
  supplied, the validation offset defaults to zero.

These rules preserve the existing same-row convenience while making a
different validation portfolio explicit.

## Validation and Scoring

Before scoring, holdout inputs are normalized to one-dimensional `float64`
arrays and checked against `len(X_val)`:

- `y_val` must be one-dimensional, non-empty, finite, have the same row count
  as `X_val`, and satisfy the fitted distribution's response domain.
- Validation weights must be one-dimensional, finite, nonnegative, have the
  same row count as `X_val`, and contain at least one positive value.
- The validation offset must be one-dimensional, finite, and have the same row
  count as `X_val`.

The compact prediction plan continues to score each canonical term exactly
once. The raw validation predictor is:

```text
intercept + validation offset + sum(term contributions)
```

Full and dropped predictors are stabilized only after the complete raw
predictor has been assembled. Dropping a term subtracts its cached
one-dimensional contribution; RE, FS, and SZ terms must never call their dense
`transform()` compatibility paths.

## Error Messages

Errors identify the public validation argument that is missing or invalid:

- separate validation rows with training weights request
  `sample_weight_val`;
- an offset-based fit without an evaluation offset requests `offset_val`;
- shape errors state the expected validation row count;
- non-finite or invalid weight values state the violated constraint.

No warning-based or silent fallback is used for separate validation objects.

## Testing

Regression coverage will establish:

1. `refit_unpenalised()` with non-uniform weights agrees with an explicitly
   weighted reference refit and differs from the unweighted result.
2. Refit-mode term-drop diagnostics agree with weighted `drop1()`.
3. Holdout diagnostics with a Poisson log-exposure offset agree with an
   independent offset-aware deviance calculation.
4. Separate validation row counts use `sample_weight_val` and `offset_val`
   without broadcasting training vectors.
5. Separate validation objects reject ambiguous training-weight and
   training-offset fallback.
6. Same-object validation preserves the existing training-vector convenience.
7. Invalid validation vectors fail for dimension, length, finiteness,
   negativity, or all-zero weights.
8. Existing RE/FS/SZ transform sentinels continue to prove compact scoring
   when validation weights and offsets are present.

Each behavior is introduced test-first and observed failing for the expected
omission or contract violation before production code changes.

## Documentation and Compatibility

The model API docstring and diagnostics documentation will distinguish training
and validation vectors and explain the same-object fallback. This is an
additive keyword-only API change. Existing same-row holdout calls continue to
work; ambiguous separate-row calls fail explicitly instead of silently using
the wrong geometry.

## Non-goals

- No changes to structured selection, Schur factors, SZ constraints, REML, or
  covariance.
- No changes to term contribution layout or compact scoring kernels.
- No automatic value-based comparison of training and validation portfolios.
- No redesign of the broader model-selection or cross-validation APIs.
