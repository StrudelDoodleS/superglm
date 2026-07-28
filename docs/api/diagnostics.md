# Diagnostics

Model diagnostic tools: residual plots, term importance, drop-term analysis,
and spline redundancy checks.

## Diagnostic plots

::: superglm.plotting.diagnostics.plot_diagnostics

## Term diagnostics

::: superglm.diagnostics.term_importance

::: superglm.diagnostics.term_drop_diagnostics

::: superglm.diagnostics.spline_redundancy

## Holdout term-drop geometry

Holdout diagnostics distinguish training vectors from validation vectors. Pass
validation weights and offsets explicitly when evaluating a separate
portfolio:

```python
drop = model.term_drop_diagnostics(
    X_train,
    y_train,
    sample_weight=weight_train,
    offset=np.log(exposure_train),
    mode="holdout",
    X_val=X_validation,
    y_val=y_validation,
    sample_weight_val=weight_validation,
    offset_val=np.log(exposure_validation),
)
```

For backwards compatibility, `sample_weight` and `offset` may be reused when
the validation objects are literally the same objects (`X_val is X` and
`y_val is y`). Separate portfolios never inherit training vectors merely
because their lengths happen to match.
