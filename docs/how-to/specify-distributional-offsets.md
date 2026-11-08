# Specify offsets in distributional predictors

An offset is a known addition to a linear predictor, with coefficient fixed at
one. Declare it with `offset=` on the family helper. It adds no estimated term,
penalty, effective degrees of freedom, or coefficient uncertainty.

```python
import numpy as np
from superglm import NegativeBinomialLS, SuperLSS, s

family = NegativeBinomialLS()
model = SuperLSS(
    family,
    family.mean(s("age"), offset=["log_exposure", "adjustment"]),
    family.theta(),
)
model.fit_reml(train, claim_count)
prediction = model.predict(new_rows)
```

Here the mean predictor is `intercept + age_effect + log_exposure + adjustment`.
The log link exponentiates that sum. Supply the logarithm yourself: an offset
does not automatically log an exposure, implement an earning law, or evaluate
an expression. Every parameter helper accepts offsets, including `mu`, `phi`,
`p`, `location`, `scale`, `shape`, `skew`, `theta`, and Dirichlet `alpha`.

`SuperLSS.fit` and `fit_reml` take no `offsets` argument, positional or keyword.
The `SuperGLM` offset API is unchanged.

## Sources and copying

These declarations are supported:

```python
family.mean("age", offset="log_exposure")
family.mean("age", offset=np.log(exposure))
family.mean("age", offset=["log_exposure", "adjustment"])
family.mean("age", offset=[np.log(exposure), adjustment])
family.mean("age", offset=["log_exposure", adjustment])
```

A flat numeric list or tuple is one row vector. A list or tuple of vectors or
string column names declares multiple sources, summed in order on the link
scale. Empty lists, scalar/vector mixtures, and two-dimensional arrays are
refused. For one constant offset, supply a full vector such as
`np.full(len(train), 0.2)`. Sources may differ between predictors. Omitting
terms gives an intercept-only predictor that can still have an offset.

The generic binder accepts the same keyword:

```python
from superglm import bind_predictor

mean = bind_predictor(family, "mean", "age", offset=["log_exposure", adjustment])
```

Arrays are independently copied and frozen when declared, and copied again
when a model takes ownership. Mutating the original array or list cannot
change the model. Use a new declaration to change fit-row array values.
Column sources are resolved afresh from each fitting or prediction frame;
pandas and Polars frames are supported. A pandas Series source must have the
same index as a pandas input frame. Arrays are positional; Series used with
Polars are positional as well.

All source lengths and pandas labels are checked on the original input rows.
Rows with zero likelihood weight are then removed, before retained source
values are required to be real, numeric, and finite. The sum must also be
finite. Errors identify original row positions. Missing or duplicate-ambiguous
columns are refused even when some rows have zero weight.

## Prediction replacements

Column declarations read the new frame, including on a different number or
order of rows. Do not also supply `offsets` for a predictor declared only with
columns: that would duplicate the source, and is refused even if the values
are equal.

Array declarations belong to fitting rows. Every new-data prediction needs
replacement values, even when the new frame has the same row count:

```python
model = SuperLSS(
    family,
    family.mean("age", offset=[np.log(exposure), adjustment]),
    family.theta(),
).fit(train, claim_count)

prediction = model.predict(
    new_rows,
    offsets={"mean": np.log(new_exposure) + new_adjustment},
)
```

The replacement is the **sum of all array sources** for that predictor. For a
mixed declaration such as `offset=["log_exposure", adjustment]`, pass only the
new adjustment in `offsets={"mean": new_adjustment}`; `log_exposure` still
comes from the new frame and is added once. Keys are family parameter names,
not helper names: Tweedie's `mu`, for example, uses `"mean"`.

Prediction mappings still work for predictors without a declaration, including
older saved models. Omitting their mapping means zero, as before. Predictions
evaluate the fixed fitted model and never run an optimizer. Link-scale standard
errors describe coefficient uncertainty and do not add uncertainty for offsets.
Term effects and coefficient summaries describe estimated contributions;
the declaration is available in `model.predictors` and `model.predictors_`.

## Refit, validation, and saving

Calling `fit` or `fit_reml` again resolves the model's declarations against
that frame. Internal term-replacement clones preserve them. Cross-validation
slices fit-row arrays positionally for each training fold and supplies the
held-out array sums at prediction; column sources are read within each fold.
For `SuperLSS`, declare offsets on the predictors before `cross_validate`;
its separate `offset=` fitting input is refused. The `SuperGLM` cross-validation
input remains supported.

Scores, residuals, posterior predictions, and other new-row diagnostics follow
the same prediction replacement rules. Prediction surfaces retain offset-only
columns in their reference rows. Array replacements for a surface describe
the swept points, not the original training rows.

Pickle and joblib preserve declarations, fitted coefficients, and retained
offset sums. Restoring a model does not turn its fit-row arrays into new-row
data. Null models and internal covariance calculations use the fit's resolved
offset sums directly, preserving each contribution once.

For a Dirichlet model, declare each concentration's offset with
`family.alpha(component, offset=...)`; `family.component_parameters[component]`
gives its prediction-mapping key. Offsets do not alter component ordering or
the family's composition-validation policy.
