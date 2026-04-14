# Validation And Model Comparison

Validation should be part of the default pricing workflow, not something added
after the model is already chosen.

## Recommended Workflow

1. split data into training and holdout
2. compare candidate models with `cross_validate()` on training data
3. inspect fold-level deltas, not just one mean metric
4. refit the chosen candidates on all training data
5. evaluate holdout Lorenz and double-lift charts
6. use diagnostics to check residual behavior and calibration

## Cross-Validation

`cross_validate()` is the main comparison tool. For REML pricing models, call
it with `fit_mode="fit_reml"` so each fold uses the same fitting story as the
final model.

```python
from sklearn.model_selection import KFold
from superglm.model_selection import cross_validate

result = cross_validate(
    model,
    X,
    y,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    sample_weight=exposure,
    fit_mode="fit_reml",
    scoring=("deviance", "nll", "gini"),
    return_oof=True,
)
```

Key outputs:

- `result.fold_scores`: per-fold metrics, fit time, convergence, and EDF
- `result.mean_scores` / `result.std_scores`: summary comparisons
- `result.oof_predictions`: out-of-fold predictions for every training row

Typical metric interpretation:

| Metric | Measures | Better |
|--------|----------|--------|
| `deviance` | probabilistic fit | lower |
| `nll` | negative log-likelihood | lower |
| `gini` | ranking / segmentation power | higher |

Out-of-fold predictions are especially useful for challenger analysis because
they let you compare models on the training portfolio without leaking each row
into its own fitted mean.

## Holdout Business Evidence

After cross-validation, refit the chosen candidates on all training data and
evaluate them on holdout.

### Lorenz Curve And Gini

```python
from superglm.validation import lorenz_curve

result = lorenz_curve(y_obs, y_pred, exposure=exposure)
print(f"Gini ratio: {result.gini_ratio:.4f}")
```

The Gini ratio measures ranking power relative to perfect foresight. It is the
standard quick view of segmentation quality.

### Double Lift Chart

Use the CAS-style double-lift chart when you need business-facing evidence that
a challenger is improving on the current model rather than just fitting well in
the abstract.

```python
from superglm.validation import double_lift_chart

result = double_lift_chart(
    y_obs=y_holdout,
    y_pred_model=mu_new,
    y_pred_current=mu_baseline,
    exposure=exposure_holdout,
    n_bins=20,
)
```

The chart sorts by the challenger/current relativity ratio, bins to equal
exposure, and shows whether the challenger tracks Actual more closely than the
baseline.

## Diagnostic Plots

Use diagnostics after you have a candidate worth defending.

```python
model.plot_diagnostics(X, y, sample_weight=exposure)
```

The four-panel diagnostic figure includes:

1. Q-Q with simulation envelope
2. exposure-weighted calibration
3. residuals versus linear predictor
4. residual histogram with normal overlay

For large datasets, the plotting code automatically switches to more efficient
rendering paths such as hexbin density summaries.

## Practical Comparison Advice

- use the same CV folds for all candidate models
- compare fold-level deltas, not just headline means
- keep holdout truly untouched until the challenger set is stable
- use both probabilistic metrics and business ranking metrics
- treat double-lift as the business communication chart

See the [Plotting & Diagnostics Demo](../notebooks/plotting_diagnostics_demo.ipynb)
for a worked example on French MTPL2.
