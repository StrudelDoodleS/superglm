# Model Validation

SuperGLM provides a complete validation toolkit for comparing and assessing
GLM/GAM models in actuarial pricing workflows.

## Diagnostic plots

Call `model.plot_diagnostics(X, y)` for a 4-panel figure using quantile
residuals (Dunn & Smyth 1996):

1. **Q-Q with simulation envelope** — simulates response data from the
   fitted model and builds a 95% pointwise reference band. Points inside
   the band indicate good fit.
2. **Calibration** — exposure-weighted observed vs predicted frequency
   across equal-exposure bins.
3. **Residuals vs linear predictor** — quantile residuals plotted against
   eta (the additive scale), with a trend line for pattern detection.
4. **Residual distribution** — histogram with N(0,1) overlay and
   dispersion/chi-squared summary.

```python
model.plot_diagnostics(X, y, sample_weight=exposure)
```

For large datasets (>20k rows), hexbin density rendering and quantile-grid
Q-Q envelopes activate automatically for performance.

## Cross-validation

Use `cross_validate()` for model selection and stability assessment:

```python
from sklearn.model_selection import KFold
from superglm.model_selection import cross_validate

result = cross_validate(
    model, X, y,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    sample_weight=exposure,
    fit_mode="fit_reml",
    scoring=("deviance", "nll", "gini"),
    return_oof=True,
)
```

**Key outputs:**

- `result.fold_scores` — per-fold DataFrame with fit time, convergence,
  effective df, and all requested metrics
- `result.mean_scores` / `result.std_scores` — summary statistics
- `result.oof_predictions` — out-of-fold predictions for every training row

**Interpreting metrics:**

| Metric | Measures | Lower/higher better |
|--------|----------|-------------------|
| deviance | Probabilistic fit | Lower |
| nll | Negative log-likelihood | Lower |
| gini | Ranking / segmentation power | Higher |

## Double lift chart

The CAS-style double lift chart ([CAS RPM 2016](https://www.casact.org/sites/default/files/presentation/rpm_2016_presentations_pm-lm-4.pdf))
compares a new model against a current/baseline model:

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

The chart sorts observations by the ratio of new-model to current-model
predictions, bins into equal-exposure quantiles, and plots three indexed
series (each indexed to its own overall average). Where the new model's
line tracks Actual more closely than Current, it is adding value.

## Lorenz curve and Gini

```python
from superglm.validation import lorenz_curve

result = lorenz_curve(y_obs, y_pred, exposure=exposure)
print(f"Gini ratio: {result.gini_ratio:.4f}")
```

The Gini ratio (model Gini / perfect-foresight Gini) measures risk
segmentation power. Values closer to 1.0 indicate better ranking.

## Recommended workflow

1. **Split** data into training (80%) and holdout (20%)
2. **Cross-validate** on training with the same folds for all candidate models
3. **Compare** fold-level deltas (mean, median, std) — models often
   trade off across metrics (e.g. better deviance, slightly worse Gini),
   so present the full delta table rather than picking a single winner
4. **Refit** selected models on all training data
5. **Evaluate** on holdout with `double_lift_chart()` and `lorenz_curve()`
   — the holdout double-lift chart is the business-facing evidence

See the [Plotting & Diagnostics Demo](../notebooks/plotting_diagnostics_demo.ipynb)
notebook for a complete worked example on the French MTPL2 dataset.

## API reference

- [Validation functions](../api/validation.md) — lift charts, Lorenz curves, loss ratio charts
- [Diagnostics](../api/diagnostics.md) — plot_diagnostics, term importance, drop-term analysis
- [Model Selection](../api/model_selection.md) — cross_validate
