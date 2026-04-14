<p align="center">
  <img src="docs/images/logo.png" alt="SuperGLM" width="300">
</p>

[![CI](https://github.com/StrudelDoodleS/superglm/actions/workflows/ci.yml/badge.svg)](https://github.com/StrudelDoodleS/superglm/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/StrudelDoodleS/superglm/graph/badge.svg?token=2HO71TA2ZY)](https://codecov.io/github/StrudelDoodleS/superglm)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://github.com/StrudelDoodleS/superglm/actions/workflows/ci.yml)

Penalised GLMs and GAM-style pricing models for insurance. SuperGLM combines
explicit feature specs, exact REML, large-`n` discrete REML, solver-backed
monotone splines, actuarial validation tooling, and deployable fitted
estimators for Poisson, Gamma, NB2, Tweedie, Binomial, and Gaussian models.

## Installation

Current installation path:

```bash
pip install git+https://github.com/StrudelDoodleS/superglm.git
```

With optional dependencies:

```bash
pip install "superglm[all] @ git+https://github.com/StrudelDoodleS/superglm.git"
```

## Recommended Workflow

For spline-based pricing models, the default path is:

1. define explicit feature specs
2. fit with `fit_reml()` and `selection_penalty=0`
3. compare candidates with `cross_validate(..., fit_mode="fit_reml")`
4. refit on all training data
5. evaluate holdout Lorenz and double-lift charts
6. serialize the fitted estimator for scoring

```python
from superglm import Categorical, Numeric, Spline, SuperGLM

features = {
    "DrivAge": Spline(kind="ps", k=14, knot_strategy="quantile_rows"),
    "VehAge": Spline(kind="cr", k=10, knot_strategy="quantile_rows"),
    "BonusMalus": Spline(kind="cr", k=12, knot_strategy="quantile_tempered"),
    "Area": Categorical(base="most_exposed"),
    "LogDensity": Numeric(),
}

model = SuperGLM(
    family="poisson",
    selection_penalty=0.0,
    features=features,
)
model.fit_reml(train_df, y_train, sample_weight=exposure_train, max_reml_iter=30)

mu_holdout = model.predict(holdout_df)
print(model.summary())
```

## Choosing A Fit Path

### `fit_reml()` with `selection_penalty=0`

This is the recommended path for spline-heavy GAM-style pricing models. Use it
when you want automatic smoothness selection, interpretable smooth terms, and
mgcv-style modeling rather than sparse screening.

```python
model = SuperGLM(
    family="poisson",
    selection_penalty=0.0,
    features=features,
)
model.fit_reml(df, y, sample_weight=exposure)
```

### `fit_reml(discrete=True)`

Use this when the model is still a REML pricing model, but the data is large
enough that exact REML becomes expensive. This is the production-scale path for
large frequency models.

```python
model = SuperGLM(
    family="poisson",
    selection_penalty=0.0,
    discrete=True,
    n_bins=256,
    features=features,
)
model.fit_reml(df, y, sample_weight=exposure)
```

### `fit()` with `selection_penalty > 0`

Use this when you want sparse screening, compression, or fixed-penalty
regularisation. This is a different modeling story from REML smoothness
selection.

```python
model = SuperGLM(
    family="poisson",
    penalty="group_elastic_net",
    selection_penalty=0.01,
    spline_penalty=0.1,
    features=features,
)
model.fit(df, y, sample_weight=exposure)
```

### `select=True`

`select=True` on spline terms adds mgcv-style double-penalty shrinkage. This is
the REML-native way to let smooth terms shrink toward linear or zero while
staying in the `fit_reml()` workflow.

```python
features = {
    "DrivAge": Spline(kind="ps", k=14, select=True),
    "VehAge": Spline(kind="cr", k=10, select=True),
    "Area": Categorical(base="most_exposed"),
}
model = SuperGLM(family="poisson", selection_penalty=0.0, features=features)
model.fit_reml(df, y, sample_weight=exposure)
```

## Validation And Model Comparison

`cross_validate()` should be part of the standard pricing workflow, not an
afterthought. It gives fold-level metrics, timing, convergence information, and
out-of-fold predictions for challenger comparisons.

```python
from sklearn.model_selection import KFold
from superglm import cross_validate
from superglm.validation import double_lift_chart, lorenz_curve

cv = cross_validate(
    model,
    train_df,
    y_train,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    sample_weight=exposure_train,
    fit_mode="fit_reml",
    scoring=("deviance", "nll", "gini"),
    return_oof=True,
)

gini = lorenz_curve(y_holdout, mu_holdout, exposure=exposure_holdout)
lift = double_lift_chart(
    y_obs=y_holdout,
    y_pred_model=mu_holdout,
    y_pred_current=mu_baseline,
    exposure=exposure_holdout,
)
```

Key outputs:

- `cv.fold_scores`: per-fold metrics, fit time, convergence, and EDF
- `cv.mean_scores` / `cv.std_scores`: summary comparisons
- `cv.oof_predictions`: out-of-fold predictions for the training rows
- `lorenz_curve(...)`: ranking power via Gini
- `double_lift_chart(...)`: business-facing champion/challenger evidence

## Monotone Splines

SuperGLM supports solver-backed monotone spline fitting. This is the preferred
way to enforce business shape constraints inside the model itself.

- `BSplineSmooth(..., monotone="increasing", monotone_mode="fit")`:
  constrained QP path
- `CubicRegressionSpline(..., monotone="decreasing", monotone_mode="fit")`:
  constrained QP path
- `PSpline(..., monotone="increasing", monotone_mode="fit")`:
  SCOP path

```python
from superglm import BSplineSmooth, PSpline, SuperGLM

qp_model = SuperGLM(
    family="gaussian",
    selection_penalty=0.0,
    features={
        "x": BSplineSmooth(n_knots=8, monotone="increasing", monotone_mode="fit"),
    },
)

scop_model = SuperGLM(
    family="gaussian",
    selection_penalty=0.0,
    features={
        "x": PSpline(n_knots=10, monotone="increasing", monotone_mode="fit"),
    },
)
```

Post-fit isotonic repair still exists, but it should be treated as a manual
fallback rather than the main monotone workflow.

## Feature Highlights

- `Spline(kind="ps")`, `Spline(kind="cr")`, and `Spline(kind="ns")` cover the
  main spline basis choices.
- `OrderedCategorical(...)` smooths ordered factor levels without forcing a
  plain one-hot representation.
- `collapse_levels(...)` lets you merge sparse categorical levels while still
  expanding back to original levels for inference and plotting.
- `interactions=[(...)]` supports spline-categorical, numeric-categorical,
  tensor, and other interaction types.
- `m=(...)` supports multi-order spline penalties with separate REML lambdas.

```python
from superglm import Categorical, OrderedCategorical, Spline, collapse_levels

area_grouping = collapse_levels(train_df["Area"], groups={"Rural": ["E", "F"]})

features = {
    "VehAge": Spline(kind="cr", k=10),
    "Area": Categorical(base="most_exposed", grouping=area_grouping),
    "BonusClass": OrderedCategorical(order=["A", "B", "C", "D"], basis="spline"),
}
```

## Weights And Offsets

Public fitting examples use `sample_weight=`. In insurance settings this means
exposure / frequency weight, not inverse-variance weight.

```python
import numpy as np

# Raw count target: offset absorbs exposure, model estimates a rate
model.fit(df, claim_counts, offset=np.log(exposure))

# Rate target: sample_weight carries exposure
model.fit(df, claim_rate, sample_weight=exposure)
```

Validation helpers such as `lorenz_curve(...)` and `double_lift_chart(...)`
still use `exposure=...`, which is correct for that API.

## Deployment

A fitted `SuperGLM` is the deployment artifact. It already contains:

- registered feature specs
- learned knot geometry and constraints
- fitted coefficients and intercept
- REML smoothing parameters

```python
import pickle

with open("pricing_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("pricing_model.pkl", "rb") as f:
    loaded = pickle.load(f)

mu = loaded.predict(score_df)
```

The loaded model can still score, print summaries, rebuild curves, and produce
relativity views without refitting.

## Advanced Penalty Objects

At the top-level model API, prefer `selection_penalty=` and `spline_penalty=`.
Low-level penalty objects still expose `lambda1`, for example:

```python
from superglm import GroupElasticNet

penalty = GroupElasticNet(lambda1=0.01, alpha=0.5)
model = SuperGLM(family="poisson", penalty=penalty, features=features)
```

That is advanced usage. It should not be your default starting point.

## Learn More

- [Recommended workflows](docs/guide/workflows.md)
- [Choosing a fitting path](docs/guide/fitting.md)
- [Monotone splines](docs/guide/monotone.md)
- [Validation and model comparison](docs/guide/validation.md)
- [Deployment](docs/guide/deployment.md)
- [Optimization and solver internals](docs/guide/optimization.md)
