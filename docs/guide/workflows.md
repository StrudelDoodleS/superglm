# Recommended Workflows

This page is the fastest way to map a real pricing task to the right SuperGLM
workflow.

## 1. Standard Pricing Model

Use this when you want a spline-based GAM-style pricing model with automatic
smoothness selection and clean post-fit inference.

```python
model = SuperGLM(
    family="poisson",
    selection_penalty=0.0,
    features=features,
)
model.fit_reml(df, y, sample_weight=exposure)
```

This is the default recommendation for tariff development.

## 2. Large-`n` Production REML

Use this when the modeling story is still REML, but the dataset is large enough
that exact REML is too expensive.

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

This is the preferred path for large frequency datasets.

## 3. Sparse Screening Or Compression

Use this when you want a sparse model with fixed penalties rather than a REML
pricing model.

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

This is useful for screening, challenger compression, and lambda-path analysis.

## 4. REML With Term Shrinkage

Use `select=True` on spline terms when you want REML to decide whether a smooth
should stay nonlinear, collapse toward linear, or shrink toward zero.

```python
features = {
    "DrivAge": Spline(kind="ps", k=14, select=True),
    "VehAge": Spline(kind="cr", k=10, select=True),
    "Area": Categorical(base="most_exposed"),
}

model = SuperGLM(
    family="poisson",
    selection_penalty=0.0,
    features=features,
)
model.fit_reml(df, y, sample_weight=exposure)
```

Use this when you want REML-native shrinkage rather than sparse group
selection.

## 5. Monotone Business Shapes

If monotonicity is part of the specification, fit it inside the model rather
than repairing it afterward.

- `BSplineSmooth(..., monotone_mode="fit")`: QP-backed monotone fit
- `CubicRegressionSpline(..., monotone_mode="fit")`: QP-backed monotone fit
- `PSpline(..., monotone_mode="fit")`: SCOP-backed monotone fit

```python
model = SuperGLM(
    family="gaussian",
    selection_penalty=0.0,
    features={
        "BonusMalus": PSpline(
            n_knots=10,
            monotone="increasing",
            monotone_mode="fit",
        ),
    },
)
model.fit_reml(df, y, sample_weight=exposure)
```

Use post-fit monotone repair only as a manual fallback.

## 6. Validation And Challenger Comparison

Use the same folds across all candidate models and keep holdout untouched until
you are ready to judge challengers.

```python
result = cross_validate(
    model,
    train_df,
    y_train,
    sample_weight=exposure_train,
    fit_mode="fit_reml",
    scoring=("deviance", "nll", "gini"),
    return_oof=True,
)
```

Then refit on all training data and evaluate:

- `lorenz_curve(...)` for ranking power
- `double_lift_chart(...)` for business-facing challenger evidence

## 7. Deployment

Serialize the fitted estimator itself.

```python
import pickle

with open("pricing_model.pkl", "wb") as f:
    pickle.dump(model, f)
```

That preserves the full fitted state: feature specs, knots, constraints,
coefficients, and REML-selected smoothing parameters.

## Where To Go Next

- [Choosing a fitting path](fitting.md)
- [Feature types](features.md)
- [Monotone splines](monotone.md)
- [Validation and model comparison](validation.md)
- [Deployment](deployment.md)
