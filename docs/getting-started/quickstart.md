# Quick Start

## Auto-detect mode

List which columns should be splines — the rest is auto-detected:

```python
from superglm import SuperGLM

model = SuperGLM(
    family="poisson",
    penalty="group_lasso",
    selection_penalty=0.01,
    splines=["DrivAge", "VehAge", "BonusMalus"],
    n_knots=10,
)
model.fit(df, y, sample_weight=exposure)
predictions = model.predict(df)
```

## Explicit mode

Full control over each feature:

```python
from superglm import SuperGLM, Spline, Categorical

model = SuperGLM(
    family="poisson",
    penalty="group_lasso",
    selection_penalty=0.01,
    features={
        "DrivAge": Spline(kind="bs", k=14),
        "VehAge": Spline(kind="cr", k=10),
        "BonusMalus": Spline(kind="ns", k=10),
        "Area": Categorical(base="most_exposed"),
    },
)
model.fit(df, y, sample_weight=exposure)
```

## Weights and offsets

`sample_weight=` is interpreted as **exposure / frequency weight** in insurance settings, not inverse-variance weight. The `exposure=` alias has been removed — use `sample_weight=` only.

Two common patterns for count models:

```python
# Raw count target: offset absorbs exposure, model estimates a rate
model.fit(df, claim_counts, offset=np.log(exposure))

# Rate target (count / exposure): weight by exposure for heteroscedasticity
model.fit(df, claim_rate, sample_weight=exposure)
```
