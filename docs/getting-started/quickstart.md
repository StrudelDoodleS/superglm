# Quick Start

## Recommended Start: Explicit Features + REML

For pricing models with spline terms, the default starting point is an explicit
feature spec plus `fit_reml()` with `selection_penalty=0`.

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
model.fit_reml(df, y, sample_weight=exposure, max_reml_iter=30)

predictions = model.predict(df)
print(model.summary())
```

## Large-`n` REML

If the dataset is large, keep the same workflow but turn on discrete REML:

```python
model = SuperGLM(
    family="poisson",
    selection_penalty=0.0,
    discrete=True,
    n_bins=256,
    features=features,
)
model.fit_reml(df, y, sample_weight=exposure, max_reml_iter=30)
```

## Auto-Detect Mode For Quick Prototypes

Auto-detect mode still exists and is useful for a quick first pass, but
explicit feature specs are the better default for pricing work.

```python
from superglm import SuperGLM

model = SuperGLM(
    family="poisson",
    selection_penalty=0.0,
    splines=["DrivAge", "VehAge", "BonusMalus"],
    n_knots=10,
)
model.fit_reml(df, y, sample_weight=exposure)
```

## Sparse Screening Or Fixed-Penalty Fitting

If your goal is sparse screening or compression rather than GAM-style REML
inference, use `fit()` with `selection_penalty > 0` instead:

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

## Weights And Offsets

`sample_weight=` is interpreted as exposure / frequency weight in insurance
settings, not inverse-variance weight. The `exposure=` fit alias has been
removed.

Two common patterns for count models:

```python
import numpy as np

# Raw count target: offset absorbs exposure, model estimates a rate
model.fit(df, claim_counts, offset=np.log(exposure))

# Rate target (count / exposure): sample_weight carries exposure
model.fit(df, claim_rate, sample_weight=exposure)
```

Next steps:

- [Recommended workflows](../guide/workflows.md)
- [Choosing a fitting path](../guide/fitting.md)
- [Validation and model comparison](../guide/validation.md)
