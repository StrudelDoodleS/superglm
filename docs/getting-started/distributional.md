# Your first distributional model

`SuperLSS` fits several parameters of a response distribution together. A
Gaussian model, for example, can let both the mean and the standard deviation
vary with the input columns. Each parameter gets its own predictor.

This walkthrough creates sample data, fits two models and compares their
predictions on held-out rows. It needs NumPy, pandas and SuperGLM, with no data
download.

## Create some data

Here the response mean depends on age and region. Its standard deviation also
increases with age. These are simulated continuous measurements, so a Gaussian
response is appropriate.

```python
import numpy as np
import pandas as pd

from superglm import GaussianLS, SuperLSS, cat, s

rng = np.random.default_rng(42)
n = 800
frame = pd.DataFrame(
    {
        "age": rng.uniform(18, 80, n),
        "region": rng.choice(["North", "South"], n),
    }
)
age = frame["age"].to_numpy()
south = (frame["region"] == "South").to_numpy()
mean = 10 + 0.08 * (age - 45) + 0.002 * (age - 45) ** 2 + 1.5 * south
sd = 0.8 + 0.018 * (age - 18)
y = rng.normal(mean, sd)

X_train, X_test = frame.iloc[:600], frame.iloc[600:]
y_train, y_test = y[:600], y[600:]
```

The split happens before fitting. Both models below learn their spline bases
and smoothing parameters from the training rows.

## Declare the predictors

```python
family = GaussianLS()

model = SuperLSS(
    family,
    family.location(s("age", kind="cr", k=8), cat("region")),
    family.scale(s("age", kind="cr", k=6)),
)
```

The family comes first. Its two helper calls describe the predictors:

- `location(...)` models the Gaussian conditional mean. Age has a smooth
  effect and region has a categorical effect.
- `scale(...)` models the Gaussian standard deviation. It has its own smooth
  age effect and no region effect.

These declarations are configuration. `model` owns the fit. The age smooths
have separate coefficients because they belong to different predictors.

`kind="cr"` selects a cubic regression spline. `k` sets its basis size;
smoothing determines how much of that flexibility the fit uses. Use a bare
string, such as `"age"`, when you want a numeric linear term instead.
Categories are always explicit with `cat(...)`, including categories stored
as numbers.

Every parameter must have a declaration. An empty call such as
`family.scale()` estimates an intercept-only predictor. It does not fix the
parameter to a numeric value, and omitting the call is an error. Each predictor
includes an intercept unless you pass `intercept=False` to its helper.

Use helpers on the same family instance that you pass to `SuperLSS`. You may
put the declarations in any order; names identify their parameters.

## Fit the model

```python
model.fit_reml(X_train, y_train, outer="efs+newton")
print(model.summary())
print(model.diagnose())
```

`fit_reml` estimates the coefficients and smoothing parameters jointly.
`outer="efs+newton"` adds Newton refinement to the smoothing updates. On this
example, the default EFS updates stop after rejecting a proposal; the
refinement reaches a stationary fit with the same convergence tolerances.
`diagnose()` reports the stopping evidence. Use `fit` when you want to hold
smoothing parameters fixed. Both fitting methods update the model and return it.

The summary separates terms by predictor. An age effect in `location` changes
the conditional mean. An age effect in `scale` changes the spread. The Gaussian
scale link is `log(scale - scale_floor)`, so its coefficients act on that
linked quantity.

## Predict the mean, spread and a quantile

```python
parameters = model.predict_parameters(X_test)
predicted_mean = model.predict(X_test)
upper_quantile = model.predict_quantile(X_test, 0.95)

predictions = parameters.assign(
    observed=y_test,
    predicted_mean=predicted_mean,
    q95=upper_quantile,
)
print(predictions.head())
```

For this family, `parameters` has `location` and `scale` columns. They contain
the mean and standard deviation on the response scale. `predict` returns the
same mean as the `location` column. `predict_link` is available when you need
the linear predictors before applying the inverse links.

The 95th percentile describes the upper part of each row's predictive
distribution. It is not a confidence bound on the estimated mean.

## Compare against constant spread

Keep the same mean terms and estimate one standard deviation for all rows:

```python
constant_scale = SuperLSS(
    family,
    family.location(s("age", kind="cr", k=8), cat("region")),
    family.scale(),
).fit_reml(X_train, y_train, outer="efs+newton")

held_out_loss = pd.Series(
    {
        "varying_scale": model.scores(X_test, y_test, which=("log",))["log"].mean(),
        "constant_scale": constant_scale.scores(
            X_test, y_test, which=("log",)
        )["log"].mean(),
    },
    name="mean_negative_log_likelihood",
)
print(held_out_loss)
```

A lower mean negative log-likelihood is better on these held-out rows. It
scores the predicted distribution, so spread matters as well as mean. For
your own data, make the split respect time or group boundaries when those
matter.

Reusing `family` does not share fitted coefficients. Constructing and fitting
`constant_scale` leaves the first model's fit intact. The models each copy
their configuration at construction.

## Choose another response family

Choose the family for the response you have. `GammaLS` models strictly
positive values. `TweedieLSS` admits both zeros and positive values.
`NegativeBinomialLS` models overdispersed counts. Their predictor names differ
because their parameters differ.

For example, a Tweedie declaration uses three helpers:

```python
from superglm import TweedieLSS

tweedie = TweedieLSS()
tweedie_model = SuperLSS(
    tweedie,
    tweedie.mu(s("age", kind="cr", k=8), cat("region")),
    tweedie.phi(s("age", kind="cr", k=6)),
    tweedie.p(),
)
```

This declares the mean, dispersion and power predictors. Their names in
results and offsets are `mean`, `dispersion` and `power`. The empty `p()` call
estimates one power value for all rows. This code only constructs the model;
fit it to a response for which the Tweedie law is appropriate.

See [family predictor names](../models/distributional.md#family-predictor-names)
for all nine families, including what their scale and shape parameters mean.
For fitting options and return values, use the
[SuperLSS API reference](../api/distributional.md). The
[distributional model guide](../models/distributional.md) covers weights,
offsets, interactions and discrete fitting; the
[checking guide](../models/distributional-inference.md) covers calibration and
predictive diagnostics.
