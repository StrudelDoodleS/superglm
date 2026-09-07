<p align="center">
  <img src="https://raw.githubusercontent.com/StrudelDoodleS/superglm/master/docs/images/logo.png" alt="SuperGLM" width="300">
</p>

[![CI](https://github.com/StrudelDoodleS/superglm/actions/workflows/ci.yml/badge.svg)](https://github.com/StrudelDoodleS/superglm/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/StrudelDoodleS/superglm/graph/badge.svg?token=2HO71TA2ZY)](https://codecov.io/github/StrudelDoodleS/superglm)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue)](https://github.com/StrudelDoodleS/superglm/actions/workflows/ci.yml)

Penalised GLMs and GAM-style pricing models for insurance. SuperGLM combines
explicit feature specs, exact REML, large-`n` discrete REML, solver-backed
monotone splines, actuarial validation tooling, and deployable fitted
estimators for Poisson, Gamma, NB2, Tweedie, Binomial, Gaussian, and Gaussian
or Gamma location–scale models.

## Installation

Install SuperGLM from PyPI:

```bash
pip install superglm
```

Plotly-based interactive charts are optional:

```bash
pip install "superglm[plotting]"
```

The local model editor is included in the normal installation.

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

Selection strength is explicit:

```python
SuperGLM()                                  # no sparse selection
SuperGLM(selection_penalty="auto")         # calibrate from the fit data
SuperGLM(selection_penalty=0.05)           # fixed selection strength
```

`None` and `0.0` disable sparse selection. Automatic calibration occurs only
when requested with `"auto"`. REML accepts only `None` or `0.0`; use spline
`select=True` when smooth terms should be eligible to shrink inside REML.

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

## Distributional Location–Scale Models

`SuperLSS` jointly models multiple parameters of a response. Gaussian
LS models conditional location and standard deviation:

```python
from superglm import Spline, SuperLSS
from superglm.distributional import GaussianLS, Predictor

lss = SuperLSS(
    family=GaussianLS(scale_floor=0.05),
    predictors=(
        Predictor("location", {"DrivAge": Spline(kind="cr", k=10)}),
        Predictor("scale", {"DrivAge": Spline(kind="cr", k=8, select=True)}),
    ),
)
lss.fit_reml(train_df, y_train)

parameters = lss.predict_parameters(holdout_df)  # location and scale
```

Use this for heteroskedastic continuous outcomes, such as transformed claim
severity. Raw claim frequency still requires a Poisson or negative-binomial
model; Gaussian LS is not a count likelihood. See
[distributional location–scale models](docs/models/distributional.md) for inference,
diagnostics, and known limits. Discrete fitting remains available for scalar
`SuperGLM` models, but `SuperLSS` currently refuses `discrete=True` until its
multi-parameter route is complete.

`GammaLS` models a strictly positive response:

```python
from superglm import Spline, SuperLSS
from superglm.distributional import GammaLS, Predictor

gamma_lss = SuperLSS(
    family=GammaLS(),
    predictors=(
        Predictor("mean", {"DrivAge": Spline(kind="cr", k=10)}),
        Predictor("scale", {"DrivAge": Spline(kind="cr", k=8, select=True)}),
    ),
).fit_reml(train_df, y_train)
```

Here `scale` is the coefficient of variation, not variance or Gamma shape. At
unit prior weight, `Var(Y | x) = mean² × scale²`; under prior precision weight
`w` (the default semantics), it is `mean² × scale² / w`. Explicit
`weight_semantics="frequency"` instead means literal integer row replication.
The mgcv/MSSM dispersion is `φ = scale²`. Gamma support is strictly positive,
so a zero response requires a different model.

The coefficient core is established IRLS/PIRLS/Fisher–Newton repeated penalized
weighted least squares, with EFS/LAML outside it for automatic smoothing; IRLS
itself is not an originality claim. `GammaLS` provides CDF, quantile, and
expected-shortfall calculations, and predictive simulation uses its quantile.

`TweedieLSS` is the dense three-predictor model for a nonnegative response with
a point mass at zero. Its predictors are ordered `mean`, `dispersion`, then
`power`; `power_lower` and `power_upper` configure an open interval strictly
inside `(1, 2)`:

```python
from superglm import LambdaPolicy, Spline, SuperLSS
from superglm.distributional import Predictor, TweedieLSS

estimate = LambdaPolicy.estimate()
tweedie_lss = SuperLSS(
    family=TweedieLSS(power_lower=1.08, power_upper=1.92),
    predictors=(
        Predictor("mean", {"DrivAge": Spline(kind="cr", k=10, lambda_policy=estimate)}),
        Predictor(
            "dispersion",
            {"DrivAge": Spline(kind="cr", k=8, lambda_policy=estimate)},
        ),
        Predictor("power", {"DrivAge": Spline(kind="cr", k=8, lambda_policy=estimate)}),
    ),
).fit_reml(
    train_df,
    y_train,
    lambdas={
        "mean:DrivAge#wiggle": 1.0,
        "dispersion:DrivAge#wiggle": 1.0,
        "power:DrivAge#wiggle": 1.0,
    },
    max_reml_iter=120,
    reml_tol=1.0e-4,
    max_log_step=1.0,
)
```

The public Tweedie route is dense and uses observed coefficient curvature.
Prior weights remain the default precision contract; explicit integer
`weight_semantics="frequency"` means literal row replication. CDF and quantile
calculations and quantile-based predictive simulation are available; Fisher
fallback and `discrete=True` are not.

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

lorenz = lorenz_curve(y_holdout, mu_holdout, exposure=exposure_holdout)
print(f"Gini ratio: {lorenz.gini_ratio:.4f}")
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

- `BSplineSmooth(..., constraint=Constraint.fit.increasing)`:
  constrained QP path
- `CubicRegressionSpline(..., constraint=Constraint.fit.decreasing)`:
  constrained QP path
- `PSpline(..., constraint=Constraint.fit.increasing)`:
  SCOP path

```python
from superglm import BSplineSmooth, Constraint, PSpline, SuperGLM

qp_model = SuperGLM(
    family="gaussian",
    selection_penalty=0.0,
    features={
        "x": BSplineSmooth(
            n_knots=8,
            constraint=Constraint.fit.increasing,
        ),
    },
)

scop_model = SuperGLM(
    family="gaussian",
    selection_penalty=0.0,
    features={
        "x": PSpline(
            n_knots=10,
            constraint=Constraint.fit.increasing,
        ),
    },
)
```

Post-fit isotonic repair still exists, but it should be treated as a manual
fallback rather than the main monotone workflow.

## Feature Highlights

- `Spline(kind="ps")`, `Spline(kind="cr")`, and `Spline(kind="ns")` cover the
  main spline basis choices.
- `OrderedCategorical(...)` smooths ordered factor levels without forcing a
  plain one-hot representation and reports one whole-smooth test rather than
  separate p-values at arbitrary level positions.
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
    "BonusClass": OrderedCategorical(
        order=["A", "B", "C", "D"],
        basis=Spline(kind="ps", k=6),
    ),
}
```

## Weights And Offsets

Weight semantics are declared, not inferred from the family. `SuperGLM(...,
weight_semantics=...)` chooses between two readings of `sample_weight=`:

- **`"prior"` (default)** — an EDM prior weight, a statement of precision:
  `Var(Y_i | x_i) = phi * V(mu_i) / w_i`. This is what you have when the
  response is an average, such as `incurred / exposure` weighted by exposure,
  and it is the reading R's `glm` and glum give their single weight argument.
- **`"frequency"`** — a replication count: once feature geometry is fixed,
  integer weights have the same likelihood and dispersion semantics as
  repeating rows.

They agree at unit weights and differ everywhere else — dispersion, standard
errors, residual degrees of freedom, REML's smoothing parameters, and learned
knot placement. See the
[families guide](docs/guide/families.md#weight-semantics) before carrying one
spelling into the other, and the
[migration note](docs/development/migrations/weight-semantics-prior.md) for
measured before-and-after figures.

The replication equivalence remains conditional on the constructed design.
Under `"frequency"`, main-effect spline boundaries and adaptive knots honor
replication mass and omit zero-weight rows. Some adaptive interaction and categorical
feature geometry can still depend on the physical row layout, however, so use
fixed or preconstructed feature geometry when exact end-to-end replication
parity matters.

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
- [Distributional location–scale models](docs/models/distributional.md)
- [Validation and model comparison](docs/guide/validation.md)
- [Deployment](docs/guide/deployment.md)
- [Optimization and solver internals](docs/guide/optimization.md)
