# SuperGLM

[![CI](https://github.com/StrudelDoodleS/superglm/actions/workflows/ci.yml/badge.svg)](https://github.com/StrudelDoodleS/superglm/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/StrudelDoodleS/superglm/graph/badge.svg?token=2HO71TA2ZY)](https://codecov.io/github/StrudelDoodleS/superglm)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://github.com/StrudelDoodleS/superglm/actions/workflows/ci.yml)

Penalised GLMs for insurance pricing. Group lasso variable selection, P-splines with SSP reparametrisation, Poisson/Gamma/NB2/Tweedie families, interactions, and statsmodels-style model summaries.

## Installation

```bash
pip install git+https://github.com/StrudelDoodleS/superglm.git
```

With optional dependencies:

```bash
pip install "superglm[sklearn] @ git+https://github.com/StrudelDoodleS/superglm.git"
pip install "superglm[all] @ git+https://github.com/StrudelDoodleS/superglm.git"
```

## Quick start

**Auto-detect mode** — list which columns should be splines, the rest is auto-detected:

```python
from superglm import SuperGLM

model = SuperGLM(
    family="poisson",
    penalty="group_lasso",
    lambda1=0.01,
    splines=["DrivAge", "VehAge", "BonusMalus"],
    n_knots=10,
)
model.fit(df, y, exposure=weights)
predictions = model.predict(df)
```

**Explicit mode** — full control over each feature:

```python
from superglm import SuperGLM, Spline, Categorical, Numeric

model = SuperGLM(
    family="poisson",
    penalty="group_lasso",
    lambda1=0.01,
    features={
        "DrivAge": Spline(kind="bs", k=14),
        "VehAge": Spline(kind="cr", k=10),
        "BonusMalus": Spline(kind="ns", k=10),
        "Area": Categorical(base="most_exposed"),
        "LogDensity": Numeric(),
    },
)
model.fit(df, y, exposure=weights)
```

## Feature types

### Splines

`Spline(kind, k)` is the recommended API for creating spline features. `kind` selects the basis type, `k` sets the basis dimension (analogous to mgcv's `k`). You can also use `n_knots` (interior knot count) instead of `k`.

```python
Spline(kind="bs", k=14)                   # P-spline (default kind)
Spline(kind="ns", k=10)                   # Natural spline (linear tails)
Spline(kind="cr", k=10)                   # Cubic regression spline (mgcv bs="cr")
Spline(kind="bs", k=14, split_linear=True) # mgcv double penalty: spline-vs-linear selection
```

| Kind | Basis | Penalty | Constraints |
|------|-------|---------|-------------|
| `"bs"` | B-spline | Second-difference | None |
| `"ns"` | B-spline | Second-difference | f''=0 at boundaries (linear tails) |
| `"cr"` | B-spline | Integrated f'' squared | Natural + identifiability |

`split_linear=True` (BS only) decomposes the penalty eigenspace into a linear subgroup and a wiggly subgroup, both penalised (mgcv-style double penalty). With `fit_reml()`, REML estimates separate lambdas for each subgroup — driving a lambda to infinity effectively zeros that component. Three-way selection: nonlinear, linear, or dropped.

The concrete classes `BasisSpline`, `NaturalSpline`, and `CubicRegressionSpline` are also available for direct use.

**Polynomial** — Orthogonal polynomial (Legendre basis). Very stable across refits — ideal for features with simple monotone or quadratic shapes.

```python
Polynomial(degree=2)            # quadratic (common insurance choice)
Polynomial(degree=3)            # cubic (default)
```

**Categorical** — One-hot encoded with a reference level. The entire factor is selected or removed as a group.

```python
Categorical(base="most_exposed")  # base = highest-exposure level (default)
Categorical(base="first")         # base = alphabetically first level
Categorical(base="B")             # explicit base level
```

**Numeric** — Single continuous feature, standardised by default. Group size 1, so group lasso reduces to standard L1.

```python
Numeric()                       # standardised (default)
Numeric(standardize=False)      # raw scale
```

## Interactions

Interactions between features are specified via the `interactions` parameter. The interaction type is auto-detected from the parent feature specs.

```python
model = SuperGLM(
    features={"age": Spline(k=14), "region": Categorical()},
    interactions=[("age", "region")],
    lambda1=0.01,
)
model.fit(df, y, exposure=weights)
```

Auto-detected interaction types:

| Parent types | Interaction class | Groups |
|---|---|---|
| Spline + Categorical | `SplineCategorical` | One spline group per non-base level |
| Polynomial + Categorical | `PolynomialCategorical` | One polynomial group per non-base level |
| Numeric + Categorical | `NumericCategorical` | Single group with per-level slopes |
| Categorical + Categorical | `CategoricalInteraction` | Single group with cross-level indicators |
| Numeric + Numeric | `NumericInteraction` | Single group (product term) |
| Polynomial + Polynomial | `PolynomialInteraction` | Single group (tensor product) |

## Penalties

```python
from superglm import GroupLasso, SparseGroupLasso, GroupElasticNet, Ridge, Adaptive

GroupLasso(lambda1=0.01)                          # group L2 — select/remove entire groups
SparseGroupLasso(lambda1=0.01, alpha=0.5)         # group L2 + elementwise L1
GroupElasticNet(lambda1=0.01, alpha=0.5)           # group lasso + ridge shrinkage
Ridge(lambda1=0.01)                               # L2 shrinkage, no selection
GroupLasso(lambda1=0.01, flavor=Adaptive())        # adaptive group lasso (two-stage)
```

If `lambda1=None` (default), it is auto-calibrated to 10% of `lambda_max` at fit time.

## Regularisation path

Fit a sequence of models from high to low regularisation with warm starts:

```python
from superglm import PathResult

model = SuperGLM(
    family=Poisson(),
    penalty=GroupLasso(),
    features={
        "DrivAge": Spline(k=14),
        "Area": Categorical(base="most_exposed"),
    },
)
result = model.fit_path(df, y, exposure=weights, n_lambda=50, lambda_ratio=1e-3)

result.lambda_seq       # (50,) decreasing lambda values
result.coef_path        # (50, p) coefficients at each lambda
result.deviance_path    # (50,) deviance at each lambda
result.n_iter_path      # (50,) PIRLS iterations per lambda
```

Or pass a custom lambda sequence:

```python
result = model.fit_path(df, y, exposure=weights, lambda_seq=[1.0, 0.1, 0.01])
```

After `fit_path`, `model.predict()` uses the last (least-regularised) fit.

## Inspecting results

```python
# Statsmodels-style summary table with SEs, p-values, and smooth tests
m = model.metrics(df, y, exposure=weights)
print(m.summary())

# Relativity DataFrames with 95% CI
rels = model.relativities(with_se=True)

# Plot all curves with CI bands and exposure histogram
model.plot_relativities(df, exposure=weights)

# Or use the standalone function
from superglm import plot_relativities
plot_relativities(rels, X=df, exposure=weights)

# Manual curve access
curve = model.reconstruct_feature("DrivAge")
plt.plot(curve["x"], curve["relativity"])
```

## Tweedie support

Fit with a fixed Tweedie power:

```python
from superglm import Tweedie

model = SuperGLM(family=Tweedie(p=1.5), penalty=GroupLasso())
```

Or estimate the power via profile likelihood:

```python
model = SuperGLM(family="tweedie", penalty=GroupLasso(lambda1=0.01))
result = model.estimate_p(df, y, exposure=weights, p_range=(1.1, 1.9))
print(result.p_hat)  # estimated Tweedie power
```

## Negative binomial (NB2) support

For overdispersed count data where the Poisson variance assumption is too restrictive:

```python
from superglm import NegativeBinomial, estimate_nb_theta

# Fixed theta
model = SuperGLM(family=NegativeBinomial(theta=1.0), penalty=GroupLasso(lambda1=0.01))
model.fit(df, y, exposure=weights)

# Profile estimate theta (MASS-style alternating GLM fit + Newton update)
result = estimate_nb_theta(model, df, y, exposure=weights)
print(result.theta)  # estimated dispersion
```

## sklearn interface

```python
from superglm import SuperGLMRegressor

model = SuperGLMRegressor(
    family="poisson",
    penalty="group_lasso",
    lambda1=0.01,
    spline_features=["DrivAge", "VehAge"],
    n_knots=10,
)
model.fit(df, y, sample_weight=weights)
model.predict(df)
```

Feature types are auto-detected: object/category columns become `Categorical`, columns in `spline_features` become `Spline`, everything else becomes `Numeric`.

## Families

| Family | Variance function | Use case |
|--------|------------------|----------|
| `Poisson()` | V(mu) = mu | Claim frequency |
| `NegativeBinomial(theta=1.0)` | V(mu) = mu + mu^2/theta | Overdispersed frequency |
| `Gamma()` | V(mu) = mu^2 | Claim severity |
| `Tweedie(p=1.5)` | V(mu) = mu^p | Pure premium (frequency x severity) |

## How it works

SuperGLM fits penalised GLMs via PIRLS (penalised iteratively reweighted least squares) with a proximal Newton block coordinate descent inner solver. Each feature group gets its own block in the BCD cycle, and the group lasso proximal operator either keeps or zeros the entire group.

SSP (smoothing spline penalty) reparametrisation transforms the B-spline basis so that the group lasso penalty acts on coefficients that are orthogonal with respect to the smoothing penalty. This means group lasso can select smooth functions without distorting their shape.
