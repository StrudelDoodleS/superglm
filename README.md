# SuperGLM

Penalised GLMs for insurance pricing. Group lasso variable selection, P-splines with SSP reparametrisation, Poisson/Gamma/Tweedie families.

## Installation

```bash
pip install superglm
```

With optional dependencies:

```bash
pip install superglm[sklearn]       # sklearn-compatible wrapper
pip install superglm[interactions]  # InterpretML for interaction detection
pip install superglm[all]           # everything
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
        "DrivAge": Spline(n_knots=10, penalty="ssp"),
        "VehAge": Spline(n_knots=10, penalty="ssp"),
        "BonusMalus": Spline(n_knots=10, penalty="ssp"),
        "Area": Categorical(base="most_exposed"),
        "LogDensity": Numeric(),
    },
)
model.fit(df, y, exposure=weights)
```

**Manual mode** — the original `add_feature()` API still works:

```python
from superglm import SuperGLM, Poisson, Spline, Categorical, Numeric, GroupLasso

model = SuperGLM(family=Poisson(), penalty=GroupLasso(lambda1=0.01))
model.add_feature("DrivAge", Spline(n_knots=10, penalty="ssp"))
model.add_feature("Area", Categorical(base="most_exposed"))
model.add_feature("LogDensity", Numeric())
model.fit(df, y, exposure=weights)
```

## Feature types

**Spline** — P-spline basis with optional SSP reparametrisation. Group lasso selects or removes the entire smooth function as a unit.

```python
Spline(n_knots=10, penalty="ssp")   # SSP reparametrised (recommended)
Spline(n_knots=10, penalty="none")  # raw B-spline basis
```

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

## Penalties

```python
from superglm import GroupLasso, SparseGroupLasso, Ridge, Adaptive

GroupLasso(lambda1=0.01)                          # group L2 — select/remove entire groups
SparseGroupLasso(lambda1=0.01, alpha=0.5)         # group L2 + elementwise L1
Ridge(lambda1=0.01)                               # L2 shrinkage, no selection
GroupLasso(lambda1=0.01, flavor=Adaptive())        # adaptive group lasso (two-stage)
```

If `lambda1=None` (default), it is auto-calibrated to 10% of `lambda_max` at fit time.

## Regularisation path

Fit a sequence of models from high to low regularisation with warm starts:

```python
from superglm import PathResult

model = SuperGLM(family=Poisson(), penalty=GroupLasso())
model.add_feature("DrivAge", Spline(n_knots=10, penalty="ssp"))
model.add_feature("Area", Categorical(base="most_exposed"))

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
# Model summary
model.summary()
# {'DrivAge': {'active': True, 'group_norm': 0.42, 'n_params': 17},
#  'Area':    {'active': False, 'group_norm': 0.0, 'n_params': 5},
#  '_model':  {'deviance': 62424.4, 'intercept': -2.31, ...}}

# Reconstruct a spline curve for plotting
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
model.summary()
```

Feature types are auto-detected: object/category columns become `Categorical`, columns in `spline_features` become `Spline`, everything else becomes `Numeric`.

## Families

| Family | Variance function | Use case |
|--------|------------------|----------|
| `Poisson()` | V(mu) = mu | Claim frequency |
| `Gamma()` | V(mu) = mu^2 | Claim severity |
| `Tweedie(p=1.5)` | V(mu) = mu^p | Pure premium (frequency x severity) |

## How it works

SuperGLM fits penalised GLMs via PIRLS (penalised iteratively reweighted least squares) with a proximal Newton block coordinate descent inner solver. Each feature group gets its own block in the BCD cycle, and the group lasso proximal operator either keeps or zeros the entire group.

SSP (smoothing spline penalty) reparametrisation transforms the B-spline basis so that the group lasso penalty acts on coefficients that are orthogonal with respect to the smoothing penalty. This means group lasso can select smooth functions without distorting their shape.
