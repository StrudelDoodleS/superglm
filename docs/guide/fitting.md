# Fitting Modes

## 1. Standard penalised fit

Use `fit()` when you want a fixed `spline_penalty` and a standard regularised GLM fit.

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

## 2. Exact REML

Use `fit_reml(discrete=False)` for the standard smoothness-selection path (`selection_penalty=0`).

```python
model = SuperGLM(family="poisson", selection_penalty=0.0, features=features)
model.fit_reml(df, y, sample_weight=exposure, max_reml_iter=30)
```

## 3. Discrete / fREML-style REML

Use `fit_reml(discrete=True)` for large data. This is the fast path for spline-heavy frequency models.

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

## 4. Shrinkage vs selection

- `select=True` on a spline adds mgcv-style double-penalty shrinkage.
- `selection_penalty > 0` activates sparse/group penalties.

Those are different tools:

- `select=True` is the more REML-aligned way to let smooth terms shrink toward zero.
- `selection_penalty > 0` is the sparse-additive path, best used for screening / compression rather than mgcv-style inference.

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
result = model.fit_path(df, y, sample_weight=exposure, n_lambda=50, lambda_ratio=1e-3)

result.lambda_seq       # (50,) decreasing lambda values
result.coef_path        # (50, p) coefficients at each lambda
result.deviance_path    # (50,) deviance at each lambda
result.n_iter_path      # (50,) PIRLS iterations per lambda
```

Or pass a custom lambda sequence:

```python
result = model.fit_path(df, y, sample_weight=exposure, lambda_seq=[1.0, 0.1, 0.01])
```

After `fit_path`, `model.predict()` uses the last (least-regularised) fit.
