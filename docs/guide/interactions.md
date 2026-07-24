# Interactions

Interactions between features are specified via the `interactions` parameter. The interaction type is auto-detected from the parent feature specs.

```python
model = SuperGLM(
    features={"age": Spline(k=14), "region": Categorical()},
    interactions=[("age", "region")],
    selection_penalty=0.01,
)
model.fit(df, y, sample_weight=exposure)
```

## Auto-detected interaction types

| Parent types | Interaction class | Groups |
|---|---|---|
| Spline + Categorical | `SplineCategorical` | One spline group per non-base level |
| Polynomial + Categorical | `PolynomialCategorical` | One polynomial group per non-base level |
| Numeric + Categorical | `NumericCategorical` | Single group with per-level slopes |
| Categorical + Categorical | `CategoricalInteraction` | Single group with cross-level indicators |
| Numeric + Numeric | `NumericInteraction` | Single group (product term) |
| Polynomial + Polynomial | `PolynomialInteraction` | Single group (tensor product) |

## Factor smooths

`FactorSmooth` is an explicit interaction for fully penalized smooth curves by
factor level:

```python
from superglm import FactorSmooth, Spline, SuperGLM

model = SuperGLM(
    family="poisson",
    features={"DrivAge": Spline(kind="ps", k=8)},
    interactions=[FactorSmooth("DrivAge", group="Region", k=6)],
    selection_penalty=0.0,
)
model.fit_reml(df, y, offset=log_exposure)
```

This is the analogue of mgcv's `s(DrivAge, Region, bs="fs")`. The main spline
is the population curve; the factor smooth supplies fully penalized regional
deviations. It may also be used without parent main effects.

Do not add a `RandomEffect` for `Region` to this model: the factor smooth
already contains the per-region constant direction. A random effect for a
different factor is supported. See [Credibility terms](credibility.md) for
interpretation, prediction modes, and solver behavior.
