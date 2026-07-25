# Choosing A Fitting Path

The most important decision is whether you are fitting a REML-selected pricing
model or a fixed-penalty sparse model.

| Situation | Recommended path | Why |
|---|---|---|
| Standard spline pricing model | `fit_reml()` with `selection_penalty=0` | Automatic smoothness selection and clean GAM-style inference |
| Large-`n` spline pricing model | `fit_reml(discrete=True)` | Same modeling story, cheaper outer iterations |
| High-cardinality random effect or factor smooth | `fit_reml()` with `direct_solve="auto"` | Compact scalar/block/constrained fitting with automatic small-model fallback |
| Smooth shrinkage inside REML | `fit_reml()` with `select=True` on spline terms | mgcv-style double-penalty shrinkage |
| Sparse screening / compression | `fit()` with `selection_penalty > 0` | Fixed-penalty sparse model rather than REML smoothness selection |
| Regularisation path analysis | `fit_path()` | Warm-started lambda path for fixed-penalty models |

## Selection Penalty Intent

Selection calibration is never implicit:

```python
SuperGLM()                                  # no sparse selection
SuperGLM(selection_penalty="auto")         # calibrate from the fit data
SuperGLM(selection_penalty=0.05)           # fixed selection strength
```

`None` and `0.0` disable sparse selection. The string `"auto"` is the only
automatic-calibration setting. REML accepts only `None` or `0.0`; its outer
optimizer owns spline smoothing, while `select=True` supplies REML-native term
shrinkage.

## Default REML Path

This is the intended path for spline-based GAM-style pricing models.

```python
model = SuperGLM(
    family="poisson",
    selection_penalty=0.0,
    features=features,
)
model.fit_reml(df, y, sample_weight=exposure, max_reml_iter=30)
```

Use this when:

- you want automatic smoothness selection
- you care about interpretable smooth terms
- you want statsmodels-style summaries and smooth-term inference

## Large-`n` REML

Turn on `discrete=True` when the model is still a REML pricing model but the
data is large enough that exact REML is too expensive.

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

This is the preferred production path for large spline-heavy frequency models.
`RandomEffect` and `FactorSmooth` retain their exact factor levels on this
path; only the continuous spline support is binned.

## Structured credibility terms

`RandomEffect` and `FactorSmooth` are REML-only terms. Their dominant
categorical block remains compact, and `direct_solve="auto"` chooses between
the ordinary Gram solver and a scalar, block, or sum-to-zero constrained solver
from the fitted geometry.

```python
model = SuperGLM(
    family="poisson",
    features={"VehBrand": RandomEffect()},
    interactions=[
        FactorSmooth("DrivAge", group="Region", basis="fs", k=6)
    ],
    discrete=True,
    n_bins=256,
    direct_solve="auto",
    selection_penalty=0.0,
)
model.fit_reml(df, y, offset=np.log(df["Exposure"]))
```

The structured path supports one dominant credibility block plus narrow dense
features, global splines, and secondary random effects. See
[Credibility terms](credibility.md) for model semantics and the French motor
example.

For `basis="sz"`, configure the matching global spline explicitly:

```python
model = SuperGLM(
    family="poisson",
    features={"DrivAge": Spline(kind="ps", k=7, m=2)},
    interactions=[
        FactorSmooth(
            "DrivAge",
            group="Region",
            basis="sz",
            kind="ps",
            k=6,
            m=2,
        )
    ],
    direct_solve="auto",
    selection_penalty=0.0,
)
model.fit_reml(df, y, offset=np.log(exposure))
```

Tabmat handles the ordinary dense, sparse, and categorical partition and the
dense-small side of this solve. The dominant factor smooth stays in compact
`codes + shared basis` form and uses compiled raw sufficient-statistic
kernels. This avoids expanding \(Kk\) columns into a generic sparse block and
then paying for its full weighted sandwich products.

## `select=True` Versus `selection_penalty > 0`

These are different tools and should not be documented as interchangeable.

- `select=True` keeps you in the REML story and adds mgcv-style double-penalty
  shrinkage to the spline term.
- `selection_penalty > 0` activates sparse/group penalties and moves you toward
  a sparse additive model workflow.

If your question is "should this smooth shrink toward linear or zero while I
stay in REML?", use `select=True`.

If your question is "which groups should survive a fixed-penalty sparse fit?",
use `selection_penalty > 0`.

## Fixed-Penalty Sparse Models

Use `fit()` when you want a fixed `spline_penalty` and sparse or shrinkage
regularisation.

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

This is a good fit for:

- feature screening
- model compression
- fixed-penalty challenger models
- lambda-path experiments

## Multi-Order Spline Penalties

Spline specs can emit multiple derivative-order penalties on one term, each
with its own REML smoothing parameter.

```python
features = {
    "DrivAge": Spline(kind="cr", k=14, m=(1, 2)),
    "VehAge": Spline(kind="ps", k=10, m=(2, 3)),
}
model = SuperGLM(
    family="poisson",
    selection_penalty=0.0,
    features=features,
)
model.fit_reml(df, y, sample_weight=exposure)
```

Current guard rails:

- `select=True + m=(...)` is not yet supported
- tensor interactions with a multi-order spline parent are not yet supported
- `kind="cr_cardinal"` currently supports only the default `m=2`
- `selection_penalty > 0` with shared-block multi-penalty terms remains guarded

## Regularisation Path

`fit_path()` is for fixed-penalty models, not the main REML path.

```python
from superglm import Categorical, GroupLasso, Poisson, Spline, SuperGLM

model = SuperGLM(
    family=Poisson(),
    penalty=GroupLasso(),
    features={
        "DrivAge": Spline(kind="ps", k=14),
        "Area": Categorical(base="most_exposed"),
    },
)
result = model.fit_path(df, y, sample_weight=exposure, n_lambda=50, lambda_ratio=1e-3)

result.lambda_seq
result.coef_path
result.deviance_path
result.n_iter_path
```

Next:

- [Recommended workflows](workflows.md)
- [Feature types](features.md)
- [Monotone splines](monotone.md)
- [REML and solvers](optimization.md)
