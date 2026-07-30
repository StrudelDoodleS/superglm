# Choosing A Fitting Path

The most important decision is whether you are fitting a REML-selected pricing
model or a fixed-penalty sparse model.

| Situation | Recommended path | Why |
|---|---|---|
| Standard spline pricing model | `fit_reml()` with `selection_penalty=0` | Automatic smoothness selection and clean GAM-style inference |
| Large-`n` spline pricing model | `fit_reml(discrete=True)` | Same modeling story, cheaper outer iterations |
| High-cardinality random effect or factor smooth | `fit_reml()` with `direct_solve="auto"` | Compact scalar/block/constrained fitting with automatic small-model fallback |
| Smooth shrinkage inside REML | `fit_reml()` with `select=True` on spline terms | reference-style double-penalty shrinkage |
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

### Measured SZ performance

These measurements were taken on 2026-07-25 with Python 3.13.11, NumPy 2.4.2,
nonuniform weights, Poisson REML, and `direct_solve="structured"`. Clean wall
times exclude cProfile and tracemalloc. They describe this machine and these
model geometries, not a universal performance promise.

| Mode | Rows | Groups | `k` | Coefficients | REML iterations | Median clean wall | Peak Python allocation | Sampled process RSS | Numerical check |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Exact | 6,000 | 50 | 6 | 304 | 7, converged | 2.47 s (3 runs) | 3.50 MiB | 383 MiB | \(2.3\times10^{-13}\) |
| Discrete | 20,000 | 300 | 10 | 3,003 | 12, converged | 2.97 s (3 runs) | 24.23 MiB | 498 MiB | stable prediction checksum |
| Discrete | 1,000,000 | 300 | 10 | 3,003 | 5, converged | 4.47 s (5 runs) | 250.19 MiB | 789 MiB | \(3.7\times10^{-8}\) from the pre-optimization checksum |

Allocation stacks were collected in separate three-REML-iteration passes; the
iteration count changes runtime, not the compact matrix dimensions that set
their peak. The million-row allocation run had a 59.10 MiB sampled RSS delta;
its RSS peak includes the interpreter, input data, and retained model state.
Exact parity is the maximum absolute prediction difference from a
same-iteration dense Gram fit. The discrete cases were structured-only to
avoid the dense \(3{,}003^2\) allocation.

The million-row case previously took a 7.67 s median at the same five REML
iterations. Compact `(group, spline-bin)` aggregation and batched dense/global
spline crosses reduced that median by 41.7%. At 20,000 rows the same retained
path reduced the 7.69 s reference by 61.3%, so there is no small-fit crossover
penalty in this measured geometry.

A bounded private-chunk Numba reduction was also tested with 1, 2, 4, and 8
configured threads. Its apparent two-thread improvement did not repeat:
a reversed five-run sweep measured 4.57 s at two threads versus 4.51 s on the
serial path. The parallel layer was therefore discarded. The retained
FactorSmooth cell reductions are serial and do not change global Numba or BLAS
thread settings.

With BLAS fixed to one thread, a five-repetition exact sweep at 2,000 rows,
`k=6`, and four ordinary numeric columns bracketed the crossover:

| Groups | Coefficients | Dense median | Structured median |
|---:|---:|---:|---:|
| 4 | 28 | 0.075 s | 0.146 s |
| 8 | 52 | 0.086 s | 0.211 s |
| 16 | 100 | 0.188 s | 0.225 s |
| 32 | 196 | 0.435 s | 0.280 s |

So the measured crossover lies between 16 and 32 groups for this case.
`direct_solve="auto"` remains the recommendation because the crossover moves
with row count, spline width, family, and hardware.

The million-row whole-fit cProfile changed as follows:

| Stack | Before cell aggregation | Retained path |
|---|---:|---:|
| Whole profiled fit | 7.92 s | 4.67 s |
| `fit_irls_direct` | 5.50 s | 2.36 s |
| `build_block_structured_system` | 3.91 s / 14 calls | 0.77 s / 14 calls |
| Legacy FactorSmooth dense cross | 2.42 s / 182 calls | absent |
| FactorSmooth sufficient statistics | 0.78 s / 14 calls | 0.10 s / 14 calls |
| Batched FactorSmooth dense-cell cross | absent | 0.23 s / 56 calls |

This is the intended hybrid: Tabmat still assembles the heterogeneous ordinary
small partition, while compiled SZ kernels aggregate the large grouped spline
block and staged batched matrix products contract compact cells. No
observation-by-factor-smooth matrix is materialized. Dense ordinary blocks and
global discretized splines sharing the FactorSmooth bin map use the optimized
cross path; unsupported or mismatched small groups retain the existing compact
fallback.

## `select=True` Versus `selection_penalty > 0`

These are different tools and should not be documented as interchangeable.

- `select=True` keeps you in the REML story and adds reference-style double-penalty
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
