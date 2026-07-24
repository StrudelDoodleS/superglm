# Credibility: Random Effects And Factor Smooths

`RandomEffect` and `FactorSmooth` are REML-selected credibility terms. They are
useful when a categorical level has its own experience, but the thinner levels
should borrow strength from the portfolio instead of receiving unrestricted
fixed effects.

They correspond to two common mgcv constructions:

| SuperGLM | mgcv analogue | What varies by level |
|---|---|---|
| `RandomEffect()` | `s(group, bs="re")` | one intercept |
| `FactorSmooth(x, group=...)` | `s(x, group, bs="fs")` | a complete smooth curve |

Both terms use every fitted level rather than dropping a reference level. REML
estimates their penalty strengths and therefore how strongly the level effects
are pulled toward the population prediction.

## Random intercept credibility

Use a random effect when each level needs one relativity.

```python
import numpy as np

from superglm import Numeric, RandomEffect, SuperGLM

model = SuperGLM(
    family="poisson",
    features={
        "LogDensity": Numeric(),
        "VehBrand": RandomEffect(),
    },
    selection_penalty=0.0,
)
model.fit_reml(
    train,
    train["ClaimNb"],
    offset=np.log(train["Exposure"]),
)
```

The estimated variance component is \(\tau^2 = \phi / \lambda\). At level
\(j\), SuperGLM reports scalar credibility

\[
Z_j = \frac{I_j}{I_j + \lambda},
\]

where \(I_j\) is the fitted working information. Thick levels have \(Z_j\)
near one and retain more of their own estimate; thin levels have \(Z_j\) near
zero and are pulled more strongly toward the population.

```python
brand = model.random_effects(
    "VehBrand",
    exposure=train["Exposure"].to_numpy(),
)

brand.lambda_value
brand.tau_squared
brand.effective_df
brand.table[
    [
        "level",
        "exposure",
        "unpooled_effect",
        "effect",
        "posterior_se",
        "credibility",
    ]
]
```

## Factor-smooth credibility

Use a factor smooth when the deviation itself changes over a continuous
variable. For example, the driver-age curve can differ by region:

```python
import numpy as np

from superglm import FactorSmooth, Spline, SuperGLM

model = SuperGLM(
    family="poisson",
    features={
        # The population curve
        "DrivAge": Spline(kind="ps", k=8),
    },
    interactions=[
        # Fully penalized regional deviations from that curve
        FactorSmooth("DrivAge", group="Region", k=6),
    ],
    selection_penalty=0.0,
)
model.fit_reml(
    train,
    train["ClaimNb"],
    offset=np.log(train["Exposure"]),
)
```

One shared P-spline basis is repeated implicitly for every factor level. The
term has shared `wiggle` and null-space smoothing parameters, but each level
gets its own coefficient block. The null-space penalties make the deviations
fully penalized, matching the role of mgcv's `bs="fs"` basis.

For level \(j\), the scalar summary generalizes ordinary credibility to the
whole coefficient block:

\[
Z_j =
\frac{\operatorname{tr}\left(I - (I_j + P)^{-1}P\right)}{k}.
\]

It is the mean fraction of the level's \(k\) coefficient directions retained
after shrinkage. It is a useful compact ranking, not a claim that all parts of
the curve receive identical shrinkage.

```python
regional_age = model.factor_smooth(
    "DrivAge:Region:fs",
    grid=80,
    levels=["R11", "R24"],
)

regional_age.lambdas
regional_age.variance_components
regional_age.table[
    ["level", "fit_weight", "effective_df", "credibility", "sufficient_support"]
]
regional_age.curves[
    ["level", "DrivAge", "effect", "posterior_se", "lower", "upper"]
]
```

Add a global `Spline("DrivAge")` when the scientific question is “what is the
portfolio age curve, and how does each region deviate from it?” A factor
smooth can also be fitted without a global smooth when complete level-specific
curves are the intended model.

Do not add `RandomEffect()` for the same grouping column as a `FactorSmooth`.
The factor smooth already contains a penalized constant direction for every
level, so the two terms would duplicate the random intercept. SuperGLM rejects
that geometry. Random effects for a different grouping column are supported.

## Conditional and population prediction

Prediction is conditional by default and includes fitted random effects and
factor-smooth deviations:

```python
conditional = model.predict(test)
population = model.predict(test, random_effects="population")
```

`random_effects="population"` zeros all `RandomEffect` and `FactorSmooth`
contributions while retaining fixed effects and global smooths. By default, an
unseen factor level also receives this zero-deviation population prediction.
Use `unseen="error"` on either term when an unknown level must fail instead.
Missing group values always fail.

## Exact, discrete, and structured fitting

Both credibility terms use compact group matrices alongside the tabmat-backed
narrow design blocks. The structured solver factors the small dense part once
and handles the dominant term as independent scalar or \(k \times k\) level
blocks. It does not form the full \((Kk) \times (Kk)\) Hessian or covariance.

```python
model = SuperGLM(
    family="poisson",
    features=features,
    interactions=[FactorSmooth("DrivAge", group="Region", k=6)],
    discrete=True,
    n_bins=256,
    direct_solve="auto",
    selection_penalty=0.0,
)
model.fit_reml(train, y, offset=offset)
```

`direct_solve="auto"` retains Gram fitting for small terms and switches to the
structured backend at the measured crossover. `direct_solve="structured"`
requires eligible credibility geometry and is useful for reproducible
benchmarking. `discrete=True` bins the continuous spline support and reuses
cached sufficient statistics across REML iterations; factor identities remain
exact.

## Real French motor example

[`examples/fremtpl2_credibility.py`](../../examples/fremtpl2_credibility.py)
uses the real
[`freMTPL2freq` OpenML data set](https://www.openml.org/d/41214)
and follows the caps used by
[scikit-learn's insurance-pricing example](https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html).
Each row is a French motor third-party-liability policy with claim count,
exposure, driver, vehicle, and geographic attributes.

Run the deterministic 30,000-policy comparison:

```bash
uv run python examples/fremtpl2_credibility.py \
  --max-rows 30000 \
  --output-dir fremtpl2_credibility_results
```

The common baseline uses smooth driver age, vehicle age, and Bonus-Malus plus
the remaining tariff controls. All challengers use Poisson claim counts with
`log(Exposure)` as an offset. The test split contains 7,500 policies and 401
observed claims.

| Challenger | Credibility addition | Held-out deviance | Change vs baseline | Fit time |
|---|---|---:|---:|---:|
| baseline | none | 0.588680 | — | 2.50 s |
| brand fixed | unrestricted vehicle-brand factor | 0.590677 | -0.339% | 2.58 s |
| `re` | vehicle-brand random effect | 0.589453 | -0.131% | 1.58 s |
| `fs` | driver-age deviations by region | 0.587040 | +0.279% | 3.81 s |
| `re + fs` | both terms | 0.587985 | +0.118% | 7.00 s |

These are one seeded demonstration, not a universal ranking. They make the
modeling distinction visible:

- unrestricted brand coefficients overfit this split;
- brand credibility partially pools those estimates and removes most of the
  held-out damage;
- regional factor smooths capture useful shape variation rather than only a
  level shift;
- combining plausible terms does not guarantee the best holdout score.

Vehicle-brand credibility ranged from `0.054` to `0.749`; the thinnest brand
effects were pulled hardest toward zero. Regional factor-smooth credibility
ranged from `0.0069` to `0.262`, reflecting the greater information required
to estimate a complete curve rather than one intercept.

The script writes held-out metrics, the random-effect table, factor-smooth
credibility and curve tables, and `credibility_demo.png`.

## Current scope

- `FactorSmooth` currently supports `kind="ps"` and requires `fit_reml()`.
- `fit()` and `fit_path()` reject factor smooths.
- Factor-smooth `k` must be at least 5.
- Missing numeric or grouping values are rejected.
- The mgcv Gaussian, Poisson, global-smooth, unseen-level, and discrete
  reference cases are pinned in the test suite.
