# Credibility And Factor-Varying Smooths

`RandomEffect` and `FactorSmooth(..., basis="fs")` are REML-selected
credibility terms. They are useful when a categorical level has its own
experience, but thinner levels should borrow strength from the portfolio
instead of receiving unrestricted fixed effects.

`FactorSmooth(..., basis="sz")` is related but has a different interpretation:
it estimates centered level deviations around a required global curve. Its
wiggle is smoothed, but its polynomial null space is not fully shrunk, so its
level table is not labelled as credibility or collapse.

| SuperGLM | mgcv analogue | What varies by level |
|---|---|---|
| `RandomEffect()` | `s(group, bs="re")` | one intercept |
| `FactorSmooth(x, group=..., basis="fs")` | `s(x, group, bs="fs")` | a fully penalized smooth curve |
| `FactorSmooth(x, group=..., basis="sz")` | `s(x, group, bs="sz", id=1)` | a centered deviation curve |

All three use every fitted level rather than dropping a reference level. REML
estimates their penalty strengths. For RE and FS this controls full shrinkage
toward the population prediction; for SZ it controls wiggle around a
sum-to-zero deviation surface.

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

The explicit `selection_penalty=0.0` documents the fitting contract; it is not
an additional tuning parameter. `fit_reml()` owns the smoothing and
variance-component lambdas and therefore accepts only `None` or `0.0` for
selection. Omitted, `None`, and zero are equivalent here. Sparse term
selection belongs to `fit()` or `fit_path()`, not the REML fit.

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

## Fully penalized FS credibility

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
        FactorSmooth("DrivAge", group="Region", basis="fs", k=6),
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

The FS null-space smoothing components are REML smoothing/variance
components. They are not sparse selection penalties, and they work with
`fit_reml()` while `selection_penalty` remains zero.

## Centered SZ deviations

Use SZ when the global curve should carry the portfolio-wide shape and the
level curves should be identifiable deviations from it:

```python
import numpy as np

from superglm import FactorSmooth, Spline, SuperGLM

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
    selection_penalty=0.0,
).fit_reml(train, train["ClaimNb"], offset=np.log(train["Exposure"]))

regional_deviation = model.factor_smooth("DrivAge:Region:sz", grid=80)
```

Equivalent marginal coefficients across levels sum to zero, so the deviation
curves also sum to zero pointwise. There is one shared `wiggle` lambda. The
polynomial null space remains unpenalized: even an extremely large wiggle
lambda can leave finite linear or low-order polynomial deviations.
Consequently `regional_deviation.collapsed` is `None`, and its table reports
support, information, EDF, and coefficient norms without `credibility` or
`shrinkage` columns.

SZ requires the matching global `Spline`. It is not a generic
`SplineCategorical` interaction: `SplineCategorical` is reference-coded and
unpooled, while SZ uses all levels symmetrically with an exact sum-to-zero
constraint.

Do not add `Categorical()` or `RandomEffect()` for the same grouping column as
a `FactorSmooth`. The factor smooth already contains the lower-order group
geometry, so those terms would duplicate it. SuperGLM rejects that
configuration. Random effects for a different grouping column are supported.

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

These terms use compact group matrices alongside tabmat-backed narrow design
blocks. The structured solver factors the small dense part once. RE and FS use
independent scalar or \(k \times k\) local blocks; SZ uses raw all-level blocks
plus a small equality-constrained border. It does not form the full
\((Kk)^2\) FS or \(((K-1)k)^2\) SZ Hessian.

```python
model = SuperGLM(
    family="poisson",
    features=features,
    interactions=[
        FactorSmooth("DrivAge", group="Region", basis="fs", k=6)
    ],
    discrete=True,
    n_bins=256,
    direct_solve="auto",
    selection_penalty=0.0,
)
model.fit_reml(train, y, offset=offset)
```

`direct_solve="auto"` retains Gram fitting for small terms and switches to the
structured backend at the measured crossover. `direct_solve="structured"`
requires eligible geometry and is useful for reproducible benchmarking.
Locally rank-deficient SZ levels remain eligible because the exact
sum-to-zero constraint can make the full system identifiable. If that global
constrained system is still unidentifiable, `"auto"` retries on Gram and
records the reason; forced `"structured"` raises the global identifiability
error.
`discrete=True` bins the continuous spline support and reuses cached
sufficient statistics across REML iterations; factor identities and the SZ
constraint remain exact.

Tabmat still owns ordinary dense, sparse, and categorical blocks and the
dense-small side of the structured system. Factor smooths retain a more
compact `codes + shared basis` representation with compiled raw-moment
kernels. Expanding the dominant term into a generic tabmat sparse block would
increase storage and weighted sandwich-product work; the structured solver
combines the two representations at their natural boundary.

## Real French motor example

[`examples/fremtpl2_credibility.py`](https://github.com/StrudelDoodleS/superglm/blob/master/examples/fremtpl2_credibility.py)
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

The table below is a 24 July 2026 snapshot generated with seed `20260724`
against the OpenML and preprocessing versions available on that date. It is
not a CI-pinned data or performance contract. Regenerate it when the upstream
data set, preprocessing, solver, or material dependencies change.

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

- `FactorSmooth` supports `basis="fs"` and `basis="sz"`, currently with
  `kind="ps"`, and requires `fit_reml()`.
- Discrete FS/SZ constructs the normal `m <= 2` marginal in bounded QR
  chunks before evaluating the final support basis. FS declarations with
  asymmetric null-component policies or `m > 2` retain a reduced-memory dense
  compatibility construction so those custom penalty coordinates do not
  change silently.
- `fit()` and `fit_path()` reject factor smooths.
- Factor-smooth `k` must be at least 5.
- Missing numeric or grouping values are rejected.
- The mgcv 1.9-4 FS and SZ Gaussian, Poisson, construction, unseen-level, and
  discrete reference cases are pinned in the test suite.
