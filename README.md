<p align="center">
  <img src="docs/images/logo.png" alt="SuperGLM" width="300">
</p>

[![CI](https://github.com/StrudelDoodleS/superglm/actions/workflows/ci.yml/badge.svg)](https://github.com/StrudelDoodleS/superglm/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/StrudelDoodleS/superglm/graph/badge.svg?token=2HO71TA2ZY)](https://codecov.io/github/StrudelDoodleS/superglm)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://github.com/StrudelDoodleS/superglm/actions/workflows/ci.yml)

Penalised GLMs for insurance pricing. SuperGLM supports standard penalised fits, exact REML, large-`n` discrete/fREML-style REML, spline double-penalty shrinkage, group penalties, interactions, and statsmodels-style summaries for Poisson, Gamma, NB2, Tweedie, Binomial, and Gaussian models.

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
    selection_penalty=0.01,
    splines=["DrivAge", "VehAge", "BonusMalus"],
    n_knots=10,
)
model.fit(df, y, sample_weight=exposure)
predictions = model.predict(df)
```

**Explicit mode** — full control over each feature:

```python
from superglm import SuperGLM, Spline, Categorical, Numeric

model = SuperGLM(
    family="poisson",
    penalty="group_lasso",
    selection_penalty=0.01,
    features={
        "DrivAge": Spline(kind="bs", k=14),
        "VehAge": Spline(kind="cr", k=10),
        "BonusMalus": Spline(kind="ns", k=10),
        "Area": Categorical(base="most_exposed"),
        "LogDensity": Numeric(),
    },
)
model.fit(df, y, sample_weight=exposure)
```

## Weights and offsets

Public examples use `sample_weight=`. In insurance settings this is interpreted as **exposure / frequency weight**, not inverse-variance weight. The older `exposure=` keyword is still accepted as a backward-compatible alias.

Two common patterns for count models:

```python
# Raw count target: offset absorbs exposure, model estimates a rate
model.fit(df, claim_counts, offset=np.log(exposure))

# Rate target (count / exposure): weight by exposure for heteroscedasticity
model.fit(df, claim_rate, sample_weight=exposure)
```

## Fitting modes

**1. Standard penalised fit**

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

**2. Exact REML**

Use `fit_reml(discrete=False)` for the standard smoothness-selection path (`selection_penalty=0`).

```python
model = SuperGLM(family="poisson", selection_penalty=0, features=features)
model.fit_reml(df, y, sample_weight=exposure, max_reml_iter=30)
```

**3. Discrete / fREML-style REML**

Use `fit_reml(discrete=True)` for large data. This is the fast path for spline-heavy frequency models.

```python
model = SuperGLM(
    family="poisson",
    selection_penalty=0,
    discrete=True,
    n_bins=256,
    features=features,
)
model.fit_reml(df, y, sample_weight=exposure, max_reml_iter=30)
```

**4. Shrinkage vs selection**

- `select=True` on a spline adds mgcv-style double-penalty shrinkage.
- `selection_penalty > 0` activates sparse/group penalties.

Those are different tools:

- `select=True` is the more REML-aligned way to let smooth terms shrink toward zero.
- `selection_penalty > 0` is the sparse-additive path, best used for screening / compression rather than mgcv-style inference.

## Feature types

### Splines

`Spline(kind, k)` is the recommended API for creating spline features. `kind` selects the basis type, `k` is the basis dimension matching mgcv's `k`. You can also use `n_knots` (interior knot count) instead of `k`.

```python
Spline(kind="bs", k=14)                   # 13-column P-spline (k-1 after identifiability)
Spline(kind="ns", k=10)                   # 9-column natural spline (k-1 after identifiability)
Spline(kind="cr", k=10)                   # 9-column cubic regression spline (k-1 after identifiability)
Spline(kind="bs", k=14, select=True)       # mgcv double penalty: spline-vs-linear selection
Spline(kind="cr", k=12, select=True)       # CR with double penalty selection
Spline(kind="bs", k=14, monotone="increasing")  # post-fit isotonic monotone constraint
```

| Kind | Basis | Penalty | Constraints | Built cols |
|------|-------|---------|-------------|-----------|
| `"bs"` | B-spline | Second-difference | Identifiability | `k - 1` |
| `"ns"` | B-spline | Second-difference | f''=0 at boundaries + identifiability | `k - 1` |
| `"cr"` | B-spline | Integrated f'' squared | Natural + identifiability | `k - 1` |

`k` matches mgcv's `k` for all kinds. The built column count is always `k - 1` because the identifiability constraint (unweighted sum-to-zero) removes one direction. mgcv absorbs this via a side constraint instead of physically removing the column.

**Knot placement strategies** — controlled by `knot_strategy=`:

| Strategy | Description |
|----------|-------------|
| `"uniform"` | Equally-spaced interior knots (default) |
| `"quantile"` | Quantiles of unique values (mgcv style) |
| `"quantile_rows"` | Quantiles of all rows (pd.qcut style) |
| `"quantile_tempered"` | Weighted blend of quantile and uniform (`knot_alpha` controls mixing) |

`select=True` (BS, CR, and CR cardinal) decomposes the penalty eigenspace into a linear subgroup and a wiggly subgroup, both penalised (mgcv-style double penalty). With `fit_reml()`, REML estimates separate lambdas for each subgroup — driving a lambda to infinity effectively zeros that component. Three-way selection: nonlinear, linear, or dropped. Not supported for NS (its constrained penalty has only 1 null eigenvalue).

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

**Numeric** — Single continuous feature passed through as-is. Group size 1, so group lasso reduces to standard L1.

```python
Numeric()                       # simple passthrough
```

**OrderedCategorical** — Ordered factor with a spline or step basis. For features with a natural ordering (e.g. policy year, damage severity grade) where you want smooth transitions between levels.

```python
from superglm import OrderedCategorical

# Spline mode: map categories to numeric values, fit a spline through them
OrderedCategorical(order=["A", "B", "C", "D", "E", "F"], basis="spline", n_knots=3)

# Step mode: one-hot + first-difference penalty (soft fusion of adjacent levels)
OrderedCategorical(order=["A", "B", "C", "D", "E", "F"], basis="step")

# Explicit numeric values instead of auto-linspace
OrderedCategorical(values={"A": 0.0, "B": 0.2, "C": 0.4, "D": 0.6, "E": 0.8, "F": 1.0}, basis="spline")
```

## Interactions

Interactions between features are specified via the `interactions` parameter. The interaction type is auto-detected from the parent feature specs.

```python
model = SuperGLM(
    features={"age": Spline(k=14), "region": Categorical()},
    interactions=[("age", "region")],
    selection_penalty=0.01,
)
model.fit(df, y, sample_weight=exposure)
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
| Spline + Spline | `TensorInteraction` | ti()-style tensor product (interaction surface only) |

## Penalties

```python
from superglm import GroupLasso, SparseGroupLasso, GroupElasticNet, Ridge, Adaptive

GroupLasso(lambda1=0.01)                          # group L2 — select/remove entire groups
SparseGroupLasso(lambda1=0.01, alpha=0.5)         # group L2 + elementwise L1
GroupElasticNet(lambda1=0.01, alpha=0.5)           # group lasso + ridge shrinkage
Ridge(lambda1=0.01)                               # L2 shrinkage, no selection
GroupLasso(lambda1=0.01, flavor=Adaptive())        # adaptive group lasso (two-stage)
```

If `lambda1=None` (default in penalty objects), it is auto-calibrated to 10% of `lambda_max` at fit time.

For spline-heavy models, `GroupElasticNet` is usually the smoother selection path than pure `GroupLasso`. `Ridge` is shrinkage only and does not remove terms.

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

## Cross-validation

Select lambda by K-fold cross-validation:

```python
from superglm import SuperGLM, CVResult

model = SuperGLM(
    family="poisson",
    penalty="group_lasso",
    features=features,
)
cv = model.fit_cv(df, y, sample_weight=exposure, n_folds=5, rule="min")

cv.best_lambda         # lambda at minimum mean CV deviance
cv.best_lambda_1se     # most regularised lambda within 1 SE of minimum
cv.mean_cv_deviance    # (n_lambda,) mean test deviance per lambda
cv.se_cv_deviance      # (n_lambda,) standard error across folds
```

`rule` controls which lambda the model refits on: `"min"` uses `best_lambda` (lowest CV deviance), `"1se"` uses `best_lambda_1se` (most regularised within 1 SE of the minimum — usually the better default). With `refit=True` (default), the model is refit on all data at the selected lambda, so `model.predict()` is ready to use.

## Inspecting results

### Summary table

`model.summary()` prints a statsmodels-style table with coefficient estimates, standard errors, p-values, and Wood (2013) smooth term tests. Parametric terms show Wald z-tests; smooth terms show the effective degrees of freedom, penalty strength, and a Bayesian chi-squared test.

```python
print(model.summary())
```

```
╔══════════════════════════ SuperGLM Results ══════════════════════════╗
║ Family:                   Poisson  No. Observations:            5000 ║
║ Link:                         Log  Df (effective):             6.745 ║
║ Method:                      REML  Penalty:              Group Lasso ║
║ Scale (phi):                1.000  Lambda1:                        0 ║
║ Log-Likelihood:           -2146.5  AIC:                       4306.5 ║
╠══════════════════════════════════════════════════════════════════════╣
║                 coef   std err     z     P>|z|   [0.025   0.975]     ║
╟──────────────────────────────────────────────────────────────────────╢
║ Intercept    -2.0693    0.0371 -55.829   0.000   -2.142   -1.997 *** ║
║                                                                      ║
╠═════════════════════════════╡ DrivAge ╞══════════════════════════════╣
║ DrivAge     [spline, 9 params, chi2(1.0)=18.8, p=<0.001]         *** ║
║               rank=9, edf=1.0, lam=2.1e+04, curve SE: 0.01-0.07      ║
║                                                                      ║
╠══════════════════════════════╡ VehAge ╞══════════════════════════════╣
║ VehAge      [spline, 7 params, chi2(2.1)=6.4, p=0.046]           *   ║
║               rank=7, edf=1.7, lam=1.3e+02, curve SE: 0.03-0.10      ║
║                                                                      ║
╠═══════════════════════════════╡ Area ╞═══════════════════════════════╣
║ Area[B]       0.2121    0.0718   2.956   0.003    0.071    0.353 **  ║
║ Area[C]      -0.0793    0.0817  -0.971   0.332   -0.239    0.081     ║
║ Area[D]       0.3012    0.0673   4.473   0.000    0.169    0.433 *** ║
╚══════════════════════════════════════════════════════════════════════╝
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

The `metrics()` path adds residuals, leverage, Cook's distance, and goodness-of-fit tests:

```python
m = model.metrics(df, y, sample_weight=exposure)
print(m.summary())              # same table, richer object
m.residuals("deviance")         # deviance residuals
m.residuals("quantile")         # quantile residuals (standard normal under correct model)
```

### Term-level output

```python
# Per-term inference (TermInference dataclass)
ti = model.term_inference("DrivAge")

# Plot a single term with CI bands and exposure density strip
model.plot("DrivAge", X=df, sample_weight=exposure)

# Plot all terms in a grid
model.plot(X=df, sample_weight=exposure, ci="both")

# Relativity DataFrames — centering="mean" shifts so geometric mean = 1
rels = model.relativities(with_se=True, centering="mean")
```

### Diagnostics

```python
model.term_importance(df, sample_weight=exposure)     # weighted variance of each term's eta contribution
model.knot_summary()                                  # knot metadata for all spline features
model.spline_redundancy(df, sample_weight=exposure)   # knot spacing, basis correlation, rank
model.diagnostics()                                   # per-group diagnostic dict
```

### Example: single-term relativity plots

Poisson frequency model on French MTPL2 (678k policies), REML smoothness selection.
95% pointwise confidence bands with exposure-weighted density strip and interior knot positions.
Click an image to open it at full size.

| Vehicle Age (`quantile_rows` knots) | Bonus-Malus (`quantile_tempered`, α=0.2) |
|:---:|:---:|
| [![VehAge](docs/images/readme_vehage.png)](docs/images/readme_vehage.png) | [![BonusMalus](docs/images/readme_bonusmalus.png)](docs/images/readme_bonusmalus.png) |

## Monotone constraints

Enforce monotonicity on spline terms via post-fit isotonic regression:

```python
# Declare the constraint at feature level
features = {
    "BonusMalus": Spline(kind="bs", k=14, monotone="increasing"),
}

# After fitting, apply the repair
model.fit_reml(df, y, sample_weight=exposure)
repair = model.apply_monotone_postfit(df, sample_weight=exposure)
```

## Tweedie support

Fit with a fixed Tweedie power:

```python
from superglm import Tweedie

model = SuperGLM(family=Tweedie(p=1.5), penalty=GroupLasso())
```

Or estimate the power via profile likelihood:

```python
model = SuperGLM(family=Tweedie(p=1.5), penalty=GroupLasso(lambda1=0.01))
result = model.estimate_p(df, y, sample_weight=exposure, p_range=(1.1, 1.9))
print(result.p_hat)  # estimated Tweedie power
print(result.ci())   # profile likelihood CI
result.profile_plot() # profile deviance curve
```

## Negative binomial (NB2) support

For overdispersed count data where the Poisson variance assumption is too restrictive:

```python
from superglm import NegativeBinomial

# Fixed theta
model = SuperGLM(family=NegativeBinomial(theta=1.0), penalty=GroupLasso(lambda1=0.01))
model.fit(df, y, sample_weight=exposure)

# Profile estimate theta (MASS-style alternating GLM fit + Newton update)
result = model.estimate_theta(df, y, sample_weight=exposure)
print(result.theta_hat)  # estimated dispersion
print(result.ci())       # profile likelihood CI
result.profile_plot()    # profile deviance curve with CI region
```

## Binomial (binary classification)

For binary outcomes (0/1):

```python
from superglm import SuperGLM, Spline, Categorical

model = SuperGLM(
    family="binomial",
    selection_penalty=0,
    features={
        "age": Spline(k=10),
        "region": Categorical(base="first"),
    },
)
model.fit(df, y)
probabilities = model.predict(df)  # returns P(Y=1)
```

The default link is logit. Alternative links (probit, cloglog, cauchit) can be passed via `link=`:

```python
from superglm import ProbitLink

model = SuperGLM(family="binomial", link=ProbitLink(), selection_penalty=0)
```

## sklearn interface

**Regressor** — for count/severity models:

```python
from superglm import SuperGLMRegressor

model = SuperGLMRegressor(
    family="poisson",
    penalty="group_lasso",
    selection_penalty=0.01,
    spline_features=["DrivAge", "VehAge"],
    n_knots=10,
)
model.fit(df, y, sample_weight=exposure)
model.predict(df)
```

**Classifier** — for binary outcomes:

```python
from superglm import SuperGLMClassifier

clf = SuperGLMClassifier(selection_penalty=0, spline_features=["age"])
clf.fit(df, y)
clf.predict(df)          # hard labels (0/1)
clf.predict_proba(df)    # (n, 2) class probabilities
clf.decision_function(df)  # log-odds
```

Feature types are auto-detected: object/category columns become `Categorical`, columns in `spline_features` become `Spline`, everything else becomes `Numeric`.

## Families

| Family | Variance function | Default link | Use case |
|--------|------------------|-------------|----------|
| `Poisson()` | V(mu) = mu | log | Claim frequency |
| `NegativeBinomial(theta=1.0)` | V(mu) = mu + mu^2/theta | log | Overdispersed frequency |
| `Gamma()` | V(mu) = mu^2 | log | Claim severity |
| `Tweedie(p=1.5)` | V(mu) = mu^p | log | Pure premium (frequency x severity) |
| `Binomial()` | V(mu) = mu(1-mu) | logit | Binary classification |
| `Gaussian()` | V(mu) = 1 | identity | Continuous response (loss ratios, etc.) |

## Link functions

Every family has a default link, but any link can be overridden:

```python
from superglm import SuperGLM, InverseLink

model = SuperGLM(family="gamma", link=InverseLink())
```

| Link | Class | String shortcut |
|------|-------|----------------|
| Log | `LogLink` | `"log"` |
| Logit | `LogitLink` | `"logit"` |
| Identity | `IdentityLink` | `"identity"` |
| Probit | `ProbitLink` | `"probit"` |
| Complementary log-log | `CloglogLink` | `"cloglog"` |
| Cauchit | `CauchitLink` | `"cauchit"` |
| Inverse (reciprocal) | `InverseLink` | `"inverse"` |
| Inverse-squared | `InverseSquaredLink` | `"inverse_squared"` |
| Square root | `SqrtLink` | `"sqrt"` |
| Power (parametric) | `PowerLink(power=p)` | -- |
| NB2 canonical | `NegativeBinomialLink(theta=t)` | -- |

All links implement `deriv2_inverse` for REML W(rho) correction support.

## How it works

SuperGLM fits penalised GLMs via PIRLS (penalised iteratively reweighted least squares) with a proximal Newton block coordinate descent inner solver. Each feature group gets its own block in the BCD cycle, and the group lasso proximal operator either keeps or zeros the entire group.

SSP (smoothing spline penalty) reparametrisation transforms the B-spline basis so that the group lasso penalty acts on coefficients that are orthogonal with respect to the smoothing penalty. This means group lasso can select smooth functions without distorting their shape.

For a detailed walkthrough of the solver stack (IRLS, PIRLS, BCD, REML, discretization), see [docs/guide/optimization.md](docs/guide/optimization.md).
