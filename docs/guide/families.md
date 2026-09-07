# Families and Dispersion Estimation

## Supported families

| Family | Variance function | Default link | Use case |
|--------|------------------|-------------|----------|
| `Gaussian()` | V(μ) = 1 | identity | Continuous outcomes |
| `Poisson()` | V(μ) = μ | log | Claim frequency |
| `NegativeBinomial(theta=1.0)` | V(μ) = μ + μ²/θ | log | Overdispersed frequency |
| `Gamma()` | V(μ) = μ² | log | Claim severity |
| `Tweedie(p=1.5)` | V(μ) = μᵖ | log | Pure premium (frequency × severity) |
| `Binomial()` | V(μ) = μ(1 − μ) | logit | Binary classification |

The native API examples below assume `features` is an explicit mapping from
input columns to feature specs, such as `{"age": Numeric()}`. SuperGLM does
not guess an omitted native feature configuration at fit time.

## Binomial (binary classification)

For binary outcomes (y in {0, 1}):

```python
from superglm import SuperGLM

model = SuperGLM(family="binomial", selection_penalty=0, features=features)
model.fit(df, y)
probabilities = model.predict(df)  # returns P(Y=1)
```

The default link is logit. Alternative links can be passed via `link=`:

```python
from superglm import SuperGLM, ProbitLink, CloglogLink

# Probit link (latent variable interpretation)
model = SuperGLM(
    family="binomial",
    link=ProbitLink(),
    selection_penalty=0,
    features=features,
)

# Complementary log-log (asymmetric alternative)
model = SuperGLM(
    family="binomial",
    link=CloglogLink(),
    selection_penalty=0,
    features=features,
)
```

For sklearn-compatible binary classification, use `SuperGLMClassifier`:

```python
from superglm import SuperGLMClassifier

clf = SuperGLMClassifier(selection_penalty=0, spline_features=["age"])
clf.fit(df, y)
clf.predict(df)            # hard labels (0/1)
clf.predict_proba(df)      # (n, 2) class probabilities
clf.decision_function(df)  # log-odds
```

Scale is known (phi = 1) for binomial, so no dispersion estimation is needed.

## Weight semantics

What `sample_weight` says about a row is a **declared modelling choice**, not a
consequence of the family. Set it with `weight_semantics`:

```python
features = {"vehicle_age": Spline(n_knots=8), "region": Categorical()}

# The default: sample_weight is a precision, e.g. exposure beside an average.
model = SuperGLM(family="gamma", features=features, weight_semantics="prior")

# The alternative: sample_weight counts identical rows.
model = SuperGLM(family="gamma", features=features, weight_semantics="frequency")
```

- **`"prior"` (default)** — an EDM prior weight, a statement of *precision*:
  `Var(Y_i | x_i) = phi * V(mu_i) / w_i`, contributing
  `log f(y_i; mu_i, phi / w_i)`. This is what you have when the response is an
  average: `incurred / exposure` weighted by exposure, or an average severity
  weighted by claim count. It is what R's `glm` and glum give their single
  weight argument, and what statsmodels calls `var_weights` and Stata calls
  `aweight`.
- **`"frequency"`** — a replication count, contributing
  `w_i * log f(y_i; mu_i, phi)`. Row `i` stands in for `w_i` identical rows, so
  an integer weight is exactly equivalent to repeating it. This is
  statsmodels' `freq_weights`, Stata's `fweight`, and SAS `GENMOD`'s `FREQ`.

The two agree only at `w == 1`. Integer weights do **not** make them coincide:
at `w == 2` the likelihood size is `n` under `"prior"` and `2n` under
`"frequency"`, so the residual degrees of freedom, `phi`, every Wald standard
error and BIC all differ. Fractional prior weights have a precision
interpretation; fractional frequency weights no longer count literal row
replications. Exposure is continuous, which is why `"prior"` is the default.

For scalar Poisson and negative-binomial models, prior-weighted rates may be
fractional or deliberately adjusted, such as a 20% uplift. No count-integrality
warning is emitted, and fitting and likelihood evaluation do not round them.
If `sample_weight * y` is fractional, the reported likelihood uses a
gamma-function continuation, not an exact count probability; see the
[count-likelihood caveat](../development/migrations/weight-semantics-prior.md#counting-families-have-a-lattice).
Frequency-replication and prior-weighted binomial warnings are unchanged.

### What the choice moves

With the same family parameters, design and penalty, both contracts give the
same mean score equations. This does **not** guarantee identical complete fits:
estimated NB `theta`, REML smoothing parameters, and learned design geometry
can differ and change the fitted means. Other affected quantities include:

| quantity | `"prior"` | `"frequency"` |
|---|---|---|
| likelihood size | rows carrying positive weight | `sum(sample_weight)` |
| residual d.f. | `n_positive - edf` | `sum(w) - edf` |
| `phi`, Wald SEs and intervals | from that d.f. | from that d.f. |
| effective `n` in AIC/BIC | same size | same size |
| REML criterion → `lambda` → `edf` | prior-weight saturated likelihood | replicated saturated likelihood |
| learned spline knots and bins | physical rows | weight mass |

Unweighted fits and fits with `w == 1` are identical under both.

A zero weight is admissible under either contract and means the row leaves the
likelihood: under `"frequency"` it drops out of `sum(w)` on its own, and under
`"prior"` it is excluded from the row count, matching R's rule that
"observations with zero weight [are] not used for calculating dispersion". The
one exception is a Tweedie fit under `"prior"`, which requires strictly positive
weights — its compound-Poisson normalizer carries `log w`, so `w = 0` is an
unevaluable density rather than an uninformative row.

### Replication parity

The frequency-weight replication statement is conditional on an identical
constructed design. Main-effect spline boundaries and the `quantile_rows` and
`quantile_tempered` knot strategies use frequency mass and ignore zero-weight
rows, matching integer row expansion and omission without materializing copies.
Prior weights intentionally leave spline geometry determined by physical rows.
Tensor-interaction marginal centering and interaction-local spline geometry use
that same stream, so a scalar integer-frequency tensor fit matches literal row
expansion through fitting, REML smoothing selection and prediction. Legacy
custom tensor marginals that cannot accept a geometry stream are accepted only
for unit/physical geometry and refuse non-unit replication mass explicitly.

One limitation is declared rather than silent: `estimate_p` profiles the
Tweedie power against the prior-weight likelihood, so it refuses
`weight_semantics="frequency"` with non-unit weights rather than answer under
the wrong one. Expand the rows the counts stand for, or profile under
`"prior"`.

`sample_weight` never enters the linear predictor; use an offset when exposure
should scale the conditional mean.

## Negative binomial: estimating theta

For overdispersed count data where the Poisson variance assumption is too restrictive:

```python
import numpy as np

from superglm import SuperGLM, NegativeBinomial

log_exposure = np.log(exposure)

# Fixed theta
model = SuperGLM(
    family=NegativeBinomial(theta=1.0),
    selection_penalty=0.01,
    features=features,
)
model.fit(df, y, offset=log_exposure)

# Profile estimate theta (alternating GLM fit + safeguarded profile solve)
result = model.estimate_theta(df, y, offset=log_exposure)
print(result.theta_hat)  # estimated dispersion
```

Here `y` contains raw counts, so exposure enters through the log offset rather
than through `sample_weight`.

With `NegativeBinomial("auto")`, `fit()` and `fit_reml()` estimate theta
automatically. Under `fit_reml`, theta is first calibrated at the configured
smoothing and then — since 0.29.0 — **re-estimated at the REML fit and
alternated with warm-started refits to a joint fixed point**, because a theta
frozen before smoothing selection absorbs lack-of-fit at the calibration
smoothing into spurious overdispersion (biasing theta low and overstating
`V(mu) = mu + mu²/theta`). The published `family.theta`,
`model._nb_profile_result.theta_hat`, and the fit always describe the same
final state.

`estimate_theta()` uses the classical alternating scheme (Venables & Ripley
2002, ch. 7.4): fit the GLM at the current theta, then update theta by a
bracketed root find on the closed-form NB2 profile score given the fitted
means (Lawless 1987), started from a method-of-moments estimate. Converges
in 2–3 outer iterations. The theta search range defaults to the numerical
guard rails `(1e-8, 1e8)`; an estimate that lands on an active bound is
reported with `converged=False` plus an `NBThetaBoundWarning` rather than
published as a converged interior value.

### Profile confidence interval

```python
ci = result.ci(alpha=0.05)  # (lower, upper) via profile likelihood ratio
```

### Profile plot

```python
result.profile_plot()  # profile deviance curve + CI region
```

## Tweedie: estimating the power parameter

The examples below assume `y` is a per-exposure response, such as pure premium
per unit of exposure.

Fit with a fixed Tweedie power:

```python
from superglm import SuperGLM, Tweedie

model = SuperGLM(
    family=Tweedie(p=1.5),
    selection_penalty=0.01,
    features=features,
)
model.fit(df, y, sample_weight=exposure)
```

Or estimate the power via profile likelihood:

```python
model = SuperGLM(
    family=Tweedie(p=1.5),
    selection_penalty=0.01,
    features=features,
)
result = model.estimate_p(
    df,
    y,
    sample_weight=exposure,
    p_bounds=(1.1, 1.9),
    ci_alpha=0.05,
)
print(result.p_hat)  # estimated Tweedie power
print(model.summary(alpha=0.05))
```

`phi_method="mle"` and `method="auto"` are the defaults. For ordinary MLE
profiles, `auto` evaluates the exact Tweedie likelihood and its joint *p*/φ
derivatives in one compiled series sweep, then takes safeguarded Newton steps.
Each accepted *p* still has a fully profiled φ, and the winning point is checked
against neighboring exact profiles. Unsafe curvature, series work, constraints,
validation, or bounds outside the stable `[1.05, 1.95]` joint range automatically
fall back to the defensive Brent profile. Explicit
`method="brent"` retains the nested scalar search. `phi_method="pearson"` is an explicit fast plug-in
option when exploratory speed matters, but it is not a likelihood profile
and cannot support a likelihood-ratio confidence interval.

As with other local likelihood optimizers, joint ML and Brent convergence do
not prove a global optimum on an arbitrarily multimodal surface. Use
`method="grid_refine"` for an explicit broad search when boundary behavior or
multiple basins are a substantive concern.

Positive densities normally use the Wright–Bessel series. A diagnosed
saddlepoint fallback is used only when that exact evaluation is not finite or
certifiable. Inspect `density_method`, `density_exact`, `saddlepoint_fraction`,
`near_power_boundary`, `outer_boundary`, and `warnings` on the result. Profiles
at a search bound, and especially near *p*=1 and *p*=2, are naturally unstable;
optimizer convergence alone does not make such an estimate reliable.

`sample_weight` follows the exponential-dispersion-model prior-weight convention:
`Var(Yᵢ | xᵢ) = φ μᵢᵖ / wᵢ` (equivalently, observation-specific dispersion
φ / wᵢ). These are prior weights, not replication counts. Zero-weight observations must be removed
consistently from `X`, `y`, `sample_weight`, and `offset` before profiling; the
profiler rejects non-positive prior weights.
`sample_weight` does not enter the linear predictor or automatically scale the
conditional mean. Use an explicit offset when exposure should also enter the mean.

With `fit_mode="reml"`, REML selects spline smoothing penalties within each
candidate fit. The *p*/φ profile is then evaluated conditionally; it does not jointly estimate *p* and φ
using an mgcv-style REML objective.

### Searching and publishing under different regimes

A REML-mode search runs a full smoothing-parameter selection inside every
candidate *p* evaluation, so its cost is roughly the number of search steps
times the cost of one `fit_reml`. `search_fit_mode` decouples the two: select
*p* under ordinary ML, then publish one REML fit at the selected *p*.

```python
result = model.estimate_p(
    df,
    y,
    sample_weight=exposure,
    fit_mode="reml",        # what gets published
    search_fit_mode="fit",  # what selects p
)
```

Whichever coupling you choose, the published fit is publication-grade.
Candidate fits exist only to rank powers, so they run at a loose smoothing
tolerance; the published refit runs the tight publication default and
re-profiles dispersion against its own fitted mean. `result.phi_hat`, the
coefficients, and their standard errors always describe the model you get
back, never the fits the search discarded.

> **Upgrading from 0.19.x:** a default-tolerance `fit_reml` (and therefore
> every published `estimate_p` fit) now runs the smoothing optimizer to the
> determined answer (`reml_tol` 1e-6 → 1e-9 on the Newton engines), and
> `phi_hat` is re-profiled at the published fit. Standard errors on designs
> with flat log-lambda directions can move by tens of percent relative to
> 0.19.x — once, to the values the old tolerance had left undetermined.
> Predictions are essentially unchanged. This is not a regression to file;
> pass `reml_tol=1e-6` to reproduce the old numbers.

The two couplings differ in which objective *chooses* p, and in what can go
wrong on the way:

- **Agreement.** Selecting *p* under ML and publishing under REML is an
  approximation. On the synthetic benchmark fixture in
  `benchmarks/tweedie_reml_search_cost.py` it moved *p̂* by 1.1e-7 and ran
  about 3× faster, and across a synthetic sweep spanning a 600× range of
  fitted penalty strength the mode disagreement was uncorrelated with how
  strongly REML actually shrinks. That is evidence, not a guarantee —
  compare against `fit_mode="reml"` alone on your own data before adopting
  it.
- **Robustness.** A coupled search evaluates REML at every candidate power,
  and some powers can have no certifiable penalized mode there; the search
  routes around them and warns when the selected optimum sits against that
  boundary, since the true optimum may lie beyond it (a censored estimate).
  A decoupled search never meets that wall — but its single REML publication
  can then fail at the selected *p*, with a typed
  `superglm.PublicationModeError` that reports the certifiability score and
  names the ways out: search under `fit_mode="reml"`, publish the ML fit, or
  restrict `p_bounds`.

Likelihood-ratio confidence intervals remain available for either coupling,
eagerly via `ci_alpha` or lazily via `result.ci()`. The interval inverts the
profile that was searched, around that profile's own value at `p_hat`
(recorded as `result.search_nll`), so it describes the regime named by
`search_fit_mode`; `result.nll` describes the published fit's re-profiled
dispersion. `trace_plot` and `profile_plot` measure against the same searched
reference.

### Profile confidence interval

```python
ci = result.ci(alpha=0.05)  # (lower, upper) via profile LRT
```

!!! note
    `result.ci()` is explicit and potentially expensive: each new boundary probe
    can require a full model refit. It is available only for MLE dispersion
    profiles. It updates the detached returned result, not the model's
    independently owned published profile state. Pass `ci_alpha=0.05` to
    `estimate_p()` when the interval should be computed transactionally and
    cached for `model.summary(alpha=0.05)`. Omitting `ci_alpha` retains the lazy,
    no-extra-CI-work path.

### Search trace and profile plots

```python
result.trace_plot()    # cached search evaluations; performs no new fits
result.profile_plot()  # dense profile curve; may fit additional uncached p values
```

`trace_plot()` sorts by *p* and connects only evaluations already cached by
`estimate_p()`, making it the cheap diagnostic. `profile_plot()` evaluates a dense
grid and fits any uncached *p* values, so it can be substantially expensive when
`phi_method="mle"`.

## How REML treats the dispersion

For the estimated-scale families — Gaussian, Gamma, and Tweedie — `fit_reml`
profiles the dispersion out of the Laplace-approximate REML criterion using
the family's **exact saturated log-likelihood** (Wood 2011, Eq. 4; Wood, Pya
& Säfken 2016, §3.3). For Tweedie this matters structurally: the compound
Poisson–gamma response has an atom at zero, so a zero row's saturated
contribution is dispersion-free while positive rows carry the Dunn–Smyth
series normalizer. Since 0.29.0 the criterion evaluates that likelihood
exactly; earlier releases substituted a Gaussian-shaped
`0.5·(n − Mp)·log(Dp)` term that overweighted the deviance arm in proportion
to the zero fraction, which mis-set smoothing on zero-heavy Tweedie fits and
could run weakly identified variance components to the boundary. Poisson,
negative binomial, and binomial fix `phi = 1` and never enter this profile.

A custom, user-supplied distribution with `scale_known=False` has no exact
profiler; `fit_reml` falls back to the Gaussian-shaped substitution and warns
that smoothing selection is approximate for such a family.

## Which dispersion a Tweedie fit publishes

Three entry points publish three different (individually standard) Tweedie
dispersion estimators. They agree to a few percent on well-behaved data but
are **not** interchangeable:

- **`fit_reml`** publishes the Pearson estimator
  `phi = sum(w·(y−mu)²/V(mu)) / (n − edf)` in `result.phi`. The REML
  criterion internally uses the exact profile MLE to choose the smoothing;
  the published value remains Pearson.
- **`estimate_p`** publishes the profile **MLE** of phi, re-profiled at the
  published fit's mean (`result.phi_hat`).
- The **QP monotone passthrough** path publishes the deviance-based
  `Dp / (n − Mp)`.

This inventory is deliberate documentation of the current state, not an
endorsement: a future release may unify them (the exact MLE is already
computed internally). Until then, compare dispersions across entry points
only with this table in hand. For reference, mgcv's default reported scale
is a Fletcher-(2012)-improved Pearson estimator, which is a fourth
convention again.

## What AIC/BIC count — and what they do not

`metrics().aic` is `−2·loglik(mu_hat, phi_hat) + 2·edf` with `edf = tr(F)`,
and BIC uses the same likelihood with a `log(n)·edf` complexity term (`n` is
the family's likelihood size: `sum(w)` for frequency-weighted families,
the row count for Tweedie prior weights). Two deliberate limitations:

- **No estimated family parameter is counted.** An auto-estimated NB2 theta,
  an estimated Tweedie power `p`, and the estimated dispersion `phi` add
  zero to the complexity term. Within one family this matches statsmodels
  (which never counts the scale) though not R's `logLik.glm` (which counts
  the Gaussian sigma); across families it means a
  `Poisson` vs `NegativeBinomial("auto")` comparison, or a fixed-`p` vs
  `estimate_p` comparison, gives the model with the extra estimated
  parameter that parameter for free. Penalize such comparisons by hand
  (one parameter ≈ 2 AIC points) or compare on held-out deviance.
- **No smoothing-uncertainty correction.** `edf = tr(F)` is the conditional
  effective dimension given the selected lambdas; the Wood–Pya–Säfken (2016)
  corrected AIC, which accounts for smoothing-parameter estimation, is not
  implemented.

## No quasi families

There is no quasi-Poisson or quasi-binomial path: Poisson and Binomial pin
`phi = 1` with no overdispersion escape hatch. For overdispersed counts use
`NegativeBinomial("auto")` (theta absorbs the overdispersion) or a Tweedie
in `1 < p < 2`; for overdispersed binary data no shipped family applies.
`Binomial` is Bernoulli-only by contract — grouped binomial data must be
expanded to one row per trial (or use frequency weights on 0/1 rows).
