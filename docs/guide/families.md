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

`sample_weight` has an explicit family-specific contract:

- Poisson, negative binomial, binomial, Gaussian, and Gamma use
  **case/frequency weights**. Once feature geometry is fixed, integer weights
  are likelihood-equivalent to repeating the corresponding rows. For Gaussian
  and Gamma, Pearson dispersion therefore uses
  `sum(sample_weight) - edf`.
- Tweedie uses strictly positive **EDM prior weights**:
  `Var(Y_i | x_i) = phi * mu_i**p / w_i`. Its weights are not replication
  counts, and dispersion uses the observation count rather than the weight
  sum.

These contracts are not interchangeable. In particular, a Gamma response
containing an average severity with claim count as `sample_weight` is often
intended as a prior-weight model with
`Var(Y_i | x_i) = phi * V(mu_i) / w_i` and residual degrees of freedom
`n - edf`. SuperGLM's Gamma fit does not implement that contract: it interprets
the count as a frequency weight and uses `sum(w) - edf`. For calibrated Gamma
dispersion and Wald inference, retain the individual claim-severity rows (and
their ordinary case weights), or use a model with an explicit prior-weight
likelihood. Do not pass claim counts beside aggregated average severities and
interpret the resulting inference as prior-weight inference.

The frequency-weight replication statement is conditional on an identical
constructed design. Main-effect non-Tweedie spline boundaries and the
`quantile_rows` and `quantile_tempered` knot strategies use frequency mass and
ignore zero-weight rows, matching integer row expansion and omission without
materializing copies. Tweedie prior weights intentionally leave spline
geometry determined by physical rows. Some tensor-interaction marginal
centering, interaction-local spline geometry, and categorical/ordered/factor
feature geometry can still depend on the physical row layout. Use fixed or
preconstructed feature geometry when exact end-to-end replication parity
matters.

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

# Profile estimate theta (MASS-style alternating GLM + Newton update)
result = model.estimate_theta(df, y, offset=log_exposure)
print(result.theta_hat)  # estimated dispersion
```

Here `y` contains raw counts, so exposure enters through the log offset rather
than through `sample_weight`.

`estimate_theta()` uses the MASS-style alternating algorithm: fit the GLM at current theta, then take a Newton step on the NB2 profile log-likelihood for theta given the fitted means. Converges in 2–3 outer iterations.

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
