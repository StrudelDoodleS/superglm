# Families and Dispersion Estimation

## Supported families

| Family | Variance function | Default link | Use case |
|--------|------------------|-------------|----------|
| `Poisson()` | V(μ) = μ | log | Claim frequency |
| `NegativeBinomial(theta=1.0)` | V(μ) = μ + μ²/θ | log | Overdispersed frequency |
| `Gamma()` | V(μ) = μ² | log | Claim severity |
| `Tweedie(p=1.5)` | V(μ) = μᵖ | log | Pure premium (frequency × severity) |
| `Binomial()` | V(μ) = μ(1 − μ) | logit | Binary classification |

## Binomial (binary classification)

For binary outcomes (y in {0, 1}):

```python
from superglm import SuperGLM

model = SuperGLM(family="binomial", selection_penalty=0)
model.fit(df, y)
probabilities = model.predict(df)  # returns P(Y=1)
```

The default link is logit. Alternative links can be passed via `link=`:

```python
from superglm import SuperGLM, ProbitLink, CloglogLink

# Probit link (latent variable interpretation)
model = SuperGLM(family="binomial", link=ProbitLink(), selection_penalty=0)

# Complementary log-log (asymmetric alternative)
model = SuperGLM(family="binomial", link=CloglogLink(), selection_penalty=0)
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

## Negative binomial: estimating theta

For overdispersed count data where the Poisson variance assumption is too restrictive:

```python
from superglm import SuperGLM, NegativeBinomial

# Fixed theta
model = SuperGLM(family=NegativeBinomial(theta=1.0), selection_penalty=0.01)
model.fit(df, y, sample_weight=exposure)

# Profile estimate theta (MASS-style alternating GLM + Newton update)
result = model.estimate_theta(df, y, sample_weight=exposure)
print(result.theta_hat)  # estimated dispersion
```

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

model = SuperGLM(family=Tweedie(p=1.5), selection_penalty=0.01)
model.fit(df, y, sample_weight=exposure)
```

Or estimate the power via profile likelihood:

```python
model = SuperGLM(family=Tweedie(p=1.5), selection_penalty=0.01)
result = model.estimate_p(df, y, sample_weight=exposure, p_bounds=(1.1, 1.9))
print(result.p_hat)  # estimated Tweedie power
```

`phi_method="mle"` and `method="brent"` are the defaults. For each candidate
*p*, `estimate_p()` fits the GLM, profiles φ by nested maximum likelihood, and
evaluates the Tweedie log-likelihood. `phi_method="pearson"` is an explicit fast plug-in
option when exploratory speed matters, but it is not a likelihood profile
and cannot support a likelihood-ratio confidence interval.

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
    profiles. The interval is cached, so summaries and exports display it after
    this explicit call without evaluating the profile again.

### Search trace and profile plots

```python
result.trace_plot()    # cached search evaluations; performs no new fits
result.profile_plot()  # dense profile curve; may fit additional uncached p values
```

`trace_plot()` sorts by *p* and connects only evaluations already cached by
`estimate_p()`, making it the cheap diagnostic. `profile_plot()` evaluates a dense
grid and fits any uncached *p* values, so it can be substantially expensive when
`phi_method="mle"`.
