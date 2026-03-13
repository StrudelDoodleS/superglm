# Families and Dispersion Estimation

## Supported families

| Family | Variance function | Use case |
|--------|------------------|----------|
| `Poisson()` | V(μ) = μ | Claim frequency |
| `NegativeBinomial(theta=1.0)` | V(μ) = μ + μ²/θ | Overdispersed frequency |
| `Gamma()` | V(μ) = μ² | Claim severity |
| `Tweedie(p=1.5)` | V(μ) = μᵖ | Pure premium (frequency × severity) |

## Negative binomial: estimating theta

For overdispersed count data where the Poisson variance assumption is too restrictive:

```python
from superglm import SuperGLM, NegativeBinomial

# Fixed theta
model = SuperGLM(family=NegativeBinomial(theta=1.0), lambda1=0.01)
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

Fit with a fixed Tweedie power:

```python
from superglm import SuperGLM, Tweedie

model = SuperGLM(family=Tweedie(p=1.5), lambda1=0.01)
model.fit(df, y, sample_weight=exposure)
```

Or estimate the power via profile likelihood:

```python
model = SuperGLM(family="tweedie", lambda1=0.01)
result = model.estimate_p(df, y, sample_weight=exposure, p_range=(1.1, 1.9))
print(result.p_hat)  # estimated Tweedie power
```

`estimate_p()` profiles over the power parameter *p*: for each candidate *p*, it fits the GLM, estimates the dispersion φ via Pearson, and evaluates the exact Tweedie log-likelihood (Wright-Bessel + saddlepoint fallback). Brent's method finds the maximiser.

### Profile confidence interval

```python
ci = result.ci(alpha=0.05)  # (lower, upper) via profile LRT
```

!!! note
    Tweedie profile CIs are more expensive than NB2 because each boundary evaluation requires a full model refit.

### Profile plot

```python
result.profile_plot()  # profile deviance curve + CI region + MLE line
```
