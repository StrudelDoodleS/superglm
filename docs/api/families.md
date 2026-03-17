# Families

All families implement the `Distribution` protocol: `variance(mu)`, `deviance_unit(y, mu)`, `log_likelihood(y, mu, weights, phi)`.

## Family classes

::: superglm.Poisson

::: superglm.Gaussian

::: superglm.Gamma

::: superglm.Binomial

::: superglm.NegativeBinomial

::: superglm.Tweedie

## Convenience constructors (`families` module)

Simple families can be passed as strings (`"poisson"`, `"gamma"`, `"gaussian"`, `"binomial"`).
Parameterized families must use objects: `NegativeBinomial(theta=...)`, `Tweedie(p=...)`.

The `families` module provides shorthand constructors:

```python
from superglm import families

families.poisson()              # Poisson()
families.gaussian()             # Gaussian()
families.gamma()                # Gamma()
families.binomial()             # Binomial()
families.nb2(theta=1.0)         # NegativeBinomial(theta=1.0)
families.nb2(theta="auto")      # NegativeBinomial(theta="auto")
families.tweedie(p=1.5)         # Tweedie(p=1.5)
```

::: superglm.families
    options:
      members:
        - poisson
        - gaussian
        - gamma
        - binomial
        - nb2
        - tweedie
