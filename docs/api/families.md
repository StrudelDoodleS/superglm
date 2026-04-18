# Families

Family objects define the response distribution used during fitting, scoring,
and inference. Convenience constructors in `superglm.families` are a shorthand
for building those family objects.

## Factories

::: superglm.families
    options:
      members:
        - poisson
        - gaussian
        - gamma
        - binomial
        - nb2
        - tweedie

## Family Classes

Known-scale families keep `phi=1`. Negative binomial overdispersion is
controlled by `theta`, not by a meaningful fitted `phi`.

::: superglm.Poisson

::: superglm.Gaussian

::: superglm.Gamma

::: superglm.Binomial

::: superglm.NegativeBinomial

::: superglm.Tweedie
