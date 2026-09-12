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

## Distributional families

These families supply the parameter definitions and predictor helpers used by
`SuperLSS`. Declare every parameter, including an empty helper call for an
intercept-only predictor. The [family naming table](../models/distributional.md#family-predictor-names)
shows helper names beside the corresponding result columns.

::: superglm.GaussianLS
    options:
      inherited_members: true
      members: [location, scale]

::: superglm.GammaLS
    options:
      inherited_members: true
      members: [mean, scale]

::: superglm.NegativeBinomialLS
    options:
      inherited_members: true
      members: [mean, theta]

::: superglm.TweedieLSS
    options:
      inherited_members: true
      members: [mu, phi, p]

::: superglm.GeneralizedParetoLSS
    options:
      inherited_members: true
      members: [scale, shape]

::: superglm.LogNormalLS
    options:
      inherited_members: true
      members: [mean, location, scale]

::: superglm.GeneralizedGammaLSS
    options:
      inherited_members: true
      members: [mean, location, scale, shape]

::: superglm.TwoPieceNormalLSS
    options:
      inherited_members: true
      members: [location, scale, skew]

::: superglm.TwoPieceLogNormalLSS
    options:
      inherited_members: true
      members: [mean, location, scale, skew]
