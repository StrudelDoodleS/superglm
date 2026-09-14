# Declarations and families

Inside a predictor, {py:func}`~superglm.s` declares a smooth of one numeric
column, {py:func}`~superglm.cat` a categorical effect with a reference level,
{py:func}`~superglm.re` a random effect with a coefficient for every level,
{py:func}`~superglm.term` attaches any other feature specification to a
column, and {py:func}`~superglm.ti` and {py:func}`~superglm.interaction`
declare interactions between terms already in that predictor. Those helpers
return {py:class}`~superglm.BoundTerm` and
{py:class}`~superglm.BoundInteraction`; the family helper wraps them in a
{py:class}`~superglm.BoundPredictor`, which
{py:func}`~superglm.bind_predictor` also creates by parameter name for custom
families, and {py:class}`~superglm.Predictor` is the immutable configuration
underneath.

The families, each named for the parameters it models:

- {py:class}`~superglm.GaussianLS`: location and scale of a Gaussian response.
- {py:class}`~superglm.GammaLS`: mean and coefficient of variation of a
  positive response.
- {py:class}`~superglm.LogNormalLS`: mean (or location) and scale of a
  log-normal response.
- {py:class}`~superglm.NegativeBinomialLS`: mean and NB2 size of a count
  response.
- {py:class}`~superglm.GeneralizedGammaLSS`: mean (or location), scale and
  shape of a generalized gamma, which nests the log-normal, Weibull and gamma.
- {py:class}`~superglm.GeneralizedParetoLSS`: scale and shape of threshold
  excesses.
- {py:class}`~superglm.TweedieLSS`: mean, dispersion and variance power of a
  nonnegative response with a point mass at zero.
- {py:class}`~superglm.TwoPieceLogNormalLSS`: mean (or location), scale and
  skew of a two-piece log-normal.
- {py:class}`~superglm.TwoPieceNormalLSS`: location, scale and skew of a
  two-piece normal on the real line.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   superglm.Predictor
   superglm.BoundPredictor
   superglm.bind_predictor
   superglm.BoundTerm
   superglm.BoundInteraction
   superglm.term
   superglm.s
   superglm.cat
   superglm.re
   superglm.ti
   superglm.interaction
   superglm.GaussianLS
   superglm.GammaLS
   superglm.LogNormalLS
   superglm.NegativeBinomialLS
   superglm.GeneralizedGammaLSS
   superglm.GeneralizedParetoLSS
   superglm.TweedieLSS
   superglm.TwoPieceLogNormalLSS
   superglm.TwoPieceNormalLSS
```
