# Families

Each family is named for the parameters it models, and each parameter has a
helper method on the family (`location`, `scale`, `mu`, `phi`, and so on) that
takes the declarations on the [Declarations](declarations.md) page and returns
the predictor for that parameter.

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
