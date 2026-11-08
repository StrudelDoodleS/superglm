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
- {py:class}`~superglm.LomaxLSS`: monetary scale and positive tail exponent
  of a nonnegative severity. Both parameters have log links.
- {py:class}`~superglm.LogLogisticLSS`: scale (the median) and positive tail
  exponent of a strictly positive severity, also known as Fisk.
- {py:class}`~superglm.TweedieLSS`: mean, dispersion and variance power of a
  nonnegative response with a point mass at zero.
- {py:class}`~superglm.TwoPieceLogNormalLSS`: mean (or location), scale and
  skew of a two-piece log-normal.
- {py:class}`~superglm.TwoPieceNormalLSS`: location, scale and skew of a
  two-piece normal on the real line.

`GammaLS` and `LogNormalLS` are special cases of `GeneralizedGammaLSS`, and
`GaussianLS` of `TwoPieceNormalLSS`. [Why nested families have their own
class](../../explanation/nested-distributional-families.md) says why each keeps
a class and gives the evidence.

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
   superglm.LomaxLSS
   superglm.LogLogisticLSS
   superglm.TweedieLSS
   superglm.TwoPieceLogNormalLSS
   superglm.TwoPieceNormalLSS
```

## Power-tail severity families

`LomaxLSS()` and `LogLogisticLSS()` use natural parameter order `(scale, shape)`
and admit every positive shape. `predict()` returns the conditional mean,
including `+inf` at shape at or below one. Variance is infinite at shape at
or below two. Finite mathematical results outside float64 range raise a
numerical-range error; the law is never clipped to impose a finite moment.
Parameters, CDFs and interior quantiles remain useful without finite moments.
Lomax also supplies complete-row Fisher information and independently checked
expected shortfall (infinite at shape at or below one). Log-logistic supplies
neither optional operation. Named posterior summaries refuse non-finite draws.

Both support exact observations, left/right censoring and lower, upper or
two-sided per-row truncation, including their combined normalization. They do
not add general interval-censored observations. Unit prior weights and integer
frequency weights are supported; non-unit prior weights are refused. Neither
family supplies a random-sampling protocol. Exact Lomax responses may equal
zero; exact log-logistic responses must be positive. Censoring and probability
windows may touch zero; no epsilon is added to observations.

Lomax is the GPD row law under `xi=1/shape`, `sigma=scale/shape`. Its full shape
domain reaches infinite-mean laws unavailable in the current GPD adapter.
Different links and penalties mean their regression models need not coincide.
See [the Lomax recipe](../../how-to/fit-a-lomax-distributional-model.md).
