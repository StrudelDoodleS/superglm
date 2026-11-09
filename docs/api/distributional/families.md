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
- {py:class}`~superglm.DirichletLSS`: one positive concentration parameter
  per named component of a strictly positive continuous composition. See
  [Fit a Dirichlet distributional model](../../how-to/fit-a-dirichlet-distributional-model.md)
  for fitting, labeled mean predictions, joint scoring and scalar share inference.

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
   superglm.DirichletLSS
```

## Observation and weight support

All eleven scalar families bind exact observations, left/right censoring,
one-sided or two-sided truncation, and their combination. Each evaluates the
normalized likelihood, score and every Hessian cross term (orders 0–2).
This does not add general interval censoring.

Scalar responses are paired with input rows by position, including pandas
Series with different index labels. Supply real numeric responses; boolean
and complex values are refused. Encode indicators explicitly as numeric
zero/one values when that is their intended meaning. Labeled Dirichlet
responses additionally require matching row indices and component identities.

| Scalar family | Nonunit prior row law | CDF and quantile | Bounded analytic curvature directions |
| --- | --- | --- | --- |
| `GaussianLS` | Variance divided by weight | yes | built-in links |
| `GammaLS` | CV squared divided by weight | yes | built-in log links |
| `NegativeBinomialLS` | Counts `wY ~ NB2(w mean, w theta)` | no | refused |
| `TweedieLSS` | Dispersion divided by weight | yes | refused |
| `LogNormalLS` | refused | yes | built-in links |
| `GeneralizedGammaLSS` | refused | yes | within numerical admission |
| `GeneralizedParetoLSS` | refused | yes | within numerical admission |
| `LomaxLSS` | refused | yes | admitted analytic windows only |
| `LogLogisticLSS` | refused | yes | restricted at small shape, below |
| `TwoPieceNormalLSS` | refused | yes | refused |
| `TwoPieceLogNormalLSS` | refused | yes | refused |

Integer frequency weights multiply the entire normalized row contribution
once, including its truncation denominator. Unit prior and unit frequency
weights give the same likelihood. A bounded plan refuses complete-row Fisher
information. Missing analytic curvature directions use the existing numerical
fallback; `outer="efs"` uses only core order-two channels. Both log-normal and
two-piece log-normal parametrisations have the same observation support.

Windows mean `lower < Y <= upper`; right censoring means `Y > c`, left censoring
means `Y <= c`. Thus a zero lower truncation bound excludes an NB or Tweedie
zero atom. For positive continuous laws, right censoring at or below zero is
a probability-one event, while left censoring there is impossible. Exact
responses still obey each family's density support. Original windows are
validated before support clipping, with errors naming original input rows.

For prior NB rates, censoring endpoints compare against the represented
binary64 values `k/w`. A threshold equal to a represented rate includes that
count on the closed upper side. Finite count-boundary encodings must be below
`2**53`; this is a numerical restriction, not a truncation of the infinite NB
law. Bounded NB uses the finite NB law, including near the Poisson limit.

Mathematical support is broader than numerical admission. NB fractions and
count sums, and Tweedie's compound Poisson–gamma sum, have explicit convergence,
range and work limits. An unresolved trial is rejected by the solver; an
unresolved public window query raises a numerical-range/convergence error.
Neither substitutes a tiny probability or interprets work exhaustion as a
zero-probability event. Two-piece curvature remains non-smooth at the join;
censoring does not grant unrestricted third/fourth derivatives.

Public log scoring requires a resolved likelihood evaluation. Numerical
failures raise with original input-row positions. Binding rejects structurally
impossible observed or conditioning windows.

See [observation guidance](../../how-to/fit-a-distributional-model.md#truncated-and-censored-rows)
and [diagnostic limits](../../how-to/check-a-distributional-fit.md#truncated-and-censored-rows).

## Dirichlet core contract

`DirichletLSS(components=("first", "second", "third"))` is available from
`superglm`, `superglm.distributional` and `superglm.distributional.families`.
Use `family.alpha("first", ...)` to declare that component's log-alpha
predictor. The component-to-parameter mapping is `family.component_parameters`;
the canonical natural parameters are `alpha_0`, `alpha_1`, and so on.

The response is a row-by-component matrix of known positive continuous shares
summing to one, with at least two distinct named components. Zeros, negative
shares, counts, percentages and materially nonunit sums are rejected; no
pseudocount is added. Within `8*gamma_(K+1)` times row magnitude of unit sum,
where `u=2**-53` and `gamma_n=n*u/(1-n*u)`, the supplied shares denote their
mathematical normalization. The plan retains the supplied values and the
normalization policy as identity evidence.

The family supplies joint likelihood derivatives, complete-row Fisher
information, mean composition `alpha/sum(alpha)`, and the response covariance
via `family.response_covariance(theta)`. Response covariance describes random
compositions; it is distinct from coefficient covariance and uncertainty in
an estimated mean composition. Log-alpha effects change both mean shares and
total concentration. Identical compositions can send concentration to infinity;
the fit discloses this rather than claiming an identified finite concentration.
Only complete observations, integer frequency weights and unit prior weights
are supported. No scalar CDF, quantile or variance API is supplied. This
reference describes the family/core contract. Scalar diagnostic methods in
the table above do not apply to this vector response.

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

One-sided censored intercepts and categorical levels are checked for scale
separation, including with truncation, using the existing `separation` policy.
The default warns that no finite scale estimate exists; `"error"` raises and
`"ignore"` suppresses the separation warning. Other convergence disclosures remain.

Numerical derivative admission is narrower than the mathematical shape domain.
For log-logistic finite windows with positive endpoints, derivatives are declined
when `shape*max(1, abs(log(lo/scale)), abs(log(hi/scale))) < 2^-8`.
This prevents cancellation between density and truncation derivatives at very
small shape. Probability and likelihood values remain available. Analytic higher
directions require the same expression to be at least `1/8`; otherwise the
existing derivative fallback applies. See the
[numerical extension notes](../../development/internals/distributional-family-development.md).

Lomax is the GPD row law under `xi=1/shape`, `sigma=scale/shape`. Its full shape
domain reaches infinite-mean laws unavailable in the current GPD adapter.
Different links and penalties mean their regression models need not coincide.
See [the Lomax recipe](../../how-to/fit-a-lomax-distributional-model.md).
