# Distributional models

`SuperLSS` fits several parameters of a response distribution together, one
predictor per parameter. Pass a family first, then one predictor declaration
per parameter using the family's helper methods; build the terms inside each
declaration with `s`, `cat`, `re`, `ti`, `term` and `interaction`. Start
with the [tutorial](../tutorials/distributional-model.md). The strip below
follows a model through its life; each section under it opens with the
members you reach for first, and its table lists every member in the group,
each with its own page.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/class-no-members-distributional.rst

   superglm.SuperLSS
```

::::{grid} 2 3 5 5
:gutter: 2

:::{grid-item-card} 1 · Declare
:link: "#declare-and-fit"
:link-type: url
A family, then one predictor per parameter.
:::
:::{grid-item-card} 2 · Fit
:link: "#declare-and-fit"
:link-type: url
`fit_reml` estimates smoothing jointly; `fit` holds it fixed.
:::
:::{grid-item-card} 3 · Predict
:link: "#predict"
:link-type: url
Means, every parameter, quantiles, simulated draws.
:::
:::{grid-item-card} 4 · Check
:link: "#check-the-fit"
:link-type: url
Residuals, binned moments, calibration, scores.
:::
:::{grid-item-card} 5 · Price
:link: "#price-and-portfolio-views"
:link-type: url
Risk curves, density fans, the book total.
:::
::::

## Declare and fit

Construct the model with a family and one declaration per family parameter,
made with that family's helper methods. {py:meth}`~superglm.SuperLSS.fit_reml`
fits the coefficients and estimates the smoothing parameters jointly, by
generalised Fellner-Schall updates with optional Newton refinement, and is the
normal path; {py:meth}`~superglm.SuperLSS.fit` holds the smoothing parameters
fixed at the `lambdas` you pass. {py:meth}`~superglm.SuperLSS.diagnose`
explains how the fit ran and how smoothing stopped: phase timings, iteration
and refit counts, and the terminal state of every smoothing component.
{py:attr}`~superglm.SuperLSS.predictors` and {py:attr}`~superglm.SuperLSS.family`
echo the declaration as independent copies. The
[how-to on fitting a distributional model](../how-to/fit-a-distributional-model.md)
covers the declaration syntax.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ~superglm.SuperLSS.fit
   ~superglm.SuperLSS.fit_reml
   ~superglm.SuperLSS.diagnose
   ~superglm.SuperLSS.predictors
   ~superglm.SuperLSS.family
```

## Predict

{py:meth}`~superglm.SuperLSS.predict` returns the conditional mean per row on
the response scale the model was fitted to.
{py:meth}`~superglm.SuperLSS.predict_parameters` returns every fitted
parameter on its natural scale, one column per name in
{py:attr}`~superglm.SuperLSS.parameter_names_`, and
{py:meth}`~superglm.SuperLSS.predict_link` the same columns on the link scale,
offsets included. {py:meth}`~superglm.SuperLSS.predict_cdf` and
{py:meth}`~superglm.SuperLSS.predict_quantile` evaluate the fitted law per row.
For uncertainty by simulation, {py:meth}`~superglm.SuperLSS.posterior_draws`
draws coefficients from the fit's Bayesian posterior,
{py:meth}`~superglm.SuperLSS.posterior_bounds` pushes them through a
parameter, a predictive quantile, an exceedance probability or an expected
shortfall to give per-row intervals, and
{py:meth}`~superglm.SuperLSS.posterior_predictive` simulates responses.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ~superglm.SuperLSS.predict
   ~superglm.SuperLSS.predict_parameters
   ~superglm.SuperLSS.predict_link
   ~superglm.SuperLSS.predict_cdf
   ~superglm.SuperLSS.predict_quantile
   ~superglm.SuperLSS.posterior_predictive
   ~superglm.SuperLSS.posterior_draws
   ~superglm.SuperLSS.posterior_bounds
```

## Read the fit

{py:meth}`~superglm.SuperLSS.summary` gives one row per intercept and per
term of every parameter: effective degrees of freedom, smoothing parameter,
the Wood (2013) statistic with its p-value and, for single-coefficient terms,
the estimate and standard error. {py:meth}`~superglm.SuperLSS.term_inference`
sweeps one term of one parameter over its training range with pointwise and
simultaneous bands, and {py:meth}`~superglm.SuperLSS.term_test` is the test
that the term is flat; all three read the training frame, so a model restored
from bytes needs `X_train=`. The fitted attributes are the fit's own state:
{py:attr}`~superglm.SuperLSS.coef_` and
{py:attr}`~superglm.SuperLSS.coef_by_predictor_`,
{py:attr}`~superglm.SuperLSS.covariance_`,
{py:attr}`~superglm.SuperLSS.smoothing_parameters_`,
{py:attr}`~superglm.SuperLSS.result_`, and the
{py:attr}`~superglm.SuperLSS.family_`, {py:attr}`~superglm.SuperLSS.predictors_`
and {py:attr}`~superglm.SuperLSS.parameter_names_` the fit resolved.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ~superglm.SuperLSS.summary
   ~superglm.SuperLSS.term_inference
   ~superglm.SuperLSS.term_test
   ~superglm.SuperLSS.parameter_names_
   ~superglm.SuperLSS.family_
   ~superglm.SuperLSS.predictors_
   ~superglm.SuperLSS.coef_
   ~superglm.SuperLSS.coef_by_predictor_
   ~superglm.SuperLSS.covariance_
   ~superglm.SuperLSS.result_
   ~superglm.SuperLSS.smoothing_parameters_
```

## Check the fit

Every check is built on {py:meth}`~superglm.SuperLSS.residuals`: the
probability-integral transform of each row under its fitted law, or its
normal inverse, which a correct family makes uniform or standard normal;
{py:meth}`~superglm.SuperLSS.residual_set` is the full payload the residual
comes from. {py:meth}`~superglm.SuperLSS.check` bins those residuals along a
covariate and reports mean, standard deviation and skewness per bin with
bootstrap bands, so it says where the fit is wrong and in which moment;
{py:meth}`~superglm.SuperLSS.check_2d` reports the mean on a grid of two
covariates. {py:meth}`~superglm.SuperLSS.actual_expected` reports realised
against predicted totals per bin as a ratio of weighted sums, and
{py:meth}`~superglm.SuperLSS.calibration` answers the coverage, tail, quantile
and reliability questions in one payload. {py:meth}`~superglm.SuperLSS.scores`
gives proper scores per row, the log score and the CRPS, and
{py:meth}`~superglm.SuperLSS.compare` the paired score difference against
another fitted candidate. The
[how-to on checking a distributional fit](../how-to/check-a-distributional-fit.md)
reads each of these in turn.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ~superglm.SuperLSS.residuals
   ~superglm.SuperLSS.residual_set
   ~superglm.SuperLSS.check
   ~superglm.SuperLSS.check_2d
   ~superglm.SuperLSS.actual_expected
   ~superglm.SuperLSS.calibration
   ~superglm.SuperLSS.scores
   ~superglm.SuperLSS.compare
```

## Price and portfolio views

{py:meth}`~superglm.SuperLSS.risk_curves` sweeps one covariate and returns
predicted response quantiles with posterior bands drawn from one shared draw
set, so the curves are coherent with one another;
{py:meth}`~superglm.SuperLSS.density_fan` is the same sweep but returns the
whole conditional density at each point, the picture that shows a shape
change. {py:meth}`~superglm.SuperLSS.parameter_spread` shows how far the
fitted parameters spread across rows and, among identically priced rows, how
far the tail probability does. {py:meth}`~superglm.SuperLSS.portfolio`
simulates the total over a book of rows, optionally by segment, carrying the
dependence the shared coefficient draws induce.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ~superglm.SuperLSS.risk_curves
   ~superglm.SuperLSS.density_fan
   ~superglm.SuperLSS.parameter_spread
   ~superglm.SuperLSS.portfolio
```

## Plot

{py:meth}`~superglm.SuperLSS.plot` draws a grid of term panels per parameter
with pointwise and simultaneous bands, one figure per parameter, which is the
view that sets what drives the location against what drives the scale.
{py:meth}`~superglm.SuperLSS.plot_data` returns the JSON-clean payload behind
any figure without drawing it, keyed by `kind`, so a front end can draw it
without the model. {py:meth}`~superglm.SuperLSS.plot_diagnostics` is the
six-panel distributional diagnostic: the first three panels ask whether the
family is right, the last three where it is wrong.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ~superglm.SuperLSS.plot
   ~superglm.SuperLSS.plot_data
   ~superglm.SuperLSS.plot_diagnostics
```

## Smoothing certification and telemetry

After {py:meth}`~superglm.SuperLSS.fit_reml`,
{py:attr}`~superglm.SuperLSS.smoothing_certified_` says whether the fit met
strict matched certification and
{py:attr}`~superglm.SuperLSS.smoothing_convergence_reason_` how automatic
smoothing stopped; both are `None` for a fixed fit.
{py:attr}`~superglm.SuperLSS.smoothing_unresolved_upper_bound_` lists the
smoothing components with unresolved pressure at the finite cap, and
{py:attr}`~superglm.SuperLSS.exact_face_components_` those accepted at the
exact infinity face. {py:attr}`~superglm.SuperLSS.coefficient_curvature`
reports which curvature the coefficient solve was asked to use, observed or
Fisher, and {py:meth}`~superglm.SuperLSS.training_telemetry` returns the
immutable audit metadata for the accepted fit.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ~superglm.SuperLSS.smoothing_certified_
   ~superglm.SuperLSS.smoothing_convergence_reason_
   ~superglm.SuperLSS.smoothing_unresolved_upper_bound_
   ~superglm.SuperLSS.exact_face_components_
   ~superglm.SuperLSS.coefficient_curvature
   ~superglm.SuperLSS.training_telemetry
```

## Save and load

{py:meth}`~superglm.SuperLSS.to_bytes` serialises one fitted revision and
{py:meth}`~superglm.SuperLSS.from_bytes` restores it after schema and
integrity checks. Load artifacts only from trusted sources. A restored model
carries no training frame or machine timing, so the reading methods need
`X_train=` and {py:meth}`~superglm.SuperLSS.diagnose` says the timing is
absent.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ~superglm.SuperLSS.to_bytes
   ~superglm.SuperLSS.from_bytes
```

## Configuration

{py:attr}`~superglm.SuperLSS.discrete` and {py:attr}`~superglm.SuperLSS.n_bins`
echo the discrete-fitting setting: grouped marginal designs and row chunks,
which can reduce design memory, and the bin count per feature.
{py:attr}`~superglm.SuperLSS.separation` is the build-time policy for
categorical cells whose responses all sit on a boundary the family declares:
warn, error or ignore. {py:attr}`~superglm.SuperLSS.weight_semantics` says
what a `sample_weight` entry means, how precisely a row was measured
(`"prior"`, the default) or how many identical rows it stands for
(`"frequency"`).

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ~superglm.SuperLSS.discrete
   ~superglm.SuperLSS.n_bins
   ~superglm.SuperLSS.separation
   ~superglm.SuperLSS.weight_semantics
```

## Declarations and families

Inside a predictor, {py:func}`~superglm.s` declares a smooth of one numeric
column, {py:func}`~superglm.cat` a categorical effect with a reference level,
{py:func}`~superglm.re` a random effect with a coefficient for every level,
{py:func}`~superglm.term` attaches any other feature specification to a
column, and {py:func}`~superglm.ti` and {py:func}`~superglm.interaction`
declare interactions between terms already in that predictor. Those helpers
return {py:class}`~superglm.BoundTerm` and
{py:class}`~superglm.BoundInteraction`; the family helper wraps them in a
{py:class}`~superglm.BoundPredictor`, which {py:func}`~superglm.bind_predictor`
also creates by parameter name for custom families, and
{py:class}`~superglm.Predictor` is the immutable configuration underneath.

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
   :toctree: generated
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
