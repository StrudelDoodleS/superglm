# Fit

Construct the model with a family and one declaration per family parameter,
made with that family's helper methods. {py:meth}`~superglm.SuperLSS.fit_reml`
fits the coefficients and estimates the smoothing parameters jointly, by
generalised Fellner-Schall updates with optional Newton refinement, and is the
normal path; {py:meth}`~superglm.SuperLSS.fit` holds the smoothing parameters
fixed at the `lambdas` you pass. {py:meth}`~superglm.SuperLSS.diagnose`
returns a {py:class}`~superglm.FitDiagnosticReport` that explains how the fit
ran and how smoothing stopped: phase timings, iteration and refit counts, and
the terminal state of every smoothing component.
{py:meth}`~superglm.SuperLSS.apply_shape_postfit` repairs the terms whose
fitted curves break a `Constraint.postfit` rule, as
{py:meth}`~superglm.SuperGLM.apply_shape_postfit` does.
{py:attr}`~superglm.SuperLSS.predictors` and
{py:attr}`~superglm.SuperLSS.family` echo the declaration as independent
copies. The [how-to on fitting a distributional
model](../../how-to/fit-a-distributional-model.md) covers the declaration
syntax.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:
   :template: autosummary/class-no-members-distributional.rst

   superglm.SuperLSS
```

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   ~superglm.SuperLSS.fit
   ~superglm.SuperLSS.fit_reml
   ~superglm.SuperLSS.diagnose
   ~superglm.SuperLSS.apply_shape_postfit
   ~superglm.SuperLSS.predictors
   ~superglm.SuperLSS.family
   superglm.FitDiagnosticReport
```

## SplitSplice

{py:class}`~superglm.SplitSplice` fits a truncated body (`GammaLS` or
`LogNormalLS`), a binomial exceedance model, and a generalized Pareto tail at
one fixed loss threshold. Alongside `predict`, `predict_cdf`,
`predict_survival`, `predict_quantile` and `predict_layer`, it shares the
experimental per-row law methods (`predict_pdf`, `predict_mean_excess`,
`distribution_curve` and `plot_distribution`) with `SuperLSS`. See the
[distributional model guide](../../how-to/fit-a-distributional-model.md#seeing-the-fitted-law-of-one-row)
for a fitting, threshold-selection and prediction example.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   superglm.SplitSplice
```

## Choosing a tail threshold

Before fitting a body-plus-tail model, inspect candidate thresholds directly on
the claims with {py:func}`~superglm.mean_excess` and
{py:func}`~superglm.threshold_stability`, then draw their results
({py:class}`~superglm.MeanExcessResult` and
{py:class}`~superglm.ThresholdStabilityResult`) with
{py:func}`~superglm.plot_mean_excess` and
{py:func}`~superglm.plot_threshold_stability`. Both accept frequency counts;
`threshold_stability` also reads right-censored claims settled at a policy
limit and can refit a user-supplied `GeneralizedParetoLSS` tail with covariate
terms, whereas `mean_excess` reads complete claims only. See
[Choosing a tail threshold](../../how-to/fit-a-distributional-model.md#choosing-a-tail-threshold)
for how to read both plots, their band assumptions and their limits.

These threshold functions and result classes are experimental: their names,
arguments and return layouts may change in a minor release without a
deprecation period, while their numbers are held to the same tests as the rest
of `superglm`. The first call in a session emits one
{py:class}`~superglm.ExperimentalWarning`, silenced with the standard warnings
filter.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   superglm.mean_excess
   superglm.MeanExcessResult
   superglm.plot_mean_excess
   superglm.threshold_stability
   superglm.ThresholdStabilityResult
   superglm.plot_threshold_stability
```
