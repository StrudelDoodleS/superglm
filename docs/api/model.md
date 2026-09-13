# Model

`SuperGLM` is the estimator for penalised GLMs and GAM-style pricing models.
Construct it with a family and a feature specification, fit it with
{py:meth}`~superglm.SuperGLM.fit_reml` for REML smoothness selection or
{py:meth}`~superglm.SuperGLM.fit` for fixed penalties, then read the fit
through {py:meth}`~superglm.SuperGLM.summary`,
{py:meth}`~superglm.SuperGLM.term_inference` and the plotting methods. The
strip below follows a model through its life; each section under it opens
with the members you reach for first, and its table lists every member in
the group, each with its own page.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/class-no-members-model.rst

   superglm.SuperGLM
```

::::{grid} 2 3 5 5
:gutter: 2

:::{grid-item-card} 1 · Build
:link: "#configuration-and-fitted-attributes"
:link-type: url
A family, a feature spec, a penalty policy.
:::
:::{grid-item-card} 2 · Fit
:link: "#fit"
:link-type: url
`fit_reml` for REML; `fit` for a fixed penalty.
:::
:::{grid-item-card} 3 · Read the fit
:link: "#read-the-fit"
:link-type: url
Summary, per-term curves, diagnostics.
:::
:::{grid-item-card} 4 · Predict
:link: "#predict"
:link-type: url
Means, relativities, reconstructed effects.
:::
:::{grid-item-card} 5 · Deploy
:link: "#export-for-deployment"
:link-type: url
Rating tables and the payload behind them.
:::
::::

## Fit

{py:meth}`~superglm.SuperGLM.fit_reml` is the normal path: it estimates a
smoothing parameter for every penalised term by optimising a Laplace
approximate REML objective, and it does not accept a selection penalty.
{py:meth}`~superglm.SuperGLM.fit` holds the smoothing parameters fixed at the
configured `spline_penalty` and is the path for sparse and group selection;
{py:meth}`~superglm.SuperGLM.fit_path` walks a regularisation path from
`lambda_max` down to `lambda_min`, warm-starting each step from the last, and
{py:meth}`~superglm.SuperGLM.refit_unpenalised` refits on the active features
alone with no selection penalty. {py:meth}`~superglm.SuperGLM.estimate_p` and
{py:meth}`~superglm.SuperGLM.estimate_theta` profile the Tweedie power and the
NB2 dispersion respectively, then refit at the estimate. Before any of these,
{py:meth}`~superglm.SuperGLM.bind_levels` fixes every categorical level
universe from the full frame so that later fits on slices share it, and
{py:meth}`~superglm.SuperGLM.clone_unfitted` returns an independent copy of
the configuration without the fit. The
[how-to on choosing a fitting path](../how-to/choose-a-fitting-path.md) says
which to use when.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ~superglm.SuperGLM.fit
   ~superglm.SuperGLM.fit_reml
   ~superglm.SuperGLM.fit_path
   ~superglm.SuperGLM.refit_unpenalised
   ~superglm.SuperGLM.estimate_p
   ~superglm.SuperGLM.estimate_theta
   ~superglm.SuperGLM.bind_levels
   ~superglm.SuperGLM.clone_unfitted
```

## Predict

{py:meth}`~superglm.SuperGLM.predict` returns the mean on the response scale
for new rows, with an optional offset and a choice of conditional or
population random effects. {py:meth}`~superglm.SuperGLM.relativities` returns
plot-ready relativity tables for every feature, the multiplicative form a
rating engine wants; {py:meth}`~superglm.SuperGLM.reconstruct_feature` returns
one feature's fitted curve or effect on its original scale.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ~superglm.SuperGLM.predict
   ~superglm.SuperGLM.relativities
   ~superglm.SuperGLM.reconstruct_feature
```

## Read the fit

Start with {py:meth}`~superglm.SuperGLM.summary`, the statsmodels-style
coefficient table. {py:meth}`~superglm.SuperGLM.term_inference` is the object
behind every curve and band, the per-term curve, uncertainty and metadata in
one place; {py:meth}`~superglm.SuperGLM.simultaneous_bands` returns simultaneous
confidence bands for a spline feature, which hold jointly across the curve. {py:meth}`~superglm.SuperGLM.metrics` computes the fit
statistics and {py:meth}`~superglm.SuperGLM.drop1` the drop-one deviance per
feature; {py:meth}`~superglm.SuperGLM.term_importance` scores each term by the
weighted variance of its contribution to the linear predictor, and
{py:meth}`~superglm.SuperGLM.term_drop_diagnostics` by what dropping it
costs in AIC, BIC or holdout loss. {py:meth}`~superglm.SuperGLM.random_effects`
and {py:meth}`~superglm.SuperGLM.factor_smooth` report variance components,
level diagnostics and smooth curves for random-effect and factor-smooth terms,
while {py:meth}`~superglm.SuperGLM.knot_summary`,
{py:meth}`~superglm.SuperGLM.design_summary` and the
{py:attr}`~superglm.SuperGLM.result` property expose what was actually built
and fitted.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ~superglm.SuperGLM.summary
   ~superglm.SuperGLM.term_inference
   ~superglm.SuperGLM.simultaneous_bands
   ~superglm.SuperGLM.random_effects
   ~superglm.SuperGLM.factor_smooth
   ~superglm.SuperGLM.metrics
   ~superglm.SuperGLM.drop1
   ~superglm.SuperGLM.term_importance
   ~superglm.SuperGLM.term_drop_diagnostics
   ~superglm.SuperGLM.knot_summary
   ~superglm.SuperGLM.design_summary
   ~superglm.SuperGLM.result
```

## Plot

{py:meth}`~superglm.SuperGLM.plot` is the single entry point for drawing
terms: all main effects, one, a subset, or an interaction, with pointwise or
simultaneous bands. {py:meth}`~superglm.SuperGLM.plot_data` returns the plain
DataFrames, arrays and metadata behind those figures so you can rebuild them
in matplotlib, plotly, Excel or a reporting system.
{py:meth}`~superglm.SuperGLM.plot_diagnostics` is the four-panel residual
figure on quantile residuals, with a simulation-based Q-Q envelope.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ~superglm.SuperGLM.plot
   ~superglm.SuperGLM.plot_data
   ~superglm.SuperGLM.plot_diagnostics
```

## Diagnose

{py:meth}`~superglm.SuperGLM.diagnostics` is the per-group dictionary for
programmatic and audit access. {py:meth}`~superglm.SuperGLM.spline_redundancy`
reports knot spacing, basis correlation and effective rank, and
{py:meth}`~superglm.SuperGLM.discretization_impact` measures what binning the
smooth terms into rating-table bins and grids does to the predictions.
{py:meth}`~superglm.SuperGLM.iteration_diagnostics` (available after
`fit(record_diagnostics=True)`), {py:meth}`~superglm.SuperGLM.reml_diagnostics`
and {py:meth}`~superglm.SuperGLM.training_telemetry` are the solver's own
records: plain JSON-serialisable objects with no tracking dependency, ready
for MLflow, files, logs or a governance system.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ~superglm.SuperGLM.diagnostics
   ~superglm.SuperGLM.spline_redundancy
   ~superglm.SuperGLM.discretization_impact
   ~superglm.SuperGLM.iteration_diagnostics
   ~superglm.SuperGLM.reml_diagnostics
   ~superglm.SuperGLM.training_telemetry
```

## Constrain shapes after fitting

Constraints declared with `Constraint.fit` are enforced during fitting and
need none of this. For spline terms declared with `Constraint.postfit`,
{py:meth}`~superglm.SuperGLM.apply_shape_postfit` repairs monotone and
curvature constraints by projecting the fitted curve and mapping it back to
spline coefficients; {py:meth}`~superglm.SuperGLM.monotonize` is the same
repair under its original name and is idempotent, and
{py:meth}`~superglm.SuperGLM.apply_monotone_postfit` is its compatibility
alias. The [how-to on constraining a smooth](../how-to/constrain-a-smooth.md)
walks through both routes.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ~superglm.SuperGLM.apply_shape_postfit
   ~superglm.SuperGLM.monotonize
   ~superglm.SuperGLM.apply_monotone_postfit
```

## Screen interactions

{py:meth}`~superglm.SuperGLM.screen_interactions` ranks candidate pairs of
fitted features by PSST, the penalised smooth score test: one O(n) pass per
pair and no refits, asking how much of the model's leftover working signal
each interaction could absorb once the pair's own main effects are profiled
out. Run it before adding interactions to the spec; the
[how-to on screening interactions](../how-to/screen-interactions.md) covers
reading the `z` and `kind` columns.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ~superglm.SuperGLM.screen_interactions
```

## Export for deployment

{py:meth}`~superglm.SuperGLM.export_rating_tables` writes the deployment
rating tables for the fitted model; {py:meth}`~superglm.SuperGLM.rating_table_payload`
builds the renderer-independent payload behind them, for when you need the
tables as objects rather than files. The
[how-to on deploying a fitted model](../how-to/deploy-a-fitted-model.md)
shows the export end to end.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ~superglm.SuperGLM.export_rating_tables
   ~superglm.SuperGLM.rating_table_payload
```

## Configuration and fitted attributes

{py:attr}`~superglm.SuperGLM.family`, {py:attr}`~superglm.SuperGLM.link`,
{py:attr}`~superglm.SuperGLM.features`, {py:attr}`~superglm.SuperGLM.penalty`,
{py:attr}`~superglm.SuperGLM.lambda2` and
{py:attr}`~superglm.SuperGLM.selection_penalty` echo the configuration you
passed. The trailing-underscore attributes follow the scikit-learn convention
and are resolved by the latest successful fit:
{py:attr}`~superglm.SuperGLM.selection_penalty_`,
{py:attr}`~superglm.SuperGLM.distribution_` and
{py:attr}`~superglm.SuperGLM.theta_`, the NB2 dispersion.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ~superglm.SuperGLM.family
   ~superglm.SuperGLM.link
   ~superglm.SuperGLM.features
   ~superglm.SuperGLM.penalty
   ~superglm.SuperGLM.lambda2
   ~superglm.SuperGLM.selection_penalty
   ~superglm.SuperGLM.selection_penalty_
   ~superglm.SuperGLM.distribution_
   ~superglm.SuperGLM.theta_
```

## Results and records

{py:class}`~superglm.PathResult` is the immutable container
{py:meth}`~superglm.SuperGLM.fit_path` returns, and
{py:class}`~superglm.REMLResult` the record of the smoothing-parameter
estimation behind {py:meth}`~superglm.SuperGLM.fit_reml`.
{py:class}`~superglm.LambdaPolicy` controls one penalty component's smoothing
parameter, estimated by REML or held fixed; {py:func}`~superglm.warmup`
compiles the optional fitting kernels before the first fit.
{py:class}`~superglm.ModelSummary`, {py:class}`~superglm.ModelMetrics`,
{py:class}`~superglm.FitDiagnosticReport` and
{py:class}`~superglm.DiscretizationResult` are what the reading and diagnosing
methods return, and {py:func}`~superglm.discretization_impact` is the
function form of the method of the same name.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   superglm.PathResult
   superglm.REMLResult
   superglm.LambdaPolicy
   superglm.warmup
   superglm.ModelSummary
   superglm.ModelMetrics
   superglm.FitDiagnosticReport
   superglm.DiscretizationResult
   superglm.discretization_impact
```
