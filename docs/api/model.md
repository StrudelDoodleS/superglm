# Model

`SuperGLM` is the estimator for penalised GLMs and GAM-style pricing models.
Construct it with a family and a feature specification, fit it with
`fit_reml` for REML smoothness selection or `fit` for fixed penalties, then
read the fit through `summary`, `term_inference` and the plotting methods.
Its members are grouped below by what you do with them; each has its own
page.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/class-no-members.rst

   superglm.SuperGLM
```

## Fit

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   superglm.SuperGLM.fit
   superglm.SuperGLM.fit_reml
   superglm.SuperGLM.fit_path
   superglm.SuperGLM.refit_unpenalised
   superglm.SuperGLM.estimate_p
   superglm.SuperGLM.estimate_theta
   superglm.SuperGLM.bind_levels
   superglm.SuperGLM.clone_unfitted
```

## Predict

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   superglm.SuperGLM.predict
   superglm.SuperGLM.relativities
   superglm.SuperGLM.reconstruct_feature
```

## Read the fit

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   superglm.SuperGLM.summary
   superglm.SuperGLM.term_inference
   superglm.SuperGLM.simultaneous_bands
   superglm.SuperGLM.random_effects
   superglm.SuperGLM.factor_smooth
   superglm.SuperGLM.metrics
   superglm.SuperGLM.drop1
   superglm.SuperGLM.term_importance
   superglm.SuperGLM.term_drop_diagnostics
   superglm.SuperGLM.knot_summary
   superglm.SuperGLM.design_summary
   superglm.SuperGLM.result
```

## Plot

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   superglm.SuperGLM.plot
   superglm.SuperGLM.plot_data
   superglm.SuperGLM.plot_diagnostics
```

## Diagnose

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   superglm.SuperGLM.diagnostics
   superglm.SuperGLM.spline_redundancy
   superglm.SuperGLM.discretization_impact
   superglm.SuperGLM.iteration_diagnostics
   superglm.SuperGLM.reml_diagnostics
   superglm.SuperGLM.training_telemetry
```

## Constrain shapes after fitting

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   superglm.SuperGLM.apply_shape_postfit
   superglm.SuperGLM.monotonize
   superglm.SuperGLM.apply_monotone_postfit
```

## Screen interactions

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   superglm.SuperGLM.screen_interactions
```

## Export for deployment

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   superglm.SuperGLM.export_rating_tables
   superglm.SuperGLM.rating_table_payload
```

## Configuration and fitted attributes

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   superglm.SuperGLM.family
   superglm.SuperGLM.link
   superglm.SuperGLM.features
   superglm.SuperGLM.penalty
   superglm.SuperGLM.lambda2
   superglm.SuperGLM.selection_penalty
   superglm.SuperGLM.selection_penalty_
   superglm.SuperGLM.distribution_
   superglm.SuperGLM.theta_
```

## Related objects

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
