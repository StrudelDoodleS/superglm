# Model

`SuperGLM` is the estimator for penalised GLMs and GAM-style pricing models.
Construct it with a family and a feature specification, fit it with
`fit_reml` for REML smoothness selection or `fit` for fixed penalties, then
read the fit through `summary`, `term_inference` and the plotting methods.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   superglm.SuperGLM
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
