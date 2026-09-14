# Inference

Start with {py:meth}`~superglm.SuperGLM.summary`, the statsmodels-style
coefficient table. {py:meth}`~superglm.SuperGLM.term_inference` is the object
behind every curve and band, the per-term curve, uncertainty and metadata in
one place; {py:meth}`~superglm.SuperGLM.simultaneous_bands` returns
simultaneous confidence bands for a spline feature, which hold jointly across
the curve. {py:meth}`~superglm.SuperGLM.metrics` computes the fit statistics
and {py:meth}`~superglm.SuperGLM.drop1` the drop-one deviance per feature;
{py:meth}`~superglm.SuperGLM.term_importance` scores each term by the weighted
variance of its contribution to the linear predictor, and
{py:meth}`~superglm.SuperGLM.term_drop_diagnostics` by what dropping it costs
in AIC, BIC or holdout loss. {py:meth}`~superglm.SuperGLM.random_effects` and
{py:meth}`~superglm.SuperGLM.factor_smooth` report variance components, level
diagnostics and smooth curves for random-effect and factor-smooth terms, while
{py:meth}`~superglm.SuperGLM.knot_summary`,
{py:meth}`~superglm.SuperGLM.design_summary` and the
{py:attr}`~superglm.SuperGLM.result` property expose what was actually built
and fitted.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
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
