# Fit

{py:meth}`~superglm.SuperGLM.fit_reml` is the normal path: it estimates a
smoothing parameter for every penalised term by optimising a Laplace
approximate REML objective, except terms whose
{py:class}`~superglm.LambdaPolicy` pins the value, and it does not accept a
selection penalty. {py:meth}`~superglm.SuperGLM.fit` holds the smoothing
parameters fixed at the configured `spline_penalty` and is the path for sparse
and group selection; {py:meth}`~superglm.SuperGLM.fit_path` walks a
regularisation path from `lambda_max` down to `lambda_min`, warm-starting each
step from the last, and {py:meth}`~superglm.SuperGLM.refit_unpenalised` refits
on the active features alone with no selection penalty.
{py:meth}`~superglm.SuperGLM.estimate_p` and
{py:meth}`~superglm.SuperGLM.estimate_theta` profile the Tweedie power and the
NB2 dispersion respectively, then refit at the estimate. Before any of these,
{py:meth}`~superglm.SuperGLM.bind_levels` fixes every categorical level
universe from the full frame so that later fits on slices share it, and
{py:meth}`~superglm.SuperGLM.clone_unfitted` returns an independent copy of
the configuration without the fit. The [how-to on choosing a fitting
path](../../how-to/choose-a-fitting-path.md) says which to use when.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
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
