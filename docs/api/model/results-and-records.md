# Results and records

{py:class}`~superglm.PathResult` is the immutable container
{py:meth}`~superglm.SuperGLM.fit_path` returns, and
{py:class}`~superglm.REMLResult` the record of the smoothing-parameter
estimation behind {py:meth}`~superglm.SuperGLM.fit_reml`. Two entries here are
inputs rather than results: {py:class}`~superglm.LambdaPolicy`, passed to a
term as `lambda_policy=`, controls one penalty component's smoothing
parameter, estimated by REML or held fixed, and {py:func}`~superglm.warmup`
compiles the optional fitting kernels before the first fit.
{py:class}`~superglm.ModelSummary`, {py:class}`~superglm.ModelMetrics` and
{py:class}`~superglm.DiscretizationResult` are what the reading and diagnosing
methods return, and {py:func}`~superglm.discretization_impact` is the function
form of the method of the same name.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   superglm.PathResult
   superglm.REMLResult
   superglm.LambdaPolicy
   superglm.warmup
   superglm.ModelSummary
   superglm.ModelMetrics
   superglm.DiscretizationResult
   superglm.discretization_impact
```
