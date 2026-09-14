# Constrain shapes

Constraints declared with `Constraint.fit` are enforced during fitting and
need none of this. For spline terms declared with `Constraint.postfit`,
{py:meth}`~superglm.SuperGLM.apply_shape_postfit` repairs monotone and
curvature constraints by projecting the fitted curve and mapping it back to
spline coefficients; {py:meth}`~superglm.SuperGLM.monotonize` is the same
repair under its original name and is idempotent, and
{py:meth}`~superglm.SuperGLM.apply_monotone_postfit` is its compatibility
alias. The [how-to on constraining a
smooth](../../how-to/constrain-a-smooth.md) walks through both routes.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   ~superglm.SuperGLM.apply_shape_postfit
   ~superglm.SuperGLM.monotonize
   ~superglm.SuperGLM.apply_monotone_postfit
```
