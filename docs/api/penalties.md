# Penalties

Penalty objects are the low-level interface behind `selection_penalty=` and
`spline_penalty=`. Prefer the model-level arguments; reach for these classes
when you need a specific group structure or an adaptive weighting.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   superglm.GroupElasticNet
   superglm.GroupLasso
   superglm.SparseGroupLasso
   superglm.Ridge
   superglm.Adaptive
```
